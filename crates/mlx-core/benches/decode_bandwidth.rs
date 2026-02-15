//! Decode-loop bandwidth benchmark.
//!
//! Measures the token generation throughput (tokens/sec) during the decode
//! phase of LLM inference. Each iteration simulates generating a single token:
//!   1. Embed the previous token
//!   2. Apply RoPE
//!   3. Linear projection (simulating Q/K/V)
//!   4. Single-token attention against growing KV cache
//!
//! The benchmark measures how throughput plateaus as the KV cache grows.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

/// Benchmark a single decode step with varying KV-cache lengths.
/// This measures how token generation throughput changes as context grows.
fn bench_decode_step(c: &mut Criterion) {
    let head_dim = 64;
    let model_dim = 128;

    let cache_lengths: &[(usize, &str)] = &[
        (32, "cache_32"),
        (128, "cache_128"),
        (512, "cache_512"),
        (1024, "cache_1024"),
    ];

    let mut group = c.benchmark_group("decode_bandwidth");
    group.sample_size(20);

    for &(cache_len, name) in cache_lengths {
        // FLOPs per token: linear projection + attention
        // Linear: 2 * 1 * model_dim * model_dim
        // Attention: 2 * 1 * cache_len * head_dim
        let flops = 2 * model_dim * model_dim + 2 * cache_len * head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        // Pre-generate data
        let token_data: Vec<f32> = (0..model_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let weight_data: Vec<f32> = (0..model_dim * model_dim)
            .map(|i| ((i as f32) * 0.0007).cos())
            .collect();
        let kv_data: Vec<f32> = (0..cache_len * head_dim)
            .map(|i| ((i as f32) * 0.003).sin())
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();

        group.bench_function(BenchmarkId::new("decode_step", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));

                    // Current token embedding: [1, model_dim]
                    let token = stream.add_constant(
                        token_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![1, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Weight for projection: [model_dim, model_dim]
                    let w = stream.add_constant(
                        weight_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![model_dim as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // KV cache: [cache_len, head_dim]
                    let k_cache = stream.add_constant(
                        kv_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![cache_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    let v_cache = stream.add_constant(
                        kv_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![cache_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    (stream, token, w, k_cache, v_cache)
                },
                |(stream, token, w, k_cache, v_cache)| {
                    // Step 1: Linear projection [1, model_dim] @ [model_dim, model_dim]
                    let projected = stream.add_op(
                        OpKind::MatMul,
                        SmallVec::from_slice(&[token, w]),
                        TensorMeta {
                            shape: Shape::new(vec![1, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 2: RoPE on the projected query
                    let q = stream.add_op(
                        OpKind::Rope {
                            rotary_dim: head_dim,
                            pos_offset: cache_len,
                            theta: 10_000.0,
                        },
                        SmallVec::from_slice(&[projected]),
                        TensorMeta {
                            shape: Shape::new(vec![1, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 3: Narrow Q to head_dim for single-head attention
                    let q_head = stream.add_op(
                        OpKind::Narrow {
                            axis: 1,
                            start: 0,
                            length: head_dim as i64,
                        },
                        SmallVec::from_slice(&[q]),
                        TensorMeta {
                            shape: Shape::new(vec![1, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 4: Attention against KV cache
                    let output = stream.add_op(
                        OpKind::Attention {
                            scale,
                            causal: false,
                        },
                        SmallVec::from_slice(&[q_head, k_cache, v_cache]),
                        TensorMeta {
                            shape: Shape::new(vec![1, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    stream.eval(output).expect("decode step eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark sustained decode throughput over multiple consecutive tokens.
fn bench_decode_sustained(c: &mut Criterion) {
    let head_dim = 64;
    let num_tokens = 32;
    let cache_start = 64;

    let mut group = c.benchmark_group("decode_sustained");
    group.sample_size(10);

    let q_data: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    group.bench_function("sustained_32_tokens", |bench| {
        bench.iter_batched(
            || {
                // Build KV cache for all positions
                let kv_data: Vec<f32> = (0..(cache_start + num_tokens) * head_dim)
                    .map(|i| ((i as f32) * 0.003).sin())
                    .collect();
                (q_data.clone(), kv_data)
            },
            |(q_data, kv_data)| {
                let scale = 1.0 / (head_dim as f32).sqrt();

                for t in 0..num_tokens {
                    let cache_len = cache_start + t;
                    let stream = Stream::new(Box::new(CpuRefBackend));

                    let q = stream.add_constant(
                        q_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![1, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    let k = stream.add_constant(
                        kv_data[..cache_len * head_dim].to_vec(),
                        TensorMeta {
                            shape: Shape::new(vec![cache_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    let v = stream.add_constant(
                        kv_data[..cache_len * head_dim].to_vec(),
                        TensorMeta {
                            shape: Shape::new(vec![cache_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    let attn = stream.add_op(
                        OpKind::Attention {
                            scale,
                            causal: false,
                        },
                        SmallVec::from_slice(&[q, k, v]),
                        TensorMeta {
                            shape: Shape::new(vec![1, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    stream.eval(attn).expect("decode eval");
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_decode_step, bench_decode_sustained);
criterion_main!(benches);

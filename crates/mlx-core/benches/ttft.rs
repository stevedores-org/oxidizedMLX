//! TTFT (Time to First Token) microbenchmark.
//!
//! Measures compute-bound prefill latency: the time from receiving a prompt
//! (as a sequence of embedded tokens) to producing the first output logit.
//! This simulates the prefill phase of LLM inference where the full prompt
//! is processed in one forward pass.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

/// Simulate a single-layer prefill: embedding lookup → linear → RMSNorm → attention → output projection.
/// This measures the latency of the compute-bound prefill phase.
fn bench_ttft_prefill(c: &mut Criterion) {
    let configs: &[(usize, usize, &str)] = &[
        (32, 128, "small_32tok_128dim"),
        (128, 256, "medium_128tok_256dim"),
        (512, 256, "large_512tok_256dim"),
    ];

    let mut group = c.benchmark_group("ttft_prefill");
    group.sample_size(20);

    for &(seq_len, model_dim, name) in configs {
        // FLOPs estimate: matmul-dominated = 2 * seq * dim * dim (Q/K/V projections + output)
        let flops = 4 * 2 * seq_len * model_dim * model_dim;
        group.throughput(Throughput::Elements(flops as u64));

        // Pre-generate random-ish data
        let input_data: Vec<f32> = (0..seq_len * model_dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let weight_data: Vec<f32> = (0..model_dim * model_dim)
            .map(|i| ((i as f32) * 0.0007).cos())
            .collect();

        group.bench_function(BenchmarkId::new("prefill", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));

                    // Input: [seq_len, model_dim]
                    let x = stream.add_constant(
                        input_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Weight: [model_dim, model_dim] (simulates Q projection)
                    let w = stream.add_constant(
                        weight_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![model_dim as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    (stream, x, w)
                },
                |(stream, x, w)| {
                    // Step 1: Q projection: x @ W -> [seq_len, model_dim]
                    let q = stream.add_op(
                        OpKind::MatMul,
                        SmallVec::from_slice(&[x, w]),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 2: RMSNorm
                    let normed = stream.add_op(
                        OpKind::RmsNorm { eps: 1e-5 },
                        SmallVec::from_slice(&[q]),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 3: SiLU activation
                    let activated = stream.add_op(
                        OpKind::Silu,
                        SmallVec::from_slice(&[normed]),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    // Step 4: Output projection: activated @ W -> [seq_len, model_dim]
                    let output = stream.add_op(
                        OpKind::MatMul,
                        SmallVec::from_slice(&[activated, w]),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, model_dim as i64]),
                            dtype: DType::F32,
                        },
                    );

                    stream.eval(output).expect("prefill eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark the attention computation portion of prefill separately.
/// This isolates the QK^T → softmax → V matmul pipeline.
fn bench_ttft_attention(c: &mut Criterion) {
    let configs: &[(usize, usize, &str)] = &[
        (32, 64, "attn_32tok_64dim"),
        (128, 64, "attn_128tok_64dim"),
        (256, 128, "attn_256tok_128dim"),
    ];

    let mut group = c.benchmark_group("ttft_attention");
    group.sample_size(20);

    for &(seq_len, head_dim, name) in configs {
        // Attention FLOPs: 2*seq*seq*dim (QK^T) + 2*seq*seq*dim (PV)
        let flops = 4 * seq_len * seq_len * head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        let q_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.013).cos())
            .collect();
        let v_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();

        group.bench_function(BenchmarkId::new("attention", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));
                    let q = stream.add_constant(
                        q_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    let k = stream.add_constant(
                        k_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    let v = stream.add_constant(
                        v_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, q, k, v)
                },
                |(stream, q, k, v)| {
                    let attn = stream.add_op(
                        OpKind::Attention {
                            scale,
                            causal: true,
                        },
                        SmallVec::from_slice(&[q, k, v]),
                        TensorMeta {
                            shape: Shape::new(vec![seq_len as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(attn).expect("attention eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ttft_prefill, bench_ttft_attention);
criterion_main!(benches);

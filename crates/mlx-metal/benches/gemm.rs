#[cfg(target_os = "macos")]
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(target_os = "macos")]
use mlx_core::graph::{OpKind, TensorMeta};
#[cfg(target_os = "macos")]
use mlx_core::types::{DType, Shape};

#[cfg(target_os = "macos")]
fn bench_gemm(c: &mut Criterion) {
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1, 4096, 4096, "decode_1x4096x4096"),
        (128, 4096, 4096, "prefill_128x4096x4096"),
        (256, 4096, 11008, "ffn_256x4096x11008"),
        (512, 4096, 4096, "large_512x4096x4096"),
    ];

    let mut group = c.benchmark_group("gemm_f32");

    for &(m, k, n, name) in shapes {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 1000) as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 1000) as f32) * 0.001).collect();

        group.bench_function(BenchmarkId::new("tiled", name), |bench| {
            bench.iter_batched(
                || mlx_metal::metal_stream().expect("Metal should be available"),
                |stream| {
                    let a = stream.add_constant(
                        a_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64, k as i64]),
                            dtype: DType::F32,
                        },
                    );
                    let b = stream.add_constant(
                        b_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![k as i64, n as i64]),
                            dtype: DType::F32,
                        },
                    );
                    let c = stream.add_op(
                        OpKind::MatMul,
                        smallvec::SmallVec::from_slice(&[a, b]),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64, n as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(c).expect("matmul eval");
                    stream.get_buffer(c).expect("buffer")
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group!(benches, bench_gemm);

#[cfg(not(target_os = "macos"))]
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(not(target_os = "macos"))]
fn bench_gemm(_c: &mut Criterion) {}

#[cfg(not(target_os = "macos"))]
criterion_group!(benches, bench_gemm);

criterion_main!(benches);

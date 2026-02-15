#[cfg(target_os = "macos")]
use criterion::{BatchSize, BenchmarkId};
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_os = "macos")]
use mlx_core::graph::{OpKind, TensorMeta};
#[cfg(target_os = "macos")]
use mlx_core::types::{DType, Shape};

#[cfg(target_os = "macos")]
fn bench_norm(c: &mut Criterion) {
    let shapes: &[(usize, usize, &str)] = &[
        (128, 256, "128x256"),
        (256, 512, "256x512"),
        (512, 1024, "512x1024"),
    ];

    let mut group = c.benchmark_group("metal_norm_f32");

    for &(tokens, dim, name) in shapes {
        let numel = tokens * dim;
        let x_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.001).collect();

        group.bench_function(BenchmarkId::new("layer_norm", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = mlx_metal::metal_stream().expect("Metal should be available");
                    let x = stream.add_constant(
                        x_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, x)
                },
                |(stream, x)| {
                    let y = stream.add_op(
                        OpKind::LayerNorm { eps: 1e-5 },
                        smallvec::SmallVec::from_slice(&[x]),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(y).expect("layer_norm eval");
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("rms_norm", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = mlx_metal::metal_stream().expect("Metal should be available");
                    let x = stream.add_constant(
                        x_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, x)
                },
                |(stream, x)| {
                    let y = stream.add_op(
                        OpKind::RmsNorm { eps: 1e-5 },
                        smallvec::SmallVec::from_slice(&[x]),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(y).expect("rms_norm eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

#[cfg(not(target_os = "macos"))]
fn bench_norm(_c: &mut Criterion) {}

criterion_group!(benches, bench_norm);
criterion_main!(benches);

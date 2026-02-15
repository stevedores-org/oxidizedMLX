use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

fn bench_reductions(c: &mut Criterion) {
    let shapes: &[(usize, usize, &str)] = &[
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (512, 512, "512x512"),
    ];

    let mut group = c.benchmark_group("cpu_reductions_f32");

    for &(m, n, name) in shapes {
        let numel = m * n;
        let x_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.001).collect();

        group.bench_function(BenchmarkId::new("sum_all", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));
                    let x = stream.add_constant(
                        x_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64, n as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, x)
                },
                |(stream, x)| {
                    let y = stream.add_op(
                        OpKind::Sum { axis: None },
                        SmallVec::from_slice(&[x]),
                        TensorMeta {
                            shape: Shape::new(vec![1]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(y).expect("sum_all eval");
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("sum_axis1", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));
                    let x = stream.add_constant(
                        x_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64, n as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, x)
                },
                |(stream, x)| {
                    let y = stream.add_op(
                        OpKind::Sum { axis: Some(1) },
                        SmallVec::from_slice(&[x]),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(y).expect("sum_axis eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reductions);
criterion_main!(benches);

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

fn bench_matmul(c: &mut Criterion) {
    let shapes: &[(usize, usize, usize, &str)] = &[
        (1, 256, 256, "decode_1x256x256"),
        (32, 256, 256, "prefill_32x256x256"),
        (64, 512, 512, "mid_64x512x512"),
    ];

    let mut group = c.benchmark_group("cpu_matmul_f32");

    for &(m, k, n, name) in shapes {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        group.bench_function(BenchmarkId::new("matmul", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));
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
                    (stream, a, b)
                },
                |(stream, a, b)| {
                    let c = stream.add_op(
                        OpKind::MatMul,
                        SmallVec::from_slice(&[a, b]),
                        TensorMeta {
                            shape: Shape::new(vec![m as i64, n as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(c).expect("matmul eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);

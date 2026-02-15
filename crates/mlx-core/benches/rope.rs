use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

fn bench_rope(c: &mut Criterion) {
    let shapes: &[(usize, usize, usize, &str)] = &[
        (128, 128, 128, "128x128"),
        (256, 128, 128, "256x128"),
        (512, 64, 64, "512x64"),
    ];

    let mut group = c.benchmark_group("cpu_rope_f32");

    for &(tokens, head_dim, rotary_dim, name) in shapes {
        let numel = tokens * head_dim;
        let x_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.001).collect();

        group.bench_function(BenchmarkId::new("rope", name), |bench| {
            bench.iter_batched(
                || {
                    let stream = Stream::new(Box::new(CpuRefBackend));
                    let x = stream.add_constant(
                        x_data.clone(),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    (stream, x)
                },
                |(stream, x)| {
                    let y = stream.add_op(
                        OpKind::Rope {
                            rotary_dim,
                            pos_offset: 0,
                            theta: 10000.0,
                        },
                        SmallVec::from_slice(&[x]),
                        TensorMeta {
                            shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                            dtype: DType::F32,
                        },
                    );
                    stream.eval(y).expect("rope eval");
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rope);
criterion_main!(benches);

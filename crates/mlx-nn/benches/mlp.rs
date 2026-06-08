use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mlx_core::{Device, Shape, Tensor};
use mlx_nn::{Linear, Mlp, Module};

fn bench_mlp(c: &mut Criterion) {
    let dev = Device::Cpu;
    let batch_size = 32;
    let input_dim = 128;
    let hidden_dim = 512;
    let output_dim = 128;

    let mut group = c.benchmark_group("mlp_f32");

    // Initialize MLP weights
    let w1 = Tensor::zeros(
        &Shape::new(vec![hidden_dim, input_dim]),
        mlx_core::DType::F32,
        &dev,
    )
    .unwrap();
    let w2 = Tensor::zeros(
        &Shape::new(vec![output_dim, hidden_dim]),
        mlx_core::DType::F32,
        &dev,
    )
    .unwrap();

    let l1 = Linear::new(w1, None);
    let l2 = Linear::new(w2, None);
    let mlp = Mlp::new(vec![l1, l2], None);

    let x_data = vec![0.0f32; (batch_size * input_dim) as usize];
    let x = Tensor::from_data_with_dtype(
        x_data,
        &Shape::new(vec![batch_size, input_dim]),
        mlx_core::DType::F32,
        &dev,
    )
    .unwrap();

    group.bench_function(BenchmarkId::new("forward", "32x128x512x128"), |b| {
        b.iter(|| {
            let y = mlp.forward(&x).unwrap();
            // Force eager evaluation of lazy graph
            y.to_vec_f32().unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_mlp);
criterion_main!(benches);

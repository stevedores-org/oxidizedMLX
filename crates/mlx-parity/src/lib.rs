//! Parity testing infrastructure: Metal GPU vs CPU reference backend.
//!
//! Provides deterministic data generation and helpers to run the same op on
//! both backends and compare results within floating-point tolerance.

use mlx_core::backend::Stream;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

/// Floating-point comparison tolerances.
pub struct Tolerance {
    pub atol: f32,
    pub rtol: f32,
}

/// Default tolerance for f32 parity checks.
pub const FP32_TOLERANCE: Tolerance = Tolerance {
    atol: 1e-4,
    rtol: 1e-4,
};

/// Generate deterministic f32 data from a seed. Values in [-1, 1].
pub fn gen_data(seed: u64, len: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

/// Run a MatMul on the CPU reference backend.
pub fn run_matmul_cpu(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    run_matmul_on_stream(&mlx_cpu::cpu_stream(), m, k, n, a, b)
}

/// Run a MatMul on the Metal GPU backend.
#[cfg(target_os = "macos")]
pub fn run_matmul_metal(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let stream = mlx_metal::metal_stream().expect("Metal should be available on macOS");
    run_matmul_on_stream(&stream, m, k, n, a, b)
}

fn run_matmul_on_stream(
    stream: &Arc<Stream>,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let a_node = stream.add_constant(
        a.to_vec(),
        TensorMeta {
            shape: Shape::new(vec![m as i64, k as i64]),
            dtype: DType::F32,
        },
    );
    let b_node = stream.add_constant(
        b.to_vec(),
        TensorMeta {
            shape: Shape::new(vec![k as i64, n as i64]),
            dtype: DType::F32,
        },
    );
    let out = stream.add_op(
        OpKind::MatMul,
        smallvec::SmallVec::from_slice(&[a_node, b_node]),
        TensorMeta {
            shape: Shape::new(vec![m as i64, n as i64]),
            dtype: DType::F32,
        },
    );
    stream.eval(out).expect("eval should succeed");
    stream.get_buffer(out).expect("buffer should exist")
}

/// Run element-wise Add on the CPU reference backend.
pub fn run_add_cpu(len: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    run_add_on_stream(&mlx_cpu::cpu_stream(), len, a, b)
}

/// Run element-wise Add on the Metal GPU backend.
#[cfg(target_os = "macos")]
pub fn run_add_metal(len: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let stream = mlx_metal::metal_stream().expect("Metal should be available on macOS");
    run_add_on_stream(&stream, len, a, b)
}

fn run_add_on_stream(stream: &Arc<Stream>, len: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let a_node = stream.add_constant(
        a.to_vec(),
        TensorMeta {
            shape: Shape::new(vec![len as i64]),
            dtype: DType::F32,
        },
    );
    let b_node = stream.add_constant(
        b.to_vec(),
        TensorMeta {
            shape: Shape::new(vec![len as i64]),
            dtype: DType::F32,
        },
    );
    let out = stream.add_op(
        OpKind::Add,
        smallvec::SmallVec::from_slice(&[a_node, b_node]),
        TensorMeta {
            shape: Shape::new(vec![len as i64]),
            dtype: DType::F32,
        },
    );
    stream.eval(out).expect("eval should succeed");
    stream.get_buffer(out).expect("buffer should exist")
}

/// Compare CPU and Metal results, panicking on mismatch.
pub fn check_parity(name: &str, cpu_result: &[f32], metal_result: &[f32], tol: &Tolerance) {
    assert_eq!(
        cpu_result.len(),
        metal_result.len(),
        "{name}: length mismatch: cpu={} metal={}",
        cpu_result.len(),
        metal_result.len()
    );
    mlx_conformance::assert_allclose(cpu_result, metal_result, tol.atol, tol.rtol);
}

//! Backend Parity Gate — Suite 7.
//!
//! Compares Metal GPU output against CPU reference output for every supported
//! operation. Each test builds the same computation graph on a fresh CPU stream
//! and a fresh Metal stream with identical input data, evaluates both, and
//! asserts element-wise closeness.

use mlx_core::backend::Stream;
use mlx_core::graph::{NodeId, OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;
use std::sync::Arc;

pub use mlx_conformance::assert_allclose;
pub use mlx_cpu::cpu_stream;
pub use mlx_metal::metal_stream;

/// Deterministic f32 data generation using a simple LCG.
///
/// Produces `n` values in roughly [-1, 1] from the given seed.
/// Not cryptographically random — just reproducible across platforms.
pub fn gen_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            // LCG: Numerical Recipes parameters
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            ((state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Generate positive f32 data (for ops like Log that need positive inputs).
pub fn gen_positive_data(n: usize, seed: u64) -> Vec<f32> {
    gen_data(n, seed).iter().map(|x| x.abs() + 0.01).collect()
}

/// Run the same operation on CPU and Metal streams, assert closeness.
///
/// `build_graph` receives a stream and returns the output NodeId.
/// It is called twice — once for CPU, once for Metal.
#[cfg(target_os = "macos")]
pub fn run_parity<F>(build_graph: F, atol: f32, rtol: f32)
where
    F: Fn(&Arc<Stream>) -> NodeId,
{
    let cpu = cpu_stream();
    let metal = metal_stream().expect("Metal required for parity tests");

    let cpu_out = build_graph(&cpu);
    let metal_out = build_graph(&metal);

    cpu.eval(cpu_out).expect("CPU eval failed");
    metal.eval(metal_out).expect("Metal eval failed");

    let cpu_result = cpu.get_buffer(cpu_out).expect("CPU buffer missing");
    let metal_result = metal.get_buffer(metal_out).expect("Metal buffer missing");

    assert_allclose(&metal_result, &cpu_result, atol, rtol);
}

/// Helper: create a TensorMeta from shape dims and F32 dtype.
pub fn meta(dims: &[i64]) -> TensorMeta {
    TensorMeta {
        shape: Shape::new(dims.to_vec()),
        dtype: DType::F32,
    }
}

/// Helper: add a constant and a binary op to a stream, return output NodeId.
pub fn binary_op(stream: &Arc<Stream>, op: OpKind, a: &[f32], b: &[f32], m: TensorMeta) -> NodeId {
    let na = stream.add_constant(a.to_vec(), m.clone());
    let nb = stream.add_constant(b.to_vec(), m.clone());
    stream.add_op(op, SmallVec::from_slice(&[na, nb]), m)
}

/// Helper: add a constant and a unary op to a stream, return output NodeId.
pub fn unary_op(stream: &Arc<Stream>, op: OpKind, a: &[f32], m: TensorMeta) -> NodeId {
    let na = stream.add_constant(a.to_vec(), m.clone());
    stream.add_op(op, SmallVec::from_slice(&[na]), m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_data_deterministic() {
        let a = gen_data(100, 42);
        let b = gen_data(100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_gen_data_different_seeds() {
        let a = gen_data(100, 42);
        let b = gen_data(100, 43);
        assert_ne!(a, b);
    }

    #[test]
    fn test_gen_data_range() {
        let data = gen_data(10000, 1);
        for &x in &data {
            assert!(x >= -1.0 && x <= 1.0, "value {x} out of range");
        }
    }

    #[test]
    fn test_gen_positive_data() {
        let data = gen_positive_data(1000, 1);
        for &x in &data {
            assert!(x > 0.0, "value {x} should be positive");
        }
    }
}

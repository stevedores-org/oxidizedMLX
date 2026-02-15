//! Golden conformance testing infrastructure.
//!
//! Compares Rust tensor operation outputs against Python MLX as the oracle.
//! Uses deterministic seeded RNG and configurable tolerances.

use serde::{Deserialize, Serialize};

/// Spec sent to the Python reference runner.
#[derive(Serialize)]
pub struct OpSpec {
    pub op: String,
    pub seed: u64,
    pub a: TensorSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b: Option<TensorSpec>,
}

/// Tensor data for the reference runner.
#[derive(Serialize)]
pub struct TensorSpec {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

/// Output from the Python reference runner.
#[derive(Deserialize)]
pub struct OpOutput {
    pub out: Vec<f32>,
}

/// Assert two f32 slices are element-wise close.
pub fn assert_allclose(a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: rust={} python={}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "mismatch at [{i}]: rust={x} python={y} diff={diff} tol={tol}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allclose_exact() {
        assert_allclose(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e-6, 1e-6);
    }

    #[test]
    fn test_allclose_within_tolerance() {
        assert_allclose(&[1.0001], &[1.0], 1e-3, 1e-3);
    }

    #[test]
    #[should_panic(expected = "mismatch")]
    fn test_allclose_fails() {
        assert_allclose(&[1.0], &[2.0], 1e-6, 1e-6);
    }
}

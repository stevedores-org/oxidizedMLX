//! Neural network modules for MLX.
//!
//! Provides common building blocks: `Linear`, `Embedding`, `LayerNorm`, and
//! `RMSNorm`. Each module stores its parameters as `Tensor` values and exposes
//! a `forward()` method.

mod linear;
mod norm;

pub use linear::Linear;
pub use norm::{LayerNorm, RmsNorm};

use mlx_core::{Result, Tensor};

/// Trait implemented by all NN modules.
pub trait Module {
    /// Run the forward pass.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::{Device, Shape, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn s(dims: &[i64]) -> Shape {
        Shape::new(dims.to_vec())
    }

    #[test]
    fn test_linear_no_bias() {
        // Linear(in=3, out=2, bias=false): y = x @ W^T
        // W is [out, in] = [2, 3]
        let weight =
            Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &s(&[2, 3]), &cpu()).unwrap();
        let linear = Linear::new(weight, None);

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[1, 3]), &cpu()).unwrap();
        let y = linear.forward(&x).unwrap();
        // [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
        let result = y.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&result, &[1.0, 2.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_linear_with_bias() {
        let weight =
            Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &s(&[2, 3]), &cpu()).unwrap();
        let bias = Tensor::from_f32(&[0.5, -0.5], &s(&[2]), &cpu()).unwrap();
        let linear = Linear::new(weight, Some(bias));

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[1, 3]), &cpu()).unwrap();
        let y = linear.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&result, &[1.5, 1.5], 1e-5, 1e-5);
    }

    #[test]
    fn test_linear_batch() {
        // Batch of 2 vectors, in=2, out=2
        let weight = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &s(&[2, 2]), &cpu()).unwrap();
        let linear = Linear::new(weight, None);

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &s(&[2, 2]), &cpu()).unwrap();
        let y = linear.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        // Identity weight: output = input
        mlx_conformance::assert_allclose(&result, &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(3, 1e-5);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[1, 3]), &cpu()).unwrap();
        let y = ln.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        // Should be normalized: mean ≈ 0, std ≈ 1
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
        mlx_conformance::assert_allclose(&result, &[-1.2247, 0.0, 1.2247], 1e-3, 1e-3);
    }

    #[test]
    fn test_rms_norm() {
        let rn = RmsNorm::new(3, 1e-5);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[1, 3]), &cpu()).unwrap();
        let y = rn.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        // RMS of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602
        let rms = (14.0f32 / 3.0).sqrt();
        let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms];
        mlx_conformance::assert_allclose(&result, &expected, 1e-4, 1e-4);
    }
}

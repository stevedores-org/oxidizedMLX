//! Neural network modules for MLX.
//!
//! Provides common building blocks: `Linear`, `Embedding`, `LayerNorm`, and
//! `RMSNorm`. Each module stores its parameters as `Tensor` values and exposes
//! a `forward()` method.

mod attention;
mod dropout;
mod embed;
mod linear;
mod norm;

pub use attention::MultiHeadAttention;
pub use dropout::Dropout;
pub use embed::Embedding;
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
    fn test_embedding() {
        // Weight: 4 embeddings, dim 3. Row i is [i*3+0, i*3+1, i*3+2] as f32.
        let weight_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight = Tensor::from_f32(&weight_data, &s(&[4, 3]), &cpu()).unwrap();
        let emb = Embedding::new(weight);

        // Indices [0, 2, 1] -> rows 0, 2, 1
        let indices = Tensor::from_f32(&[0.0, 2.0, 1.0], &s(&[3]), &cpu()).unwrap();
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape().0.as_slice(), &[3, 3]);
        let result = y.to_vec_f32().unwrap();
        // Row 0: [0,1,2], Row 2: [6,7,8], Row 1: [3,4,5]
        mlx_conformance::assert_allclose(
            &result,
            &[0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0],
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn test_embedding_batch() {
        // Weight [2, 2]: rows [1,2], [3,4]
        let weight = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &s(&[2, 2]), &cpu()).unwrap();
        let emb = Embedding::new(weight);
        // Indices shape [2, 1]: [[0], [1]]
        let indices = Tensor::from_f32(&[0.0, 1.0], &s(&[2, 1]), &cpu()).unwrap();
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape().0.as_slice(), &[2, 1, 2]);
        let result = y.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&result, &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let mut drop = Dropout::new(0.5);
        drop.eval();
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[3]), &cpu()).unwrap();
        let y = drop.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        // In eval mode, output == input
        mlx_conformance::assert_allclose(&result, &[1.0, 2.0, 3.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_dropout_zero_prob() {
        let drop = Dropout::new(0.0);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[3]), &cpu()).unwrap();
        let y = drop.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&result, &[1.0, 2.0, 3.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_dropout_training_mode() {
        let drop = Dropout::new(0.5);
        let x = Tensor::from_f32(&[1.0; 1000], &s(&[1000]), &cpu()).unwrap();
        let y = drop.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        // In training mode, roughly half should be zero, non-zero should be scaled by 2.0
        let zeros = result.iter().filter(|&&v| v == 0.0).count();
        let non_zeros = result.iter().filter(|&&v| v != 0.0).count();
        assert!(zeros > 300, "expected many zeros, got {zeros}");
        assert!(non_zeros > 300, "expected many non-zeros, got {non_zeros}");
        // Non-zero values should be scaled by 1/(1-0.5) = 2.0
        for &v in result.iter().filter(|&&v| v != 0.0) {
            assert!((v - 2.0).abs() < 1e-5, "expected ~2.0, got {v}");
        }
    }

    #[test]
    fn test_dropout_full_drop() {
        let drop = Dropout::new(1.0);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &s(&[3]), &cpu()).unwrap();
        let y = drop.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&result, &[0.0, 0.0, 0.0], 1e-5, 1e-5);
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

    #[test]
    fn test_multi_head_attention_smoke() {
        // model_dim=4, n_heads=2, head_dim=2, seq_len=2
        // Identity-ish weights for simplicity
        let wq_w = Tensor::from_f32(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &s(&[4, 4]),
            &cpu(),
        )
        .unwrap();
        let wo_w = wq_w.clone();
        let wq = Linear::new(wq_w.clone(), None);
        let wk = Linear::new(wq_w.clone(), None);
        let wv = Linear::new(wq_w, None);
        let wo = Linear::new(wo_w, None);

        let mha = MultiHeadAttention::new(wq, wk, wv, wo, 2);

        let x = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &s(&[2, 4]),
            &cpu(),
        )
        .unwrap();
        let y = mha.forward_causal(&x).unwrap();
        assert_eq!(y.shape(), &s(&[2, 4]));
        // Just verify it runs and produces correct shape
        let result = y.to_vec_f32().unwrap();
        assert_eq!(result.len(), 8);
    }
}

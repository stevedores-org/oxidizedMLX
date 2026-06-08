//! Multi-layer perceptron (MLP) module.

use crate::{Dropout, Linear, Module};
use mlx_core::{Result, Tensor};

/// A Multi-Layer Perceptron (MLP) layer.
///
/// Consists of linear layers with SiLU activations and optional dropout in between.
pub struct Mlp {
    layers: Vec<Linear>,
    dropout: Option<Dropout>,
}

impl Mlp {
    /// Create a new MLP from a list of linear layers and optional dropout probability.
    pub fn new(layers: Vec<Linear>, dropout_p: Option<f32>) -> Self {
        let dropout = dropout_p.map(Dropout::new);
        Self { layers, dropout }
    }

    /// Set the MLP and its dropout to training mode.
    pub fn train(&mut self) {
        if let Some(ref mut d) = self.dropout {
            d.train();
        }
    }

    /// Set the MLP and its dropout to eval mode.
    pub fn eval(&mut self) {
        if let Some(ref mut d) = self.dropout {
            d.eval();
        }
    }

    /// Get a slice of the linear layers.
    pub fn layers(&self) -> &[Linear] {
        &self.layers
    }
}

impl Module for Mlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply activation & dropout to all hidden layers (not output layer)
            if i < num_layers - 1 {
                x = x.silu();

                if let Some(ref d) = self.dropout {
                    x = d.forward(&x)?;
                }
            }
        }

        Ok(x)
    }
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
    fn test_mlp_forward_smoke() {
        // Input: [1, 2]
        // Layer 1: [2, 3], no bias
        // Layer 2: [3, 2], no bias
        let w1 = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &s(&[3, 2]), &cpu()).unwrap();
        let w2 = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &s(&[2, 3]), &cpu()).unwrap();

        let l1 = Linear::new(w1, None);
        let l2 = Linear::new(w2, None);

        let mlp = Mlp::new(vec![l1, l2], None);

        let x = Tensor::from_f32(&[1.0, 2.0], &s(&[1, 2]), &cpu()).unwrap();
        let y = mlp.forward(&x).unwrap();

        assert_eq!(y.shape(), &s(&[1, 2]));
        let result = y.to_vec_f32().unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_mlp_parity_vs_manual() {
        // Rigorous parity check vs manual math
        let dev = cpu();

        // Input x: [1, 2] = [1.0, 2.0]
        // W1: [3, 2] = [[0.5, -0.5], [0.0, 1.0], [2.0, -1.0]]
        // b1: [3] = [0.1, -0.2, 0.3]
        // W2: [2, 3] = [[1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]
        // b2: [2] = [-0.5, 0.5]

        let w1_data = &[0.5, -0.5, 0.0, 1.0, 2.0, -1.0];
        let b1_data = &[0.1, -0.2, 0.3];
        let w2_data = &[1.0, 2.0, 0.0, -1.0, 0.0, 1.0];
        let b2_data = &[-0.5, 0.5];

        let w1 = Tensor::from_f32(w1_data, &s(&[3, 2]), &dev).unwrap();
        let b1 = Tensor::from_f32(b1_data, &s(&[3]), &dev).unwrap();
        let w2 = Tensor::from_f32(w2_data, &s(&[2, 3]), &dev).unwrap();
        let b2 = Tensor::from_f32(b2_data, &s(&[2]), &dev).unwrap();

        let l1 = Linear::new(w1, Some(b1));
        let l2 = Linear::new(w2, Some(b2));

        let mlp = Mlp::new(vec![l1, l2], None);

        let x = Tensor::from_f32(&[1.0, 2.0], &s(&[1, 2]), &dev).unwrap();
        let y = mlp.forward(&x).unwrap();
        let result = y.to_vec_f32().unwrap();

        // Manual computation:
        // x = [1.0, 2.0]
        // z1 = x @ W1^T + b1
        //    z1 = [-0.4, 1.8, 0.3]
        // h1 = silu(z1) ≈ [-0.16052, 1.54467, 0.17233]
        // z2 = h1 @ W2^T + b2 ≈ [2.42882, 0.83285]

        let expected = &[2.42882, 0.83285];
        mlx_conformance::assert_allclose(&result, expected, 1e-4, 1e-4);
    }
}

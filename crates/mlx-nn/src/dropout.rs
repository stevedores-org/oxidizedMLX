//! Dropout module — randomly zeros elements during training.

use mlx_core::{Result, Tensor};
use rand::Rng;

use crate::Module;

/// Dropout: randomly zeros elements with probability `p` during training.
///
/// During training, each element is independently set to zero with probability
/// `p`, and the remaining elements are scaled by `1 / (1 - p)` to preserve the
/// expected value. In eval mode, the input is passed through unchanged.
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// Create a new Dropout with drop probability `p`.
    pub fn new(p: f32) -> Self {
        Self { p, training: true }
    }

    /// Set to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set to eval mode (no dropout).
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let n = input.numel() as usize;
        let mut rng = rand::rng();
        let scale = 1.0 / (1.0 - self.p);
        let mask: Vec<f32> = (0..n)
            .map(|_| {
                if rng.random::<f32>() >= self.p {
                    scale
                } else {
                    0.0
                }
            })
            .collect();
        let mask_t =
            Tensor::from_data_with_dtype(mask, input.shape(), input.dtype(), input.device())?;
        input.mul(&mask_t)
    }
}

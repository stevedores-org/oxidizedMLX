//! Normalization layers: LayerNorm and RMSNorm.

use mlx_core::{Result, Tensor};

use crate::Module;

/// Layer normalization over the last dimension.
///
/// Normalizes to zero mean and unit variance, controlled by `eps` for
/// numerical stability.
pub struct LayerNorm {
    /// Reserved for learnable scale/shift parameters (Phase 2).
    #[allow(dead_code)]
    dim: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { dim, eps }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.layer_norm(self.eps))
    }
}

/// RMS normalization over the last dimension.
///
/// Like LayerNorm but skips the mean-centering step, normalizing only by
/// the root-mean-square.
pub struct RmsNorm {
    /// Reserved for learnable scale parameter (Phase 2).
    #[allow(dead_code)]
    dim: usize,
    eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self { dim, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.rms_norm(self.eps))
    }
}

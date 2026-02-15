//! Normalization layers: LayerNorm and RMSNorm.

use mlx_core::{MlxError, Result, Tensor};

use crate::Module;

/// Layer normalization over the last dimension.
///
/// Normalizes to zero mean and unit variance, controlled by `eps` for
/// numerical stability. The `dim` parameter specifies the expected size
/// of the last dimension and is validated during `forward()`.
pub struct LayerNorm {
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
        let last_dim = *input.shape().0.last().ok_or_else(|| {
            MlxError::InvalidArgument("LayerNorm requires at least 1D input".into())
        })? as usize;
        if last_dim != self.dim {
            return Err(MlxError::InvalidArgument(format!(
                "LayerNorm expected last dim {}, got {}",
                self.dim, last_dim
            )));
        }
        Ok(input.layer_norm(self.eps))
    }
}

/// RMS normalization over the last dimension.
///
/// Like LayerNorm but skips the mean-centering step, normalizing only by
/// the root-mean-square. The `dim` parameter specifies the expected size
/// of the last dimension and is validated during `forward()`.
pub struct RmsNorm {
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
        let last_dim =
            *input.shape().0.last().ok_or_else(|| {
                MlxError::InvalidArgument("RmsNorm requires at least 1D input".into())
            })? as usize;
        if last_dim != self.dim {
            return Err(MlxError::InvalidArgument(format!(
                "RmsNorm expected last dim {}, got {}",
                self.dim, last_dim
            )));
        }
        Ok(input.rms_norm(self.eps))
    }
}

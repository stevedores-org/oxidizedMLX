//! Linear (fully-connected) layer.

use mlx_core::{Result, Tensor};

use crate::Module;

/// A linear (fully-connected) layer: `y = x @ W^T + b`.
///
/// Weight has shape `[out_features, in_features]`. Bias (optional) has shape
/// `[out_features]`.
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Create a new Linear layer from pre-existing weight and bias tensors.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get a reference to the bias tensor (if any).
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // y = input @ weight^T
        let wt = self.weight.transpose(None)?;
        let mut y = input.matmul(&wt)?;
        if let Some(ref bias) = self.bias {
            // Broadcast bias [out_features] to match output [batch, out_features]
            y = y.add(&bias.broadcast_to(y.shape())?)?;
        }
        Ok(y)
    }
}

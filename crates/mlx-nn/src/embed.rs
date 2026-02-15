//! Embedding layer: integer index lookup into a learnable weight matrix.
//!
//! Used for token embeddings in LLMs. Input is a tensor of integer indices
//! (passed as f32 with values 0, 1, …); output is the corresponding rows
//! of the weight matrix.

use mlx_core::{MlxError, Result, Tensor};

use crate::Module;

/// Embedding layer: maps integer indices to vectors.
///
/// Weight has shape `[num_embeddings, embedding_dim]`. Forward accepts a tensor
/// of indices (shape `[*]`, elements are f32 representing 0, 1, 2, …) and
/// returns the embedded vectors with shape `[*, embedding_dim]`.
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// Create an embedding layer from a pre-existing weight tensor.
    ///
    /// Weight must be 2D with shape `[num_embeddings, embedding_dim]`.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Number of embeddings (vocabulary size).
    pub fn num_embeddings(&self) -> i64 {
        self.weight.shape().0[0]
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> i64 {
        self.weight.shape().0[1]
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape().ndim() == 0 {
            return Err(MlxError::InvalidArgument(
                "Embedding input (indices) must have at least one dimension".into(),
            ));
        }
        self.weight.embedding_lookup(input)
    }
}

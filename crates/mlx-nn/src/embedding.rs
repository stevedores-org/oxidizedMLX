//! Embedding layer — looks up rows from a weight matrix by index.

use mlx_core::{Result, Tensor};

use crate::Module;

/// Embedding layer: maps integer indices to dense vectors.
///
/// Weight has shape `[vocab_size, embed_dim]`. Input indices are 1D `[seq_len]`
/// (stored as f32, cast to usize internally). Output is `[seq_len, embed_dim]`.
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// Create a new Embedding from a pre-existing weight tensor `[vocab_size, embed_dim]`.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.weight.embedding_lookup(input)
    }
}

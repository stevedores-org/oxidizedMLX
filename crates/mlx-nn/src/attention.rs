//! Multi-head attention module.

use mlx_core::{Result, Tensor};

use crate::{Linear, Module};

/// Multi-head attention.
///
/// Projects input through Q, K, V linear layers, splits into `n_heads` heads,
/// runs per-head scaled dot-product attention (via the fused `Attention` op),
/// concatenates heads, and applies an output projection.
pub struct MultiHeadAttention {
    n_heads: usize,
    head_dim: usize,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention from pre-built linear layers.
    ///
    /// - `wq`, `wk`, `wv`: projection layers with weight `[n_heads * head_dim, model_dim]`
    /// - `wo`: output projection `[model_dim, n_heads * head_dim]`
    /// - `n_heads`: number of attention heads
    pub fn new(wq: Linear, wk: Linear, wv: Linear, wo: Linear, n_heads: usize) -> Self {
        // Infer head_dim from wq weight shape: [n_heads * head_dim, model_dim]
        let total_dim = wq.weight().shape().0[0] as usize;
        let head_dim = total_dim / n_heads;
        Self {
            n_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
        }
    }

    // TODO: add cross-attention variant that accepts separate key/value inputs.

    /// Forward pass with causal masking (self-attention, auto-regressive).
    ///
    /// `x` has shape `[seq_len, model_dim]`.
    /// Returns `[seq_len, model_dim]`.
    pub fn forward_causal(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape().0[0] as usize;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Project Q, K, V: [seq, model_dim] -> [seq, n_heads * head_dim]
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to [n_heads, seq, head_dim]
        let q = q.reshape(&mlx_core::Shape::new(vec![
            seq_len as i64,
            self.n_heads as i64,
            self.head_dim as i64,
        ]))?;
        let q = q.transpose(Some(&[1, 0, 2]))?; // [n_heads, seq, head_dim]

        let k = k.reshape(&mlx_core::Shape::new(vec![
            seq_len as i64,
            self.n_heads as i64,
            self.head_dim as i64,
        ]))?;
        let k = k.transpose(Some(&[1, 0, 2]))?;

        let v = v.reshape(&mlx_core::Shape::new(vec![
            seq_len as i64,
            self.n_heads as i64,
            self.head_dim as i64,
        ]))?;
        let v = v.transpose(Some(&[1, 0, 2]))?;

        // Per-head attention using narrow to extract each head, then the fused Attention op.
        // TODO: replace with a batched attention op to avoid O(n_heads) graph nodes.
        let mut head_outputs = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let q_h = q.narrow(0, h as i64, 1)?; // [1, seq, head_dim]
            let q_h = q_h.reshape(&mlx_core::Shape::new(vec![
                seq_len as i64,
                self.head_dim as i64,
            ]))?; // [seq, head_dim]

            let k_h = k.narrow(0, h as i64, 1)?;
            let k_h = k_h.reshape(&mlx_core::Shape::new(vec![
                seq_len as i64,
                self.head_dim as i64,
            ]))?;

            let v_h = v.narrow(0, h as i64, 1)?;
            let v_h = v_h.reshape(&mlx_core::Shape::new(vec![
                seq_len as i64,
                self.head_dim as i64,
            ]))?;

            let attn_h = q_h.attention(&k_h, &v_h, scale, true)?; // [seq, head_dim]
            head_outputs.push(attn_h);
        }

        // Concatenate heads: collect [seq, head_dim] tensors -> [seq, n_heads * head_dim]
        let head_refs: Vec<&Tensor> = head_outputs.iter().collect();
        let concat = Tensor::cat(&head_refs, 1)?; // [seq, n_heads * head_dim]

        // Output projection
        // wo expects [seq, n_heads * head_dim] -> [seq, model_dim]
        self.wo.forward(&concat)
    }
}

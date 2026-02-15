//! Rotary positional embeddings (RoPE) for Metal GPU dispatch.

/// Parameters for interleaved RoPE on `[tokens, head_dim]` tensors.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    pub tokens: u32,
    pub head_dim: u32,
    /// Must be even and `<= head_dim`.
    pub rotary_dim: u32,
    pub pos_offset: u32,
    /// Base frequency (typically 10000.0).
    pub theta: f32,
}

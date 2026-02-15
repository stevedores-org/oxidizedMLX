//! Safe, ergonomic Rust API for MLX tensors and operations.
//!
//! `mlx-core` provides the foundational types (`Tensor`, `Device`, `DType`, `Shape`)
//! and a backend-agnostic interface for lazy tensor computation.
//!
//! # Architecture
//!
//! Operations on tensors build a lazy computation graph. Calling `eval()` (or
//! `to_vec_f32()`) triggers a topological walk that dispatches each node to the
//! active `Backend`. A default CPU reference backend is provided out of the box.
//!
//! # Backends
//!
//! - Built-in CPU reference: simple, safe Rust — always available
//! - `ffi` feature: delegates to the MLX C++ runtime via `mlx-sys`
//! - `mlx-cpu` crate: optimized pure-Rust CPU backend (future)
//! - `mlx-metal` crate: native Apple Silicon acceleration (future)

pub mod backend;
pub mod cpu_kernels;
pub mod graph;
pub mod tensor;
pub mod types;

pub use graph::NodeId;
pub use tensor::{Device, Tensor};
pub use types::{DType, Shape};

pub type Result<T> = std::result::Result<T, MlxError>;

#[derive(thiserror::Error, Debug)]
pub enum MlxError {
    #[error("FFI returned null pointer")]
    NullPtr,

    #[error("FFI call failed: {0}")]
    FfiFailed(&'static str),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<i64>, got: Vec<i64> },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Backend not available: {0}")]
    BackendUnavailable(&'static str),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

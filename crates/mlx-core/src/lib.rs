//! Safe, ergonomic Rust API for MLX tensors and operations.
//!
//! `mlx-core` provides the foundational types (`Tensor`, `Device`, `DType`, `Shape`)
//! and a backend-agnostic interface for lazy tensor computation.
//!
//! # Backends
//!
//! - `ffi` feature: delegates to the MLX C++ runtime via `mlx-sys`
//! - CPU backend (future): pure Rust correctness oracle
//! - Metal backend (future): native Apple Silicon acceleration

pub mod graph;
pub mod tensor;
pub mod types;

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
}

//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Provides a minimal Metal runtime for smoke testing GPU dispatch.

use mlx_core::Result;

#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod command;
#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
mod pipeline;

#[cfg(target_os = "macos")]
pub use buffers::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;

#[cfg(not(target_os = "macos"))]
mod stubs {
    use mlx_core::{MlxError, Result};

    /// Stub context for non-macOS platforms.
    pub struct MetalContext;

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Err(MlxError::BackendUnavailable("metal (macOS only)"))
        }

        pub fn device_name(&self) -> String {
            "unsupported".to_string()
        }

        pub fn run_add_u32(&self, _a: &[u32], _b: &[u32]) -> Result<Vec<u32>> {
            Err(MlxError::BackendUnavailable("metal (macOS only)"))
        }
    }

    /// Stub buffer type for non-macOS platforms.
    pub struct MetalBuffer<T> {
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T> MetalBuffer<T> {
        pub fn from_slice_shared(_ctx: &MetalContext, _data: &[T]) -> Result<Self> {
            Err(MlxError::BackendUnavailable("metal (macOS only)"))
        }

        pub fn read_to_vec(&self) -> Vec<T> {
            Vec::new()
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stubs::{MetalBuffer, MetalContext};

/// Convenience wrapper to run the `add_u32` kernel.
#[cfg(target_os = "macos")]
pub fn run_add_u32(a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    let ctx = MetalContext::new()?;
    ctx.run_add_u32(a, b)
}

/// Non-macOS stub for `run_add_u32`.
#[cfg(not(target_os = "macos"))]
pub fn run_add_u32(_a: &[u32], _b: &[u32]) -> Result<Vec<u32>> {
    Err(mlx_core::MlxError::BackendUnavailable("metal (macOS only)"))
}

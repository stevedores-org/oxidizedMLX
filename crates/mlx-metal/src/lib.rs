//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Provides a unified memory buffer model that enables zero-copy CPU/GPU
//! sharing on Apple Silicon via Metal and a minimal runtime for GPU dispatch.

#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod command;
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod instrument;
#[cfg(target_os = "macos")]
mod pipeline;
#[cfg(target_os = "macos")]
pub mod unified;

#[cfg(target_os = "macos")]
pub use buffers::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use instrument::BufferTelemetry;
#[cfg(target_os = "macos")]
pub use unified::{HostAllocation, UnifiedBuffer};

#[cfg(not(target_os = "macos"))]
mod stubs {
    use crate::Result;

    /// Stub context for non-macOS platforms.
    #[derive(Clone, Copy)]
    pub struct MetalContext;

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Err(crate::MetalError::NoDevice)
        }

        pub fn device_name(&self) -> String {
            "unsupported".to_string()
        }

        pub fn run_add_u32(&self, _a: &[u32], _b: &[u32]) -> Result<Vec<u32>> {
            Err(crate::MetalError::NoDevice)
        }
    }

    /// Stub buffer type for non-macOS platforms.
    pub struct MetalBuffer<T> {
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T> MetalBuffer<T> {
        pub fn from_slice_shared(_ctx: &MetalContext, _data: &[T]) -> Result<Self> {
            Err(crate::MetalError::NoDevice)
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
    Err(crate::MetalError::NoDevice)
}

/// Errors arising from Metal buffer operations.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device found")]
    NoDevice,
    #[error("buffer creation failed: {0}")]
    BufferCreationFailed(String),
    #[error("GPU command buffer is in flight; mutable host access denied")]
    GpuInFlight,
    #[error("host pointer is not page-aligned")]
    NotPageAligned,
    #[error("zero-length buffer is not permitted")]
    ZeroLength,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, MetalError>;

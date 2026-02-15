//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Provides a unified memory buffer model that enables zero-copy CPU/GPU
//! sharing on Apple Silicon via Metal's `newBufferWithBytesNoCopy`.

#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod instrument;
#[cfg(target_os = "macos")]
pub mod unified;

#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use instrument::BufferTelemetry;
#[cfg(target_os = "macos")]
pub use unified::{HostAllocation, UnifiedBuffer};

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
}

pub type Result<T> = std::result::Result<T, MetalError>;

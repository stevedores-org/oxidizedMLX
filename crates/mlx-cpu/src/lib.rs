//! Pure Rust CPU backend — correctness oracle for conformance testing.
//!
//! This crate re-exports the built-in CPU reference backend from `mlx-core`
//! and will later provide optimized kernels (SIMD, rayon parallelism) as an
//! upgrade path.

pub use mlx_core::cpu_kernels::CpuRefBackend;

/// Create a new [`mlx_core::backend::Context`] backed by the CPU reference backend.
pub fn cpu_context() -> std::sync::Arc<mlx_core::backend::Context> {
    std::sync::Arc::new(mlx_core::backend::Context::new(Box::new(CpuRefBackend)))
}

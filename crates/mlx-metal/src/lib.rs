//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Provides a unified memory buffer model that enables zero-copy CPU/GPU
//! sharing on Apple Silicon via Metal and a minimal runtime for GPU dispatch.

#[cfg(target_os = "macos")]
mod attention;
#[cfg(target_os = "macos")]
mod backend;
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
pub use backend::MetalBackend;
#[cfg(target_os = "macos")]
pub use buffers::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use instrument::BufferTelemetry;
#[cfg(target_os = "macos")]
pub use unified::{HostAllocation, UnifiedBuffer};

#[cfg(target_os = "macos")]
pub fn metal_stream() -> mlx_core::Result<std::sync::Arc<mlx_core::backend::Stream>> {
    Ok(std::sync::Arc::new(mlx_core::backend::Stream::new(
        Box::new(MetalBackend::new()?),
    )))
}

#[cfg(not(target_os = "macos"))]
mod stubs {
    use mlx_core::Result;

    /// Stub context for non-macOS platforms.
    #[derive(Clone, Copy)]
    pub struct MetalContext;

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Err(mlx_core::MlxError::BackendUnavailable(
                "Metal requires macOS",
            ))
        }

        pub fn device_name(&self) -> String {
            "unsupported".to_string()
        }
    }

    pub struct MetalBackend;
    impl MetalBackend {
        pub fn new() -> Result<Self> {
            Err(mlx_core::MlxError::BackendUnavailable(
                "Metal requires macOS",
            ))
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use stubs::{MetalBackend, MetalContext};

#[cfg(not(target_os = "macos"))]
pub fn metal_stream() -> mlx_core::Result<std::sync::Arc<mlx_core::backend::Stream>> {
    Err(mlx_core::MlxError::BackendUnavailable(
        "Metal is only available on macOS",
    ))
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

impl From<MetalError> for mlx_core::MlxError {
    fn from(e: MetalError) -> Self {
        mlx_core::MlxError::InvalidArgument(format!("Metal error: {}", e))
    }
}

pub type Result<T> = std::result::Result<T, MetalError>;

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use mlx_core::graph::{OpKind, TensorMeta};
    use mlx_core::types::{DType, Shape};

    #[test]
    fn test_metal_add_smoke() {
        let stream = metal_stream().expect("Metal should be available on macOS");

        let a = stream.add_constant(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorMeta {
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );
        let b = stream.add_constant(
            vec![5.0, 6.0, 7.0, 8.0],
            TensorMeta {
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );
        let c = stream.add_op(
            OpKind::Add,
            smallvec::SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );

        stream.eval(c).expect("eval should succeed");
        let result = stream.get_buffer(c).expect("buffer should exist");
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    /// CPU triple-loop reference for MatMul.
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += a[row * k + i] * b[i * n + col];
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    /// Assert element-wise closeness: |a - b| <= atol + rtol * |b|.
    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        let atol = 1e-4_f32;
        let rtol = 1e-4_f32;
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(
                diff <= tol,
                "element {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
            );
        }
    }

    fn run_matmul_test(m: usize, k: usize, n: usize) {
        let stream = metal_stream().expect("Metal should be available on macOS");

        // Deterministic data: simple ascending pattern.
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let expected = cpu_matmul(&a_data, &b_data, m, k, n);

        let a = stream.add_constant(
            a_data,
            TensorMeta {
                shape: Shape::new(vec![m as i64, k as i64]),
                dtype: DType::F32,
            },
        );
        let b = stream.add_constant(
            b_data,
            TensorMeta {
                shape: Shape::new(vec![k as i64, n as i64]),
                dtype: DType::F32,
            },
        );
        let c = stream.add_op(
            OpKind::MatMul,
            smallvec::SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![m as i64, n as i64]),
                dtype: DType::F32,
            },
        );

        stream.eval(c).expect("eval should succeed");
        let result = stream.get_buffer(c).expect("buffer should exist");
        assert_close(&result, &expected);
    }

    #[test]
    fn test_matmul_2x2x2() {
        run_matmul_test(2, 2, 2);
    }

    #[test]
    fn test_matmul_4x4x4() {
        run_matmul_test(4, 4, 4);
    }

    #[test]
    fn test_matmul_64x64x64() {
        run_matmul_test(64, 64, 64);
    }

    #[test]
    fn test_matmul_1x4096x4096() {
        run_matmul_test(1, 4096, 4096);
    }

    #[test]
    fn test_matmul_128x4096x4096() {
        run_matmul_test(128, 4096, 4096);
    }

    #[test]
    fn test_matmul_non_power_of_2() {
        run_matmul_test(17, 31, 23);
    }
}

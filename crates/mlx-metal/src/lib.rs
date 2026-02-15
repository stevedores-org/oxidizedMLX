//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Provides a `MetalBackend` implementing `mlx_core::backend::Backend` that
//! dispatches compute kernels to the GPU via Apple's Metal API. On non-macOS
//! platforms the crate compiles as a stub that returns an error on construction.

// ─── macOS implementation ───────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod metal_impl {
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, DeviceRef,
        MTLResourceOptions, MTLSize,
    };
    use mlx_core::backend::{Backend, NodeInput};
    use mlx_core::graph::{OpKind, TensorMeta};
    use mlx_core::{MlxError, Result};
    use std::sync::Arc;

    #[allow(dead_code)]
    mod gemm;

    /// Metal Shading Language source for element-wise add.
    const ADD_F32_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out      [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
"#;

    /// Owns the Metal device and command queue. Created once per backend.
    #[allow(dead_code)]
    pub struct MetalContext {
        device: Device,
        queue: CommandQueue,
        add_pipeline: ComputePipelineState,
        naive_gemm_pipeline: ComputePipelineState,
        tiled_gemm_pipeline: ComputePipelineState,
    }

    impl MetalContext {
        /// Initialize Metal: find the system GPU, create a command queue, and
        /// compile the built-in kernel library.
        pub fn new() -> Result<Self> {
            let device = Device::system_default()
                .ok_or(MlxError::BackendUnavailable("no Metal GPU found"))?;
            let queue = device.new_command_queue();

            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(ADD_F32_SOURCE, &opts)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal compile error: {e}")))?;

            let add_fn = library
                .get_function("add_f32", None)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal function error: {e}")))?;
            let add_pipeline = device
                .new_compute_pipeline_state_with_function(&add_fn)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal pipeline error: {e}")))?;

            // Compile GEMM kernels.
            let gemm_library = device
                .new_library_with_source(gemm::GEMM_F32_SOURCE, &opts)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal GEMM compile error: {e}")))?;

            let naive_gemm_fn = gemm_library
                .get_function("naive_gemm_f32", None)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal function error: {e}")))?;
            let naive_gemm_pipeline = device
                .new_compute_pipeline_state_with_function(&naive_gemm_fn)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal pipeline error: {e}")))?;

            let tiled_gemm_fn = gemm_library
                .get_function("tiled_gemm_f32", None)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal function error: {e}")))?;
            let tiled_gemm_pipeline = device
                .new_compute_pipeline_state_with_function(&tiled_gemm_fn)
                .map_err(|e| MlxError::InvalidArgument(format!("Metal pipeline error: {e}")))?;

            Ok(Self {
                device,
                queue,
                add_pipeline,
                naive_gemm_pipeline,
                tiled_gemm_pipeline,
            })
        }

        /// Reference to the underlying Metal device.
        pub fn device(&self) -> &DeviceRef {
            &self.device
        }

        /// Create a shared-memory buffer from an `f32` slice.
        fn data_to_buffer(&self, data: &[f32]) -> Buffer {
            // Metal buffers must have a non-zero length. When `data` is empty,
            // allocate a minimal buffer of one `f32` and skip the copy.
            let mut byte_len = std::mem::size_of_val(data) as u64;
            if byte_len == 0 {
                byte_len = std::mem::size_of::<f32>() as u64;
            }
            let buffer = self
                .device
                .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
            if !data.is_empty() {
                unsafe {
                    let dst = buffer.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            }
            buffer
        }
    }

    /// Metal compute backend. Dispatches evaluated ops to the GPU.
    pub struct MetalBackend {
        ctx: Arc<MetalContext>,
    }

    // SAFETY: Metal command buffers are thread-safe once created, and we
    // synchronise via `wait_until_completed` before reading back.
    unsafe impl Send for MetalBackend {}
    unsafe impl Sync for MetalBackend {}

    impl MetalBackend {
        /// Create a new Metal backend, discovering the system GPU.
        pub fn new() -> Result<Self> {
            Ok(Self {
                ctx: Arc::new(MetalContext::new()?),
            })
        }

        /// Element-wise add on the GPU.
        fn eval_add(&self, inputs: &[NodeInput<'_>], meta: &TensorMeta) -> Result<Vec<f32>> {
            if inputs.len() != 2 {
                return Err(MlxError::InvalidArgument(
                    "Add requires exactly 2 inputs".into(),
                ));
            }

            let numel = meta.shape.numel() as usize;

            // Validate input lengths match expected output size.
            if inputs[0].data.len() != numel || inputs[1].data.len() != numel {
                return Err(MlxError::ShapeMismatch {
                    expected: meta.shape.0.clone(),
                    got: vec![inputs[0].data.len() as i64, inputs[1].data.len() as i64],
                });
            }

            // Fast-path: empty tensor — skip GPU dispatch entirely.
            if numel == 0 {
                return Ok(Vec::new());
            }

            let a_buf = self.ctx.data_to_buffer(inputs[0].data);
            let b_buf = self.ctx.data_to_buffer(inputs[1].data);

            let numel_u64 = numel as u64;
            let out_bytes = numel_u64 * std::mem::size_of::<f32>() as u64;
            let out_buf = self
                .ctx
                .device()
                .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

            let cmd_buf = self.ctx.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.ctx.add_pipeline);
            encoder.set_buffer(0, Some(&a_buf), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);

            let thread_group_size = MTLSize::new(
                self.ctx
                    .add_pipeline
                    .thread_execution_width()
                    .min(numel_u64),
                1,
                1,
            );
            let grid_size = MTLSize::new(numel_u64, 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let result = unsafe {
                let ptr = out_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, numel).to_vec()
            };

            Ok(result)
        }

        /// Matrix multiplication on the GPU (tiled kernel).
        fn eval_matmul(&self, inputs: &[NodeInput<'_>], meta: &TensorMeta) -> Result<Vec<f32>> {
            if inputs.len() != 2 {
                return Err(MlxError::InvalidArgument(
                    "MatMul requires exactly 2 inputs".into(),
                ));
            }

            // Both inputs must be 2-D.
            if inputs[0].shape.ndim() != 2 || inputs[1].shape.ndim() != 2 {
                return Err(MlxError::InvalidArgument(
                    "MatMul inputs must be 2-D".into(),
                ));
            }

            let m = inputs[0].shape.0[0] as u32;
            let k = inputs[0].shape.0[1] as u32;
            let k2 = inputs[1].shape.0[0] as u32;
            let n = inputs[1].shape.0[1] as u32;

            if k != k2 {
                return Err(MlxError::ShapeMismatch {
                    expected: vec![m as i64, k as i64],
                    got: vec![k2 as i64, n as i64],
                });
            }

            // Validate that buffer lengths match their declared shapes.
            if inputs[0].data.len() != (m * k) as usize || inputs[1].data.len() != (k * n) as usize {
                return Err(MlxError::InvalidArgument(format!(
                    "MatMul input buffer length mismatch: a={}, b={}",
                    inputs[0].data.len(),
                    inputs[1].data.len()
                )));
            }

            // Validate that the output metadata matches the computed shape.
            if meta.shape.ndim() != 2 || meta.shape.0[0] != m as i64 || meta.shape.0[1] != n as i64 {
                return Err(MlxError::ShapeMismatch {
                    expected: vec![m as i64, n as i64],
                    got: meta.shape.0.clone(),
                });
            }

            let numel = (m as usize) * (n as usize);

            // Fast-path: empty tensor.
            if numel == 0 {
                return Ok(Vec::new());
            }

            let a_buf = self.ctx.data_to_buffer(inputs[0].data);
            let b_buf = self.ctx.data_to_buffer(inputs[1].data);

            let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
            let out_buf = self
                .ctx
                .device()
                .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

            let params = gemm::GemmParams { m, n, k };
            let params_bytes = std::mem::size_of::<gemm::GemmParams>() as u64;
            let params_buf = self
                .ctx
                .device()
                .new_buffer(params_bytes, MTLResourceOptions::StorageModeShared);
            unsafe {
                let dst = params_buf.contents() as *mut gemm::GemmParams;
                *dst = params;
            }

            let cmd_buf = self.ctx.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.ctx.tiled_gemm_pipeline);
            encoder.set_buffer(0, Some(&a_buf), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            encoder.set_buffer(3, Some(&params_buf), 0);

            let thread_groups = MTLSize::new(n.div_ceil(16) as u64, m.div_ceil(16) as u64, 1);
            let threads_per_group = MTLSize::new(16, 16, 1);
            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let result = unsafe {
                let ptr = out_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, numel).to_vec()
            };

            Ok(result)
        }
    }

    impl Backend for MetalBackend {
        fn eval_node(
            &self,
            op: &OpKind,
            inputs: &[NodeInput<'_>],
            meta: &TensorMeta,
        ) -> Result<Vec<f32>> {
            match op {
                OpKind::Constant | OpKind::Parameter => Err(MlxError::InvalidArgument(
                    "Constant/Parameter nodes should be pre-materialized".into(),
                )),
                OpKind::Add => self.eval_add(inputs, meta),
                OpKind::MatMul => self.eval_matmul(inputs, meta),
                _ => Err(MlxError::InvalidArgument(format!(
                    "Metal: unsupported op {:?}",
                    op
                ))),
            }
        }
    }

    /// Create a new [`mlx_core::backend::Stream`] backed by the Metal GPU.
    pub fn metal_stream() -> Result<std::sync::Arc<mlx_core::backend::Stream>> {
        Ok(std::sync::Arc::new(mlx_core::backend::Stream::new(
            Box::new(MetalBackend::new()?),
        )))
    }
}

#[cfg(target_os = "macos")]
pub use metal_impl::{MetalBackend, MetalContext, metal_stream};

// ─── Non-macOS stub ─────────────────────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> mlx_core::Result<Self> {
        Err(mlx_core::MlxError::BackendUnavailable(
            "Metal is only available on macOS",
        ))
    }
}

#[cfg(not(target_os = "macos"))]
pub fn metal_stream() -> mlx_core::Result<std::sync::Arc<mlx_core::backend::Stream>> {
    Err(mlx_core::MlxError::BackendUnavailable(
        "Metal is only available on macOS",
    ))
}

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

    // ── MatMul correctness tests ────────────────────────────────────────────

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

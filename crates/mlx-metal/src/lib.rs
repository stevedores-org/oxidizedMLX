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
    pub struct MetalContext {
        device: Device,
        queue: CommandQueue,
        add_pipeline: ComputePipelineState,
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

            Ok(Self {
                device,
                queue,
                add_pipeline,
            })
        }

        /// Reference to the underlying Metal device.
        pub fn device(&self) -> &DeviceRef {
            &self.device
        }

        /// Create a shared-memory buffer from an `f32` slice.
        fn data_to_buffer(&self, data: &[f32]) -> Buffer {
            let byte_len = std::mem::size_of_val(data) as u64;
            let buffer = self
                .device
                .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
            unsafe {
                let dst = buffer.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
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

            let a_buf = self.ctx.data_to_buffer(inputs[0].data);
            let b_buf = self.ctx.data_to_buffer(inputs[1].data);

            let numel = meta.shape.numel() as u64;
            let out_bytes = numel * std::mem::size_of::<f32>() as u64;
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
                self.ctx.add_pipeline.thread_execution_width().min(numel),
                1,
                1,
            );
            let grid_size = MTLSize::new(numel, 1, 1);
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let result = unsafe {
                let ptr = out_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, numel as usize).to_vec()
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
}

//! Native Metal backend for Apple Silicon GPU acceleration.
//!
//! Replaces the MLX C++ Metal runtime with a Rust-native implementation
//! using `metal-rs` / `objc2`. Includes command queue management, pipeline
//! cache, and buffer allocation for unified memory.

#[cfg(target_os = "macos")]
use metal::*;
use mlx_core::backend::{Backend, NodeInput};
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::{MlxError, Result};

pub struct MetalBackend {
    #[cfg(target_os = "macos")]
    device: Device,
    #[cfg(target_os = "macos")]
    queue: CommandQueue,
}

impl MetalBackend {
    #[cfg(target_os = "macos")]
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        Some(Self { device, queue })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Option<Self> {
        None
    }
}

impl Backend for MetalBackend {
    fn eval_node(
        &self,
        op: &OpKind,
        inputs: &[NodeInput<'_>],
        output_meta: &TensorMeta,
    ) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        {
            match op {
                OpKind::RoPE {
                    base,
                    offset,
                    traditional,
                } => self.run_rope(inputs, *base, *offset, *traditional, output_meta),
                _ => Err(MlxError::InvalidArgument(format!(
                    "Metal backend does not support op: {:?}",
                    op
                ))),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (op, inputs, output_meta);
            Err(MlxError::BackendUnavailable("Metal requires macOS"))
        }
    }
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    fn run_rope(
        &self,
        inputs: &[NodeInput<'_>],
        base: f32,
        offset: usize,
        traditional: bool,
        _output_meta: &TensorMeta,
    ) -> Result<Vec<f32>> {
        let a = inputs
            .get(0)
            .ok_or_else(|| MlxError::InvalidArgument("missing input".into()))?;
        let shape = &a.shape.0;
        let ndim = shape.len();
        if ndim < 2 {
            return Err(MlxError::InvalidArgument(
                "RoPE requires at least 2D tensor".into(),
            ));
        }

        let head_dim = shape[ndim - 1] as u32;
        let seq_len = shape[ndim - 2] as u32;
        let batch_size = (a.data.len() / (seq_len as usize * head_dim as usize)) as u32;

        let source = include_str!("rope.metal");
        let options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(|_| MlxError::FfiFailed("Failed to compile Metal RoPE kernel"))?;
        let func = library
            .get_function("rope_kernel", None)
            .map_err(|_| MlxError::FfiFailed("Failed to get RoPE kernel function"))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|_| MlxError::FfiFailed("Failed to create compute pipeline"))?;

        let in_buffer = self.device.new_buffer_with_data(
            a.data.as_ptr() as *const _,
            (a.data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buffer = self.device.new_buffer(
            (a.data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        #[repr(C)]
        struct RoPEParams {
            head_dim: u32,
            seq_len: u32,
            offset: u32,
            base: f32,
            traditional: bool,
        }

        let params = RoPEParams {
            head_dim,
            seq_len,
            offset: offset as u32,
            base,
            traditional,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&in_buffer), 0);
        encoder.set_buffer(1, Some(&out_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<RoPEParams>() as u64,
            &params as *const _ as *const _,
        );

        let grid_size = MTLSize {
            width: (head_dim / 2) as u64,
            height: seq_len as u64,
            depth: batch_size as u64,
        };
        
        let w = pipeline.thread_execution_width();
        let h = pipeline.max_total_threads_per_threadgroup() / w;
        let threadgroup_size = MTLSize {
            width: w,
            height: h,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = out_buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, a.data.len()).to_vec() };

        Ok(result)
    }
}

pub fn metal_stream() -> Option<std::sync::Arc<mlx_core::backend::Stream>> {
    let backend = MetalBackend::new()?;
    Some(std::sync::Arc::new(mlx_core::backend::Stream::new(Box::new(backend))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::Shape;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_rope() {
        let stream = metal_stream().expect("Metal should be available on macOS");
        let data = [1.0, 0.0, 0.0, 1.0];
        let shape = Shape::new(vec![1, 4]);
        
        let meta = TensorMeta {
            shape: shape.clone(),
            dtype: mlx_core::DType::F32,
        };
        let id_a = stream.add_constant(data.to_vec(), meta.clone());
        let id_rope = stream.add_op(
            OpKind::RoPE {
                base: 10000.0,
                offset: 0,
                traditional: true,
            },
            smallvec::SmallVec::from_slice(&[id_a]),
            meta,
        );
        
        stream.eval(id_rope).unwrap();
        let result = stream.get_buffer(id_rope).unwrap();
        
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }
}

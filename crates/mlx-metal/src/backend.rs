use metal::{MTLResourceOptions, MTLSize};
use mlx_core::backend::{Backend, NodeInput};
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::{MlxError, Result};
use std::sync::Arc;

use crate::context::MetalContext;
use crate::rope::RopeParams;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
}

pub struct MetalBackend {
    ctx: Arc<MetalContext>,
}

// SAFETY: MetalBackend holds an Arc<MetalContext> containing metal::Device and
// metal::CommandQueue. Both are Objective-C ref-counted wrappers whose retain/
// release operations are atomic. Metal command queues are documented as thread-
// safe, and device operations used here (buffer allocation, pipeline creation)
// are safe to call from any thread.
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    pub fn new() -> crate::Result<Self> {
        Ok(Self {
            ctx: Arc::new(MetalContext::new()?),
        })
    }

    fn data_to_buffer(&self, data: &[f32]) -> Result<metal::Buffer> {
        let device = self.ctx.device();
        let mut byte_len = std::mem::size_of_val(data) as u64;
        if byte_len == 0 {
            byte_len = std::mem::size_of::<f32>() as u64;
        }
        let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        if !data.is_empty() {
            let dst = buffer.contents() as *mut f32;
            if dst.is_null() {
                return Err(MlxError::InvalidArgument(
                    "Metal buffer allocation failed (contents pointer is null)".into(),
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        }
        Ok(buffer)
    }

    fn eval_add(&self, inputs: &[NodeInput<'_>], meta: &TensorMeta) -> Result<Vec<f32>> {
        if inputs.len() != 2 {
            return Err(MlxError::InvalidArgument(
                "Add requires exactly 2 inputs".into(),
            ));
        }

        let numel = meta.shape.numel() as usize;

        if inputs[0].data.len() != numel || inputs[1].data.len() != numel {
            return Err(MlxError::ShapeMismatch {
                expected: meta.shape.0.clone(),
                got: vec![inputs[0].data.len() as i64, inputs[1].data.len() as i64],
            });
        }

        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = self.data_to_buffer(inputs[0].data)?;
        let b_buf = self.data_to_buffer(inputs[1].data)?;

        let numel_u64 = numel as u64;
        let out_bytes = numel_u64 * std::mem::size_of::<f32>() as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let pipeline = self
            .ctx
            .pipelines()
            .get_add_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);

        let thread_group_size =
            MTLSize::new(pipeline.thread_execution_width().min(numel_u64), 1, 1);
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

    fn eval_matmul(&self, inputs: &[NodeInput<'_>], meta: &TensorMeta) -> Result<Vec<f32>> {
        if inputs.len() != 2 {
            return Err(MlxError::InvalidArgument(
                "MatMul requires exactly 2 inputs".into(),
            ));
        }

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

        if inputs[0].data.len() != (m * k) as usize || inputs[1].data.len() != (k * n) as usize {
            return Err(MlxError::InvalidArgument(format!(
                "MatMul input buffer length mismatch: a={}, b={}",
                inputs[0].data.len(),
                inputs[1].data.len()
            )));
        }

        if meta.shape.ndim() != 2 || meta.shape.0[0] != m as i64 || meta.shape.0[1] != n as i64 {
            return Err(MlxError::ShapeMismatch {
                expected: vec![m as i64, n as i64],
                got: meta.shape.0.clone(),
            });
        }

        let numel = (m as usize) * (n as usize);

        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = self.data_to_buffer(inputs[0].data)?;
        let b_buf = self.data_to_buffer(inputs[1].data)?;

        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = GemmParams { m, n, k };
        let params_buf = self.ctx.device().new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<GemmParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_tiled_gemm_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&b_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_buffer(3, Some(&params_buf), 0);

        let thread_groups = MTLSize::new((n as u64).div_ceil(16), (m as u64).div_ceil(16), 1);
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

    fn eval_rope(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        rotary_dim: usize,
        pos_offset: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        if inputs.len() != 1 {
            return Err(MlxError::InvalidArgument(
                "Rope requires exactly 1 input".into(),
            ));
        }
        if meta.shape.ndim() != 2 {
            return Err(MlxError::InvalidArgument(
                "Rope input must be 2-D [tokens, head_dim]".into(),
            ));
        }

        let tokens = meta.shape.0[0] as usize;
        let head_dim = meta.shape.0[1] as usize;
        let numel = tokens * head_dim;

        if rotary_dim > head_dim {
            return Err(MlxError::InvalidArgument(
                "rotary_dim must be <= head_dim".into(),
            ));
        }
        if !rotary_dim.is_multiple_of(2) {
            return Err(MlxError::InvalidArgument("rotary_dim must be even".into()));
        }
        if inputs[0].data.len() != numel {
            return Err(MlxError::InvalidArgument(format!(
                "Rope input length mismatch: expected {numel}, got {}",
                inputs[0].data.len()
            )));
        }
        if numel == 0 {
            return Ok(Vec::new());
        }

        let x_buf = self.data_to_buffer(inputs[0].data)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = RopeParams {
            tokens: tokens as u32,
            head_dim: head_dim as u32,
            rotary_dim: rotary_dim as u32,
            pos_offset: pos_offset as u32,
            theta,
        };
        let params_buf = self.ctx.device().new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<RopeParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_rope_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        let numel_u64 = numel as u64;
        let thread_group_size =
            MTLSize::new(pipeline.thread_execution_width().min(numel_u64), 1, 1);
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
            OpKind::Rope {
                rotary_dim,
                pos_offset,
                theta,
            } => self.eval_rope(inputs, meta, *rotary_dim, *pos_offset, *theta),
            _ => Err(MlxError::InvalidArgument(format!(
                "Metal: unsupported op {:?}",
                op
            ))),
        }
    }
}

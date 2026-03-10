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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SoftmaxParams {
    row_size: u32,
    num_rows: u32,
    scale: f32,
    has_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NormParams {
    row_size: u32,
    eps: f32,
}

pub struct MetalBackend {
    pub(crate) ctx: Arc<MetalContext>,
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

    pub(crate) fn data_to_buffer(&self, data: &[f32]) -> Result<metal::Buffer> {
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
        self.eval_elementwise_binary(inputs, meta, "add")
    }

    fn eval_elementwise_binary(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        op_prefix: &str,
    ) -> Result<Vec<f32>> {
        if inputs.len() != 2 {
            return Err(MlxError::InvalidArgument(format!(
                "{} requires exactly 2 inputs",
                op_prefix
            )));
        }

        let numel = meta.shape.numel() as usize;

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

        let pipeline = match op_prefix {
            "add" => self.ctx.pipelines().get_add_f32(self.ctx.device()),
            "sub" => self.ctx.pipelines().get_sub_f32(self.ctx.device()),
            "mul" => self.ctx.pipelines().get_mul_f32(self.ctx.device()),
            "div" => self.ctx.pipelines().get_div_f32(self.ctx.device()),
            _ => {
                return Err(MlxError::InvalidArgument(format!(
                    "Unsupported binary op: {}",
                    op_prefix
                )));
            }
        }
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

    fn eval_elementwise_unary(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        op_prefix: &str,
    ) -> Result<Vec<f32>> {
        if inputs.len() != 1 {
            return Err(MlxError::InvalidArgument(format!(
                "{} requires exactly 1 input",
                op_prefix
            )));
        }

        let numel = meta.shape.numel() as usize;

        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_buf = self.data_to_buffer(inputs[0].data)?;

        let numel_u64 = numel as u64;
        let out_bytes = numel_u64 * std::mem::size_of::<f32>() as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let pipeline = match op_prefix {
            "neg" => self.ctx.pipelines().get_neg_f32(self.ctx.device()),
            "exp" => self.ctx.pipelines().get_exp_f32(self.ctx.device()),
            "log" => self.ctx.pipelines().get_log_f32(self.ctx.device()),
            "silu" => self.ctx.pipelines().get_silu_f32(self.ctx.device()),
            "gelu" => self.ctx.pipelines().get_gelu_f32(self.ctx.device()),
            _ => {
                return Err(MlxError::InvalidArgument(format!(
                    "Unsupported unary op: {}",
                    op_prefix
                )));
            }
        }
        .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);

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

    fn eval_transpose(
        &self,
        inputs: &[NodeInput<'_>],
        _meta: &TensorMeta,
        axes: &Option<Vec<usize>>,
    ) -> Result<Vec<f32>> {
        let x = inputs[0].data;
        let shape = &inputs[0].shape;

        if shape.0.len() != 2 {
            return Err(MlxError::InvalidArgument(
                "Metal: Transpose currently only supports 2D tensors".into(),
            ));
        }

        let is_swap = match axes {
            None => true,
            Some(v) if v.len() == 2 && v[0] == 1 && v[1] == 0 => true,
            _ => false,
        };

        if !is_swap {
            return Err(MlxError::InvalidArgument(
                "Metal: Transpose currently only supports swapping dims 0 and 1".into(),
            ));
        }

        let rows = shape.0[0] as u32;
        let cols = shape.0[1] as u32;
        let numel = (rows * cols) as usize;

        if numel == 0 {
            return Ok(Vec::new());
        }

        let x_buf = self.data_to_buffer(x)?;
        let out_bytes = (numel * 4) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let dims = [rows, cols];
        let dims_buf = self.ctx.device().new_buffer_with_data(
            dims.as_ptr() as *const _,
            8,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_transpose_2d_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&dims_buf), 0);

        encoder.dispatch_threads(
            MTLSize::new(cols as u64, rows as u64, 1),
            MTLSize::new(16, 16, 1),
        );
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { Ok(std::slice::from_raw_parts(out_buf.contents() as *const f32, numel).to_vec()) }
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
            OpKind::Sub => self.eval_elementwise_binary(inputs, meta, "sub"),
            OpKind::Mul => self.eval_elementwise_binary(inputs, meta, "mul"),
            OpKind::Div => self.eval_elementwise_binary(inputs, meta, "div"),
            OpKind::Neg => self.eval_elementwise_unary(inputs, meta, "neg"),
            OpKind::Exp => self.eval_elementwise_unary(inputs, meta, "exp"),
            OpKind::Log => self.eval_elementwise_unary(inputs, meta, "log"),
            OpKind::Silu => self.eval_elementwise_unary(inputs, meta, "silu"),
            OpKind::Gelu => self.eval_elementwise_unary(inputs, meta, "gelu"),
            OpKind::Transpose { axes } => self.eval_transpose(inputs, meta, axes),
            OpKind::MatMul => self.eval_matmul(inputs, meta),
            OpKind::LayerNorm { eps } => self.eval_layer_norm(inputs, meta, *eps),
            OpKind::RmsNorm { eps } => self.eval_rms_norm(inputs, meta, *eps),
            OpKind::ScaledMaskedSoftmax { scale, causal } => {
                self.eval_scaled_masked_softmax(inputs, meta, *scale, *causal)
            }
            OpKind::Attention { scale, causal } => {
                self.eval_attention(inputs, meta, *scale, *causal)
            }
            OpKind::Rope {
                rotary_dim,
                pos_offset,
                theta,
            } => self.eval_rope(inputs, meta, *rotary_dim, *pos_offset, *theta),
            OpKind::Softmax { axis } => self.eval_softmax(inputs, meta, *axis),
            _ => Err(MlxError::InvalidArgument(format!(
                "Metal: unsupported op {:?}",
                op
            ))),
        }
    }
}

impl MetalBackend {
    fn eval_softmax(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        axis: i32,
    ) -> Result<Vec<f32>> {
        let x = inputs[0].data;
        let shape = &meta.shape;
        let numel: usize = shape.0.iter().product::<i64>() as usize;

        if numel == 0 {
            return Ok(Vec::new());
        }

        // Handle negative axis
        let axis = if axis < 0 {
            (shape.0.len() as i32 + axis) as usize
        } else {
            axis as usize
        };

        if axis >= shape.0.len() {
            return Err(MlxError::InvalidArgument(format!(
                "Softmax axis {} out of range for shape {:?}",
                axis, shape
            )));
        }

        let row_size = shape.0[axis] as u32;
        let num_rows = (numel as u32) / row_size;

        let x_buf = self.data_to_buffer(x)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = SoftmaxParams {
            row_size,
            num_rows,
            scale: 1.0,
            has_mask: 0,
        };
        let params_buf = self.ctx.device().new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<SoftmaxParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_scaled_softmax_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);
        // Buffer 3 is mask, not used if has_mask is 0
        encoder.set_buffer(3, None, 0);

        // Dispatch: one thread per row (as implemented in the simple kernel)
        let thread_group_size = MTLSize::new(1, 1, 1);
        let grid_size = MTLSize::new(num_rows as u64, 1, 1);
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

    fn eval_rms_norm(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let x = inputs[0].data;
        let shape = &meta.shape;
        let numel: usize = shape.0.iter().product::<i64>() as usize;
        let row_size = shape.0.last().copied().unwrap_or(1) as u32;
        let num_rows = (numel as u32) / row_size;

        let x_buf = self.data_to_buffer(x)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = NormParams { row_size, eps };
        let params_buf = self.ctx.device().new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<NormParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_rms_norm_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        let grid_size = MTLSize::new(num_rows as u64, 1, 1);
        let thread_group_size = MTLSize::new(1, 1, 1);
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

    fn eval_layer_norm(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let x = inputs[0].data;
        let shape = &meta.shape;
        let numel: usize = shape.0.iter().product::<i64>() as usize;
        let row_size = shape.0.last().copied().unwrap_or(1) as u32;
        let num_rows = (numel as u32) / row_size;

        let x_buf = self.data_to_buffer(x)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = self
            .ctx
            .device()
            .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = NormParams { row_size, eps };
        let params_buf = self.ctx.device().new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<NormParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_layer_norm_f32(self.ctx.device())
            .map_err(|e| MlxError::InvalidArgument(e.to_string()))?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        let grid_size = MTLSize::new(num_rows as u64, 1, 1);
        let thread_group_size = MTLSize::new(1, 1, 1);
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

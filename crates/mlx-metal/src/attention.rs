use metal::{MTLResourceOptions, MTLSize};
use mlx_core::backend::NodeInput;
use mlx_core::graph::TensorMeta;
use mlx_core::{MlxError, Result};

use crate::backend::MetalBackend;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SoftmaxMaskedParams {
    pub tq: u32,
    pub tk: u32,
    pub scale: f32,
    pub causal: u32, // 0 or 1
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TransposeParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
}

impl MetalBackend {
    pub(crate) fn eval_scaled_masked_softmax(
        &self,
        inputs: &[NodeInput<'_>],
        meta: &TensorMeta,
        scale: f32,
        causal: bool,
    ) -> Result<Vec<f32>> {
        if inputs.len() != 1 {
            return Err(MlxError::InvalidArgument(
                "ScaledMaskedSoftmax requires exactly 1 input".into(),
            ));
        }
        if inputs[0].shape.ndim() != 2 {
            return Err(MlxError::InvalidArgument(
                "ScaledMaskedSoftmax requires 2D input".into(),
            ));
        }

        let tq = meta.shape.0[0] as u32;
        let tk = meta.shape.0[1] as u32;
        let numel = (tq as usize) * (tk as usize);

        if numel == 0 {
            return Ok(Vec::new());
        }

        let device = self.ctx.device();
        let scores_buf = self.data_to_buffer(inputs[0].data)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()) as u64;
        let out_buf = device.new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

        let params = SoftmaxMaskedParams {
            tq,
            tk,
            scale,
            causal: u32::from(causal),
        };
        let params_buf = device.new_buffer_with_data(
            &params as *const _ as *const _,
            std::mem::size_of::<SoftmaxMaskedParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipeline = self
            .ctx
            .pipelines()
            .get_softmax_masked_f32(device)
            ?;

        let queue = self.ctx.queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&scores_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        let grid_size = MTLSize::new(tq as u64, 1, 1);
        let thread_group_size = MTLSize::new((tq as u64).min(256), 1, 1);
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

    pub(crate) fn eval_attention(
        &self,
        inputs: &[NodeInput<'_>],
        _meta: &TensorMeta,
        scale: f32,
        causal: bool,
    ) -> Result<Vec<f32>> {
        if inputs.len() != 3 {
            return Err(MlxError::InvalidArgument(
                "Attention requires exactly 3 inputs [Q, K, V]".into(),
            ));
        }

        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];

        if q.shape.ndim() != 2 || k.shape.ndim() != 2 || v.shape.ndim() != 2 {
            return Err(MlxError::InvalidArgument(
                "Attention inputs must be 2D".into(),
            ));
        }

        let tq = q.shape.0[0] as u32;
        let dh = q.shape.0[1] as u32;
        let tk = k.shape.0[0] as u32;
        let dh_k = k.shape.0[1] as u32;

        if dh != dh_k {
            return Err(MlxError::InvalidArgument(format!(
                "Q head_dim {} != K head_dim {}",
                dh, dh_k
            )));
        }

        let out_numel = (tq as usize) * (dh as usize);
        if out_numel == 0 {
            return Ok(Vec::new());
        }

        let device = self.ctx.device();
        let queue = self.ctx.queue();
        let f32_size = std::mem::size_of::<f32>() as u64;

        // Upload Q, K, V to GPU
        let q_buf = self.data_to_buffer(q.data)?;
        let k_buf = self.data_to_buffer(k.data)?;
        let v_buf = self.data_to_buffer(v.data)?;

        // ── Step 1: Transpose K [Tk, Dh] → K^T [Dh, Tk] ──
        let kt_numel = (tk as usize) * (dh as usize);
        let kt_buf = device.new_buffer(
            kt_numel as u64 * f32_size,
            MTLResourceOptions::StorageModeShared,
        );

        let transpose_params = TransposeParams {
            rows: tk,
            cols: dh,
        };
        let transpose_params_buf = device.new_buffer_with_data(
            &transpose_params as *const _ as *const _,
            std::mem::size_of::<TransposeParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let transpose_pipeline = self
            .ctx
            .pipelines()
            .get_transpose_f32(device)
            ?;

        let cmd_buf = queue.new_command_buffer();

        // Dispatch transpose
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&transpose_pipeline);
        encoder.set_buffer(0, Some(&k_buf), 0);
        encoder.set_buffer(1, Some(&kt_buf), 0);
        encoder.set_buffer(2, Some(&transpose_params_buf), 0);

        let grid = MTLSize::new(dh as u64, tk as u64, 1);
        let tg = MTLSize::new((dh as u64).min(16), (tk as u64).min(16), 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();

        // ── Step 2: scores = Q @ K^T [Tq, Dh] @ [Dh, Tk] → [Tq, Tk] ──
        let scores_numel = (tq as usize) * (tk as usize);
        let scores_buf = device.new_buffer(
            scores_numel as u64 * f32_size,
            MTLResourceOptions::StorageModeShared,
        );

        let gemm1_params = GemmParams {
            m: tq,
            n: tk,
            k: dh,
        };
        let gemm1_params_buf = device.new_buffer_with_data(
            &gemm1_params as *const _ as *const _,
            std::mem::size_of::<GemmParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let gemm_pipeline = self
            .ctx
            .pipelines()
            .get_tiled_gemm_f32(device)
            ?;

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&gemm_pipeline);
        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&kt_buf), 0);
        encoder.set_buffer(2, Some(&scores_buf), 0);
        encoder.set_buffer(3, Some(&gemm1_params_buf), 0);

        let thread_groups =
            MTLSize::new((tk as u64).div_ceil(16), (tq as u64).div_ceil(16), 1);
        let threads_per_group = MTLSize::new(16, 16, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();

        // ── Step 3: softmax_masked on scores → probs [Tq, Tk] ──
        let probs_buf = device.new_buffer(
            scores_numel as u64 * f32_size,
            MTLResourceOptions::StorageModeShared,
        );

        let softmax_params = SoftmaxMaskedParams {
            tq,
            tk,
            scale,
            causal: u32::from(causal),
        };
        let softmax_params_buf = device.new_buffer_with_data(
            &softmax_params as *const _ as *const _,
            std::mem::size_of::<SoftmaxMaskedParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let softmax_pipeline = self
            .ctx
            .pipelines()
            .get_softmax_masked_f32(device)
            ?;

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&softmax_pipeline);
        encoder.set_buffer(0, Some(&scores_buf), 0);
        encoder.set_buffer(1, Some(&probs_buf), 0);
        encoder.set_buffer(2, Some(&softmax_params_buf), 0);

        let grid_size = MTLSize::new(tq as u64, 1, 1);
        let thread_group_size = MTLSize::new((tq as u64).min(256), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();

        // ── Step 4: Y = P @ V [Tq, Tk] @ [Tk, Dh] → [Tq, Dh] ──
        let out_buf = device.new_buffer(
            out_numel as u64 * f32_size,
            MTLResourceOptions::StorageModeShared,
        );

        let gemm2_params = GemmParams {
            m: tq,
            n: dh,
            k: tk,
        };
        let gemm2_params_buf = device.new_buffer_with_data(
            &gemm2_params as *const _ as *const _,
            std::mem::size_of::<GemmParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&gemm_pipeline);
        encoder.set_buffer(0, Some(&probs_buf), 0);
        encoder.set_buffer(1, Some(&v_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_buffer(3, Some(&gemm2_params_buf), 0);

        let thread_groups =
            MTLSize::new((dh as u64).div_ceil(16), (tq as u64).div_ceil(16), 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();

        // Commit and wait
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let result = unsafe {
            let ptr = out_buf.contents() as *const f32;
            std::slice::from_raw_parts(ptr, out_numel).to_vec()
        };
        Ok(result)
    }
}

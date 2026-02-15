//! Command encoding helpers.

use metal::{MTLCommandBufferStatus, MTLSize};

use crate::buffers::MetalBuffer;
use crate::context::MetalContext;
use crate::{MetalError, Result};

pub(crate) fn run_add_u32_impl(ctx: &MetalContext, a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(MetalError::InvalidArgument(format!(
            "input length mismatch: a={}, b={}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(Vec::new());
    }

    let a_buf = MetalBuffer::from_slice_shared(ctx, a)?;
    let b_buf = MetalBuffer::from_slice_shared(ctx, b)?;
    let out_buf = MetalBuffer::<u32>::new_shared_uninitialized(ctx, a.len())?;

    let pipeline = ctx.pipelines().get_add_u32(ctx.device())?;

    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a_buf.raw()), 0);
    encoder.set_buffer(1, Some(b_buf.raw()), 0);
    encoder.set_buffer(2, Some(out_buf.raw()), 0);

    let width = a.len() as u64;
    let tew = pipeline.thread_execution_width();
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threads_per_group = tew.min(max_threads).max(1).min(width);

    encoder.dispatch_threads(
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalError::InvalidArgument(format!(
            "command buffer failed with status {:?}",
            command_buffer.status()
        )));
    }

    Ok(out_buf.read_to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SoftmaxParams {
    row_size: u32,
    num_rows: u32,
    scale: f32,
    has_mask: u32,
}

pub(crate) fn run_softmax_f32_impl(
    ctx: &MetalContext,
    input: &[f32],
    row_size: usize,
) -> Result<Vec<f32>> {
    let numel = input.len();
    if numel == 0 {
        return Ok(Vec::new());
    }
    if numel % row_size != 0 {
        return Err(MetalError::InvalidArgument(format!(
            "input length {} not divisible by row_size {}",
            numel, row_size
        )));
    }

    let num_rows = numel / row_size;
    let a_buf = MetalBuffer::from_slice_shared(ctx, input)?;
    let out_buf = MetalBuffer::<f32>::new_shared_uninitialized(ctx, numel)?;

    let params = SoftmaxParams {
        row_size: row_size as u32,
        num_rows: num_rows as u32,
        scale: 1.0,
        has_mask: 0,
    };
    let params_buf = ctx.device().new_buffer_with_data(
        &params as *const _ as *const _,
        std::mem::size_of::<SoftmaxParams>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let pipeline = ctx.pipelines().get_scaled_softmax_f32(ctx.device())?;

    let command_buffer = ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a_buf.raw()), 0);
    encoder.set_buffer(1, Some(out_buf.raw()), 0);
    encoder.set_buffer(2, Some(&params_buf), 0);
    encoder.set_buffer(3, None, 0); // No mask

    // One thread per row
    encoder.dispatch_threads(
        MTLSize {
            width: num_rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalError::InvalidArgument(format!(
            "command buffer failed with status {:?}",
            command_buffer.status()
        )));
    }

    Ok(out_buf.read_to_vec())
}

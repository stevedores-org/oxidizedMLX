//! Command encoding helpers.

use metal::{MTLCommandBufferStatus, MTLSize};
use mlx_core::{MlxError, Result};

use crate::buffers::MetalBuffer;
use crate::context::MetalContext;

pub(crate) fn run_add_u32_impl(ctx: &MetalContext, a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(MlxError::InvalidArgument(format!(
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
        let status = command_buffer.status();
        return Err(MlxError::InvalidArgument(format!(
            "command buffer failed with status {:?}",
            status
        )));
    }

    out_buf.read_to_vec()
}

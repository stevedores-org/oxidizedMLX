use crate::gemm::{gemm_fp16, gemm_fp16_b_transposed, GemmParams};
use crate::{MetalBuffer, MetalContext, MetalError, Result};

#[derive(Clone, Copy, Debug)]
pub struct AttnParams {
    pub tq: usize,
    pub tk: usize,
    pub dh: usize,
    pub scale: f32,
    pub causal: bool,
}

pub fn attention_fp16(
    ctx: &MetalContext,
    q: &MetalBuffer<u16>,
    k: &MetalBuffer<u16>,
    v: &MetalBuffer<u16>,
    out: &MetalBuffer<u16>,
    p: AttnParams,
) -> Result<()> {
    if q.len() != p.tq * p.dh {
        return Err(MetalError::Invalid("q len mismatch"));
    }
    if k.len() != p.tk * p.dh {
        return Err(MetalError::Invalid("k len mismatch"));
    }
    if v.len() != p.tk * p.dh {
        return Err(MetalError::Invalid("v len mismatch"));
    }
    if out.len() != p.tq * p.dh {
        return Err(MetalError::Invalid("out len mismatch"));
    }

    let scores = MetalBuffer::<u16>::new_shared(ctx, p.tq * p.tk)?;
    let probs = MetalBuffer::<u16>::new_shared(ctx, p.tq * p.tk)?;

    gemm_fp16_b_transposed(
        ctx,
        q,
        k,
        &scores,
        GemmParams {
            m: p.tq,
            n: p.tk,
            k: p.dh,
        },
    )?;

    softmax_masked_fp16(ctx, &scores, &probs, p)?;

    gemm_fp16(
        ctx,
        &probs,
        v,
        out,
        GemmParams {
            m: p.tq,
            n: p.dh,
            k: p.tk,
        },
    )?;

    Ok(())
}

fn softmax_masked_fp16(
    ctx: &MetalContext,
    scores: &MetalBuffer<u16>,
    probs: &MetalBuffer<u16>,
    p: AttnParams,
) -> Result<()> {
    let msl = include_str!("kernels/softmax_masked_fp16.metal");
    let pipe = ctx.pipeline_from_msl("softmax_masked_fp16", msl, "softmax_masked_fp16")?;

    let tq_u32 = p.tq as u32;
    let tk_u32 = p.tk as u32;
    let causal_u32 = if p.causal { 1u32 } else { 0u32 };

    ctx.submit_and_wait(|cmd_buf| {
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipe);
        encoder.set_buffer(0, Some(scores.raw()), 0);
        encoder.set_buffer(1, Some(probs.raw()), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &tq_u32 as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &tk_u32 as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &p.scale as *const f32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &causal_u32 as *const u32 as *const _);

        let tg = metal::MTLSize { width: 256, height: 1, depth: 1 };
        let grid = metal::MTLSize { width: p.tq as u64, height: 1, depth: 1 };
        encoder.dispatch_threadgroups(grid, tg);
        encoder.end_encoding();
        Ok(())
    })
}

use crate::{MetalBuffer, MetalContext, MetalError, Result};

#[derive(Clone, Copy, Debug)]
pub struct GemmParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub fn gemm_fp16(
    ctx: &MetalContext,
    a: &MetalBuffer<u16>,
    b: &MetalBuffer<u16>,
    out: &MetalBuffer<u16>,
    p: GemmParams,
) -> Result<()> {
    if a.len() != p.m * p.k {
        return Err(MetalError::Invalid("a len mismatch"));
    }
    if b.len() != p.k * p.n {
        return Err(MetalError::Invalid("b len mismatch"));
    }
    if out.len() != p.m * p.n {
        return Err(MetalError::Invalid("out len mismatch"));
    }

    let msl = include_str!("kernels/gemm_fp16.metal");
    let pipe = ctx.pipeline_from_msl("gemm_fp16", msl, "gemm_fp16")?;

    let m_u32 = p.m as u32;
    let n_u32 = p.n as u32;
    let k_u32 = p.k as u32;

    ctx.submit_and_wait(|cmd_buf| {
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipe);
        encoder.set_buffer(0, Some(a.raw()), 0);
        encoder.set_buffer(1, Some(b.raw()), 0);
        encoder.set_buffer(2, Some(out.raw()), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

        let tg = metal::MTLSize { width: 16, height: 16, depth: 1 };
        let grid = metal::MTLSize {
            width: ((p.n + 15) / 16) as u64,
            height: ((p.m + 15) / 16) as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        Ok(())
    })
}

pub fn gemm_fp16_b_transposed(
    ctx: &MetalContext,
    a: &MetalBuffer<u16>,
    b: &MetalBuffer<u16>,
    out: &MetalBuffer<u16>,
    p: GemmParams,
) -> Result<()> {
    if a.len() != p.m * p.k {
        return Err(MetalError::Invalid("a len mismatch"));
    }
    if b.len() != p.n * p.k {
        return Err(MetalError::Invalid("b len mismatch"));
    }
    if out.len() != p.m * p.n {
        return Err(MetalError::Invalid("out len mismatch"));
    }

    let msl = include_str!("kernels/gemm_abt_fp16.metal");
    let pipe = ctx.pipeline_from_msl("gemm_abt_fp16", msl, "gemm_abt_fp16")?;

    let m_u32 = p.m as u32;
    let n_u32 = p.n as u32;
    let k_u32 = p.k as u32;

    ctx.submit_and_wait(|cmd_buf| {
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipe);
        encoder.set_buffer(0, Some(a.raw()), 0);
        encoder.set_buffer(1, Some(b.raw()), 0);
        encoder.set_buffer(2, Some(out.raw()), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &m_u32 as *const u32 as *const _);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &k_u32 as *const u32 as *const _);

        let tg = metal::MTLSize { width: 16, height: 16, depth: 1 };
        let grid = metal::MTLSize {
            width: ((p.n + 15) / 16) as u64,
            height: ((p.m + 15) / 16) as u64,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        Ok(())
    })
}

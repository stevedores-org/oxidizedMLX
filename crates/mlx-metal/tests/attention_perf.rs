#[cfg(target_os = "macos")]
mod tests {
    use mlx_metal::{attention_fp16, AttnParams, MetalBuffer, MetalContext};
    use std::time::Instant;

    #[test]
    #[ignore]
    fn attention_perf_baseline() {
        let ctx = MetalContext::new().expect("ctx");
        let tq = 128usize;
        let tk = 128usize;
        let dh = 128usize;
        let scale = 1.0 / (dh as f32).sqrt();

        let q = vec![0u16; tq * dh];
        let k = vec![0u16; tk * dh];
        let v = vec![0u16; tk * dh];

        let q = MetalBuffer::from_slice_shared(&ctx, &q).unwrap();
        let k = MetalBuffer::from_slice_shared(&ctx, &k).unwrap();
        let v = MetalBuffer::from_slice_shared(&ctx, &v).unwrap();
        let out = MetalBuffer::<u16>::new_shared(&ctx, tq * dh).unwrap();

        let params = AttnParams {
            tq,
            tk,
            dh,
            scale,
            causal: true,
        };

        for _ in 0..3 {
            attention_fp16(&ctx, &q, &k, &v, &out, params).unwrap();
        }

        let start = Instant::now();
        for _ in 0..10 {
            attention_fp16(&ctx, &q, &k, &v, &out, params).unwrap();
        }
        let elapsed = start.elapsed();
        println!("attention_fp16 {}x{}x{}: {:?}", tq, tk, dh, elapsed);
    }
}

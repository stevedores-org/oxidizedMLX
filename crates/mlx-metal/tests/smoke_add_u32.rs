#[cfg(target_os = "macos")]
mod tests {
    use mlx_metal::MetalContext;

    #[test]
    fn smoke_add_u32() {
        let _ = tracing_subscriber::fmt::try_init();
        let ctx = MetalContext::new().expect("context");

        let a: Vec<u32> = (0..1024).collect();
        let b: Vec<u32> = (1000..2024).collect();
        let expected: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let out = ctx.run_add_u32(&a, &b).expect("run add");
        assert_eq!(out, expected);
    }

    #[test]
    fn smoke_add_u32_len_one() {
        let ctx = MetalContext::new().expect("context");
        let out = ctx.run_add_u32(&[7], &[9]).expect("run add");
        assert_eq!(out, vec![16]);
    }

    #[test]
    fn smoke_add_u32_len_zero() {
        let ctx = MetalContext::new().expect("context");
        let out = ctx.run_add_u32(&[], &[]).expect("run add");
        assert!(out.is_empty());
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn smoke_add_u32_non_macos() {
    // Metal tests are macOS-only; this ensures the test binary compiles on Linux CI.
}

#[cfg(target_os = "macos")]
mod tests {
    use mlx_metal::MetalContext;

    #[test]
    fn test_softmax_f32() {
        let _ = tracing_subscriber::fmt::try_init();
        let ctx = MetalContext::new().expect("context");

        // Input: 2 rows of 4 elements
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let row_size = 4;
        
        let out = ctx.run_softmax_f32(&input, row_size).expect("run softmax");
        
        assert_eq!(out.len(), 8);
        
        // Row 1: softmax([1, 2, 3, 4])
        let row1 = &out[0..4];
        let sum1: f32 = row1.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!(row1[3] > row1[2]);
        assert!(row1[2] > row1[1]);
        assert!(row1[1] > row1[0]);

        // Row 2: softmax([0, 0, 0, 0]) -> [0.25, 0.25, 0.25, 0.25]
        let row2 = &out[4..8];
        let sum2: f32 = row2.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-6);
        for &val in row2 {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }
}

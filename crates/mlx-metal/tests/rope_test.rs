#[cfg(target_os = "macos")]
mod tests {
    use mlx_core::Tensor;
    use mlx_core::types::Shape;
    use mlx_metal::metal_stream;

    #[test]
    fn test_rope() {
        let stream = metal_stream().expect("metal stream");

        // Input: [tokens=2, head_dim=4]
        let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let shape = Shape::new(vec![2, 4]);
        let x = Tensor::from_f32_on_stream(&data, &shape, &stream).unwrap();

        let rotary_dim = 2;
        let pos_offset = 0;
        let theta = 10000.0;

        let out = x.rope(rotary_dim, pos_offset, theta);
        let result = out.to_vec_f32().unwrap();

        assert_eq!(result.len(), 8);

        // Elements 2, 3 and 6, 7 (indices) should be unchanged because rotary_dim=2
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[6], 1.0);
        assert_eq!(result[7], 1.0);

        // Elements 0, 1 should be rotated
        // For t=0: angle = 0, so cos=1, sin=0 -> unchanged
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);

        // Elements 4, 5 (t=1): angle = 1 * 10000^(0) = 1.0
        // cost=cos(1) ≈ 0.5403, sint=sin(1) ≈ 0.8415
        // out[4] = 1.0 * cost - 1.0 * sint ≈ -0.3012
        // out[5] = 1.0 * sint + 1.0 * cost ≈ 1.3818
        assert!((result[4] - (0.5403023 - 0.84147098)).abs() < 1e-4);
        assert!((result[5] - (0.84147098 + 0.5403023)).abs() < 1e-4);
    }
}

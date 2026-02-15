#[cfg(target_os = "macos")]
mod tests {
    use mlx_core::Tensor;
    use mlx_core::types::{Shape};
    use mlx_metal::metal_stream;

    #[test]
    fn test_rms_norm() {
        let stream = metal_stream().expect("metal stream");
        
        // Input: [2, 4]
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        let shape = Shape::new(vec![2, 4]);
        let x = Tensor::from_f32_on_stream(&data, &shape, &stream).unwrap();
        
        let eps = 1e-5;
        let out = x.rms_norm(eps);
        let result = out.to_vec_f32().unwrap();
        
        assert_eq!(result.len(), 8);
        
        // Row 1: rms(1,2,3,4) = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) ≈ 2.7386
        // Row 2: rms(1,1,1,1) = sqrt((1+1+1+1)/4 + eps) = sqrt(1.0 + eps) ≈ 1.0
        
        let row2 = &result[4..8];
        for &val in row2 {
            assert!((val - 1.0f32).abs() < 1e-3f32);
        }
    }

    #[test]
    fn test_layer_norm() {
        let stream = metal_stream().expect("metal stream");
        
        // Input: [1, 4]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![1, 4]);
        let x = Tensor::from_f32_on_stream(&data, &shape, &stream).unwrap();
        
        let eps = 1e-5;
        let out = x.layer_norm(eps);
        let result = out.to_vec_f32().unwrap();
        
        assert_eq!(result.len(), 4);
        
        // Mean = 2.5
        // Var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
        // Std = sqrt(1.25) ≈ 1.118
        
        // Sum should be approx 0
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < 1e-5);
    }
}

#[cfg(target_os = "macos")]
mod tests {
    use mlx_core::Tensor;
    use mlx_core::types::{Shape};
    use mlx_metal::metal_stream;

    #[test]
    fn test_attention_block() {
        let _ = tracing_subscriber::fmt::try_init();
        let stream = metal_stream().expect("metal stream");
        
        let seq_len = 8;
        let d_k = 16;
        
        // Q: [8, 16]
        let q_data: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) / 100.0).collect();
        let q = Tensor::from_f32_on_stream(&q_data, &Shape::new(vec![seq_len as i64, d_k as i64]), &stream).unwrap();
        
        // K: [8, 16]
        let k_data: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32 + 10.0) / 100.0).collect();
        let k = Tensor::from_f32_on_stream(&k_data, &Shape::new(vec![seq_len as i64, d_k as i64]), &stream).unwrap();
        
        // V: [8, 16]
        let v_data: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32 + 20.0) / 100.0).collect();
        let v = Tensor::from_f32_on_stream(&v_data, &Shape::new(vec![seq_len as i64, d_k as i64]), &stream).unwrap();
        
        // 1. Q @ K.T
        let k_t = k.transpose(None).unwrap();
        let scores = q.matmul(&k_t).unwrap(); // [8, 8]
        
        // 2. Scale
        let scale = (d_k as f32).sqrt();
        let scale_data = vec![1.0 / scale; seq_len * seq_len];
        let scale_tensor = Tensor::from_f32_on_stream(&scale_data, &Shape::new(vec![seq_len as i64, seq_len as i64]), &stream).unwrap();
        let scaled_scores = scores.mul(&scale_tensor).unwrap();
        
        // 3. Softmax
        // Use axis -1 (last)
        let probs = scaled_scores.softmax(-1).unwrap();
        
        // 4. Probs @ V
        let out = probs.matmul(&v).unwrap(); // [8, 16]
        
        let result = out.to_vec_f32().unwrap();
        assert_eq!(result.len(), seq_len * d_k);
        
        // Basic check: all values should be positive and finite
        for &val in &result {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }
    }
}

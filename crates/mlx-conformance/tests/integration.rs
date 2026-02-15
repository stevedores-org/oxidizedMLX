use mlx_autograd::value_and_grad;
use mlx_core::{Device, Shape, Tensor};

#[test]
fn test_end_to_end_sgd_linear_regression() {
    // Problem: y = 3x + 2 + noise
    // But we use deterministic small input to test convergence step.
    let device = Device::Cpu;

    // 1. Data
    // X: [[1.0], [2.0], [3.0], [4.0]]
    // Y: [[5.0], [8.0], [11.0], [14.0]] (perfect fit)
    let x_data = [1.0, 2.0, 3.0, 4.0];
    let y_data = [5.0, 8.0, 11.0, 14.0];
    let shape = Shape::new(vec![4, 1]);
    let x = Tensor::from_f32(&x_data, &shape, &device).unwrap();
    let y = Tensor::from_f32(&y_data, &shape, &device).unwrap();

    // 2. Initialize weights to 0.0, bias to 0.0
    // W shape: [out, in] = [1, 1]
    let w_init = Tensor::zeros(&Shape::new(vec![1, 1]), mlx_core::DType::F32, &device).unwrap();
    let b_init = Tensor::zeros(&Shape::new(vec![1]), mlx_core::DType::F32, &device).unwrap();
    
    // Parameters (W is 0.0, true is 3.0; B is 0.0, true is 2.0)
    let mut w = w_init;
    let mut b = b_init;
    let lr = 0.02;
    let n_samples = 4.0;
    let n_tensor = Tensor::from_f32(&[n_samples], &Shape::scalar(), &device).unwrap();
    let lr_tensor = Tensor::from_f32(&[lr], &Shape::scalar(), &device).unwrap();

    for _epoch in 0..1000 {
        // Compute grad for W (treating b as constant for this backward pass)
        let (_, dw) = value_and_grad(|w_arg| {
            // Forward: y = x @ w.T + b
            let w_t = w_arg.transpose(None)?; 
            // Explicit broadcast b to match batch size
            let b_broadcast = b.broadcast_to(&Shape::new(vec![4, 1]))?;
            let y_pred = x.matmul(&w_t)?.add(&b_broadcast)?;
            let diff = y_pred.sub(&y)?;
            let sq = diff.mul(&diff)?;
            let mse = sq.sum_all()?.div(&n_tensor)?;
            Ok(mse)
        }, &w).unwrap();
        
        // Compute grad for b (treating w as constant)
        let (_, db) = value_and_grad(|b_arg| {
            let w_t = w.transpose(None)?;
            let b_broadcast = b_arg.broadcast_to(&Shape::new(vec![4, 1]))?;
            let y_pred = x.matmul(&w_t)?.add(&b_broadcast)?;
            let diff = y_pred.sub(&y)?;
            let sq = diff.mul(&diff)?;
            let mse = sq.sum_all()?.div(&n_tensor)?;
            Ok(mse)
        }, &b).unwrap();

        // Update W: w = w - lr * dw
        let lr_w = lr_tensor.broadcast_to(w.shape()).unwrap();
        let step_w = dw.mul(&lr_w).unwrap();
        w = w.sub(&step_w).unwrap();

        // Update b: b = b - lr * db
        let lr_b = lr_tensor.broadcast_to(b.shape()).unwrap();
        let step_b = db.mul(&lr_b).unwrap();
        b = b.sub(&step_b).unwrap();

        // Detach parameters to prevent infinite graph growth
        let w_data = w.to_vec_f32().unwrap();
        w = Tensor::from_f32(&w_data, &Shape::new(vec![1, 1]), &device).unwrap();

        let b_data = b.to_vec_f32().unwrap();
        b = Tensor::from_f32(&b_data, &Shape::new(vec![1]), &device).unwrap();
    }

    // Check convergence
    let w_final = w.to_vec_f32().unwrap()[0];
    let b_final = b.to_vec_f32().unwrap()[0];

    // Expected: W ~ 3.0, b ~ 2.0
    // With 50 epochs lr=0.01, it should be close.
    // Let's assert strictly enough to prove learning roughly occurred.
    assert!((w_final - 3.0).abs() < 0.5, "W should be ~3.0, got {}", w_final);
    assert!((b_final - 2.0).abs() < 0.5, "b should be ~2.0, got {}", b_final);
}

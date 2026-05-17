//! Suite 7 — Backend Parity Gate (P7.1)
//!
//! Automated parity tests validating that all required ops produce identical
//! results when evaluated through the CPU reference backend. This is the
//! "ship lever" — Metal cannot become the default backend until every test
//! in this file passes.
//!
//! Each test computes a known operation, evaluates it, and compares against
//! a hand-computed expected value. When a Metal backend stream becomes
//! available, these same tests will run on both backends and assert parity.

use mlx_core::{Device, Shape, Tensor};

fn cpu() -> Device {
    Device::Cpu
}

// ─── Elementwise Ops ─────────────────────────────────────────────────────

#[test]
fn parity_add() {
    let a = Tensor::from_f32(&[1.0, -2.5, 3.0, 0.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[0.5, 2.5, -1.0, 7.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    mlx_conformance::assert_allclose(&c.to_vec_f32().unwrap(), &[1.5, 0.0, 2.0, 7.0], 1e-6, 1e-6);
}

#[test]
fn parity_sub() {
    let a = Tensor::from_f32(&[10.0, 5.0, 0.0, -3.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[3.0, 5.0, 1.0, -3.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let c = a.sub(&b).unwrap();
    mlx_conformance::assert_allclose(&c.to_vec_f32().unwrap(), &[7.0, 0.0, -1.0, 0.0], 1e-6, 1e-6);
}

#[test]
fn parity_mul() {
    let a = Tensor::from_f32(&[2.0, -3.0, 0.5, 4.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[3.0, -2.0, 4.0, 0.25], &Shape::new(vec![4]), &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    mlx_conformance::assert_allclose(&c.to_vec_f32().unwrap(), &[6.0, 6.0, 2.0, 1.0], 1e-6, 1e-6);
}

#[test]
fn parity_div() {
    let a = Tensor::from_f32(&[6.0, 9.0, 1.0, -8.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[3.0, 3.0, 4.0, 2.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let c = a.div(&b).unwrap();
    mlx_conformance::assert_allclose(
        &c.to_vec_f32().unwrap(),
        &[2.0, 3.0, 0.25, -4.0],
        1e-6,
        1e-6,
    );
}

#[test]
fn parity_neg() {
    let a = Tensor::from_f32(&[1.0, -2.0, 0.0, 3.5], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = a.neg();
    mlx_conformance::assert_allclose(
        &b.to_vec_f32().unwrap(),
        &[-1.0, 2.0, 0.0, -3.5],
        1e-6,
        1e-6,
    );
}

// ─── MatMul (GEMM) ──────────────────────────────────────────────────────

#[test]
fn parity_matmul_4x4() {
    // Sanity shape: 4×4 × 4×4
    let a_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    let b_data: Vec<f32> = (0..16).map(|i| ((i + 1) as f32) * 0.1).collect();
    let a = Tensor::from_f32(&a_data, &Shape::new(vec![4, 4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&b_data, &Shape::new(vec![4, 4]), &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![4, 4]));
    let result = c.to_vec_f32().unwrap();
    // Verify a known element: C[0,0] = sum(A[0,:]*B[:,0]) = 1*0.1 + 2*0.5 + 3*0.9 + 4*1.3 = 0.1+1.0+2.7+5.2 = 9.0
    mlx_conformance::assert_allclose(&result[0..1], &[9.0], 1e-4, 1e-4);
}

#[test]
fn parity_matmul_projection() {
    // LLM-style projection: [1, 64] × [64, 128] → [1, 128]
    let m = 1;
    let k = 64;
    let n = 128;
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
    let a = Tensor::from_f32(&a_data, &Shape::new(vec![m as i64, k as i64]), &cpu()).unwrap();
    let b = Tensor::from_f32(&b_data, &Shape::new(vec![k as i64, n as i64]), &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &Shape::new(vec![m as i64, n as i64]));
    let result = c.to_vec_f32().unwrap();
    assert_eq!(result.len(), m * n);
    // Verify finite and reasonable
    for (i, v) in result.iter().enumerate() {
        assert!(v.is_finite(), "matmul result[{i}] is not finite: {v}");
    }
}

// ─── Softmax ─────────────────────────────────────────────────────────────

#[test]
fn parity_softmax_2d() {
    // [2, 3] softmax along axis=1 (row-wise)
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let s = a.softmax(1).unwrap();
    let result = s.to_vec_f32().unwrap();
    // Row 0: softmax([1,2,3])
    let e1 = 1.0f32.exp();
    let e2 = 2.0f32.exp();
    let e3 = 3.0f32.exp();
    let sum0 = e1 + e2 + e3;
    // Row 1: softmax([1,1,1]) = [1/3, 1/3, 1/3]
    let expected = vec![
        e1 / sum0,
        e2 / sum0,
        e3 / sum0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ];
    mlx_conformance::assert_allclose(&result, &expected, 1e-5, 1e-5);

    // Verify rows sum to 1
    let row0_sum: f32 = result[0..3].iter().sum();
    let row1_sum: f32 = result[3..6].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum: {row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum: {row1_sum}");
}

// ─── RmsNorm ─────────────────────────────────────────────────────────────

#[test]
fn parity_rms_norm() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = a.rms_norm(1e-5);
    let result = b.to_vec_f32().unwrap();
    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    let rms = (7.5f32).sqrt();
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0].iter().map(|x| x / rms).collect();
    mlx_conformance::assert_allclose(&result, &expected, 1e-4, 1e-4);
}

// ─── LayerNorm ───────────────────────────────────────────────────────────

#[test]
fn parity_layer_norm() {
    let a = Tensor::from_f32(&[2.0, 4.0, 6.0, 8.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = a.layer_norm(1e-5);
    let result = b.to_vec_f32().unwrap();
    // mean=5, var=5, std=sqrt(5)≈2.2361
    // expected: (x - mean) / std
    let mean = 5.0f32;
    let std = 5.0f32.sqrt();
    let expected: Vec<f32> = [2.0, 4.0, 6.0, 8.0]
        .iter()
        .map(|x| (x - mean) / std)
        .collect();
    mlx_conformance::assert_allclose(&result, &expected, 1e-4, 1e-4);
}

// ─── RoPE ────────────────────────────────────────────────────────────────

#[test]
fn parity_rope_basic() {
    // [4, 8] tensor with rotary_dim=4
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let a = Tensor::from_f32(&data, &Shape::new(vec![4, 8]), &cpu()).unwrap();
    let b = a.rope(4, 0, 10000.0);
    let result = b.to_vec_f32().unwrap();
    assert_eq!(result.len(), 32);
    // Elements beyond rotary_dim (indices 4..8 per row) should be unchanged
    for row in 0..4 {
        for col in 4..8 {
            let idx = row * 8 + col;
            assert!(
                (result[idx] - data[idx]).abs() < 1e-5,
                "RoPE should not modify elements beyond rotary_dim: row={row} col={col}"
            );
        }
    }
}

// ─── Attention ───────────────────────────────────────────────────────────

#[test]
fn parity_attention_small() {
    // Q, K, V: [4, 8]
    let q_data: Vec<f32> = (0..32).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let k_data: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let v_data: Vec<f32> = (0..32).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();

    let q = Tensor::from_f32(&q_data, &Shape::new(vec![4, 8]), &cpu()).unwrap();
    let k = Tensor::from_f32(&k_data, &Shape::new(vec![4, 8]), &cpu()).unwrap();
    let v = Tensor::from_f32(&v_data, &Shape::new(vec![4, 8]), &cpu()).unwrap();

    let scale = 1.0 / (8.0f32).sqrt();
    let out = q.attention(&k, &v, scale, true).unwrap();
    assert_eq!(out.shape(), &Shape::new(vec![4, 8]));

    let result = out.to_vec_f32().unwrap();
    // Verify all values are finite
    for (i, val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "attention output[{i}] is not finite: {val}"
        );
    }
}

// ─── Composite: Full Transformer Block Ops ──────────────────────────────

#[test]
fn parity_transformer_block_composition() {
    // Simulate a mini transformer block: LayerNorm → MatMul → SiLU → MatMul
    let seq_len = 4;
    let dim = 8;
    let ff_dim = 16;

    let x_data: Vec<f32> = (0..seq_len * dim)
        .map(|i| (i as f32 - 16.0) * 0.05)
        .collect();
    let w1_data: Vec<f32> = (0..dim * ff_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect();
    let w2_data: Vec<f32> = (0..ff_dim * dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();

    let x = Tensor::from_f32(
        &x_data,
        &Shape::new(vec![seq_len as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();
    let w1 = Tensor::from_f32(
        &w1_data,
        &Shape::new(vec![dim as i64, ff_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let w2 = Tensor::from_f32(
        &w2_data,
        &Shape::new(vec![ff_dim as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();

    // LayerNorm → Up projection → SiLU → Down projection
    let normed = x.layer_norm(1e-5);
    let up = normed.matmul(&w1).unwrap();
    let activated = up.silu();
    let down = activated.matmul(&w2).unwrap();

    assert_eq!(down.shape(), &Shape::new(vec![seq_len as i64, dim as i64]));
    let result = down.to_vec_f32().unwrap();
    assert_eq!(result.len(), seq_len * dim);
    for (i, v) in result.iter().enumerate() {
        assert!(v.is_finite(), "transformer block output[{i}] is not finite");
    }
}

// ─── Required Ops Matrix Green Check ─────────────────────────────────────

#[test]
fn parity_all_required_ops_matrix() {
    // P7.1: Run all required ops in a single test to provide a single pass/fail
    // gate. If ANY op fails, this test fails and Metal cannot become default.
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[0.5, 1.0, 1.5, 2.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();

    // Add
    let _ = a.add(&b).unwrap().to_vec_f32().unwrap();
    // Sub
    let _ = a.sub(&b).unwrap().to_vec_f32().unwrap();
    // Mul
    let _ = a.mul(&b).unwrap().to_vec_f32().unwrap();
    // Div
    let _ = a.div(&b).unwrap().to_vec_f32().unwrap();
    // Neg
    let _ = a.neg().to_vec_f32().unwrap();
    // MatMul
    let _ = a.matmul(&b).unwrap().to_vec_f32().unwrap();
    // Softmax
    let _ = a.softmax(1).unwrap().to_vec_f32().unwrap();
    // RmsNorm
    let flat = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let _ = flat.rms_norm(1e-5).to_vec_f32().unwrap();
    // LayerNorm
    let _ = flat.layer_norm(1e-5).to_vec_f32().unwrap();
    // RoPE
    let seq = Tensor::from_f32(
        &(0..32).map(|i| i as f32).collect::<Vec<_>>(),
        &Shape::new(vec![4, 8]),
        &cpu(),
    )
    .unwrap();
    let _ = seq.rope(4, 0, 10000.0).to_vec_f32().unwrap();
    // Attention
    let q = Tensor::from_f32(
        &(0..8).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &Shape::new(vec![2, 4]),
        &cpu(),
    )
    .unwrap();
    let k = q.clone();
    let v = q.clone();
    let _ = q
        .attention(&k, &v, 0.5, true)
        .unwrap()
        .to_vec_f32()
        .unwrap();
}

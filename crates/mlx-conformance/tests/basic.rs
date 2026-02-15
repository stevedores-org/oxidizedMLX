//! Basic golden conformance tests (Rust lazy graph + CPU backend).
//!
//! These tests validate the Rust lazy graph evaluation against known values.
//! When the Python reference runner is available, these will also compare
//! against Python MLX outputs.

use mlx_core::{Device, Shape, Tensor};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn golden_add() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let b = Tensor::from_f32(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let c = a.add(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    let expected = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
    mlx_conformance::assert_allclose(&result, &expected, 1e-5, 1e-5);
}

#[test]
fn golden_sub() {
    let a = Tensor::from_f32(&[5.0, 7.0, 9.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let c = a.sub(&b).unwrap();
    mlx_conformance::assert_allclose(&c.to_vec_f32().unwrap(), &[4.0, 5.0, 6.0], 1e-6, 1e-6);
}

#[test]
fn golden_mul() {
    let a = Tensor::from_f32(&[2.0, 3.0, 4.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[0.5, 2.0, 0.25], &Shape::new(vec![3]), &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&result, &[1.0, 6.0, 1.0], 1e-6, 1e-6);
}

#[test]
fn golden_div() {
    let a = Tensor::from_f32(&[10.0, 9.0, 8.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[2.0, 3.0, 4.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let c = a.div(&b).unwrap();
    mlx_conformance::assert_allclose(&c.to_vec_f32().unwrap(), &[5.0, 3.0, 2.0], 1e-6, 1e-6);
}

#[test]
fn golden_neg() {
    let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = a.neg();
    mlx_conformance::assert_allclose(&b.to_vec_f32().unwrap(), &[-1.0, 2.0, -3.0], 1e-6, 1e-6);
}

#[test]
fn golden_matmul() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&result, &[19.0, 22.0, 43.0, 50.0], 1e-5, 1e-5);
}

#[test]
fn golden_sum_axis0() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let s = a.sum_axis(0).unwrap();
    mlx_conformance::assert_allclose(&s.to_vec_f32().unwrap(), &[5.0, 7.0, 9.0], 1e-6, 1e-6);
}

#[test]
fn golden_sum_axis1() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let s = a.sum_axis(1).unwrap();
    mlx_conformance::assert_allclose(&s.to_vec_f32().unwrap(), &[6.0, 15.0], 1e-6, 1e-6);
}

#[test]
fn golden_sum_all() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let s = a.sum_all().unwrap();
    mlx_conformance::assert_allclose(&s.to_vec_f32().unwrap(), &[10.0], 1e-6, 1e-6);
}

#[test]
fn golden_softmax() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let s = a.softmax(0).unwrap();
    let result = s.to_vec_f32().unwrap();
    let e1 = 1.0f32.exp();
    let e2 = 2.0f32.exp();
    let e3 = 3.0f32.exp();
    let sum = e1 + e2 + e3;
    let expected = vec![e1 / sum, e2 / sum, e3 / sum];
    mlx_conformance::assert_allclose(&result, &expected, 1e-5, 1e-5);
}

#[test]
fn golden_reshape() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let b = a.reshape(&Shape::new(vec![3, 2])).unwrap();
    assert_eq!(b.shape(), &Shape::new(vec![3, 2]));
    mlx_conformance::assert_allclose(
        &b.to_vec_f32().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-6,
        1e-6,
    );
}

#[test]
fn golden_transpose() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let b = a.transpose(None).unwrap();
    assert_eq!(b.shape(), &Shape::new(vec![3, 2]));
    mlx_conformance::assert_allclose(
        &b.to_vec_f32().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        1e-6,
        1e-6,
    );
}

#[test]
fn golden_lazy_chain() {
    // Verify lazy evaluation: (a + b) * c
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let c = Tensor::from_f32(&[2.0, 2.0, 2.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let d = a.add(&b).unwrap().mul(&c).unwrap();
    mlx_conformance::assert_allclose(&d.to_vec_f32().unwrap(), &[10.0, 14.0, 18.0], 1e-6, 1e-6);
}

#[test]
fn golden_layer_norm() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = a.layer_norm(1e-5);
    let vals = b.to_vec_f32().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / 3.0;
    assert!(
        mean.abs() < 1e-5,
        "layer_norm mean should be ~0, got {mean}"
    );
    mlx_conformance::assert_allclose(&vals, &[-1.2247, 0.0, 1.2247], 1e-3, 1e-3);
}

#[test]
fn golden_silu() {
    let a = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], &Shape::new(vec![4]), &cpu()).unwrap();
    let b = a.silu();
    let vals = b.to_vec_f32().unwrap();
    assert!((vals[0]).abs() < 1e-6); // silu(0) = 0
    assert!((vals[1] - 0.7311).abs() < 1e-3); // silu(1) ≈ 0.7311
    assert!((vals[2] - (-0.2689)).abs() < 1e-3); // silu(-1) ≈ -0.2689
}

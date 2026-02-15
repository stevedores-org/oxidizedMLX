//! Basic golden conformance tests (Rust CPU path).
//!
//! These tests validate the Rust eager CPU implementation against known values.
//! When the Python reference runner is available, these will also compare
//! against Python MLX outputs.

use mlx_core::{Device, Shape, Tensor};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn golden_add() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &Shape::new(vec![2, 3]), &cpu())
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
fn golden_mul() {
    let a = Tensor::from_f32(&[2.0, 3.0, 4.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[0.5, 2.0, 0.25], &Shape::new(vec![3]), &cpu()).unwrap();
    let c = a.mul(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&result, &[1.0, 6.0, 1.0], 1e-6, 1e-6);
}

#[test]
fn golden_matmul() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    mlx_conformance::assert_allclose(&result, &[19.0, 22.0, 43.0, 50.0], 1e-5, 1e-5);
}

#[test]
fn golden_sum() {
    let a = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &cpu(),
    )
    .unwrap();
    let s = a.sum_axis(0).unwrap();
    let result = s.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&result, &[5.0, 7.0, 9.0], 1e-6, 1e-6);
}

#[test]
fn golden_softmax() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let s = a.softmax(0).unwrap();
    let result = s.to_vec_f32().unwrap();
    // Known softmax values for [1,2,3]
    let e1 = 1.0f32.exp();
    let e2 = 2.0f32.exp();
    let e3 = 3.0f32.exp();
    let sum = e1 + e2 + e3;
    let expected = vec![e1 / sum, e2 / sum, e3 / sum];
    mlx_conformance::assert_allclose(&result, &expected, 1e-5, 1e-5);
}

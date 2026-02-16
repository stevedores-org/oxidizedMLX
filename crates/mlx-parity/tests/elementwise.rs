//! Parity tests for elementwise operations: Add, Sub, Mul, Div, Neg, Exp, Log.

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;

const ATOL: f32 = 1e-6;
const RTOL: f32 = 1e-6;

// ── Binary ops ──────────────────────────────────────────────────────────────

#[test]
fn parity_add_1d() {
    let a = gen_data(1024, 42);
    let b = gen_data(1024, 43);
    let m = meta(&[1024]);
    run_parity(|s| binary_op(s, OpKind::Add, &a, &b, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_add_2d() {
    let a = gen_data(64 * 64, 44);
    let b = gen_data(64 * 64, 45);
    let m = meta(&[64, 64]);
    run_parity(|s| binary_op(s, OpKind::Add, &a, &b, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_sub_1d() {
    let a = gen_data(1024, 46);
    let b = gen_data(1024, 47);
    let m = meta(&[1024]);
    run_parity(|s| binary_op(s, OpKind::Sub, &a, &b, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_mul_1d() {
    let a = gen_data(1024, 48);
    let b = gen_data(1024, 49);
    let m = meta(&[1024]);
    run_parity(|s| binary_op(s, OpKind::Mul, &a, &b, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_div_1d() {
    // Use positive data for divisor to avoid div-by-zero
    let a = gen_data(1024, 50);
    let b = gen_positive_data(1024, 51);
    let m = meta(&[1024]);
    run_parity(|s| binary_op(s, OpKind::Div, &a, &b, m.clone()), ATOL, RTOL);
}

// ── Unary ops ───────────────────────────────────────────────────────────────

#[test]
fn parity_neg() {
    let a = gen_data(1024, 52);
    let m = meta(&[1024]);
    run_parity(|s| unary_op(s, OpKind::Neg, &a, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_exp() {
    // Use small values to avoid overflow
    let a: Vec<f32> = gen_data(1024, 53).iter().map(|x| x * 0.5).collect();
    let m = meta(&[1024]);
    run_parity(|s| unary_op(s, OpKind::Exp, &a, m.clone()), ATOL, RTOL);
}

#[test]
fn parity_log() {
    let a = gen_positive_data(1024, 54);
    let m = meta(&[1024]);
    run_parity(|s| unary_op(s, OpKind::Log, &a, m.clone()), ATOL, RTOL);
}

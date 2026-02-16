//! Parity tests for activation functions: Silu, Gelu, Softmax.

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;

const ACT_ATOL: f32 = 1e-5;
const ACT_RTOL: f32 = 1e-5;

#[test]
fn parity_silu_1d() {
    let a = gen_data(256, 60);
    let m = meta(&[256]);
    run_parity(
        |s| unary_op(s, OpKind::Silu, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

#[test]
fn parity_silu_2d() {
    let a = gen_data(32 * 64, 61);
    let m = meta(&[32, 64]);
    run_parity(
        |s| unary_op(s, OpKind::Silu, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

#[test]
fn parity_gelu_1d() {
    let a = gen_data(256, 62);
    let m = meta(&[256]);
    run_parity(
        |s| unary_op(s, OpKind::Gelu, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

#[test]
fn parity_gelu_2d() {
    let a = gen_data(32 * 64, 63);
    let m = meta(&[32, 64]);
    run_parity(
        |s| unary_op(s, OpKind::Gelu, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

#[test]
fn parity_softmax_small() {
    let a = gen_data(4 * 8, 64);
    let m = meta(&[4, 8]);
    run_parity(
        |s| unary_op(s, OpKind::Softmax { axis: -1 }, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

#[test]
fn parity_softmax_large() {
    let a = gen_data(32 * 128, 65);
    let m = meta(&[32, 128]);
    run_parity(
        |s| unary_op(s, OpKind::Softmax { axis: -1 }, &a, m.clone()),
        ACT_ATOL,
        ACT_RTOL,
    );
}

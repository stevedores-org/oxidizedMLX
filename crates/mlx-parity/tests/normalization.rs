//! Parity tests for LayerNorm and RmsNorm.

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-3;
const EPS: f32 = 1e-5;

// ── LayerNorm ───────────────────────────────────────────────────────────────

#[test]
fn parity_layer_norm_4x8() {
    let a = gen_data(4 * 8, 90);
    let m = meta(&[4, 8]);
    run_parity(
        |s| unary_op(s, OpKind::LayerNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_layer_norm_32x128() {
    let a = gen_data(32 * 128, 91);
    let m = meta(&[32, 128]);
    run_parity(
        |s| unary_op(s, OpKind::LayerNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_layer_norm_128x256() {
    let a = gen_data(128 * 256, 92);
    let m = meta(&[128, 256]);
    run_parity(
        |s| unary_op(s, OpKind::LayerNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

// ── RmsNorm ─────────────────────────────────────────────────────────────────

#[test]
fn parity_rms_norm_4x8() {
    let a = gen_data(4 * 8, 93);
    let m = meta(&[4, 8]);
    run_parity(
        |s| unary_op(s, OpKind::RmsNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_rms_norm_32x128() {
    let a = gen_data(32 * 128, 94);
    let m = meta(&[32, 128]);
    run_parity(
        |s| unary_op(s, OpKind::RmsNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_rms_norm_128x256() {
    let a = gen_data(128 * 256, 95);
    let m = meta(&[128, 256]);
    run_parity(
        |s| unary_op(s, OpKind::RmsNorm { eps: EPS }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

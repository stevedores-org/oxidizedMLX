//! Parity tests for ScaledMaskedSoftmax and fused Attention.

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;
use smallvec::SmallVec;

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-3;

// ── ScaledMaskedSoftmax ─────────────────────────────────────────────────────

#[test]
fn parity_scaled_masked_softmax_causal() {
    let tq = 4;
    let tk = 8;
    let a = gen_data(tq * tk, 110);
    let m = meta(&[tq as i64, tk as i64]);
    let scale = 1.0 / (tk as f32).sqrt();
    run_parity(
        |s| unary_op(s, OpKind::ScaledMaskedSoftmax { scale, causal: true }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_scaled_masked_softmax_noncausal() {
    let tq = 16;
    let tk = 16;
    let a = gen_data(tq * tk, 111);
    let m = meta(&[tq as i64, tk as i64]);
    let scale = 1.0 / (tk as f32).sqrt();
    run_parity(
        |s| unary_op(s, OpKind::ScaledMaskedSoftmax { scale, causal: false }, &a, m.clone()),
        ATOL,
        RTOL,
    );
}

// ── Full Attention ──────────────────────────────────────────────────────────

fn run_attention_parity(tq: usize, tk: usize, dh: usize, causal: bool) {
    let q = gen_data(tq * dh, 120);
    let k = gen_data(tk * dh, 121);
    let v = gen_data(tk * dh, 122);
    let q_meta = meta(&[tq as i64, dh as i64]);
    let k_meta = meta(&[tk as i64, dh as i64]);
    let v_meta = meta(&[tk as i64, dh as i64]);
    let out_meta = meta(&[tq as i64, dh as i64]);
    let scale = 1.0 / (dh as f32).sqrt();

    run_parity(
        |s| {
            let nq = s.add_constant(q.clone(), q_meta.clone());
            let nk = s.add_constant(k.clone(), k_meta.clone());
            let nv = s.add_constant(v.clone(), v_meta.clone());
            s.add_op(
                OpKind::Attention { scale, causal },
                SmallVec::from_slice(&[nq, nk, nv]),
                out_meta.clone(),
            )
        },
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_attention_small_causal() {
    run_attention_parity(4, 4, 16, true);
}

#[test]
fn parity_attention_medium_noncausal() {
    run_attention_parity(32, 32, 64, false);
}

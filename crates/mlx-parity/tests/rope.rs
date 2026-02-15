//! Parity tests for Rotary Positional Embeddings (RoPE).

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;

#[test]
fn parity_rope_4x16() {
    let tokens = 4;
    let head_dim = 16;
    let a = gen_data(tokens * head_dim, 100);
    let m = meta(&[tokens as i64, head_dim as i64]);
    run_parity(
        |s| {
            unary_op(
                s,
                OpKind::Rope {
                    rotary_dim: head_dim,
                    pos_offset: 0,
                    theta: 10000.0,
                },
                &a,
                m.clone(),
            )
        },
        5e-4,
        5e-3,
    );
}

#[test]
fn parity_rope_128x128() {
    let tokens = 128;
    let head_dim = 128;
    let a = gen_data(tokens * head_dim, 101);
    let m = meta(&[tokens as i64, head_dim as i64]);
    run_parity(
        |s| {
            unary_op(
                s,
                OpKind::Rope {
                    rotary_dim: head_dim,
                    pos_offset: 0,
                    theta: 10000.0,
                },
                &a,
                m.clone(),
            )
        },
        1e-3,
        1e-2,
    );
}

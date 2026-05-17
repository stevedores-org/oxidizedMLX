//! Parity tests for Transpose (2D).

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;

// Transpose should be exact (just data movement, no arithmetic).
const ATOL: f32 = 0.0;
const RTOL: f32 = 0.0;

#[test]
fn parity_transpose_4x8() {
    let a = gen_data(4 * 8, 130);
    let in_meta = meta(&[4, 8]);
    let out_meta = meta(&[8, 4]);
    run_parity(
        |s| {
            let na = s.add_constant(a.clone(), in_meta.clone());
            s.add_op(
                OpKind::Transpose { axes: None },
                smallvec::SmallVec::from_slice(&[na]),
                out_meta.clone(),
            )
        },
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_transpose_64x128() {
    let a = gen_data(64 * 128, 131);
    let in_meta = meta(&[64, 128]);
    let out_meta = meta(&[128, 64]);
    run_parity(
        |s| {
            let na = s.add_constant(a.clone(), in_meta.clone());
            s.add_op(
                OpKind::Transpose { axes: None },
                smallvec::SmallVec::from_slice(&[na]),
                out_meta.clone(),
            )
        },
        ATOL,
        RTOL,
    );
}

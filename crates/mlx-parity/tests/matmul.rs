//! Parity tests for MatMul at various sizes.

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;
use smallvec::SmallVec;

const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-4;

fn run_matmul_parity(m: i64, k: i64, n: i64) {
    let a = gen_data((m * k) as usize, 70 + m as u64);
    let b = gen_data((k * n) as usize, 80 + n as u64);
    let a_meta = meta(&[m, k]);
    let b_meta = meta(&[k, n]);
    let out_meta = meta(&[m, n]);

    run_parity(
        |s| {
            let na = s.add_constant(a.clone(), a_meta.clone());
            let nb = s.add_constant(b.clone(), b_meta.clone());
            s.add_op(OpKind::MatMul, SmallVec::from_slice(&[na, nb]), out_meta.clone())
        },
        ATOL,
        RTOL,
    );
}

#[test]
fn parity_matmul_4x4() {
    run_matmul_parity(4, 4, 4);
}

#[test]
fn parity_matmul_64x64() {
    run_matmul_parity(64, 64, 64);
}

#[test]
fn parity_matmul_128x4096() {
    run_matmul_parity(128, 4096, 128);
}

#[test]
fn parity_matmul_non_pow2() {
    run_matmul_parity(17, 31, 23);
}

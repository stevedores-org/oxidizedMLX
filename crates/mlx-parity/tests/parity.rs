//! Parity tests: Metal GPU backend vs CPU reference backend.
//!
//! All tests are gated on macOS since Metal is only available there.

#![cfg(target_os = "macos")]

use mlx_parity::{
    FP32_TOLERANCE, check_parity, gen_data, run_add_cpu, run_add_metal, run_matmul_cpu,
    run_matmul_metal,
};

// ── MatMul parity ───────────────────────────────────────────────────────────

fn matmul_parity(m: usize, k: usize, n: usize, seed: u64) {
    let a = gen_data(seed, m * k);
    let b = gen_data(seed + 1, k * n);

    let cpu = run_matmul_cpu(m, k, n, &a, &b);
    let metal = run_matmul_metal(m, k, n, &a, &b);

    check_parity(
        &format!("matmul_{m}x{k}x{n}"),
        &cpu,
        &metal,
        &FP32_TOLERANCE,
    );
}

#[test]
fn parity_matmul_4x4x4() {
    matmul_parity(4, 4, 4, 42);
}

#[test]
fn parity_matmul_128x128x128() {
    matmul_parity(128, 128, 128, 100);
}

#[test]
fn parity_matmul_128x4096x4096() {
    matmul_parity(128, 4096, 4096, 200);
}

#[test]
fn parity_matmul_17x31x23() {
    matmul_parity(17, 31, 23, 300);
}

// ── Add parity ──────────────────────────────────────────────────────────────

fn add_parity(len: usize, seed: u64) {
    let a = gen_data(seed, len);
    let b = gen_data(seed + 1, len);

    let cpu = run_add_cpu(len, &a, &b);
    let metal = run_add_metal(len, &a, &b);

    check_parity(&format!("add_{len}"), &cpu, &metal, &FP32_TOLERANCE);
}

#[test]
fn parity_add_small() {
    add_parity(256, 500);
}

#[test]
fn parity_add_large() {
    add_parity(1_000_000, 600);
}

// ── RmsNorm parity (ignored until Metal implements it) ──────────────────────

#[test]
#[ignore = "Metal does not yet implement RmsNorm"]
fn parity_rmsnorm_2x128() {
    // TODO: implement when Metal supports RmsNorm
}

#[test]
#[ignore = "Metal does not yet implement RmsNorm"]
fn parity_rmsnorm_128x4096() {
    // TODO: implement when Metal supports RmsNorm
}

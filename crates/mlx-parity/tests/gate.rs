//! Backend Parity Gate — The Ship Lever.
//!
//! This is the acceptance test for promoting Metal to default backend.
//! It runs every parity check in sequence. All must pass.
//!
//! Run with: `cargo test -p mlx-parity`

#![cfg(target_os = "macos")]

use mlx_core::graph::OpKind;
use mlx_parity::*;
use smallvec::SmallVec;

/// The gate test: exercises all supported Metal ops against CPU reference.
#[test]
fn backend_parity_gate() {
    // ── Elementwise binary ──────────────────────────────────────────────
    let a = gen_data(1024, 1);
    let b = gen_data(1024, 2);
    let b_pos = gen_positive_data(1024, 3);
    let m1k = meta(&[1024]);

    for (name, op) in [
        ("Add", OpKind::Add),
        ("Sub", OpKind::Sub),
        ("Mul", OpKind::Mul),
    ] {
        let a = a.clone();
        let b = b.clone();
        let m = m1k.clone();
        run_parity(|s| binary_op(s, op.clone(), &a, &b, m.clone()), 1e-6, 1e-6);
        eprintln!("  PASS {name}");
    }

    {
        let a = a.clone();
        let b = b_pos.clone();
        let m = m1k.clone();
        run_parity(|s| binary_op(s, OpKind::Div, &a, &b, m.clone()), 1e-6, 1e-6);
        eprintln!("  PASS Div");
    }

    // ── Elementwise unary ───────────────────────────────────────────────
    {
        let a = a.clone();
        let m = m1k.clone();
        run_parity(|s| unary_op(s, OpKind::Neg, &a, m.clone()), 1e-6, 1e-6);
        eprintln!("  PASS Neg");
    }
    {
        let a: Vec<f32> = gen_data(1024, 4).iter().map(|x| x * 0.5).collect();
        let m = m1k.clone();
        run_parity(|s| unary_op(s, OpKind::Exp, &a, m.clone()), 1e-6, 1e-6);
        eprintln!("  PASS Exp");
    }
    {
        let a = gen_positive_data(1024, 5);
        let m = m1k.clone();
        run_parity(|s| unary_op(s, OpKind::Log, &a, m.clone()), 1e-6, 1e-6);
        eprintln!("  PASS Log");
    }

    // ── Activations ─────────────────────────────────────────────────────
    {
        let a = gen_data(256, 10);
        let m = meta(&[256]);
        run_parity(|s| unary_op(s, OpKind::Silu, &a, m.clone()), 1e-5, 1e-5);
        eprintln!("  PASS Silu");
    }
    {
        let a = gen_data(256, 11);
        let m = meta(&[256]);
        run_parity(|s| unary_op(s, OpKind::Gelu, &a, m.clone()), 1e-5, 1e-5);
        eprintln!("  PASS Gelu");
    }
    {
        let a = gen_data(32 * 128, 12);
        let m = meta(&[32, 128]);
        run_parity(
            |s| unary_op(s, OpKind::Softmax { axis: -1 }, &a, m.clone()),
            1e-5,
            1e-5,
        );
        eprintln!("  PASS Softmax");
    }

    // ── MatMul ──────────────────────────────────────────────────────────
    for (m_dim, k_dim, n_dim) in [(4, 4, 4), (64, 64, 64), (17, 31, 23)] {
        let a = gen_data(m_dim * k_dim, 20);
        let b = gen_data(k_dim * n_dim, 21);
        let am = meta(&[m_dim as i64, k_dim as i64]);
        let bm = meta(&[k_dim as i64, n_dim as i64]);
        let om = meta(&[m_dim as i64, n_dim as i64]);
        run_parity(
            |s| {
                let na = s.add_constant(a.clone(), am.clone());
                let nb = s.add_constant(b.clone(), bm.clone());
                s.add_op(OpKind::MatMul, SmallVec::from_slice(&[na, nb]), om.clone())
            },
            1e-4,
            1e-4,
        );
        eprintln!("  PASS MatMul {m_dim}x{k_dim}x{n_dim}");
    }

    // ── Normalization ───────────────────────────────────────────────────
    {
        let a = gen_data(32 * 128, 30);
        let m = meta(&[32, 128]);
        run_parity(
            |s| unary_op(s, OpKind::LayerNorm { eps: 1e-5 }, &a, m.clone()),
            1e-3,
            1e-3,
        );
        eprintln!("  PASS LayerNorm");
    }
    {
        let a = gen_data(32 * 128, 31);
        let m = meta(&[32, 128]);
        run_parity(
            |s| unary_op(s, OpKind::RmsNorm { eps: 1e-5 }, &a, m.clone()),
            1e-3,
            1e-3,
        );
        eprintln!("  PASS RmsNorm");
    }

    // ── RoPE ────────────────────────────────────────────────────────────
    {
        let a = gen_data(4 * 16, 40);
        let m = meta(&[4, 16]);
        run_parity(
            |s| {
                unary_op(
                    s,
                    OpKind::Rope {
                        rotary_dim: 16,
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
        eprintln!("  PASS Rope");
    }

    // ── Attention ───────────────────────────────────────────────────────
    {
        let tq = 4usize;
        let tk = 8usize;
        let a = gen_data(tq * tk, 50);
        let m = meta(&[tq as i64, tk as i64]);
        let scale = 1.0 / (tk as f32).sqrt();
        run_parity(
            |s| unary_op(s, OpKind::ScaledMaskedSoftmax { scale, causal: true }, &a, m.clone()),
            1e-3,
            1e-3,
        );
        eprintln!("  PASS ScaledMaskedSoftmax");
    }
    {
        let tq = 4usize;
        let dh = 16usize;
        let tk = tq;
        let q = gen_data(tq * dh, 51);
        let k = gen_data(tk * dh, 52);
        let v = gen_data(tk * dh, 53);
        let qm = meta(&[tq as i64, dh as i64]);
        let km = meta(&[tk as i64, dh as i64]);
        let vm = meta(&[tk as i64, dh as i64]);
        let om = meta(&[tq as i64, dh as i64]);
        let scale = 1.0 / (dh as f32).sqrt();
        run_parity(
            |s| {
                let nq = s.add_constant(q.clone(), qm.clone());
                let nk = s.add_constant(k.clone(), km.clone());
                let nv = s.add_constant(v.clone(), vm.clone());
                s.add_op(
                    OpKind::Attention {
                        scale,
                        causal: true,
                    },
                    SmallVec::from_slice(&[nq, nk, nv]),
                    om.clone(),
                )
            },
            1e-3,
            1e-3,
        );
        eprintln!("  PASS Attention");
    }

    // ── Transpose ───────────────────────────────────────────────────────
    {
        let a = gen_data(64 * 128, 60);
        let im = meta(&[64, 128]);
        let om = meta(&[128, 64]);
        run_parity(
            |s| {
                let na = s.add_constant(a.clone(), im.clone());
                s.add_op(
                    OpKind::Transpose { axes: None },
                    SmallVec::from_slice(&[na]),
                    om.clone(),
                )
            },
            0.0,
            0.0,
        );
        eprintln!("  PASS Transpose");
    }

    eprintln!("\n  Backend Parity Gate: ALL PASSED");
}

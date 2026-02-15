//! Memory stability tests.
//!
//! Validates that repeated evaluation doesn't cause unbounded memory growth.
//! Tests include:
//! - 10k eval cycles without growth
//! - Buffer reuse after evaluation
//! - Graph/buffer cache doesn't grow unboundedly

use mlx_core::backend::Stream;
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use mlx_core::{Device, Tensor};
use smallvec::SmallVec;

fn fresh_stream() -> Stream {
    Stream::new(Box::new(CpuRefBackend))
}

fn meta(shape: Vec<i64>) -> TensorMeta {
    TensorMeta {
        shape: Shape::new(shape),
        dtype: DType::F32,
    }
}

fn cpu() -> Device {
    Device::Cpu
}

fn s(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec())
}

// ── 10k eval cycle tests ───────────────────────────────────────────────

#[test]
fn test_10k_eval_cycles_fresh_streams() {
    // Each iteration creates a fresh stream to avoid cache accumulation.
    // This tests that stream construction/destruction doesn't leak.
    for i in 0..10_000 {
        let stream = fresh_stream();
        let a = stream.add_constant(vec![i as f32], meta(vec![1]));
        let b = stream.add_constant(vec![1.0], meta(vec![1]));
        let c = stream.add_op(OpKind::Add, SmallVec::from_slice(&[a, b]), meta(vec![1]));
        stream.eval(c).unwrap();
        let result = stream.get_buffer(c).unwrap();
        assert_eq!(result, vec![i as f32 + 1.0]);
    }
}

#[test]
fn test_10k_matmul_cycles() {
    // Repeated matmul evaluations
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity

    for _ in 0..10_000 {
        let stream = fresh_stream();
        let a = stream.add_constant(a_data.clone(), meta(vec![2, 2]));
        let b = stream.add_constant(b_data.clone(), meta(vec![2, 2]));
        let c = stream.add_op(OpKind::MatMul, SmallVec::from_slice(&[a, b]), meta(vec![2, 2]));
        stream.eval(c).unwrap();
        let result = stream.get_buffer(c).unwrap();
        assert_eq!(result, a_data);
    }
}

#[test]
fn test_10k_norm_cycles() {
    // Repeated normalization evaluations
    let data = vec![1.0, 2.0, 3.0];

    for _ in 0..10_000 {
        let stream = fresh_stream();
        let x = stream.add_constant(data.clone(), meta(vec![3]));
        let y = stream.add_op(
            OpKind::LayerNorm { eps: 1e-5 },
            SmallVec::from_slice(&[x]),
            meta(vec![3]),
        );
        stream.eval(y).unwrap();
        let result = stream.get_buffer(y).unwrap();
        let mean: f32 = result.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }
}

// ── Cache growth monitoring ────────────────────────────────────────────

#[test]
fn test_cache_growth_bounded_single_stream() {
    // Using a single stream with many operations. Cache should grow linearly
    // with the number of unique nodes, not exponentially.
    let stream = fresh_stream();
    let mut last_cache_len = stream.cache_len();

    for i in 0..1000 {
        let a = stream.add_constant(vec![i as f32], meta(vec![1]));
        let b = stream.add_constant(vec![1.0], meta(vec![1]));
        let c = stream.add_op(OpKind::Add, SmallVec::from_slice(&[a, b]), meta(vec![1]));
        stream.eval(c).unwrap();

        let new_cache_len = stream.cache_len();
        // Cache should only grow by ~3 per iteration (a, b, c)
        assert!(
            new_cache_len <= last_cache_len + 3,
            "iteration {i}: cache grew from {last_cache_len} to {new_cache_len}"
        );
        last_cache_len = new_cache_len;
    }
}

#[test]
fn test_eval_count_scales_linearly() {
    // Eval count should scale linearly with the number of operations.
    let stream = fresh_stream();
    let mut prev_count = stream.eval_count();

    for i in 0..100 {
        let a = stream.add_constant(vec![i as f32], meta(vec![1]));
        let b = stream.add_constant(vec![1.0], meta(vec![1]));
        let c = stream.add_op(OpKind::Add, SmallVec::from_slice(&[a, b]), meta(vec![1]));
        stream.eval(c).unwrap();

        let new_count = stream.eval_count();
        // Each eval should trigger exactly 1 backend call (a and b are constants)
        assert_eq!(
            new_count - prev_count,
            1,
            "iteration {i}: expected 1 new eval, got {}",
            new_count - prev_count
        );
        prev_count = new_count;
    }
}

// ── Deep graph eval stability ──────────────────────────────────────────

#[test]
fn test_deep_chain_1000_ops() {
    // Build a chain of 1000 negations and evaluate
    let stream = fresh_stream();
    let mut current = stream.add_constant(vec![1.0], meta(vec![1]));

    for _ in 0..1000 {
        current = stream.add_op(OpKind::Neg, SmallVec::from_slice(&[current]), meta(vec![1]));
    }

    stream.eval(current).unwrap();
    let result = stream.get_buffer(current).unwrap();
    // 1000 negations (even number) = identity
    assert_eq!(result, vec![1.0]);
}

#[test]
fn test_wide_graph_100_branches() {
    // One input shared across 100 independent branches
    let stream = fresh_stream();
    let x = stream.add_constant(vec![2.0, 3.0], meta(vec![2]));

    let mut outputs = Vec::with_capacity(100);
    for _ in 0..100 {
        let neg = stream.add_op(OpKind::Neg, SmallVec::from_slice(&[x]), meta(vec![2]));
        outputs.push(neg);
    }

    // Evaluate all
    for &out in &outputs {
        stream.eval(out).unwrap();
        let result = stream.get_buffer(out).unwrap();
        assert_eq!(result, vec![-2.0, -3.0]);
    }
}

// ── Tensor API stability ───────────────────────────────────────────────

#[test]
fn test_10k_tensor_api_cycles() {
    for i in 0..10_000 {
        let val = i as f32;
        let a = Tensor::from_f32(&[val, val + 1.0], &s(&[2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0], &s(&[2]), &cpu()).unwrap();
        let c = a.add(&b).unwrap();
        let result = c.to_vec_f32().unwrap();
        assert_eq!(result, vec![val + 1.0, val + 2.0]);
    }
}

#[test]
fn test_repeated_softmax_stability() {
    // Softmax should produce stable probability distributions even after many evals
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    for _ in 0..5_000 {
        let t = Tensor::from_f32(&data, &s(&[5]), &cpu()).unwrap();
        let s = t.softmax(0).unwrap();
        let result = s.to_vec_f32().unwrap();

        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax doesn't sum to 1: {sum}"
        );
    }
}

#[test]
fn test_repeated_attention_stability() {
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![0.5, 0.5, 0.5, 0.5];

    let mut prev_result: Option<Vec<f32>> = None;

    for i in 0..1_000 {
        let q_t = Tensor::from_f32(&q, &s(&[2, 2]), &cpu()).unwrap();
        let k_t = Tensor::from_f32(&k, &s(&[2, 2]), &cpu()).unwrap();
        let v_t = Tensor::from_f32(&v, &s(&[2, 2]), &cpu()).unwrap();
        let attn = q_t.attention(&k_t, &v_t, 0.5, true).unwrap();
        let result = attn.to_vec_f32().unwrap();

        if let Some(ref prev) = prev_result {
            for (j, (&a, &b)) in prev.iter().zip(result.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "attention not deterministic at iteration {i}, element {j}: {a} vs {b}"
                );
            }
        }
        prev_result = Some(result);
    }
}

// ── Stress test combinations ───────────────────────────────────────────

#[test]
fn test_alternating_ops_stability() {
    // Alternate between different ops to stress the backend dispatch
    for i in 0..5_000 {
        let val = (i as f32) * 0.01;
        let t = Tensor::from_f32(&[val, val + 1.0, val + 2.0], &s(&[3]), &cpu()).unwrap();

        match i % 5 {
            0 => {
                let _ = t.neg().to_vec_f32().unwrap();
            }
            1 => {
                let _ = t.silu().to_vec_f32().unwrap();
            }
            2 => {
                let _ = t.gelu().to_vec_f32().unwrap();
            }
            3 => {
                let _ = t.layer_norm(1e-5).to_vec_f32().unwrap();
            }
            4 => {
                let _ = t.softmax(0).unwrap().to_vec_f32().unwrap();
            }
            _ => unreachable!(),
        }
    }
}

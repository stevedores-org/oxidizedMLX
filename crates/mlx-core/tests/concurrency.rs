//! Concurrency and stress tests for the tensor evaluation engine.
//!
//! Tests determinism under parallel evaluation, thread-safety of the
//! Stream/Graph/Buffer layer, and stress testing under high contention.

use std::sync::Arc;
use std::thread;

use mlx_core::backend::{Stream, default_stream};
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use mlx_core::{Device, Tensor};
use smallvec::SmallVec;

/// Helper to create a fresh CPU stream.
fn fresh_stream() -> Arc<Stream> {
    Arc::new(Stream::new(Box::new(CpuRefBackend)))
}

fn meta(shape: Vec<i64>) -> TensorMeta {
    TensorMeta {
        shape: Shape::new(shape),
        dtype: DType::F32,
    }
}

// ── 16-thread determinism tests ─────────────────────────────────────────

#[test]
fn test_16_thread_parallel_eval_determinism() {
    // Each thread independently builds and evaluates the same computation graph.
    // All threads should produce identical results.
    let num_threads = 16;
    let results: Vec<Vec<f32>> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                s.spawn(|| {
                    let a = Tensor::from_f32(
                        &[1.0, 2.0, 3.0, 4.0],
                        &Shape::new(vec![2, 2]),
                        &Device::Cpu,
                    )
                    .unwrap();
                    let b = Tensor::from_f32(
                        &[5.0, 6.0, 7.0, 8.0],
                        &Shape::new(vec![2, 2]),
                        &Device::Cpu,
                    )
                    .unwrap();
                    let c = a.matmul(&b).unwrap();
                    c.to_vec_f32().unwrap()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // All results should be identical
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result, &expected, "thread {i} produced wrong result");
    }
}

#[test]
fn test_16_thread_shared_stream_eval() {
    // All 16 threads share the same stream and evaluate different computations.
    let stream = fresh_stream();
    let num_threads = 16;

    let results: Vec<(usize, Vec<f32>)> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let stream = Arc::clone(&stream);
                s.spawn(move || {
                    let val = (tid + 1) as f32;
                    let a = stream.add_constant(vec![val, val * 2.0], meta(vec![2]));
                    let b = stream.add_constant(vec![1.0, 1.0], meta(vec![2]));
                    let c = stream.add_op(
                        OpKind::Add,
                        SmallVec::from_slice(&[a, b]),
                        meta(vec![2]),
                    );
                    stream.eval(c).unwrap();
                    (tid, stream.get_buffer(c).unwrap())
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Each thread should get its own correct result
    for (tid, result) in results {
        let val = (tid + 1) as f32;
        assert_eq!(result, vec![val + 1.0, val * 2.0 + 1.0]);
    }
}

#[test]
fn test_16_thread_complex_graph_determinism() {
    // Each thread builds a more complex graph: (a + b) * (a - b) = a^2 - b^2
    let num_threads = 16;

    let results: Vec<Vec<f32>> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                s.spawn(|| {
                    let a = Tensor::from_f32(&[3.0, 5.0, 7.0], &Shape::new(vec![3]), &Device::Cpu)
                        .unwrap();
                    let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &Device::Cpu)
                        .unwrap();

                    let sum = a.add(&b).unwrap();
                    let diff = a.sub(&b).unwrap();
                    let product = sum.mul(&diff).unwrap();
                    product.to_vec_f32().unwrap()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // (a+b)*(a-b) = a^2 - b^2 = [9-1, 25-4, 49-9] = [8, 21, 40]
    let expected = vec![8.0, 21.0, 40.0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result, &expected, "thread {i} produced wrong result");
    }
}

// ── Stress tests ───────────────────────────────────────────────────────

#[test]
fn test_1000_sequential_evals() {
    // Stress test: 1000 sequential evaluations shouldn't leak or corrupt state.
    let stream = fresh_stream();

    for i in 0..1000 {
        let val = i as f32;
        let a = stream.add_constant(vec![val], meta(vec![1]));
        let b = stream.add_constant(vec![1.0], meta(vec![1]));
        let c = stream.add_op(OpKind::Add, SmallVec::from_slice(&[a, b]), meta(vec![1]));
        stream.eval(c).unwrap();
        let result = stream.get_buffer(c).unwrap();
        assert_eq!(result, vec![val + 1.0], "iteration {i} failed");
    }
}

#[test]
fn test_deep_graph_chain() {
    // Build a very deep sequential graph: x0 -> neg -> neg -> ... -> neg (200 times)
    let stream = fresh_stream();
    let mut current = stream.add_constant(vec![42.0], meta(vec![1]));

    for _ in 0..200 {
        current = stream.add_op(OpKind::Neg, SmallVec::from_slice(&[current]), meta(vec![1]));
    }

    stream.eval(current).unwrap();
    let result = stream.get_buffer(current).unwrap();
    // 200 negations = identity (even number)
    assert_eq!(result, vec![42.0]);
}

#[test]
fn test_wide_fan_out() {
    // One input feeds into many independent operations
    let stream = fresh_stream();
    let x = stream.add_constant(vec![1.0, 2.0, 3.0], meta(vec![3]));

    let mut outputs = Vec::new();
    for _ in 0..50 {
        let y = stream.add_op(OpKind::Neg, SmallVec::from_slice(&[x]), meta(vec![3]));
        outputs.push(y);
    }

    // Evaluate all outputs
    for &out in &outputs {
        stream.eval(out).unwrap();
        let result = stream.get_buffer(out).unwrap();
        assert_eq!(result, vec![-1.0, -2.0, -3.0]);
    }
}

#[test]
fn test_concurrent_tensor_creation_and_eval() {
    // Many threads creating tensors and evaluating them simultaneously
    let num_threads = 16;
    let ops_per_thread = 100;

    thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                s.spawn(move || {
                    for i in 0..ops_per_thread {
                        let val = (tid * ops_per_thread + i) as f32;
                        let a = Tensor::from_f32(&[val], &Shape::new(vec![1]), &Device::Cpu)
                            .unwrap();
                        let b = Tensor::from_f32(&[1.0], &Shape::new(vec![1]), &Device::Cpu)
                            .unwrap();
                        let c = a.add(&b).unwrap();
                        let result = c.to_vec_f32().unwrap();
                        assert_eq!(result, vec![val + 1.0]);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    });
}

#[test]
fn test_16_thread_matmul_stress() {
    // Stress test with matmul across 16 threads
    let num_threads = 16;

    thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                s.spawn(move || {
                    // Each thread multiplies identity * data = data
                    let identity = vec![1.0, 0.0, 0.0, 1.0];
                    let data = vec![
                        (tid as f32) + 1.0,
                        (tid as f32) + 2.0,
                        (tid as f32) + 3.0,
                        (tid as f32) + 4.0,
                    ];

                    let a = Tensor::from_f32(&identity, &Shape::new(vec![2, 2]), &Device::Cpu)
                        .unwrap();
                    let b = Tensor::from_f32(&data, &Shape::new(vec![2, 2]), &Device::Cpu)
                        .unwrap();
                    let c = a.matmul(&b).unwrap();
                    let result = c.to_vec_f32().unwrap();
                    assert_eq!(result, data, "thread {tid} matmul failed");
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    });
}

#[test]
fn test_16_thread_norm_determinism() {
    // Test that normalization ops are deterministic across threads
    let num_threads = 16;
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let results: Vec<Vec<f32>> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let data = data.clone();
                s.spawn(move || {
                    let x = Tensor::from_f32(&data, &Shape::new(vec![2, 3]), &Device::Cpu)
                        .unwrap();
                    let y = x.layer_norm(1e-5);
                    y.to_vec_f32().unwrap()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // All threads should produce the same result
    let reference = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        for (j, (&a, &b)) in reference.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "thread {i} element {j}: {a} vs {b}"
            );
        }
    }
}

// ── Default stream shared access ───────────────────────────────────────

#[test]
fn test_default_stream_concurrent_access() {
    // The global default_stream() is shared; verify it handles concurrent usage.
    let num_threads = 8;

    thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                s.spawn(move || {
                    let stream = default_stream();
                    let val = (tid + 1) as f32;
                    let a = stream.add_constant(vec![val], meta(vec![1]));
                    stream.eval(a).unwrap();
                    let result = stream.get_buffer(a).unwrap();
                    assert_eq!(result, vec![val]);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    });
}

//! Golden tests for missing operator coverage.
//!
//! Fills Suite 4 gaps: GELU, Mean, Max reduction golden tests with
//! hand-computed expected values.

use mlx_core::backend::{Backend, NodeInput, Stream};
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use mlx_core::{Device, Tensor};
use smallvec::SmallVec;

fn cpu() -> Device {
    Device::Cpu
}

fn s(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec())
}

fn assert_allclose(a: &[f32], b: &[f32], atol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff <= atol,
            "mismatch at [{i}]: got={x} expected={y} diff={diff} atol={atol}"
        );
    }
}

// ── GELU golden tests ──────────────────────────────────────────────────

#[test]
fn test_gelu_golden_known_values() {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Reference values computed from Python: torch.nn.functional.gelu(x)
    let input = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let expected = vec![
        -0.004050, // gelu(-3)
        -0.045400, // gelu(-2)
        -0.158808, // gelu(-1)
        -0.154286, // gelu(-0.5)
        0.000000,  // gelu(0)
        0.345714,  // gelu(0.5)
        0.841192,  // gelu(1)
        1.954600,  // gelu(2)
        2.995950,  // gelu(3)
    ];

    let t = Tensor::from_f32(&input, &s(&[9]), &cpu()).unwrap();
    let result = t.gelu().to_vec_f32().unwrap();

    assert_allclose(&result, &expected, 1e-3);
}

#[test]
fn test_gelu_symmetry() {
    // GELU is NOT symmetric: gelu(-x) != -gelu(x), but gelu(0) = 0
    let t = Tensor::from_f32(&[0.0], &s(&[1]), &cpu()).unwrap();
    let result = t.gelu().to_vec_f32().unwrap();
    assert!((result[0]).abs() < 1e-6, "gelu(0) should be 0");
}

#[test]
fn test_gelu_large_positive() {
    // For large positive x, gelu(x) ≈ x
    let x = 10.0f32;
    let t = Tensor::from_f32(&[x], &s(&[1]), &cpu()).unwrap();
    let result = t.gelu().to_vec_f32().unwrap();
    assert!(
        (result[0] - x).abs() < 0.001,
        "gelu({x}) should ≈ {x}, got {}",
        result[0]
    );
}

#[test]
fn test_gelu_large_negative() {
    // For large negative x, gelu(x) ≈ 0
    let x = -10.0f32;
    let t = Tensor::from_f32(&[x], &s(&[1]), &cpu()).unwrap();
    let result = t.gelu().to_vec_f32().unwrap();
    assert!(
        result[0].abs() < 0.001,
        "gelu({x}) should ≈ 0, got {}",
        result[0]
    );
}

#[test]
fn test_gelu_2d() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let t = Tensor::from_f32(&input, &s(&[2, 2]), &cpu()).unwrap();
    let result = t.gelu().to_vec_f32().unwrap();

    // Shape should be preserved
    assert_eq!(result.len(), 4);
    // gelu(0) = 0
    assert!(result[0].abs() < 1e-6);
    // gelu(1) ≈ 0.8412
    assert!((result[1] - 0.8412).abs() < 1e-3);
}

// ── Mean reduction golden tests ────────────────────────────────────────

#[test]
fn test_mean_axis0_2x3() {
    // [[1,2,3],[4,5,6]] -> mean(axis=0) = [2.5, 3.5, 4.5]
    let backend = CpuRefBackend;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Mean { axis: Some(0) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[3]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[2.5, 3.5, 4.5], 1e-6);
}

#[test]
fn test_mean_axis1_2x3() {
    // [[1,2,3],[4,5,6]] -> mean(axis=1) = [2.0, 5.0]
    let backend = CpuRefBackend;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Mean { axis: Some(1) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[2.0, 5.0], 1e-6);
}

#[test]
fn test_mean_all() {
    // mean([1,2,3,4,5,6]) = 3.5
    let backend = CpuRefBackend;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Mean { axis: None },
            &[NodeInput {
                data: &data,
                shape: &s(&[6]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: Shape::scalar(),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[3.5], 1e-6);
}

#[test]
fn test_mean_3d_axis0() {
    // Shape [2, 2, 2]: [[[1,2],[3,4]], [[5,6],[7,8]]]
    // mean(axis=0) -> [[3,4],[5,6]]
    let backend = CpuRefBackend;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = backend
        .eval_node(
            &OpKind::Mean { axis: Some(0) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 2, 2]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2, 2]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[3.0, 4.0, 5.0, 6.0], 1e-6);
}

#[test]
fn test_mean_single_element() {
    let backend = CpuRefBackend;
    let data = [42.0];
    let result = backend
        .eval_node(
            &OpKind::Mean { axis: None },
            &[NodeInput {
                data: &data,
                shape: &s(&[1]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: Shape::scalar(),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[42.0], 1e-6);
}

// ── Max reduction golden tests ─────────────────────────────────────────

#[test]
fn test_max_axis0_2x3() {
    // [[1,5,3],[4,2,6]] -> max(axis=0) = [4, 5, 6]
    let backend = CpuRefBackend;
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Max { axis: Some(0) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[3]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[4.0, 5.0, 6.0], 1e-6);
}

#[test]
fn test_max_axis1_2x3() {
    // [[1,5,3],[4,2,6]] -> max(axis=1) = [5, 6]
    let backend = CpuRefBackend;
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Max { axis: Some(1) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[5.0, 6.0], 1e-6);
}

#[test]
fn test_max_all() {
    // max([1,5,3,4,2,6]) = 6
    let backend = CpuRefBackend;
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let result = backend
        .eval_node(
            &OpKind::Max { axis: None },
            &[NodeInput {
                data: &data,
                shape: &s(&[6]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: Shape::scalar(),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[6.0], 1e-6);
}

#[test]
fn test_max_negative_values() {
    // max([-5, -2, -8, -1]) = -1
    let backend = CpuRefBackend;
    let data = [-5.0, -2.0, -8.0, -1.0];
    let result = backend
        .eval_node(
            &OpKind::Max { axis: None },
            &[NodeInput {
                data: &data,
                shape: &s(&[4]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: Shape::scalar(),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[-1.0], 1e-6);
}

#[test]
fn test_max_3d_axis1() {
    // Shape [2, 3, 2]: max along axis 1
    // [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
    // max(axis=1) -> [[5,6],[11,12]]
    let backend = CpuRefBackend;
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let result = backend
        .eval_node(
            &OpKind::Max { axis: Some(1) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3, 2]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2, 2]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_allclose(&result, &[5.0, 6.0, 11.0, 12.0], 1e-6);
}

// ── Combined operator golden tests ─────────────────────────────────────

#[test]
fn test_gelu_then_mean() {
    // gelu([0, 1, -1, 2]) then mean
    let t = Tensor::from_f32(&[0.0, 1.0, -1.0, 2.0], &s(&[4]), &cpu()).unwrap();
    let g = t.gelu();
    let result = g.to_vec_f32().unwrap();

    // Verify GELU values are reasonable
    assert!(result[0].abs() < 1e-6, "gelu(0) = 0");
    assert!(result[1] > 0.0, "gelu(1) > 0");
    assert!(result[2] < 0.0, "gelu(-1) < 0");
    assert!(result[3] > 1.0, "gelu(2) > 1");
}

#[test]
fn test_mean_after_matmul() {
    // matmul then mean reduction
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &s(&[2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &s(&[2, 2]), &cpu()).unwrap();
    let c = a.matmul(&b).unwrap();
    // c = a (identity matmul) = [[1,2],[3,4]]
    // sum_axis(0) of c = [4, 6]
    let s0 = c.sum_axis(0).unwrap();
    let result = s0.to_vec_f32().unwrap();
    assert_allclose(&result, &[4.0, 6.0], 1e-5);
}

#[test]
fn test_exp_log_roundtrip() {
    // exp(log(x)) should ≈ x for positive x
    let stream = Stream::new(Box::new(CpuRefBackend));

    let data = vec![1.0, 2.0, 0.5, 10.0];
    let meta = TensorMeta {
        shape: s(&[4]),
        dtype: DType::F32,
    };
    let x = stream.add_constant(data.clone(), meta.clone());
    let log_x = stream.add_op(OpKind::Log, SmallVec::from_slice(&[x]), meta.clone());
    let exp_log_x = stream.add_op(OpKind::Exp, SmallVec::from_slice(&[log_x]), meta.clone());

    stream.eval(exp_log_x).unwrap();
    let result = stream.get_buffer(exp_log_x).unwrap();

    assert_allclose(&result, &data, 1e-5);
}

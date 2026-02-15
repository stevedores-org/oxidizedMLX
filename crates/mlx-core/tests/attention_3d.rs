//! 3D+ tensor dimension tests for attention and related operations.
//!
//! Fills Suite 5 gap: validates that attention, transpose, reshape, and
//! normalization work correctly with 3D and higher-dimensional tensors.

use mlx_core::backend::{Backend, NodeInput};
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use mlx_core::{Device, Tensor};

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

// ── 3D Transpose tests ─────────────────────────────────────────────────

#[test]
fn test_transpose_3d_swap_01() {
    // Shape [2, 3, 2] with perm [1, 0, 2] -> [3, 2, 2]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 2]), &cpu()).unwrap();
    let result = t.transpose(Some(&[1, 0, 2])).unwrap();
    assert_eq!(result.shape(), &s(&[3, 2, 2]));

    let vals = result.to_vec_f32().unwrap();
    // Original: [[[0,1],[2,3],[4,5]], [[6,7],[8,9],[10,11]]]
    // After [1,0,2]: [[[0,1],[6,7]], [[2,3],[8,9]], [[4,5],[10,11]]]
    assert_allclose(&vals, &[0.0, 1.0, 6.0, 7.0, 2.0, 3.0, 8.0, 9.0, 4.0, 5.0, 10.0, 11.0], 1e-6);
}

#[test]
fn test_transpose_3d_swap_12() {
    // Shape [2, 3, 4] with perm [0, 2, 1] -> [2, 4, 3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.transpose(Some(&[0, 2, 1])).unwrap();
    assert_eq!(result.shape(), &s(&[2, 4, 3]));
    let vals = result.to_vec_f32().unwrap();
    assert_eq!(vals.len(), 24);
}

#[test]
fn test_transpose_3d_reverse() {
    // Shape [2, 3, 4] with perm [2, 1, 0] -> [4, 3, 2]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.transpose(Some(&[2, 1, 0])).unwrap();
    assert_eq!(result.shape(), &s(&[4, 3, 2]));

    // Verify specific elements
    let vals = result.to_vec_f32().unwrap();
    // Element [0,0,0] in original -> should be at [0,0,0] in transposed = 0.0
    assert!((vals[0] - 0.0).abs() < 1e-6);
}

// ── 3D Reshape tests ───────────────────────────────────────────────────

#[test]
fn test_reshape_3d_to_2d() {
    // [2, 3, 4] -> [6, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.reshape(&s(&[6, 4])).unwrap();
    assert_eq!(result.shape(), &s(&[6, 4]));
    let vals = result.to_vec_f32().unwrap();
    assert_eq!(vals, data);
}

#[test]
fn test_reshape_2d_to_3d() {
    // [6, 4] -> [2, 3, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[6, 4]), &cpu()).unwrap();
    let result = t.reshape(&s(&[2, 3, 4])).unwrap();
    assert_eq!(result.shape(), &s(&[2, 3, 4]));
    let vals = result.to_vec_f32().unwrap();
    assert_eq!(vals, data);
}

#[test]
fn test_reshape_3d_to_3d() {
    // [2, 3, 4] -> [4, 3, 2]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.reshape(&s(&[4, 3, 2])).unwrap();
    assert_eq!(result.shape(), &s(&[4, 3, 2]));
}

// ── 3D Normalization tests ─────────────────────────────────────────────

#[test]
fn test_layer_norm_3d() {
    // LayerNorm on a 3D tensor [2, 3, 4] normalizes over last dim (4)
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.layer_norm(1e-5);
    let vals = result.to_vec_f32().unwrap();

    assert_eq!(vals.len(), 24);

    // Each group of 4 elements should have mean ≈ 0
    for row in 0..6 {
        let start = row * 4;
        let mean: f32 = vals[start..start + 4].iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-4,
            "row {row} mean not zero: {mean}"
        );
    }
}

#[test]
fn test_rms_norm_3d() {
    // RMSNorm on a 3D tensor [2, 3, 4] normalizes over last dim (4)
    let data: Vec<f32> = (1..25).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.rms_norm(1e-5);
    let vals = result.to_vec_f32().unwrap();

    assert_eq!(vals.len(), 24);

    // Each group of 4 elements should have RMS ≈ 1
    for row in 0..6 {
        let start = row * 4;
        let rms: f32 = (vals[start..start + 4]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / 4.0)
            .sqrt();
        assert!(
            (rms - 1.0).abs() < 0.1,
            "row {row} rms not ~1: {rms}"
        );
    }
}

// ── 3D Softmax tests ───────────────────────────────────────────────────

#[test]
fn test_softmax_3d_last_axis() {
    // Softmax on [2, 3, 4] along axis=-1 (=2)
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4]), &cpu()).unwrap();
    let result = t.softmax(2).unwrap();
    let vals = result.to_vec_f32().unwrap();

    // Each group of 4 should sum to 1
    for row in 0..6 {
        let start = row * 4;
        let sum: f32 = vals[start..start + 4].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax row {row} doesn't sum to 1: {sum}"
        );
    }
}

#[test]
fn test_softmax_3d_middle_axis() {
    // Softmax on [2, 3, 4] along axis=1
    let backend = CpuRefBackend;
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
    let result = backend
        .eval_node(
            &OpKind::Softmax { axis: 1 },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3, 4]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2, 3, 4]),
                dtype: DType::F32,
            },
        )
        .unwrap();

    assert_eq!(result.len(), 24);

    // For each (batch, inner) pair, sum over dim=1 should be 1
    for b in 0..2 {
        for i in 0..4 {
            let mut sum = 0.0f32;
            for d in 0..3 {
                sum += result[b * 12 + d * 4 + i];
            }
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax batch={b} inner={i} sum={sum}"
            );
        }
    }
}

// ── 3D Reduction tests ─────────────────────────────────────────────────

#[test]
fn test_sum_3d_all_axes() {
    // Sum [2, 3, 4] along each axis
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let backend = CpuRefBackend;

    // Axis 0: [2,3,4] -> [3,4]
    let result_0 = backend
        .eval_node(
            &OpKind::Sum { axis: Some(0) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3, 4]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[3, 4]),
                dtype: DType::F32,
            },
        )
        .unwrap();
    assert_eq!(result_0.len(), 12);
    // Element [0,0]: data[0] + data[12] = 0 + 12 = 12
    assert!((result_0[0] - 12.0).abs() < 1e-6);

    // Axis 1: [2,3,4] -> [2,4]
    let result_1 = backend
        .eval_node(
            &OpKind::Sum { axis: Some(1) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3, 4]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2, 4]),
                dtype: DType::F32,
            },
        )
        .unwrap();
    assert_eq!(result_1.len(), 8);
    // Element [0,0]: data[0] + data[4] + data[8] = 0 + 4 + 8 = 12
    assert!((result_1[0] - 12.0).abs() < 1e-6);

    // Axis 2: [2,3,4] -> [2,3]
    let result_2 = backend
        .eval_node(
            &OpKind::Sum { axis: Some(2) },
            &[NodeInput {
                data: &data,
                shape: &s(&[2, 3, 4]),
                dtype: DType::F32,
            }],
            &TensorMeta {
                shape: s(&[2, 3]),
                dtype: DType::F32,
            },
        )
        .unwrap();
    assert_eq!(result_2.len(), 6);
    // Element [0,0]: sum(0,1,2,3) = 6
    assert!((result_2[0] - 6.0).abs() < 1e-6);
}

// ── Multi-head attention simulation ────────────────────────────────────

#[test]
fn test_multi_head_attention_via_reshape() {
    // Simulate multi-head attention by:
    // 1. Create QKV: [seq_len, n_heads * head_dim]
    // 2. Reshape to [seq_len, n_heads, head_dim]
    // 3. Transpose to [n_heads, seq_len, head_dim]
    // 4. Extract individual heads via narrow
    // 5. Run attention per head
    let seq_len = 4;
    let n_heads = 2;
    let head_dim = 2;
    let model_dim = n_heads * head_dim;

    let q_data: Vec<f32> = (0..seq_len * model_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();

    let q = Tensor::from_f32(&q_data, &s(&[seq_len as i64, model_dim as i64]), &cpu()).unwrap();

    // Reshape: [4, 4] -> [4, 2, 2]
    let q_3d = q
        .reshape(&s(&[seq_len as i64, n_heads as i64, head_dim as i64]))
        .unwrap();
    assert_eq!(q_3d.shape(), &s(&[4, 2, 2]));

    // Transpose: [4, 2, 2] -> [2, 4, 2] (heads first)
    let q_heads = q_3d.transpose(Some(&[1, 0, 2])).unwrap();
    assert_eq!(q_heads.shape(), &s(&[2, 4, 2]));

    // Narrow to extract head 0: [1, 4, 2]
    let q_h0 = q_heads.narrow(0, 0, 1).unwrap();
    assert_eq!(q_h0.shape(), &s(&[1, 4, 2]));

    // Reshape to [4, 2] for attention
    let q_h0_2d = q_h0.reshape(&s(&[4, 2])).unwrap();
    assert_eq!(q_h0_2d.shape(), &s(&[4, 2]));

    // Verify data is accessible
    let vals = q_h0_2d.to_vec_f32().unwrap();
    assert_eq!(vals.len(), 8);
}

#[test]
fn test_attention_per_head_then_concat() {
    // Run attention for 2 heads, then concatenate results
    let seq_len = 3;
    let head_dim = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Head 0
    let q0 = Tensor::from_f32(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        &s(&[seq_len as i64, head_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let k0 = q0.clone();
    let v0 = q0.clone();
    let attn0 = q0.attention(&k0, &v0, scale, true).unwrap();
    assert_eq!(attn0.shape(), &s(&[3, 2]));

    // Head 1
    let q1 = Tensor::from_f32(
        &[0.5, 0.5, 1.0, 0.0, 0.0, 1.0],
        &s(&[seq_len as i64, head_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let k1 = q1.clone();
    let v1 = q1.clone();
    let attn1 = q1.attention(&k1, &v1, scale, true).unwrap();
    assert_eq!(attn1.shape(), &s(&[3, 2]));

    // Concatenate along dim 1: [3, 2] + [3, 2] -> [3, 4]
    let concat = Tensor::cat(&[&attn0, &attn1], 1).unwrap();
    assert_eq!(concat.shape(), &s(&[3, 4]));

    let vals = concat.to_vec_f32().unwrap();
    assert_eq!(vals.len(), 12);
}

// ── 3D Narrow tests ───────────────────────────────────────────────────

#[test]
fn test_narrow_3d_axis0() {
    // [3, 2, 2] -> narrow(axis=0, start=1, len=2) -> [2, 2, 2]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[3, 2, 2]), &cpu()).unwrap();
    let result = t.narrow(0, 1, 2).unwrap();
    assert_eq!(result.shape(), &s(&[2, 2, 2]));
    let vals = result.to_vec_f32().unwrap();
    assert_allclose(&vals, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], 1e-6);
}

#[test]
fn test_narrow_3d_axis1() {
    // [2, 3, 2] -> narrow(axis=1, start=0, len=2) -> [2, 2, 2]
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 2]), &cpu()).unwrap();
    let result = t.narrow(1, 0, 2).unwrap();
    assert_eq!(result.shape(), &s(&[2, 2, 2]));
    let vals = result.to_vec_f32().unwrap();
    // First batch: rows 0,1 = [0,1,2,3]; Second batch: rows 0,1 = [6,7,8,9]
    assert_allclose(&vals, &[0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
}

// ── 3D Broadcast tests ─────────────────────────────────────────────────

#[test]
fn test_broadcast_2d_to_3d() {
    // [1, 3] broadcast to [2, 1, 3] -> [2, 1, 3]
    let data = vec![1.0, 2.0, 3.0];
    let t = Tensor::from_f32(&data, &s(&[1, 3]), &cpu()).unwrap();
    let result = t.broadcast_to(&s(&[2, 1, 3])).unwrap();
    assert_eq!(result.shape(), &s(&[2, 1, 3]));
    let vals = result.to_vec_f32().unwrap();
    assert_allclose(&vals, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-6);
}

// ── 4D tensor tests ────────────────────────────────────────────────────

#[test]
fn test_reshape_4d() {
    // [2, 3, 4, 5] -> [6, 20]
    let n = 2 * 3 * 4 * 5;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 3, 4, 5]), &cpu()).unwrap();
    let result = t.reshape(&s(&[6, 20])).unwrap();
    assert_eq!(result.shape(), &s(&[6, 20]));
    let vals = result.to_vec_f32().unwrap();
    assert_eq!(vals, data);
}

#[test]
fn test_layer_norm_4d() {
    // LayerNorm on [2, 2, 2, 3] normalizes over last dim (3)
    let n = 2 * 2 * 2 * 3;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &s(&[2, 2, 2, 3]), &cpu()).unwrap();
    let result = t.layer_norm(1e-5);
    let vals = result.to_vec_f32().unwrap();
    assert_eq!(vals.len(), n as usize);

    // Each group of 3 should have mean ≈ 0
    for row in 0..8 {
        let start = row * 3;
        let mean: f32 = vals[start..start + 3].iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-4, "4D layer_norm row {row} mean: {mean}");
    }
}

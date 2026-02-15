//! Built-in CPU reference backend — correctness oracle.
//!
//! This is an intentionally simple, safe Rust implementation of every op.
//! It prioritizes correctness and readability over performance.

use crate::backend::{Backend, NodeInput};
use crate::graph::{OpKind, TensorMeta};
use crate::{MlxError, Result};

/// Reference CPU backend.
pub struct CpuRefBackend;

impl Backend for CpuRefBackend {
    fn eval_node(
        &self,
        op: &OpKind,
        inputs: &[NodeInput<'_>],
        output_meta: &TensorMeta,
    ) -> Result<Vec<f32>> {
        match op {
            OpKind::Constant | OpKind::Parameter => Err(MlxError::InvalidArgument(
                "Constant/Parameter nodes should be pre-materialized".into(),
            )),
            OpKind::Add => binary_elementwise(inputs, |a, b| a + b),
            OpKind::Mul => binary_elementwise(inputs, |a, b| a * b),
            OpKind::Sub => binary_elementwise(inputs, |a, b| a - b),
            OpKind::Div => binary_elementwise(inputs, |a, b| a / b),
            OpKind::Neg => {
                let a = require_input(inputs, 0)?;
                Ok(a.data.iter().map(|x| -x).collect())
            }
            OpKind::Sum { axis } => reduce_sum(inputs, *axis),
            OpKind::Mean { axis } => reduce_mean(inputs, *axis),
            OpKind::Max { axis } => reduce_max(inputs, *axis),
            OpKind::MatMul => matmul(inputs),
            OpKind::Reshape { .. } => {
                let a = require_input(inputs, 0)?;
                Ok(a.data.to_vec())
            }
            OpKind::Transpose { axes } => transpose(inputs, axes.as_deref()),
            OpKind::Softmax { axis } => softmax(inputs, *axis),
            OpKind::Silu => {
                let a = require_input(inputs, 0)?;
                Ok(a.data.iter().map(|&x| x * sigmoid(x)).collect())
            }
            OpKind::Gelu => {
                let a = require_input(inputs, 0)?;
                Ok(a.data
                    .iter()
                    .map(|&x| {
                        0.5 * x
                            * (1.0
                                + ((2.0 / std::f32::consts::PI).sqrt()
                                    * (x + 0.044715 * x * x * x))
                                    .tanh())
                    })
                    .collect())
            }
            OpKind::LayerNorm { eps } => layer_norm(inputs, *eps, output_meta),
            OpKind::RmsNorm { eps } => rms_norm(inputs, *eps, output_meta),
            OpKind::Broadcast { target_shape } => broadcast(inputs, target_shape),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn require_input<'a>(inputs: &'a [NodeInput<'_>], idx: usize) -> Result<&'a NodeInput<'a>> {
    inputs
        .get(idx)
        .ok_or_else(|| MlxError::InvalidArgument(format!("expected input at index {idx}")))
}

fn binary_elementwise(inputs: &[NodeInput<'_>], f: fn(f32, f32) -> f32) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let b = require_input(inputs, 1)?;
    if a.data.len() != b.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: a.shape.0.clone(),
            got: b.shape.0.clone(),
        });
    }
    Ok(a.data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| f(x, y))
        .collect())
}

fn reduce_sum(inputs: &[NodeInput<'_>], axis: Option<i32>) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    match axis {
        None => Ok(vec![a.data.iter().sum()]),
        Some(axis) => reduce_along_axis(a, axis, |slice| slice.iter().sum()),
    }
}

fn reduce_mean(inputs: &[NodeInput<'_>], axis: Option<i32>) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    match axis {
        None => {
            let n = a.data.len() as f32;
            Ok(vec![a.data.iter().sum::<f32>() / n])
        }
        Some(axis) => {
            let ndim = a.shape.ndim() as i32;
            let ax = if axis < 0 { ndim + axis } else { axis } as usize;
            let dim = a.shape.0[ax] as f32;
            reduce_along_axis(a, axis, |slice| slice.iter().sum::<f32>() / dim)
        }
    }
}

fn reduce_max(inputs: &[NodeInput<'_>], axis: Option<i32>) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    match axis {
        None => Ok(vec![
            a.data.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ]),
        Some(axis) => reduce_along_axis(a, axis, |slice| {
            slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }),
    }
}

fn reduce_along_axis(
    a: &NodeInput<'_>,
    axis: i32,
    reducer: impl Fn(&[f32]) -> f32,
) -> Result<Vec<f32>> {
    let ndim = a.shape.ndim() as i32;
    let ax = if axis < 0 { ndim + axis } else { axis };
    if ax < 0 || ax >= ndim {
        return Err(MlxError::InvalidArgument(format!(
            "axis {axis} out of range for ndim {ndim}"
        )));
    }
    let ax = ax as usize;

    let outer: usize = a.shape.0[..ax].iter().product::<i64>() as usize;
    let dim: usize = a.shape.0[ax] as usize;
    let inner: usize = a.shape.0[ax + 1..].iter().product::<i64>() as usize;

    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut slice = Vec::with_capacity(dim);
            for d in 0..dim {
                slice.push(a.data[o * dim * inner + d * inner + i]);
            }
            result.push(reducer(&slice));
        }
    }
    Ok(result)
}

fn matmul(inputs: &[NodeInput<'_>]) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let b = require_input(inputs, 1)?;

    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(MlxError::InvalidArgument(
            "matmul requires 2D tensors".into(),
        ));
    }

    let m = a.shape.0[0] as usize;
    let k = a.shape.0[1] as usize;
    let k2 = b.shape.0[0] as usize;
    let n = b.shape.0[1] as usize;

    if k != k2 {
        return Err(MlxError::ShapeMismatch {
            expected: vec![m as i64, k as i64],
            got: vec![k2 as i64, n as i64],
        });
    }

    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = sum;
        }
    }
    Ok(data)
}

fn transpose(inputs: &[NodeInput<'_>], axes: Option<&[usize]>) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let ndim = a.shape.ndim();

    let perm: Vec<usize> = match axes {
        Some(ax) => ax.to_vec(),
        None => (0..ndim).rev().collect(),
    };

    if perm.len() != ndim {
        return Err(MlxError::InvalidArgument(
            "transpose axes length must match ndim".into(),
        ));
    }

    let old_shape: Vec<usize> = a.shape.0.iter().map(|&d| d as usize).collect();
    let new_shape: Vec<usize> = perm.iter().map(|&ax| old_shape[ax]).collect();

    // Compute strides for the old shape.
    let mut old_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }

    let total = a.data.len();
    let mut result = vec![0.0f32; total];

    for (flat, out) in result.iter_mut().enumerate() {
        // Convert flat index → multi-index in NEW shape.
        let mut remaining = flat;
        let mut old_flat = 0;
        for dim_idx in 0..ndim {
            let new_dim_size: usize = new_shape[dim_idx + 1..].iter().product::<usize>().max(1);
            let coord = remaining / new_dim_size;
            remaining %= new_dim_size;
            // This coord in the new tensor corresponds to perm[dim_idx] axis in old tensor.
            old_flat += coord * old_strides[perm[dim_idx]];
        }
        *out = a.data[old_flat];
    }

    Ok(result)
}

fn softmax(inputs: &[NodeInput<'_>], axis: i32) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let ndim = a.shape.ndim() as i32;
    let ax = if axis < 0 { ndim + axis } else { axis };
    if ax < 0 || ax >= ndim {
        return Err(MlxError::InvalidArgument(format!(
            "axis {axis} out of range for ndim {ndim}"
        )));
    }
    let ax = ax as usize;

    let outer: usize = a.shape.0[..ax].iter().product::<i64>() as usize;
    let dim: usize = a.shape.0[ax] as usize;
    let inner: usize = a.shape.0[ax + 1..].iter().product::<i64>() as usize;

    let mut data = a.data.to_vec();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            let mut sum_exp = 0.0f32;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                data[idx] = (data[idx] - max_val).exp();
                sum_exp += data[idx];
            }
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                data[idx] /= sum_exp;
            }
        }
    }
    Ok(data)
}

fn layer_norm(inputs: &[NodeInput<'_>], eps: f32, _meta: &TensorMeta) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    // LayerNorm normalizes over the last dimension.
    let ndim = a.shape.ndim();
    if ndim == 0 {
        return Ok(a.data.to_vec());
    }
    let last_dim = a.shape.0[ndim - 1] as usize;
    let outer = a.data.len() / last_dim;

    let mut result = vec![0.0f32; a.data.len()];
    for o in 0..outer {
        let start = o * last_dim;
        let end = start + last_dim;
        let slice = &a.data[start..end];

        let mean = slice.iter().sum::<f32>() / last_dim as f32;
        let var = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;
        let std = (var + eps).sqrt();

        for (i, &x) in slice.iter().enumerate() {
            result[start + i] = (x - mean) / std;
        }
    }
    Ok(result)
}

fn broadcast(inputs: &[NodeInput<'_>], target_shape: &crate::Shape) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let in_shape = &a.shape.0;
    let out_shape = &target_shape.0;
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let pad = out_ndim - in_ndim;
    let total: usize = out_shape.iter().product::<i64>() as usize;

    let mut result = vec![0.0f32; total];
    for (out_flat, out) in result.iter_mut().enumerate() {
        let mut remaining = out_flat;
        let mut in_flat = 0usize;
        let mut in_stride = 1usize;

        for d in (0..out_ndim).rev() {
            let out_dim = out_shape[d] as usize;
            let coord = remaining % out_dim;
            remaining /= out_dim;

            if d >= pad {
                let in_d = d - pad;
                let in_dim = in_shape[in_d] as usize;
                let in_coord = if in_dim == 1 { 0 } else { coord };
                in_flat += in_coord * in_stride;
                in_stride *= in_dim;
            }
        }
        *out = a.data[in_flat];
    }
    Ok(result)
}

fn rms_norm(inputs: &[NodeInput<'_>], eps: f32, _meta: &TensorMeta) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let ndim = a.shape.ndim();
    if ndim == 0 {
        return Ok(a.data.to_vec());
    }
    let last_dim = a.shape.0[ndim - 1] as usize;
    let outer = a.data.len() / last_dim;

    let mut result = vec![0.0f32; a.data.len()];
    for o in 0..outer {
        let start = o * last_dim;
        let end = start + last_dim;
        let slice = &a.data[start..end];

        let rms = (slice.iter().map(|x| x * x).sum::<f32>() / last_dim as f32 + eps).sqrt();

        for (i, &x) in slice.iter().enumerate() {
            result[start + i] = x / rms;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorMeta;
    use crate::types::Shape;

    fn meta(shape: Vec<i64>) -> TensorMeta {
        TensorMeta {
            shape: Shape::new(shape),
            dtype: crate::DType::F32,
        }
    }

    fn input(data: &[f32], shape: Vec<i64>) -> NodeInput<'_> {
        // We need to leak the shape to get a reference. Use a workaround.
        NodeInput {
            data,
            shape: Box::leak(Box::new(Shape::new(shape))),
            dtype: crate::DType::F32,
        }
    }

    #[test]
    fn test_add() {
        let backend = CpuRefBackend;
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [4.0, 5.0, 6.0];
        let result = backend
            .eval_node(
                &OpKind::Add,
                &[input(&a_data, vec![3]), input(&b_data, vec![3])],
                &meta(vec![3]),
            )
            .unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let backend = CpuRefBackend;
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [5.0, 6.0, 7.0, 8.0];
        let result = backend
            .eval_node(
                &OpKind::MatMul,
                &[input(&a_data, vec![2, 2]), input(&b_data, vec![2, 2])],
                &meta(vec![2, 2]),
            )
            .unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_softmax() {
        let backend = CpuRefBackend;
        let data = [1.0, 2.0, 3.0];
        let result = backend
            .eval_node(
                &OpKind::Softmax { axis: 0 },
                &[input(&data, vec![3])],
                &meta(vec![3]),
            )
            .unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_neg() {
        let backend = CpuRefBackend;
        let data = [1.0, -2.0, 3.0];
        let result = backend
            .eval_node(&OpKind::Neg, &[input(&data, vec![3])], &meta(vec![3]))
            .unwrap();
        assert_eq!(result, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_layer_norm() {
        let backend = CpuRefBackend;
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = backend
            .eval_node(
                &OpKind::LayerNorm { eps: 1e-5 },
                &[input(&data, vec![2, 3])],
                &meta(vec![2, 3]),
            )
            .unwrap();
        // Each row should be normalized to mean≈0, std≈1
        let row1_mean: f32 = result[0..3].iter().sum::<f32>() / 3.0;
        assert!(row1_mean.abs() < 1e-5);
    }

    #[test]
    fn test_reduce_sum_axis() {
        let backend = CpuRefBackend;
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = backend
            .eval_node(
                &OpKind::Sum { axis: Some(0) },
                &[input(&data, vec![2, 3])],
                &meta(vec![3]),
            )
            .unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_all() {
        let backend = CpuRefBackend;
        let data = [1.0, 2.0, 3.0];
        let result = backend
            .eval_node(
                &OpKind::Sum { axis: None },
                &[input(&data, vec![3])],
                &meta(vec![]),
            )
            .unwrap();
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_silu() {
        let backend = CpuRefBackend;
        let data = [0.0, 1.0, -1.0];
        let result = backend
            .eval_node(&OpKind::Silu, &[input(&data, vec![3])], &meta(vec![3]))
            .unwrap();
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!((result[0]).abs() < 1e-6);
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }
}

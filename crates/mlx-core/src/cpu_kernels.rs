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
            OpKind::Rope {
                rotary_dim,
                pos_offset,
                theta,
            } => cpu_rope(inputs, output_meta, *rotary_dim, *pos_offset, *theta),
            OpKind::LayerNormVjp { eps } => layer_norm_vjp(inputs, *eps),
            OpKind::RmsNormVjp { eps } => rms_norm_vjp(inputs, *eps),
            OpKind::SoftmaxVjp { axis } => softmax_vjp(inputs, *axis),
            OpKind::SiluVjp => silu_vjp(inputs),
            OpKind::GeluVjp => gelu_vjp(inputs),
            OpKind::Sqrt => {
                let a = require_input(inputs, 0)?;
                Ok(a.data.iter().map(|&x| x.sqrt()).collect())
            }
            OpKind::RoPE {
                base,
                offset,
                traditional,
            } => rope(inputs, *base, *offset, *traditional),
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

/// LayerNorm backward: inputs = [grad_output, original_input].
///
/// For x of shape [..., D] (normalized over last dim, no affine params):
///   x_hat = (x - mean) / std
///   dx_i = (1/std) * (dy_i - mean(dy) - x_hat_i * mean(dy * x_hat))
fn layer_norm_vjp(inputs: &[NodeInput<'_>], eps: f32) -> Result<Vec<f32>> {
    let dy = require_input(inputs, 0)?;
    let x = require_input(inputs, 1)?;
    if dy.shape != x.shape || dy.data.len() != x.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: x.shape.0.clone(),
            got: dy.shape.0.clone(),
        });
    }
    let ndim = x.shape.ndim();
    if ndim == 0 {
        return Ok(dy.data.to_vec());
    }
    let d = x.shape.0[ndim - 1] as usize;
    if d == 0 || x.data.is_empty() {
        return Ok(vec![0.0f32; x.data.len()]);
    }
    let d_f = d as f32;
    let outer = x.data.len() / d;

    let mut result = vec![0.0f32; x.data.len()];
    for o in 0..outer {
        let start = o * d;
        let end = start + d;
        let x_slice = &x.data[start..end];
        let dy_slice = &dy.data[start..end];

        // Forward recomputation
        let mean = x_slice.iter().sum::<f32>() / d_f;
        let var = x_slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d_f;
        let std = (var + eps).sqrt();
        let inv_std = 1.0 / std;

        // x_hat = (x - mean) / std
        let x_hat: Vec<f32> = x_slice.iter().map(|v| (v - mean) * inv_std).collect();

        // mean(dy) and mean(dy * x_hat)
        let mean_dy = dy_slice.iter().sum::<f32>() / d_f;
        let mean_dy_xhat: f32 = dy_slice
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / d_f;

        // dx_i = inv_std * (dy_i - mean_dy - x_hat_i * mean_dy_xhat)
        for i in 0..d {
            result[start + i] = inv_std * (dy_slice[i] - mean_dy - x_hat[i] * mean_dy_xhat);
        }
    }
    Ok(result)
}

/// RmsNorm backward: inputs = [grad_output, original_input].
///
/// For x of shape [..., D] (normalized over last dim, no affine params):
///   rms = sqrt(mean(x^2) + eps)
///   y = x / rms
///   dx_i = (1/rms) * (dy_i - y_i * mean(dy * y))
fn rms_norm_vjp(inputs: &[NodeInput<'_>], eps: f32) -> Result<Vec<f32>> {
    let dy = require_input(inputs, 0)?;
    let x = require_input(inputs, 1)?;
    if dy.shape != x.shape || dy.data.len() != x.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: x.shape.0.clone(),
            got: dy.shape.0.clone(),
        });
    }
    let ndim = x.shape.ndim();
    if ndim == 0 {
        return Ok(dy.data.to_vec());
    }
    let d = x.shape.0[ndim - 1] as usize;
    if d == 0 || x.data.is_empty() {
        return Ok(vec![0.0f32; x.data.len()]);
    }
    let d_f = d as f32;
    let outer = x.data.len() / d;

    let mut result = vec![0.0f32; x.data.len()];
    for o in 0..outer {
        let start = o * d;
        let end = start + d;
        let x_slice = &x.data[start..end];
        let dy_slice = &dy.data[start..end];

        // Forward recomputation
        let rms = (x_slice.iter().map(|v| v * v).sum::<f32>() / d_f + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // y = x / rms
        let y: Vec<f32> = x_slice.iter().map(|v| v * inv_rms).collect();

        // mean(dy * y)
        let mean_dy_y: f32 = dy_slice
            .iter()
            .zip(y.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / d_f;

        // dx_i = inv_rms * (dy_i - y_i * mean_dy_y)
        for i in 0..d {
            result[start + i] = inv_rms * (dy_slice[i] - y[i] * mean_dy_y);
        }
    }
    Ok(result)
}

/// Softmax backward: inputs = [grad_output, softmax_output].
///
/// dx_i = s_i * (dy_i - sum(dy * s))
fn softmax_vjp(inputs: &[NodeInput<'_>], axis: i32) -> Result<Vec<f32>> {
    let dy = require_input(inputs, 0)?;
    let s = require_input(inputs, 1)?;
    if dy.data.len() != s.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: s.shape.0.clone(),
            got: dy.shape.0.clone(),
        });
    }
    let ndim = s.shape.ndim() as i32;
    let ax = if axis < 0 { ndim + axis } else { axis };
    if ax < 0 || ax >= ndim {
        return Err(MlxError::InvalidArgument(format!(
            "axis {axis} out of range for ndim {ndim}"
        )));
    }
    let ax = ax as usize;

    let outer: usize = s.shape.0[..ax].iter().product::<i64>() as usize;
    let dim: usize = s.shape.0[ax] as usize;
    let inner: usize = s.shape.0[ax + 1..].iter().product::<i64>() as usize;

    let mut result = vec![0.0f32; s.data.len()];
    for o in 0..outer {
        for i in 0..inner {
            // dot = sum(dy * s) along the axis
            let mut dot = 0.0f32;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                dot += dy.data[idx] * s.data[idx];
            }
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                result[idx] = s.data[idx] * (dy.data[idx] - dot);
            }
        }
    }
    Ok(result)
}

/// SiLU backward: inputs = [grad_output, original_input].
///
/// d_silu/dx = σ(x) * (1 + x * (1 - σ(x)))
fn silu_vjp(inputs: &[NodeInput<'_>]) -> Result<Vec<f32>> {
    let dy = require_input(inputs, 0)?;
    let x = require_input(inputs, 1)?;
    if dy.data.len() != x.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: x.shape.0.clone(),
            got: dy.shape.0.clone(),
        });
    }
    Ok(dy
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&dy_i, &x_i)| {
            let sig = sigmoid(x_i);
            dy_i * sig * (1.0 + x_i * (1.0 - sig))
        })
        .collect())
}

/// GELU backward (tanh approximation): inputs = [grad_output, original_input].
///
/// gelu(x) = 0.5x(1 + tanh(a(x + bx³)))
/// d_gelu/dx = 0.5(1 + tanh(...)) + 0.5x * sech²(...) * a(1 + 3bx²)
fn gelu_vjp(inputs: &[NodeInput<'_>]) -> Result<Vec<f32>> {
    let dy = require_input(inputs, 0)?;
    let x = require_input(inputs, 1)?;
    if dy.data.len() != x.data.len() {
        return Err(MlxError::ShapeMismatch {
            expected: x.shape.0.clone(),
            got: dy.shape.0.clone(),
        });
    }
    let a = (2.0f32 / std::f32::consts::PI).sqrt();
    let b = 0.044715f32;
    Ok(dy
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&dy_i, &x_i)| {
            let inner = a * (x_i + b * x_i * x_i * x_i);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let dgelu =
                0.5 * (1.0 + tanh_inner) + 0.5 * x_i * sech2 * a * (1.0 + 3.0 * b * x_i * x_i);
            dy_i * dgelu
        })
        .collect())
}

fn cpu_rope(
    inputs: &[NodeInput<'_>],
    meta: &TensorMeta,
    rotary_dim: usize,
    pos_offset: usize,
    theta: f32,
) -> Result<Vec<f32>> {
    let x = require_input(inputs, 0)?;
    if meta.shape.ndim() != 2 {
        return Err(MlxError::InvalidArgument(
            "Rope input must be 2-D [tokens, head_dim]".into(),
        ));
    }
    let tokens = meta.shape.0[0] as usize;
    let head_dim = meta.shape.0[1] as usize;
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(MlxError::InvalidArgument(
            "rotary_dim must be even and <= head_dim".into(),
        ));
    }

    let mut out = x.data.to_vec();
    for t in 0..tokens {
        for i in 0..rotary_dim / 2 {
            let inv_freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
            let angle = (pos_offset + t) as f32 * inv_freq;
            let (s, c) = angle.sin_cos();

            let base = t * head_dim + i * 2;
            let x0 = x.data[base];
            let x1 = x.data[base + 1];

            out[base] = x0 * c - x1 * s;
            out[base + 1] = x0 * s + x1 * c;
        }
    }
    Ok(out)
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

fn rope(inputs: &[NodeInput<'_>], base: f32, offset: usize, traditional: bool) -> Result<Vec<f32>> {
    let a = require_input(inputs, 0)?;
    let ndim = a.shape.ndim();
    if ndim < 1 {
        return Err(MlxError::InvalidArgument(
            "RoPE requires at least 1 dimension".into(),
        ));
    }

    let head_dim = a.shape.0[ndim - 1] as usize;
    if !head_dim.is_multiple_of(2) {
        return Err(MlxError::InvalidArgument(format!(
            "RoPE head_dim must be even, got {head_dim}"
        )));
    }
    let half_dim = head_dim / 2;

    let total = a.data.len();
    let num_heads_total = total / head_dim;

    let mut result = vec![0.0f32; total];

    for i in 0..num_heads_total {
        // Calculate position based on offset.
        // Assuming flattening over batch/seq for now.
        // More robust logic would use shape explicitly.
        // Here we simplify assuming linear indexing corresponds to position.
        // Wait, issue specified (tokens, head_dim) -> i corresponds to token index (pos).

        let pos = (offset + i) as f32;

        for d in 0..half_dim {
            let theta = pos * base.powf(-(2.0 * d as f32 / head_dim as f32));
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            if traditional {
                // Pairs are adjacent: (2d, 2d+1)
                let idx0 = i * head_dim + 2 * d;
                let idx1 = idx0 + 1;

                let x0 = a.data[idx0];
                let x1 = a.data[idx1];

                result[idx0] = x0 * cos_theta - x1 * sin_theta;
                result[idx1] = x0 * sin_theta + x1 * cos_theta;
            } else {
                // OpenAI style: pairs are (d, d + half_dim)
                let idx0 = i * head_dim + d;
                let idx1 = i * head_dim + d + half_dim;

                let x0 = a.data[idx0];
                let x1 = a.data[idx1];

                result[idx0] = x0 * cos_theta - x1 * sin_theta;
                result[idx1] = x0 * sin_theta + x1 * cos_theta;
            }
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
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_rope_offsets() {
        let backend = CpuRefBackend;
        let base = 10_000.0;
        let offset = 100usize;
        let traditional = false;
        // Shape: 1 seq, 4 head_dim. total = 4 floats.
        let data = [1.0, 0.0, 0.0, 1.0];
        let result = backend
            .eval_node(
                &OpKind::RoPE {
                    base,
                    offset,
                    traditional,
                },
                &[input(&data, vec![1, 4])],
                &meta(vec![1, 4]),
            )
            .unwrap();

        // Expected values (same logic as before)
        let cos100 = 100.0f32.cos();
        let sin100 = 100.0f32.sin();
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();

        assert!((result[0] - cos100).abs() < 1e-5);
        assert!((result[2] - sin100).abs() < 1e-5);
        assert!((result[1] - (-sin1)).abs() < 1e-5);
        assert!((result[3] - cos1).abs() < 1e-5);
    }

    #[test]
    fn test_rope_large() {
        let backend = CpuRefBackend;
        let shape = vec![128, 128];
        let numel = 128 * 128;
        let data = vec![1.0; numel];
        let result = backend.eval_node(
            &OpKind::RoPE {
                base: 10000.0,
                offset: 0,
                traditional: true,
            },
            &[input(&data, shape.clone())],
            &meta(shape.clone()),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), numel);
    }
}

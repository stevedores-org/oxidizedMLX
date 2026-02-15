//! Shape inference for graph ops.
//!
//! Given an `OpKind` and input shapes, computes the output shape. This is used
//! by the graph builder to set `TensorMeta` on newly created nodes.

use mlx_core::Shape;
use mlx_core::graph::OpKind;

/// Error returned when shapes are incompatible for an op.
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("shape mismatch: {0}")]
    Mismatch(String),

    #[error("invalid axis {axis} for ndim {ndim}")]
    InvalidAxis { axis: i32, ndim: usize },

    #[error("matmul inner dimensions mismatch: {k1} vs {k2}")]
    MatmulMismatch { k1: i64, k2: i64 },
}

/// Infer the output shape for a given op and input shapes.
pub fn infer_shape(op: &OpKind, inputs: &[&Shape]) -> Result<Shape, ShapeError> {
    match op {
        // Binary elementwise ops: shapes must match (or be broadcastable).
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input 0".into()))?;
            let b = inputs
                .get(1)
                .ok_or(ShapeError::Mismatch("missing input 1".into()))?;
            crate::broadcast_shapes(a, b)
                .ok_or_else(|| ShapeError::Mismatch(format!("cannot broadcast {a} with {b}")))
        }

        // Unary ops preserve shape.
        OpKind::Neg | OpKind::Silu | OpKind::Gelu | OpKind::Constant | OpKind::Parameter => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            Ok((*a).clone())
        }

        // Normalization preserves shape.
        OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            Ok((*a).clone())
        }

        // Softmax preserves shape.
        OpKind::Softmax { axis } => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            validate_axis(*axis, a.ndim())?;
            Ok((*a).clone())
        }

        // Reductions remove the specified axis.
        OpKind::Sum { axis } | OpKind::Mean { axis } | OpKind::Max { axis } => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            match axis {
                None => Ok(Shape::new(vec![1])),
                Some(ax) => {
                    let resolved = resolve_axis(*ax, a.ndim())?;
                    let mut dims = a.0.clone();
                    dims.remove(resolved);
                    if dims.is_empty() {
                        dims.push(1);
                    }
                    Ok(Shape::new(dims))
                }
            }
        }

        // MatMul: [M, K] @ [K, N] → [M, N]
        OpKind::MatMul => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input 0".into()))?;
            let b = inputs
                .get(1)
                .ok_or(ShapeError::Mismatch("missing input 1".into()))?;
            if a.ndim() != 2 || b.ndim() != 2 {
                return Err(ShapeError::Mismatch("matmul requires 2D tensors".into()));
            }
            let k1 = a.0[1];
            let k2 = b.0[0];
            if k1 != k2 {
                return Err(ShapeError::MatmulMismatch { k1, k2 });
            }
            Ok(Shape::new(vec![a.0[0], b.0[1]]))
        }

        // Reshape: output shape is specified in the op.
        OpKind::Reshape { new_shape } => Ok(new_shape.clone()),

        // Broadcast: output shape is the target shape.
        OpKind::Broadcast { target_shape } => Ok(target_shape.clone()),

        // Backward ops: output shape = input shape (same as grad_output shape).
        OpKind::LayerNormVjp { .. } | OpKind::RmsNormVjp { .. } => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            Ok((*a).clone())
        }

        // Transpose: permute dimensions.
        OpKind::Transpose { axes } => {
            let a = inputs
                .first()
                .ok_or(ShapeError::Mismatch("missing input".into()))?;
            let perm: Vec<usize> = match axes {
                Some(ax) => ax.clone(),
                None => (0..a.ndim()).rev().collect(),
            };
            let new_dims: Vec<i64> = perm.iter().map(|&ax| a.0[ax]).collect();
            Ok(Shape::new(new_dims))
        }
    }
}

fn validate_axis(axis: i32, ndim: usize) -> Result<usize, ShapeError> {
    resolve_axis(axis, ndim)
}

fn resolve_axis(axis: i32, ndim: usize) -> Result<usize, ShapeError> {
    let ndim_i = ndim as i32;
    let resolved = if axis < 0 { ndim_i + axis } else { axis };
    if resolved < 0 || resolved >= ndim_i {
        return Err(ShapeError::InvalidAxis { axis, ndim });
    }
    Ok(resolved as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(dims: &[i64]) -> Shape {
        Shape::new(dims.to_vec())
    }

    #[test]
    fn test_binary_same_shape() {
        let a = s(&[2, 3]);
        let result = infer_shape(&OpKind::Add, &[&a, &a]).unwrap();
        assert_eq!(result, s(&[2, 3]));
    }

    #[test]
    fn test_binary_broadcast() {
        let a = s(&[2, 1]);
        let b = s(&[1, 3]);
        let result = infer_shape(&OpKind::Mul, &[&a, &b]).unwrap();
        assert_eq!(result, s(&[2, 3]));
    }

    #[test]
    fn test_binary_incompatible() {
        let a = s(&[2, 3]);
        let b = s(&[2, 4]);
        assert!(infer_shape(&OpKind::Add, &[&a, &b]).is_err());
    }

    #[test]
    fn test_unary_preserves_shape() {
        let a = s(&[3, 4]);
        assert_eq!(infer_shape(&OpKind::Neg, &[&a]).unwrap(), s(&[3, 4]));
        assert_eq!(infer_shape(&OpKind::Silu, &[&a]).unwrap(), s(&[3, 4]));
    }

    #[test]
    fn test_sum_axis() {
        let a = s(&[2, 3, 4]);
        let result = infer_shape(&OpKind::Sum { axis: Some(1) }, &[&a]).unwrap();
        assert_eq!(result, s(&[2, 4]));
    }

    #[test]
    fn test_sum_all() {
        let a = s(&[2, 3]);
        let result = infer_shape(&OpKind::Sum { axis: None }, &[&a]).unwrap();
        assert_eq!(result, s(&[1]));
    }

    #[test]
    fn test_sum_negative_axis() {
        let a = s(&[2, 3, 4]);
        let result = infer_shape(&OpKind::Sum { axis: Some(-1) }, &[&a]).unwrap();
        assert_eq!(result, s(&[2, 3]));
    }

    #[test]
    fn test_matmul() {
        let a = s(&[2, 3]);
        let b = s(&[3, 4]);
        let result = infer_shape(&OpKind::MatMul, &[&a, &b]).unwrap();
        assert_eq!(result, s(&[2, 4]));
    }

    #[test]
    fn test_matmul_mismatch() {
        let a = s(&[2, 3]);
        let b = s(&[4, 5]);
        assert!(infer_shape(&OpKind::MatMul, &[&a, &b]).is_err());
    }

    #[test]
    fn test_transpose_default() {
        let a = s(&[2, 3]);
        let result = infer_shape(&OpKind::Transpose { axes: None }, &[&a]).unwrap();
        assert_eq!(result, s(&[3, 2]));
    }

    #[test]
    fn test_transpose_custom() {
        let a = s(&[2, 3, 4]);
        let result = infer_shape(
            &OpKind::Transpose {
                axes: Some(vec![2, 0, 1]),
            },
            &[&a],
        )
        .unwrap();
        assert_eq!(result, s(&[4, 2, 3]));
    }

    #[test]
    fn test_reshape() {
        let a = s(&[2, 3]);
        let result = infer_shape(
            &OpKind::Reshape {
                new_shape: s(&[3, 2]),
            },
            &[&a],
        )
        .unwrap();
        assert_eq!(result, s(&[3, 2]));
    }

    #[test]
    fn test_softmax_preserves_shape() {
        let a = s(&[2, 3]);
        let result = infer_shape(&OpKind::Softmax { axis: 1 }, &[&a]).unwrap();
        assert_eq!(result, s(&[2, 3]));
    }

    #[test]
    fn test_layer_norm_preserves_shape() {
        let a = s(&[4, 8]);
        let result = infer_shape(&OpKind::LayerNorm { eps: 1e-5 }, &[&a]).unwrap();
        assert_eq!(result, s(&[4, 8]));
    }
}

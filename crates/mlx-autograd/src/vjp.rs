//! VJP (Vector-Jacobian Product) implementations for each supported op.

use mlx_core::graph::OpKind;
use mlx_core::{MlxError, Result, Shape, Tensor};

/// Compute VJP for a single op: given the output gradient, return gradients
/// for each input.
pub fn vjp(
    op: &OpKind,
    inputs: &[Tensor],
    _output: &Tensor,
    grad_output: &Tensor,
) -> Result<Vec<Tensor>> {
    match op {
        // ── Elementwise ──────────────────────────────────────────────
        OpKind::Add => {
            // d(a+b)/da = 1, d(a+b)/db = 1
            Ok(vec![grad_output.clone(), grad_output.clone()])
        }

        OpKind::Sub => {
            // d(a-b)/da = 1, d(a-b)/db = -1
            Ok(vec![grad_output.clone(), grad_output.neg()])
        }

        OpKind::Mul => {
            // d(a*b)/da = b, d(a*b)/db = a
            let grad_a = grad_output.mul(&inputs[1])?;
            let grad_b = grad_output.mul(&inputs[0])?;
            Ok(vec![grad_a, grad_b])
        }

        OpKind::Div => {
            // d(a/b)/da = 1/b
            // d(a/b)/db = -a/b^2
            let grad_a = grad_output.div(&inputs[1])?;
            let neg_a = inputs[0].neg();
            let b_sq = inputs[1].mul(&inputs[1])?;
            let grad_b = grad_output.mul(&neg_a)?.div(&b_sq)?;
            Ok(vec![grad_a, grad_b])
        }

        OpKind::Neg => {
            // d(-a)/da = -1
            Ok(vec![grad_output.neg()])
        }

        // ── Reductions ───────────────────────────────────────────────
        OpKind::Sum { axis } => {
            let input_shape = inputs[0].shape();
            match axis {
                None => {
                    // sum_all: broadcast scalar grad to input shape
                    let grad_input = grad_output.broadcast_to(input_shape)?;
                    Ok(vec![grad_input])
                }
                Some(ax) => {
                    // sum_axis: unsqueeze the reduced axis, then broadcast
                    let resolved = if *ax < 0 {
                        (input_shape.ndim() as i32 + ax) as usize
                    } else {
                        *ax as usize
                    };
                    let mut unsqueezed = grad_output.shape().0.clone();
                    unsqueezed.insert(resolved, 1);
                    let reshaped = grad_output.reshape(&Shape::new(unsqueezed))?;
                    let grad_input = reshaped.broadcast_to(input_shape)?;
                    Ok(vec![grad_input])
                }
            }
        }

        // ── Linear algebra ───────────────────────────────────────────
        OpKind::MatMul => {
            // C = A @ B
            // dL/dA = dL/dC @ B^T
            // dL/dB = A^T @ dL/dC
            let b_t = inputs[1].transpose(None)?;
            let a_t = inputs[0].transpose(None)?;
            let grad_a = grad_output.matmul(&b_t)?;
            let grad_b = a_t.matmul(grad_output)?;
            Ok(vec![grad_a, grad_b])
        }

        // ── Shape ops (pass-through or rearrange grad) ───────────────
        OpKind::Reshape { .. } => {
            // Reshape grad back to input shape
            let grad_input = grad_output.reshape(inputs[0].shape())?;
            Ok(vec![grad_input])
        }

        OpKind::Transpose { axes } => {
            // Invert the permutation
            let perm = match axes {
                Some(ax) => ax.clone(),
                None => (0..inputs[0].shape().ndim()).rev().collect(),
            };
            let mut inv_perm = vec![0usize; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                inv_perm[p] = i;
            }
            let grad_input = grad_output.transpose(Some(&inv_perm))?;
            Ok(vec![grad_input])
        }

        // ── Broadcasting (grad is reduced back to original shape) ────
        OpKind::Broadcast { .. } => {
            // Reverse of broadcast: sum over the broadcasted dimensions
            let input_shape = inputs[0].shape();
            let output_shape = grad_output.shape();
            let mut grad_input = grad_output.clone();

            // Sum over dimensions that were added (leading dims from padding)
            let pad = output_shape.ndim() - input_shape.ndim();
            for _ in 0..pad {
                grad_input = grad_input.sum_axis(0)?;
            }

            // Sum over dimensions that were broadcast from size 1
            for i in 0..input_shape.ndim() {
                if input_shape.0[i] == 1 && grad_input.shape().0[i] != 1 {
                    grad_input = grad_input.sum_axis(i as i32)?;
                    // sum_axis removes the dim, we need to keep it as size 1
                    let mut reshape_dims = grad_input.shape().0.clone();
                    reshape_dims.insert(i, 1);
                    grad_input = grad_input.reshape(&Shape::new(reshape_dims))?;
                }
            }

            Ok(vec![grad_input])
        }

        _ => Err(MlxError::InvalidArgument(format!(
            "VJP not implemented for {:?}",
            op
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::{Device, Shape, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn t(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, &Shape::new(shape.to_vec()), &cpu()).unwrap()
    }

    #[test]
    fn test_vjp_add() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let grad_out = t(&[1.0, 1.0], &[2]);
        let grads = vjp(&OpKind::Add, &[a, b], &t(&[4.0, 6.0], &[2]), &grad_out).unwrap();
        assert_eq!(grads[0].to_vec_f32().unwrap(), vec![1.0, 1.0]);
        assert_eq!(grads[1].to_vec_f32().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_vjp_mul() {
        let a = t(&[2.0, 3.0], &[2]);
        let b = t(&[4.0, 5.0], &[2]);
        let grad_out = t(&[1.0, 1.0], &[2]);
        let grads = vjp(&OpKind::Mul, &[a, b], &t(&[8.0, 15.0], &[2]), &grad_out).unwrap();
        // grad_a = grad * b = [4, 5], grad_b = grad * a = [2, 3]
        assert_eq!(grads[0].to_vec_f32().unwrap(), vec![4.0, 5.0]);
        assert_eq!(grads[1].to_vec_f32().unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_vjp_neg() {
        let a = t(&[1.0, 2.0], &[2]);
        let grad_out = t(&[1.0, 1.0], &[2]);
        let grads = vjp(&OpKind::Neg, &[a], &t(&[-1.0, -2.0], &[2]), &grad_out).unwrap();
        assert_eq!(grads[0].to_vec_f32().unwrap(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_vjp_sum_all() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let grad_out = t(&[1.0], &[1]);
        let grads = vjp(
            &OpKind::Sum { axis: None },
            &[a],
            &t(&[6.0], &[1]),
            &grad_out,
        )
        .unwrap();
        assert_eq!(grads[0].to_vec_f32().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_vjp_sum_axis() {
        // Input [2,3], sum axis 0 -> [3]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let grad_out = t(&[1.0, 1.0, 1.0], &[3]);
        let grads = vjp(
            &OpKind::Sum { axis: Some(0) },
            &[a],
            &t(&[5.0, 7.0, 9.0], &[3]),
            &grad_out,
        )
        .unwrap();
        let result = grads[0].to_vec_f32().unwrap();
        // Each element in [2,3] should get grad 1.0 (broadcast from [3])
        assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_vjp_broadcast_scalar() {
        // Broadcast [1] -> [3]: gradient should sum to 3
        let input = t(&[2.0], &[1]);
        let grad_out = t(&[1.0, 1.0, 1.0], &[3]);
        let grads = vjp(
            &OpKind::Broadcast {
                target_shape: Shape::new(vec![3]),
            },
            &[input],
            &t(&[2.0, 2.0, 2.0], &[3]),
            &grad_out,
        )
        .unwrap();
        // Gradient should sum over broadcasted dimension
        assert_eq!(grads[0].to_vec_f32().unwrap(), vec![3.0]);
    }

    #[test]
    fn test_vjp_broadcast_2d() {
        // Broadcast [2,1] -> [2,3]: each row collects sum of its broadcasted axis
        let input = t(&[2.0, 5.0], &[2, 1]);
        let grad_out = t(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
        let grads = vjp(
            &OpKind::Broadcast {
                target_shape: Shape::new(vec![2, 3]),
            },
            &[input],
            &t(&[2.0, 2.0, 2.0, 5.0, 5.0, 5.0], &[2, 3]),
            &grad_out,
        )
        .unwrap();
        // Each row sums its 3 gradient values: [1+1+1, 1+1+1] = [3, 3]
        // But result shape should be [2,1]
        let result = grads[0].to_vec_f32().unwrap();
        assert_eq!(result, vec![3.0, 3.0]);
    }
}

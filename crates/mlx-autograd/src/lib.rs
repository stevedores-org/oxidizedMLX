//! Reverse-mode automatic differentiation (autograd).
//!
//! Provides `grad(f)` and `value_and_grad(f)` for scalar losses, with a VJP
//! (Vector-Jacobian Product) registry covering elementwise ops, matmul, and
//! reductions.

mod vjp;

use std::collections::HashMap;
use std::sync::Arc;

use mlx_core::backend::Stream;
use mlx_core::graph::{Node, OpKind};
use mlx_core::{Device, MlxError, NodeId, Result, Tensor};

/// Compute the gradient of a scalar function w.r.t. input.
///
/// ```ignore
/// let dx = grad(|x| x.mul(&x)?.sum_all(), &x)?;
/// ```
pub fn grad<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: FnOnce(&Tensor) -> Result<Tensor>,
{
    let (_, g) = value_and_grad(f, x)?;
    Ok(g)
}

/// Compute both the value and gradient of a scalar function.
pub fn value_and_grad<F>(f: F, x: &Tensor) -> Result<(Tensor, Tensor)>
where
    F: FnOnce(&Tensor) -> Result<Tensor>,
{
    let loss = f(x)?;
    if loss.numel() != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "grad requires scalar loss, got {} elements",
            loss.numel()
        )));
    }

    // Evaluate forward graph to materialize all values.
    loss.eval()?;

    // Seed: d(loss)/d(loss) = 1.0
    let ones = Tensor::ones(loss.shape(), loss.dtype(), loss.device())?;

    let grad = backward(&loss, x, ones)?;
    Ok((loss, grad))
}

/// Reverse-mode backward pass.
fn backward(loss: &Tensor, wrt: &Tensor, seed: Tensor) -> Result<Tensor> {
    let stream = loss.stream();
    let order = stream.topo_sort(&[loss.node_id()]);

    let mut grads: HashMap<NodeId, Tensor> = HashMap::new();
    grads.insert(loss.node_id(), seed);

    for &node_id in order.iter().rev() {
        let grad_output = match grads.remove(&node_id) {
            Some(g) => g,
            None => continue,
        };

        let node = match stream.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Source nodes don't have inputs to propagate to, but keep their grad.
        if matches!(node.op, OpKind::Constant | OpKind::Parameter) {
            grads.insert(node_id, grad_output);
            continue;
        }

        // Reconstruct lightweight tensor handles for inputs and output.
        let input_tensors: Vec<Tensor> = node
            .inputs
            .iter()
            .map(|&id| tensor_from_node(&stream, id, loss.device()))
            .collect::<Result<Vec<_>>>()?;

        let output_tensor =
            tensor_from_node_info(&node, node_id, loss.device(), Arc::clone(&stream));

        // Compute VJPs for this op.
        let input_grads = vjp::vjp(&node.op, &input_tensors, &output_tensor, &grad_output)?;

        // Accumulate into grad table.
        for (inp, ig) in input_tensors.iter().zip(input_grads) {
            let id = inp.node_id();
            match grads.remove(&id) {
                Some(existing) => {
                    grads.insert(id, existing.add(&ig)?);
                }
                None => {
                    grads.insert(id, ig);
                }
            }
        }
    }

    grads.remove(&wrt.node_id()).ok_or_else(|| {
        MlxError::InvalidArgument("gradient not found — input may not affect the loss".into())
    })
}

/// Create a Tensor handle from a node ID by looking up its metadata in the stream.
fn tensor_from_node(stream: &Arc<Stream>, id: NodeId, device: &Device) -> Result<Tensor> {
    let node = stream
        .get_node(id)
        .ok_or_else(|| MlxError::InvalidArgument(format!("node {:?} not found in graph", id)))?;
    Ok(Tensor::from_node_id(
        id,
        node.meta.shape.clone(),
        node.meta.dtype,
        device.clone(),
        Arc::clone(stream),
    ))
}

/// Create a Tensor handle directly from a Node (avoids extra graph lookup).
fn tensor_from_node_info(
    node: &Node,
    node_id: NodeId,
    device: &Device,
    stream: Arc<Stream>,
) -> Tensor {
    Tensor::from_node_id(
        node_id,
        node.meta.shape.clone(),
        node.meta.dtype,
        device.clone(),
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::{Device, Shape, Tensor};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn s(dims: &[i64]) -> Shape {
        Shape::new(dims.to_vec())
    }

    fn t(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, &s(shape), &cpu()).unwrap()
    }

    /// Numerical gradient via finite differences (central difference).
    fn numerical_grad<F>(f: &F, x_data: &[f32], shape: &[i64], eps: f32) -> Vec<f32>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let mut grad = vec![0.0f32; x_data.len()];
        for i in 0..x_data.len() {
            let mut x_plus = x_data.to_vec();
            let mut x_minus = x_data.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let tp = Tensor::from_f32(&x_plus, &s(shape), &cpu()).unwrap();
            let tm = Tensor::from_f32(&x_minus, &s(shape), &cpu()).unwrap();
            let fp = f(&tp).unwrap().to_vec_f32().unwrap()[0];
            let fm = f(&tm).unwrap().to_vec_f32().unwrap()[0];
            grad[i] = (fp - fm) / (2.0 * eps);
        }
        grad
    }

    // ── Basic VJP tests ──────────────────────────────────────────────

    #[test]
    fn test_grad_identity() {
        // f(x) = sum(x), grad = ones
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let dx = grad(|x| x.sum_all(), &x).unwrap();
        let vals = dx.to_vec_f32().unwrap();
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_grad_add() {
        // f(x) = sum(x + x) = 2*sum(x), grad = 2*ones
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let dx = grad(|x| x.add(x)?.sum_all(), &x).unwrap();
        let vals = dx.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals, &[2.0, 2.0, 2.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_grad_mul_self() {
        // f(x) = sum(x * x) = sum(x^2), grad = 2*x
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let dx = grad(|x| x.mul(x)?.sum_all(), &x).unwrap();
        let vals = dx.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals, &[2.0, 4.0, 6.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_grad_neg() {
        // f(x) = sum(-x), grad = -ones
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let dx = grad(|x| x.neg().sum_all(), &x).unwrap();
        let vals = dx.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals, &[-1.0, -1.0, -1.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_grad_sub() {
        // f(x) = sum(x - c), grad w.r.t. x = ones
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let c = t(&[0.5, 0.5, 0.5], &[3]);
        let dx = grad(
            |x| {
                let diff = x.sub(&c)?;
                diff.sum_all()
            },
            &x,
        )
        .unwrap();
        let vals = dx.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals, &[1.0, 1.0, 1.0], 1e-5, 1e-5);
    }

    #[test]
    fn test_grad_div() {
        // f(x) = sum(x / c), grad w.r.t. x = 1/c
        let x = t(&[2.0, 4.0, 6.0], &[3]);
        let c = t(&[2.0, 2.0, 2.0], &[3]);
        let dx = grad(
            |x| {
                let q = x.div(&c)?;
                q.sum_all()
            },
            &x,
        )
        .unwrap();
        let vals = dx.to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals, &[0.5, 0.5, 0.5], 1e-5, 1e-5);
    }

    // ── MatMul gradient ──────────────────────────────────────────────

    #[test]
    fn test_grad_matmul() {
        // f(X) = sum(X @ W), grad w.r.t. X = ones @ W^T
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let w = t(&[0.5, 0.3, 0.7, 0.1], &[2, 2]);
        let dx = grad(
            |x| {
                let y = x.matmul(&w)?;
                y.sum_all()
            },
            &x,
        )
        .unwrap();
        let vals = dx.to_vec_f32().unwrap();
        // grad = ones([2,2]) @ W^T = [[1,1],[1,1]] @ [[0.5,0.7],[0.3,0.1]]
        // = [[0.8, 0.8], [0.8, 0.8]]
        mlx_conformance::assert_allclose(&vals, &[0.8, 0.8, 0.8, 0.8], 1e-5, 1e-5);
    }

    // ── Finite-difference gradient checks ────────────────────────────

    #[test]
    fn test_grad_check_mul() {
        let x_data = [1.0, 2.0, 3.0];
        let c = t(&[2.0, 3.0, 4.0], &[3]);

        let f = |x: &Tensor| x.mul(&c)?.sum_all();

        let x = t(&x_data, &[3]);
        let analytical = grad(f, &x).unwrap().to_vec_f32().unwrap();
        let numerical = numerical_grad(&f, &x_data, &[3], 1e-3);

        mlx_conformance::assert_allclose(&analytical, &numerical, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_check_div() {
        let x_data = [2.0, 4.0, 6.0];
        let c = t(&[2.0, 3.0, 4.0], &[3]);

        let f = |x: &Tensor| x.div(&c)?.sum_all();

        let x = t(&x_data, &[3]);
        let analytical = grad(f, &x).unwrap().to_vec_f32().unwrap();
        let numerical = numerical_grad(&f, &x_data, &[3], 1e-3);

        mlx_conformance::assert_allclose(&analytical, &numerical, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_check_chain() {
        // f(x) = sum((x * x) + x) = sum(x^2 + x), grad = 2x + 1
        let x_data = [1.0, 2.0, 3.0];

        let f = |x: &Tensor| {
            let sq = x.mul(x)?;
            let added = sq.add(x)?;
            added.sum_all()
        };

        let x = t(&x_data, &[3]);
        let analytical = grad(f, &x).unwrap().to_vec_f32().unwrap();
        let numerical = numerical_grad(&f, &x_data, &[3], 1e-3);

        mlx_conformance::assert_allclose(&analytical, &numerical, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_check_matmul() {
        let x_data = [1.0, 2.0, 3.0, 4.0];
        let w = t(&[0.5, 0.3, 0.7, 0.1], &[2, 2]);

        let f = |x: &Tensor| x.matmul(&w)?.sum_all();

        let x = t(&x_data, &[2, 2]);
        let analytical = grad(f, &x).unwrap().to_vec_f32().unwrap();
        let numerical = numerical_grad(&f, &x_data, &[2, 2], 1e-3);

        mlx_conformance::assert_allclose(&analytical, &numerical, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_sum_axis() {
        // f(x) = sum_all(sum_axis0(x)), where x is [2,3]
        // sum_axis0([2,3]) -> [3], then sum_all -> scalar
        // This is equivalent to sum_all(x), so grad = ones
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let f = |x: &Tensor| {
            let s = x.sum_axis(0)?;
            s.sum_all()
        };

        let x = t(&x_data, &[2, 3]);
        let analytical = grad(f, &x).unwrap().to_vec_f32().unwrap();
        let numerical = numerical_grad(&f, &x_data, &[2, 3], 1e-3);

        mlx_conformance::assert_allclose(&analytical, &numerical, 1e-3, 1e-3);
    }

    #[test]
    fn test_value_and_grad() {
        let x = t(&[2.0, 3.0], &[2]);
        let (val, dx) = value_and_grad(|x| x.mul(x)?.sum_all(), &x).unwrap();
        assert_eq!(val.to_vec_f32().unwrap(), vec![13.0]); // 4 + 9
        mlx_conformance::assert_allclose(&dx.to_vec_f32().unwrap(), &[4.0, 6.0], 1e-5, 1e-5);
    }
}

//! Tensor type — a lazy handle to a node in the computation graph.
//!
//! Operations on tensors record nodes in the graph. Actual computation is
//! deferred until `eval()` (or `to_vec_f32()`) is called, at which point the
//! stream topologically sorts the subgraph and dispatches to the backend.

use std::sync::Arc;

use smallvec::SmallVec;

use crate::backend::{Stream, default_stream};
use crate::graph::{OpKind, TensorMeta};
use crate::{DType, MlxError, NodeId, Result, Shape};

/// Compute device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    /// Return the default device for this platform.
    pub fn default_device() -> Self {
        #[cfg(target_os = "macos")]
        {
            Device::Gpu
        }
        #[cfg(not(target_os = "macos"))]
        {
            Device::Cpu
        }
    }
}

/// A tensor handle.
///
/// In the lazy graph model a `Tensor` is a lightweight reference to a node in
/// the computation graph. Operations build up the graph; actual computation
/// happens when `eval()` is called (or implicitly via `to_vec_f32()`).
#[derive(Clone)]
pub struct Tensor {
    node_id: NodeId,
    shape: Shape,
    dtype: DType,
    device: Device,
    stream: Arc<Stream>,
}

impl Tensor {
    // ── Constructors ────────────────────────────────────────────────────

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel() as usize;
        Self::from_data(vec![0.0; n], shape, dtype, device)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel() as usize;
        Self::from_data(vec![1.0; n], shape, dtype, device)
    }

    /// Create a tensor from f32 data.
    pub fn from_f32(data: &[f32], shape: &Shape, device: &Device) -> Result<Self> {
        let expected = shape.numel() as usize;
        if data.len() != expected {
            return Err(MlxError::InvalidArgument(format!(
                "data length {} does not match shape {} (expected {})",
                data.len(),
                shape,
                expected,
            )));
        }
        Self::from_data(data.to_vec(), shape, DType::F32, device)
    }

    fn from_data(data: Vec<f32>, shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let stream = default_stream();
        let meta = TensorMeta {
            shape: shape.clone(),
            dtype,
        };
        let node_id = stream.add_constant(data, meta);
        Ok(Self {
            node_id,
            shape: shape.clone(),
            dtype,
            device: device.clone(),
            stream,
        })
    }

    fn lazy_op(
        &self,
        op: OpKind,
        inputs: SmallVec<[NodeId; 2]>,
        shape: Shape,
        dtype: DType,
    ) -> Self {
        let meta = TensorMeta {
            shape: shape.clone(),
            dtype,
        };
        let node_id = self.stream.add_op(op, inputs, meta);
        Tensor {
            node_id,
            shape,
            dtype,
            device: self.device.clone(),
            stream: Arc::clone(&self.stream),
        }
    }

    // ── Elementwise ops ─────────────────────────────────────────────────

    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::Add,
            SmallVec::from_slice(&[self.node_id, rhs.node_id]),
            self.shape.clone(),
            self.dtype,
        ))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::Sub,
            SmallVec::from_slice(&[self.node_id, rhs.node_id]),
            self.shape.clone(),
            self.dtype,
        ))
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::Mul,
            SmallVec::from_slice(&[self.node_id, rhs.node_id]),
            self.shape.clone(),
            self.dtype,
        ))
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::Div,
            SmallVec::from_slice(&[self.node_id, rhs.node_id]),
            self.shape.clone(),
            self.dtype,
        ))
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        self.lazy_op(
            OpKind::Neg,
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        )
    }

    // ── Reductions ──────────────────────────────────────────────────────

    /// Sum along an axis.
    pub fn sum_axis(&self, axis: i32) -> Result<Tensor> {
        let ndim = self.shape.ndim() as i32;
        let ax = if axis < 0 { ndim + axis } else { axis };
        if ax < 0 || ax >= ndim {
            return Err(MlxError::InvalidArgument(format!(
                "axis {axis} out of range for ndim {ndim}"
            )));
        }
        let mut new_dims: Vec<i64> = self.shape.0.clone();
        new_dims.remove(ax as usize);
        if new_dims.is_empty() {
            new_dims.push(1);
        }
        Ok(self.lazy_op(
            OpKind::Sum { axis: Some(axis) },
            SmallVec::from_slice(&[self.node_id]),
            Shape::new(new_dims),
            self.dtype,
        ))
    }

    /// Sum all elements to a scalar.
    pub fn sum_all(&self) -> Result<Tensor> {
        Ok(self.lazy_op(
            OpKind::Sum { axis: None },
            SmallVec::from_slice(&[self.node_id]),
            Shape::new(vec![1]),
            self.dtype,
        ))
    }

    // ── Linear algebra ──────────────────────────────────────────────────

    /// Matrix multiplication (2D only for now).
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape.ndim() != 2 || rhs.shape.ndim() != 2 {
            return Err(MlxError::InvalidArgument(
                "matmul requires 2D tensors".to_string(),
            ));
        }
        let m = self.shape.0[0];
        let k = self.shape.0[1];
        let k2 = rhs.shape.0[0];
        let n = rhs.shape.0[1];
        if k != k2 {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::MatMul,
            SmallVec::from_slice(&[self.node_id, rhs.node_id]),
            Shape::new(vec![m, n]),
            self.dtype,
        ))
    }

    // ── Shape manipulation ──────────────────────────────────────────────

    /// Reshape the tensor.
    pub fn reshape(&self, new_shape: &Shape) -> Result<Tensor> {
        if self.shape.numel() != new_shape.numel() {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: new_shape.0.clone(),
            });
        }
        Ok(self.lazy_op(
            OpKind::Reshape {
                new_shape: new_shape.clone(),
            },
            SmallVec::from_slice(&[self.node_id]),
            new_shape.clone(),
            self.dtype,
        ))
    }

    /// Transpose (reverses axes by default, or use specified permutation).
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Tensor> {
        let perm: Vec<usize> = match axes {
            Some(ax) => {
                if ax.len() != self.shape.ndim() {
                    return Err(MlxError::InvalidArgument(
                        "transpose axes length must match ndim".into(),
                    ));
                }
                ax.to_vec()
            }
            None => (0..self.shape.ndim()).rev().collect(),
        };
        let new_dims: Vec<i64> = perm.iter().map(|&ax| self.shape.0[ax]).collect();
        Ok(self.lazy_op(
            OpKind::Transpose { axes: Some(perm) },
            SmallVec::from_slice(&[self.node_id]),
            Shape::new(new_dims),
            self.dtype,
        ))
    }

    // ── Activations ─────────────────────────────────────────────────────

    /// Softmax along an axis.
    pub fn softmax(&self, axis: i32) -> Result<Tensor> {
        let ndim = self.shape.ndim() as i32;
        let ax = if axis < 0 { ndim + axis } else { axis };
        if ax < 0 || ax >= ndim {
            return Err(MlxError::InvalidArgument(format!(
                "axis {axis} out of range for ndim {ndim}"
            )));
        }
        Ok(self.lazy_op(
            OpKind::Softmax { axis },
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        ))
    }

    /// SiLU (Sigmoid Linear Unit) activation.
    pub fn silu(&self) -> Tensor {
        self.lazy_op(
            OpKind::Silu,
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        )
    }

    /// GELU (Gaussian Error Linear Unit) activation.
    pub fn gelu(&self) -> Tensor {
        self.lazy_op(
            OpKind::Gelu,
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        )
    }

    // ── Normalization ───────────────────────────────────────────────────

    /// Layer normalization over the last dimension.
    pub fn layer_norm(&self, eps: f32) -> Tensor {
        self.lazy_op(
            OpKind::LayerNorm { eps },
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        )
    }

    /// RMS normalization over the last dimension.
    pub fn rms_norm(&self, eps: f32) -> Tensor {
        self.lazy_op(
            OpKind::RmsNorm { eps },
            SmallVec::from_slice(&[self.node_id]),
            self.shape.clone(),
            self.dtype,
        )
    }

    // ── Materialization ─────────────────────────────────────────────────

    /// Materialize the tensor — triggers evaluation of the computation graph.
    pub fn eval(&self) -> Result<()> {
        self.stream.eval(self.node_id)
    }

    /// Copy data out as Vec<f32>. Triggers evaluation if needed.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.eval()?;
        self.stream
            .get_buffer(self.node_id)
            .ok_or_else(|| MlxError::InvalidArgument("buffer not found after eval".into()))
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Get the tensor shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the tensor dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the tensor device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of elements.
    pub fn numel(&self) -> i64 {
        self.shape.numel()
    }

    /// Get the graph node ID.
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Get the stream this tensor belongs to.
    pub fn stream(&self) -> Arc<Stream> {
        Arc::clone(&self.stream)
    }

    /// Reconstruct a tensor handle from a node ID and metadata.
    ///
    /// Used by autograd to create handles for graph introspection.
    pub fn from_node_id(
        node_id: NodeId,
        shape: Shape,
        dtype: DType,
        device: Device,
        stream: Arc<Stream>,
    ) -> Self {
        Self {
            node_id,
            shape,
            dtype,
            device,
            stream,
        }
    }

    /// Broadcast this tensor to the target shape (numpy-style rules).
    pub fn broadcast_to(&self, target: &Shape) -> Result<Tensor> {
        if &self.shape == target {
            return Ok(self.clone());
        }
        // Validate broadcast compatibility: dimensions are compared from the right.
        let in_ndim = self.shape.ndim();
        let out_ndim = target.ndim();
        if in_ndim > out_ndim {
            return Err(MlxError::InvalidArgument(format!(
                "cannot broadcast shape {} to {}",
                self.shape, target
            )));
        }
        let pad = out_ndim - in_ndim;
        for i in 0..in_ndim {
            let in_dim = self.shape.0[i];
            let out_dim = target.0[pad + i];
            if in_dim != 1 && in_dim != out_dim {
                return Err(MlxError::InvalidArgument(format!(
                    "cannot broadcast shape {} to {}",
                    self.shape, target
                )));
            }
        }
        Ok(self.lazy_op(
            OpKind::Broadcast {
                target_shape: target.clone(),
            },
            SmallVec::from_slice(&[self.node_id]),
            target.clone(),
            self.dtype,
        ))
    }
}

impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&Shape::new(vec![2, 3]), DType::F32, &cpu()).unwrap();
        assert_eq!(t.to_vec_f32().unwrap(), vec![0.0; 6]);
        assert_eq!(t.shape(), &Shape::new(vec![2, 3]));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&Shape::new(vec![3]), DType::F32, &cpu()).unwrap();
        assert_eq!(t.to_vec_f32().unwrap(), vec![1.0; 3]);
    }

    #[test]
    fn test_from_f32() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
        assert_eq!(t.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_from_f32_shape_mismatch() {
        let r = Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![3]), &cpu());
        assert!(r.is_err());
    }

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_f32(&[5.0, 7.0, 9.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let c = a.sub(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_f32(&[2.0, 3.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div() {
        let a = Tensor::from_f32(&[10.0, 9.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = a.div(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![5.0, 3.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let b = a.neg();
        assert_eq!(b.to_vec_f32().unwrap(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &Shape::new(vec![2, 3]),
            &cpu(),
        )
        .unwrap();
        let s0 = a.sum_axis(0).unwrap();
        assert_eq!(s0.to_vec_f32().unwrap(), vec![5.0, 7.0, 9.0]);
        let s1 = a.sum_axis(1).unwrap();
        assert_eq!(s1.to_vec_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_all() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let s = a.sum_all().unwrap();
        assert_eq!(s.to_vec_f32().unwrap(), vec![6.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let s = a.softmax(0).unwrap();
        let vals = s.to_vec_f32().unwrap();
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &Shape::new(vec![2, 3]),
            &cpu(),
        )
        .unwrap();
        let b = a.reshape(&Shape::new(vec![3, 2])).unwrap();
        assert_eq!(b.shape(), &Shape::new(vec![3, 2]));
        assert_eq!(b.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &Shape::new(vec![2, 3]),
            &cpu(),
        )
        .unwrap();
        let b = a.transpose(None).unwrap();
        assert_eq!(b.shape(), &Shape::new(vec![3, 2]));
        // [[1,2,3],[4,5,6]] transposed = [[1,4],[2,5],[3,6]]
        assert_eq!(b.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_operator_add() {
        let a = Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = (&a + &b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_operator_neg() {
        let a = Tensor::from_f32(&[1.0, -2.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = -&a;
        assert_eq!(b.to_vec_f32().unwrap(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_lazy_chain() {
        // Build a chain: (a + b) * c — nothing evaluated until to_vec_f32
        let a = Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = Tensor::from_f32(&[2.0, 3.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let d = a.add(&b).unwrap().mul(&c).unwrap();
        // Only now does evaluation happen:
        assert_eq!(d.to_vec_f32().unwrap(), vec![8.0, 18.0]);
    }

    #[test]
    fn test_silu() {
        let a = Tensor::from_f32(&[0.0, 1.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = a.silu();
        let vals = b.to_vec_f32().unwrap();
        assert!((vals[0]).abs() < 1e-6);
        assert!((vals[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let b = a.layer_norm(1e-5);
        let vals = b.to_vec_f32().unwrap();
        let mean: f32 = vals.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }
}

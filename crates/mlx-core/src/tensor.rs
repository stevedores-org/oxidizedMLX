//! Tensor type — a lazy handle to a node in the computation graph.

use crate::{DType, MlxError, Result, Shape};

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
/// In the lazy graph model, a `Tensor` is a lightweight reference to a node in
/// the computation graph. Operations on tensors build up the graph; actual
/// computation happens when `eval()` is called.
///
/// Currently this is a placeholder that stores data eagerly for the CPU-only
/// path. The FFI backend path (feature `ffi`) will wrap opaque MLX handles.
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
    device: Device,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel() as usize;
        Ok(Self {
            data: vec![0.0; n],
            shape: shape.clone(),
            dtype,
            device: device.clone(),
        })
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        let n = shape.numel() as usize;
        Ok(Self {
            data: vec![1.0; n],
            shape: shape.clone(),
            dtype,
            device: device.clone(),
        })
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
        Ok(Self {
            data: data.to_vec(),
            shape: shape.clone(),
            dtype: DType::F32,
            device: device.clone(),
        })
    }

    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape != rhs.shape {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: rhs.shape.0.clone(),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Matrix multiplication (2D only for now).
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.shape.ndim() != 2 || rhs.shape.ndim() != 2 {
            return Err(MlxError::InvalidArgument(
                "matmul requires 2D tensors".to_string(),
            ));
        }
        let m = self.shape.0[0] as usize;
        let k = self.shape.0[1] as usize;
        let k2 = rhs.shape.0[0] as usize;
        let n = rhs.shape.0[1] as usize;
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
                    sum += self.data[i * k + p] * rhs.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Ok(Tensor {
            data,
            shape: Shape::new(vec![m as i64, n as i64]),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Sum along an axis.
    pub fn sum_axis(&self, axis: i32) -> Result<Tensor> {
        let ndim = self.shape.ndim() as i32;
        let ax = if axis < 0 { ndim + axis } else { axis };
        if ax < 0 || ax >= ndim {
            return Err(MlxError::InvalidArgument(format!(
                "axis {axis} out of range for ndim {ndim}"
            )));
        }
        let ax = ax as usize;

        let mut new_shape: Vec<i64> = self.shape.0.clone();
        new_shape.remove(ax);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer: usize = self.shape.0[..ax].iter().product::<i64>() as usize;
        let dim: usize = self.shape.0[ax] as usize;
        let inner: usize = self.shape.0[ax + 1..].iter().product::<i64>() as usize;
        let inner = if inner == 0 { 1 } else { inner };

        let mut data = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for d in 0..dim {
                for i in 0..inner {
                    data[o * inner + i] += self.data[o * dim * inner + d * inner + i];
                }
            }
        }

        Ok(Tensor {
            data,
            shape: Shape::new(new_shape),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Sum all elements to a scalar.
    pub fn sum_all(&self) -> Result<Tensor> {
        let sum: f32 = self.data.iter().sum();
        Ok(Tensor {
            data: vec![sum],
            shape: Shape::new(vec![1]),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Softmax along an axis.
    pub fn softmax(&self, axis: i32) -> Result<Tensor> {
        let ndim = self.shape.ndim() as i32;
        let ax = if axis < 0 { ndim + axis } else { axis };
        if ax < 0 || ax >= ndim {
            return Err(MlxError::InvalidArgument(format!(
                "axis {axis} out of range for ndim {ndim}"
            )));
        }
        let ax = ax as usize;

        let outer: usize = self.shape.0[..ax].iter().product::<i64>() as usize;
        let dim: usize = self.shape.0[ax] as usize;
        let inner: usize = self.shape.0[ax + 1..].iter().product::<i64>() as usize;
        let inner = if inner == 0 { 1 } else { inner };

        let mut data = self.data.clone();

        for o in 0..outer {
            for i in 0..inner {
                // Numerically stable softmax: subtract max first
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

        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Reshape the tensor.
    pub fn reshape(&self, new_shape: &Shape) -> Result<Tensor> {
        if self.shape.numel() != new_shape.numel() {
            return Err(MlxError::ShapeMismatch {
                expected: self.shape.0.clone(),
                got: new_shape.0.clone(),
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// Materialize the tensor (no-op for eager CPU path).
    pub fn eval(&self) {
        // In the lazy graph model, this would trigger computation.
        // For the eager CPU path, data is already materialized.
    }

    /// Copy data out as Vec<f32>.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.eval();
        Ok(self.data.clone())
    }

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
}

impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
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
    fn test_mul() {
        let a = Tensor::from_f32(&[2.0, 3.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &cpu()).unwrap();
        let c = a.matmul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
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
        // sum axis=0 -> [5, 7, 9]
        let s0 = a.sum_axis(0).unwrap();
        assert_eq!(s0.to_vec_f32().unwrap(), vec![5.0, 7.0, 9.0]);
        // sum axis=1 -> [6, 15]
        let s1 = a.sum_axis(1).unwrap();
        assert_eq!(s1.to_vec_f32().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &cpu()).unwrap();
        let s = a.softmax(0).unwrap();
        let vals = s.to_vec_f32().unwrap();
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // softmax is monotonic
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
    fn test_operator_add() {
        let a = Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &Shape::new(vec![2]), &cpu()).unwrap();
        let c = (&a + &b).unwrap();
        assert_eq!(c.to_vec_f32().unwrap(), vec![4.0, 6.0]);
    }
}

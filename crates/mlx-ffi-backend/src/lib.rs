//! MLX backend that delegates ops to the mlx-sys C ABI shim.

use mlx_core::backend::{Backend, NodeInput};
use mlx_core::graph::OpKind;
use mlx_core::types::DType;
use mlx_core::{MlxError, Result};
use mlx_sys as sys;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Handle(u64);

/// Backend that evaluates nodes by calling into the MLX C ABI shim.
pub struct MlxFfiBackend {
    device: NonNull<sys::mlx_device_t>,
    arena: Mutex<HashMap<Handle, NonNull<sys::mlx_tensor_t>>>,
    next: AtomicU64,
}

// SAFETY: The MLX C++ runtime is internally thread-safe. The NonNull pointers
// wrap MLX device/tensor handles protected by the arena Mutex.
unsafe impl Send for MlxFfiBackend {}
unsafe impl Sync for MlxFfiBackend {}

impl MlxFfiBackend {
    /// Create a backend using MLX's default device.
    pub fn new_default_device() -> Result<Self> {
        let dev = unsafe { sys::mlxrs_default_device() };
        let device = NonNull::new(dev).ok_or(MlxError::BackendUnavailable(
            "mlxrs_default_device returned null",
        ))?;
        Ok(Self {
            device,
            arena: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
        })
    }

    fn alloc_handle(&self, t: NonNull<sys::mlx_tensor_t>) -> Handle {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let handle = Handle(id);
        self.arena.lock().insert(handle, t);
        handle
    }

    fn track_tensor(&self, t: NonNull<sys::mlx_tensor_t>) -> Handle {
        self.alloc_handle(t)
    }

    fn free_tensor(&self, h: Handle) {
        if let Some(t) = self.arena.lock().remove(&h) {
            unsafe { sys::mlxrs_free_tensor(t.as_ptr()) }
        }
    }

    #[allow(dead_code)]
    fn dtype_to_sys(dtype: DType) -> sys::mlx_dtype_t {
        match dtype {
            DType::F32 => sys::mlx_dtype_t::F32,
            DType::F16 => sys::mlx_dtype_t::F16,
            DType::BF16 => sys::mlx_dtype_t::BF16,
            DType::I32 => sys::mlx_dtype_t::I32,
            DType::I64 => sys::mlx_dtype_t::I64,
        }
    }

    fn make_from_f32(&self, input: &NodeInput<'_>) -> Result<Handle> {
        if input.dtype != DType::F32 {
            return Err(MlxError::InvalidArgument(
                "mlx-ffi-backend only supports F32 inputs".into(),
            ));
        }
        let ptr = unsafe {
            sys::mlxrs_from_f32(
                self.device.as_ptr(),
                input.shape.0.as_ptr(),
                input.shape.0.len(),
                input.data.as_ptr(),
                input.data.len(),
            )
        };
        let t = NonNull::new(ptr)
            .ok_or(MlxError::BackendUnavailable("mlxrs_from_f32 returned null"))?;
        Ok(self.track_tensor(t))
    }

    fn read_f32(&self, t: NonNull<sys::mlx_tensor_t>) -> Result<Vec<f32>> {
        unsafe { sys::mlxrs_eval(t.as_ptr()) };
        let numel = unsafe { sys::mlxrs_numel(t.as_ptr()) };
        if numel < 0 {
            return Err(MlxError::InvalidArgument("mlxrs_numel failed".into()));
        }
        let mut out = vec![0.0f32; numel as usize];
        let rc = unsafe { sys::mlxrs_to_f32_vec(t.as_ptr(), out.as_mut_ptr(), out.len()) };
        if rc != 0 {
            return Err(MlxError::InvalidArgument("mlxrs_to_f32_vec failed".into()));
        }
        Ok(out)
    }

    fn call_unary(
        &self,
        input: &NodeInput<'_>,
        f: unsafe extern "C" fn(*mut sys::mlx_tensor_t) -> *mut sys::mlx_tensor_t,
    ) -> Result<Vec<f32>> {
        let h = self.make_from_f32(input)?;
        let t = self
            .arena
            .lock()
            .get(&h)
            .copied()
            .expect("tensor handle missing");
        let out_ptr = unsafe { f(t.as_ptr()) };
        let out = NonNull::new(out_ptr)
            .ok_or(MlxError::BackendUnavailable("mlxrs unary op returned null"))?;
        let out_handle = self.track_tensor(out);
        let result = self.read_f32(out)?;
        self.free_tensor(out_handle);
        self.free_tensor(h);
        Ok(result)
    }

    fn call_binary(
        &self,
        a: &NodeInput<'_>,
        b: &NodeInput<'_>,
        f: unsafe extern "C" fn(
            *mut sys::mlx_tensor_t,
            *mut sys::mlx_tensor_t,
        ) -> *mut sys::mlx_tensor_t,
    ) -> Result<Vec<f32>> {
        let ha = self.make_from_f32(a)?;
        let hb = self.make_from_f32(b)?;
        let ta = self
            .arena
            .lock()
            .get(&ha)
            .copied()
            .expect("tensor handle missing");
        let tb = self
            .arena
            .lock()
            .get(&hb)
            .copied()
            .expect("tensor handle missing");
        let out_ptr = unsafe { f(ta.as_ptr(), tb.as_ptr()) };
        let out = NonNull::new(out_ptr).ok_or(MlxError::BackendUnavailable(
            "mlxrs binary op returned null",
        ))?;
        let out_handle = self.track_tensor(out);
        let result = self.read_f32(out)?;
        self.free_tensor(out_handle);
        self.free_tensor(ha);
        self.free_tensor(hb);
        Ok(result)
    }

    fn call_sum(&self, input: &NodeInput<'_>, axis: Option<i32>) -> Result<Vec<f32>> {
        let h = self.make_from_f32(input)?;
        let t = self
            .arena
            .lock()
            .get(&h)
            .copied()
            .expect("tensor handle missing");
        let out_ptr = match axis {
            Some(ax) => unsafe { sys::mlxrs_sum(t.as_ptr(), ax as _) },
            None => unsafe { sys::mlxrs_sum_all(t.as_ptr()) },
        };
        let out =
            NonNull::new(out_ptr).ok_or(MlxError::BackendUnavailable("mlxrs_sum returned null"))?;
        let out_handle = self.track_tensor(out);
        let result = self.read_f32(out)?;
        self.free_tensor(out_handle);
        self.free_tensor(h);
        Ok(result)
    }
}

fn require_input<'a>(inputs: &'a [NodeInput<'_>], idx: usize) -> Result<&'a NodeInput<'a>> {
    inputs.get(idx).ok_or_else(|| {
        MlxError::InvalidArgument(format!("mlx-ffi-backend expected input at index {idx}"))
    })
}

impl Drop for MlxFfiBackend {
    fn drop(&mut self) {
        let arena = std::mem::take(&mut *self.arena.lock());
        for (_h, t) in arena {
            unsafe { sys::mlxrs_free_tensor(t.as_ptr()) };
        }
        unsafe { sys::mlxrs_free_device(self.device.as_ptr()) };
    }
}

impl Backend for MlxFfiBackend {
    fn eval_node(
        &self,
        op: &OpKind,
        inputs: &[NodeInput<'_>],
        _output_meta: &mlx_core::graph::TensorMeta,
    ) -> Result<Vec<f32>> {
        match op {
            OpKind::Add => {
                let a = require_input(inputs, 0)?;
                let b = require_input(inputs, 1)?;
                self.call_binary(a, b, sys::mlxrs_add)
            }
            OpKind::Sub => Err(MlxError::InvalidArgument(
                "Sub not supported by FFI backend".into(),
            )),
            OpKind::Mul => {
                let a = require_input(inputs, 0)?;
                let b = require_input(inputs, 1)?;
                self.call_binary(a, b, sys::mlxrs_mul)
            }
            OpKind::Div => Err(MlxError::InvalidArgument(
                "Div not supported by FFI backend".into(),
            )),
            OpKind::Neg => {
                let a = require_input(inputs, 0)?;
                self.call_unary(a, sys::mlxrs_neg)
            }
            OpKind::MatMul => {
                let a = require_input(inputs, 0)?;
                let b = require_input(inputs, 1)?;
                self.call_binary(a, b, sys::mlxrs_matmul)
            }
            OpKind::Sum { axis } => {
                let a = require_input(inputs, 0)?;
                self.call_sum(a, *axis)
            }
            OpKind::Mean { .. } => Err(MlxError::InvalidArgument(
                "Mean not supported by FFI backend".into(),
            )),
            OpKind::Max { .. } => Err(MlxError::InvalidArgument(
                "Max not supported by FFI backend".into(),
            )),
            OpKind::Reshape { new_shape } => {
                let a = require_input(inputs, 0)?;
                let h = self.make_from_f32(a)?;
                let t = self
                    .arena
                    .lock()
                    .get(&h)
                    .copied()
                    .expect("tensor handle missing");
                let out_ptr = unsafe {
                    sys::mlxrs_reshape(t.as_ptr(), new_shape.0.as_ptr(), new_shape.0.len())
                };
                let out = NonNull::new(out_ptr)
                    .ok_or(MlxError::BackendUnavailable("mlxrs_reshape returned null"))?;
                let out_handle = self.track_tensor(out);
                let result = self.read_f32(out)?;
                self.free_tensor(out_handle);
                self.free_tensor(h);
                Ok(result)
            }
            OpKind::Transpose { .. } => {
                let a = require_input(inputs, 0)?;
                self.call_unary(a, sys::mlxrs_transpose)
            }
            OpKind::Softmax { axis } => {
                let a = require_input(inputs, 0)?;
                let h = self.make_from_f32(a)?;
                let t = self
                    .arena
                    .lock()
                    .get(&h)
                    .copied()
                    .expect("tensor handle missing");
                let out_ptr = unsafe { sys::mlxrs_softmax(t.as_ptr(), *axis as _) };
                let out = NonNull::new(out_ptr)
                    .ok_or(MlxError::BackendUnavailable("mlxrs_softmax returned null"))?;
                let out_handle = self.track_tensor(out);
                let result = self.read_f32(out)?;
                self.free_tensor(out_handle);
                self.free_tensor(h);
                Ok(result)
            }
            OpKind::Silu => Err(MlxError::InvalidArgument(
                "Silu not supported by FFI backend".into(),
            )),
            OpKind::Gelu => Err(MlxError::InvalidArgument(
                "Gelu not supported by FFI backend".into(),
            )),
            OpKind::LayerNorm { .. } => Err(MlxError::InvalidArgument(
                "LayerNorm not supported by FFI backend".into(),
            )),
            OpKind::RmsNorm { .. } => Err(MlxError::InvalidArgument(
                "RmsNorm not supported by FFI backend".into(),
            )),
            OpKind::Broadcast { .. }
            | OpKind::LayerNormVjp { .. }
            | OpKind::RmsNormVjp { .. }
            | OpKind::Rope { .. }
            | OpKind::RoPE { .. } => Err(MlxError::InvalidArgument(format!(
                "{op:?} not supported by FFI backend",
            ))),
            OpKind::Constant | OpKind::Parameter => Err(MlxError::InvalidArgument(
                "Constant/Parameter should be pre-materialized by Stream".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::graph::TensorMeta;
    use mlx_core::types::{DType, Shape};

    fn meta(shape: &[i64]) -> TensorMeta {
        TensorMeta {
            shape: Shape::new(shape.to_vec()),
            dtype: DType::F32,
        }
    }

    #[test]
    fn eval_add_via_ffi_backend() {
        let backend = MlxFfiBackend::new_default_device().unwrap();
        let a = NodeInput {
            data: &[1.0, 2.0, 3.0],
            shape: &Shape::new(vec![3]),
            dtype: DType::F32,
        };
        let b = NodeInput {
            data: &[4.0, 5.0, 6.0],
            shape: &Shape::new(vec![3]),
            dtype: DType::F32,
        };
        let out = backend
            .eval_node(&OpKind::Add, &[a, b], &meta(&[3]))
            .unwrap();
        mlx_conformance::assert_allclose(&out, &[5.0, 7.0, 9.0], 1e-6, 1e-6);
    }

    #[test]
    fn eval_matmul_via_ffi_backend() {
        let backend = MlxFfiBackend::new_default_device().unwrap();
        let a = NodeInput {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &Shape::new(vec![2, 2]),
            dtype: DType::F32,
        };
        let b = NodeInput {
            data: &[5.0, 6.0, 7.0, 8.0],
            shape: &Shape::new(vec![2, 2]),
            dtype: DType::F32,
        };
        let out = backend
            .eval_node(&OpKind::MatMul, &[a, b], &meta(&[2, 2]))
            .unwrap();
        mlx_conformance::assert_allclose(&out, &[19.0, 22.0, 43.0, 50.0], 1e-6, 1e-6);
    }

    #[test]
    fn eval_sum_axis_via_ffi_backend() {
        let backend = MlxFfiBackend::new_default_device().unwrap();
        let a = NodeInput {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &Shape::new(vec![2, 2]),
            dtype: DType::F32,
        };
        let out = backend
            .eval_node(&OpKind::Sum { axis: Some(0) }, &[a], &meta(&[2]))
            .unwrap();
        mlx_conformance::assert_allclose(&out, &[4.0, 6.0], 1e-6, 1e-6);
    }
}

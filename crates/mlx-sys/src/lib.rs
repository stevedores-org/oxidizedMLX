//! C ABI shim for MLX tensor operations.
//!
//! With the default `native` feature, all `mlxrs_*` functions are implemented
//! in pure Rust using the `mlx-core` tensor API. With the `cpp` feature, they
//! link against the external MLX C++ library built via cmake.

#![allow(non_camel_case_types)]

// libc types re-exported for cpp feature extern declarations.
#[cfg(feature = "cpp")]
use libc::{c_int, size_t};

// ── Opaque handle types ─────────────────────────────────────────────────

/// Opaque handle to an MLX tensor.
#[repr(C)]
pub struct mlx_tensor_t {
    _private: [u8; 0],
}

/// Opaque handle to an MLX device.
#[repr(C)]
pub struct mlx_device_t {
    _private: [u8; 0],
}

/// Data types supported by the FFI layer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum mlx_dtype_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    I64 = 4,
}

// ── C++ FFI declarations (enabled with `cpp` feature) ───────────────────

#[cfg(feature = "cpp")]
extern "C" {
    pub fn mlxrs_default_device() -> *mut mlx_device_t;
    pub fn mlxrs_zeros(
        device: *mut mlx_device_t,
        dtype: mlx_dtype_t,
        shape_ptr: *const i64,
        shape_len: size_t,
    ) -> *mut mlx_tensor_t;
    pub fn mlxrs_ones(
        device: *mut mlx_device_t,
        dtype: mlx_dtype_t,
        shape_ptr: *const i64,
        shape_len: size_t,
    ) -> *mut mlx_tensor_t;
    pub fn mlxrs_from_f32(
        device: *mut mlx_device_t,
        shape_ptr: *const i64,
        shape_len: size_t,
        data_ptr: *const f32,
        data_len: size_t,
    ) -> *mut mlx_tensor_t;
    pub fn mlxrs_add(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_mul(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_neg(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_matmul(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_sum(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t;
    pub fn mlxrs_sum_all(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_reshape(
        a: *mut mlx_tensor_t,
        shape_ptr: *const i64,
        shape_len: size_t,
    ) -> *mut mlx_tensor_t;
    pub fn mlxrs_transpose(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_softmax(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t;
    pub fn mlxrs_eval(t: *mut mlx_tensor_t);
    pub fn mlxrs_to_f32_vec(t: *mut mlx_tensor_t, out_ptr: *mut f32, out_len: size_t) -> c_int;
    pub fn mlxrs_numel(t: *mut mlx_tensor_t) -> i64;
    pub fn mlxrs_ndim(t: *mut mlx_tensor_t) -> c_int;
    pub fn mlxrs_shape(t: *mut mlx_tensor_t, out_ptr: *mut i64, out_len: size_t) -> c_int;
    pub fn mlxrs_free_tensor(t: *mut mlx_tensor_t);
    pub fn mlxrs_free_device(d: *mut mlx_device_t);
}

// ── Pure-Rust native implementation (enabled with `native` feature) ─────

#[cfg(feature = "native")]
mod native_impl;

#[cfg(feature = "native")]
pub use native_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a tensor from f32 data via C ABI.
    unsafe fn make_tensor(data: &[f32], shape: &[i64]) -> *mut mlx_tensor_t {
        unsafe {
            let dev = mlxrs_default_device();
            let t = mlxrs_from_f32(dev, shape.as_ptr(), shape.len(), data.as_ptr(), data.len());
            mlxrs_free_device(dev);
            t
        }
    }

    // Helper: read tensor data as Vec<f32> via C ABI.
    unsafe fn read_f32(t: *mut mlx_tensor_t) -> Vec<f32> {
        unsafe {
            mlxrs_eval(t);
            let n = mlxrs_numel(t) as usize;
            let mut out = vec![0.0f32; n];
            let rc = mlxrs_to_f32_vec(t, out.as_mut_ptr(), n);
            assert_eq!(rc, 0, "mlxrs_to_f32_vec failed");
            out
        }
    }

    // Helper: read tensor shape via C ABI.
    unsafe fn read_shape(t: *mut mlx_tensor_t) -> Vec<i64> {
        unsafe {
            let ndim = mlxrs_ndim(t) as usize;
            let mut out = vec![0i64; ndim];
            let rc = mlxrs_shape(t, out.as_mut_ptr(), ndim);
            assert_eq!(rc, 0, "mlxrs_shape failed");
            out
        }
    }

    // ── Creation tests ───────────────────────────────────────────────

    #[test]
    fn test_from_f32() {
        unsafe {
            let t = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            assert!(!t.is_null());
            assert_eq!(read_shape(t), vec![2, 2]);
            assert_eq!(read_f32(t), vec![1.0, 2.0, 3.0, 4.0]);
            mlxrs_free_tensor(t);
        }
    }

    #[test]
    fn test_zeros() {
        unsafe {
            let dev = mlxrs_default_device();
            let t = mlxrs_zeros(dev, mlx_dtype_t::F32, [2, 3].as_ptr(), 2);
            assert!(!t.is_null());
            assert_eq!(read_f32(t), vec![0.0; 6]);
            mlxrs_free_tensor(t);
            mlxrs_free_device(dev);
        }
    }

    #[test]
    fn test_ones() {
        unsafe {
            let dev = mlxrs_default_device();
            let t = mlxrs_ones(dev, mlx_dtype_t::F32, [3].as_ptr(), 1);
            assert!(!t.is_null());
            assert_eq!(read_f32(t), vec![1.0; 3]);
            mlxrs_free_tensor(t);
            mlxrs_free_device(dev);
        }
    }

    // ── Week-1 op tests ──────────────────────────────────────────────

    #[test]
    fn test_add() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
            let b = make_tensor(&[4.0, 5.0, 6.0], &[3]);
            let c = mlxrs_add(a, b);
            assert!(!c.is_null());
            mlx_conformance::assert_allclose(&read_f32(c), &[5.0, 7.0, 9.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
            mlxrs_free_tensor(c);
        }
    }

    #[test]
    fn test_mul() {
        unsafe {
            let a = make_tensor(&[2.0, 3.0, 4.0], &[3]);
            let b = make_tensor(&[0.5, 2.0, 0.25], &[3]);
            let c = mlxrs_mul(a, b);
            assert!(!c.is_null());
            mlx_conformance::assert_allclose(&read_f32(c), &[1.0, 6.0, 1.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
            mlxrs_free_tensor(c);
        }
    }

    #[test]
    fn test_neg() {
        unsafe {
            let a = make_tensor(&[1.0, -2.0, 3.0], &[3]);
            let b = mlxrs_neg(a);
            assert!(!b.is_null());
            mlx_conformance::assert_allclose(&read_f32(b), &[-1.0, 2.0, -3.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
        }
    }

    #[test]
    fn test_matmul() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
            let c = mlxrs_matmul(a, b);
            assert!(!c.is_null());
            assert_eq!(read_shape(c), vec![2, 2]);
            mlx_conformance::assert_allclose(&read_f32(c), &[19.0, 22.0, 43.0, 50.0], 1e-5, 1e-5);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
            mlxrs_free_tensor(c);
        }
    }

    #[test]
    fn test_sum_axis() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let s = mlxrs_sum(a, 0);
            assert!(!s.is_null());
            assert_eq!(read_shape(s), vec![3]);
            mlx_conformance::assert_allclose(&read_f32(s), &[5.0, 7.0, 9.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(s);
        }
    }

    #[test]
    fn test_sum_all() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
            let s = mlxrs_sum_all(a);
            assert!(!s.is_null());
            mlx_conformance::assert_allclose(&read_f32(s), &[10.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(s);
        }
    }

    #[test]
    fn test_reshape() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let b = mlxrs_reshape(a, [3, 2].as_ptr(), 2);
            assert!(!b.is_null());
            assert_eq!(read_shape(b), vec![3, 2]);
            mlx_conformance::assert_allclose(
                &read_f32(b),
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                1e-6,
                1e-6,
            );
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
        }
    }

    #[test]
    fn test_transpose() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let b = mlxrs_transpose(a);
            assert!(!b.is_null());
            assert_eq!(read_shape(b), vec![3, 2]);
            mlx_conformance::assert_allclose(
                &read_f32(b),
                &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                1e-6,
                1e-6,
            );
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
        }
    }

    #[test]
    fn test_softmax() {
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
            let s = mlxrs_softmax(a, 0);
            assert!(!s.is_null());
            let result = read_f32(s);
            let sum: f32 = result.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
            assert!(result[0] < result[1]);
            assert!(result[1] < result[2]);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(s);
        }
    }

    #[test]
    fn test_lazy_chain() {
        // (a + b) * b via C ABI — tests lazy evaluation through shim
        unsafe {
            let a = make_tensor(&[1.0, 2.0, 3.0], &[3]);
            let b = make_tensor(&[2.0, 2.0, 2.0], &[3]);
            let sum = mlxrs_add(a, b);
            let product = mlxrs_mul(sum, b);
            assert!(!product.is_null());
            mlx_conformance::assert_allclose(&read_f32(product), &[6.0, 8.0, 10.0], 1e-6, 1e-6);
            mlxrs_free_tensor(a);
            mlxrs_free_tensor(b);
            mlxrs_free_tensor(sum);
            mlxrs_free_tensor(product);
        }
    }
}

//! Unsafe FFI bindings to the MLX C ABI shim.
//!
//! This crate provides raw, opaque-handle-based access to MLX operations.
//! It is not intended for direct use — use `mlx-core` for a safe API.

#![allow(non_camel_case_types)]

use libc::{c_int, size_t};

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

extern "C" {
    // ── Device ──────────────────────────────────────────────────────────
    pub fn mlxrs_default_device() -> *mut mlx_device_t;

    // ── Tensor creation ─────────────────────────────────────────────────
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

    // ── Elementwise ops ─────────────────────────────────────────────────
    pub fn mlxrs_add(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_mul(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;
    pub fn mlxrs_neg(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;

    // ── Linear algebra ──────────────────────────────────────────────────
    pub fn mlxrs_matmul(a: *mut mlx_tensor_t, b: *mut mlx_tensor_t) -> *mut mlx_tensor_t;

    // ── Reductions ──────────────────────────────────────────────────────
    pub fn mlxrs_sum(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t;
    pub fn mlxrs_sum_all(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;

    // ── Shape manipulation ──────────────────────────────────────────────
    pub fn mlxrs_reshape(
        a: *mut mlx_tensor_t,
        shape_ptr: *const i64,
        shape_len: size_t,
    ) -> *mut mlx_tensor_t;
    pub fn mlxrs_transpose(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t;

    // ── Activation functions ────────────────────────────────────────────
    pub fn mlxrs_softmax(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t;

    // ── Materialization ─────────────────────────────────────────────────
    pub fn mlxrs_eval(t: *mut mlx_tensor_t);
    pub fn mlxrs_to_f32_vec(t: *mut mlx_tensor_t, out_ptr: *mut f32, out_len: size_t) -> c_int;
    pub fn mlxrs_numel(t: *mut mlx_tensor_t) -> i64;
    pub fn mlxrs_ndim(t: *mut mlx_tensor_t) -> c_int;
    pub fn mlxrs_shape(t: *mut mlx_tensor_t, out_ptr: *mut i64, out_len: size_t) -> c_int;

    // ── Lifecycle ───────────────────────────────────────────────────────
    pub fn mlxrs_free_tensor(t: *mut mlx_tensor_t);
    pub fn mlxrs_free_device(d: *mut mlx_device_t);
}

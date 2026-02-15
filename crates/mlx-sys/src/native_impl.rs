//! Pure-Rust implementation of the `mlxrs_*` C ABI functions.
//!
//! Each function delegates to the `mlx-core` Tensor API, using opaque pointers
//! (Box<Tensor>/Box<Device>) cast through the zero-sized marker types.
//!
//! # Safety
//!
//! All functions in this module follow C ABI conventions: callers must pass
//! valid, non-null pointers obtained from other `mlxrs_*` functions. Handles
//! must be freed exactly once via `mlxrs_free_tensor` / `mlxrs_free_device`.

#![allow(clippy::missing_safety_doc)]

use libc::{c_int, size_t};

use mlx_core::tensor::Device;
use mlx_core::{DType, Shape, Tensor};

use crate::{mlx_device_t, mlx_dtype_t, mlx_tensor_t};

// ── Pointer conversion helpers ──────────────────────────────────────────

fn box_tensor(t: Tensor) -> *mut mlx_tensor_t {
    Box::into_raw(Box::new(t)) as *mut mlx_tensor_t
}

/// # Safety
///
/// `p` must be a valid pointer obtained from `box_tensor` that has not been freed.
/// The returned reference is valid only as long as `p` is live — callers must not
/// use the reference after calling `mlxrs_free_tensor(p)`.
unsafe fn ref_tensor<'a>(p: *mut mlx_tensor_t) -> &'a Tensor {
    unsafe { &*(p as *const Tensor) }
}

fn box_device(d: Device) -> *mut mlx_device_t {
    Box::into_raw(Box::new(d)) as *mut mlx_device_t
}

/// # Safety
///
/// `p` must be a valid pointer obtained from `box_device` that has not been freed.
/// The returned reference is valid only as long as `p` is live — callers must not
/// use the reference after calling `mlxrs_free_device(p)`.
unsafe fn ref_device<'a>(p: *mut mlx_device_t) -> &'a Device {
    unsafe { &*(p as *const Device) }
}

/// Build a slice from a C pointer+length.
///
/// Returns `Some(&[])` when `len == 0` (even if `ptr` is null). Returns `None`
/// when `len > 0` and `ptr` is null.
unsafe fn safe_slice<'a, T>(ptr: *const T, len: size_t) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn convert_dtype(dt: mlx_dtype_t) -> DType {
    match dt {
        mlx_dtype_t::F32 => DType::F32,
        mlx_dtype_t::F16 => DType::F16,
        mlx_dtype_t::BF16 => DType::BF16,
        mlx_dtype_t::I32 => DType::I32,
        mlx_dtype_t::I64 => DType::I64,
    }
}

// ── Device ──────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_default_device() -> *mut mlx_device_t {
    box_device(Device::default_device())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_device_type(d: *mut mlx_device_t) -> crate::mlx_device_type_t {
    let dev = unsafe { ref_device(d) };
    match dev {
        Device::Cpu => crate::mlx_device_type_t::CPU,
        Device::Gpu => crate::mlx_device_type_t::GPU,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_cpu_device() -> *mut mlx_device_t {
    box_device(Device::Cpu)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_gpu_device() -> *mut mlx_device_t {
    box_device(Device::Gpu)
}

// ── Tensor creation ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_zeros(
    device: *mut mlx_device_t,
    dtype: mlx_dtype_t,
    shape_ptr: *const i64,
    shape_len: size_t,
) -> *mut mlx_tensor_t {
    if device.is_null() {
        return std::ptr::null_mut();
    }
    let dev = unsafe { ref_device(device) };
    let dims = match unsafe { safe_slice(shape_ptr, shape_len) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let shape = Shape::new(dims.to_vec());
    match Tensor::zeros(&shape, convert_dtype(dtype), dev) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_ones(
    device: *mut mlx_device_t,
    dtype: mlx_dtype_t,
    shape_ptr: *const i64,
    shape_len: size_t,
) -> *mut mlx_tensor_t {
    if device.is_null() {
        return std::ptr::null_mut();
    }
    let dev = unsafe { ref_device(device) };
    let dims = match unsafe { safe_slice(shape_ptr, shape_len) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let shape = Shape::new(dims.to_vec());
    match Tensor::ones(&shape, convert_dtype(dtype), dev) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_from_f32(
    device: *mut mlx_device_t,
    shape_ptr: *const i64,
    shape_len: size_t,
    data_ptr: *const f32,
    data_len: size_t,
) -> *mut mlx_tensor_t {
    if device.is_null() {
        return std::ptr::null_mut();
    }
    let dev = unsafe { ref_device(device) };
    let dims = match unsafe { safe_slice(shape_ptr, shape_len) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let data = match unsafe { safe_slice(data_ptr, data_len) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let shape = Shape::new(dims.to_vec());
    match Tensor::from_f32(data, &shape, dev) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

// ── Elementwise ops ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_add(
    a: *mut mlx_tensor_t,
    b: *mut mlx_tensor_t,
) -> *mut mlx_tensor_t {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (a, b) = unsafe { (ref_tensor(a), ref_tensor(b)) };
    match a.add(b) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_mul(
    a: *mut mlx_tensor_t,
    b: *mut mlx_tensor_t,
) -> *mut mlx_tensor_t {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (a, b) = unsafe { (ref_tensor(a), ref_tensor(b)) };
    match a.mul(b) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_neg(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    box_tensor(a.neg())
}

// ── Linear algebra ──────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_matmul(
    a: *mut mlx_tensor_t,
    b: *mut mlx_tensor_t,
) -> *mut mlx_tensor_t {
    if a.is_null() || b.is_null() {
        return std::ptr::null_mut();
    }
    let (a, b) = unsafe { (ref_tensor(a), ref_tensor(b)) };
    match a.matmul(b) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

// ── Reductions ──────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_sum(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    match a.sum_axis(axis) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_sum_all(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    match a.sum_all() {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

// ── Shape manipulation ──────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_reshape(
    a: *mut mlx_tensor_t,
    shape_ptr: *const i64,
    shape_len: size_t,
) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    let dims = match unsafe { safe_slice(shape_ptr, shape_len) } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let new_shape = Shape::new(dims.to_vec());
    match a.reshape(&new_shape) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_transpose(a: *mut mlx_tensor_t) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    match a.transpose(None) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

// ── Activation functions ────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_softmax(a: *mut mlx_tensor_t, axis: c_int) -> *mut mlx_tensor_t {
    if a.is_null() {
        return std::ptr::null_mut();
    }
    let a = unsafe { ref_tensor(a) };
    match a.softmax(axis) {
        Ok(t) => box_tensor(t),
        Err(_) => std::ptr::null_mut(),
    }
}

// ── Materialization ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_eval(t: *mut mlx_tensor_t) {
    if t.is_null() {
        return;
    }
    let t = unsafe { ref_tensor(t) };
    let _ = t.eval();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_to_f32_vec(
    t: *mut mlx_tensor_t,
    out_ptr: *mut f32,
    out_len: size_t,
) -> c_int {
    if t.is_null() {
        return -1;
    }
    let t = unsafe { ref_tensor(t) };
    match t.to_vec_f32() {
        Ok(data) => {
            let required_len = data.len();
            if required_len > (c_int::MAX as usize) {
                return -3;
            }
            if out_len == 0 {
                return required_len as c_int;
            }
            if out_len < required_len {
                return required_len as c_int; // buffer too small, return required length
            }
            if required_len > 0 && out_ptr.is_null() {
                return -1;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), out_ptr, required_len);
            }
            0 // success
        }
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_numel(t: *mut mlx_tensor_t) -> i64 {
    if t.is_null() {
        return -1;
    }
    let t = unsafe { ref_tensor(t) };
    t.numel()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_ndim(t: *mut mlx_tensor_t) -> c_int {
    if t.is_null() {
        return -1;
    }
    let t = unsafe { ref_tensor(t) };
    t.shape().ndim() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_shape(
    t: *mut mlx_tensor_t,
    out_ptr: *mut i64,
    out_len: size_t,
) -> c_int {
    if t.is_null() {
        return -1;
    }
    let t = unsafe { ref_tensor(t) };
    let dims = &t.shape().0;
    let needed = dims.len();
    if needed > (c_int::MAX as usize) {
        return -3;
    }
    if out_len == 0 {
        return needed as c_int;
    }
    let copy_len = needed.min(out_len);
    if copy_len > 0 && out_ptr.is_null() {
        return -1;
    }
    unsafe { std::ptr::copy_nonoverlapping(dims.as_ptr(), out_ptr, copy_len) };
    if out_len < needed {
        needed as c_int // signal that buffer was too small
    } else {
        0
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_free_tensor(t: *mut mlx_tensor_t) {
    if !t.is_null() {
        unsafe {
            drop(Box::from_raw(t as *mut Tensor));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mlxrs_free_device(d: *mut mlx_device_t) {
    if !d.is_null() {
        unsafe {
            drop(Box::from_raw(d as *mut Device));
        }
    }
}

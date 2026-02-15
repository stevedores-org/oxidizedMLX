//! Shared buffer allocation and mapping helpers.

use mlx_core::{MlxError, Result};
use metal::{Buffer, MTLResourceOptions};
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use crate::context::MetalContext;

/// Typed Metal buffer wrapper.
#[derive(Clone)]
pub struct MetalBuffer<T> {
    raw: Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> MetalBuffer<T> {
    /// Create a shared buffer from a host slice.
    pub fn from_slice_shared(ctx: &MetalContext, data: &[T]) -> Result<Self> {
        let len = data.len();
        let byte_len = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| MlxError::InvalidArgument("buffer size overflow".to_string()))?;

        let raw = ctx
            .device()
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

        unsafe {
            let dst = raw.contents() as *mut T;
            if dst.is_null() {
                return Err(MlxError::InvalidArgument(
                    "Metal buffer contents pointer was null".to_string(),
                ));
            }
            ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
        }

        Ok(Self {
            raw,
            len,
            _marker: PhantomData,
        })
    }

    pub(crate) fn new_shared_uninitialized(ctx: &MetalContext, len: usize) -> Result<Self> {
        let byte_len = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| MlxError::InvalidArgument("buffer size overflow".to_string()))?;

        let raw = ctx
            .device()
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            raw,
            len,
            _marker: PhantomData,
        })
    }

    pub(crate) fn raw(&self) -> &Buffer {
        &self.raw
    }

    /// Read the buffer contents into a Vec.
    pub fn read_to_vec(&self) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }
        unsafe {
            let src = self.raw.contents() as *const T;
            if src.is_null() {
                return Vec::new();
            }
            let slice = std::slice::from_raw_parts(src, self.len);
            slice.to_vec()
        }
    }
}

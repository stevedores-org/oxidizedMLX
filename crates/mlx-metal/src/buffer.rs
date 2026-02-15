use crate::{MetalError, Result};
use metal::*;
use std::{marker::PhantomData, mem, ptr};

pub struct MetalBuffer<T: Copy> {
    raw: Buffer,
    len: usize,
    _pd: PhantomData<T>,
}

impl<T: Copy> MetalBuffer<T> {
    pub fn new_shared(ctx: &crate::MetalContext, len: usize) -> Result<Self> {
        let byte_len = len
            .checked_mul(mem::size_of::<T>())
            .ok_or(MetalError::Invalid("buffer size overflow"))?;
        let opts = MTLResourceOptions::StorageModeShared;
        let raw = ctx.device().new_buffer(byte_len as u64, opts);
        Ok(Self {
            raw,
            len,
            _pd: PhantomData,
        })
    }

    pub fn from_slice_shared(ctx: &crate::MetalContext, data: &[T]) -> Result<Self> {
        let mut b = Self::new_shared(ctx, data.len())?;
        b.write_from(data)?;
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn raw(&self) -> &Buffer {
        &self.raw
    }

    pub fn write_from(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(MetalError::Invalid("write len mismatch"));
        }
        unsafe {
            let dst = self.raw.contents() as *mut T;
            ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    pub fn read_to_vec(&self) -> Vec<T> {
        let mut out = vec![unsafe { mem::zeroed() }; self.len];
        unsafe {
            let src = self.raw.contents() as *const T;
            ptr::copy_nonoverlapping(src, out.as_mut_ptr(), self.len);
        }
        out
    }
}

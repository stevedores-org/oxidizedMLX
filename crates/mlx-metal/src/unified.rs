use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use metal::{Buffer as MtlBuffer, Device, MTLResourceOptions};

use crate::instrument::BufferTelemetry;
use crate::{MetalError, Result};

/// Returns the VM page size for this platform.
fn page_size() -> usize {
    // SAFETY: reading a constant from libc
    unsafe { libc::vm_page_size }
}

/// Rounds `n` up to the nearest multiple of `align`.
fn round_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// HostAllocation<T>
// ---------------------------------------------------------------------------

/// Page-aligned heap allocation suitable for Metal no-copy buffers.
///
/// The allocation is rounded up to the nearest page boundary so that
/// `Device::new_buffer_with_bytes_no_copy` can map it directly.
pub struct HostAllocation<T: Copy> {
    ptr: NonNull<T>,
    /// Number of logical `T` elements.
    len: usize,
    /// Layout used for alloc/dealloc (page-aligned, page-rounded size).
    layout: Layout,
}

// SAFETY: The allocation is exclusively owned; no interior mutability.
unsafe impl<T: Copy + Send> Send for HostAllocation<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for HostAllocation<T> {}

impl<T: Copy> HostAllocation<T> {
    /// Allocate space for `len` elements of `T`, page-aligned.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(MetalError::ZeroLength);
        }

        let byte_len = len * std::mem::size_of::<T>();
        let ps = page_size();
        let rounded = round_up(byte_len, ps);

        let layout =
            Layout::from_size_align(rounded, ps).map_err(|e| MetalError::BufferCreationFailed(e.to_string()))?;

        // SAFETY: layout has non-zero size (len > 0 and T is not ZST in practice).
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw as *mut T).ok_or_else(|| {
            MetalError::BufferCreationFailed("allocation returned null".into())
        })?;

        Ok(Self { ptr, len, layout })
    }

    /// Allocate and copy data from `slice`.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let alloc = Self::new(data.len())?;
        // SAFETY: alloc has room for at least `data.len()` elements,
        // and both regions are valid, non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), alloc.ptr.as_ptr(), data.len());
        }
        Ok(alloc)
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: we own `len` initialised elements starting at `ptr`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Logical byte length (`len * size_of::<T>()`).
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// The full layout (page-rounded size, page-aligned).
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl<T: Copy> Drop for HostAllocation<T> {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this exact layout.
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// ---------------------------------------------------------------------------
// UnifiedBuffer<T>
// ---------------------------------------------------------------------------

/// Tracks how a `UnifiedBuffer` was created.
enum BufferOrigin<T: Copy> {
    /// Created via `new_buffer_with_bytes_no_copy`; we own the host memory.
    NoCopy { host: HostAllocation<T> },
    /// Created via `new_buffer` (Metal owns the storage).
    Shared,
}

/// A Metal buffer backed by Apple Silicon unified memory.
///
/// For `NoCopy` buffers the Rust-side `HostAllocation` and the GPU-visible
/// `MTLBuffer::contents()` pointer are the *same physical address*, proving
/// true zero-copy UMA behaviour.
pub struct UnifiedBuffer<T: Copy + Send + Sync> {
    // Declared before `origin` so it drops first (releases MTLBuffer before
    // freeing the host allocation).
    mtl_buffer: MtlBuffer,
    origin: BufferOrigin<T>,
    len: usize,
    gpu_in_flight: AtomicBool,
}

impl<T: Copy + Send + Sync> UnifiedBuffer<T> {
    /// Create a no-copy buffer backed by a pre-existing `HostAllocation`.
    ///
    /// Metal maps the same physical pages — no `memcpy` occurs.
    pub(crate) fn from_host_no_copy(
        device: &Device,
        host: HostAllocation<T>,
    ) -> Result<Self> {
        let len = host.len();
        let options = MTLResourceOptions::StorageModeShared;

        // Use layout.size() (page-rounded) so Metal sees a page-multiple length.
        let mtl_buffer = device.new_buffer_with_bytes_no_copy(
            host.as_ptr() as *const std::ffi::c_void,
            host.layout().size() as u64,
            options,
            None,
        );

        Ok(Self {
            mtl_buffer,
            origin: BufferOrigin::NoCopy { host },
            len,
            gpu_in_flight: AtomicBool::new(false),
        })
    }

    /// Create a shared buffer of `len` elements, owned by Metal.
    pub(crate) fn shared_uninitialized(device: &Device, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(MetalError::ZeroLength);
        }
        let byte_len = (len * std::mem::size_of::<T>()) as u64;
        let options = MTLResourceOptions::StorageModeShared;
        let mtl_buffer = device.new_buffer(byte_len, options);
        Ok(Self {
            mtl_buffer,
            origin: BufferOrigin::Shared,
            len,
            gpu_in_flight: AtomicBool::new(false),
        })
    }

    /// Read the buffer contents as a typed slice (CPU-side).
    pub fn as_host_slice(&self) -> &[T] {
        let ptr = self.mtl_buffer.contents() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Mutable access to buffer contents. Panics if the GPU is in flight.
    pub fn as_host_slice_mut(&mut self) -> &mut [T] {
        assert!(
            !self.gpu_in_flight.load(Ordering::Acquire),
            "cannot mutably access buffer while GPU command is in flight"
        );
        let ptr = self.mtl_buffer.contents() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Borrow the underlying `MTLBuffer` for GPU dispatch.
    pub fn mtl_buffer(&self) -> &MtlBuffer {
        &self.mtl_buffer
    }

    /// Mark the buffer as in-flight (GPU is using it).
    pub fn set_gpu_in_flight(&self, in_flight: bool) {
        self.gpu_in_flight.store(in_flight, Ordering::Release);
    }

    /// Snapshot telemetry for this buffer.
    pub fn telemetry(&self) -> BufferTelemetry {
        let gpu_contents_ptr = self.mtl_buffer.contents() as usize;
        match &self.origin {
            BufferOrigin::NoCopy { host } => BufferTelemetry {
                host_ptr: host.as_ptr() as usize,
                gpu_contents_ptr,
                bytes_copied: 0,
                byte_length: host.byte_len(),
                is_no_copy: true,
            },
            BufferOrigin::Shared => BufferTelemetry {
                host_ptr: gpu_contents_ptr,
                gpu_contents_ptr,
                bytes_copied: 0,
                byte_length: self.len * std::mem::size_of::<T>(),
                is_no_copy: false,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use crate::MetalContext;

    #[test]
    fn unified_no_copy_preserves_pointer_identity() {
        let ctx = MetalContext::new().expect("need Metal device");
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let host = HostAllocation::from_slice(&data).unwrap();
        let host_addr = host.as_ptr() as usize;

        let buf = ctx.buffer_from_host_no_copy(host).unwrap();
        let telem = buf.telemetry();

        assert_eq!(telem.host_ptr, host_addr);
        telem.assert_zero_copy();

        // Verify data roundtrip
        let readback = buf.as_host_slice();
        assert_eq!(readback, &data[..]);
    }

    #[test]
    fn shared_output_readback_no_memcpy() {
        let ctx = MetalContext::new().expect("need Metal device");
        let buf: UnifiedBuffer<f32> = ctx.buffer_shared_uninitialized(256).unwrap();

        // Write via CPU
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buf.mtl_buffer().contents() as *mut f32, 256)
        };
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32 * 2.0;
        }

        // Read back — same pointer, zero copy
        let telem = buf.telemetry();
        assert_eq!(telem.bytes_copied, 0);

        let readback = buf.as_host_slice();
        assert_eq!(readback[0], 0.0);
        assert_eq!(readback[128], 256.0);
    }

    #[test]
    fn host_allocation_page_aligned() {
        let alloc: HostAllocation<u8> = HostAllocation::new(100).unwrap();
        let ps = page_size();
        assert_eq!(alloc.as_ptr() as usize % ps, 0);
        assert_eq!(alloc.layout().size() % ps, 0);
    }

    #[test]
    fn leak_test_create_destroy_cycles() {
        let ctx = MetalContext::new().expect("need Metal device");
        for _ in 0..10_000 {
            let host = HostAllocation::<f32>::new(64).unwrap();
            let _buf = ctx.buffer_from_host_no_copy(host).unwrap();
        }
        // If we get here without OOM, the test passes.
    }

    #[test]
    #[should_panic(expected = "cannot mutably access buffer while GPU command is in flight")]
    fn mutable_access_panics_during_gpu_flight() {
        let ctx = MetalContext::new().expect("need Metal device");
        let host = HostAllocation::<f32>::new(16).unwrap();
        let mut buf = ctx.buffer_from_host_no_copy(host).unwrap();
        buf.set_gpu_in_flight(true);
        let _slice = buf.as_host_slice_mut(); // should panic
    }

    #[test]
    fn zero_length_rejected() {
        assert!(HostAllocation::<f32>::new(0).is_err());

        let ctx = MetalContext::new().expect("need Metal device");
        assert!(ctx.buffer_shared_uninitialized::<f32>(0).is_err());
    }
}

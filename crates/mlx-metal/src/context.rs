use metal::{CommandQueue, Device};

use crate::unified::{HostAllocation, UnifiedBuffer};
use crate::{MetalError, Result};

/// Owns a Metal device and command queue, and provides factory methods
/// for creating unified memory buffers.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
}

impl MetalContext {
    /// Create a context using the system default Metal device.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    /// Create a zero-copy buffer from a page-aligned host allocation.
    pub fn buffer_from_host_no_copy<T: Copy + Send + Sync>(
        &self,
        host: HostAllocation<T>,
    ) -> Result<UnifiedBuffer<T>> {
        UnifiedBuffer::from_host_no_copy(&self.device, host)
    }

    /// Create a Metal-owned shared buffer with `len` uninitialised elements.
    pub fn buffer_shared_uninitialized<T: Copy + Send + Sync>(
        &self,
        len: usize,
    ) -> Result<UnifiedBuffer<T>> {
        UnifiedBuffer::shared_uninitialized(&self.device, len)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
}

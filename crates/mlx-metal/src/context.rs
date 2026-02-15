//! Metal device and command queue context.

use metal::{CommandQueue, Device};
use std::sync::Arc;
use tracing::info;

use crate::command::{run_add_u32_impl, run_softmax_f32_impl};
use crate::pipeline::PipelineCache;
use crate::unified::{HostAllocation, UnifiedBuffer};
use crate::{MetalError, Result};

/// Shared Metal context: device + command queue.
#[derive(Clone)]
pub struct MetalContext {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: Arc<PipelineCache>,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        info!(device = %device.name(), "Initialized Metal context");
        Ok(Self {
            device,
            queue,
            pipelines: Arc::new(PipelineCache::new()),
        })
    }

    /// Human-readable device name.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Run the trivial add_u32 kernel for smoke testing.
    pub fn run_add_u32(&self, a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
        run_add_u32_impl(self, a, b)
    }

    /// Run the row-wise softmax kernel.
    pub fn run_softmax_f32(&self, input: &[f32], row_size: usize) -> Result<Vec<f32>> {
        run_softmax_f32_impl(self, input, row_size)
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

    pub(crate) fn pipelines(&self) -> &PipelineCache {
        &self.pipelines
    }
}

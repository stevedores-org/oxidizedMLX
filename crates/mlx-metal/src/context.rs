//! Metal device and command queue context.

use metal::{CommandQueue, Device};
use mlx_core::{MlxError, Result};
use std::sync::Arc;
use tracing::info;

use crate::command::run_add_u32_impl;
use crate::pipeline::PipelineCache;

/// Shared Metal context: device + command queue.
#[derive(Clone)]
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipelines: Arc<PipelineCache>,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or(MlxError::BackendUnavailable("Metal device not available"))?;
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

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub(crate) fn pipelines(&self) -> &PipelineCache {
        &self.pipelines
    }
}

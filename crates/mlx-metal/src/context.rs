use crate::{MetalError, Result};
use metal::*;
use parking_lot::Mutex;
use std::collections::HashMap;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MetalError::Metal("no system default device"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn pipeline_from_msl(
        &self,
        cache_key: &str,
        msl_source: &str,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        if let Some(p) = self.pipelines.lock().get(cache_key).cloned() {
            return Ok(p);
        }

        let options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(msl_source, &options)
            .map_err(|e| MetalError::Compile(format!("{e:?}")))?;

        let func = library
            .get_function(function_name, None)
            .map_err(|_| MetalError::Metal("get_function failed"))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|_| MetalError::Metal("new_compute_pipeline_state failed"))?;

        self.pipelines
            .lock()
            .insert(cache_key.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    pub fn submit_and_wait(
        &self,
        encode: impl FnOnce(&CommandBufferRef) -> Result<()>,
    ) -> Result<()> {
        let cmd_buf = self.queue.new_command_buffer();
        encode(cmd_buf)?;
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        Ok(())
    }
}

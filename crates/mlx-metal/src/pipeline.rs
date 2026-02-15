//! Metal library compilation and compute pipeline cache.

use mlx_core::{MlxError, Result};
use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::Mutex;

const ADD_U32_NAME: &str = "add_u32";

fn add_u32_source() -> &'static str {
    include_str!("kernels/add_u32.metal")
}

pub(crate) struct PipelineCache {
    cache: Mutex<HashMap<String, ComputePipelineState>>,
}

impl PipelineCache {
    pub(crate) fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_add_u32(&self, device: &Device) -> Result<ComputePipelineState> {
        self.get_or_build(device, ADD_U32_NAME, add_u32_source())
    }

    fn get_or_build(
        &self,
        device: &Device,
        key: &str,
        source: &str,
    ) -> Result<ComputePipelineState> {
        if let Some(pipeline) = self.cache.lock().expect("pipeline cache lock").get(key) {
            return Ok(pipeline.to_owned());
        }

        let library = compile_library(device, source)?;
        let function = library
            .get_function(key, None)
            .map_err(|e| MlxError::InvalidArgument(format!("failed to get function {key}: {e}")))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                MlxError::InvalidArgument(format!("failed to build pipeline {key}: {e}"))
            })?;

        self.cache
            .lock()
            .expect("pipeline cache lock")
            .insert(key.to_string(), pipeline.to_owned());

        Ok(pipeline)
    }
}

fn compile_library(device: &Device, source: &str) -> Result<Library> {
    let options = CompileOptions::new();
    device
        .new_library_with_source(source, &options)
        .map_err(|e| MlxError::InvalidArgument(format!("failed to compile Metal library: {e}")))
}

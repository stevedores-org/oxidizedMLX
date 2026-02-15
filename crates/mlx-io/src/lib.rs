//! Tensor I/O: safetensors loading, GGUF support, mmap-based weight files.

use mlx_core::{Device, MlxError, Result, Shape, Tensor};
use safetensors::tensor::{SafeTensors, serialize};
use std::collections::HashMap;
use std::path::Path;

/// Load tensors from a .safetensors file.
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path).map_err(|e| MlxError::Io(e.to_string()))?;

    // SAFETY: We assume the file is not modified by other processes while mapped.
    // This is the standard pattern for efficient tensor loading in MLX/Safetensors.
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| MlxError::Io(e.to_string()))?
    };

    let tensors =
        SafeTensors::deserialize(&mmap).map_err(|e| MlxError::Serialization(e.to_string()))?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = Shape::new(view.shape().iter().map(|&d| d as i64).collect::<Vec<_>>());
        let data = view.data();

        let f32_data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => {
                // SAFETY: chunks_exact(4) and try_into().unwrap() are safe because
                // the Safetensors view guarantees data length is a multiple of dtype size.
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            }
            _ => {
                return Err(MlxError::Serialization(format!(
                    "Unsupported safetensors dtype: {:?} for tensor {}",
                    view.dtype(),
                    name
                )));
            }
        };

        let tensor = Tensor::from_vec(f32_data, &shape, device)?;
        result.insert(name, tensor);
    }
    Ok(result)
}

/// Save tensors to a .safetensors file.
pub fn save_safetensors<P: AsRef<Path>>(path: P, tensors: &HashMap<String, Tensor>) -> Result<()> {
    // Phase 1: Materialize all tensor data into owned buffers.
    // We do this first because Tensor::to_vec_f32() might trigger evaluation,
    // and we need stable references to the byte buffers when creating Safetensors views.
    let mut data_map = HashMap::new();
    for (name, tensor) in tensors {
        let data = tensor.to_vec_f32()?;
        let shape = tensor
            .shape()
            .0
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        data_map.insert(name.clone(), (shape, bytes));
    }

    // Phase 2: Create views and serialize.
    let mut views = HashMap::new();
    for (name, (shape, bytes)) in &data_map {
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .map_err(|e| MlxError::Serialization(e.to_string()))?;
        views.insert(name.as_str(), view);
    }

    let bytes = serialize(views, &None).map_err(|e| MlxError::Serialization(e.to_string()))?;

    std::fs::write(path, bytes).map_err(|e| MlxError::Io(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::{Device, Shape, Tensor};
    use tempfile::tempdir;

    #[test]
    fn test_save_load_safetensors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        let device = Device::Cpu;

        let mut tensors = HashMap::new();
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &device).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0], &Shape::new(vec![2]), &device).unwrap();
        tensors.insert("a".to_string(), a.clone());
        tensors.insert("b".to_string(), b.clone());

        save_safetensors(&path, &tensors).unwrap();

        let loaded = load_safetensors(&path, &device).unwrap();
        assert_eq!(loaded.len(), 2);

        let a_loaded = loaded.get("a").unwrap();
        assert_eq!(a_loaded.shape(), a.shape());
        assert_eq!(a_loaded.to_vec_f32().unwrap(), a.to_vec_f32().unwrap());

        let b_loaded = loaded.get("b").unwrap();
        assert_eq!(b_loaded.shape(), b.shape());
        assert_eq!(b_loaded.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    }
}

//! Tensor I/O: safetensors loading/saving, mmap-based weight files.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use mlx_core::{DType, Device, MlxError, Result, Shape, Tensor};

/// Load all tensors from a safetensors file into memory.
pub fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = fs::read(path)?;
    let st = SafeTensors::deserialize(&data)
        .map_err(|e| MlxError::InvalidArgument(format!("safetensors parse error: {e}")))?;
    deserialize_tensors(&st, device)
}

/// Load all tensors from a safetensors file using memory-mapped I/O.
pub fn load_safetensors_mmap(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let file = fs::File::open(path)?;
    // SAFETY: The file must not be modified while the mmap is alive.
    // This is the standard usage pattern for read-only model weight files.
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| MlxError::InvalidArgument(format!("safetensors parse error: {e}")))?;
    deserialize_tensors(&st, device)
}

/// Save tensors to a safetensors file.
pub fn save_safetensors(path: &Path, tensors: &HashMap<String, Tensor>) -> Result<()> {
    let mut data_map: Vec<(String, Vec<u8>, safetensors::Dtype, Vec<usize>)> =
        Vec::with_capacity(tensors.len());

    for (name, tensor) in tensors {
        let f32_data = tensor.to_vec_f32()?;
        let st_dtype = map_dtype_reverse(tensor.dtype());
        let shape: Vec<usize> = tensor.shape().0.iter().map(|&d| d as usize).collect();
        let bytes = f32_to_bytes(&f32_data, tensor.dtype());
        data_map.push((name.clone(), bytes, st_dtype, shape));
    }

    let views: Vec<(&str, TensorView<'_>)> = data_map
        .iter()
        .map(|(name, bytes, dtype, shape)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();

    safetensors::serialize_to_file(views, &None, path)
        .map_err(|e| MlxError::InvalidArgument(format!("safetensors save error: {e}")))?;

    Ok(())
}

fn deserialize_tensors(st: &SafeTensors<'_>, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut result = HashMap::new();
    for (name, view) in st.tensors() {
        let dtype = map_dtype(view.dtype())?;
        let shape = Shape::new(view.shape().iter().map(|&d| d as i64).collect::<Vec<_>>());
        let f32_data = convert_to_f32(view.dtype(), view.data())?;
        let tensor = Tensor::from_data_with_dtype(f32_data, &shape, dtype, device)?;
        result.insert(name, tensor);
    }
    Ok(result)
}

fn convert_to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::BF16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::I32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::I64 => {
            let floats: Vec<f32> = data
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect();
            Ok(floats)
        }
        other => Err(MlxError::InvalidArgument(format!(
            "unsupported safetensors dtype: {other:?}"
        ))),
    }
}

fn map_dtype(st_dtype: safetensors::Dtype) -> Result<DType> {
    match st_dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        other => Err(MlxError::InvalidArgument(format!(
            "unsupported safetensors dtype: {other:?}"
        ))),
    }
}

fn map_dtype_reverse(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::F32 => safetensors::Dtype::F32,
        DType::F16 => safetensors::Dtype::F16,
        DType::BF16 => safetensors::Dtype::BF16,
        DType::I32 => safetensors::Dtype::I32,
        DType::I64 => safetensors::Dtype::I64,
    }
}

fn f32_to_bytes(data: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F16 => data
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect(),
        DType::BF16 => data
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect(),
        DType::I32 => data
            .iter()
            .flat_map(|v| (*v as i32).to_le_bytes())
            .collect(),
        DType::I64 => data
            .iter()
            .flat_map(|v| (*v as i64).to_le_bytes())
            .collect(),
    }
}

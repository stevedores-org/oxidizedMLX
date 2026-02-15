use std::collections::HashMap;
use std::path::Path;

use mlx_core::{DType, Device, Shape, Tensor};
use mlx_io::{load_safetensors, load_safetensors_mmap, save_safetensors};

fn cpu() -> Device {
    Device::Cpu
}

#[test]
fn roundtrip_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.safetensors");

    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_f32(&data, &Shape::new(vec![2, 3]), &cpu()).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), t);

    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors(&path, &cpu()).unwrap();

    assert_eq!(loaded.len(), 1);
    let w = &loaded["weight"];
    assert_eq!(w.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.to_vec_f32().unwrap(), data);
}

#[test]
fn roundtrip_mmap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_mmap.safetensors");

    let data = vec![10.0_f32, 20.0, 30.0];
    let t = Tensor::from_f32(&data, &Shape::new(vec![3]), &cpu()).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("bias".to_string(), t);

    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors_mmap(&path, &cpu()).unwrap();

    assert_eq!(loaded.len(), 1);
    let b = &loaded["bias"];
    assert_eq!(b.shape(), &Shape::new(vec![3]));
    assert_eq!(b.to_vec_f32().unwrap(), data);
}

#[test]
fn roundtrip_multiple_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "a".to_string(),
        Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![2]), &cpu()).unwrap(),
    );
    tensors.insert(
        "b".to_string(),
        Tensor::from_f32(&[3.0, 4.0, 5.0, 6.0], &Shape::new(vec![2, 2]), &cpu()).unwrap(),
    );

    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors(&path, &cpu()).unwrap();

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded["a"].to_vec_f32().unwrap(), vec![1.0, 2.0]);
    assert_eq!(loaded["b"].to_vec_f32().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(loaded["b"].shape(), &Shape::new(vec![2, 2]));
}

#[test]
fn empty_tensors_map() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.safetensors");

    let tensors: HashMap<String, Tensor> = HashMap::new();
    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors(&path, &cpu()).unwrap();
    assert!(loaded.is_empty());
}

#[test]
fn error_missing_file() {
    let result = load_safetensors(Path::new("/nonexistent/path.safetensors"), &cpu());
    assert!(result.is_err());
}

#[test]
fn error_corrupt_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.safetensors");
    std::fs::write(&path, b"not a valid safetensors file").unwrap();

    let result = load_safetensors(&path, &cpu());
    assert!(result.is_err());
}

#[test]
fn roundtrip_preserves_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape.safetensors");

    let shapes = [
        Shape::new(vec![1]),
        Shape::new(vec![2, 3]),
        Shape::new(vec![1, 1, 4]),
    ];
    let mut tensors = HashMap::new();
    for (i, shape) in shapes.iter().enumerate() {
        let n = shape.numel() as usize;
        let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
        let t = Tensor::from_f32(&data, shape, &cpu()).unwrap();
        tensors.insert(format!("t{i}"), t);
    }

    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors(&path, &cpu()).unwrap();

    for (i, shape) in shapes.iter().enumerate() {
        assert_eq!(loaded[&format!("t{i}")].shape(), shape);
    }
}

#[test]
fn roundtrip_multi_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_dtype.safetensors");

    let data = vec![1.0_f32, 2.0, 3.0];
    let shape = Shape::new(vec![3]);

    let mut tensors = HashMap::new();

    // F32 tensor
    tensors.insert(
        "f32".to_string(),
        Tensor::from_f32(&data, &shape, &cpu()).unwrap(),
    );

    // F16 tensor — stored as f32 internally but with F16 dtype metadata
    tensors.insert(
        "f16".to_string(),
        Tensor::from_data_with_dtype(data.clone(), &shape, DType::F16, &cpu()).unwrap(),
    );

    // BF16 tensor
    tensors.insert(
        "bf16".to_string(),
        Tensor::from_data_with_dtype(data.clone(), &shape, DType::BF16, &cpu()).unwrap(),
    );

    // I32 tensor
    tensors.insert(
        "i32".to_string(),
        Tensor::from_data_with_dtype(vec![1.0, 2.0, 3.0], &shape, DType::I32, &cpu()).unwrap(),
    );

    save_safetensors(&path, &tensors).unwrap();
    let loaded = load_safetensors(&path, &cpu()).unwrap();

    assert_eq!(loaded["f32"].dtype(), DType::F32);
    assert_eq!(loaded["f16"].dtype(), DType::F16);
    assert_eq!(loaded["bf16"].dtype(), DType::BF16);
    assert_eq!(loaded["i32"].dtype(), DType::I32);

    // F32 should be exact
    assert_eq!(loaded["f32"].to_vec_f32().unwrap(), data);

    // F16/BF16 may lose precision in the roundtrip (f32 → f16 → f32)
    let f16_vals = loaded["f16"].to_vec_f32().unwrap();
    for (a, b) in f16_vals.iter().zip(data.iter()) {
        assert!((a - b).abs() < 0.01, "f16 mismatch: {a} vs {b}");
    }

    let bf16_vals = loaded["bf16"].to_vec_f32().unwrap();
    for (a, b) in bf16_vals.iter().zip(data.iter()) {
        assert!((a - b).abs() < 0.02, "bf16 mismatch: {a} vs {b}");
    }

    // I32 should be exact for small integers
    assert_eq!(loaded["i32"].to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0]);
}

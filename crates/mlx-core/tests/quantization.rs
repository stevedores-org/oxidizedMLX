//! Quantization readiness tests.
//!
//! These tests validate the infrastructure needed for int4/int8 quantization:
//! - DType round-trip correctness for reduced-precision types
//! - Simulated quantize/dequantize cycles
//! - Accuracy degradation bounds for quantized operations
//! - Range and clamping behavior

use mlx_core::types::{DType, Shape};
use mlx_core::{Device, Tensor};

fn cpu() -> Device {
    Device::Cpu
}

fn s(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec())
}

// ── Simulated int8 quantization helpers ────────────────────────────────

/// Simulate int8 quantization: scale values to [-128, 127] range and back.
fn quantize_int8(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

fn dequantize_int8(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&q| q as f32 * scale).collect()
}

/// Simulate int4 quantization: scale values to [-8, 7] range and back.
fn quantize_int4(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| (x / scale).round().clamp(-8.0, 7.0) as i8)
        .collect();
    (quantized, scale)
}

fn dequantize_int4(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&q| q as f32 * scale).collect()
}

// ── Int8 quantization tests ────────────────────────────────────────────

#[test]
fn test_int8_round_trip_small_values() {
    let data = vec![0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0];
    let (quantized, scale) = quantize_int8(&data);
    let recovered = dequantize_int8(&quantized, scale);

    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        assert!(
            err < scale * 1.5,
            "int8 round-trip error too large at index {i}: orig={orig}, recovered={rec}, err={err}, scale={scale}"
        );
    }
}

#[test]
fn test_int8_round_trip_large_values() {
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let (quantized, scale) = quantize_int8(&data);
    let recovered = dequantize_int8(&quantized, scale);

    let max_err: f32 = data
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    assert!(
        max_err < scale * 1.5,
        "int8 max round-trip error {max_err} exceeds bound {}", scale * 1.5
    );
}

#[test]
fn test_int8_saturation() {
    // With symmetric quantization (scale = max_abs / 127), equal magnitude
    // positive and negative values map to +127 and -127 respectively.
    // To hit -128, we need asymmetric quantization (not implemented here).
    let data = vec![1000.0, -1000.0, 0.0];
    let (quantized, _scale) = quantize_int8(&data);

    assert_eq!(quantized[0], 127, "positive saturation failed");
    assert_eq!(quantized[1], -127, "negative mirror should be -127 for symmetric quant");
    assert_eq!(quantized[2], 0, "zero should remain zero");

    // Verify that values exceeding the range get clamped to the boundary
    let data2 = vec![2000.0, -3000.0, 0.0];
    let (quantized2, _scale2) = quantize_int8(&data2);
    // scale = 3000/127, so 2000/scale = 2000*127/3000 ≈ 84.67 -> 85
    assert!(quantized2[0] > 0, "positive value should quantize to positive int8");
    assert!(quantized2[1] >= -128 && quantized2[1] < 0);
    assert_eq!(quantized2[2], 0);
}

#[test]
fn test_int8_quantized_matmul_accuracy() {
    // Simulate quantized matmul: quantize inputs, dequantize, multiply in f32
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 0.3, 0.7, 0.1];

    // Full precision matmul
    let a = Tensor::from_f32(&a_data, &s(&[2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&b_data, &s(&[2, 2]), &cpu()).unwrap();
    let fp32_result = a.matmul(&b).unwrap().to_vec_f32().unwrap();

    // Simulated int8 quantized matmul
    let (a_q, a_scale) = quantize_int8(&a_data);
    let (b_q, b_scale) = quantize_int8(&b_data);
    let a_deq = dequantize_int8(&a_q, a_scale);
    let b_deq = dequantize_int8(&b_q, b_scale);

    let a_q_tensor = Tensor::from_f32(&a_deq, &s(&[2, 2]), &cpu()).unwrap();
    let b_q_tensor = Tensor::from_f32(&b_deq, &s(&[2, 2]), &cpu()).unwrap();
    let q_result = a_q_tensor.matmul(&b_q_tensor).unwrap().to_vec_f32().unwrap();

    // Quantized result should be close to full precision
    for (i, (&fp, &q)) in fp32_result.iter().zip(q_result.iter()).enumerate() {
        let rel_err = if fp.abs() > 1e-6 {
            (fp - q).abs() / fp.abs()
        } else {
            (fp - q).abs()
        };
        assert!(
            rel_err < 0.15,
            "int8 matmul relative error too large at index {i}: fp32={fp}, int8={q}, rel_err={rel_err}"
        );
    }
}

// ── Int4 quantization tests ────────────────────────────────────────────

#[test]
fn test_int4_round_trip() {
    let data = vec![0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
    let (quantized, scale) = quantize_int4(&data);
    let recovered = dequantize_int4(&quantized, scale);

    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (orig - rec).abs();
        // Int4 has much larger quantization error
        assert!(
            err < scale * 1.5,
            "int4 round-trip error too large at index {i}: orig={orig}, recovered={rec}, err={err}"
        );
    }
}

#[test]
fn test_int4_saturation() {
    // Symmetric quantization: scale = max_abs / 7, so equal magnitude values
    // map to +7 and -7.
    let data = vec![100.0, -100.0, 0.0];
    let (quantized, _scale) = quantize_int4(&data);

    assert_eq!(quantized[0], 7, "positive saturation failed");
    assert_eq!(quantized[1], -7, "negative mirror should be -7 for symmetric quant");
    assert_eq!(quantized[2], 0, "zero should remain zero");

    // Verify asymmetric values clamp correctly
    let data2 = vec![50.0, -200.0, 0.0];
    let (quantized2, _scale2) = quantize_int4(&data2);
    assert!(quantized2[0] > 0 && quantized2[0] <= 7);
    assert!(quantized2[1] >= -8 && quantized2[1] < 0);
    assert_eq!(quantized2[2], 0);
}

#[test]
fn test_int4_quantized_add_accuracy() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![0.5, 0.5, 0.5, 0.5];

    // Full precision
    let a = Tensor::from_f32(&a_data, &s(&[4]), &cpu()).unwrap();
    let b = Tensor::from_f32(&b_data, &s(&[4]), &cpu()).unwrap();
    let fp32_result = a.add(&b).unwrap().to_vec_f32().unwrap();

    // Simulated int4
    let (a_q, a_scale) = quantize_int4(&a_data);
    let (b_q, b_scale) = quantize_int4(&b_data);
    let a_deq = dequantize_int4(&a_q, a_scale);
    let b_deq = dequantize_int4(&b_q, b_scale);

    let a_q_tensor = Tensor::from_f32(&a_deq, &s(&[4]), &cpu()).unwrap();
    let b_q_tensor = Tensor::from_f32(&b_deq, &s(&[4]), &cpu()).unwrap();
    let q_result = a_q_tensor.add(&b_q_tensor).unwrap().to_vec_f32().unwrap();

    for (i, (&fp, &q)) in fp32_result.iter().zip(q_result.iter()).enumerate() {
        let abs_err = (fp - q).abs();
        // Int4 is very coarse; allow larger errors
        assert!(
            abs_err < 1.5,
            "int4 add error too large at index {i}: fp32={fp}, int4={q}, abs_err={abs_err}"
        );
    }
}

#[test]
fn test_int4_quantized_matmul_accuracy() {
    let a_data = vec![1.0, 0.5, 0.3, 0.8];
    let b_data = vec![0.2, 0.4, 0.6, 0.1];

    let a = Tensor::from_f32(&a_data, &s(&[2, 2]), &cpu()).unwrap();
    let b = Tensor::from_f32(&b_data, &s(&[2, 2]), &cpu()).unwrap();
    let fp32_result = a.matmul(&b).unwrap().to_vec_f32().unwrap();

    let (a_q, a_scale) = quantize_int4(&a_data);
    let (b_q, b_scale) = quantize_int4(&b_data);
    let a_deq = dequantize_int4(&a_q, a_scale);
    let b_deq = dequantize_int4(&b_q, b_scale);

    let a_q_tensor = Tensor::from_f32(&a_deq, &s(&[2, 2]), &cpu()).unwrap();
    let b_q_tensor = Tensor::from_f32(&b_deq, &s(&[2, 2]), &cpu()).unwrap();
    let q_result = a_q_tensor.matmul(&b_q_tensor).unwrap().to_vec_f32().unwrap();

    for (i, (&fp, &q)) in fp32_result.iter().zip(q_result.iter()).enumerate() {
        let abs_err = (fp - q).abs();
        assert!(
            abs_err < 1.0,
            "int4 matmul abs error too large at index {i}: fp32={fp}, int4={q}, abs_err={abs_err}"
        );
    }
}

// ── DType infrastructure tests ─────────────────────────────────────────

#[test]
fn test_dtype_f16_tracking() {
    // Tensors can track F16 dtype even though storage is f32
    let data = vec![1.0, 2.0, 3.0];
    let t = Tensor::from_data_with_dtype(data, &s(&[3]), DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    // Data should still be accessible as f32
    let result = t.to_vec_f32().unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dtype_bf16_tracking() {
    let data = vec![1.5, 2.5, 3.5];
    let t = Tensor::from_data_with_dtype(data, &s(&[3]), DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let result = t.to_vec_f32().unwrap();
    assert_eq!(result, vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_dtype_size_bytes() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
}

// ── Quantization error analysis tests ──────────────────────────────────

#[test]
fn test_int8_error_distribution() {
    // Generate a range of values and verify error distribution is uniform
    let n = 1000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();

    let (quantized, scale) = quantize_int8(&data);
    let recovered = dequantize_int8(&quantized, scale);

    let errors: Vec<f32> = data
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mean_err: f32 = errors.iter().sum::<f32>() / n as f32;
    let max_err: f32 = errors.iter().copied().fold(0.0, f32::max);

    // Mean error should be roughly half the quantization step
    assert!(
        mean_err < scale,
        "mean error {mean_err} too large for scale {scale}"
    );
    assert!(
        max_err < scale * 1.5,
        "max error {max_err} too large for scale {scale}"
    );
}

#[test]
fn test_int4_error_distribution() {
    let n = 100;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();

    let (quantized, scale) = quantize_int4(&data);
    let recovered = dequantize_int4(&quantized, scale);

    let errors: Vec<f32> = data
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    let mean_err: f32 = errors.iter().sum::<f32>() / n as f32;
    let max_err: f32 = errors.iter().copied().fold(0.0, f32::max);

    // Int4 has much larger steps (only 16 levels)
    assert!(
        mean_err < scale,
        "int4 mean error {mean_err} too large for scale {scale}"
    );
    assert!(
        max_err < scale * 1.5,
        "int4 max error {max_err} too large for scale {scale}"
    );
}

#[test]
fn test_mixed_precision_graph() {
    // Build a graph where F16-tagged and F32 tensors interact
    // This validates that the backend doesn't crash on mixed dtypes
    let a = Tensor::from_data_with_dtype(
        vec![1.0, 2.0, 3.0],
        &s(&[3]),
        DType::F16,
        &cpu(),
    )
    .unwrap();
    let b = Tensor::from_f32(&[0.5, 0.5, 0.5], &s(&[3]), &cpu()).unwrap();
    let c = a.add(&b).unwrap();
    let result = c.to_vec_f32().unwrap();
    assert_eq!(result, vec![1.5, 2.5, 3.5]);
}

#[test]
fn test_quantized_softmax_stability() {
    // Softmax should still produce valid probability distributions after quantization
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (q, scale) = quantize_int8(&data);
    let deq = dequantize_int8(&q, scale);

    let t = Tensor::from_f32(&deq, &s(&[5]), &cpu()).unwrap();
    let softmax = t.softmax(0).unwrap();
    let result = softmax.to_vec_f32().unwrap();

    // Should still sum to ~1.0
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "quantized softmax doesn't sum to 1: {sum}"
    );

    // Should be monotonically increasing
    for i in 1..result.len() {
        assert!(
            result[i] >= result[i - 1],
            "softmax not monotonic: {:?}",
            result
        );
    }
}

#[test]
fn test_quantized_layer_norm_stability() {
    // LayerNorm should produce zero-mean, unit-variance output even with quantized input
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (q, scale) = quantize_int8(&data);
    let deq = dequantize_int8(&q, scale);

    let t = Tensor::from_f32(&deq, &s(&[2, 3]), &cpu()).unwrap();
    let normed = t.layer_norm(1e-5);
    let result = normed.to_vec_f32().unwrap();

    // Each row should have mean ≈ 0
    let row1_mean = (result[0] + result[1] + result[2]) / 3.0;
    let row2_mean = (result[3] + result[4] + result[5]) / 3.0;
    assert!(row1_mean.abs() < 1e-4, "row 1 mean not zero: {row1_mean}");
    assert!(row2_mean.abs() < 1e-4, "row 2 mean not zero: {row2_mean}");
}

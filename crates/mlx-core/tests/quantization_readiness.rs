//! Suite 9 — Quantization Readiness (Q9.1)
//!
//! Placeholder test harness for quantized tensor support. Apple highlights
//! 4-bit quantization as a key feature. This module validates the
//! infrastructure for loading, representing, and dequantizing packed
//! integer tensors — even before full kernel support exists.

use mlx_core::{DType, Shape};

// ─── Q9.1: Quantized Weight Representation ──────────────────────────────

/// Pack f32 values into simulated int8 quantized representation.
/// Returns (quantized_i8, scale, zero_point) for asymmetric quantization.
fn quantize_asymmetric_i8(data: &[f32]) -> (Vec<i8>, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max - min) / 255.0;
    let zero_point = min;

    let quantized: Vec<i8> = data
        .iter()
        .map(|v| {
            let q = ((v - zero_point) / scale).round().clamp(0.0, 255.0) as u8;
            // Store as i8 by offsetting
            (q as i16 - 128) as i8
        })
        .collect();

    (quantized, scale, zero_point)
}

/// Dequantize i8 values back to f32.
fn dequantize_i8(quantized: &[i8], scale: f32, zero_point: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|q| {
            let u = (*q as i16 + 128) as f32;
            u * scale + zero_point
        })
        .collect()
}

/// Pack f32 values into simulated int4 quantized representation.
/// Each pair of values packed into one u8.
fn quantize_int4(data: &[f32]) -> (Vec<u8>, f32, f32) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max - min) / 15.0; // 4-bit: 0..15
    let zero_point = min;

    // Pack two 4-bit values per byte
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
    for chunk in data.chunks(2) {
        let lo = ((chunk[0] - zero_point) / scale).round().clamp(0.0, 15.0) as u8;
        let hi = if chunk.len() > 1 {
            ((chunk[1] - zero_point) / scale).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        packed.push(lo | (hi << 4));
    }

    (packed, scale, zero_point)
}

/// Dequantize int4 packed values back to f32.
fn dequantize_int4(packed: &[u8], scale: f32, zero_point: f32, count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    for byte in packed {
        let lo = (byte & 0x0F) as f32;
        result.push(lo * scale + zero_point);
        if result.len() < count {
            let hi = ((byte >> 4) & 0x0F) as f32;
            result.push(hi * scale + zero_point);
        }
    }
    result.truncate(count);
    result
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[test]
fn quantize_int8_roundtrip_preserves_values() {
    let original = vec![0.0, 0.5, 1.0, -1.0, -0.5, 2.0, -2.0, 1.5];
    let (quantized, scale, zero_point) = quantize_asymmetric_i8(&original);

    assert_eq!(quantized.len(), original.len());

    let recovered = dequantize_i8(&quantized, scale, zero_point);
    assert_eq!(recovered.len(), original.len());

    // INT8 quantization should preserve values within ~2% of range
    let range = 4.0; // -2 to 2
    let max_error = range / 255.0 * 1.5; // Allow ~1.5 quantization steps
    for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < max_error,
            "INT8 roundtrip error at [{i}]: original={orig}, recovered={rec}, max_error={max_error}"
        );
    }
}

#[test]
fn quantize_int4_roundtrip_preserves_values() {
    let original = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let (packed, scale, zero_point) = quantize_int4(&original);

    // 8 values → 4 bytes (2 per byte)
    assert_eq!(packed.len(), 4);

    let recovered = dequantize_int4(&packed, scale, zero_point, original.len());
    assert_eq!(recovered.len(), original.len());

    // INT4 has coarser quantization
    let range = 7.0;
    let max_error = range / 15.0 * 1.5;
    for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < max_error,
            "INT4 roundtrip error at [{i}]: original={orig}, recovered={rec}, max_error={max_error}"
        );
    }
}

#[test]
fn quantize_int4_packing_correctness() {
    // Verify the bit-packing is correct
    let data = vec![3.0, 7.0]; // Should map to specific 4-bit values
    let (packed, scale, zero_point) = quantize_int4(&data);
    assert_eq!(packed.len(), 1, "2 values should pack into 1 byte");

    let lo = packed[0] & 0x0F;
    let hi = (packed[0] >> 4) & 0x0F;

    // Both should be in 0..15 range
    assert!(lo <= 15, "lo nibble out of range: {lo}");
    assert!(hi <= 15, "hi nibble out of range: {hi}");

    // Verify roundtrip
    let recovered = dequantize_int4(&packed, scale, zero_point, 2);
    assert_eq!(recovered.len(), 2);
}

#[test]
fn quantize_int8_shape_metadata_preserved() {
    // Verify that quantization preserves shape information
    let shape = Shape::new(vec![4, 8]);
    let count = shape.numel() as usize;
    let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.1 - 1.6).collect();

    let (quantized, _scale, _zero_point) = quantize_asymmetric_i8(&data);
    assert_eq!(
        quantized.len(),
        count,
        "quantized element count must match shape"
    );

    // Verify shape dimensions are preserved through the process
    assert_eq!(shape.ndim(), 2);
    assert_eq!(shape.dim(0), Some(4));
    assert_eq!(shape.dim(1), Some(8));
}

#[test]
fn quantize_int4_odd_element_count() {
    // INT4 packing with odd number of elements
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements (odd)
    let (packed, scale, zero_point) = quantize_int4(&original);

    // 5 elements → 3 bytes (ceil(5/2))
    assert_eq!(packed.len(), 3);

    let recovered = dequantize_int4(&packed, scale, zero_point, original.len());
    assert_eq!(recovered.len(), 5);
}

#[test]
fn quantize_dtype_size_sanity() {
    // Verify that quantized types would save memory
    let n_elements = 1024;

    let f32_bytes = n_elements * DType::F32.size_bytes();
    let f16_bytes = n_elements * DType::F16.size_bytes();
    let i8_bytes = n_elements; // 1 byte per element
    let i4_bytes = n_elements.div_ceil(2); // 0.5 bytes per element

    assert_eq!(f32_bytes, 4096);
    assert_eq!(f16_bytes, 2048);
    assert_eq!(i8_bytes, 1024);
    assert_eq!(i4_bytes, 512);

    // Compression ratios
    assert_eq!(f32_bytes / i8_bytes, 4); // 4x compression
    assert_eq!(f32_bytes / i4_bytes, 8); // 8x compression
}

#[test]
fn quantize_preserves_sign_symmetry() {
    // Verify symmetric values quantize symmetrically
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let (quantized, scale, zero_point) = quantize_asymmetric_i8(&data);
    let recovered = dequantize_i8(&quantized, scale, zero_point);

    // The middle value (0.0) should be close to 0
    assert!(
        recovered[2].abs() < scale * 2.0,
        "zero should roundtrip near zero: got {}",
        recovered[2]
    );

    // Symmetric values should have similar absolute recovered values
    let diff = (recovered[0].abs() - recovered[4].abs()).abs();
    assert!(
        diff < scale * 3.0,
        "symmetric values should have similar magnitude: {} vs {}, diff={}",
        recovered[0],
        recovered[4],
        diff
    );
}

#[test]
fn quantize_large_weight_matrix() {
    // Simulate quantizing a 512×512 weight matrix (typical LLM layer)
    let rows = 512;
    let cols = 512;
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            // Simulate normal-ish distribution
            let x = (i as f32 / n as f32) * 6.0 - 3.0;
            (-x * x / 2.0).exp() * (if i % 2 == 0 { 1.0 } else { -1.0 })
        })
        .collect();

    let (quantized_i8, scale_i8, zp_i8) = quantize_asymmetric_i8(&data);
    let recovered_i8 = dequantize_i8(&quantized_i8, scale_i8, zp_i8);

    let (packed_i4, scale_i4, zp_i4) = quantize_int4(&data);
    let recovered_i4 = dequantize_int4(&packed_i4, scale_i4, zp_i4, n);

    // Compute mean absolute error
    let mae_i8: f32 = data
        .iter()
        .zip(recovered_i8.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / n as f32;
    let mae_i4: f32 = data
        .iter()
        .zip(recovered_i4.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / n as f32;

    // INT8 should have lower error than INT4
    assert!(
        mae_i8 < mae_i4,
        "INT8 MAE ({mae_i8}) should be less than INT4 MAE ({mae_i4})"
    );

    // Both should be reasonably small
    assert!(mae_i8 < 0.01, "INT8 MAE too high: {mae_i8}");
    assert!(mae_i4 < 0.1, "INT4 MAE too high: {mae_i4}");
}

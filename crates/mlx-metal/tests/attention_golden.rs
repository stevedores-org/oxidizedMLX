//! Golden tests: Metal attention pipeline vs CPU reference.

#[cfg(target_os = "macos")]
mod metal_tests {
    use mlx_core::graph::{OpKind, TensorMeta};
    use mlx_core::types::{DType, Shape};
    use mlx_metal::metal_stream;

    /// CPU reference: scaled masked softmax.
    fn cpu_scaled_masked_softmax(scores: &[f32], tq: usize, tk: usize, scale: f32, causal: bool) -> Vec<f32> {
        let mut data = vec![0.0f32; tq * tk];
        for i in 0..tq {
            for j in 0..tk {
                let idx = i * tk + j;
                let mut val = scores[idx] * scale;
                if causal && j > i {
                    val = -1e9;
                }
                data[idx] = val;
            }
            let row_start = i * tk;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..tk {
                if data[row_start + j] > max_val {
                    max_val = data[row_start + j];
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..tk {
                data[row_start + j] = (data[row_start + j] - max_val).exp();
                sum_exp += data[row_start + j];
            }
            for j in 0..tk {
                data[row_start + j] /= sum_exp;
            }
        }
        data
    }

    /// CPU reference: matmul [M,K] @ [K,N] -> [M,N].
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += a[row * k + i] * b[i * n + col];
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    /// CPU reference: transpose 2D [rows, cols] -> [cols, rows].
    fn cpu_transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        out
    }

    /// CPU reference: full attention.
    #[allow(clippy::too_many_arguments)]
    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, dh: usize, scale: f32, causal: bool) -> Vec<f32> {
        let kt = cpu_transpose(k, tk, dh);
        let scores = cpu_matmul(q, &kt, tq, dh, tk);
        let probs = cpu_scaled_masked_softmax(&scores, tq, tk, scale, causal);
        cpu_matmul(&probs, v, tq, tk, dh)
    }

    /// Assert element-wise closeness with attention-appropriate tolerance.
    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(
                diff <= tol,
                "element {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
            );
        }
    }

    /// Deterministic test data.
    fn gen_data(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.01).sin() * 0.5).collect()
    }

    // ── ScaledMaskedSoftmax tests ──

    #[test]
    fn test_softmax_masked_standalone() {
        let stream = metal_stream().expect("Metal available");
        let tq = 4usize;
        let tk = 8usize;
        let scale = 0.125f32;
        let scores = gen_data(tq * tk, 1.0);

        let expected = cpu_scaled_masked_softmax(&scores, tq, tk, scale, true);

        let s = stream.add_constant(
            scores,
            TensorMeta { shape: Shape::new(vec![tq as i64, tk as i64]), dtype: DType::F32 },
        );
        let out = stream.add_op(
            OpKind::ScaledMaskedSoftmax { scale, causal: true },
            smallvec::SmallVec::from_slice(&[s]),
            TensorMeta { shape: Shape::new(vec![tq as i64, tk as i64]), dtype: DType::F32 },
        );

        stream.eval(out).expect("eval");
        let result = stream.get_buffer(out).expect("buffer");
        assert_close(&result, &expected, 1e-5, 1e-5);
    }

    #[test]
    fn test_softmax_masked_causal_identity() {
        let stream = metal_stream().expect("Metal available");
        let tq = 4usize;
        let tk = 4usize;
        let scale = 1.0f32;
        // Uniform scores
        let scores = vec![1.0f32; tq * tk];

        let expected = cpu_scaled_masked_softmax(&scores, tq, tk, scale, true);

        let s = stream.add_constant(
            scores,
            TensorMeta { shape: Shape::new(vec![tq as i64, tk as i64]), dtype: DType::F32 },
        );
        let out = stream.add_op(
            OpKind::ScaledMaskedSoftmax { scale, causal: true },
            smallvec::SmallVec::from_slice(&[s]),
            TensorMeta { shape: Shape::new(vec![tq as i64, tk as i64]), dtype: DType::F32 },
        );

        stream.eval(out).expect("eval");
        let result = stream.get_buffer(out).expect("buffer");

        // Verify upper triangle is ~zero (causal mask)
        for i in 0..tq {
            for j in (i + 1)..tk {
                assert!(
                    result[i * tk + j] < 1e-5,
                    "row {i} col {j} should be ~0, got {}",
                    result[i * tk + j]
                );
            }
        }
        assert_close(&result, &expected, 1e-5, 1e-5);
    }

    // ── Full Attention tests ──

    fn run_attention_test(tq: usize, tk: usize, dh: usize, causal: bool) {
        let stream = metal_stream().expect("Metal available");
        let scale = 1.0 / (dh as f32).sqrt();

        let q_data = gen_data(tq * dh, 0.0);
        let k_data = gen_data(tk * dh, 100.0);
        let v_data = gen_data(tk * dh, 200.0);

        let expected = cpu_attention(&q_data, &k_data, &v_data, tq, tk, dh, scale, causal);

        let q = stream.add_constant(
            q_data,
            TensorMeta { shape: Shape::new(vec![tq as i64, dh as i64]), dtype: DType::F32 },
        );
        let k = stream.add_constant(
            k_data,
            TensorMeta { shape: Shape::new(vec![tk as i64, dh as i64]), dtype: DType::F32 },
        );
        let v = stream.add_constant(
            v_data,
            TensorMeta { shape: Shape::new(vec![tk as i64, dh as i64]), dtype: DType::F32 },
        );
        let out = stream.add_op(
            OpKind::Attention { scale, causal },
            smallvec::SmallVec::from_slice(&[q, k, v]),
            TensorMeta { shape: Shape::new(vec![tq as i64, dh as i64]), dtype: DType::F32 },
        );

        stream.eval(out).expect("eval");
        let result = stream.get_buffer(out).expect("buffer");
        assert_close(&result, &expected, 1e-3, 1e-3);
    }

    #[test]
    fn test_attention_small() {
        run_attention_test(4, 4, 16, true);
    }

    #[test]
    fn test_attention_medium() {
        run_attention_test(32, 32, 64, true);
    }

    #[test]
    fn test_attention_asymmetric() {
        run_attention_test(8, 16, 32, true);
    }

    #[test]
    fn test_attention_no_mask() {
        run_attention_test(4, 4, 16, false);
    }

    #[test]
    fn test_attention_invalid_shapes() {
        let stream = metal_stream().expect("Metal available");

        // Mismatched head dimensions: Q [4, 16], K [4, 8], V [4, 8]
        let q = stream.add_constant(
            vec![0.0f32; 4 * 16],
            TensorMeta { shape: Shape::new(vec![4, 16]), dtype: DType::F32 },
        );
        let k = stream.add_constant(
            vec![0.0f32; 4 * 8],
            TensorMeta { shape: Shape::new(vec![4, 8]), dtype: DType::F32 },
        );
        let v = stream.add_constant(
            vec![0.0f32; 4 * 8],
            TensorMeta { shape: Shape::new(vec![4, 8]), dtype: DType::F32 },
        );
        let out = stream.add_op(
            OpKind::Attention { scale: 0.25, causal: true },
            smallvec::SmallVec::from_slice(&[q, k, v]),
            TensorMeta { shape: Shape::new(vec![4, 16]), dtype: DType::F32 },
        );

        let result = stream.eval(out);
        assert!(result.is_err(), "should fail with mismatched head dimensions");
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn attention_tests_require_macos() {
    eprintln!("Attention golden tests require macOS with Metal support");
}

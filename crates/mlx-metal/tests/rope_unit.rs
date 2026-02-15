//! Unit and correctness tests for Metal RoPE kernel.

#[cfg(target_os = "macos")]
mod tests {
    use mlx_core::graph::{OpKind, TensorMeta};
    use mlx_core::types::{DType, Shape};

    // ── CPU reference implementation ────────────────────────────────────

    fn cpu_rope(
        x: &[f32],
        tokens: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_offset: usize,
        theta: f32,
    ) -> Vec<f32> {
        let mut out = x.to_vec();
        for t in 0..tokens {
            for i in 0..rotary_dim / 2 {
                let inv_freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let angle = (pos_offset + t) as f32 * inv_freq;
                let (s, c) = angle.sin_cos();

                let base = t * head_dim + i * 2;
                let x0 = x[base];
                let x1 = x[base + 1];

                out[base] = x0 * c - x1 * s;
                out[base + 1] = x0 * s + x1 * c;
            }
        }
        out
    }

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

    // ── Validation tests ────────────────────────────────────────────────

    #[test]
    fn rope_rejects_odd_rotary_dim() {
        let stream = mlx_metal::metal_stream().unwrap();

        let x = stream.add_constant(
            vec![0.0f32; 4 * 8],
            TensorMeta {
                shape: Shape::new(vec![4, 8]),
                dtype: DType::F32,
            },
        );

        let c = stream.add_op(
            OpKind::Rope {
                rotary_dim: 7,
                pos_offset: 0,
                theta: 10000.0,
            },
            smallvec::SmallVec::from_slice(&[x]),
            TensorMeta {
                shape: Shape::new(vec![4, 8]),
                dtype: DType::F32,
            },
        );

        assert!(stream.eval(c).is_err());
    }

    #[test]
    fn rope_rejects_rotary_dim_gt_head_dim() {
        let stream = mlx_metal::metal_stream().unwrap();

        let x = stream.add_constant(
            vec![0.0f32; 2 * 8],
            TensorMeta {
                shape: Shape::new(vec![2, 8]),
                dtype: DType::F32,
            },
        );

        let c = stream.add_op(
            OpKind::Rope {
                rotary_dim: 16,
                pos_offset: 0,
                theta: 10000.0,
            },
            smallvec::SmallVec::from_slice(&[x]),
            TensorMeta {
                shape: Shape::new(vec![2, 8]),
                dtype: DType::F32,
            },
        );

        assert!(stream.eval(c).is_err());
    }

    // ── Correctness tests (Metal vs CPU reference) ──────────────────────

    fn run_rope_test(
        tokens: usize,
        head_dim: usize,
        rotary_dim: usize,
        pos_offset: usize,
        theta: f32,
    ) {
        let numel = tokens * head_dim;
        let x_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 0.5).collect();

        let expected = cpu_rope(&x_data, tokens, head_dim, rotary_dim, pos_offset, theta);

        let stream = mlx_metal::metal_stream().unwrap();

        let x = stream.add_constant(
            x_data,
            TensorMeta {
                shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                dtype: DType::F32,
            },
        );

        let c = stream.add_op(
            OpKind::Rope {
                rotary_dim,
                pos_offset,
                theta,
            },
            smallvec::SmallVec::from_slice(&[x]),
            TensorMeta {
                shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                dtype: DType::F32,
            },
        );

        stream.eval(c).expect("eval should succeed");
        let result = stream.get_buffer(c).expect("buffer should exist");
        assert_close(&result, &expected, 5e-4, 5e-3);
    }

    #[test]
    fn rope_4x16_full_rotation() {
        run_rope_test(4, 16, 16, 0, 10000.0);
    }

    #[test]
    fn rope_4x16_partial_rotation() {
        run_rope_test(4, 16, 8, 0, 10000.0);
    }

    #[test]
    fn rope_with_pos_offset() {
        run_rope_test(4, 16, 16, 100, 10000.0);
    }

    #[test]
    fn rope_128x128_llm_sized() {
        run_rope_test(128, 128, 128, 0, 10000.0);
    }

    #[test]
    fn rope_non_power_of_2() {
        run_rope_test(17, 32, 24, 5, 10000.0);
    }

    #[test]
    fn rope_passthrough_beyond_rotary_dim() {
        let tokens = 2;
        let head_dim = 8;
        let rotary_dim = 4;
        let numel = tokens * head_dim;
        let x_data: Vec<f32> = (0..numel).map(|i| i as f32).collect();

        let expected = cpu_rope(&x_data, tokens, head_dim, rotary_dim, 0, 10000.0);

        // Elements beyond rotary_dim should be unchanged.
        for t in 0..tokens {
            for d in rotary_dim..head_dim {
                let idx = t * head_dim + d;
                assert_eq!(
                    expected[idx], x_data[idx],
                    "element at t={t}, d={d} should be unchanged"
                );
            }
        }

        // Also verify through Metal.
        let stream = mlx_metal::metal_stream().unwrap();
        let x = stream.add_constant(
            x_data,
            TensorMeta {
                shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                dtype: DType::F32,
            },
        );
        let c = stream.add_op(
            OpKind::Rope {
                rotary_dim,
                pos_offset: 0,
                theta: 10000.0,
            },
            smallvec::SmallVec::from_slice(&[x]),
            TensorMeta {
                shape: Shape::new(vec![tokens as i64, head_dim as i64]),
                dtype: DType::F32,
            },
        );
        stream.eval(c).unwrap();
        let result = stream.get_buffer(c).unwrap();
        assert_close(&result, &expected, 5e-4, 5e-3);
    }

    #[test]
    fn rope_empty_input() {
        let stream = mlx_metal::metal_stream().unwrap();
        let x = stream.add_constant(
            Vec::<f32>::new(),
            TensorMeta {
                shape: Shape::new(vec![0, 16]),
                dtype: DType::F32,
            },
        );
        let c = stream.add_op(
            OpKind::Rope {
                rotary_dim: 16,
                pos_offset: 0,
                theta: 10000.0,
            },
            smallvec::SmallVec::from_slice(&[x]),
            TensorMeta {
                shape: Shape::new(vec![0, 16]),
                dtype: DType::F32,
            },
        );
        stream.eval(c).unwrap();
        let result = stream.get_buffer(c).unwrap();
        assert!(result.is_empty());
    }
}

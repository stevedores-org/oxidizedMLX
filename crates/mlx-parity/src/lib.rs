//! Backend parity gate: compares two backends (e.g., Metal vs CPU) for all ops.
//!
//! This crate provides a test harness that runs every `OpKind` on two backends
//! and asserts that outputs match within tolerance. This is the gate that must
//! pass before shipping Metal as the default backend.

use mlx_core::backend::{Backend, NodeInput, Stream};
use mlx_core::cpu_kernels::CpuRefBackend;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use mlx_core::Result;
use smallvec::SmallVec;

/// Configuration for a parity test case.
pub struct ParityCase {
    pub name: &'static str,
    pub op: OpKind,
    pub inputs: Vec<(Vec<f32>, Shape)>,
    pub output_shape: Shape,
    pub atol: f32,
    pub rtol: f32,
}

/// Run a parity check between the CPU reference backend and a test backend.
/// Returns the (cpu_result, test_result) pair on success.
pub fn check_parity(
    test_backend: &dyn Backend,
    case: &ParityCase,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let cpu = CpuRefBackend;

    let meta = TensorMeta {
        shape: case.output_shape.clone(),
        dtype: DType::F32,
    };

    // Build NodeInput references
    let shapes: Vec<Shape> = case.inputs.iter().map(|(_, s)| s.clone()).collect();
    let inputs: Vec<NodeInput<'_>> = case
        .inputs
        .iter()
        .zip(shapes.iter())
        .map(|((data, _), shape)| NodeInput {
            data: data.as_slice(),
            shape,
            dtype: DType::F32,
        })
        .collect();

    let cpu_result = cpu.eval_node(&case.op, &inputs, &meta)?;
    let test_result = test_backend.eval_node(&case.op, &inputs, &meta)?;

    mlx_conformance::assert_allclose(&test_result, &cpu_result, case.atol, case.rtol);

    Ok((cpu_result, test_result))
}

/// Run a parity check using the Stream-level API (lazy graph evaluation).
/// This tests the full end-to-end path including graph construction and eval scheduling.
pub fn check_stream_parity(
    cpu_stream: &Stream,
    test_stream: &Stream,
    case: &ParityCase,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Add inputs to both streams
    let mut cpu_inputs = SmallVec::<[mlx_core::NodeId; 2]>::new();
    let mut test_inputs = SmallVec::<[mlx_core::NodeId; 2]>::new();

    for (data, shape) in &case.inputs {
        let meta = TensorMeta {
            shape: shape.clone(),
            dtype: DType::F32,
        };
        cpu_inputs.push(cpu_stream.add_constant(data.clone(), meta.clone()));
        test_inputs.push(test_stream.add_constant(data.clone(), meta));
    }

    let out_meta = TensorMeta {
        shape: case.output_shape.clone(),
        dtype: DType::F32,
    };

    let cpu_out = cpu_stream.add_op(case.op.clone(), cpu_inputs, out_meta.clone());
    let test_out = test_stream.add_op(case.op.clone(), test_inputs, out_meta);

    cpu_stream.eval(cpu_out)?;
    test_stream.eval(test_out)?;

    let cpu_result = cpu_stream.get_buffer(cpu_out).unwrap();
    let test_result = test_stream.get_buffer(test_out).unwrap();

    mlx_conformance::assert_allclose(&test_result, &cpu_result, case.atol, case.rtol);

    Ok((cpu_result, test_result))
}

/// Generate all standard parity test cases for the ops matrix.
pub fn standard_ops_matrix() -> Vec<ParityCase> {
    vec![
        // Elementwise binary ops
        ParityCase {
            name: "add_1d",
            op: OpKind::Add,
            inputs: vec![
                (vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4])),
                (vec![0.1, 0.2, 0.3, 0.4], Shape::new(vec![4])),
            ],
            output_shape: Shape::new(vec![4]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "add_2d",
            op: OpKind::Add,
            inputs: vec![
                (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3])),
                (vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape::new(vec![2, 3])),
            ],
            output_shape: Shape::new(vec![2, 3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "sub_1d",
            op: OpKind::Sub,
            inputs: vec![
                (vec![5.0, 7.0, 9.0], Shape::new(vec![3])),
                (vec![1.0, 2.0, 3.0], Shape::new(vec![3])),
            ],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "mul_1d",
            op: OpKind::Mul,
            inputs: vec![
                (vec![2.0, 3.0, 4.0], Shape::new(vec![3])),
                (vec![0.5, 0.5, 0.5], Shape::new(vec![3])),
            ],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "div_1d",
            op: OpKind::Div,
            inputs: vec![
                (vec![10.0, 20.0, 30.0], Shape::new(vec![3])),
                (vec![2.0, 4.0, 5.0], Shape::new(vec![3])),
            ],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // Elementwise unary ops
        ParityCase {
            name: "neg_1d",
            op: OpKind::Neg,
            inputs: vec![(vec![1.0, -2.0, 3.0], Shape::new(vec![3]))],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "exp_1d",
            op: OpKind::Exp,
            inputs: vec![(vec![0.0, 1.0, -1.0, 0.5], Shape::new(vec![4]))],
            output_shape: Shape::new(vec![4]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "log_1d",
            op: OpKind::Log,
            inputs: vec![(vec![1.0, 2.718, 10.0, 0.5], Shape::new(vec![4]))],
            output_shape: Shape::new(vec![4]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "sqrt_1d",
            op: OpKind::Sqrt,
            inputs: vec![(vec![1.0, 4.0, 9.0, 16.0], Shape::new(vec![4]))],
            output_shape: Shape::new(vec![4]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // Activations
        ParityCase {
            name: "silu_1d",
            op: OpKind::Silu,
            inputs: vec![(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new(vec![5]))],
            output_shape: Shape::new(vec![5]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "gelu_1d",
            op: OpKind::Gelu,
            inputs: vec![(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new(vec![5]))],
            output_shape: Shape::new(vec![5]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "softmax_1d",
            op: OpKind::Softmax { axis: 0 },
            inputs: vec![(vec![1.0, 2.0, 3.0], Shape::new(vec![3]))],
            output_shape: Shape::new(vec![3]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "softmax_2d_axis1",
            op: OpKind::Softmax { axis: 1 },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![2, 3]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        // Reductions
        ParityCase {
            name: "sum_axis0",
            op: OpKind::Sum { axis: Some(0) },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "sum_all",
            op: OpKind::Sum { axis: None },
            inputs: vec![(vec![1.0, 2.0, 3.0], Shape::new(vec![3]))],
            output_shape: Shape::scalar(),
            atol: 1e-6,
            rtol: 1e-6,
        },
        ParityCase {
            name: "mean_axis1",
            op: OpKind::Mean { axis: Some(1) },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![2]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "max_axis0",
            op: OpKind::Max { axis: Some(0) },
            inputs: vec![(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // MatMul
        ParityCase {
            name: "matmul_2x2",
            op: OpKind::MatMul,
            inputs: vec![
                (vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2])),
                (vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2])),
            ],
            output_shape: Shape::new(vec![2, 2]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "matmul_4x3_3x2",
            op: OpKind::MatMul,
            inputs: vec![
                (
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    Shape::new(vec![4, 3]),
                ),
                (
                    (0..6).map(|i| (i as f32) * 0.2).collect(),
                    Shape::new(vec![3, 2]),
                ),
            ],
            output_shape: Shape::new(vec![4, 2]),
            atol: 1e-4,
            rtol: 1e-4,
        },
        // Normalization
        ParityCase {
            name: "layer_norm_2x3",
            op: OpKind::LayerNorm { eps: 1e-5 },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![2, 3]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        ParityCase {
            name: "rms_norm_2x3",
            op: OpKind::RmsNorm { eps: 1e-5 },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![2, 3]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        // Transpose
        ParityCase {
            name: "transpose_2x3",
            op: OpKind::Transpose {
                axes: Some(vec![1, 0]),
            },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]))],
            output_shape: Shape::new(vec![3, 2]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // RoPE
        ParityCase {
            name: "rope_4x4",
            op: OpKind::Rope {
                rotary_dim: 4,
                pos_offset: 0,
                theta: 10_000.0,
            },
            inputs: vec![(
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                Shape::new(vec![4, 4]),
            )],
            output_shape: Shape::new(vec![4, 4]),
            atol: 1e-5,
            rtol: 1e-5,
        },
        // Attention
        ParityCase {
            name: "attention_causal_4x4",
            op: OpKind::Attention {
                scale: 0.5,
                causal: true,
            },
            inputs: vec![
                (
                    (0..16).map(|i| ((i as f32) * 0.1).sin()).collect(),
                    Shape::new(vec![4, 4]),
                ),
                (
                    (0..16).map(|i| ((i as f32) * 0.13).cos()).collect(),
                    Shape::new(vec![4, 4]),
                ),
                (
                    (0..16).map(|i| ((i as f32) * 0.17).sin()).collect(),
                    Shape::new(vec![4, 4]),
                ),
            ],
            output_shape: Shape::new(vec![4, 4]),
            atol: 1e-4,
            rtol: 1e-4,
        },
        // Embedding
        ParityCase {
            name: "embedding_lookup",
            op: OpKind::Embedding,
            inputs: vec![
                (
                    (0..12).map(|i| i as f32).collect(),
                    Shape::new(vec![4, 3]),
                ),
                (vec![0.0, 2.0, 1.0, 3.0], Shape::new(vec![4])),
            ],
            output_shape: Shape::new(vec![4, 3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // Narrow
        ParityCase {
            name: "narrow_axis0",
            op: OpKind::Narrow {
                axis: 0,
                start: 1,
                length: 2,
            },
            inputs: vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![3, 2]))],
            output_shape: Shape::new(vec![2, 2]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // Concatenate
        ParityCase {
            name: "concatenate_axis0",
            op: OpKind::Concatenate { axis: 0 },
            inputs: vec![
                (vec![1.0, 2.0], Shape::new(vec![1, 2])),
                (vec![3.0, 4.0], Shape::new(vec![1, 2])),
            ],
            output_shape: Shape::new(vec![2, 2]),
            atol: 1e-6,
            rtol: 1e-6,
        },
        // Broadcast
        ParityCase {
            name: "broadcast_1_to_3",
            op: OpKind::Broadcast {
                target_shape: Shape::new(vec![3]),
            },
            inputs: vec![(vec![5.0], Shape::new(vec![1]))],
            output_shape: Shape::new(vec![3]),
            atol: 1e-6,
            rtol: 1e-6,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_self_parity() {
        // Sanity check: CPU backend should be perfectly consistent with itself
        let cases = standard_ops_matrix();
        for case in &cases {
            let cpu = CpuRefBackend;
            let result = check_parity(&cpu, case);
            assert!(
                result.is_ok(),
                "CPU self-parity failed for {}: {:?}",
                case.name,
                result.err()
            );
        }
    }

    #[test]
    fn test_stream_level_cpu_parity() {
        let cases = standard_ops_matrix();
        for case in &cases {
            let cpu_stream = Stream::new(Box::new(CpuRefBackend));
            let test_stream = Stream::new(Box::new(CpuRefBackend));
            let result = check_stream_parity(&cpu_stream, &test_stream, case);
            assert!(
                result.is_ok(),
                "Stream-level CPU self-parity failed for {}: {:?}",
                case.name,
                result.err()
            );
        }
    }

    #[test]
    fn test_ops_matrix_coverage() {
        // Verify the ops matrix covers all major OpKind variants
        let cases = standard_ops_matrix();
        let op_names: Vec<&str> = cases.iter().map(|c| c.name).collect();

        // Check we have at least one test for each category
        assert!(op_names.iter().any(|n| n.starts_with("add")), "missing Add tests");
        assert!(op_names.iter().any(|n| n.starts_with("sub")), "missing Sub tests");
        assert!(op_names.iter().any(|n| n.starts_with("mul")), "missing Mul tests");
        assert!(op_names.iter().any(|n| n.starts_with("div")), "missing Div tests");
        assert!(op_names.iter().any(|n| n.starts_with("neg")), "missing Neg tests");
        assert!(op_names.iter().any(|n| n.starts_with("exp")), "missing Exp tests");
        assert!(op_names.iter().any(|n| n.starts_with("log")), "missing Log tests");
        assert!(op_names.iter().any(|n| n.starts_with("sqrt")), "missing Sqrt tests");
        assert!(op_names.iter().any(|n| n.starts_with("silu")), "missing Silu tests");
        assert!(op_names.iter().any(|n| n.starts_with("gelu")), "missing Gelu tests");
        assert!(op_names.iter().any(|n| n.starts_with("softmax")), "missing Softmax tests");
        assert!(op_names.iter().any(|n| n.starts_with("sum")), "missing Sum tests");
        assert!(op_names.iter().any(|n| n.starts_with("mean")), "missing Mean tests");
        assert!(op_names.iter().any(|n| n.starts_with("max")), "missing Max tests");
        assert!(op_names.iter().any(|n| n.starts_with("matmul")), "missing MatMul tests");
        assert!(op_names.iter().any(|n| n.starts_with("layer_norm")), "missing LayerNorm tests");
        assert!(op_names.iter().any(|n| n.starts_with("rms_norm")), "missing RmsNorm tests");
        assert!(op_names.iter().any(|n| n.starts_with("transpose")), "missing Transpose tests");
        assert!(op_names.iter().any(|n| n.starts_with("rope")), "missing RoPE tests");
        assert!(op_names.iter().any(|n| n.starts_with("attention")), "missing Attention tests");
        assert!(op_names.iter().any(|n| n.starts_with("embedding")), "missing Embedding tests");
        assert!(op_names.iter().any(|n| n.starts_with("narrow")), "missing Narrow tests");
        assert!(op_names.iter().any(|n| n.starts_with("concatenate")), "missing Concatenate tests");
        assert!(op_names.iter().any(|n| n.starts_with("broadcast")), "missing Broadcast tests");
    }
}

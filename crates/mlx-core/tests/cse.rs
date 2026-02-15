//! CSE (Common Subexpression Elimination) integration tests.

use std::sync::Arc;

use mlx_core::backend::Stream;
use mlx_core::graph::{OpKind, TensorMeta};
use mlx_core::types::{DType, Shape};
use smallvec::SmallVec;

fn meta_f32(shape: Vec<i64>) -> TensorMeta {
    TensorMeta {
        shape: Shape::new(shape),
        dtype: DType::F32,
    }
}

/// Create a fresh stream so tests don't interfere via the global default.
fn fresh_stream() -> Arc<Stream> {
    Arc::new(Stream::new(Box::new(mlx_core::cpu_kernels::CpuRefBackend)))
}

#[test]
fn cse_dedups_identical_ops() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0, 2.0], meta_f32(vec![2]));
    let b = s.add_constant(vec![3.0, 4.0], meta_f32(vec![2]));

    let c1 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );
    let c2 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );

    assert_eq!(c1, c2, "identical add ops should return the same NodeId");
}

#[test]
fn cse_dedups_identical_constants() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0, 2.0, 3.0], meta_f32(vec![3]));
    let b = s.add_constant(vec![1.0, 2.0, 3.0], meta_f32(vec![3]));

    assert_eq!(a, b, "identical constants should return the same NodeId");
}

#[test]
fn cse_different_inputs_not_deduped() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0], meta_f32(vec![1]));
    let b = s.add_constant(vec![2.0], meta_f32(vec![1]));
    let c = s.add_constant(vec![3.0], meta_f32(vec![1]));

    let ab = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![1]),
    );
    let ac = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, c]),
        meta_f32(vec![1]),
    );

    assert_ne!(ab, ac, "different inputs should produce different NodeIds");
}

#[test]
fn cse_different_ops_not_deduped() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0, 2.0], meta_f32(vec![2]));
    let b = s.add_constant(vec![3.0, 4.0], meta_f32(vec![2]));

    let add = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );
    let mul = s.add_op(
        OpKind::Mul,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );

    assert_ne!(add, mul, "different ops should produce different NodeIds");
}

#[test]
fn cse_graph_node_count() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0], meta_f32(vec![1]));
    let b = s.add_constant(vec![2.0], meta_f32(vec![1]));

    assert_eq!(s.graph_node_count(), 2, "two constants should exist");

    let _c1 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![1]),
    );
    assert_eq!(
        s.graph_node_count(),
        3,
        "first add should create one new node"
    );

    let _c2 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![1]),
    );
    assert_eq!(
        s.graph_node_count(),
        3,
        "duplicate add should not create a new node"
    );
}

#[test]
fn cse_deduped_ops_produce_correct_results() {
    let s = fresh_stream();
    let a = s.add_constant(vec![1.0, 2.0], meta_f32(vec![2]));
    let b = s.add_constant(vec![3.0, 4.0], meta_f32(vec![2]));

    let c1 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );
    let c2 = s.add_op(
        OpKind::Add,
        SmallVec::from_slice(&[a, b]),
        meta_f32(vec![2]),
    );

    s.eval(c1).unwrap();
    s.eval(c2).unwrap();

    assert_eq!(s.get_buffer(c1).unwrap(), vec![4.0, 6.0]);
    assert_eq!(s.get_buffer(c2).unwrap(), vec![4.0, 6.0]);
}

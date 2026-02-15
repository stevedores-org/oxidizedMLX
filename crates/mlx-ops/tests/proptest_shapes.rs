//! Property tests for shape inference, broadcasting, and dtype promotion.
//!
//! These tests use proptest to generate random shapes and verify invariants
//! that must hold for any valid input.

use mlx_core::graph::OpKind;
use mlx_core::{DType, Shape};
use mlx_ops::{broadcast_shapes, infer_shape, promote};
use proptest::prelude::*;

// ── Strategies ───────────────────────────────────────────────────────────

/// Generate a random dimension value (1..=8 to keep tests fast).
fn dim() -> impl Strategy<Value = i64> {
    1i64..=8
}

/// Generate a random shape with rank 0..=4.
fn arb_shape() -> impl Strategy<Value = Shape> {
    prop::collection::vec(dim(), 0..=4).prop_map(Shape::new)
}

/// Generate a broadcastable pair of shapes.
fn broadcastable_pair() -> impl Strategy<Value = (Shape, Shape)> {
    prop::collection::vec(dim(), 1..=4).prop_flat_map(|target| {
        let len = target.len();
        (
            0..=len,
            prop::collection::vec(prop::bool::ANY, len),
            Just(target),
        )
            .prop_map(|(skip, masks, t)| {
                // Build `a` by taking a suffix of `t` (different rank) and masking some dims to 1.
                // This exercises both rank-extension and per-dimension broadcasting behavior.
                let a_dims: Vec<i64> = t[skip..]
                    .iter()
                    .zip(masks[skip..].iter())
                    .map(|(&d, &keep)| if keep { d } else { 1 })
                    .collect();
                (Shape::new(a_dims), Shape::new(t))
            })
    })
}

/// Generate a random DType.
fn arb_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::F32),
        Just(DType::F16),
        Just(DType::BF16),
        Just(DType::I32),
        Just(DType::I64),
    ]
}

/// Numeric priority for dtype promotion (higher = wider).
///
/// Keep in sync with `crates/mlx-ops/src/dtype_promotion.rs`.
fn dtype_priority(dt: DType) -> u8 {
    match dt {
        DType::I32 => 1,
        DType::I64 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::F32 => 5,
    }
}

/// Generate a 2D shape for matmul.
fn matmul_shapes() -> impl Strategy<Value = (Shape, Shape)> {
    (dim(), dim(), dim()).prop_map(|(m, k, n)| (Shape::new(vec![m, k]), Shape::new(vec![k, n])))
}

/// Generate a shape and a valid axis for it (including negative indexing).
fn shape_with_axis(rank: std::ops::RangeInclusive<usize>) -> impl Strategy<Value = (Shape, i32)> {
    prop::collection::vec(dim(), rank).prop_flat_map(|dims| {
        let ndim = dims.len() as i32;
        let shape = Shape::new(dims);
        (
            Just(shape),
            (0..ndim, prop::bool::ANY)
                .prop_map(move |(axis, negative)| if negative { axis - ndim } else { axis }),
        )
    })
}

// ── Broadcasting property tests ──────────────────────────────────────────

proptest! {
    /// Broadcasting is commutative.
    #[test]
    fn broadcast_commutative(a in arb_shape(), b in arb_shape()) {
        let ab = broadcast_shapes(&a, &b);
        let ba = broadcast_shapes(&b, &a);
        prop_assert_eq!(ab, ba);
    }

    /// A shape broadcasts with itself to itself.
    #[test]
    fn broadcast_self_identity(a in arb_shape()) {
        let result = broadcast_shapes(&a, &a);
        prop_assert_eq!(result, Some(a));
    }

    /// Known-broadcastable pairs always produce a valid result.
    #[test]
    fn broadcast_valid_pairs((a, b) in broadcastable_pair()) {
        let result = broadcast_shapes(&a, &b);
        prop_assert!(result.is_some());
    }

    /// Broadcast result rank is max(rank(a), rank(b)).
    #[test]
    fn broadcast_result_rank(a in arb_shape(), b in arb_shape()) {
        if let Some(result) = broadcast_shapes(&a, &b) {
            let expected_rank = a.ndim().max(b.ndim());
            prop_assert_eq!(result.ndim(), expected_rank);
        }
    }

    /// Each dimension of the broadcast result >= corresponding input dimensions.
    #[test]
    fn broadcast_dims_at_least_inputs((a, b) in broadcastable_pair()) {
        let result = broadcast_shapes(&a, &b).unwrap();
        for (i, &rd) in result.0.iter().rev().enumerate() {
            if i < a.0.len() {
                let ad = a.0[a.0.len() - 1 - i];
                prop_assert!(rd >= ad);
            }
            if i < b.0.len() {
                let bd = b.0[b.0.len() - 1 - i];
                prop_assert!(rd >= bd);
            }
        }
    }

    /// Broadcasting with a scalar always succeeds and returns the other shape.
    #[test]
    fn broadcast_scalar(a in arb_shape()) {
        let scalar = Shape::scalar();
        let result = broadcast_shapes(&a, &scalar);
        prop_assert_eq!(result, Some(a));
    }
}

// ── Shape inference property tests ───────────────────────────────────────

proptest! {
    /// Unary ops preserve the input shape.
    #[test]
    fn unary_preserves_shape(a in arb_shape()) {
        for op in &[OpKind::Neg, OpKind::Silu, OpKind::Gelu] {
            let result = infer_shape(op, &[&a]).unwrap();
            prop_assert_eq!(result, a.clone());
        }
    }

    /// LayerNorm/RmsNorm preserve the input shape.
    #[test]
    fn norm_preserves_shape(a in arb_shape()) {
        for op in &[
            OpKind::LayerNorm { eps: 1e-5 },
            OpKind::RmsNorm { eps: 1e-5 },
        ] {
            let result = infer_shape(op, &[&a]).unwrap();
            prop_assert_eq!(result, a.clone());
        }
    }

    /// MatMul: [M,K] @ [K,N] → [M,N]
    #[test]
    fn matmul_shape_correct((a, b) in matmul_shapes()) {
        let result = infer_shape(&OpKind::MatMul, &[&a, &b]).unwrap();
        prop_assert_eq!(result.0[0], a.0[0]);
        prop_assert_eq!(result.0[1], b.0[1]);
        prop_assert_eq!(result.ndim(), 2);
    }

    /// MatMul with mismatched inner dims always fails.
    #[test]
    fn matmul_mismatch_fails(m in dim(), k1 in dim(), k2 in dim(), n in dim()) {
        prop_assume!(k1 != k2);
        let a = Shape::new(vec![m, k1]);
        let b = Shape::new(vec![k2, n]);
        prop_assert!(infer_shape(&OpKind::MatMul, &[&a, &b]).is_err());
    }

    /// Sum(axis=None) always produces a scalar shape.
    #[test]
    fn sum_all_is_scalar(a in arb_shape()) {
        let result = infer_shape(&OpKind::Sum { axis: None }, &[&a]).unwrap();
        prop_assert_eq!(result, Shape::scalar());
    }

    /// Sum(axis=0) removes exactly one dimension for rank >= 2.
    #[test]
    fn sum_axis_removes_one_dim((shape, axis) in shape_with_axis(2..=4)) {
        let result = infer_shape(&OpKind::Sum { axis: Some(axis) }, &[&shape]).unwrap();
        prop_assert_eq!(result.ndim(), shape.ndim() - 1);
    }

    /// Transpose(None) reverses dimensions.
    #[test]
    fn transpose_reverses(dims in prop::collection::vec(dim(), 1..=4)) {
        let shape = Shape::new(dims.clone());
        let result = infer_shape(&OpKind::Transpose { axes: None }, &[&shape]).unwrap();
        let expected: Vec<i64> = dims.iter().rev().copied().collect();
        prop_assert_eq!(result.0, expected);
    }

    /// Transpose preserves numel.
    #[test]
    fn transpose_preserves_numel(dims in prop::collection::vec(dim(), 1..=4)) {
        let shape = Shape::new(dims);
        let result = infer_shape(&OpKind::Transpose { axes: None }, &[&shape]).unwrap();
        prop_assert_eq!(result.numel(), shape.numel());
    }

    /// Softmax preserves shape for valid axis.
    #[test]
    fn softmax_preserves_shape((shape, axis) in shape_with_axis(1..=4)) {
        let result = infer_shape(&OpKind::Softmax { axis }, &[&shape]).unwrap();
        prop_assert_eq!(result, shape);
    }
}

// ── DType promotion property tests ───────────────────────────────────────

proptest! {
    /// Promotion is commutative.
    #[test]
    fn promote_commutative(a in arb_dtype(), b in arb_dtype()) {
        prop_assert_eq!(promote(a, b), promote(b, a));
    }

    /// Promotion is at least as wide as both inputs (by promotion priority, not by byte width).
    #[test]
    fn promote_at_least_as_wide(a in arb_dtype(), b in arb_dtype()) {
        let result = promote(a, b);
        prop_assert!(dtype_priority(result) >= dtype_priority(a).max(dtype_priority(b)));
    }

    /// Promoting a dtype with itself returns the same dtype.
    #[test]
    fn promote_self_identity(a in arb_dtype()) {
        prop_assert_eq!(promote(a, a), a);
    }

    /// Promotion result is an upper bound: promoting it with either input yields itself.
    #[test]
    fn promote_is_upper_bound(a in arb_dtype(), b in arb_dtype()) {
        let result = promote(a, b);
        prop_assert_eq!(promote(result, a), result);
        prop_assert_eq!(promote(result, b), result);
    }

    /// Promoting any type with F32 gives F32 (widest float).
    #[test]
    fn promote_with_f32(a in arb_dtype()) {
        prop_assert_eq!(promote(a, DType::F32), DType::F32);
    }
}

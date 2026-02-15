//! Broadcasting rules following NumPy/MLX semantics.

use mlx_core::Shape;

/// Compute the broadcast shape of two shapes, or None if incompatible.
///
/// Rules (NumPy-style):
/// 1. Align shapes from the trailing dimension.
/// 2. For each dimension pair: must be equal, or one must be 1.
/// 3. The output dimension is the max of the two.
pub fn broadcast_shapes(a: &Shape, b: &Shape) -> Option<Shape> {
    let a_dims = &a.0;
    let b_dims = &b.0;
    let max_ndim = a_dims.len().max(b_dims.len());

    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < a_dims.len() {
            a_dims[a_dims.len() - 1 - i]
        } else {
            1
        };
        let db = if i < b_dims.len() {
            b_dims[b_dims.len() - 1 - i]
        } else {
            1
        };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return None;
        }
    }

    result.reverse();
    Some(Shape::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_shapes() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 3]);
        assert_eq!(broadcast_shapes(&a, &b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_scalar_broadcast() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::scalar();
        assert_eq!(broadcast_shapes(&a, &b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_one_broadcast() {
        let a = Shape::new(vec![2, 1]);
        let b = Shape::new(vec![1, 3]);
        assert_eq!(broadcast_shapes(&a, &b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_rank_extension() {
        let a = Shape::new(vec![3]);
        let b = Shape::new(vec![2, 3]);
        assert_eq!(broadcast_shapes(&a, &b), Some(Shape::new(vec![2, 3])));
    }

    #[test]
    fn test_incompatible() {
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![2, 4]);
        assert_eq!(broadcast_shapes(&a, &b), None);
    }

    #[test]
    fn test_higher_rank() {
        let a = Shape::new(vec![1, 3, 1]);
        let b = Shape::new(vec![2, 1, 4]);
        assert_eq!(
            broadcast_shapes(&a, &b),
            Some(Shape::new(vec![2, 3, 4]))
        );
    }
}

//! Core type definitions: DType, Shape.

/// Supported data types for tensor elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
}

impl DType {
    /// Size in bytes of a single element.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

/// Tensor shape (dimensions).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<i64>);

impl Shape {
    pub fn new(dims: impl Into<Vec<i64>>) -> Self {
        Self(dims.into())
    }

    /// Scalar (rank-0) shape.
    pub fn scalar() -> Self {
        Self(vec![])
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        self.0.iter().product()
    }

    /// Get dimension at axis (supports negative indexing).
    pub fn dim(&self, axis: i32) -> Option<i64> {
        let ndim = self.0.len() as i32;
        let idx = if axis < 0 { ndim + axis } else { axis };
        if idx >= 0 && idx < ndim {
            Some(self.0[idx as usize])
        } else {
            None
        }
    }

    /// Compute the broadcast shape of two shapes, or None if incompatible.
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
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_numel() {
        assert_eq!(Shape::new(vec![2, 3, 4]).numel(), 24);
        assert_eq!(Shape::scalar().numel(), 1);
        assert_eq!(Shape::new(vec![0, 5]).numel(), 0);
    }

    #[test]
    fn test_shape_dim_negative_index() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(-1), Some(4));
        assert_eq!(s.dim(-3), Some(2));
        assert_eq!(s.dim(3), None);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::I64.size_bytes(), 8);
    }
}

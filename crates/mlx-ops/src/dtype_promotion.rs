//! DType promotion rules following MLX / NumPy semantics.
//!
//! When two tensors with different dtypes are combined in a binary op,
//! the result dtype is determined by these promotion rules.

use mlx_core::DType;

/// Promote two dtypes to a common result dtype.
///
/// Rules:
/// - Same dtype → same dtype
/// - Float + Float → wider float
/// - Int + Float → the float type
/// - Int + Int → wider int
pub fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    let pa = priority(a);
    let pb = priority(b);
    if pa >= pb { a } else { b }
}

/// Numeric priority for dtype promotion (higher = wider).
pub fn priority(dt: DType) -> u8 {
    match dt {
        DType::I32 => 1,
        DType::I64 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::F32 => 5,
    }
}

/// Check whether a dtype is a floating-point type.
pub fn is_float(dt: DType) -> bool {
    matches!(dt, DType::F32 | DType::F16 | DType::BF16)
}

/// Check whether a dtype is an integer type.
pub fn is_integer(dt: DType) -> bool {
    matches!(dt, DType::I32 | DType::I64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_dtype() {
        assert_eq!(promote(DType::F32, DType::F32), DType::F32);
        assert_eq!(promote(DType::I32, DType::I32), DType::I32);
    }

    #[test]
    fn test_float_promotion() {
        assert_eq!(promote(DType::F16, DType::F32), DType::F32);
        assert_eq!(promote(DType::BF16, DType::F32), DType::F32);
        assert_eq!(promote(DType::F16, DType::BF16), DType::BF16);
    }

    #[test]
    fn test_int_float_promotion() {
        assert_eq!(promote(DType::I32, DType::F32), DType::F32);
        assert_eq!(promote(DType::I64, DType::F16), DType::F16);
        assert_eq!(promote(DType::I32, DType::F16), DType::F16);
    }

    #[test]
    fn test_int_promotion() {
        assert_eq!(promote(DType::I32, DType::I64), DType::I64);
    }

    #[test]
    fn test_symmetry() {
        for &a in &[DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64] {
            for &b in &[DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64] {
                assert_eq!(
                    promote(a, b),
                    promote(b, a),
                    "promote({a:?}, {b:?}) not symmetric"
                );
            }
        }
    }

    #[test]
    fn test_is_float() {
        assert!(is_float(DType::F32));
        assert!(is_float(DType::F16));
        assert!(is_float(DType::BF16));
        assert!(!is_float(DType::I32));
        assert!(!is_float(DType::I64));
    }
}

//! Op registry, shape inference, broadcasting rules, and dtype promotion.

pub mod broadcast;
pub mod dtype_promotion;
pub mod shape_inference;

pub use broadcast::broadcast_shapes;
pub use dtype_promotion::promote;
pub use shape_inference::infer_shape;

//! Reverse-mode automatic differentiation (autograd).
//!
//! Provides `grad(f)` for scalar losses, a VJP registry per op, and
//! composable transforms following MLX's design.

// TODO(milestone-5): implement tape/graph-based AD + VJP registry

//! Pure Rust CPU backend — correctness oracle for conformance testing.
//!
//! This backend implements tensor operations in safe Rust with no external
//! dependencies. It is intentionally simple rather than fast, serving as the
//! reference implementation against which FFI and Metal backends are validated.

// TODO(milestone-4): implement Backend trait + kernel dispatch

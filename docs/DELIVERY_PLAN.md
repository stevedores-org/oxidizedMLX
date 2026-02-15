# Rust MLX Delivery Plan (Epics + User Stories)

This is a delivery plan to turn an MLX-like stack into a Rust-first runtime without a big-bang rewrite. The ordering is deliberate: **correctness and conformance first**, performance and GPU/Metal later.

## Epic 0: Repo Baseline (DX + CI)

Goal: contributors can clone, run `just ci`, and get fast feedback without special environment variables.

User stories:

- As a contributor, I can run `just ci` on a fresh clone and it passes without `MLX_SRC`.
- As a maintainer, CI runs fmt/clippy/tests on Linux and macOS for `main` and `develop`.
- As a maintainer, the repo has a single place to understand crates, env vars, and workflows (`README.md`, `docs/*`).

Acceptance:

- `just ci` passes without requiring the upstream MLX checkout.
- `mlx-sys` is excluded from default lint/test paths unless explicitly requested.

## Epic 1: Backend Abstraction (The “Narrow Waist”)

Goal: make “tensor API” independent from “execution backend”.

User stories:

- As an operator author, I can implement an op once (in `mlx-ops`) against a backend trait.
- As a backend author, I can plug in a new backend (CPU / Metal / FFI) with minimal surface area.
- As a user, I can select a device/backend at tensor creation time.

Acceptance:

- A `Backend` trait (or equivalent) exists with explicit contracts for memory, shape, dtype, and dispatch.
- A minimal set of ops route through backend dispatch (add/mul/matmul/reductions).

## Epic 2: Core Tensor API (Safety + Ergonomics)

Goal: define a stable, safe Rust public API for tensors and scalar types.

User stories:

- As a user, I can create tensors from slices, shapes, and scalar fills.
- As a user, I can query dtype/shape/device and get stable error messages.
- As a maintainer, API invariants are enforced by types and tests.

Acceptance:

- `Tensor` has clear ownership rules and no footguns around aliasing.
- Errors are typed (not stringly-typed) at the public boundary.

## Epic 3: CPU Reference Backend (Correctness Oracle)

Goal: have a deterministic correctness-first backend used for tests and conformance.

User stories:

- As a maintainer, I can trust CPU backend results as the oracle for correctness tests.
- As a contributor, I can add kernels for ops without touching FFI/Metal.
- As a user, I can run models end-to-end on CPU even if slower.

Acceptance:

- Core ops (elementwise, matmul, reductions, broadcast) implemented and covered by unit + property tests.
- Deterministic behavior for random seeds and floating point edge cases (documented where not possible).

## Epic 4: Conformance Harness (Rust vs Python MLX)

Goal: compare behavior against upstream MLX (Python) to lock down semantics.

User stories:

- As a maintainer, I can run “conformance” locally to compare outputs for a suite of ops.
- As a contributor, I can add a new op and update conformance vectors in one place.
- As a maintainer, I can gate merging on conformance results for a defined subset.

Acceptance:

- A harness exists that runs a fixed suite of ops across shapes/dtypes and compares tolerances.
- Conformance reports failures with repro snippets (inputs + op + expected vs got).

## Epic 5: Autograd (Reverse-Mode AD)

Goal: enable training and gradient-based optimization.

User stories:

- As a user, I can call `backward()` and get gradients for leaf parameters.
- As an op implementer, I can register VJPs (vector-Jacobian products) per op.
- As a maintainer, gradients are validated via numerical checks where feasible.

Acceptance:

- Tape/graph representation with topological traversal and gradient accumulation.
- Gradient tests for representative ops (matmul, sum, relu-like, broadcast).

## Epic 6: NN Modules + Optimizers

Goal: basic training loop experience (modules, parameters, optimizers).

User stories:

- As a user, I can define modules (Linear, LayerNorm, Embedding) with parameters.
- As a user, I can run forward/backward + optimizer step (SGD/AdamW).
- As a user, I can save/load weights in a stable format (state dict).

Acceptance:

- Parameter container and state dict conventions are defined and tested.
- AdamW + SGD implemented with unit tests against known small cases.

## Epic 7: IO (safetensors + mmap)

Goal: fast, safe model weight loading and basic tensor serialization.

User stories:

- As a user, I can load safetensors files via mmap without copying unnecessarily.
- As a user, I can validate metadata (dtype/shape) before allocation.
- As a maintainer, IO is fuzz/property tested for robustness.

Acceptance:

- `mlx-io` implements safetensors read with clear safety boundaries.
- Corrupt/invalid files produce actionable errors.

## Epic 8: FFI Integration (Upstream MLX C++ via C ABI)

Goal: use upstream MLX where it makes sense, behind a stable C ABI shim.

User stories:

- As a backend author, I can implement an FFI backend that satisfies the same backend trait.
- As a maintainer, I can control the FFI surface area and keep it minimal.
- As a user, I can opt into the FFI backend without breaking the pure-Rust path.

Acceptance:

- `mlx-sys` builds from `MLX_SRC` and exposes a minimal, versioned C ABI.
- Safety boundaries are explicit (no hidden global state assumptions).

## Epic 9: Metal Backend (macOS GPU)

Goal: a native Metal backend (or thin wrapper) with stable semantics.

User stories:

- As a user on macOS, I can run core ops on Metal and match CPU semantics within tolerance.
- As a maintainer, I can triage correctness vs performance regressions via benchmarks + conformance.

Acceptance:

- Minimal runtime scaffolding (queue, buffers, kernels) with conformance for the covered ops.

## Epic 10: Benchmarks + Performance Program

Goal: measure and improve performance without regressing correctness.

User stories:

- As a maintainer, I can run benches for core kernels and see trends.
- As a contributor, I can add a kernel and prove it’s faster with benchmarks.

Acceptance:

- Criterion benches exist for matmul, reductions, and elementwise ops across sizes/dtypes.

## Epic 11: Packaging (Crate UX, SemVer, Feature Flags)

Goal: make this publishable and consumable.

User stories:

- As a user, I can depend on `mlx-core` and select backends via cargo features.
- As a maintainer, I can ship breaking changes behind semver increments with clear changelogs.

Acceptance:

- Feature flags: `cpu` (default), `ffi`, `metal`, `conformance` (dev-only).
- Docs and examples build in CI.


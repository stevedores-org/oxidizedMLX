# Architecture + TDD Plan (Staged Conversion)

This document describes a staged approach to making MLX Rust-first without a big-bang rewrite.

## Target End State

Two layers, at least initially:

1. Rust-first API + runtime (safe, ergonomic).
2. Backend(s) that start as the existing MLX C++/Metal implementation and are replaced progressively.

This lets us ship Rust integration early while porting subsystems one-by-one.

## Workspace Layout

Current workspace already reflects the intended modules:

- `mlx-sys`: unsafe FFI to a minimal C ABI shim.
- `mlx-core`: tensor API + lazy graph IR.
- `mlx-ops`: op registry + inference.
- `mlx-cpu`: correctness backend.
- `mlx-metal`: native Apple Silicon backend.
- `mlx-autograd`: AD + VJP registry.
- `mlx-nn`, `mlx-optim`, `mlx-io`: productization layer.

## Core Concepts

### Tensor Is a Handle (Lazy Node)

MLX is lazy; arrays materialize only when needed.

Rust `Tensor` should be a lightweight handle to:

- `Shape`/`DType`/`Device`
- a node id in a graph arena
- optionally a realized buffer/handle after `eval()`

### Graph IR + Scheduler

Minimum viable IR supports:

- constants / params
- elementwise ops
- reductions
- matmul
- shape ops (reshape/transpose)
- a single materialization boundary (`eval()`)

Scheduler responsibilities:

- topological order
- caching/memoization
- backend dispatch

### Backend Trait (Pluggable)

Start with `ffi` backend (call MLX via C ABI), then add `cpu` and `metal`.

## TDD Plan

Test pyramid (in increasing cost):

1. Pure Rust unit tests (fast)
2. Property tests (shapes/broadcast fuzz)
3. Golden conformance tests vs Python MLX (authoritative)
4. Gradient checks (slow, tiny sizes)
5. Perf regression (bench harness, not CI-gating)

Recommended early golden op coverage:

- add/mul
- reductions (sum)
- matmul
- softmax

## Migration Phases

- Phase 0: conformance harness
- Phase 1: C ABI shim + `mlx-sys` + minimal safe API
- Phase 2: Rust graph + lazy semantics, backend still FFI
- Phase 3: CPU backend for CI correctness
- Phase 4: autograd MVP
- Phase 5: native Metal backend + hot kernels

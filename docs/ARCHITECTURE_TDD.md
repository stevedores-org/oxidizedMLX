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

## Migration Status & Revised Plan (Ref: #87)

The project has transitioned from foundational API design to functional training. The revised roadmap focuses on memory efficiency and GPU performance.

### ✅ Completed Milestones

- **Phase 0-1 (Foundations):** `mlx-sys` C-ABI shim and `mlx-core` graph IR are stable.
- **Phase 2-3 (Correctness):** `mlx-cpu` provides bit-wise parity for core ops; `mlx-conformance` runs automated tests against Python reference.
- **Phase 4 (Autograd & NN):** `mlx-autograd` (VJP registry), `mlx-nn` (Module/Linear/LayerNorm), and `mlx-optim` (AdamW) are implemented and verified with a 5-step training convergence test.
- **I/O:** `mlx-io` supports `.safetensors` weight loading/saving.

### 🚧 Current Work: Phase 5 (Unified Eval & Metal Consolidation)

The primary bottleneck is the fragmentation of the Metal runtime and the "Eager Host Copy" model in the `Backend` trait.

1.  **Unified Buffer Abstraction:**
    -   Replace `Vec<f32>` in `Backend::eval_node` with a `Buffer` trait/enum.
    -   Support "Zero-copy" transfers between backends where possible (Shared Memory).
    -   Enable `LazyBuffer` residency management (LRU cache for GPU memory).

2.  **Metal Runtime Consolidation:**
    -   Unify the 6+ fragmented Metal PRs (#70, #71, #77, etc.) into a single cohesive runtime.
    -   Implement the `MetalBackend` using the new Unified Buffer abstraction to avoid host-roundtrips during `eval()`.
    -   Prioritize `GEMM` (MatMul) and `Convolution` kernels.

3.  **Graph Optimizations:**
    -   Implement "Operator Fusion" (e.g., `Add` + `Mul` -> `FusedAddMul`).
    -   Memory-efficient derivative calculation (gradient checkpointing).

### 🚀 Future: Phase 6 (Productization & Scale)

-   **Distributed Training:** Basic support for `AllReduce` via NCCL or custom shims.
-   **Stable Diffusion / Llama Verification:** Running full-scale weights through the stack.
-   **Python Bindings:** Exporting the Rust runtime back to Python for use as a high-performance backend.

## Workspace Layout

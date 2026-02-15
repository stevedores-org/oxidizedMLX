# Attack Plan for Issue #100: Remaining Architecture Work

This document outlines the phased approach to completing the remaining work for the oxidizedMLX architecture, as tracked in issue #100.

## Phase 1: Foundation & I/O (High Priority)

**Objective:** Enable loading and saving model weights in the industry-standard Safetensors format.

- **Task 1.1:** Implement `mlx-io` crate.
  - Add `safetensors` and `memmap2` dependencies. [DONE]
  - Implement `load_safetensors` with `mmap` support. [DONE]
  - Implement `save_safetensors`. [DONE]
  - Add robust error handling for file I/O and parsing. [DONE]
- **Task 1.2:** Add unit tests and integration tests with mock tensor data. [DONE]

## Phase 2: Metal Backend Parity (High Priority)

**Objective:** Expand GPU support to cover the necessary operations for LLM inference.

- **Task 2.1:** Implement missing Elementwise kernels.
  - `Sub`, `Mul`, `Div`, `Neg` [DONE]
  - `Exp`, `Log` [PENDING]
- **Task 2.2:** Implement Activation kernels.
  - `Softmax`, `Silu`, `Gelu` [DONE]
- **Task 2.3:** Implement Normalization kernels.
  - `LayerNorm`, `RmsNorm` [DONE]
- **Task 2.4:** Implement Reduction kernels.
  - `Sum`, `Mean`, `Max` [PENDING]
- **Task 2.5:** Shape ops.
  - `Transpose` (support for arbitrary axes) [PENDING - currently only 2D swap]
  - `Broadcast` [PENDING]

## Phase 3: Neural Network Layers (High Priority)

**Objective:** Provide high-level building blocks for model construction.

- **Task 3.1:** Implement `Embedding` layer in `mlx-nn`.
- **Task 3.2:** Implement Multi-head Attention (MHA) / Grouped-query Attention (GQA) components.
- **Task 3.3:** Implement `Dropout` (with training mode support).

## Phase 4: CI, Conformance & DX (Medium Priority)

**Objective:** Automate correctness checks and improve the developer experience.

- **Task 4.1:** Wire up `mlx-conformance` in CI.
  - Activate golden test comparison against Python MLX.
- **Task 4.2:** Implement Learning Rate Schedulers in `mlx-optim`.
  - `StepLR`, `CosineAnnealing`.
- **Task 4.3:** Add multi-agent / concurrency stress tests.

## Phase 5: Advanced Features & Optimization (Lower Priority)

**Objective:** Future-proofing and performance tuning.

- **Task 5.1:** Advanced transforms (`vmap`, `jvp`).
- **Task 5.2:** Operator fusion for Metal.
- **Task 5.3:** Unified Buffer abstraction to replace `Vec<f32>`.
- **Task 5.4:** CPU kernel optimization (SIMD, Rayon).

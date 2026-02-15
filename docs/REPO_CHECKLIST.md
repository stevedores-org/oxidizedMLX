# “MLX in Rust” (oxidizedMLX) Validation Checklist

This checklist validates progress aligned with the Strangler Fig approach and the [Delivery Plan](DELIVERY_PLAN.md).

## 0) Repo baseline and contributor DX (Epic 0 / M0)

* [x] **Fresh clone → `just ci` passes** without `MLX_SRC` (default path excludes FFI). ([GitHub][1])
  * Verification: `just ci`
  * [x] fmt + clippy + tests run on Linux + macOS CI
  * [x] clear “optional FFI path” docs (`MLX_SRC`, `just test-ffi`, `just clippy-ffi`) ([GitHub][1])
* [x] **Repo docs are canonical**
  * [x] `README.md` explains crates + env vars + workflows ([GitHub][1])
  * [x] `docs/DELIVERY_PLAN.md` and `docs/REPO_CHECKLIST.md` stay current ([GitHub][2])
* [x] **Dev tooling**
  * Verification: `just smoke`
  * [x] `just smoke` works and exercises something meaningful (CPU path at minimum) ([GitHub][1])
  * [x] Nix flake builds the workspace (optional but highly desirable) ([GitHub][1])

## 1) “Narrow waist” backend abstraction (Epic 1 / M1)

* [x] **Backend trait contract exists** (dtype/shape/device semantics, memory rules, error model). ([GitHub][2])
  * Verification: `cargo test -p mlx-core`
* [x] **Tensor creation selects backend/device** (explicit, not global hidden state). ([GitHub][2])
* [x] **Minimal ops routed through backend dispatch**
  * [x] add / mul
  * [x] matmul
  * [x] reductions (at least sum)
  * [x] reshape/broadcast plumbing (as needed)
* [x] **Single error strategy** at public boundary (typed errors, not string-only). ([GitHub][2])

## 2) Core tensor API stabilization (Epic 2 / M2)

* [x] **`Tensor` API is stable & safe:**
  * Verification: `cargo test -p mlx-core`
  * [x] `from_slice`, `zeros/ones/full`, `reshape`, `transpose/view` (as defined)
  * [x] `dtype() / shape() / device()` are correct
* [x] **dtype policy is explicit and tested** (f16/bf16/f32 + ints/bool as desired). ([GitHub][3])
* [x] **Broadcasting rules + shape inference are property-tested** (proptest). ([GitHub][3])
  * Verification: `just proptest`

## 3) CPU backend as correctness oracle (Epic 3 / M4)

* [x] **CPU backend implements backend trait end-to-end.** ([GitHub][3])
  * Verification: `cargo test -p mlx-core`
* [x] **Deterministic kernels + deterministic RNG policy for tests.** ([GitHub][3])
* [x] **Coverage:**
  * [x] elementwise ops (core set)
  * [x] reductions with axis handling
  * [x] matmul (naive OK first; add optimized later)
* [x] **Unit tests + property tests validate CPU backend as the oracle.**

## 4) Conformance harness (Rust vs Python MLX) (Epic 4 / M7)

* [x] **Conformance harness exists and runs a fixed suite of ops across shapes/dtypes.** ([GitHub][2])
  * Verification: `just conformance`
* [x] **Reports failures with repro inputs + expected vs got deltas.** ([GitHub][2])
* [x] **CI gating policy:**
  * [x] at least a “core” conformance subset gates PRs
  * [x] full suite may run nightly / optional job

## 5) FFI backend (Upstream MLX via C ABI shim) (Epic 8 / M8)

* [x] **`mlx-sys` builds from `MLX_SRC` and exposes a **minimal, versioned** C ABI.** ([GitHub][1])
  * Verification: `just test-ffi`
* [x] **FFI backend implements the same backend trait behind `ffi` feature flag.** ([GitHub][3])
* [x] **CI job exists for FFI builds (opt-in or restricted).** ([GitHub][3])
* [x] **Safety boundaries are explicit** (no hidden global state assumptions). ([GitHub][2])

## 6) Metal backend program (Epic 9 → your E8 series)

### E8-S1 Metal runtime scaffolding

* [x] Create device/queue; submit command buffers
  * Verification: `cargo test -p mlx-metal`
* [x] Smoke test runs a trivial kernel and returns correct output
  * Verification: `cargo test -p mlx-metal`

### E8-S2 Unified memory buffer model (UMA “no-copy”)

* [x] Shared buffer model supports no-copy where possible
* [ ] Instrumentation proves “no-copy path” (pointer identity / bytes_copied=0)

### E8-S3 Metal GEMM

* [ ] GEMM correctness vs CPU/FFI for a test matrix
* [ ] Benchmark harness exists; reports TFLOPs / time for LLM-ish shapes

### E8-S4 RMSNorm / LayerNorm

* [ ] Golden tests vs reference
* [ ] Perf baseline recorded (µs/row, GB/s style metrics)

### E8-S5 RoPE

* [x] Unit tests + golden tests match reference implementation

### E8-S6 Attention components

* [ ] softmax + masking + matmul compose correctly
* [ ] Functional attention block golden test passes

### E8-S7 Backend parity gate (maintainer control)

* [ ] Parity suite compares metal vs CPU/FFI on defined ops
* [ ] Feature flag controls default backend (metal not default until parity passes)

## 7) Autograd (Epic 5 / M5)

* [x] **Tape/graph representation + reverse topo traversal exists.** ([GitHub][2])
  * Verification: `cargo test -p mlx-autograd`
* [x] **VJP registry for core ops** (add/mul/matmul/sum/reductions). ([GitHub][2])
* [x] **Gradient checks** (finite differences) for representative ops. ([GitHub][2])

## 8) NN modules + optimizers (Epic 6 / M9) and IO (Epic 7 / M9)

* [x] **Parameter / module system + state dict conventions.** ([GitHub][2])
  * Verification: `cargo test -p mlx-nn`
* [x] **Optimizers: SGD + AdamW + schedulers with tests.** ([GitHub][2])
  * Verification: `cargo test -p mlx-optim`
* [x] **safetensors + mmap loader with validation + robustness tests.** ([GitHub][2])
  * Verification: `cargo test -p mlx-io`

## 9) Benchmarks & performance program (Epic 10)

* [ ] **Criterion benches exist for:**
  * Verification: `just bench`
  * [ ] matmul (CPU + metal + ffi)
  * [ ] reductions
  * [ ] normalization
  * [ ] rope / attention microbenches
* [ ] **Perf regressions are tracked** (baseline artifacts/logs per release or per main).

## 10) Packaging & feature flags (Epic 11)

* [ ] **Feature flags are clean and documented:**
  * Verification: `cargo test --features ...`
  * [ ] `cpu` default
  * [ ] `metal`
  * [ ] `ffi`
  * [ ] `conformance` (dev-only)
* [ ] **Examples build in CI; public API has semver plan.** ([GitHub][2])

[1]: https://github.com/stevedores-org/oxidizedMLX "GitHub - stevedores-org/oxidizedMLX: Oxidized MLX"
[2]: https://raw.githubusercontent.com/stevedores-org/oxidizedMLX/main/docs/DELIVERY_PLAN.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/stevedores-org/oxidizedMLX/main/docs/REPO_CHECKLIST.md "raw.githubusercontent.com"

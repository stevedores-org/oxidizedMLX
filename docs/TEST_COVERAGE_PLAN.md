# Rust Test Coverage Plan — oxidizedMLX

> Comprehensive plan to bring the Rust codebase from ~78 tests to full coverage.
> Organized as **Epics → User Stories → Acceptance Criteria**.

---

## Current State

| Crate | Existing Unit Tests | Existing Integration Tests | Estimated Coverage |
|---|---|---|---|
| `mlx-core::types` | 3 | — | ~38% |
| `mlx-core::graph` | 1 | — | ~20% |
| `mlx-core::backend` | 2 | — | ~29% |
| `mlx-core::tensor` | 21 | — | ~54% |
| `mlx-core::cpu_kernels` | 8 | — | ~56% |
| `mlx-ops::broadcast` | 6 | — | ~67% |
| `mlx-ops::dtype_promotion` | 6 (incl. 1 symmetry) | — | ~67% |
| `mlx-ops::shape_inference` | 13 | — | ~80% |
| `mlx-conformance` | 3 (utility) | 15 (golden) | ~67% |
| `mlx-cpu` | 0 | — | 0% |
| `mlx-cli` | 0 | — | 0% |
| `mlx-roadmap` | 0 | — | 0% |
| **Total** | **63** | **15** | **~50%** |

### Key Gaps Identified

1. **No error-path tests** — Shape mismatches, invalid axes, missing inputs are validated in code but not exercised in tests
2. **No tests for Mean/Max reductions** — `reduce_mean` and `reduce_max` CPU kernels have zero test coverage
3. **No GELU test** — Polynomial approximation in `cpu_kernels.rs:46-58` is completely untested
4. **No RMS norm test** — `rms_norm` kernel (`cpu_kernels.rs:305-327`) has no unit test
5. **Graph module barely tested** — Only 1 test for `topo_sort`; `get()`, `len()`, `is_empty()`, `add_node()` untested directly
6. **Backend eval() paths untested** — Complex lock management and error handling in `backend.rs:77-150` not covered
7. **No operator trait tests** for `Sub`, `Mul` (only `Add` and `Neg` operator overloads tested)
8. **No property-based tests** — `proptest` is a dev-dependency in `mlx-ops` but never used
9. **Zero tests for `mlx-cpu`, `mlx-cli`, `mlx-roadmap`**
10. **No Display/Debug trait output tests** — `DType::Display`, `Shape::Display` never asserted

---

## Epic 1: mlx-core — Types & Foundational Structs

**Goal:** Achieve >95% coverage of `types.rs` and derived trait behavior.

### Story 1.1: Shape edge cases
**File:** `crates/mlx-core/src/types.rs`

- [ ] Test `Shape::scalar()` returns ndim=0 and numel=1
- [ ] Test `Shape::new(vec![])` behaves like scalar
- [ ] Test `Shape::new(vec![1])` — single-element dimension
- [ ] Test `Shape::dim()` returns `None` for out-of-range positive axis
- [ ] Test `Shape::dim()` returns `None` for out-of-range negative axis (e.g. `dim(-4)` on rank-3)
- [ ] Test `Shape::numel()` on shape with a zero dimension (e.g. `[0, 5]` → 0)
- [ ] Test `Shape` equality: identical shapes are equal, different shapes are not

### Story 1.2: Display trait formatting
**File:** `crates/mlx-core/src/types.rs`

- [ ] Assert `format!("{}", DType::F32)` == `"f32"` (and all other variants)
- [ ] Assert `format!("{}", Shape::new(vec![2,3]))` == `"[2, 3]"`
- [ ] Assert `format!("{}", Shape::scalar())` == `"[]"`
- [ ] Assert `format!("{}", Shape::new(vec![1]))` == `"[1]"`

### Story 1.3: DType size_bytes exhaustive
**File:** `crates/mlx-core/src/types.rs`

- [ ] Verify `BF16.size_bytes() == 2`
- [ ] Verify `I32.size_bytes() == 4`
- [ ] (Existing tests cover F32, F16, I64 — add BF16 and I32)

### Story 1.4: MlxError Display formatting
**File:** `crates/mlx-core/src/lib.rs`

- [ ] Assert `MlxError::NullPtr.to_string()` contains "null pointer"
- [ ] Assert `MlxError::FfiFailed("xyz").to_string()` contains "xyz"
- [ ] Assert `MlxError::ShapeMismatch` message contains both expected and got
- [ ] Assert `MlxError::InvalidArgument` message contains the argument string
- [ ] Assert `MlxError::BackendUnavailable` message contains the backend name

---

## Epic 2: mlx-core — Computation Graph

**Goal:** Achieve >90% coverage of `graph.rs`. The graph is the foundation of lazy evaluation and must be bulletproof.

### Story 2.1: Graph construction and lookup
**File:** `crates/mlx-core/src/graph.rs`

- [ ] Test `Graph::new()` starts empty (`len() == 0`, `is_empty() == true`)
- [ ] Test `add_node` returns monotonically increasing `NodeId`s
- [ ] Test `get()` returns `Some` for added nodes and `None` for unknown IDs
- [ ] Test `len()` increments with each `add_node` call
- [ ] Test `is_empty()` returns false after first node added

### Story 2.2: Topological sort correctness
**File:** `crates/mlx-core/src/graph.rs`

- [ ] Test topo_sort on a linear chain A→B→C
- [ ] Test topo_sort on a diamond graph: A→C, B→C, C→D (A,B before C before D)
- [ ] Test topo_sort with multiple roots (outputs = [D, E] where D and E share a common subgraph)
- [ ] Test topo_sort of a single constant node (no inputs)
- [ ] Test topo_sort handles the same node requested twice in outputs (deduplication)

### Story 2.3: Node metadata integrity
**File:** `crates/mlx-core/src/graph.rs`

- [ ] Test that `get(id).unwrap().op` matches the `OpKind` passed to `add_node`
- [ ] Test that `get(id).unwrap().inputs` matches the inputs passed
- [ ] Test that `get(id).unwrap().meta.shape` and `.dtype` match what was provided

---

## Epic 3: mlx-core — Backend & Stream

**Goal:** Achieve >85% coverage of `backend.rs`. This is the evaluation engine.

### Story 3.1: Stream constant management
**File:** `crates/mlx-core/src/backend.rs`

- [ ] Test `add_constant` stores data retrievable via `get_buffer` (even before eval)
- [ ] Test `eval` on an already-materialized constant is a no-op (idempotent)
- [ ] Test `get_buffer` returns `None` for a non-existent node ID

### Story 3.2: Stream eval scheduling
**File:** `crates/mlx-core/src/backend.rs`

- [ ] Test eval of a 3-node chain: const → neg → sum_all
- [ ] Test eval of a diamond graph: two constants → add → result
- [ ] Test that calling `eval` twice on the same output doesn't re-evaluate (idempotent)
- [ ] Test that multiple independent subgraphs can coexist in the same stream

### Story 3.3: Stream error paths
**File:** `crates/mlx-core/src/backend.rs`

- [ ] Test that evaluating a `Constant` op node (not pre-materialized) returns `InvalidArgument` error
- [ ] Test `get_buffer` on an unevaluated op node returns `None`

---

## Epic 4: mlx-core — Tensor API Error Paths

**Goal:** Every `Result`-returning method in `tensor.rs` must have its error path tested.

### Story 4.1: Binary op shape mismatch errors
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `add()` with mismatched shapes returns `ShapeMismatch`
- [ ] Test `sub()` with mismatched shapes returns `ShapeMismatch`
- [ ] Test `mul()` with mismatched shapes returns `ShapeMismatch`
- [ ] Test `div()` with mismatched shapes returns `ShapeMismatch`

### Story 4.2: Matmul validation errors
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `matmul()` with 1D tensor returns `InvalidArgument` ("matmul requires 2D")
- [ ] Test `matmul()` with 3D tensor returns `InvalidArgument`
- [ ] Test `matmul()` with inner dimension mismatch returns `ShapeMismatch`

### Story 4.3: Reduction axis validation
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `sum_axis()` with axis out of range (positive) returns `InvalidArgument`
- [ ] Test `sum_axis()` with axis out of range (negative, e.g. -3 on 2D) returns `InvalidArgument`
- [ ] Test `sum_axis()` with negative axis resolves correctly (e.g. -1 on [2,3] → axis 1)
- [ ] Test `sum_axis()` on 1D tensor reducing axis 0 produces shape [1]

### Story 4.4: Softmax/Transpose validation
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `softmax()` with out-of-range axis returns `InvalidArgument`
- [ ] Test `softmax()` with negative axis resolves correctly
- [ ] Test `transpose()` with wrong-length axes vec returns `InvalidArgument`
- [ ] Test `transpose()` with custom permutation returns correct shape and data

### Story 4.5: Reshape validation
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `reshape()` with incompatible numel returns `ShapeMismatch`
- [ ] Test `reshape()` from [6] to [2,3] and back

### Story 4.6: from_f32 validation
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `from_f32()` with empty data and empty shape succeeds
- [ ] Test `from_f32()` with data length > shape numel returns `InvalidArgument`
- [ ] Test `from_f32()` with data length < shape numel returns `InvalidArgument`

---

## Epic 5: mlx-core — CPU Kernels (Missing Operations)

**Goal:** Every `OpKind` variant handled in `cpu_kernels.rs` must have at least one dedicated test.

### Story 5.1: Mean reduction
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `reduce_mean` with `axis: None` (all-reduce) on [1, 2, 3] → [2.0]
- [ ] Test `reduce_mean` with `axis: Some(0)` on shape [2, 3]
- [ ] Test `reduce_mean` with `axis: Some(1)` on shape [2, 3]
- [ ] Test `reduce_mean` with `axis: Some(-1)` (negative axis)

### Story 5.2: Max reduction
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `reduce_max` with `axis: None` (all-reduce) — include negative values
- [ ] Test `reduce_max` with `axis: Some(0)` on shape [2, 3]
- [ ] Test `reduce_max` with `axis: Some(1)` on shape [2, 3]
- [ ] Test `reduce_max` on data with all identical values

### Story 5.3: GELU activation
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `gelu(0.0)` ≈ 0.0
- [ ] Test `gelu(1.0)` ≈ 0.8412 (known value)
- [ ] Test `gelu(-1.0)` ≈ -0.1588 (known value)
- [ ] Test GELU is approximately `x * Φ(x)` for a range of values

### Story 5.4: RMS norm
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `rms_norm` on 1D input — output RMS should be ≈1.0 (normalized)
- [ ] Test `rms_norm` on 2D input (batch dimension) — each row normalized independently
- [ ] Test `rms_norm` with different epsilon values (1e-5, 1e-8)
- [ ] Test `rms_norm` on constant input (all same values)

### Story 5.5: Sub and Div kernels
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `Sub` kernel directly via `eval_node` (currently untested at kernel level)
- [ ] Test `Div` kernel directly via `eval_node` (currently untested at kernel level)
- [ ] Test `Div` with divisor containing zeros (should produce Inf/NaN)

### Story 5.6: Transpose edge cases
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test transpose on 3D tensor with default (reversed) axes
- [ ] Test transpose on 3D tensor with custom permutation [1, 2, 0]
- [ ] Test transpose on 1D tensor (no-op)
- [ ] Test transpose with identity permutation [0, 1, 2] (should be no-op)

### Story 5.7: Kernel error paths
**File:** `crates/mlx-core/src/cpu_kernels.rs`

- [ ] Test `eval_node(Constant, ...)` returns error (constants should be pre-materialized)
- [ ] Test `eval_node(Parameter, ...)` returns error
- [ ] Test binary op with mismatched input lengths returns `ShapeMismatch`
- [ ] Test `matmul` with non-2D inputs returns error
- [ ] Test `matmul` with inner dimension mismatch returns error
- [ ] Test `reduce_along_axis` with out-of-range axis returns error
- [ ] Test `softmax` with out-of-range axis returns error
- [ ] Test `transpose` with wrong-length axes returns error

---

## Epic 6: mlx-core — Tensor Operator Overloads

**Goal:** All `std::ops` trait implementations tested.

### Story 6.1: Operator trait coverage
**File:** `crates/mlx-core/src/tensor.rs`

- [ ] Test `&a - &b` (ops::Sub) produces correct result
- [ ] Test `&a * &b` (ops::Mul) produces correct result
- [ ] Test `&a - &b` with mismatched shapes returns Err
- [ ] Test `&a * &b` with mismatched shapes returns Err
- [ ] Test operator chaining: `(&(&a + &b)? * &c)?`

---

## Epic 7: mlx-ops — Broadcasting, Shape Inference, DType Promotion

**Goal:** >95% coverage with property-based tests to catch edge cases.

### Story 7.1: Property-based broadcasting tests (proptest)
**File:** `crates/mlx-ops/src/broadcast.rs`

- [ ] Property: `broadcast_shapes(a, a) == Some(a)` for all valid shapes
- [ ] Property: `broadcast_shapes(a, b) == broadcast_shapes(b, a)` (commutativity)
- [ ] Property: broadcasting with scalar always succeeds
- [ ] Property: result shape ndim >= max(a.ndim, b.ndim)
- [ ] Test 4D broadcasting (e.g. batch + channel dimensions)
- [ ] Test broadcasting with shape containing 0 dimension

### Story 7.2: Shape inference — missing coverage
**File:** `crates/mlx-ops/src/shape_inference.rs`

- [ ] Test `infer_shape` for `Mean` reduction (axis and all-reduce variants)
- [ ] Test `infer_shape` for `Max` reduction (axis and all-reduce variants)
- [ ] Test `infer_shape` for `Gelu` (unary, preserves shape)
- [ ] Test `infer_shape` for `RmsNorm` (preserves shape)
- [ ] Test `infer_shape` for `Constant` and `Parameter` (preserve shape)
- [ ] Test `infer_shape` with missing inputs (0 inputs for binary op → error)
- [ ] Test `infer_shape` with invalid axis on Softmax → `InvalidAxis` error
- [ ] Test `infer_shape` with axis on Sum equal to ndim → `InvalidAxis` error

### Story 7.3: DType promotion edge cases
**File:** `crates/mlx-ops/src/dtype_promotion.rs`

- [ ] Test `is_integer` for all integer types (I32, I64 → true; rest → false)
- [ ] Test promotion transitivity: `promote(promote(a,b), c) == promote(a, promote(b,c))`
- [ ] Property-based test: promote always returns one of the two inputs
- [ ] Property-based test: promote result priority >= both input priorities

---

## Epic 8: mlx-conformance — Testing Infrastructure & Golden Tests

**Goal:** Harden the test infrastructure and expand golden test coverage.

### Story 8.1: assert_allclose edge cases
**File:** `crates/mlx-conformance/src/lib.rs`

- [ ] Test `assert_allclose` with empty slices (should pass)
- [ ] Test `assert_allclose` with NaN values (should fail — NaN != NaN)
- [ ] Test `assert_allclose` with Inf values (matching Infs should pass)
- [ ] Test `assert_allclose` with length mismatch panics with correct message

### Story 8.2: Expand golden conformance tests
**File:** `crates/mlx-conformance/tests/basic.rs`

- [ ] Add `golden_gelu` — validate GELU against known reference values
- [ ] Add `golden_rms_norm` — validate RMS normalization
- [ ] Add `golden_mean_axis` — mean reduction with axis
- [ ] Add `golden_mean_all` — mean all-reduce
- [ ] Add `golden_max_axis` — max reduction with axis
- [ ] Add `golden_max_all` — max all-reduce
- [ ] Add `golden_transpose_3d` — 3D tensor transposition
- [ ] Add `golden_reshape_higher_rank` — reshape to higher rank

### Story 8.3: Serialization round-trip tests
**File:** `crates/mlx-conformance/src/lib.rs`

- [ ] Test `OpSpec` serializes to valid JSON with expected fields
- [ ] Test `OpSpec` with optional `b: None` omits the field (`skip_serializing_if`)
- [ ] Test `TensorSpec` serializes shape and data correctly
- [ ] Test `OpOutput` deserializes from JSON string

---

## Epic 9: mlx-cpu — CPU Backend Crate

**Goal:** Test the public API surface of `mlx-cpu`.

### Story 9.1: Re-export and stream creation
**File:** `crates/mlx-cpu/src/lib.rs`

- [ ] Test `cpu_stream()` returns a working stream
- [ ] Test that `cpu_stream()` can be used to add constants and evaluate them
- [ ] Test that `CpuRefBackend` is re-exported and implements `Backend`

---

## Epic 10: mlx-cli — Smoke Test Binary

**Goal:** Ensure the CLI runs without panic.

### Story 10.1: CLI smoke subcommand
**File:** `crates/mlx-cli/src/main.rs`

- [ ] Integration test: run the binary with `smoke` subcommand and assert exit code 0
- [ ] Integration test: assert stdout contains "All smoke tests passed."
- [ ] Test: run with no subcommand → non-zero exit (clap missing subcommand)
- [ ] Test: run with invalid subcommand → non-zero exit

---

## Epic 11: mlx-roadmap — Repo Automation

**Goal:** Unit test the pure logic functions without requiring GitHub API access.

### Story 11.1: Repo slug parsing
**File:** `crates/mlx-roadmap/src/main.rs`

- [ ] Test `parse_repo_slug("owner/repo")` → `Ok(Repo { owner: "owner", name: "repo" })`
- [ ] Test `parse_repo_slug("owner/")` → `Err` (empty name)
- [ ] Test `parse_repo_slug("/repo")` → `Err` (empty owner)
- [ ] Test `parse_repo_slug("noslash")` → `Err` (missing separator)
- [ ] Test `parse_repo_slug("a/b/c")` → `Err` (too many parts)

### Story 11.2: Plan generation
**File:** `crates/mlx-roadmap/src/main.rs`

- [ ] Test `plan()` returns non-empty labels, milestones, and epic issues
- [ ] Test all labels have non-empty name, color (6 hex chars), and description
- [ ] Test milestone count matches expected (10)
- [ ] Test each epic issue has a non-empty title and body

### Story 11.3: Transient error detection
**File:** `crates/mlx-roadmap/src/main.rs`

- [ ] Test `is_transient_gh_error` returns true for "error connecting to api.github.com"
- [ ] Test `is_transient_gh_error` returns true for "check your internet connection"
- [ ] Test `is_transient_gh_error` returns true for "githubstatus.com"
- [ ] Test `is_transient_gh_error` returns false for "404 Not Found"

---

## Epic 12: Numerical Correctness & Stress Tests

**Goal:** Validate mathematical correctness beyond simple happy-path cases.

### Story 12.1: Floating-point edge cases
- [ ] Test all elementwise ops with `f32::INFINITY`, `f32::NEG_INFINITY`, `f32::NAN`
- [ ] Test softmax numerical stability with large values (e.g. [1000.0, 1001.0, 1002.0])
- [ ] Test layer_norm with constant input (variance → 0, eps prevents division by zero)
- [ ] Test division by zero produces `Inf` (not panic)
- [ ] Test GELU with large positive and large negative values

### Story 12.2: Higher-dimensional tensors
- [ ] Test sum reduction on 3D tensor [2, 3, 4] along each axis (0, 1, 2)
- [ ] Test softmax on 2D tensor along axis 0 and axis 1
- [ ] Test layer_norm on 3D tensor [batch, seq, hidden]
- [ ] Test transpose on 4D tensor

### Story 12.3: Single-element and degenerate shapes
- [ ] Test all ops on shape [1] (single element)
- [ ] Test sum_all on scalar-like shape [1]
- [ ] Test reshape from [1,1,1] to [1]
- [ ] Test matmul on [1,1] @ [1,1]

---

## Epic 13: Property-Based Testing Infrastructure

**Goal:** Add proptest strategies and use them across crates.

### Story 13.1: Create shared test strategies
**Location:** New file `crates/mlx-ops/tests/proptest_strategies.rs` or as a test module

- [ ] `arb_shape()` — generates valid shapes (rank 0-4, dims 1-8)
- [ ] `arb_dtype()` — generates any `DType` variant
- [ ] `arb_broadcastable_pair()` — generates two shapes guaranteed to be broadcastable
- [ ] `arb_tensor_data(shape)` — generates f32 data matching a shape

### Story 13.2: Property tests for broadcasting
**File:** `crates/mlx-ops/src/broadcast.rs`

- [ ] Commutativity: `broadcast(a, b) == broadcast(b, a)`
- [ ] Associativity with three shapes
- [ ] Result numel >= max(a.numel, b.numel)

### Story 13.3: Property tests for dtype promotion
**File:** `crates/mlx-ops/src/dtype_promotion.rs`

- [ ] Idempotence: `promote(a, a) == a`
- [ ] Commutativity (already tested, migrate to proptest)
- [ ] Result is always one of the two inputs

---

## Priority & Ordering

| Priority | Epic | Rationale |
|---|---|---|
| **P0** | Epic 5 (CPU Kernels — Missing Ops) | Mean, Max, GELU, RMS norm have zero coverage in the correctness oracle |
| **P0** | Epic 4 (Tensor Error Paths) | Error paths are the #1 source of untested code in the public API |
| **P0** | Epic 2 (Graph) | Foundation of lazy eval; only 1 test today |
| **P1** | Epic 3 (Backend & Stream) | Complex lock management, error handling |
| **P1** | Epic 7 (mlx-ops proptest) | Catch edge cases in broadcasting/shape inference |
| **P1** | Epic 8 (Conformance expansion) | Golden tests for newly-covered ops |
| **P1** | Epic 12 (Numerical correctness) | Floating-point edge cases can cause silent bugs |
| **P2** | Epic 1 (Types) | Low risk but easy wins |
| **P2** | Epic 6 (Operator overloads) | Small gap, quick to close |
| **P2** | Epic 9 (mlx-cpu) | Thin wrapper, low risk |
| **P2** | Epic 13 (Proptest infra) | Enables ongoing quality |
| **P3** | Epic 10 (CLI) | Manual smoke test exists |
| **P3** | Epic 11 (Roadmap) | GitHub automation, not core logic |

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Total test functions | 78 | ~250+ |
| Crates with zero tests | 3 (mlx-cpu, mlx-cli, mlx-roadmap) | 0 |
| OpKind variants with kernel tests | 10/18 | 18/18 |
| Error paths tested | ~2 | ~25+ |
| Property-based test suites | 0 | 3+ |
| Public methods with no test | ~15 | 0 |

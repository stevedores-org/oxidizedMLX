# Agent Brief: Oxidizing MLX into a Rust-Metal Stack

## Mission

Convert Apple's MLX framework from a C++-controlled runtime into a Rust-first machine learning system with:

- safe lazy tensors
- Rust-owned graph + scheduler
- gradual backend replacement
- eventual native Metal kernel dispatch
- support for next-gen architectures (MoE, DeltaNet)

Do this incrementally ("Ship of Theseus"), never breaking correctness.

## Core Constraints (Non-Negotiable)

### MLX Semantics Must Hold

- Lazy evaluation (ops build graph, don't compute)
- Unified memory behavior (zero-copy where possible)
- Correct broadcasting + dtype rules
- Multi-device CPU/GPU execution

### Safety Goals

- No segfaults, no UB
- No use-after-free
- No CPU mutation while GPU borrows buffer

### Migration Rule

Never rewrite everything at once.
Always keep a fallback backend.

## System Architecture (Agent Mental Model)

### Layer 1: Rust Control Plane (Always Owned by Rust)

- Tensor API
- Lazy graph arena
- Scheduler + topo sort
- Optimization passes (CSE, DCE, fusion)
- Autograd tape + VJP registry

### Layer 2: Pluggable Compute Backends

Backend progression:

1. **FFI backend** → calls MLX C++ kernels
2. **CPU backend** → correctness oracle
3. **Metal backend** → native Rust GPU runtime
4. **Advanced kernels** → MoE + Scan primitives

## Execution Roadmap (Agent Plan)

### EPIC 1 — "Iron Bridge": FFI + Conformance Harness

**Goal:** Rust can call MLX safely and prove correctness.

**Agent Tasks:**

- Build mlx-sys with stable C ABI shim
- Wrap pointers in RAII handles (Drop)
- Ensure invalid ops return Result::Err
- Build golden reference test runner:
  - Python MLX generates outputs
  - Rust compares with allclose

**Success Criteria:**

- 90% of core ops covered by golden tests
- No panics or segfaults
- Memory stable after 1M tensor allocations

### EPIC 2 — "Oxidized Graph": Rust Lazy Graph Runtime

**Goal:** Stop using C++ for graph construction. Rust owns graph semantics.

**Key Design:**

Arena graph:
- `Vec<Node>`
- Node references = `NodeId(u32)`
- No Rc cycles, no pointer chasing

**Agent Tasks:**

- Implement Tensor as graph handle
- Operations append nodes lazily
- `eval()` triggers topo schedule (Kahn)
- Add optimization passes:
  - Dead Code Elimination
  - Common Subexpression Elimination
  - Operator Fusion (later)

**Success Criteria:**

- Graph builds instantly without compute
- `eval()` executes correct dependency order
- Dangling branches never dispatch kernels

### EPIC 3 — "Metal Heart": Native Rust GPU Backend

**Goal:** Replace C++ execution engine with Rust+Metal.

**Critical Apple Constraint:**

Unified Memory ≠ automatic safety.

Must enforce:
- GPU borrows buffer
- CPU blocked until completion
- Use MTLSharedEvent / fences

**Agent Tasks:**

- Implement Metal runtime:
  - pipeline cache
  - command buffers
  - dispatch encoder
- Port core kernels:
  - add
  - mul
  - matmul (tiled + simdgroup)
- Default buffers = MTLStorageModeShared
- Add async evaluation:
  - `eval().await`

**Success Criteria:**

- No memcpy for tensor readback
- Native kernels match CPU golden outputs
- Multiple async evals queue correctly

### EPIC 4 — "Qwen3-Next Payoff": Advanced Architectures

**Goal:** Prove Rust stack enables modern model primitives.

#### Feature A: Gated DeltaNet Scan Primitive

**Problem:** Sequential recurrence must become parallel GPU scan.

**Agent Tasks:**
- Implement chunked scan kernel
- Expose `Scan(op, axis)` node
- Verify against CPU sequential reference

**Success:** 10× speedup at long context

#### Feature B: Sparse MoE Routing

**Problem:** Dynamic token→expert dispatch without sync blowup.

**Agent Tasks:**
- Top-K router kernel (values + indices)
- Scatter/Gather permute tokens
- Batch expert dispatch
- Use indirect command buffers

**Success:** Efficient MoE inference without expert imbalance collapse

#### Feature C: Reverse-Mode Autograd

**Agent Tasks:**
- Maintain arena tape
- Traverse reverse topo
- VJP registry per op
- Gradient checks via finite difference

**Success:** `grad(custom_op)` works safely + correctly

## Testing Strategy (Agent Must Obsess Over)

### Pyramid

1. Unit tests (fast)
2. Property fuzz tests (proptest)
3. Golden conformance vs Python MLX
4. Gradient finite difference checks
5. Bench regression (non-gating)

## Key Engineering Invariants

Agent must enforce:

- Graph nodes are immutable once inserted
- Tensor handles are cheap (`NodeId + Arc<Context>`)
- No backend can mutate buffer without borrow rules
- All unsafe code isolated in `mlx-sys` + `mlx-metal`
- CPU backend always exists as correctness oracle

## Deliverables Timeline (Agent Scheduling)

| Timeframe | Deliverable |
|-----------|-------------|
| Month 1–2 | Golden tests + safe FFI wrapper |
| Month 3–4 | Rust lazy graph + scheduler + CSE |
| Month 5–6 | Metal backend + add/mul/matmul |
| Month 7–8 | Scan + MoE + Autograd engine |

## Agent Next Actions (Concrete)

If I were your agent right now, I would do:

1. Finish Epic 1 conformance harness
2. Expand FFI shim to cover reshape/transpose
3. Lock down graph CSE + DCE passes
4. Start Metal backend with Add kernel only
5. Benchmark matmul vs MLX baseline

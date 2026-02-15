# Backend Parity Suite Matrix

Metal backend may not become default until all REQUIRED ops are green.

Legend:
- CPU = Rust reference backend
- FFI = MLX upstream backend (ground truth)
- Metal = Native Apple GPU backend
- Tests:
  - U = Unit tests
  - P = Property tests (proptest)
  - G = Golden vs Python MLX
  - R = Parity vs CPU/FFI
  - B = Bench baseline

---

## Required Ops Before Metal Default

| Op Category     | Operation        | CPU | FFI | Metal | Required Tests        | Default Gate |
|-----------------|------------------|-----|-----|-------|-----------------------|--------------|
| Elementwise     | Add              | ✅  | ✅  | ✅    | U + R                 | ✅ Required  |
| Elementwise     | Sub              | ✅  | ✅  | ✅    | U + R                 | ✅ Required  |
| Elementwise     | Mul              | ✅  | ✅  | ✅    | U + R                 | ✅ Required  |
| Elementwise     | Div              | ✅  | ✅  | ✅    | U + R                 | ✅ Required  |
| Elementwise     | Neg              | ✅  | ✅  | ✅    | U + R                 | ✅ Required  |
| GEMM            | MatMul           | ✅  | ✅  | ✅    | U + G + R + B         | ✅ Required  |
| Activations     | Silu             | ✅  | ✅  | ✅    | U + G + R             | ✅ Required  |
| Activations     | Gelu             | ✅  | ✅  | ✅    | U + G + R             | ✅ Required  |
| Softmax         | Softmax(axis=-1) | ✅  | ✅  | ✅    | U + G + R             | ✅ Required  |
| Masking         | Causal Mask      | ✅  | ✅  | ✅    | U + G                 | ✅ Required  |
| Norm            | RMSNorm          | ✅  | ✅  | ✅    | U + G + R + B         | ✅ Required  |
| Norm            | LayerNorm        | ✅  | ✅  | ✅    | U + G + R + B         | ✅ Required  |
| Positional      | RoPE             | ✅  | ✅  | ✅    | U + G + R             | ✅ Required  |
| Reduction       | Sum              | ✅  | ✅  | ⬜    | U + P + R             | ⚠ Optional   |
| Reduction       | Mean             | ✅  | ✅  | ⬜    | U + P + R             | ⚠ Optional   |
| Reduction       | Max              | ✅  | ✅  | ⬜    | U + P + R             | ⚠ Optional   |
| Shape Ops       | Reshape          | ✅  | ✅  | ⬜    | U + P                 | ⚠ Optional   |
| Shape Ops       | Transpose        | ✅  | ✅  | ✅    | U + P                 | ⚠ Optional   |
| Shape Ops       | Broadcast        | ✅  | ✅  | ⬜    | U + P                 | ⚠ Optional   |
| Attention       | SDPA Block       | ✅  | ✅  | ✅    | Golden Functional Test| ✅ Required  |

---

## Metal Default Flip Criteria

Metal backend may become default when:

- All REQUIRED ops above are implemented in Metal
- Parity suite passes on Apple Silicon runner:

```bash
cargo test -p mlx-parity --features default-backend-metal
```

- Golden attention block test passes:

```bash
cargo test -p mlx-metal attention_block_goldens
```

- Feature flag is flipped in release builds:

```toml
[features]
default = ["default-backend-metal"]
```

Until then, Metal remains opt-in:

```bash
cargo test --features metal
MLX_RS_BACKEND=metal
```

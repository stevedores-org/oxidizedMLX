#!/usr/bin/env python3
"""Generate golden test data for Metal RoPE kernel validation.

Writes .bin files (raw f32 little-endian) and meta.txt for each test case.

Usage:
    python3 tools/py_ref_runner/gen_rope_goldens.py
"""

import os
import struct
import numpy as np


def rope_interleaved(
    x: np.ndarray,
    rotary_dim: int,
    pos_offset: int,
    theta: float = 10000.0,
) -> np.ndarray:
    """Reference RoPE: interleaved layout on [tokens, head_dim]."""
    tokens, head_dim = x.shape
    out = x.copy()
    assert rotary_dim % 2 == 0
    assert rotary_dim <= head_dim
    for t in range(tokens):
        pos = pos_offset + t
        for i in range(rotary_dim // 2):
            inv_freq = theta ** (-2.0 * i / rotary_dim)
            angle = pos * inv_freq
            c = np.cos(angle)
            s = np.sin(angle)
            x0 = float(out[t, 2 * i])
            x1 = float(out[t, 2 * i + 1])
            out[t, 2 * i] = x0 * c - x1 * s
            out[t, 2 * i + 1] = x0 * s + x1 * c
    return out


def write_f32_bin(path: str, arr: np.ndarray) -> None:
    arr.astype(np.float32).tofile(path)


def gen_case(
    out_dir: str,
    tokens: int,
    head_dim: int,
    rotary_dim: int,
    pos_offset: int,
    seed: int = 0,
    theta: float = 10000.0,
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(tokens, head_dim)).astype(np.float32)
    y = rope_interleaved(x, rotary_dim, pos_offset, theta).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    write_f32_bin(os.path.join(out_dir, "x.bin"), x)
    write_f32_bin(os.path.join(out_dir, "y.bin"), y)
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write(
            f"tokens={tokens}\n"
            f"head_dim={head_dim}\n"
            f"rotary_dim={rotary_dim}\n"
            f"pos_offset={pos_offset}\n"
            f"theta={theta}\n"
            f"seed={seed}\n"
        )
    print(f"  {out_dir}: [{tokens}, {head_dim}] rotary_dim={rotary_dim}")


if __name__ == "__main__":
    base = os.path.join("tools", "goldens", "rope")
    print("Generating RoPE golden test data:")
    gen_case(
        os.path.join(base, "case_small"),
        tokens=4, head_dim=16, rotary_dim=16, pos_offset=0, seed=0,
    )
    gen_case(
        os.path.join(base, "case_partial"),
        tokens=4, head_dim=16, rotary_dim=8, pos_offset=0, seed=42,
    )
    gen_case(
        os.path.join(base, "case_llm"),
        tokens=128, head_dim=128, rotary_dim=128, pos_offset=100, seed=1,
    )
    print("Done.")

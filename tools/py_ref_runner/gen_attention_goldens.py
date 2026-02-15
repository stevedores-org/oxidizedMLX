#!/usr/bin/env python3
"""Generate attention reference outputs using NumPy.

This script is for documentation and debugging only; it is NOT required for CI.
The Rust golden tests compute CPU reference values inline.

Usage:
    python gen_attention_goldens.py
"""

import numpy as np


def scaled_masked_softmax(
    scores: np.ndarray, scale: float, causal: bool
) -> np.ndarray:
    """Fused scale + causal-mask + softmax along last axis."""
    tq, tk = scores.shape
    s = scores * scale
    if causal:
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=1)
        s = np.where(mask, -1e9, s)
    # Numerically stable softmax
    s_max = s.max(axis=-1, keepdims=True)
    e = np.exp(s - s_max)
    return e / e.sum(axis=-1, keepdims=True)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float, causal: bool
) -> np.ndarray:
    """Single-head scaled dot-product attention."""
    scores = q @ k.T  # [Tq, Tk]
    probs = scaled_masked_softmax(scores, scale, causal)
    return probs @ v  # [Tq, Dh]


def gen_data(n: int, seed: float) -> np.ndarray:
    """Same deterministic data as the Rust tests."""
    return np.array([np.sin((i + seed) * 0.01) * 0.5 for i in range(n)], dtype=np.float32)


def main():
    configs = [
        {"tq": 4, "tk": 4, "dh": 16, "causal": True, "name": "small"},
        {"tq": 32, "tk": 32, "dh": 64, "causal": True, "name": "medium"},
        {"tq": 8, "tk": 16, "dh": 32, "causal": True, "name": "asymmetric"},
        {"tq": 4, "tk": 4, "dh": 16, "causal": False, "name": "no_mask"},
    ]

    for cfg in configs:
        tq, tk, dh = cfg["tq"], cfg["tk"], cfg["dh"]
        causal = cfg["causal"]
        scale = 1.0 / np.sqrt(dh)

        q = gen_data(tq * dh, 0.0).reshape(tq, dh)
        k = gen_data(tk * dh, 100.0).reshape(tk, dh)
        v = gen_data(tk * dh, 200.0).reshape(tk, dh)

        y = attention(q, k, v, scale, causal)

        print(f"=== {cfg['name']} (Tq={tq}, Tk={tk}, Dh={dh}, causal={causal}) ===")
        print(f"  output shape: {y.shape}")
        print(f"  output[:4]: {y.flatten()[:4]}")
        print(f"  output sum:  {y.sum():.6f}")
        print()


if __name__ == "__main__":
    main()

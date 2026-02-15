"""Python MLX reference runner for golden conformance tests.

Usage:
    python tools/py_ref_runner/run.py input.json output.json

Input JSON spec:
    {
        "op": "add",
        "seed": 42,
        "a": {"shape": [2, 3], "data": [1.0, 2.0, ...]},
        "b": {"shape": [2, 3], "data": [4.0, 5.0, ...]}  // optional
    }

Output JSON:
    {"out": [5.0, 7.0, ...]}
"""

import json
import sys

import mlx.core as mx
import numpy as np


def main(inp_path: str, out_path: str) -> None:
    with open(inp_path, "r") as f:
        spec = json.load(f)

    op = spec["op"]
    np.random.seed(spec.get("seed", 42))

    def to_mx(x: dict) -> mx.array:
        a = np.array(x["data"], dtype=np.float32).reshape(x["shape"])
        return mx.array(a)

    a = to_mx(spec["a"])
    b = to_mx(spec["b"]) if "b" in spec else None

    ops = {
        "add": lambda: a + b,
        "mul": lambda: a * b,
        "matmul": lambda: a @ b,
        "sum0": lambda: mx.sum(a, axis=0),
        "sum1": lambda: mx.sum(a, axis=1),
        "sum_all": lambda: mx.sum(a),
        "softmax": lambda: mx.softmax(a, axis=-1),
    }

    if op not in ops:
        raise ValueError(f"unknown op: {op}")

    y = ops[op]()
    mx.eval(y)
    y_np = np.array(y, dtype=np.float32).reshape(-1).tolist()

    with open(out_path, "w") as f:
        json.dump({"out": y_np}, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.json output.json", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

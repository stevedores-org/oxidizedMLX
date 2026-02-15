import json
import math
import os
import random


def matmul(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i][kk] * b[kk][j]
            out[i][j] = s
    return out


def transpose(x):
    return [list(row) for row in zip(*x)]


def softmax_rows(x):
    out = []
    for row in x:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.append([v / s for v in exps])
    return out


def attention(q, k, v, scale, causal):
    scores = matmul(q, transpose(k))
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] *= scale
            if causal and j > i:
                scores[i][j] = -1e9
    probs = softmax_rows(scores)
    out = matmul(probs, v)
    return out


def randn(rng, mu=0.0, sigma=1.0):
    # Box-Muller
    u1 = rng.random()
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z


def gen_matrix(rng, rows, cols, mu=0.0, sigma=1.0):
    return [[randn(rng, mu, sigma) for _ in range(cols)] for _ in range(rows)]


def flatten(x):
    return [v for row in x for v in row]


def gen_case(out_dir, tq, tk, dh, seed=0, causal=True):
    rng = random.Random(seed)
    q = gen_matrix(rng, tq, dh)
    k = gen_matrix(rng, tk, dh)
    v = gen_matrix(rng, tk, dh)
    scale = 1.0 / math.sqrt(dh)

    out = attention(q, k, v, scale, causal)

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "tq": tq,
        "tk": tk,
        "dh": dh,
        "scale": scale,
        "causal": causal,
        "q": flatten(q),
        "k": flatten(k),
        "v": flatten(v),
        "out": flatten(out),
    }
    with open(os.path.join(out_dir, "golden.json"), "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    base = os.path.join("tools", "goldens", "attn")
    gen_case(os.path.join(base, "case_small"), tq=4, tk=4, dh=16, seed=0, causal=True)
    gen_case(os.path.join(base, "case_medium"), tq=32, tk=32, dh=64, seed=1, causal=True)
    gen_case(os.path.join(base, "case_asym"), tq=8, tk=16, dh=32, seed=2, causal=True)

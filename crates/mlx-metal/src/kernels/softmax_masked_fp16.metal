#include <metal_stdlib>
using namespace metal;

kernel void softmax_masked_fp16(
    device const half* scores [[ buffer(0) ]],
    device half* probs [[ buffer(1) ]],
    constant uint& Tq [[ buffer(2) ]],
    constant uint& Tk [[ buffer(3) ]],
    constant float& scale [[ buffer(4) ]],
    constant uint& causal [[ buffer(5) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint3 tgp [[ threadgroup_position_in_grid ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    uint row = tgp.x;
    if (row >= Tq) {
        return;
    }
    uint base = row * Tk;

    threadgroup float shm[256];

    float max_val = -INFINITY;
    for (uint j = tid; j < Tk; j += tg_size) {
        if (causal != 0 && j > row) {
            continue;
        }
        float v = (float)scores[base + j] * scale;
        if (v > max_val) {
            max_val = v;
        }
    }
    shm[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shm[tid] = max(shm[tid], shm[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float row_max = shm[0];
    float sum = 0.0f;
    for (uint j = tid; j < Tk; j += tg_size) {
        if (causal != 0 && j > row) {
            continue;
        }
        float v = (float)scores[base + j] * scale;
        sum += exp(v - row_max);
    }
    shm[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float denom = shm[0];
    for (uint j = tid; j < Tk; j += tg_size) {
        if (causal != 0 && j > row) {
            probs[base + j] = (half)0.0f;
        } else {
            float v = (float)scores[base + j] * scale;
            float p = exp(v - row_max) / denom;
            probs[base + j] = (half)p;
        }
    }
}

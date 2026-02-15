#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
    uint row_size;
    uint num_rows;
    float scale;
    uint has_mask;
};

// A row-wise softmax kernel. Each threadgroup handles one or more rows.
// For simplicity, this version uses one thread per element for small rows,
// or a serial loop for larger rows if we don't implement full multi-pass reduction yet.
// However, a common efficient way is to use SIMD shuffle/reduction within a threadgroup.

kernel void scaled_softmax_f32(device const float* in    [[buffer(0)]],
                               device float*       out   [[buffer(1)]],
                               constant SoftmaxParams& p [[buffer(2)]],
                               device const float* mask  [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    uint row_idx = gid;
    if (row_idx >= p.num_rows) return;

    device const float* row_in = in + row_idx * p.row_size;
    device float* row_out = out + row_idx * p.row_size;
    device const float* row_mask = p.has_mask ? mask + row_idx * p.row_size : nullptr;

    // 1. Find Max
    float max_val = -INFINITY;
    for (uint i = 0; i < p.row_size; i++) {
        float val = row_in[i] * p.scale;
        if (p.has_mask) {
            val += row_mask[i];
        }
        if (val > max_val) max_val = val;
    }

    // 2. Compute Sum of Exps
    float sum_exp = 0.0f;
    for (uint i = 0; i < p.row_size; i++) {
        float val = row_in[i] * p.scale;
        if (p.has_mask) {
            val += row_mask[i];
        }
        float e = exp(val - max_val);
        row_out[i] = e; // Temporarily store exp
        sum_exp += e;
    }

    // 3. Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < p.row_size; i++) {
        row_out[i] *= inv_sum;
    }
}

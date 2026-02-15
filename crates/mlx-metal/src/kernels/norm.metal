#include <metal_stdlib>
using namespace metal;

struct NormParams {
    uint row_size;
    float eps;
};

kernel void rms_norm_f32(
    device const float* x   [[ buffer(0) ]],
    device float* out       [[ buffer(1) ]],
    constant NormParams& p  [[ buffer(2) ]],
    uint row_idx            [[ thread_position_in_grid ]]
) {
    uint base = row_idx * p.row_size;

    // 1. Compute mean(x^2)
    float sum_sq = 0.0f;
    for (uint i = 0; i < p.row_size; ++i) {
        float val = x[base + i];
        sum_sq += val * val;
    }
    float rms = sqrt(sum_sq / p.row_size + p.eps);
    float inv_rms = 1.0f / rms;

    // 2. Normalize
    for (uint i = 0; i < p.row_size; ++i) {
        out[base + i] = x[base + i] * inv_rms;
    }
}

kernel void layer_norm_f32(
    device const float* x   [[ buffer(0) ]],
    device float* out       [[ buffer(1) ]],
    constant NormParams& p  [[ buffer(2) ]],
    uint row_idx            [[ thread_position_in_grid ]]
) {
    uint base = row_idx * p.row_size;

    // 1. Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < p.row_size; ++i) {
        sum += x[base + i];
    }
    float mean = sum / p.row_size;

    // 2. Compute variance
    float sum_sq_diff = 0.0f;
    for (uint i = 0; i < p.row_size; ++i) {
        float diff = x[base + i] - mean;
        sum_sq_diff += diff * diff;
    }
    float var = sum_sq_diff / p.row_size;
    float inv_std = 1.0f / sqrt(var + p.eps);

    // 3. Normalize
    for (uint i = 0; i < p.row_size; ++i) {
        out[base + i] = (x[base + i] - mean) * inv_std;
    }
}

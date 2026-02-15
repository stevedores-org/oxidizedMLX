#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void mul_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void sub_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void div_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* out     [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] / b[id];
}

kernel void exp_f32(device const float* a [[buffer(0)]],
                    device float* out     [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = exp(a[id]);
}

kernel void log_f32(device const float* a [[buffer(0)]],
                    device float* out     [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = log(a[id]);
}

kernel void neg_f32(device const float* a [[buffer(0)]],
                    device float* out     [[buffer(1)]],
                    uint id [[thread_position_in_grid]]) {
    out[id] = -a[id];
}

kernel void transpose_2d_f32(device const float* in [[buffer(0)]],
                             device float* out      [[buffer(1)]],
                             constant uint* dims    [[buffer(2)]], // [rows, cols]
                             uint2 gid [[thread_position_in_grid]]) {
    uint rows = dims[0];
    uint cols = dims[1];
    if (gid.x >= cols || gid.y >= rows) return;
    out[gid.x * rows + gid.y] = in[gid.y * cols + gid.x];
}

kernel void silu_f32(device const float* a [[buffer(0)]],
                     device float* out     [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    float x = a[id];
    out[id] = x / (1.0f + exp(-x));
}

kernel void gelu_f32(device const float* a [[buffer(0)]],
                     device float* out     [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    float x = a[id];
    // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float inner = 0.79788456f * (x + 0.044715f * x * x * x);
    out[id] = 0.5f * x * (1.0f + tanh(inner));
}

#include <metal_stdlib>
using namespace metal;

struct RoPEParams {
    uint head_dim;
    uint seq_len;
    uint offset;
    float base;
    bool traditional;
};

kernel void rope_kernel(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant RoPEParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint s = gid.y;
    uint i = gid.x;

    if (s >= params.seq_len || i >= params.head_dim / 2) {
        return;
    }

    uint head_dim = params.head_dim;
    uint base_idx = b * params.seq_len * head_dim + s * head_dim;
    
    uint i0, i1;
    if (params.traditional) {
        i0 = 2 * i;
        i1 = 2 * i + 1;
    } else {
        i0 = i;
        i1 = i + head_dim / 2;
    }

    float pos = (float)(s + params.offset);
    float theta = pos / pow(params.base, (float)(2 * i) / (float)head_dim);
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    float x0 = in[base_idx + i0];
    float x1 = in[base_idx + i1];

    out[base_idx + i0] = x0 * cos_t - x1 * sin_t;
    out[base_idx + i1] = x1 * cos_t + x0 * sin_t;
}

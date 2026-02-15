#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint tokens;
    uint head_dim;
    uint rotary_dim;
    uint pos_offset;
    float theta;
};

/// Interleaved RoPE on last-dimension pairs.
///
/// x: [tokens, head_dim]   (contiguous f32)
/// out: [tokens, head_dim]
///
/// For each pair (x[..., 2i], x[..., 2i+1]) within rotary_dim:
///   y0 = x0 * cos(angle) - x1 * sin(angle)
///   y1 = x0 * sin(angle) + x1 * cos(angle)
/// where angle = (pos_offset + t) * theta^{-2i / rotary_dim}.
///
/// Elements beyond rotary_dim are copied unchanged.
/// Only even-index threads compute pairs to avoid double writes.
kernel void rope_f32(
    device const float* x       [[ buffer(0) ]],
    device float* out           [[ buffer(1) ]],
    constant RopeParams& p      [[ buffer(2) ]],
    uint gid                    [[ thread_position_in_grid ]]
) {
    uint total = p.tokens * p.head_dim;
    if (gid >= total) return;

    uint t = gid / p.head_dim;
    uint d = gid % p.head_dim;

    // Outside rotary_dim: copy unchanged.
    if (d >= p.rotary_dim) {
        out[gid] = x[gid];
        return;
    }

    // Only even-index threads write the pair.
    if ((d & 1u) != 0u) return;

    uint i = d >> 1;
    float inv_freq = precise::pow(p.theta, -2.0f * float(i) / float(p.rotary_dim));
    float angle = float(p.pos_offset + t) * inv_freq;
    float c = precise::cos(angle);
    float s = precise::sin(angle);

    uint base = t * p.head_dim + (i << 1);
    float x0 = x[base];
    float x1 = x[base + 1];

    out[base]     = x0 * c - x1 * s;
    out[base + 1] = x0 * s + x1 * c;
}

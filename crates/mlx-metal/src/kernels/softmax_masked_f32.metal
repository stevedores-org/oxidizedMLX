#include <metal_stdlib>
using namespace metal;

struct SoftmaxMaskedParams {
    uint tq;
    uint tk;
    float scale;
    uint causal;  // 0 or 1
};

kernel void softmax_masked_f32(
    device const float* scores [[buffer(0)]],
    device float* probs        [[buffer(1)]],
    constant SoftmaxMaskedParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= p.tq) return;

    uint base = row * p.tk;

    // Step 1: Scale + mask, find max
    float max_val = -1e30f;
    for (uint j = 0; j < p.tk; j++) {
        float val = scores[base + j] * p.scale;
        if (p.causal != 0 && j > row) {
            val = -1e9f;
        }
        probs[base + j] = val;
        if (val > max_val) {
            max_val = val;
        }
    }

    // Step 2: exp(x - max) and sum
    float sum_exp = 0.0f;
    for (uint j = 0; j < p.tk; j++) {
        float e = exp(probs[base + j] - max_val);
        probs[base + j] = e;
        sum_exp += e;
    }

    // Step 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint j = 0; j < p.tk; j++) {
        probs[base + j] *= inv_sum;
    }
}

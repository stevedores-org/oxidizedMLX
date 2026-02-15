#include <metal_stdlib>
using namespace metal;

kernel void gemm_abt_fp16(
    device const half* a [[ buffer(0) ]],
    device const half* b [[ buffer(1) ]],
    device half* out [[ buffer(2) ]],
    constant uint& M [[ buffer(3) ]],
    constant uint& N [[ buffer(4) ]],
    constant uint& K [[ buffer(5) ]],
    uint2 tid [[ threadgroup_position_in_grid ]],
    uint2 tid_local [[ thread_position_in_threadgroup ]]
) {
    uint row = tid.y * 16 + tid_local.y;
    uint col = tid.x * 16 + tid_local.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    uint a_base = row * K;
    uint b_base = col * K;
    for (uint k = 0; k < K; k++) {
        float av = (float)a[a_base + k];
        float bv = (float)b[b_base + k];
        acc += av * bv;
    }
    out[row * N + col] = (half)acc;
}

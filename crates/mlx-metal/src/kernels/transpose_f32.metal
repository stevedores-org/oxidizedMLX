#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant TransposeParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (row >= p.rows || col >= p.cols) return;

    output[col * p.rows + row] = input[row * p.cols + col];
}

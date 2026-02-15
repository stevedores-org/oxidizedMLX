/// Metal Shading Language source for GEMM (f32).
///
/// Contains two kernels:
/// - `naive_gemm_f32`: one thread per output element, simple K-loop (correctness baseline)
/// - `tiled_gemm_f32`: 16×16 threadgroup tiles with shared memory (performance kernel)
pub const GEMM_F32_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint M;
    uint N;
    uint K;
};

// Naive GEMM: one thread computes one output element.
kernel void naive_gemm_f32(device const float* A   [[buffer(0)]],
                           device const float* B   [[buffer(1)]],
                           device float*       Out [[buffer(2)]],
                           constant GemmParams& p  [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= p.M || col >= p.N) return;

    float acc = 0.0f;
    for (uint i = 0; i < p.K; i++) {
        acc += A[row * p.K + i] * B[i * p.N + col];
    }
    Out[row * p.N + col] = acc;
}

// Tiled GEMM: 16×16 tiles loaded into threadgroup shared memory.
constant uint TM = 16;
constant uint TN = 16;
constant uint TK = 16;

kernel void tiled_gemm_f32(device const float* A   [[buffer(0)]],
                           device const float* B   [[buffer(1)]],
                           device float*       Out [[buffer(2)]],
                           constant GemmParams& p  [[buffer(3)]],
                           uint2 group_id  [[threadgroup_position_in_grid]],
                           uint2 local_id  [[thread_position_in_threadgroup]]) {
    // Tile origin in global coords.
    uint row = group_id.y * TM + local_id.y;
    uint col = group_id.x * TN + local_id.x;

    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    float acc = 0.0f;

    uint num_tiles = (p.K + TK - 1) / TK;
    for (uint t = 0; t < num_tiles; t++) {
        // Load A tile: row from global row, col from tile offset
        uint a_col = t * TK + local_id.x;
        As[local_id.y][local_id.x] = (row < p.M && a_col < p.K)
            ? A[row * p.K + a_col]
            : 0.0f;

        // Load B tile: row from tile offset, col from global col
        uint b_row = t * TK + local_id.y;
        Bs[local_id.y][local_id.x] = (b_row < p.K && col < p.N)
            ? B[b_row * p.N + col]
            : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TK; i++) {
            acc += As[local_id.y][i] * Bs[i][local_id.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p.M && col < p.N) {
        Out[row * p.N + col] = acc;
    }
}
"#;

/// Parameters passed to the GEMM kernel via constant buffer (buffer 3).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GemmParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

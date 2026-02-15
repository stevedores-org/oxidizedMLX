/*
 * mlxrs.h — C ABI contract for oxidizedMLX ↔ MLX C++ interop.
 *
 * ABI version: 1
 *
 * All functions use opaque handles. Callers must free handles exactly once
 * via mlxrs_free_tensor / mlxrs_free_device. Passing NULL to free is a no-op.
 *
 * Two implementations exist:
 *   1. "native" — pure Rust (mlx-sys with `native` feature, default)
 *   2. "cpp"    — links against MLX C++ via cmake (mlx-sys with `cpp` feature)
 */

#ifndef MLXRS_H
#define MLXRS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── ABI version ──────────────────────────────────────────────────────── */

#define MLXRS_ABI_VERSION 1

/* ── Opaque handle types ──────────────────────────────────────────────── */

typedef struct mlx_tensor_t mlx_tensor_t;
typedef struct mlx_device_t mlx_device_t;

/* ── Enumerations ─────────────────────────────────────────────────────── */

typedef enum {
    MLXRS_DEVICE_CPU = 0,
    MLXRS_DEVICE_GPU = 1,
} mlx_device_type_t;

typedef enum {
    MLXRS_DTYPE_F32  = 0,
    MLXRS_DTYPE_F16  = 1,
    MLXRS_DTYPE_BF16 = 2,
    MLXRS_DTYPE_I32  = 3,
    MLXRS_DTYPE_I64  = 4,
} mlx_dtype_t;

/* ── Device discovery ─────────────────────────────────────────────────── */

/* Returns the default device: GPU on Apple Silicon, CPU elsewhere. */
mlx_device_t* mlxrs_default_device(void);

/* Query the device type of a device handle. */
mlx_device_type_t mlxrs_device_type(mlx_device_t* d);

/* Create a specific device. */
mlx_device_t* mlxrs_cpu_device(void);
mlx_device_t* mlxrs_gpu_device(void);

/* ── Tensor creation ──────────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_zeros(mlx_device_t* device, mlx_dtype_t dtype,
                           const int64_t* shape_ptr, size_t shape_len);
mlx_tensor_t* mlxrs_ones(mlx_device_t* device, mlx_dtype_t dtype,
                          const int64_t* shape_ptr, size_t shape_len);
mlx_tensor_t* mlxrs_from_f32(mlx_device_t* device,
                              const int64_t* shape_ptr, size_t shape_len,
                              const float* data_ptr, size_t data_len);

/* ── Elementwise ops ──────────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_add(mlx_tensor_t* a, mlx_tensor_t* b);
mlx_tensor_t* mlxrs_mul(mlx_tensor_t* a, mlx_tensor_t* b);
mlx_tensor_t* mlxrs_neg(mlx_tensor_t* a);

/* ── Linear algebra ───────────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_matmul(mlx_tensor_t* a, mlx_tensor_t* b);

/* ── Reductions ───────────────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_sum(mlx_tensor_t* a, int axis);
mlx_tensor_t* mlxrs_sum_all(mlx_tensor_t* a);

/* ── Shape manipulation ───────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_reshape(mlx_tensor_t* a, const int64_t* shape_ptr,
                             size_t shape_len);
mlx_tensor_t* mlxrs_transpose(mlx_tensor_t* a);

/* ── Activation functions ─────────────────────────────────────────────── */

mlx_tensor_t* mlxrs_softmax(mlx_tensor_t* a, int axis);

/* ── Materialization ──────────────────────────────────────────────────── */

void  mlxrs_eval(mlx_tensor_t* t);
int   mlxrs_to_f32_vec(mlx_tensor_t* t, float* out_ptr, size_t out_len);
int64_t mlxrs_numel(mlx_tensor_t* t);
int   mlxrs_ndim(mlx_tensor_t* t);
int   mlxrs_shape(mlx_tensor_t* t, int64_t* out_ptr, size_t out_len);

/* ── Lifecycle ────────────────────────────────────────────────────────── */

void mlxrs_free_tensor(mlx_tensor_t* t);
void mlxrs_free_device(mlx_device_t* d);

#ifdef __cplusplus
}
#endif

#endif /* MLXRS_H */

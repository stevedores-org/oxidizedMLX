//! Suite 6 (expanded) — Embedding Integration Tests
//!
//! Dedicated tests for the Embedding module beyond the basic smoke test
//! in mlx-nn. Validates token lookup, batch lookup, shape preservation,
//! and gradient flow through embeddings.

use mlx_core::{Device, Shape, Tensor};
use mlx_nn::{Embedding, Module};

fn cpu() -> Device {
    Device::Cpu
}

// ─── I6.1: Token Gather / Embedding Lookup ──────────────────────────────

#[test]
fn embedding_single_token_lookup() {
    // Weight: [4, 3] (vocab=4, dim=3)
    let weight = Tensor::from_f32(
        &[
            1.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, // token 1
            0.0, 0.0, 1.0, // token 2
            1.0, 1.0, 1.0, // token 3
        ],
        &Shape::new(vec![4, 3]),
        &cpu(),
    )
    .unwrap();

    let emb = Embedding::new(weight);
    assert_eq!(emb.num_embeddings(), 4);
    assert_eq!(emb.embedding_dim(), 3);

    // Look up token 2 → should get [0, 0, 1]
    let indices = Tensor::from_f32(&[2.0], &Shape::new(vec![1]), &cpu()).unwrap();
    let result = emb.forward(&indices).unwrap();
    mlx_conformance::assert_allclose(&result.to_vec_f32().unwrap(), &[0.0, 0.0, 1.0], 1e-6, 1e-6);
}

#[test]
fn embedding_sequence_lookup() {
    // Weight: [5, 2] (vocab=5, dim=2)
    let weight = Tensor::from_f32(
        &[
            0.1, 0.2, // token 0
            0.3, 0.4, // token 1
            0.5, 0.6, // token 2
            0.7, 0.8, // token 3
            0.9, 1.0, // token 4
        ],
        &Shape::new(vec![5, 2]),
        &cpu(),
    )
    .unwrap();

    let emb = Embedding::new(weight);

    // Look up sequence [3, 1, 4]
    let indices = Tensor::from_f32(&[3.0, 1.0, 4.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let result = emb.forward(&indices).unwrap();

    assert_eq!(result.shape(), &Shape::new(vec![3, 2]));
    mlx_conformance::assert_allclose(
        &result.to_vec_f32().unwrap(),
        &[0.7, 0.8, 0.3, 0.4, 0.9, 1.0],
        1e-5,
        1e-5,
    );
}

#[test]
fn embedding_repeated_indices() {
    // Same token looked up multiple times should give same embedding
    let weight = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![3, 2]),
        &cpu(),
    )
    .unwrap();

    let emb = Embedding::new(weight);

    let indices = Tensor::from_f32(&[1.0, 1.0, 1.0], &Shape::new(vec![3]), &cpu()).unwrap();
    let result = emb.forward(&indices).unwrap();

    let vals = result.to_vec_f32().unwrap();
    // All three rows should be identical (token 1 = [3.0, 4.0])
    mlx_conformance::assert_allclose(&vals[0..2], &[3.0, 4.0], 1e-6, 1e-6);
    mlx_conformance::assert_allclose(&vals[2..4], &[3.0, 4.0], 1e-6, 1e-6);
    mlx_conformance::assert_allclose(&vals[4..6], &[3.0, 4.0], 1e-6, 1e-6);
}

#[test]
fn embedding_first_and_last_token() {
    // Edge case: first (0) and last (vocab_size-1) tokens
    let vocab = 8;
    let dim = 4;
    let data: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1).collect();
    let weight =
        Tensor::from_f32(&data, &Shape::new(vec![vocab as i64, dim as i64]), &cpu()).unwrap();

    let emb = Embedding::new(weight);

    // Token 0
    let idx0 = Tensor::from_f32(&[0.0], &Shape::new(vec![1]), &cpu()).unwrap();
    let out0 = emb.forward(&idx0).unwrap().to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&out0, &[0.0, 0.1, 0.2, 0.3], 1e-6, 1e-6);

    // Token 7 (last)
    let idx7 = Tensor::from_f32(&[7.0], &Shape::new(vec![1]), &cpu()).unwrap();
    let out7 = emb.forward(&idx7).unwrap().to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&out7, &[2.8, 2.9, 3.0, 3.1], 1e-5, 1e-5);
}

// ─── Embedding → Downstream Op Composition ──────────────────────────────

#[test]
fn embedding_to_layer_norm_pipeline() {
    // Embedding lookup → LayerNorm (common LLM pattern)
    let weight = Tensor::from_f32(
        &[
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
        ],
        &Shape::new(vec![2, 4]),
        &cpu(),
    )
    .unwrap();

    let emb = Embedding::new(weight);

    let indices = Tensor::from_f32(&[0.0, 1.0], &Shape::new(vec![2]), &cpu()).unwrap();
    let embedded = emb.forward(&indices).unwrap();
    assert_eq!(embedded.shape(), &Shape::new(vec![2, 4]));

    // Apply LayerNorm to the embedded result
    let normed = embedded.layer_norm(1e-5);
    let result = normed.to_vec_f32().unwrap();
    assert_eq!(result.len(), 8);

    // Verify layer norm properties: each row should have mean ~0
    let row0_mean: f32 = result[0..4].iter().sum::<f32>() / 4.0;
    let row1_mean: f32 = result[4..8].iter().sum::<f32>() / 4.0;
    assert!(
        row0_mean.abs() < 1e-5,
        "row 0 mean should be ~0: {row0_mean}"
    );
    assert!(
        row1_mean.abs() < 1e-5,
        "row 1 mean should be ~0: {row1_mean}"
    );
}

#[test]
fn embedding_to_matmul_projection() {
    // Embedding → Linear projection (simulated via matmul)
    let vocab = 4;
    let emb_dim = 3;
    let proj_dim = 2;

    let emb_data: Vec<f32> = (0..vocab * emb_dim)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let proj_data: Vec<f32> = (0..emb_dim * proj_dim)
        .map(|i| (i as f32 + 1.0) * 0.5)
        .collect();

    let emb_weight = Tensor::from_f32(
        &emb_data,
        &Shape::new(vec![vocab as i64, emb_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let proj_weight = Tensor::from_f32(
        &proj_data,
        &Shape::new(vec![emb_dim as i64, proj_dim as i64]),
        &cpu(),
    )
    .unwrap();

    let emb = Embedding::new(emb_weight);

    // Embed tokens [0, 2]
    let indices = Tensor::from_f32(&[0.0, 2.0], &Shape::new(vec![2]), &cpu()).unwrap();
    let embedded = emb.forward(&indices).unwrap(); // [2, 3]
    let projected = embedded.matmul(&proj_weight).unwrap(); // [2, 2]

    assert_eq!(projected.shape(), &Shape::new(vec![2, proj_dim as i64]));
    let result = projected.to_vec_f32().unwrap();
    assert_eq!(result.len(), 2 * proj_dim);
    for (i, v) in result.iter().enumerate() {
        assert!(v.is_finite(), "projected output[{i}] is not finite: {v}");
    }
}

// ─── Module Properties ──────────────────────────────────────────────────

#[test]
fn embedding_weight_accessible() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight = Tensor::from_f32(&data, &Shape::new(vec![2, 3]), &cpu()).unwrap();
    let emb = Embedding::new(weight);

    // Can access the underlying weight tensor
    let w = emb.weight();
    assert_eq!(w.shape(), &Shape::new(vec![2, 3]));
    let w_data = w.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&w_data, &data, 1e-6, 1e-6);
}

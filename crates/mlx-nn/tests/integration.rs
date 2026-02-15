//! Integration tests: Embedding → forward path.
//!
//! Tests the end-to-end inference pipeline: safetensors load → embedding →
//! linear → norm → activation → attention → output projection.

use mlx_core::{Device, Shape, Tensor};
use mlx_nn::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention, RmsNorm};

fn cpu() -> Device {
    Device::Cpu
}

fn s(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec())
}

// ── Embedding → Linear → Norm pipeline ─────────────────────────────────

#[test]
fn test_embedding_to_linear_forward() {
    // Simulate: token_ids -> embedding -> linear -> output
    let vocab_size = 8;
    let embed_dim = 4;
    let hidden_dim = 4;

    // Create embedding weights: [vocab_size, embed_dim]
    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let embed_weight = Tensor::from_f32(&embed_data, &s(&[vocab_size as i64, embed_dim as i64]), &cpu()).unwrap();
    let embedding = Embedding::new(embed_weight);

    // Token indices: [3] (look up tokens 0, 3, 7)
    let indices = Tensor::from_f32(&[0.0, 3.0, 7.0], &s(&[3]), &cpu()).unwrap();

    // Step 1: Embedding lookup -> [3, 4]
    let embedded = embedding.forward(&indices).unwrap();
    assert_eq!(embedded.shape(), &s(&[3, embed_dim as i64]));

    // Step 2: Linear projection -> [3, hidden_dim]
    let weight = Tensor::from_f32(
        &[1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0],
        &s(&[hidden_dim as i64, embed_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear = Linear::new(weight, None);
    let projected = linear.forward(&embedded).unwrap();
    assert_eq!(projected.shape(), &s(&[3, hidden_dim as i64]));

    // Identity projection: output should equal embedded values
    let embed_vals = embedded.to_vec_f32().unwrap();
    let proj_vals = projected.to_vec_f32().unwrap();
    mlx_conformance::assert_allclose(&proj_vals, &embed_vals, 1e-5, 1e-5);
}

#[test]
fn test_embedding_to_layer_norm() {
    let vocab_size = 4;
    let embed_dim = 3;

    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let embed_weight = Tensor::from_f32(&embed_data, &s(&[vocab_size as i64, embed_dim as i64]), &cpu()).unwrap();
    let embedding = Embedding::new(embed_weight);
    let ln = LayerNorm::new(embed_dim, 1e-5);

    let indices = Tensor::from_f32(&[0.0, 1.0, 2.0], &s(&[3]), &cpu()).unwrap();
    let embedded = embedding.forward(&indices).unwrap();
    let normed = ln.forward(&embedded).unwrap();

    assert_eq!(normed.shape(), &s(&[3, embed_dim as i64]));

    let vals = normed.to_vec_f32().unwrap();
    // Each row should have mean ≈ 0
    for row in 0..3 {
        let start = row * embed_dim;
        let mean: f32 = vals[start..start + embed_dim].iter().sum::<f32>() / embed_dim as f32;
        assert!(mean.abs() < 1e-4, "row {row} mean not zero: {mean}");
    }
}

#[test]
fn test_embedding_to_rms_norm() {
    let vocab_size = 4;
    let embed_dim = 3;

    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let embed_weight = Tensor::from_f32(&embed_data, &s(&[vocab_size as i64, embed_dim as i64]), &cpu()).unwrap();
    let embedding = Embedding::new(embed_weight);
    let rn = RmsNorm::new(embed_dim, 1e-5);

    let indices = Tensor::from_f32(&[0.0, 2.0, 3.0], &s(&[3]), &cpu()).unwrap();
    let embedded = embedding.forward(&indices).unwrap();
    let normed = rn.forward(&embedded).unwrap();

    assert_eq!(normed.shape(), &s(&[3, embed_dim as i64]));
    let vals = normed.to_vec_f32().unwrap();
    assert_eq!(vals.len(), 9);
}

// ── Full transformer-like forward pass ─────────────────────────────────

#[test]
fn test_full_forward_embedding_to_attention() {
    // Simulate a minimal transformer block:
    // tokens -> embedding -> linear(Q,K,V) -> attention -> output projection
    let vocab_size = 8;
    let model_dim = 4;
    let n_heads = 2;
    let seq_len = 3;

    // Embedding
    let embed_data: Vec<f32> = (0..vocab_size * model_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let embed_weight = Tensor::from_f32(
        &embed_data,
        &s(&[vocab_size as i64, model_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let embedding = Embedding::new(embed_weight);

    // Token indices
    let indices = Tensor::from_f32(
        &[1.0, 4.0, 7.0],
        &s(&[seq_len as i64]),
        &cpu(),
    )
    .unwrap();

    // Step 1: Embed tokens -> [3, 4]
    let hidden = embedding.forward(&indices).unwrap();
    assert_eq!(hidden.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Step 2: LayerNorm
    let ln = LayerNorm::new(model_dim, 1e-5);
    let normed = ln.forward(&hidden).unwrap();
    assert_eq!(normed.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Step 3: Multi-head attention
    // Identity-ish weights for Q, K, V, O projections
    let eye4: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let w = Tensor::from_f32(&eye4, &s(&[model_dim as i64, model_dim as i64]), &cpu()).unwrap();
    let wq = Linear::new(w.clone(), None);
    let wk = Linear::new(w.clone(), None);
    let wv = Linear::new(w.clone(), None);
    let wo = Linear::new(w, None);
    let mha = MultiHeadAttention::new(wq, wk, wv, wo, n_heads);

    let attn_out = mha.forward_causal(&normed).unwrap();
    assert_eq!(attn_out.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Step 4: Residual connection (add back the original)
    let residual = attn_out.add(&normed).unwrap();
    assert_eq!(residual.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Step 5: Final layer norm
    let final_ln = LayerNorm::new(model_dim, 1e-5);
    let output = final_ln.forward(&residual).unwrap();
    assert_eq!(output.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Verify output is valid
    let vals = output.to_vec_f32().unwrap();
    assert_eq!(vals.len(), seq_len * model_dim);
    assert!(vals.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
}

// ── Embedding → GELU/SiLU feedforward ──────────────────────────────────

#[test]
fn test_embedding_to_gelu_feedforward() {
    // tokens -> embedding -> linear -> gelu -> linear -> output
    let vocab_size = 4;
    let embed_dim = 2;

    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let embed_weight = Tensor::from_f32(
        &embed_data,
        &s(&[vocab_size as i64, embed_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let embedding = Embedding::new(embed_weight);

    let indices = Tensor::from_f32(&[0.0, 2.0], &s(&[2]), &cpu()).unwrap();
    let hidden = embedding.forward(&indices).unwrap();
    assert_eq!(hidden.shape(), &s(&[2, 2]));

    // Linear up-projection
    let w_up = Tensor::from_f32(
        &[1.0, 0.5, 0.5, 1.0],
        &s(&[embed_dim as i64, embed_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear_up = Linear::new(w_up, None);
    let up = linear_up.forward(&hidden).unwrap();

    // GELU activation
    let activated = up.gelu();

    // Linear down-projection
    let w_down = Tensor::from_f32(
        &[1.0, 0.0, 0.0, 1.0],
        &s(&[embed_dim as i64, embed_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear_down = Linear::new(w_down, None);
    let output = linear_down.forward(&activated).unwrap();

    assert_eq!(output.shape(), &s(&[2, 2]));
    let vals = output.to_vec_f32().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
}

#[test]
fn test_embedding_to_silu_feedforward() {
    let vocab_size = 4;
    let embed_dim = 2;

    let embed_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32) * 0.3)
        .collect();
    let embed_weight = Tensor::from_f32(
        &embed_data,
        &s(&[vocab_size as i64, embed_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let embedding = Embedding::new(embed_weight);

    let indices = Tensor::from_f32(&[1.0, 3.0], &s(&[2]), &cpu()).unwrap();
    let hidden = embedding.forward(&indices).unwrap();

    // SiLU activation
    let activated = hidden.silu();

    let vals = activated.to_vec_f32().unwrap();
    assert_eq!(vals.len(), 4);
    assert!(vals.iter().all(|v| v.is_finite()));
}

// ── Multi-step inference simulation ────────────────────────────────────

#[test]
fn test_two_layer_forward_pass() {
    // Two transformer-style blocks back to back
    let model_dim = 4;
    let seq_len = 2;

    let data: Vec<f32> = (0..seq_len * model_dim)
        .map(|i| ((i as f32) * 0.2).sin())
        .collect();
    let x = Tensor::from_f32(&data, &s(&[seq_len as i64, model_dim as i64]), &cpu()).unwrap();

    // Block 1: LayerNorm -> SiLU -> Linear
    let ln1 = LayerNorm::new(model_dim, 1e-5);
    let h1 = ln1.forward(&x).unwrap();
    let h1 = h1.silu();
    let w1 = Tensor::from_f32(
        &[1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0],
        &s(&[model_dim as i64, model_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear1 = Linear::new(w1, None);
    let h1 = linear1.forward(&h1).unwrap();

    // Block 2: RMSNorm -> GELU -> Linear
    let rn2 = RmsNorm::new(model_dim, 1e-5);
    let h2 = rn2.forward(&h1).unwrap();
    let h2 = h2.gelu();
    let w2 = Tensor::from_f32(
        &[0.5, 0.0, 0.0, 0.0,
          0.0, 0.5, 0.0, 0.0,
          0.0, 0.0, 0.5, 0.0,
          0.0, 0.0, 0.0, 0.5],
        &s(&[model_dim as i64, model_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear2 = Linear::new(w2, None);
    let output = linear2.forward(&h2).unwrap();

    assert_eq!(output.shape(), &s(&[seq_len as i64, model_dim as i64]));
    let vals = output.to_vec_f32().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
}

// ── Shape preservation through pipeline ────────────────────────────────

#[test]
fn test_shape_preservation_full_pipeline() {
    let seq_len = 5;
    let model_dim = 8;

    let data: Vec<f32> = (0..seq_len * model_dim)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let x = Tensor::from_f32(&data, &s(&[seq_len as i64, model_dim as i64]), &cpu()).unwrap();

    // Each operation should preserve the [seq_len, model_dim] shape
    let ln = LayerNorm::new(model_dim, 1e-5);
    let normed = ln.forward(&x).unwrap();
    assert_eq!(normed.shape(), &s(&[seq_len as i64, model_dim as i64]));

    let activated = normed.gelu();
    assert_eq!(activated.shape(), &s(&[seq_len as i64, model_dim as i64]));

    let w = Tensor::from_f32(
        &vec![0.0; model_dim * model_dim]
            .iter()
            .enumerate()
            .map(|(i, _)| if i % (model_dim + 1) == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>(),
        &s(&[model_dim as i64, model_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let linear = Linear::new(w, None);
    let projected = linear.forward(&activated).unwrap();
    assert_eq!(projected.shape(), &s(&[seq_len as i64, model_dim as i64]));

    // Residual
    let residual = projected.add(&x).unwrap();
    assert_eq!(residual.shape(), &s(&[seq_len as i64, model_dim as i64]));
}

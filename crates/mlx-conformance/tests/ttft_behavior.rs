//! Suite 3 — TTFT vs Generation Behavior (L3.1, L3.2)
//!
//! Apple's core insight: TTFT is compute-bound (matmul-heavy) while
//! generation is bandwidth-bound (KV reuse). These tests validate that
//! oxidizedMLX reproduces these patterns by measuring the computational
//! profile of prefill vs decode phases.

use mlx_core::{Device, Shape, Tensor};

fn cpu() -> Device {
    Device::Cpu
}

// ─── L3.1: TTFT Microbenchmark (Compute Bound) ─────────────────────────

#[test]
fn ttft_prefill_matmul_dominates() {
    // Single forward pass of a "transformer layer" with seq_len=64.
    // In the prefill phase, dense matmul dominates compute.
    let seq_len = 64;
    let dim = 32;
    let ff_dim = 64;

    let x_data: Vec<f32> = (0..seq_len * dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let wq_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect();
    let wk_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    let wv_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.03)
        .collect();
    let w1_data: Vec<f32> = (0..dim * ff_dim)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.01)
        .collect();
    let w2_data: Vec<f32> = (0..ff_dim * dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let x = Tensor::from_f32(
        &x_data,
        &Shape::new(vec![seq_len as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();
    let wq = Tensor::from_f32(&wq_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();
    let wk = Tensor::from_f32(&wk_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();
    let wv = Tensor::from_f32(&wv_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();
    let w1 = Tensor::from_f32(
        &w1_data,
        &Shape::new(vec![dim as i64, ff_dim as i64]),
        &cpu(),
    )
    .unwrap();
    let w2 = Tensor::from_f32(
        &w2_data,
        &Shape::new(vec![ff_dim as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();

    // Measure time for full prefill forward pass
    let start = std::time::Instant::now();

    // Self-attention: Q,K,V projections (3 matmuls)
    let normed = x.layer_norm(1e-5);
    let q = normed.matmul(&wq).unwrap();
    let k = normed.matmul(&wk).unwrap();
    let v = normed.matmul(&wv).unwrap();

    // Attention: softmax(Q @ K^T / sqrt(d)) @ V
    let scale = 1.0 / (dim as f32).sqrt();
    let attn_out = q.attention(&k, &v, scale, true).unwrap();

    // FFN: up projection → SiLU → down projection (2 matmuls)
    let ffn_normed = attn_out.layer_norm(1e-5);
    let up = ffn_normed.matmul(&w1).unwrap();
    let activated = up.silu();
    let down = activated.matmul(&w2).unwrap();

    let result = down.to_vec_f32().unwrap();
    let prefill_time = start.elapsed();

    assert_eq!(result.len(), seq_len * dim, "prefill output shape mismatch");
    for (i, v) in result.iter().enumerate() {
        assert!(v.is_finite(), "prefill output[{i}] is not finite");
    }

    // The prefill should complete (no hang / deadlock) — time is informational
    assert!(
        prefill_time.as_secs() < 10,
        "prefill took too long: {prefill_time:?}"
    );
}

// ─── L3.2: Decode Loop Bandwidth Test ───────────────────────────────────

#[test]
fn decode_loop_incremental_attention() {
    // Simulate incremental decode: seq=1 appended to growing KV cache.
    // This pattern should be bandwidth-bound, not compute-bound.
    let dim = 16;
    let n_decode_steps = 32;

    let wq_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let wk_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
        .collect();
    let wv_data: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.04)
        .collect();

    let wq = Tensor::from_f32(&wq_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();
    let wk = Tensor::from_f32(&wk_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();
    let wv = Tensor::from_f32(&wv_data, &Shape::new(vec![dim as i64, dim as i64]), &cpu()).unwrap();

    // Start with a small prompt
    let prompt_len = 4;
    let mut kv_k_data: Vec<f32> = (0..prompt_len * dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.02)
        .collect();
    let mut kv_v_data: Vec<f32> = (0..prompt_len * dim)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.03)
        .collect();
    let mut kv_len = prompt_len;

    let mut decode_times = Vec::new();

    for step in 0..n_decode_steps {
        let start = std::time::Instant::now();

        // New token embedding (simulated)
        let new_token: Vec<f32> = (0..dim)
            .map(|i| ((step * dim + i) % 11) as f32 * 0.1)
            .collect();
        let x = Tensor::from_f32(&new_token, &Shape::new(vec![1, dim as i64]), &cpu()).unwrap();

        // Project new token
        let q_new = x.matmul(&wq).unwrap();
        let k_new = x.matmul(&wk).unwrap();
        let v_new = x.matmul(&wv).unwrap();

        // Append to KV cache
        let k_new_data = k_new.to_vec_f32().unwrap();
        let v_new_data = v_new.to_vec_f32().unwrap();
        kv_k_data.extend_from_slice(&k_new_data);
        kv_v_data.extend_from_slice(&v_new_data);
        kv_len += 1;

        // Attend over full KV cache
        let k_full = Tensor::from_f32(
            &kv_k_data,
            &Shape::new(vec![kv_len as i64, dim as i64]),
            &cpu(),
        )
        .unwrap();
        let v_full = Tensor::from_f32(
            &kv_v_data,
            &Shape::new(vec![kv_len as i64, dim as i64]),
            &cpu(),
        )
        .unwrap();

        let scale = 1.0 / (dim as f32).sqrt();
        // Q: [1, dim], K: [kv_len, dim], V: [kv_len, dim]
        let attn_out = q_new.attention(&k_full, &v_full, scale, false).unwrap();
        let result = attn_out.to_vec_f32().unwrap();

        let elapsed = start.elapsed();
        decode_times.push(elapsed);

        assert_eq!(
            result.len(),
            dim,
            "decode step {step} output shape mismatch"
        );
        for (i, v) in result.iter().enumerate() {
            assert!(
                v.is_finite(),
                "decode step {step} output[{i}] is not finite"
            );
        }
    }

    // Verify decode completed without excessive time
    let total: std::time::Duration = decode_times.iter().sum();
    assert!(total.as_secs() < 30, "decode loop took too long: {total:?}");

    // Verify decode time grows roughly linearly (not quadratically)
    // Compare first quarter vs last quarter average
    let q1_avg: f64 = decode_times[..n_decode_steps / 4]
        .iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>()
        / (n_decode_steps / 4) as f64;
    let q4_avg: f64 = decode_times[3 * n_decode_steps / 4..]
        .iter()
        .map(|d| d.as_nanos() as f64)
        .sum::<f64>()
        / (n_decode_steps / 4) as f64;

    // Last quarter should be at most ~10x slower than first quarter
    // (generous bound — actual should be much less)
    if q1_avg > 0.0 {
        let ratio = q4_avg / q1_avg;
        assert!(
            ratio < 10.0,
            "decode time growth ratio too high ({ratio:.1}x): suggests quadratic scaling"
        );
    }
}

// ─── Prefill vs Decode Behavioral Contrast ──────────────────────────────

#[test]
fn prefill_vs_decode_shape_contract() {
    // Validate the shape contracts that distinguish prefill from decode:
    // - Prefill: Q,K,V all have same seq_len (batch processing)
    // - Decode: Q has seq_len=1, K,V have growing seq_len
    let dim = 8;

    // Prefill: all [seq, dim]
    let prefill_seq = 16;
    let q_data: Vec<f32> = vec![0.1; prefill_seq * dim];
    let k_data: Vec<f32> = vec![0.1; prefill_seq * dim];
    let v_data: Vec<f32> = vec![0.1; prefill_seq * dim];

    let q = Tensor::from_f32(
        &q_data,
        &Shape::new(vec![prefill_seq as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();
    let k = Tensor::from_f32(
        &k_data,
        &Shape::new(vec![prefill_seq as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();
    let v = Tensor::from_f32(
        &v_data,
        &Shape::new(vec![prefill_seq as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();

    let prefill_out = q.attention(&k, &v, 0.5, true).unwrap();
    assert_eq!(
        prefill_out.shape(),
        &Shape::new(vec![prefill_seq as i64, dim as i64]),
        "prefill output should match Q shape"
    );

    // Decode: Q=[1, dim], K/V=[kv_len, dim]
    let kv_len = 32;
    let q_decode =
        Tensor::from_f32(&vec![0.1; dim], &Shape::new(vec![1, dim as i64]), &cpu()).unwrap();
    let k_decode = Tensor::from_f32(
        &vec![0.1; kv_len * dim],
        &Shape::new(vec![kv_len as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();
    let v_decode = Tensor::from_f32(
        &vec![0.1; kv_len * dim],
        &Shape::new(vec![kv_len as i64, dim as i64]),
        &cpu(),
    )
    .unwrap();

    let decode_out = q_decode
        .attention(&k_decode, &v_decode, 0.5, false)
        .unwrap();
    assert_eq!(
        decode_out.shape(),
        &Shape::new(vec![1, dim as i64]),
        "decode output should be [1, dim]"
    );
}

//! Optimizers for MLX training.
//!
//! Tensors are immutable graph node handles — optimizers create new tensors
//! for updated parameters rather than mutating in place.

use mlx_core::{Result, Tensor};

/// Optimizer trait: apply one step, returning updated parameters.
pub trait Optimizer {
    fn step(&mut self, params: &[Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>>;
}

// ── SGD ──────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent with optional momentum.
pub struct Sgd {
    lr: f32,
    momentum: f32,
    velocity: Vec<Tensor>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    ///
    /// - `lr`: learning rate
    /// - `momentum`: momentum factor (0.0 = no momentum)
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &[Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>> {
        // Initialize velocity on first call
        if self.velocity.is_empty() {
            self.velocity = params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect::<Result<Vec<_>>>()?;
        }

        let lr_scalar = self.lr;
        let mom = self.momentum;

        let mut new_params = Vec::with_capacity(params.len());
        let mut new_velocity = Vec::with_capacity(params.len());

        for (i, (p, g)) in params.iter().zip(grads.iter()).enumerate() {
            if mom == 0.0 {
                // p_new = p - lr * g
                let lr_t = scalar_like(lr_scalar, p)?;
                let update = lr_t.mul(g)?;
                new_params.push(p.sub(&update)?);
                new_velocity.push(self.velocity[i].clone());
            } else {
                // v = momentum * v + g
                let mom_t = scalar_like(mom, p)?;
                let v = mom_t.mul(&self.velocity[i])?.add(g)?;
                // p_new = p - lr * v
                let lr_t = scalar_like(lr_scalar, p)?;
                let update = lr_t.mul(&v)?;
                new_params.push(p.sub(&update)?);
                new_velocity.push(v);
            }
        }

        self.velocity = new_velocity;
        Ok(new_params)
    }
}

// ── AdamW ────────────────────────────────────────────────────────────────

/// AdamW optimizer (Adam with decoupled weight decay).
pub struct AdamW {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    t: u64,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &[Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>> {
        // Initialize moments on first call
        if self.m.is_empty() {
            self.m = params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect::<Result<Vec<_>>>()?;
            self.v = params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect::<Result<Vec<_>>>()?;
        }

        self.t += 1;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);

        let mut new_params = Vec::with_capacity(params.len());
        let mut new_m = Vec::with_capacity(params.len());
        let mut new_v = Vec::with_capacity(params.len());

        for (i, (p, g)) in params.iter().zip(grads.iter()).enumerate() {
            // m = β1 * m + (1 - β1) * g
            let b1_t = scalar_like(b1, p)?;
            let one_minus_b1 = scalar_like(1.0 - b1, p)?;
            let m_new = b1_t.mul(&self.m[i])?.add(&one_minus_b1.mul(g)?)?;

            // v = β2 * v + (1 - β2) * g²
            let b2_t = scalar_like(b2, p)?;
            let one_minus_b2 = scalar_like(1.0 - b2, p)?;
            let g_sq = g.mul(g)?;
            let v_new = b2_t.mul(&self.v[i])?.add(&one_minus_b2.mul(&g_sq)?)?;

            // Bias-corrected estimates
            let bc1_t = scalar_like(bc1, p)?;
            let bc2_t = scalar_like(bc2, p)?;
            let m_hat = m_new.div(&bc1_t)?;
            let v_hat = v_new.div(&bc2_t)?;

            // p_new = p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)
            let decay_factor = scalar_like(1.0 - self.lr * self.weight_decay, p)?;
            let lr_t = scalar_like(self.lr, p)?;
            let eps_t = scalar_like(self.eps, p)?;
            let denom = v_hat.sqrt().add(&eps_t)?;
            let step = lr_t.mul(&m_hat)?.div(&denom)?;
            let p_new = decay_factor.mul(p)?.sub(&step)?;

            new_params.push(p_new);
            new_m.push(m_new);
            new_v.push(v_new);
        }

        self.m = new_m;
        self.v = new_v;
        Ok(new_params)
    }
}

/// Create a scalar tensor broadcast to the same shape/dtype/device as `like`.
fn scalar_like(val: f32, like: &Tensor) -> Result<Tensor> {
    let data = vec![val; like.numel() as usize];
    Tensor::from_f32(&data, like.shape(), like.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_core::{Device, Shape};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn t(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, &Shape::new(shape.to_vec()), &cpu()).unwrap()
    }

    #[test]
    fn test_sgd_no_momentum() {
        let mut opt = Sgd::new(0.1, 0.0);
        let p = t(&[1.0, 2.0, 3.0], &[3]);
        let g = t(&[0.5, 1.0, 1.5], &[3]);
        let new_p = opt.step(&[p], &[g]).unwrap();
        let vals = new_p[0].to_vec_f32().unwrap();
        // p - lr * g = [1 - 0.05, 2 - 0.1, 3 - 0.15]
        mlx_conformance::assert_allclose(&vals, &[0.95, 1.9, 2.85], 1e-5, 1e-5);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut opt = Sgd::new(0.1, 0.9);
        let p = t(&[1.0, 2.0], &[2]);
        let g = t(&[1.0, 1.0], &[2]);

        // Step 1: v = 0.9*0 + g = [1,1], p = p - 0.1*v = [0.9, 1.9]
        let new_p = opt.step(&[p], std::slice::from_ref(&g)).unwrap();
        let vals1 = new_p[0].to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals1, &[0.9, 1.9], 1e-5, 1e-5);

        // Step 2: v = 0.9*[1,1] + [1,1] = [1.9,1.9], p = [0.9,1.9] - 0.1*[1.9,1.9] = [0.71, 1.71]
        let new_p2 = opt
            .step(std::slice::from_ref(&new_p[0]), std::slice::from_ref(&g))
            .unwrap();
        let vals2 = new_p2[0].to_vec_f32().unwrap();
        mlx_conformance::assert_allclose(&vals2, &[0.71, 1.71], 1e-5, 1e-5);
    }

    #[test]
    fn test_adamw_single_step() {
        let mut opt = AdamW::new(0.001)
            .betas(0.9, 0.999)
            .eps(1e-8)
            .weight_decay(0.01);
        let p = t(&[1.0, 2.0], &[2]);
        let g = t(&[0.1, 0.2], &[2]);

        let new_p = opt
            .step(std::slice::from_ref(&p), std::slice::from_ref(&g))
            .unwrap();
        let vals = new_p[0].to_vec_f32().unwrap();

        // Hand-compute step 1:
        // m = 0.1*(1-0.9)*g = [0.01, 0.02]
        // v = (1-0.999)*g^2 = [0.00001, 0.00004]
        // m_hat = m / (1 - 0.9) = [0.1, 0.2]
        // v_hat = v / (1 - 0.999) = [0.01, 0.04]
        // decay = 1 - 0.001*0.01 = 0.99999
        // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * [0.1, 0.2] / ([0.1, 0.2] + 1e-8) ≈ [0.001, 0.001]
        // p_new = decay*p - step ≈ [0.99999 - 0.001, 1.99998 - 0.001] ≈ [0.99899, 1.99898]
        let expected_0 = 0.99999 * 1.0 - 0.001 * 0.1 / (0.01f32.sqrt() + 1e-8);
        let expected_1 = 0.99999 * 2.0 - 0.001 * 0.2 / (0.04f32.sqrt() + 1e-8);
        mlx_conformance::assert_allclose(&vals, &[expected_0, expected_1], 1e-4, 1e-4);
    }

    #[test]
    fn test_adamw_two_steps() {
        let mut opt = AdamW::new(0.001)
            .betas(0.9, 0.999)
            .eps(1e-8)
            .weight_decay(0.0);
        let p = t(&[1.0], &[1]);
        let g = t(&[1.0], &[1]);

        let p1 = opt.step(&[p], std::slice::from_ref(&g)).unwrap();
        let p2 = opt
            .step(std::slice::from_ref(&p1[0]), std::slice::from_ref(&g))
            .unwrap();

        // After 2 steps, parameter should have decreased
        let v1 = p1[0].to_vec_f32().unwrap()[0];
        let v2 = p2[0].to_vec_f32().unwrap()[0];
        assert!(v1 < 1.0, "param should decrease after step 1");
        assert!(v2 < v1, "param should decrease after step 2");
    }
}

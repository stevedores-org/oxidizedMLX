#[cfg(target_os = "macos")]
mod tests {
    use mlx_metal::{attention_fp16, AttnParams, MetalBuffer, MetalContext};
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Golden {
        tq: usize,
        tk: usize,
        dh: usize,
        scale: f32,
        causal: bool,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        out: Vec<f32>,
    }

    fn load_case(name: &str) -> Golden {
        let path: PathBuf = ["tools", "goldens", "attn", name, "golden.json"].iter().collect();
        let data = fs::read_to_string(path).expect("read golden");
        serde_json::from_str(&data).expect("parse golden")
    }

    fn f32_to_f16_bits(data: &[f32]) -> Vec<u16> {
        data.iter().map(|&v| f32_to_f16_bits_scalar(v)).collect()
    }

    fn f16_bits_to_f32(data: &[u16]) -> Vec<f32> {
        data.iter().map(|&v| f16_bits_to_f32_scalar(v)).collect()
    }

    fn f32_to_f16_bits_scalar(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mut exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7fffff;
        if exp == 255 {
            if mant == 0 {
                return sign | 0x7c00;
            }
            return sign | 0x7e00;
        }
        exp -= 127;
        if exp > 15 {
            return sign | 0x7c00;
        }
        if exp < -14 {
            if exp < -24 {
                return sign;
            }
            let shift = (-exp - 1) as u32;
            let mantissa = (mant | 0x800000) >> (shift + 13);
            return sign | (mantissa as u16);
        }
        let exp_bits = ((exp + 15) as u16) << 10;
        let mant_bits = (mant >> 13) as u16;
        sign | exp_bits | mant_bits
    }

    fn f16_bits_to_f32_scalar(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = ((bits >> 10) & 0x1f) as i32;
        let mant = (bits & 0x3ff) as u32;
        let out = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut e = -14;
                let mut m = mant;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                let exp_bits = ((e + 127) as u32) << 23;
                let mant_bits = m << 13;
                sign | exp_bits | mant_bits
            }
        } else if exp == 31 {
            sign | 0x7f800000 | (mant << 13)
        } else {
            let exp_bits = ((exp - 15 + 127) as u32) << 23;
            let mant_bits = mant << 13;
            sign | exp_bits | mant_bits
        };
        f32::from_bits(out)
    }

    fn assert_allclose(got: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        for (g, e) in got.iter().zip(expected.iter()) {
            let diff = (g - e).abs();
            let tol = atol + rtol * e.abs();
            assert!(diff <= tol, "diff {diff} > tol {tol} (g={g}, e={e})");
        }
    }

    fn run_case(name: &str) {
        let g = load_case(name);
        let ctx = MetalContext::new().expect("ctx");

        let q_bits = f32_to_f16_bits(&g.q);
        let k_bits = f32_to_f16_bits(&g.k);
        let v_bits = f32_to_f16_bits(&g.v);

        let q = MetalBuffer::from_slice_shared(&ctx, &q_bits).unwrap();
        let k = MetalBuffer::from_slice_shared(&ctx, &k_bits).unwrap();
        let v = MetalBuffer::from_slice_shared(&ctx, &v_bits).unwrap();
        let out = MetalBuffer::<u16>::new_shared(&ctx, g.tq * g.dh).unwrap();

        attention_fp16(
            &ctx,
            &q,
            &k,
            &v,
            &out,
            AttnParams {
                tq: g.tq,
                tk: g.tk,
                dh: g.dh,
                scale: g.scale,
                causal: g.causal,
            },
        )
        .unwrap();

        let got = f16_bits_to_f32(&out.read_to_vec());
        assert_allclose(&got, &g.out, 3e-2, 2e-2);
    }

    #[test]
    fn golden_case_small() {
        run_case("case_small");
    }

    #[test]
    fn golden_case_medium() {
        run_case("case_medium");
    }

    #[test]
    fn golden_case_asym() {
        run_case("case_asym");
    }
}

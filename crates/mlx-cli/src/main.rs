use clap::{Parser, Subcommand};
use mlx_core::{DType, Device, Shape, Tensor};

#[derive(Parser)]
#[command(name = "mlx-cli")]
#[command(about = "oxidizedMLX development CLI")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a quick smoke test of core ops.
    Smoke,

    /// Exercise the C ABI shim (mlx-sys) and free all handles (useful for valgrind).
    CabiSmoke {
        /// Iterations of the allocate/op/free loop.
        #[arg(long, default_value_t = 1000)]
        iters: usize,
    },
}

fn main() {
    let args = Args::parse();
    match args.cmd {
        Cmd::Smoke => smoke(),
        Cmd::CabiSmoke { iters } => cabi_smoke(iters),
    }
}

fn smoke() {
    let dev = Device::default_device();
    println!("Device: {:?}", dev);
    println!("Backend: lazy graph + CPU reference\n");

    // matmul
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &dev).unwrap();
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &dev).unwrap();
    let c = a.matmul(&b).unwrap();
    println!(
        "matmul [[1,2],[3,4]] @ [[5,6],[7,8]] = {:?}",
        c.to_vec_f32().unwrap()
    );

    // add
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &dev).unwrap();
    let y = Tensor::from_f32(&[4.0, 5.0, 6.0], &Shape::new(vec![3]), &dev).unwrap();
    let z = x.add(&y).unwrap();
    println!("add [1,2,3] + [4,5,6] = {:?}", z.to_vec_f32().unwrap());

    // lazy chain: (a + b) * c — graph built lazily, evaluated on to_vec_f32
    let p = Tensor::from_f32(&[1.0, 2.0], &Shape::new(vec![2]), &dev).unwrap();
    let q = Tensor::from_f32(&[3.0, 4.0], &Shape::new(vec![2]), &dev).unwrap();
    let r = Tensor::from_f32(&[2.0, 3.0], &Shape::new(vec![2]), &dev).unwrap();
    let chain = p.add(&q).unwrap().mul(&r).unwrap();
    println!("lazy (1+3)*2, (2+4)*3 = {:?}", chain.to_vec_f32().unwrap());

    // softmax
    let s = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &dev).unwrap();
    let sm = s.softmax(0).unwrap();
    println!("softmax [1,2,3] = {:?}", sm.to_vec_f32().unwrap());

    // transpose
    let t = Tensor::from_f32(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::new(vec![2, 3]),
        &dev,
    )
    .unwrap();
    let tt = t.transpose(None).unwrap();
    println!("transpose [2,3]->[3,2] = {:?}", tt.to_vec_f32().unwrap());

    // layer norm
    let ln = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &dev).unwrap();
    let normed = ln.layer_norm(1e-5);
    println!("layer_norm [1,2,3] = {:?}", normed.to_vec_f32().unwrap());

    // zeros
    let zeros = Tensor::zeros(&Shape::new(vec![2, 3]), DType::F32, &dev).unwrap();
    println!("zeros [2,3] = {:?}", zeros.to_vec_f32().unwrap());

    println!("\nAll smoke tests passed.");
}

fn cabi_smoke(iters: usize) {
    println!("C ABI smoke (native shim): iters={iters}");

    unsafe {
        // Device lifecycle
        let dev = mlx_sys::mlxrs_default_device();
        assert!(!dev.is_null());

        for _ in 0..iters {
            let shape: [i64; 2] = [2, 2];
            let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            let b_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];

            // Allocate tensors
            let a = mlx_sys::mlxrs_from_f32(
                dev,
                shape.as_ptr(),
                shape.len(),
                a_data.as_ptr(),
                a_data.len(),
            );
            let b = mlx_sys::mlxrs_from_f32(
                dev,
                shape.as_ptr(),
                shape.len(),
                b_data.as_ptr(),
                b_data.len(),
            );
            assert!(!a.is_null());
            assert!(!b.is_null());

            // Ops
            let c = mlx_sys::mlxrs_matmul(a, b);
            assert!(!c.is_null());
            mlx_sys::mlxrs_eval(c);

            // Materialize output to ensure buffers get touched
            let n = mlx_sys::mlxrs_numel(c) as usize;
            let mut out = vec![0.0f32; n];
            let rc = mlx_sys::mlxrs_to_f32_vec(c, out.as_mut_ptr(), n);
            assert_eq!(rc, 0);

            // Free lifecycle
            mlx_sys::mlxrs_free_tensor(a);
            mlx_sys::mlxrs_free_tensor(b);
            mlx_sys::mlxrs_free_tensor(c);
        }

        mlx_sys::mlxrs_free_device(dev);
    }

    println!("C ABI smoke passed.");
}

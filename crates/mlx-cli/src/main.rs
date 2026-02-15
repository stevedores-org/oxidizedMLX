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
}

fn main() {
    let args = Args::parse();
    match args.cmd {
        Cmd::Smoke => smoke(),
    }
}

fn smoke() {
    let dev = Device::default_device();
    println!("Device: {:?}", dev);

    // matmul smoke
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &Shape::new(vec![2, 2]), &dev).unwrap();
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &Shape::new(vec![2, 2]), &dev).unwrap();
    let c = a.matmul(&b).unwrap();
    println!(
        "matmul [[1,2],[3,4]] @ [[5,6],[7,8]] = {:?}",
        c.to_vec_f32().unwrap()
    );

    // add smoke
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &dev).unwrap();
    let y = Tensor::from_f32(&[4.0, 5.0, 6.0], &Shape::new(vec![3]), &dev).unwrap();
    let z = x.add(&y).unwrap();
    println!("add [1,2,3] + [4,5,6] = {:?}", z.to_vec_f32().unwrap());

    // softmax smoke
    let s = Tensor::from_f32(&[1.0, 2.0, 3.0], &Shape::new(vec![3]), &dev).unwrap();
    let sm = s.softmax(0).unwrap();
    println!("softmax [1,2,3] = {:?}", sm.to_vec_f32().unwrap());

    // zeros smoke
    let zeros = Tensor::zeros(&Shape::new(vec![2, 3]), DType::F32, &dev).unwrap();
    println!("zeros [2,3] = {:?}", zeros.to_vec_f32().unwrap());

    println!("\nAll smoke tests passed.");
}

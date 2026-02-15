fn main() {
    println!("cargo:rerun-if-changed=src/kernels/add_u32.metal");
    println!("cargo:rerun-if-changed=src/kernels/add_f32.metal");
    println!("cargo:rerun-if-changed=src/kernels/gemm.metal");
}

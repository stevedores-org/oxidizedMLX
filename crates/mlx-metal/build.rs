fn main() {
    println!("cargo:rerun-if-changed=src/kernels/add_u32.metal");
}

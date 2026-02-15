use std::env;

fn main() {
    // Path to your MLX fork source. Set MLX_SRC=/path/to/mlx
    let mlx_src = env::var("MLX_SRC").unwrap_or_else(|_| "../../mlx".to_string());

    // Build MLX C ABI shim with cmake.
    let dst = cmake::Config::new(&mlx_src)
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );

    // Link the C ABI shim library
    println!("cargo:rustc-link-lib=static=mlxrs_capi");

    // C++ stdlib on macOS
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    println!("cargo:rerun-if-env-changed=MLX_SRC");
}

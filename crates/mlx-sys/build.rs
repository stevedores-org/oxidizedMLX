fn main() {
    // Only build the C++ MLX library when the `cpp` feature is enabled.
    #[cfg(feature = "cpp")]
    {
        use std::env;

        let mlx_src = env::var("MLX_SRC").unwrap_or_else(|_| "../../mlx".to_string());

        let dst = cmake::Config::new(&mlx_src)
            .define("CMAKE_BUILD_TYPE", "Release")
            .build();

        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib").display()
        );
        println!("cargo:rustc-link-lib=static=mlxrs_capi");

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=c++");

        println!("cargo:rerun-if-env-changed=MLX_SRC");
    }
}

fn main() {
    // Only build/link the external C++ backend when the `ffi` feature is enabled.
    //
    // This is intended to be used like:
    //   MLX_SRC=/path/to/your/mlx-fork cargo build -p mlx-sys --features ffi
    println!("cargo:rerun-if-env-changed=MLX_SRC");
    println!("cargo:rerun-if-env-changed=MLX_BUILD_TYPE");
    println!("cargo:rerun-if-env-changed=MLX_CAPI_LIB_NAME");

    #[cfg(feature = "ffi")]
    {
        use std::env;
        use std::path::PathBuf;

        let mlx_src = env::var("MLX_SRC").unwrap_or_else(|_| {
            panic!(
                "MLX_SRC is required when building mlx-sys with --features ffi (e.g., MLX_SRC=/path/to/mlx)"
            )
        });

        let build_type = env::var("MLX_BUILD_TYPE").unwrap_or_else(|_| "Release".to_string());
        let lib_name = env::var("MLX_CAPI_LIB_NAME").unwrap_or_else(|_| "mlxrs_capi".to_string());

        let dst = cmake::Config::new(&mlx_src)
            .define("CMAKE_BUILD_TYPE", &build_type)
            .build();

        // Prefer the cmake crate's install lib dir, but also tolerate different layouts.
        let lib_dirs: [PathBuf; 3] = [
            dst.join("lib"),
            dst.join("lib64"),
            dst.join("build").join("lib"),
        ];
        for d in lib_dirs.iter() {
            if d.exists() {
                println!("cargo:rustc-link-search=native={}", d.display());
            }
        }

        println!("cargo:rustc-link-lib=static={lib_name}");

        #[cfg(target_os = "macos")]
        println!("cargo:rustc-link-lib=c++");
    }
}

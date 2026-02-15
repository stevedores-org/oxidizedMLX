// GEMM benchmarks — requires Metal backend with `metal_stream()` API.
// Currently a no-op stub; the real benchmark will be enabled once the
// Metal stream API lands.

fn main() {
    #[cfg(target_os = "macos")]
    eprintln!("gemm bench: metal_stream() not yet available — skipping");
}

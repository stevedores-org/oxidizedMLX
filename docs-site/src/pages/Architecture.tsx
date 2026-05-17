import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function Architecture() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Architecture</h1>
      <p className="text-lg text-zinc-400 mb-10">How oxidizedMLX is structured and the strangler-fig migration strategy.</p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Dependency Graph</h2>
      <CodeBlock>{`mlx-nn ─┐
mlx-io ─┤
        ├→ mlx-ops → mlx-core
mlx-metal ┘              ↑
mlx-cpu ──────────────────┘
mlx-sys (optional C++ FFI)
mlx-conformance → mlx-core + mlx-cpu (oracle)
mlx-cli → mlx-core + mlx-cpu`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Backend Trait</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        All operations target an abstract backend trait. <code className="text-orange-300/90 font-mono text-[13px]">mlx-cpu</code> is the reference implementation. <code className="text-orange-300/90 font-mono text-[13px]">mlx-metal</code> provides GPU-accelerated kernels on Apple Silicon.
      </p>
      <CodeBlock>{`// Simplified backend trait
pub trait Backend {
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn reduce_sum(&self, a: &Tensor, axes: &[usize]) -> Result<Tensor>;
    // ...
}`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Conformance Testing</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        <code className="text-orange-300/90 font-mono text-[13px]">mlx-conformance</code> compares Rust outputs against the Python MLX reference to ensure bitwise correctness before the Metal backend can become default.
      </p>

      <Callout icon="🎯">
        <strong className="text-zinc-100">Design principle:</strong> The CPU backend is the oracle. Any result difference between Metal and CPU is a Metal bug, never a CPU bug.
      </Callout>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">FFI Layer</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        <code className="text-orange-300/90 font-mono text-[13px]">mlx-sys</code> provides a narrow C ABI shim to the upstream C++ MLX core. This allows using battle-tested kernels where beneficial, while keeping the Rust API primary.
      </p>
    </Layout>
  );
}

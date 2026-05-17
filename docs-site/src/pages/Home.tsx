import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import { Card, CardGrid } from "@/components/Card";
import Callout from "@/components/Callout";

export default function Home() {
  return (
    <Layout>
      <div className="pb-10 mb-10 border-b border-zinc-800/60">
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-br from-zinc-100 to-zinc-400 bg-clip-text text-transparent leading-tight">oxidizedMLX</h1>
        <p className="text-lg text-zinc-400 mt-3 leading-relaxed max-w-xl">
          Rust-first MLX-compatible tensor runtime. Strangler-fig approach: safe Rust API first, correctness-first reference backend, then Metal acceleration.
        </p>
        <div className="flex gap-3 mt-6">
          <Link to="/getting-started" className="bg-orange-500 hover:bg-orange-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm">Get Started</Link>
          <a href="https://github.com/stevedores-org/oxidizedMLX" className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300">GitHub</a>
        </div>
      </div>

      <h2 className="text-2xl font-bold tracking-tight mb-3">Approach</h2>
      <p className="text-zinc-400 text-[15px] leading-relaxed mb-4">
        Not a big-bang rewrite of MLX. A <strong className="text-zinc-200">strangler-fig</strong> strategy:
      </p>
      <div className="border border-zinc-800 rounded-xl overflow-hidden my-4">
        <table className="w-full text-[13px]"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">1</td><td className="px-5 py-3">Safe Rust API (<code className="font-mono text-orange-300/90 text-[13px]">mlx-core</code>) + correctness-first CPU backend</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">2</td><td className="px-5 py-3">Conformance harness comparing Rust vs upstream MLX (Python)</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">3</td><td className="px-5 py-3">Upstream C++ core behind a narrow C ABI shim (<code className="font-mono text-orange-300/90 text-[13px]">mlx-sys</code>)</td></tr>
          <tr><td className="px-5 py-3 text-orange-400 font-mono font-medium">4</td><td className="px-5 py-3">Metal support (<code className="font-mono text-orange-300/90 text-[13px]">mlx-metal</code>) after correctness is locked</td></tr>
        </tbody></table>
      </div>

      <Callout icon="⚡">
        <strong className="text-zinc-100">Correctness first.</strong> The CPU backend is the oracle. Metal cannot be default until conformance tests pass.
      </Callout>

      <h2 className="text-2xl font-bold tracking-tight mt-12 mb-3">Workspace Crates</h2>
      <CardGrid>
        <Card to="/crates/mlx-core" title="mlx-core" tag="core" description="Public tensor API: shape, dtype, device, graph metadata, and basic ops plumbing." />
        <Card to="/crates/mlx-ops" title="mlx-ops" tag="ops" description="Pure ops layer: broadcasting, reductions, matmul targeting a backend trait." />
        <Card to="/crates/mlx-cpu" title="mlx-cpu" tag="backend" description="Correctness-first CPU backend. Reference kernels used as oracle for conformance." />
        <Card to="/crates/mlx-metal" title="mlx-metal" tag="backend" description="Metal backend runtime for Apple Silicon GPU acceleration." />
        <Card to="/crates/mlx-nn" title="mlx-nn" tag="neural" description="Neural network modules and parameter handling." />
        <Card to="/crates/mlx-io" title="mlx-io" tag="io" description="Safetensors + mmap weight loader for model loading." />
        <Card to="/crates/mlx-sys" title="mlx-sys" tag="ffi" description="FFI bindings to upstream MLX via a small C ABI shim." />
      </CardGrid>
    </Layout>
  );
}

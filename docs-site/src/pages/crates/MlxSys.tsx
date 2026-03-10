import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";
import Callout from "@/components/Callout";

export default function MlxSys() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-sys</h1>
        <StatusBadge status="wip" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        FFI bindings to upstream MLX C++ core via a narrow C ABI shim.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Building</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        Requires an MLX source checkout pointed to by <code className="text-orange-300/90 font-mono text-[13px]">MLX_SRC</code>.
      </p>
      <CodeBlock>{`export MLX_SRC=/path/to/mlx
just test-ffi
just clippy-ffi`}</CodeBlock>

      <Callout icon="⚠️">
        <strong className="text-zinc-100">Optional dependency.</strong> The pure-Rust stack (mlx-core + mlx-cpu) works without mlx-sys. FFI is only used where upstream kernels provide a significant advantage.
      </Callout>
    </Layout>
  );
}

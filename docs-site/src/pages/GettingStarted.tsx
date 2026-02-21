import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function GettingStarted() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Getting Started</h1>
      <p className="text-lg text-zinc-400 mb-10">Set up oxidizedMLX and run your first tensor operations.</p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Prerequisites</h2>
      <p className="text-zinc-400 text-[15px] mb-4">Rust 1.70+ and Cargo. For FFI builds, you'll need an MLX source checkout.</p>
      <CodeBlock>{`# Install just (task runner)
cargo install just

# Run all checks
just ci

# CLI smoke test
just smoke`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">FFI Build (Optional)</h2>
      <CodeBlock>{`export MLX_SRC=/path/to/your/mlx
just test-ffi
just clippy-ffi`}</CodeBlock>

      <Callout icon="💡">
        The FFI build is optional. The pure-Rust CPU backend works standalone for development and testing.
      </Callout>
    </Layout>
  );
}

import Layout from "@/components/Layout";
import StatusBadge from "@/components/StatusBadge";
import Callout from "@/components/Callout";

export default function MlxMetal() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-metal</h1>
        <StatusBadge status="planned" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Metal backend runtime scaffolding for Apple Silicon GPU acceleration.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Status</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        Metal support is Phase 4 of the roadmap. Correctness must be locked down via conformance testing before Metal kernels can be trusted.
      </p>

      <Callout icon="🏗️">
        Scaffolding exists but kernels are not yet implemented. See the <a href="https://github.com/stevedores-org/oxidizedMLX/blob/main/docs/DELIVERY_PLAN.md" className="text-orange-400 underline">delivery plan</a> for timelines.
      </Callout>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Approach</h2>
      <ul className="list-disc list-inside text-zinc-400 text-[15px] space-y-2">
        <li>Implements the same <code className="text-orange-300/90 font-mono text-[13px]">Backend</code> trait as mlx-cpu</li>
        <li>Metal Performance Shaders (MPS) for matmul and convolutions</li>
        <li>Custom MSL compute kernels for element-wise and reduction ops</li>
        <li>Must pass all conformance tests before becoming default</li>
      </ul>
    </Layout>
  );
}

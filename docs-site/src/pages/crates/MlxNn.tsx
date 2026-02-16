import Layout from "@/components/Layout";
import StatusBadge from "@/components/StatusBadge";

export default function MlxNn() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-nn</h1>
        <StatusBadge status="planned" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Neural network modules and parameter handling — Linear, LayerNorm, Embedding, Attention, and more.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Planned Modules</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px]">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">Linear</td><td className="px-5 py-3">Dense layer with optional bias</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">LayerNorm</td><td className="px-5 py-3">Layer normalization</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">RMSNorm</td><td className="px-5 py-3">Root mean square normalization</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">Embedding</td><td className="px-5 py-3">Token embeddings</td></tr>
          <tr><td className="px-5 py-3 text-orange-400 font-mono font-medium">Attention</td><td className="px-5 py-3">Multi-head attention</td></tr>
        </tbody></table>
      </div>
    </Layout>
  );
}

import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";

export default function MlxOps() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-ops</h1>
        <StatusBadge status="wip" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Pure ops layer targeting a backend trait: broadcasting, reductions, matmul, element-wise operations.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Op Categories</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden text-[13px] mb-6">
        <table className="w-full"><tbody className="text-zinc-400">
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">Element-wise</td><td className="px-5 py-3">add, sub, mul, div, exp, log, sin, cos, relu, sigmoid</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">Reductions</td><td className="px-5 py-3">sum, mean, max, min, argmax, argmin over axes</td></tr>
          <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-orange-400 font-mono font-medium">Linear algebra</td><td className="px-5 py-3">matmul, transpose, reshape, view</td></tr>
          <tr><td className="px-5 py-3 text-orange-400 font-mono font-medium">Broadcasting</td><td className="px-5 py-3">NumPy-compatible broadcast semantics</td></tr>
        </tbody></table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Backend Dispatch</h2>
      <CodeBlock>{`// ops are backend-agnostic
pub fn matmul<B: Backend>(
    backend: &B,
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor> {
    backend.matmul(a, b)
}`}</CodeBlock>
    </Layout>
  );
}

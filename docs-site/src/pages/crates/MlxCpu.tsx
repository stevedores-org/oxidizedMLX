import Layout from "@/components/Layout";
import Callout from "@/components/Callout";
import StatusBadge from "@/components/StatusBadge";

export default function MlxCpu() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-cpu</h1>
        <StatusBadge status="wip" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Correctness-first CPU backend. The reference implementation used as an oracle for conformance testing against the Metal backend.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Purpose</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        <code className="text-orange-300/90 font-mono text-[13px]">mlx-cpu</code> prioritizes <strong className="text-zinc-200">correctness</strong> over performance.
        It serves two roles:
      </p>
      <ul className="list-disc list-inside text-zinc-400 text-[15px] space-y-2 mb-6">
        <li>Development backend for writing and testing ops</li>
        <li>Oracle for the conformance harness: any mismatch with Metal is a Metal bug</li>
      </ul>

      <Callout icon="🧪">
        The CPU backend passes all conformance tests by definition — it <em>is</em> the reference.
      </Callout>
    </Layout>
  );
}

import Layout from "@/components/Layout";
import StatusBadge from "@/components/StatusBadge";

export default function MlxIo() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-io</h1>
        <StatusBadge status="planned" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Safetensors + mmap weight loader for efficient model loading.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Features</h2>
      <ul className="list-disc list-inside text-zinc-400 text-[15px] space-y-2">
        <li>Memory-mapped file I/O for zero-copy weight loading</li>
        <li>Safetensors format support for safe, aligned tensor storage</li>
        <li>Lazy loading for models larger than RAM</li>
        <li>Dtype conversion on load</li>
      </ul>
    </Layout>
  );
}

import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import StatusBadge from "@/components/StatusBadge";

export default function MlxCore() {
  return (
    <Layout>
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-3xl font-extrabold tracking-tight">mlx-core</h1>
        <StatusBadge status="wip" />
      </div>
      <p className="text-lg text-zinc-400 mb-10">
        Public tensor API. Shape, dtype, device, graph metadata, and basic ops plumbing.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Key Types</h2>
      <CodeBlock>{`pub struct Tensor {
    shape: Shape,
    dtype: DType,
    device: Device,
    data: TensorData,
}

pub enum DType {
    F16, BF16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

pub enum Device {
    Cpu,
    Metal(usize),
}`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Shape & Broadcasting</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        <code className="text-orange-300/90 font-mono text-[13px]">Shape</code> tracks dimensions and strides. Broadcasting follows NumPy semantics — right-aligned, size-1 dimensions are expanded.
      </p>
    </Layout>
  );
}

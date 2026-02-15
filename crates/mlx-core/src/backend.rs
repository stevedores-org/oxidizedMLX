//! Backend trait and Stream — pluggable compute engine for tensor evaluation.
//!
//! A `Backend` knows how to execute a single graph node (op + inputs → output).
//! A `Stream` binds a `Backend` to a lazy computation `Graph`, managing
//! materialized buffers and evaluation scheduling.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

use smallvec::SmallVec;

use crate::Result;
use crate::graph::{Graph, Node, NodeId, OpKind, TensorMeta};
use crate::types::{DType, Shape};

/// CSE key for deduplicating pure operations.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CseKey {
    op: OpKind,
    inputs: SmallVec<[NodeId; 2]>,
    meta: TensorMeta,
}

/// CSE key for deduplicating constant nodes with identical data.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ConstKey {
    meta: TensorMeta,
    payload_hash: u64,
}

/// Hash an f32 slice by converting each element to its bit pattern.
fn hash_f32_slice(data: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    data.len().hash(&mut hasher);
    for &v in data {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Materialized input data passed to a backend for evaluation.
pub struct NodeInput<'a> {
    pub data: &'a [f32],
    pub shape: &'a Shape,
    pub dtype: DType,
}

/// Pluggable compute backend.
///
/// Backends evaluate individual graph nodes. The `Stream` handles scheduling
/// (topological sort) and buffer management; the backend only needs to
/// implement the actual kernel dispatch.
pub trait Backend: Send + Sync {
    /// Evaluate a single op node given its materialized inputs.
    fn eval_node(
        &self,
        op: &OpKind,
        inputs: &[NodeInput<'_>],
        output_meta: &TensorMeta,
    ) -> Result<Vec<f32>>;
}

/// A computation stream binding a graph to a backend.
///
/// Operations on tensors add nodes to the stream's graph lazily.
/// Calling `eval()` topologically sorts and evaluates pending nodes.
pub struct Stream {
    graph: Mutex<Graph>,
    backend: Box<dyn Backend>,
    buffers: Mutex<HashMap<NodeId, Vec<f32>>>,
    cse_ops: Mutex<HashMap<CseKey, NodeId>>,
    cse_consts: Mutex<HashMap<ConstKey, NodeId>>,
}

impl Stream {
    /// Create a new stream with the given backend.
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            graph: Mutex::new(Graph::new()),
            backend,
            buffers: Mutex::new(HashMap::new()),
            cse_ops: Mutex::new(HashMap::new()),
            cse_consts: Mutex::new(HashMap::new()),
        }
    }

    /// Add a constant node (data already known).
    ///
    /// Identical constants (same meta + payload) are deduplicated.
    pub fn add_constant(&self, data: Vec<f32>, meta: TensorMeta) -> NodeId {
        let key = ConstKey {
            meta: meta.clone(),
            payload_hash: hash_f32_slice(&data),
        };

        let mut cse_consts = self.cse_consts.lock().unwrap();
        if let Some(&existing) = cse_consts.get(&key) {
            // Verify payload equality to guard against hash collisions.
            let buffers = self.buffers.lock().unwrap();
            if let Some(existing_data) = buffers.get(&existing)
                && data.len() == existing_data.len()
                && data
                    .iter()
                    .zip(existing_data.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return existing;
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.add_node(OpKind::Constant, SmallVec::new(), meta);
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, data);
        cse_consts.insert(key, id);
        id
    }

    /// Add an operation node to the graph.
    ///
    /// CSE-eligible ops with identical (op, inputs, meta) are deduplicated.
    pub fn add_op(&self, op: OpKind, inputs: SmallVec<[NodeId; 2]>, meta: TensorMeta) -> NodeId {
        if op.is_cse_eligible() {
            let key = CseKey {
                op: op.clone(),
                inputs: inputs.clone(),
                meta: meta.clone(),
            };

            let mut cse_ops = self.cse_ops.lock().unwrap();
            if let Some(&existing) = cse_ops.get(&key) {
                return existing;
            }

            let mut graph = self.graph.lock().unwrap();
            let id = graph.add_node(op, inputs, meta);
            cse_ops.insert(key, id);
            id
        } else {
            let mut graph = self.graph.lock().unwrap();
            graph.add_node(op, inputs, meta)
        }
    }

    /// Evaluate all nodes needed to materialize the given output.
    pub fn eval(&self, output: NodeId) -> Result<()> {
        // Already materialized?
        if self.buffers.lock().unwrap().contains_key(&output) {
            return Ok(());
        }

        // Topo-sort the subgraph rooted at `output`.
        let order = {
            let graph = self.graph.lock().unwrap();
            graph.topo_sort(&[output])
        };

        // Evaluate each node in order. Never hold both locks simultaneously.
        for &node_id in &order {
            if self.buffers.lock().unwrap().contains_key(&node_id) {
                continue;
            }

            // Step 1: get node info (graph lock only).
            let node: Node = {
                let graph = self.graph.lock().unwrap();
                graph
                    .get(node_id)
                    .cloned()
                    .ok_or_else(|| crate::MlxError::InvalidArgument("missing graph node".into()))?
            };

            // Step 2: get input metadata from graph (graph lock only).
            let input_metas: Vec<TensorMeta> = {
                let graph = self.graph.lock().unwrap();
                node.inputs
                    .iter()
                    .map(|&id| {
                        graph
                            .get(id)
                            .expect("input node missing from graph")
                            .meta
                            .clone()
                    })
                    .collect()
            };

            // Step 3: gather input data + run backend (buffers lock only for reads).
            let input_buffers: Vec<Vec<f32>> = {
                let buffers = self.buffers.lock().unwrap();
                node.inputs
                    .iter()
                    .map(|&id| {
                        buffers
                            .get(&id)
                            .expect("input node should be evaluated before dependents")
                            .clone()
                    })
                    .collect()
            };

            let inputs: Vec<NodeInput<'_>> = input_buffers
                .iter()
                .zip(input_metas.iter())
                .map(|(data, meta)| NodeInput {
                    data: data.as_slice(),
                    shape: &meta.shape,
                    dtype: meta.dtype,
                })
                .collect();

            let result = self.backend.eval_node(&node.op, &inputs, &node.meta)?;

            // Step 4: store result (buffers lock only for write).
            self.buffers.lock().unwrap().insert(node_id, result);
        }

        Ok(())
    }

    /// Get materialized buffer data for a node (must call eval first).
    pub fn get_buffer(&self, id: NodeId) -> Option<Vec<f32>> {
        self.buffers.lock().unwrap().get(&id).cloned()
    }

    /// Get a clone of a graph node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.graph.lock().unwrap().get(id).cloned()
    }

    /// Topological sort of the subgraph rooted at the given outputs.
    pub fn topo_sort(&self, outputs: &[NodeId]) -> Vec<NodeId> {
        self.graph.lock().unwrap().topo_sort(outputs)
    }

    /// Number of nodes in the graph.
    pub fn graph_node_count(&self) -> usize {
        self.graph.lock().unwrap().len()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream").finish_non_exhaustive()
    }
}

/// Which backend to use by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultBackend {
    Cpu,
    Metal,
}

/// Determine the default backend via env var → compile-time feature → CPU fallback.
///
/// Priority:
/// 1. `MLX_RS_BACKEND` env var (`"cpu"` or `"metal"`, case-insensitive)
/// 2. Cargo feature `default-backend-metal`
/// 3. Cargo feature `default-backend-cpu`
/// 4. CPU fallback
pub fn default_backend() -> DefaultBackend {
    if let Ok(val) = std::env::var("MLX_RS_BACKEND") {
        match val.to_lowercase().as_str() {
            "metal" => return DefaultBackend::Metal,
            "cpu" => return DefaultBackend::Cpu,
            _ => {} // ignore unrecognized values, fall through
        }
    }

    if cfg!(feature = "default-backend-metal") {
        DefaultBackend::Metal
    } else {
        DefaultBackend::Cpu
    }
}

/// The default stream using the CPU reference backend.
///
/// When `default_backend()` returns `Metal`, higher-level crates (e.g. `mlx-cpu`,
/// `mlx-metal`) are responsible for constructing the appropriate stream.
/// This static always provides a working CPU fallback.
static DEFAULT_STREAM: LazyLock<Arc<Stream>> =
    LazyLock::new(|| Arc::new(Stream::new(Box::new(crate::cpu_kernels::CpuRefBackend))));

/// Get the default computation stream.
pub fn default_stream() -> Arc<Stream> {
    Arc::clone(&DEFAULT_STREAM)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorMeta;

    #[test]
    fn test_stream_constant() {
        let stream = default_stream();
        let id = stream.add_constant(
            vec![1.0, 2.0, 3.0],
            TensorMeta {
                shape: Shape::new(vec![3]),
                dtype: DType::F32,
            },
        );
        stream.eval(id).unwrap();
        assert_eq!(stream.get_buffer(id).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_stream_add_op() {
        let stream = default_stream();
        let a = stream.add_constant(
            vec![1.0, 2.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let b = stream.add_constant(
            vec![3.0, 4.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let c = stream.add_op(
            OpKind::Add,
            smallvec::SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        stream.eval(c).unwrap();
        assert_eq!(stream.get_buffer(c).unwrap(), vec![4.0, 6.0]);
    }
}

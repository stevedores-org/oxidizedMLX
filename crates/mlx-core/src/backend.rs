//! Backend trait and Context — pluggable compute engine for tensor evaluation.
//!
//! A `Backend` knows how to execute a single graph node (op + inputs → output).
//! A `Context` binds a `Backend` to a lazy computation `Graph`, managing
//! materialized buffers and evaluation scheduling.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::Result;
use crate::graph::{Graph, GraphView, Node, NodeId, OpKind, TensorMeta};
use crate::types::{DType, Shape};

/// Materialized data buffer (currently just Vec<f32> for simplicity).
pub type Materialized = Vec<f32>;

/// Materialized input data passed to a backend for evaluation.
pub struct NodeInput<'a> {
    pub data: &'a [f32],
    pub shape: &'a Shape,
    pub dtype: DType,
}

/// Pluggable compute backend.
pub trait Backend: Send + Sync {
    /// Evaluate a single op node given its materialized inputs.
    fn eval_node(
        &self,
        op: &OpKind,
        inputs: &[NodeInput<'_>],
        output_meta: &TensorMeta,
    ) -> Result<Materialized>;

    /// Realize a batch of nodes. 
    /// Default implementation just calls eval_node in order, but backends can override.
    fn realize(&self, graph: &dyn GraphView, nodes: &[NodeId], cache: &Mutex<HashMap<NodeId, Materialized>>) -> Result<Vec<Materialized>> {
        let mut results = Vec::with_capacity(nodes.len());
        for &nid in nodes {
            let node = graph.node(nid);
            
            // Collect inputs from cache
            let mut cache_lock = cache.lock().unwrap();
            let mut inputs = Vec::with_capacity(node.inputs.len());
            for &inp_id in &node.inputs {
                let data = cache_lock.get(&inp_id).ok_or_else(|| {
                    crate::MlxError::Backend("missing input in realize cache")
                })?;
                let meta = &graph.node(inp_id).meta;
                inputs.push(NodeInput {
                    data,
                    shape: &meta.shape,
                    dtype: meta.dtype,
                });
            }

            let res = self.eval_node(&node.op, &inputs, &node.meta)?;
            results.push(res.clone());
            cache_lock.insert(nid, res);
        }
        Ok(results)
    }
}

/// A computation context binding a graph to a backend.
///
/// Operations on tensors add nodes to the context's graph lazily.
/// Calling `eval()` topologically sorts and evaluates pending nodes.
pub struct Context {
    graph: Mutex<Graph>,
    backend: Box<dyn Backend>,
    buffers: Mutex<HashMap<NodeId, Materialized>>,
}

impl Context {
    /// Create a new context with the given backend.
    pub fn new(backend: Box<dyn Backend>) -> Self {
        Self {
            graph: Mutex::new(Graph::new()),
            backend,
            buffers: Mutex::new(HashMap::new()),
        }
    }

    /// Add a constant node (data already known).
    pub fn add_constant(&self, data: Materialized, meta: TensorMeta) -> NodeId {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.add_node(OpKind::Constant, smallvec::SmallVec::new(), meta);
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, data);
        id
    }

    /// Add an operation node to the graph.
    pub fn add_op(
        &self,
        op: OpKind,
        inputs: smallvec::SmallVec<[NodeId; 2]>,
        meta: TensorMeta,
    ) -> NodeId {
        let mut graph = self.graph.lock().unwrap();
        graph.add_node(op, inputs, meta)
    }

    /// Evaluate all nodes needed to materialize the given output.
    pub fn eval(&self, output: NodeId) -> Result<()> {
        self.eval_many(&[output]).map(|_| ())
    }

    /// Evaluate multiple outputs efficiently with a topo schedule.
    pub fn eval_many(&self, outputs: &[NodeId]) -> Result<Vec<Materialized>> {
        // Already materialized?
        {
            let buffers = self.buffers.lock().unwrap();
            if outputs.iter().all(|&id| buffers.contains_key(&id)) {
                return Ok(outputs
                    .iter()
                    .map(|&id| buffers.get(&id).unwrap().clone())
                    .collect());
            }
        }

        // Topo-sort the subgraph rooted at `outputs`.
        let sched = {
            let graph = self.graph.lock().unwrap();
            crate::schedule::topo_schedule(&*graph, outputs)?
        };

        // Determine which nodes actually need to be computed (not in buffers cache).
        let mut to_compute = Vec::new();
        {
            let buffers = self.buffers.lock().unwrap();
            for &nid in &sched.topo {
                if !buffers.contains_key(&nid) {
                    to_compute.push(nid);
                }
            }
        }

        if !to_compute.is_empty() {
            let mats = {
                let graph = self.graph.lock().unwrap();
                self.backend.realize(&*graph, &to_compute, &self.buffers)?
            };

            if mats.len() != to_compute.len() {
                return Err(crate::MlxError::Backend(
                    "backend returned wrong number of outputs",
                ));
            }

            let mut buffers = self.buffers.lock().unwrap();
            for (nid, mat) in to_compute.into_iter().zip(mats.into_iter()) {
                buffers.insert(nid, mat);
            }
        }

        // Return requested outputs from cache.
        let buffers = self.buffers.lock().unwrap();
        let mut out = Vec::with_capacity(outputs.len());
        for &nid in outputs {
            let m = buffers
                .get(&nid)
                .cloned()
                .ok_or_else(|| crate::MlxError::Backend("missing cached output"))?;
            out.push(m);
        }
        Ok(out)
    }

    /// Get materialized buffer data for a node (must call eval first).
    pub fn get_buffer(&self, id: NodeId) -> Option<Materialized> {
        self.buffers.lock().unwrap().get(&id).cloned()
    }

    /// Get a clone of a graph node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.graph.lock().unwrap().get(id).cloned()
    }

    /// Topological sort of the subgraph rooted at the given outputs.
    pub fn topo_sort(&self, outputs: &[NodeId]) -> Result<Vec<NodeId>> {
        let graph = self.graph.lock().unwrap();
        let sched = crate::schedule::topo_schedule(&*graph, outputs)?;
        Ok(sched.topo)
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

/// The default context using the built-in CPU reference backend.
static DEFAULT_CONTEXT: LazyLock<Arc<Context>> =
    LazyLock::new(|| Arc::new(Context::new(Box::new(crate::cpu_kernels::CpuRefBackend))));

/// Get the default computation context.
pub fn default_context() -> Arc<Context> {
    Arc::clone(&DEFAULT_CONTEXT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TensorMeta;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_context_constant() {
        let ctx = default_context();
        let id = ctx.add_constant(
            vec![1.0, 2.0, 3.0],
            TensorMeta {
                shape: Shape::new(vec![3]),
                dtype: DType::F32,
            },
        );
        ctx.eval(id).unwrap();
        assert_eq!(ctx.get_buffer(id).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_context_add_op() {
        let ctx = default_context();
        let a = ctx.add_constant(
            vec![1.0, 2.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let b = ctx.add_constant(
            vec![3.0, 4.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let c = ctx.add_op(
            OpKind::Add,
            smallvec::SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        ctx.eval(c).unwrap();
        assert_eq!(ctx.get_buffer(c).unwrap(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_eval_is_memoized_no_recompute() {
        struct CountingBackend {
            inner: crate::cpu_kernels::CpuRefBackend,
            calls: Arc<AtomicUsize>,
        }

        impl Backend for CountingBackend {
            fn eval_node(
                &self,
                op: &OpKind,
                inputs: &[NodeInput<'_>],
                output_meta: &TensorMeta,
            ) -> Result<Vec<f32>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.inner.eval_node(op, inputs, output_meta)
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let ctx = Context::new(Box::new(CountingBackend {
            inner: crate::cpu_kernels::CpuRefBackend,
            calls: Arc::clone(&calls),
        }));

        let a = ctx.add_constant(
            vec![1.0, 2.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let b = ctx.add_constant(
            vec![3.0, 4.0],
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );

        // Two op nodes: add then neg
        let add = ctx.add_op(
            OpKind::Add,
            smallvec::SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );
        let out = ctx.add_op(
            OpKind::Neg,
            smallvec::SmallVec::from_slice(&[add]),
            TensorMeta {
                shape: Shape::new(vec![2]),
                dtype: DType::F32,
            },
        );

        ctx.eval(out).unwrap();
        let after_first = calls.load(Ordering::Relaxed);
        assert_eq!(after_first, 2);
        assert_eq!(ctx.get_buffer(out).unwrap(), vec![-4.0, -6.0]);

        // Repeated eval should not call into the backend again.
        ctx.eval(out).unwrap();
        let after_second = calls.load(Ordering::Relaxed);
        assert_eq!(after_first, after_second);

        // Evaluating already-materialized intermediates should also be a no-op.
        ctx.eval(add).unwrap();
        let after_third = calls.load(Ordering::Relaxed);
        assert_eq!(after_second, after_third);
    }
}

//! Lazy computation graph IR.
//!
//! Tensors are handles to nodes in this graph. Computation is deferred until
//! `eval()` is called, at which point the scheduler topologically sorts the
//! graph and dispatches to the active backend.

use crate::types::{DType, Shape};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Unique identifier for a node in the computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u64);

/// Metadata about a tensor (known before materialization).
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub shape: Shape,
    pub dtype: DType,
}

/// A node in the lazy computation graph.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: OpKind,
    pub inputs: SmallVec<[NodeId; 2]>,
    pub meta: TensorMeta,
}

/// The set of operations supported by the graph IR.
#[derive(Clone, Debug)]
pub enum OpKind {
    // ── Sources ─────────────────────────────────────────────────────────
    /// Constant tensor (data already materialized).
    Constant,
    /// Parameter (learnable weight, data provided externally).
    Parameter,

    // ── Elementwise ─────────────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Neg,

    // ── Reductions ──────────────────────────────────────────────────────
    Sum {
        axis: Option<i32>,
    },
    Mean {
        axis: Option<i32>,
    },
    Max {
        axis: Option<i32>,
    },

    // ── Linear algebra ──────────────────────────────────────────────────
    MatMul,

    // ── Shape manipulation ──────────────────────────────────────────────
    Reshape {
        new_shape: Shape,
    },
    Transpose {
        axes: Option<Vec<usize>>,
    },

    // ── Activations ─────────────────────────────────────────────────────
    Softmax {
        axis: i32,
    },
    Silu,
    Gelu,

    // ── Normalization ───────────────────────────────────────────────────
    LayerNorm {
        eps: f32,
    },
    RmsNorm {
        eps: f32,
    },

    // ── Broadcasting ──────────────────────────────────────────────────
    /// Broadcast a tensor to a target shape (numpy-style rules).
    Broadcast {
        target_shape: Shape,
    },

    // ── Backward (VJP) ops ──────────────────────────────────────────
    /// LayerNorm backward: inputs = [grad_output, input], produces grad_input.
    LayerNormVjp {
        eps: f32,
    },
    /// RmsNorm backward: inputs = [grad_output, input], produces grad_input.
    RmsNormVjp {
        eps: f32,
    },
}

/// The computation graph arena.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    next_id: u64,
    cse: HashMap<CseKey, NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node and return its ID.
    pub fn add_node(
        &mut self,
        op: OpKind,
        inputs: SmallVec<[NodeId; 2]>,
        meta: TensorMeta,
    ) -> NodeId {
        self.add_node_raw(op, inputs, meta)
    }

    /// Add a node without CSE.
    pub fn add_node_raw(
        &mut self,
        op: OpKind,
        inputs: SmallVec<[NodeId; 2]>,
        meta: TensorMeta,
    ) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            meta,
        });
        id
    }

    /// Add a node with CSE. For constants, include a payload hash if available.
    pub fn intern_node(
        &mut self,
        op: OpKind,
        inputs: SmallVec<[NodeId; 2]>,
        meta: TensorMeta,
        const_hash: Option<u64>,
    ) -> NodeId {
        if !is_cse_eligible(&op) {
            return self.add_node_raw(op, inputs, meta);
        }

        let key = CseKey {
            op_key: OpKey::from_op(&op),
            inputs: inputs.clone(),
            meta_sig: MetaSig::new(&meta),
            const_hash,
        };

        if let Some(&existing) = self.cse.get(&key) {
            return existing;
        }

        let id = self.add_node_raw(op, inputs, meta);
        self.cse.insert(key, id);
        id
    }

    /// Get a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Topological sort of the graph rooted at `outputs`.
    pub fn topo_sort(&self, outputs: &[NodeId]) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        for &out in outputs {
            self.topo_visit(out, &mut visited, &mut order);
        }

        order
    }

    fn topo_visit(
        &self,
        id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if let Some(node) = self.get(id) {
            for &input in &node.inputs {
                self.topo_visit(input, visited, order);
            }
        }
        order.push(id);
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MetaSig {
    dtype: DType,
    shape: Vec<i64>,
}

impl MetaSig {
    fn new(meta: &TensorMeta) -> Self {
        Self {
            dtype: meta.dtype,
            shape: meta.shape.0.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum OpKey {
    Constant,
    Parameter,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Sum { axis: Option<i32> },
    Mean { axis: Option<i32> },
    Max { axis: Option<i32> },
    MatMul,
    Reshape { new_shape: Vec<i64> },
    Transpose { axes: Option<Vec<usize>> },
    Softmax { axis: i32 },
    Silu,
    Gelu,
    LayerNorm { eps_bits: u32 },
    RmsNorm { eps_bits: u32 },
    Broadcast { target_shape: Vec<i64> },
}

impl OpKey {
    fn from_op(op: &OpKind) -> Self {
        match op {
            OpKind::Constant => OpKey::Constant,
            OpKind::Parameter => OpKey::Parameter,
            OpKind::Add => OpKey::Add,
            OpKind::Sub => OpKey::Sub,
            OpKind::Mul => OpKey::Mul,
            OpKind::Div => OpKey::Div,
            OpKind::Neg => OpKey::Neg,
            OpKind::Sum { axis } => OpKey::Sum { axis: *axis },
            OpKind::Mean { axis } => OpKey::Mean { axis: *axis },
            OpKind::Max { axis } => OpKey::Max { axis: *axis },
            OpKind::MatMul => OpKey::MatMul,
            OpKind::Reshape { new_shape } => OpKey::Reshape {
                new_shape: new_shape.0.clone(),
            },
            OpKind::Transpose { axes } => OpKey::Transpose { axes: axes.clone() },
            OpKind::Softmax { axis } => OpKey::Softmax { axis: *axis },
            OpKind::Silu => OpKey::Silu,
            OpKind::Gelu => OpKey::Gelu,
            OpKind::LayerNorm { eps } => OpKey::LayerNorm {
                eps_bits: eps.to_bits(),
            },
            OpKind::RmsNorm { eps } => OpKey::RmsNorm {
                eps_bits: eps.to_bits(),
            },
            OpKind::Broadcast { target_shape } => OpKey::Broadcast {
                target_shape: target_shape.0.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CseKey {
    op_key: OpKey,
    inputs: SmallVec<[NodeId; 2]>,
    meta_sig: MetaSig,
    const_hash: Option<u64>,
}

fn is_cse_eligible(op: &OpKind) -> bool {
    !matches!(op, OpKind::Parameter)
}

pub fn hash_f32_payload(data: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.len().hash(&mut h);
    for &x in data {
        x.to_bits().hash(&mut h);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_topo_sort() {
        let mut g = Graph::new();
        let a = g.add_node(
            OpKind::Constant,
            SmallVec::new(),
            TensorMeta {
                shape: Shape::new(vec![2, 3]),
                dtype: DType::F32,
            },
        );
        let b = g.add_node(
            OpKind::Constant,
            SmallVec::new(),
            TensorMeta {
                shape: Shape::new(vec![2, 3]),
                dtype: DType::F32,
            },
        );
        let c = g.add_node(
            OpKind::Add,
            SmallVec::from_slice(&[a, b]),
            TensorMeta {
                shape: Shape::new(vec![2, 3]),
                dtype: DType::F32,
            },
        );

        let order = g.topo_sort(&[c]);
        assert_eq!(order.len(), 3);
        // a and b before c
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        let pos_c = order.iter().position(|&id| id == c).unwrap();
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_cse_dedups_constants_and_ops() {
        let mut g = Graph::new();
        let meta = TensorMeta {
            shape: Shape::new(vec![2]),
            dtype: DType::F32,
        };
        let hash = hash_f32_payload(&[1.0, 2.0]);
        let a = g.intern_node(OpKind::Constant, SmallVec::new(), meta.clone(), Some(hash));
        let b = g.intern_node(OpKind::Constant, SmallVec::new(), meta.clone(), Some(hash));
        assert_eq!(a, b);
        assert_eq!(g.len(), 1);

        let add1 = g.intern_node(
            OpKind::Add,
            SmallVec::from_slice(&[a, b]),
            meta.clone(),
            None,
        );
        let add2 = g.intern_node(
            OpKind::Add,
            SmallVec::from_slice(&[a, b]),
            meta.clone(),
            None,
        );
        assert_eq!(add1, add2);
        assert_eq!(g.len(), 2);
    }
}

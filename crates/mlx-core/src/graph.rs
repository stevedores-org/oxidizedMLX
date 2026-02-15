//! Lazy computation graph IR.
//!
//! Tensors are handles to nodes in this graph. Computation is deferred until
//! `eval()` is called, at which point the scheduler topologically sorts the
//! graph and dispatches to the active backend.

use crate::types::{DType, Shape};
use smallvec::SmallVec;

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
}

/// The computation graph arena.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    next_id: u64,
}

/// Read-only view of a computation graph for scheduling.
pub trait GraphView {
    fn node(&self, id: NodeId) -> &Node;
    fn contains(&self, id: NodeId) -> bool;
}

impl GraphView for Graph {
    fn node(&self, id: NodeId) -> &Node {
        self.get(id).expect("node not found in graph")
    }
    fn contains(&self, id: NodeId) -> bool {
        self.get(id).is_some()
    }
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
}

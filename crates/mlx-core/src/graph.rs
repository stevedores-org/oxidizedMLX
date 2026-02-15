//! Lazy computation graph IR.
//!
//! Tensors are handles to nodes in this graph. Computation is deferred until
//! `eval()` is called, at which point the scheduler topologically sorts the
//! graph and dispatches to the active backend.

use std::hash::{Hash, Hasher};

use crate::types::{DType, Shape};
use smallvec::SmallVec;

/// Unique identifier for a node in the computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u64);

/// Metadata about a tensor (known before materialization).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

impl PartialEq for OpKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Constant, Self::Constant) => true,
            (Self::Parameter, Self::Parameter) => true,
            (Self::Add, Self::Add) => true,
            (Self::Sub, Self::Sub) => true,
            (Self::Mul, Self::Mul) => true,
            (Self::Div, Self::Div) => true,
            (Self::Neg, Self::Neg) => true,
            (Self::Sum { axis: a }, Self::Sum { axis: b }) => a == b,
            (Self::Mean { axis: a }, Self::Mean { axis: b }) => a == b,
            (Self::Max { axis: a }, Self::Max { axis: b }) => a == b,
            (Self::MatMul, Self::MatMul) => true,
            (Self::Reshape { new_shape: a }, Self::Reshape { new_shape: b }) => a == b,
            (Self::Transpose { axes: a }, Self::Transpose { axes: b }) => a == b,
            (Self::Softmax { axis: a }, Self::Softmax { axis: b }) => a == b,
            (Self::Silu, Self::Silu) => true,
            (Self::Gelu, Self::Gelu) => true,
            (Self::LayerNorm { eps: a }, Self::LayerNorm { eps: b }) => a.to_bits() == b.to_bits(),
            (Self::RmsNorm { eps: a }, Self::RmsNorm { eps: b }) => a.to_bits() == b.to_bits(),
            (Self::Broadcast { target_shape: a }, Self::Broadcast { target_shape: b }) => a == b,
            _ => false,
        }
    }
}

impl Eq for OpKind {}

impl Hash for OpKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Sum { axis } | Self::Mean { axis } | Self::Max { axis } => axis.hash(state),
            Self::Reshape { new_shape } => new_shape.hash(state),
            Self::Transpose { axes } => axes.hash(state),
            Self::Softmax { axis } => axis.hash(state),
            Self::LayerNorm { eps } => eps.to_bits().hash(state),
            Self::RmsNorm { eps } => eps.to_bits().hash(state),
            Self::Broadcast { target_shape } => target_shape.hash(state),
            _ => {}
        }
    }
}

impl OpKind {
    /// Returns `true` for pure ops that can be deduplicated via CSE.
    /// `Constant` and `Parameter` are not eligible — constants are handled
    /// separately in `add_constant`, and parameters are unique by definition.
    pub fn is_cse_eligible(&self) -> bool {
        !matches!(self, Self::Constant | Self::Parameter)
    }
}

/// The computation graph arena.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    next_id: u64,
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

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
    Exp,
    Log,

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

    // ── Positional encoding ────────────────────────────────────────────
    /// Rotary positional embeddings (RoPE).
    /// Applied in-place to interleaved pairs of the last dimension.
    Rope {
        rotary_dim: usize,
        pos_offset: usize,
        theta: f32,
    },

    // ── Broadcasting ──────────────────────────────────────────────────
    /// Broadcast a tensor to a target shape (numpy-style rules).
    Broadcast {
        target_shape: Shape,
    },

    // ── Attention ──────────────────────────────────────────────────
    /// Fused scale + causal-mask + softmax along last axis.
    /// Input: scores [Tq, Tk], output: probs [Tq, Tk]
    ScaledMaskedSoftmax {
        scale: f32,
        causal: bool,
    },

    /// Full single-head attention composition.
    /// Inputs: [Q, K, V] where Q=[Tq,Dh], K=[Tk,Dh], V=[Tk,Dh]
    /// Output: Y=[Tq,Dh]
    Attention {
        scale: f32,
        causal: bool,
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
    /// Softmax backward: inputs = [grad_output, softmax_output], produces grad_input.
    SoftmaxVjp {
        axis: i32,
    },
    /// SiLU backward: inputs = [grad_output, original_input], produces grad_input.
    SiluVjp,
    /// GELU backward: inputs = [grad_output, original_input], produces grad_input.
    GeluVjp,

    // ── Elementwise (misc) ──────────────────────────────────────────
    /// Element-wise square root.
    Sqrt,

    // ── Rotary Positional Embeddings ───────────────────────────────────
    #[cfg_attr(target_os = "macos", doc = "Apply rotary positional embeddings.")]
    RoPE {
        base: f32,
        offset: usize,
        traditional: bool,
    },
}

/// The computation graph arena.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    next_id: u64,
    cse: HashMap<CseKey, NodeId>,
    const_payloads: HashMap<NodeId, Vec<f32>>,
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
        const_payload: Option<&[f32]>,
    ) -> NodeId {
        if !is_cse_eligible(&op) {
            return self.add_node_raw(op, inputs, meta);
        }

        let mut inputs = inputs;
        normalize_inputs_for_cse(&op, &mut inputs);

        let const_hash = const_payload.map(hash_f32_payload);
        let key = CseKey {
            op_key: OpKey::from_op(&op),
            inputs: inputs.clone(),
            meta_sig: MetaSig::new(&meta),
            const_hash,
        };

        if let Some(&existing) = self.cse.get(&key) {
            if matches!(op, OpKind::Constant) {
                if let (Some(payload), Some(existing_payload)) =
                    (const_payload, self.const_payload(existing))
                    && existing_payload == payload
                {
                    return existing;
                }
            } else {
                return existing;
            }
        }

        let id = self.add_node_raw(op, inputs, meta);
        if matches!(key.op_key, OpKey::Constant)
            && let Some(payload) = const_payload
        {
            self.const_payloads.insert(id, payload.to_vec());
        }
        self.cse.insert(key, id);
        id
    }

    pub fn const_payload(&self, id: NodeId) -> Option<&[f32]> {
        self.const_payloads.get(&id).map(|v| v.as_slice())
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
    Exp,
    Log,
    Sum {
        axis: Option<i32>,
    },
    Mean {
        axis: Option<i32>,
    },
    Max {
        axis: Option<i32>,
    },
    MatMul,
    Reshape {
        new_shape: Vec<i64>,
    },
    Transpose {
        axes: Option<Vec<usize>>,
    },
    Softmax {
        axis: i32,
    },
    Silu,
    Gelu,
    LayerNorm {
        eps_bits: u32,
    },
    RmsNorm {
        eps_bits: u32,
    },
    Broadcast {
        target_shape: Vec<i64>,
    },
    LayerNormVjp {
        eps_bits: u32,
    },
    RmsNormVjp {
        eps_bits: u32,
    },
    ScaledMaskedSoftmax {
        scale_bits: u32,
        causal: bool,
    },
    Attention {
        scale_bits: u32,
        causal: bool,
    },
    Rope {
        rotary_dim: usize,
        pos_offset: usize,
        theta_bits: u32,
    },
    RoPE {
        base_bits: u32,
        offset: usize,
        traditional: bool,
    },
    SoftmaxVjp {
        axis: i32,
    },
    SiluVjp,
    GeluVjp,
    Sqrt,
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
            OpKind::Exp => OpKey::Exp,
            OpKind::Log => OpKey::Log,
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
            OpKind::LayerNormVjp { eps } => OpKey::LayerNormVjp {
                eps_bits: eps.to_bits(),
            },
            OpKind::RmsNormVjp { eps } => OpKey::RmsNormVjp {
                eps_bits: eps.to_bits(),
            },
            OpKind::ScaledMaskedSoftmax { scale, causal } => OpKey::ScaledMaskedSoftmax {
                scale_bits: scale.to_bits(),
                causal: *causal,
            },
            OpKind::Attention { scale, causal } => OpKey::Attention {
                scale_bits: scale.to_bits(),
                causal: *causal,
            },
            OpKind::Rope {
                rotary_dim,
                pos_offset,
                theta,
            } => OpKey::Rope {
                rotary_dim: *rotary_dim,
                pos_offset: *pos_offset,
                theta_bits: theta.to_bits(),
            },
            OpKind::RoPE {
                base,
                offset,
                traditional,
            } => OpKey::RoPE {
                base_bits: base.to_bits(),
                offset: *offset,
                traditional: *traditional,
            },
            OpKind::SoftmaxVjp { axis } => OpKey::SoftmaxVjp { axis: *axis },
            OpKind::SiluVjp => OpKey::SiluVjp,
            OpKind::GeluVjp => OpKey::GeluVjp,
            OpKind::Sqrt => OpKey::Sqrt,
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
    // Constants and Parameters must never be deduplicated: two tensors with
    // identical data may flow through different parts of the graph and receive
    // independent gradients during backpropagation.
    !matches!(op, OpKind::Constant | OpKind::Parameter)
}

pub fn hash_f32_payload(data: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.len().hash(&mut h);
    for &x in data {
        x.to_bits().hash(&mut h);
    }
    h.finish()
}

fn normalize_inputs_for_cse(op: &OpKind, inputs: &mut SmallVec<[NodeId; 2]>) {
    if matches!(op, OpKind::Add | OpKind::Mul) && inputs.len() == 2 && inputs[0].0 > inputs[1].0 {
        inputs.swap(0, 1);
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

    #[test]
    fn test_cse_does_not_dedup_constants() {
        let mut g = Graph::new();
        let meta = TensorMeta {
            shape: Shape::new(vec![2]),
            dtype: DType::F32,
        };
        let a = g.intern_node(
            OpKind::Constant,
            SmallVec::new(),
            meta.clone(),
            Some(&[1.0, 2.0]),
        );
        let b = g.intern_node(
            OpKind::Constant,
            SmallVec::new(),
            meta.clone(),
            Some(&[1.0, 2.0]),
        );
        // Constants must NOT be deduplicated — they may receive independent gradients
        assert_ne!(a, b);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_cse_dedups_ops() {
        let mut g = Graph::new();
        let meta = TensorMeta {
            shape: Shape::new(vec![2]),
            dtype: DType::F32,
        };
        let a = g.add_node_raw(OpKind::Constant, SmallVec::new(), meta.clone());
        let b = g.add_node_raw(OpKind::Constant, SmallVec::new(), meta.clone());

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
        assert_eq!(g.len(), 3); // 2 constants + 1 add
    }
}

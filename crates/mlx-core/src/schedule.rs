use crate::graph::{GraphView, NodeId};
use crate::{MlxError, Result};
use std::collections::{HashSet, VecDeque, HashMap};

#[derive(Debug, Clone)]
pub struct Schedule {
    /// Nodes in an order where dependencies appear before dependents.
    pub topo: Vec<NodeId>,
    /// The requested output nodes (subset of topo, typically at the end).
    pub outputs: Vec<NodeId>,
}

/// Compute a topo order for the subgraph required to produce `outputs`.
///
/// - Only includes nodes reachable from outputs.
/// - Ensures dependencies appear before dependents.
/// - Detects cycles (should never happen, but guards against bad graph construction).
pub fn topo_schedule(graph: &dyn GraphView, outputs: &[NodeId]) -> Result<Schedule> {
    // 1) Collect reachable nodes by reverse traversal (from outputs to inputs).
    let mut reachable: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = outputs.to_vec();

    while let Some(nid) = stack.pop() {
        if !reachable.insert(nid) {
            continue;
        }
        let n = graph.node(nid);
        for &inp in n.inputs.iter() {
            stack.push(inp);
        }
    }

    // 2) Compute in-degrees within reachable subgraph + adjacency (dep -> users).
    let mut indeg: HashMap<NodeId, usize> = HashMap::new();
    let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    for &nid in reachable.iter() {
        indeg.entry(nid).or_insert(0);
        let node = graph.node(nid);
        for &inp in node.inputs.iter() {
            if !reachable.contains(&inp) {
                continue;
            }
            *indeg.entry(nid).or_insert(0) += 1;
            users.entry(inp).or_default().push(nid);
        }
    }

    // 3) Kahn’s algorithm.
    let mut q = VecDeque::new();
    for (&nid, &d) in indeg.iter() {
        if d == 0 {
            q.push_back(nid);
        }
    }

    let mut topo = Vec::with_capacity(reachable.len());
    while let Some(nid) = q.pop_front() {
        topo.push(nid);
        if let Some(us) = users.get(&nid) {
            for &u in us.iter() {
                let e = indeg.get_mut(&u).unwrap();
                *e -= 1;
                if *e == 0 {
                    q.push_back(u);
                }
            }
        }
    }

    if topo.len() != reachable.len() {
        return Err(MlxError::Graph("cycle detected in graph"));
    }

    Ok(Schedule { topo, outputs: outputs.to_vec() })
}

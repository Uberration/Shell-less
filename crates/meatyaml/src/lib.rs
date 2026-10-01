//! MEATYAML: source code for executable composition.
//!
//! The compiler lowers source into MEAT IR — a [`Graph`] of typed nodes with
//! explicit data [`Edge`]s — plus the authority the program declares. Source
//! conveniences such as `from: previous` exist only here; the IR is the
//! contract every backend consumes.
//!
//! ```yaml
//! agent:
//!   name: echoer
//! authority:
//!   - path: /tools/echo
//!     rights: [invoke]
//!   - path: /state/result
//!     rights: [write]
//! flow:
//!   - id: make_meat
//!     invoke:
//!       path: /tools/echo
//!       input:
//!         text: MEAT
//!   - id: store
//!     write:
//!       path: /state/result
//!       from: make_meat
//! ```

mod parse;

use meatfs::{AuthorityRequest, NodeId, Path, Rights, Value};
use std::fmt;

pub use parse::compile;

/// A compiled MEATYAML program: IR plus declared authority.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub agent: String,
    /// Declared authority scopes: the ceiling the graph was checked against.
    pub authority: Vec<(Path, Rights)>,
    pub graph: Graph,
}

impl Program {
    /// The exact authority the graph needs, one entry per touched path.
    pub fn request(&self) -> AuthorityRequest {
        self.graph
            .nodes
            .iter()
            .fold(AuthorityRequest::new(&self.agent), |req, node| req.want(node.op.target().clone(), node.op.rights()))
    }
}

/// Content identity of a graph: a hash of its canonical encoding.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphId(u128);

impl GraphId {
    pub fn short(self) -> String {
        format!("gph:{:08x}", (self.0 >> 96) as u32)
    }
}

impl fmt::Display for GraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gph:{:032x}", self.0)
    }
}

impl fmt::Debug for GraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// MEAT IR.
///
/// * Data flows only along `edges`; each node has at most one incoming edge,
///   which feeds its data input.
/// * Effects are ordered by [`NodeId`]; the compiler assigns ids in source
///   order and edges always point from a lower id to a higher one.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub id: GraphId,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    fn new(nodes: Vec<Node>, edges: Vec<Edge>) -> Graph {
        let id = GraphId(fnv1a_128(canonical(&nodes, &edges).as_bytes()));
        Graph { id, nodes, edges }
    }

    /// The node whose output feeds `node`'s data input.
    pub fn input_of(&self, node: NodeId) -> Option<NodeId> {
        self.edges.iter().find(|e| e.to == node).map(|e| e.from)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    /// Source label (`id:` in MEATYAML). Not part of graph identity.
    pub label: Option<String>,
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Read(ReadOp),
    Write(WriteOp),
    Invoke(InvokeOp),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadOp {
    pub target: Path,
}

/// Writes `value`, or the incoming edge's data when `value` is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOp {
    pub target: Path,
    pub value: Option<Value>,
}

/// Invokes with `input`, or the incoming edge's data when `input` is `None`.
/// With neither, the input is `null`.
#[derive(Debug, Clone, PartialEq)]
pub struct InvokeOp {
    pub target: Path,
    pub input: Option<Value>,
}

impl Op {
    pub fn target(&self) -> &Path {
        match self {
            Op::Read(ReadOp { target }) | Op::Write(WriteOp { target, .. }) | Op::Invoke(InvokeOp { target, .. }) => {
                target
            }
        }
    }

    pub fn rights(&self) -> Rights {
        match self {
            Op::Read(_) => Rights::READ,
            Op::Write(_) => Rights::WRITE,
            Op::Invoke(_) => Rights::INVOKE,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Read(_) => "read",
            Op::Write(_) => "write",
            Op::Invoke(_) => "invoke",
        }
    }

    /// The literal data carried by the node itself, if any.
    pub fn literal(&self) -> Option<&Value> {
        match self {
            Op::Read(_) => None,
            Op::Write(WriteOp { value, .. }) => value.as_ref(),
            Op::Invoke(InvokeOp { input, .. }) => input.as_ref(),
        }
    }
}

/// The output of `from` feeds the data input of `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
}

/// Canonical text form of the semantic content of a graph. Labels are
/// excluded; maps are already ordered, so equal graphs encode equally.
fn canonical(nodes: &[Node], edges: &[Edge]) -> String {
    let mut out = String::from("meat-ir/1\n");
    for node in nodes {
        let literal = node.op.literal().map(Value::to_string).unwrap_or_default();
        out += &format!("node {} {} {} {}\n", node.id, node.op.name(), node.op.target(), literal);
    }
    for edge in edges {
        out += &format!("edge {} {}\n", edge.from, edge.to);
    }
    out
}

fn fnv1a_128(bytes: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    bytes.iter().fold(OFFSET, |hash, &b| (hash ^ u128::from(b)).wrapping_mul(PRIME))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    /// Location within the program, e.g. `flow[1].write.from`.
    pub at: String,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.at.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.at, self.message)
        }
    }
}

impl std::error::Error for Error {}

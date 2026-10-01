//! MEATYAML: source code for executable composition.
//!
//! The compiler lowers source into MEAT IR — a [`Graph`] of typed nodes,
//! explicit [`Edge`]s and named [`GraphOutput`]s — plus the authority the
//! program declares. Source conveniences (`from: previous`, labels, step
//! order) exist only here; the IR is the contract every backend consumes.
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
//! outputs:
//!   result:
//!     from: store
//! ```

mod parse;

use meatfs::{AuthorityRequest, NodeId, Path, Rights, Value};
use std::collections::BTreeMap;
use std::fmt;

pub use parse::{compile, compile_with};

/// A compiled MEATYAML program: IR plus declared authority.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub agent: String,
    /// Declared authority scopes: the ceiling the graph was checked against.
    pub authority: Vec<(Path, Rights)>,
    pub graph: Graph,
}

impl Program {
    /// The exact authority the graph needs: for each node in id order, its
    /// target (if it has one) and then its `uses`, each a separate want.
    /// `compose` nodes need no authority.
    pub fn request(&self) -> AuthorityRequest {
        self.graph.nodes.iter().fold(AuthorityRequest::new(&self.agent), |req, node| {
            let req = match node.op.target() {
                Some(target) => req.want(target.clone(), node.op.rights()),
                None => req,
            };
            node.op.uses().iter().fold(req, |req, u| req.want(u.path.clone(), u.rights))
        })
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
/// Execution order is defined only by `edges`:
/// * a [`EdgeKind::Data`] edge feeds the source's output into the target's
///   data input. Each node has at most one, except a `compose` node, which
///   has exactly one per distinct source its expression selects from;
/// * a [`EdgeKind::Order`] edge requires the source to complete first.
///
/// `NodeId`s carry no semantics beyond identity. A backend may run nodes in
/// any order consistent with the edges.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub id: GraphId,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub outputs: Vec<GraphOutput>,
}

impl Graph {
    pub fn new(nodes: Vec<Node>, edges: Vec<Edge>, outputs: Vec<GraphOutput>) -> Graph {
        let id = GraphId(fnv1a_128(canonical(&nodes, &edges, &outputs).as_bytes()));
        Graph { id, nodes, edges, outputs }
    }

    /// The node whose output feeds `node`'s data input.
    pub fn data_input(&self, node: NodeId) -> Option<NodeId> {
        self.edges.iter().find(|e| e.to == node && e.kind == EdgeKind::Data).map(|e| e.from)
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
    Compose(ValueExpr),
}

/// A pure value constructor over literals and earlier outputs of the same
/// execution. It reads no MeatFS state, invokes nothing and holds no
/// grants. There is no interpolation, evaluation or implicit conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueExpr {
    /// Exactly this value.
    Literal(Value),
    /// A part of `source`'s output; an empty path selects all of it.
    Select {
        source: NodeId,
        path: Vec<Selector>,
    },
    Map(BTreeMap<String, ValueExpr>),
    List(Vec<ValueExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    /// A field of a map.
    Key(String),
    /// An element of a list.
    Index(usize),
}

impl ValueExpr {
    /// Every node this expression selects from, ascending and deduplicated.
    pub fn sources(&self) -> Vec<NodeId> {
        fn walk(expr: &ValueExpr, out: &mut Vec<NodeId>) {
            match expr {
                ValueExpr::Literal(_) => {}
                ValueExpr::Select { source, .. } => out.push(*source),
                ValueExpr::Map(fields) => fields.values().for_each(|e| walk(e, out)),
                ValueExpr::List(items) => items.iter().for_each(|e| walk(e, out)),
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Nesting depth; a leaf has depth 1.
    pub fn depth(&self) -> usize {
        match self {
            ValueExpr::Literal(_) | ValueExpr::Select { .. } => 1,
            ValueExpr::Map(fields) => 1 + fields.values().map(ValueExpr::depth).max().unwrap_or(0),
            ValueExpr::List(items) => 1 + items.iter().map(ValueExpr::depth).max().unwrap_or(0),
        }
    }

    /// Number of expression constructors.
    pub fn count(&self) -> usize {
        match self {
            ValueExpr::Literal(_) | ValueExpr::Select { .. } => 1,
            ValueExpr::Map(fields) => 1 + fields.values().map(ValueExpr::count).sum::<usize>(),
            ValueExpr::List(items) => 1 + items.iter().map(ValueExpr::count).sum::<usize>(),
        }
    }

    /// Unambiguous canonical encoding: constructor tags, literal values,
    /// key/index distinctions and resolved node references.
    fn canonical(&self) -> String {
        match self {
            ValueExpr::Literal(v) => format!("lit({v})"),
            ValueExpr::Select { source, path } => {
                let path: Vec<String> = path
                    .iter()
                    .map(|s| match s {
                        Selector::Key(k) => format!("k{}", Value::from(k.as_str())),
                        Selector::Index(i) => format!("i{i}"),
                    })
                    .collect();
                format!("sel({source},[{}])", path.join(","))
            }
            ValueExpr::Map(fields) => {
                let fields: Vec<String> =
                    fields.iter().map(|(k, e)| format!("{}:{}", Value::from(k.as_str()), e.canonical())).collect();
                format!("map{{{}}}", fields.join(","))
            }
            ValueExpr::List(items) => {
                format!("list[{}]", items.iter().map(ValueExpr::canonical).collect::<Vec<_>>().join(","))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadOp {
    pub target: Path,
}

/// Create-or-replace: writes `value`, or the data input when `value` is
/// `None`. Creation happens when the node executes, inside the execution's
/// transaction.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOp {
    pub target: Path,
    pub value: Option<Value>,
}

/// Invokes with `input`, or the data input when `input` is `None`; with
/// neither, the input is `null`. `uses` is the only MeatFS authority an
/// effectful capability receives.
#[derive(Debug, Clone, PartialEq)]
pub struct InvokeOp {
    pub target: Path,
    pub input: Option<Value>,
    pub uses: Vec<Use>,
}

/// Authority attached to one invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct Use {
    pub path: Path,
    pub rights: Rights,
}

impl Op {
    /// The MeatFS name the node acts on; `None` for `compose`.
    pub fn target(&self) -> Option<&Path> {
        match self {
            Op::Read(ReadOp { target }) | Op::Write(WriteOp { target, .. }) | Op::Invoke(InvokeOp { target, .. }) => {
                Some(target)
            }
            Op::Compose(_) => None,
        }
    }

    pub fn rights(&self) -> Rights {
        match self {
            Op::Read(_) => Rights::READ,
            Op::Write(_) => Rights::WRITE,
            Op::Invoke(_) => Rights::INVOKE,
            Op::Compose(_) => Rights::NONE,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Read(_) => "read",
            Op::Write(_) => "write",
            Op::Invoke(_) => "invoke",
            Op::Compose(_) => "compose",
        }
    }

    /// The literal data carried by a read, write or invoke node, if any.
    pub fn literal(&self) -> Option<&Value> {
        match self {
            Op::Read(_) | Op::Compose(_) => None,
            Op::Write(WriteOp { value, .. }) => value.as_ref(),
            Op::Invoke(InvokeOp { input, .. }) => input.as_ref(),
        }
    }

    pub fn uses(&self) -> &[Use] {
        match self {
            Op::Invoke(InvokeOp { uses, .. }) => uses,
            _ => &[],
        }
    }

    /// Every MeatFS path the node may touch, with the rights it touches it
    /// with. Says nothing about effects outside MeatFS.
    pub fn footprint(&self) -> Vec<(&Path, Rights)> {
        let mut footprint: Vec<_> = self.target().map(|t| (t, self.rights())).into_iter().collect();
        footprint.extend(self.uses().iter().map(|u| (&u.path, u.rights)));
        footprint
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    /// `to` consumes the output of `from`.
    Data,
    /// `from` must complete before `to` starts.
    Order,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
}

/// A named value the graph produces: the output of `source`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphOutput {
    pub name: String,
    pub source: NodeId,
}

/// Canonical text form of the semantic content of a graph. Labels are
/// excluded; maps are already ordered, so equal graphs encode equally.
fn canonical(nodes: &[Node], edges: &[Edge], outputs: &[GraphOutput]) -> String {
    let mut out = String::from("meat-ir/3\n");
    for node in nodes {
        let detail = match &node.op {
            Op::Compose(expr) => expr.canonical(),
            op => {
                let literal = op.literal().map(Value::to_string).unwrap_or_default();
                format!("{} {literal}", op.target().expect("non-compose ops have targets"))
            }
        };
        out += &format!("node {} {} {detail}\n", node.id, node.op.name());
        for u in node.op.uses() {
            out += &format!("use {} {} {}\n", node.id, u.path, u.rights);
        }
    }
    for edge in edges {
        out += &format!("edge {:?} {} {}\n", edge.kind, edge.from, edge.to);
    }
    for output in outputs {
        out += &format!("output {} {}\n", output.name, output.source);
    }
    out
}

fn fnv1a_128(bytes: &[u8]) -> u128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013B;
    bytes.iter().fold(OFFSET, |hash, &b| (hash ^ u128::from(b)).wrapping_mul(PRIME))
}

/// What kind of compile error occurred. Stable and content-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The text is not well-formed YAML.
    Syntax,
    /// A construct has the wrong shape: not a map, not a list, too many keys.
    Shape,
    MissingKey,
    UnknownKey,
    UnknownOperation,
    UnknownConstructor,
    UnknownRight,
    InvalidPath,
    InvalidValue,
    /// A step needs authority the program does not declare.
    Undeclared,
    DuplicateId,
    UnknownReference,
    /// Both literal data and `from` were given.
    ConflictingData,
    /// A write has nothing to write.
    MissingData,
}

/// A compile error record, made under the host's capture policy. Under
/// `Omit` it holds no source text: `location` names source-controlled
/// segments only by position (`#i`) and `detail` is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    /// Structural location, e.g. `flow[2].compose.map.#0.select`. Under
    /// `Inline`, source-controlled segments appear as written.
    pub location: String,
    /// Line and column (both 1-based), where the YAML parser reports them.
    pub position: Option<(usize, usize)>,
    /// Human-readable explanation, retained only under `Inline`.
    pub detail: Option<String>,
}

impl Error {
    /// Render the record. Shows exactly what the record retained.
    pub fn render(&self) -> String {
        let mut out = format!("compile error kind={:?}", self.kind);
        if !self.location.is_empty() {
            out += &format!(" at={}", self.location);
        }
        if let Some((line, column)) = self.position {
            out += &format!(" line={line} column={column}");
        }
        match &self.detail {
            Some(detail) => out += &format!(" detail={detail}"),
            None => out += " detail=<omitted>",
        }
        out
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for Error {}

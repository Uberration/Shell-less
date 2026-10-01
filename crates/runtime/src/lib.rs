//! The first MEAT IR backend: a deterministic Rust interpreter.
//!
//! ```text
//! Program ──load──▶ Loaded (resolved execution graph) ──execute──▶ ExecutionReceipt
//!            │
//!            ├─ validate IR (edges, inputs, acyclicity) and fix an order
//!            ├─ AuthorityRequest → host Policy → GrantSet  (MeatFs::issue)
//!            └─ every node bound to an ObjectId and the GrantId it may use
//! ```
//!
//! The runtime holds no authority of its own and never looks up a name while
//! executing: nodes carry resolved object identities and grants. Given the
//! same graph, initial namespace state and pure capability inputs, execution
//! produces the same outputs, events and receipt.

use meatfs::{
    Cause, Event, ExecutionId, GrantId, GrantSet, MeatFs, NodeId, ObjectId, Policy, PrincipalId, Rights, Value,
};
use meatyaml::{Graph, GraphId, Op, Program};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The IR violates a structural rule.
    InvalidGraph(String),
    /// Authority resolution failed: policy denial, unbound name, …
    Authority(meatfs::Error),
    /// A node failed while executing.
    Node { node: NodeId, source: meatfs::Error },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidGraph(m) => write!(f, "invalid graph: {m}"),
            Error::Authority(e) => write!(f, "authority: {e}"),
            Error::Node { node, source } => write!(f, "{node}: {source}"),
        }
    }
}

impl std::error::Error for Error {}

/// A node bound to the object it touches and the grant it presents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    pub node: NodeId,
    pub object: ObjectId,
    pub grant: GrantId,
}

/// A resolved execution graph: IR plus the authority issued for it.
pub struct Loaded {
    graph: Graph,
    grants: GrantSet,
    /// Indexed like `graph.nodes`.
    bindings: Vec<Binding>,
    /// Indices into `graph.nodes`, in execution order.
    order: Vec<usize>,
}

impl Loaded {
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn grants(&self) -> &GrantSet {
        &self.grants
    }

    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Revoke the graph's authority. It cannot run again.
    pub fn retire(self, fs: &MeatFs) {
        fs.retire(self.grants);
    }
}

/// Validate the IR and return a deterministic execution order.
///
/// Rules: node ids are dense and match their index; every edge joins two
/// existing nodes; a node has at most one incoming edge; reads and nodes
/// with literal data have none; writes have data; the graph is acyclic.
/// Ready nodes run lowest-id first, which preserves source order of effects.
fn schedule(graph: &Graph) -> Result<Vec<usize>, Error> {
    let invalid = |m: String| Err(Error::InvalidGraph(m));
    let n = graph.nodes.len();
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id != NodeId(i as u32) {
            return invalid(format!("node at index {i} has id {}", node.id));
        }
    }
    let mut incoming = vec![None; n];
    let mut outgoing = vec![Vec::new(); n];
    for edge in &graph.edges {
        let (from, to) = (edge.from.0 as usize, edge.to.0 as usize);
        if from >= n || to >= n {
            return invalid(format!("edge {} → {} leaves the graph", edge.from, edge.to));
        }
        if incoming[to].replace(from).is_some() {
            return invalid(format!("{} has more than one input", edge.to));
        }
        outgoing[from].push(to);
    }
    for (node, input) in graph.nodes.iter().zip(&incoming) {
        match (&node.op, input) {
            (Op::Read(_), Some(_)) => return invalid(format!("{} reads but has an input edge", node.id)),
            (op, Some(_)) if op.literal().is_some() => {
                return invalid(format!("{} has both literal data and an input edge", node.id))
            }
            (Op::Write(w), None) if w.value.is_none() => return invalid(format!("{} writes nothing", node.id)),
            _ => {}
        }
    }

    let mut ready: BTreeSet<usize> = (0..n).filter(|&i| incoming[i].is_none()).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(i) = ready.pop_first() {
        order.push(i);
        ready.extend(outgoing[i].iter().copied());
    }
    if order.len() != n {
        return invalid("graph has a cycle".to_owned());
    }
    Ok(order)
}

/// Resolve a program into an executable graph under host `policy`.
pub fn load(fs: &MeatFs, policy: &Policy, program: &Program) -> Result<Loaded, Error> {
    let order = schedule(&program.graph)?;
    let grants = fs.issue(policy, &program.request()).map_err(Error::Authority)?;
    let bindings = program
        .graph
        .nodes
        .iter()
        .map(|node| {
            let object = fs.resolve(node.op.target()).expect("issue binds every requested name");
            let grant = grants.for_object(object).expect("issue grants every requested object").id();
            Binding { node: node.id, object, grant }
        })
        .collect();
    Ok(Loaded { graph: program.graph.clone(), grants, bindings, order })
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrantRecord {
    pub grant: GrantId,
    pub object: ObjectId,
    pub rights: Rights,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeRecord {
    pub node: NodeId,
    pub op: &'static str,
    pub object: ObjectId,
    pub grant: GrantId,
    pub output: Value,
}

/// The substrate's record of one completed execution.
///
/// Connects execution → node → grant used → object touched → event
/// produced. Deterministic plain data; attestation comes later.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    pub execution: ExecutionId,
    pub graph: GraphId,
    pub principal: PrincipalId,
    pub grants: Vec<GrantRecord>,
    /// In execution order.
    pub nodes: Vec<NodeRecord>,
    pub events: Vec<Event>,
    /// Output of the last node executed.
    pub result: Value,
}

/// Run a loaded graph to completion.
pub fn execute(fs: &MeatFs, loaded: &Loaded) -> Result<ExecutionReceipt, Error> {
    let execution = fs.new_execution();
    let mut outputs: BTreeMap<NodeId, Value> = BTreeMap::new();
    let mut nodes = Vec::with_capacity(loaded.order.len());

    for &i in &loaded.order {
        let node = &loaded.graph.nodes[i];
        let Binding { object, grant, .. } = loaded.bindings[i];
        let access = loaded.grants.access(grant).expect("bound grant is in the set");
        let access = access.caused_by(Cause { execution, node: node.id });
        let data = || match (node.op.literal(), loaded.graph.input_of(node.id)) {
            (Some(literal), _) => literal.clone(),
            (None, Some(from)) => outputs[&from].clone(),
            (None, None) => Value::Null,
        };
        let output = match &node.op {
            Op::Read(_) => fs.read(&access, object),
            Op::Invoke(_) => fs.invoke(&access, object, data()),
            Op::Write(_) => {
                let value = data();
                fs.write(&access, object, value.clone()).map(|_| value)
            }
        }
        .map_err(|source| Error::Node { node: node.id, source })?;
        nodes.push(NodeRecord { node: node.id, op: node.op.name(), object, grant, output: output.clone() });
        outputs.insert(node.id, output);
    }

    let witness = loaded.grants.access(loaded.bindings[0].grant).expect("bound grant is in the set");
    let first = loaded.graph.nodes[loaded.order[0]].id;
    let events = fs.execution_events(&witness.caused_by(Cause { execution, node: first })).map_err(Error::Authority)?;
    let grants = loaded
        .grants
        .iter()
        .map(|g| GrantRecord {
            grant: g.id(),
            object: match g.target() {
                meatfs::Target::Object(o) => *o,
                meatfs::Target::Namespace(_) => unreachable!("graphs are only issued object grants"),
            },
            rights: g.rights(),
        })
        .collect();
    let result = nodes.last().map(|n: &NodeRecord| n.output.clone()).unwrap_or_default();
    Ok(ExecutionReceipt {
        execution,
        graph: loaded.graph.id,
        principal: loaded.grants.principal(),
        grants,
        nodes,
        events,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::builtin::{Echo, Upper};
    use meatfs::{EventKind, Path, Seed};
    use meatyaml::{Edge, Node, ReadOp};

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    const ECHO: &str = include_str!("../../../examples/echo.meat.yaml");

    fn boot(seed: u64) -> MeatFs {
        let (fs, host) = MeatFs::genesis(Seed::fixed(seed));
        let root = host.access(host.iter().next().unwrap().id()).unwrap();
        capability::mount(&fs, &root, &p("/tools/echo"), Echo).unwrap();
        capability::mount(&fs, &root, &p("/tools/text/upper"), Upper).unwrap();
        fs.bind(&root, &p("/memory/context"), Value::map([("text", Value::from("boot"))])).unwrap();
        fs
    }

    fn policy() -> Policy {
        Policy::default()
            .allow(p("/tools"), Rights::INVOKE)
            .allow(p("/memory"), Rights::READ)
            .allow(p("/state"), Rights::READ | Rights::WRITE)
    }

    fn run(fs: &MeatFs, src: &str) -> Result<ExecutionReceipt, Error> {
        let program = meatyaml::compile(src).unwrap();
        execute(fs, &load(fs, &policy(), &program)?)
    }

    #[test]
    fn acceptance_receipt_connects_everything() {
        let fs = boot(42);
        let program = meatyaml::compile(ECHO).unwrap();
        let loaded = load(&fs, &policy(), &program).unwrap();
        let receipt = execute(&fs, &loaded).unwrap();

        let meat = Value::map([("text", Value::from("MEAT"))]);
        assert_eq!(receipt.result, meat);
        let echo = fs.resolve(&p("/tools/echo")).unwrap();
        let result = fs.resolve(&p("/state/result")).unwrap();

        assert_eq!(receipt.graph, program.graph.id);
        assert_eq!(receipt.principal, loaded.grants().principal());
        assert_eq!(
            receipt.grants.iter().map(|g| (g.object, g.rights)).collect::<Vec<_>>(),
            vec![(echo, Rights::INVOKE), (result, Rights::WRITE)]
        );
        assert_eq!(
            receipt.nodes.iter().map(|n| (n.node, n.object, n.grant)).collect::<Vec<_>>(),
            vec![(NodeId(0), echo, receipt.grants[0].grant), (NodeId(1), result, receipt.grants[1].grant)]
        );
        let events: Vec<_> =
            receipt.events.iter().map(|e| (e.cause.unwrap().node, e.object, e.grant, &e.kind)).collect();
        assert_eq!(
            events,
            vec![
                (NodeId(0), echo, Some(receipt.grants[0].grant), &EventKind::Invoked { ok: true }),
                (NodeId(1), result, Some(receipt.grants[1].grant), &EventKind::Written { version: 1 }),
            ]
        );
        assert!(receipt.events.iter().all(|e| e.cause.unwrap().execution == receipt.execution));
    }

    #[test]
    fn execution_is_deterministic() {
        let a = run(&boot(7), ECHO).unwrap();
        let b = run(&boot(7), ECHO).unwrap();
        assert_eq!(a, b);
        let c = run(&boot(8), ECHO).unwrap();
        assert_eq!(a.result, c.result);
        assert_ne!(a.execution, c.execution, "identities differ across seeds");
    }

    #[test]
    fn escalation_fails_at_policy() {
        let fs = boot(1);
        let e = run(
            &fs,
            r#"
agent: { name: rogue }
authority:
  - { path: /memory/context, rights: [read, write] }
flow:
  - read: /memory/context
  - write: /memory/context
"#,
        )
        .unwrap_err();
        assert_eq!(
            e,
            Error::Authority(meatfs::Error::PolicyDenied {
                path: p("/memory/context"),
                rights: Rights::READ | Rights::WRITE
            })
        );
    }

    #[test]
    fn retired_graphs_cannot_run() {
        let fs = boot(1);
        let loaded = load(&fs, &policy(), &meatyaml::compile(ECHO).unwrap()).unwrap();
        execute(&fs, &loaded).unwrap();
        let grant = loaded.bindings()[0].grant;
        let graph = loaded.graph().clone();
        loaded.retire(&fs);
        // Re-loading issues fresh grants; the retired ones are gone for good.
        let again =
            load(&fs, &policy(), &meatyaml::Program { agent: "echoer".into(), authority: vec![], graph }).unwrap();
        assert_ne!(again.bindings()[0].grant, grant);
    }

    #[test]
    fn rejects_structurally_invalid_ir() {
        let program = meatyaml::compile(ECHO).unwrap();
        let mut cyclic = program.clone();
        cyclic.graph.edges.push(Edge { from: NodeId(1), to: NodeId(0) });
        assert!(matches!(schedule(&cyclic.graph), Err(Error::InvalidGraph(_))));

        let mut fed_read = program.clone();
        fed_read.graph.nodes.push(Node { id: NodeId(2), label: None, op: Op::Read(ReadOp { target: p("/x") }) });
        fed_read.graph.edges.push(Edge { from: NodeId(0), to: NodeId(2) });
        assert!(matches!(schedule(&fed_read.graph), Err(Error::InvalidGraph(_))));
    }

    #[test]
    fn failing_node_is_reported_by_id() {
        let fs = boot(1);
        let e = run(
            &fs,
            r#"
agent: { name: a }
authority: [{ path: /tools, rights: [invoke] }]
flow:
  - invoke: { path: /tools/echo, input: { wrong: 1 } }
"#,
        )
        .unwrap_err();
        assert!(matches!(e, Error::Node { node: NodeId(0), source: meatfs::Error::Capability { .. } }));
    }
}

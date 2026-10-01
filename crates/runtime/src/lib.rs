//! The first MEAT IR backend: a deterministic, transactional Rust interpreter.
//!
//! ```text
//! Program ──load──▶ Loaded ──execute──▶ ExecutionReceipt
//!            │                 │
//!            │                 ├─ one MeatFS transaction per execution (graph-atomic)
//!            │                 ├─ nodes run when their dependencies succeed; ties → lowest NodeId
//!            │                 ├─ failure blocks dependents, then rolls everything back
//!            │                 └─ every node acts only with the grants attached to it
//!            │
//!            ├─ validate IR structure
//!            ├─ AuthorityRequest → host Policy → GrantSet   (observational: no state changes)
//!            └─ project grants per node: target grant + `uses` grants
//! ```

use meatfs::{
    CapabilityMeta, Cause, Event, ExecutionId, GrantId, GrantSet, MeatFs, NodeGrant, NodeGrants, NodeId, ObjectId,
    Path, Policy, PrincipalId, Purity, Rights, Target, Transaction, Value,
};
use meatyaml::{EdgeKind, Graph, GraphId, Op, Program};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Why a program could not be loaded. Nothing has executed and the
/// namespace is unchanged.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadError {
    /// The IR violates a structural rule.
    InvalidGraph(String),
    /// Authority resolution failed: policy denial, unbound name, …
    Authority(meatfs::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::InvalidGraph(m) => write!(f, "invalid graph: {m}"),
            LoadError::Authority(e) => write!(f, "authority: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// What a node's operation acts on, fixed at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedTarget {
    Object(ObjectId),
    /// An unbound name a write node will create when it executes.
    New(Path),
}

/// A node with its statically resolved target and its exact authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedNode {
    pub id: NodeId,
    pub target: ResolvedTarget,
    /// The grant for the node's own operation.
    pub grant: GrantId,
    /// Grants attached to an invocation: all its capability may use.
    pub uses: Vec<NodeGrant>,
}

/// A resolved execution graph: IR, issued authority, per-node projection.
pub struct Loaded {
    graph: Graph,
    grants: GrantSet,
    nodes: Vec<ResolvedNode>,
    preds: Vec<Vec<usize>>,
    succs: Vec<Vec<usize>>,
}

impl Loaded {
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn grants(&self) -> &GrantSet {
        &self.grants
    }

    pub fn nodes(&self) -> &[ResolvedNode] {
        &self.nodes
    }

    /// Revoke the graph's authority. It cannot run again.
    pub fn retire(self, fs: &MeatFs) {
        fs.retire(self.grants);
    }
}

/// Validate IR structure, returning predecessor and successor lists.
///
/// Rules: node ids are dense and match their index; every edge joins two
/// existing nodes; a node has at most one data input; reads and nodes with
/// literal data have none; writes have data; outputs name existing nodes;
/// the graph (data and order edges together) is acyclic.
#[allow(clippy::type_complexity)]
fn validate(graph: &Graph) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>), LoadError> {
    let invalid = |m: String| Err(LoadError::InvalidGraph(m));
    let n = graph.nodes.len();
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id != NodeId(i as u32) {
            return invalid(format!("node at index {i} has id {}", node.id));
        }
    }
    let mut preds = vec![Vec::new(); n];
    let mut succs = vec![Vec::new(); n];
    let mut data_in = vec![false; n];
    for edge in &graph.edges {
        let (from, to) = (edge.from.0 as usize, edge.to.0 as usize);
        if from >= n || to >= n {
            return invalid(format!("edge {} → {} leaves the graph", edge.from, edge.to));
        }
        if edge.kind == EdgeKind::Data && std::mem::replace(&mut data_in[to], true) {
            return invalid(format!("{} has more than one data input", edge.to));
        }
        preds[to].push(from);
        succs[from].push(to);
    }
    for (node, &has_input) in graph.nodes.iter().zip(&data_in) {
        match (&node.op, has_input) {
            (Op::Read(_), true) => return invalid(format!("{} reads but has a data input", node.id)),
            (op, true) if op.literal().is_some() => {
                return invalid(format!("{} has both literal data and a data input", node.id))
            }
            (Op::Write(w), false) if w.value.is_none() => return invalid(format!("{} writes nothing", node.id)),
            _ => {}
        }
    }
    if let Some(o) = graph.outputs.iter().find(|o| o.source.0 as usize >= n) {
        return invalid(format!("output `{}` names missing node {}", o.name, o.source));
    }
    let mut remaining: Vec<usize> = preds.iter().map(Vec::len).collect();
    let mut ready: Vec<usize> = (0..n).filter(|&i| remaining[i] == 0).collect();
    let mut seen = 0;
    while let Some(i) = ready.pop() {
        seen += 1;
        for &s in &succs[i] {
            remaining[s] -= 1;
            if remaining[s] == 0 {
                ready.push(s);
            }
        }
    }
    if seen != n {
        return invalid("graph has a cycle".to_owned());
    }
    Ok((preds, succs))
}

/// Resolve a program into an executable graph under host `policy`.
///
/// Observational: on success or failure, the namespace is unchanged.
pub fn load(fs: &MeatFs, policy: &Policy, program: &Program) -> Result<Loaded, LoadError> {
    let (preds, succs) = validate(&program.graph)?;
    let grants = fs.issue(policy, &program.request()).map_err(LoadError::Authority)?;

    // `request()` lists, per node, its target then its uses; walk it in step.
    let mut issued = grants.iter();
    let mut nodes = Vec::with_capacity(program.graph.nodes.len());
    for node in &program.graph.nodes {
        let grant = issued.next().expect("one grant per want");
        let target = match grant.target() {
            Target::Object(o) => ResolvedTarget::Object(*o),
            Target::Name(p) => ResolvedTarget::New(p.clone()),
            Target::Namespace(_) => unreachable!("issue never grants namespaces"),
        };
        let uses: Vec<NodeGrant> = node
            .op
            .uses()
            .iter()
            .map(|u| NodeGrant { path: u.path.clone(), grant: issued.next().expect("one grant per want").id() })
            .collect();
        if let (Op::Invoke(_), ResolvedTarget::Object(o), false) = (&node.op, &target, uses.is_empty()) {
            if fs.meta(*o).is_some_and(|m| m.purity == Purity::Pure) {
                return Err(LoadError::InvalidGraph(format!("{}: pure capability cannot be given `uses`", node.id)));
            }
        }
        nodes.push(ResolvedNode { id: node.id, target, grant: grant.id(), uses });
    }
    drop(issued);
    Ok(Loaded { graph: program.graph.clone(), grants, nodes, preds, succs })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Pending,
    Ready,
    Running,
    Succeeded,
    Failed,
    /// A dependency failed, so this node never ran.
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionErrorKind {
    AuthorityDenied,
    MissingObject,
    WrongObjectKind,
    InvalidInput,
    CapabilityFailed,
    DependencyFailed,
    TransactionConflict,
}

/// A structured failure: what failed, where, with which authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionError {
    pub execution: ExecutionId,
    pub node: Option<NodeId>,
    pub object: Option<ObjectId>,
    pub grant: Option<GrantId>,
    pub kind: ExecutionErrorKind,
    /// Human-readable context. Never needed to interpret the failure.
    pub detail: String,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.map(|n| format!("{n}: ")).unwrap_or_default();
        write!(f, "{node}{:?}: {}", self.kind, self.detail)
    }
}

impl std::error::Error for ExecutionError {}

fn classify(e: &meatfs::Error) -> (ExecutionErrorKind, Option<ObjectId>) {
    use meatfs::Error as E;
    use ExecutionErrorKind as K;
    match e {
        E::Denied { object, .. } => (K::AuthorityDenied, Some(*object)),
        E::InvalidGrant(_) | E::PolicyDenied { .. } | E::NotAttached(_) => (K::AuthorityDenied, None),
        E::Unbound(_) => (K::MissingObject, None),
        E::UnknownObject(o) => (K::MissingObject, Some(*o)),
        E::Unsupported { object, .. } => (K::WrongObjectKind, Some(*object)),
        E::AlreadyBound(_) => (K::WrongObjectKind, None),
        E::InvalidPath(_) => (K::InvalidInput, None),
        E::InvalidInput { object, .. } => (K::InvalidInput, Some(*object)),
        E::Capability { object, .. } | E::Impure(object) => (K::CapabilityFailed, Some(*object)),
        E::Conflict(o) => (K::TransactionConflict, Some(*o)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxOutcome {
    Committed,
    RolledBack,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrantRecord {
    pub grant: GrantId,
    pub target: Target,
    pub rights: Rights,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeRecord {
    pub node: NodeId,
    pub op: &'static str,
    pub state: NodeState,
    /// The object touched; for a creating write, the object it created.
    pub object: Option<ObjectId>,
    pub grant: GrantId,
    pub uses: Vec<GrantId>,
    /// For invocations: the capability's declared execution properties and
    /// implementation provenance, as published by the invoked object.
    pub capability: Option<CapabilityMeta>,
    /// The data the node consumed, if it ran.
    pub input: Option<Value>,
    pub output: Option<Value>,
    pub error: Option<ExecutionError>,
}

/// The substrate's record of one execution, successful or not.
///
/// Connects execution → node → grant used → object touched → event
/// produced. Deterministic plain data; attestation comes later.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    pub execution: ExecutionId,
    pub graph: GraphId,
    pub principal: PrincipalId,
    pub grants: Vec<GrantRecord>,
    /// Every node, by `NodeId`.
    pub nodes: Vec<NodeRecord>,
    /// The order nodes actually ran in.
    pub schedule: Vec<NodeId>,
    /// Audit events caused by this execution, including rolled-back work.
    pub events: Vec<Event>,
    pub transaction: TxOutcome,
    /// Graph outputs; empty unless the transaction committed.
    pub outputs: BTreeMap<String, Value>,
    /// The failure that rolled the execution back, if any.
    pub error: Option<ExecutionError>,
}

impl ExecutionReceipt {
    pub fn succeeded(&self) -> bool {
        self.transaction == TxOutcome::Committed
    }
}

struct Run<'a> {
    loaded: &'a Loaded,
    tx: Transaction<'a>,
    execution: ExecutionId,
    records: Vec<NodeRecord>,
}

impl Run<'_> {
    fn input(&self, i: usize) -> Value {
        let node = &self.loaded.graph.nodes[i];
        match (node.op.literal(), self.loaded.graph.data_input(node.id)) {
            (Some(literal), _) => literal.clone(),
            (None, Some(from)) => self.records[from.0 as usize].output.clone().expect("dependencies succeeded"),
            (None, None) => Value::Null,
        }
    }

    fn step(&mut self, i: usize) -> Result<(Option<ObjectId>, Value), meatfs::Error> {
        let (node, resolved, grants) = (&self.loaded.graph.nodes[i], &self.loaded.nodes[i], &self.loaded.grants);
        let access = grants.access(resolved.grant).expect("resolved grant is in the set");
        let access = access.caused_by(Cause { execution: self.execution, node: node.id });
        let input = self.input(i);
        if !matches!(node.op, Op::Read(_)) {
            self.records[i].input = Some(input.clone());
        }
        let result = match (&node.op, &resolved.target) {
            (Op::Read(_), ResolvedTarget::Object(o)) => self.tx.read(&access, *o).map(|v| (Some(*o), v)),
            (Op::Invoke(_), ResolvedTarget::Object(o)) => {
                let attached = NodeGrants::new(grants, &resolved.uses).expect("uses are in the set");
                self.tx.invoke(&access, *o, input, attached).map(|v| (Some(*o), v))
            }
            (Op::Write(_), ResolvedTarget::Object(o)) => {
                self.tx.write(&access, *o, input.clone()).map(|_| (Some(*o), input))
            }
            (Op::Write(_), ResolvedTarget::New(p)) => {
                self.tx.create(&access, p, input.clone()).map(|o| (Some(o), input))
            }
            (_, ResolvedTarget::New(p)) => Err(meatfs::Error::Unbound(p.clone())),
        };
        result
    }

    /// Structure a substrate failure of node `i`.
    fn failure(&self, i: usize, e: &meatfs::Error) -> ExecutionError {
        let resolved = &self.loaded.nodes[i];
        let (kind, object) = classify(e);
        let object = object.or(match &resolved.target {
            ResolvedTarget::Object(o) => Some(*o),
            ResolvedTarget::New(_) => None,
        });
        ExecutionError {
            execution: self.execution,
            node: Some(resolved.id),
            object,
            grant: Some(resolved.grant),
            kind,
            detail: e.to_string(),
        }
    }

    /// Mark every transitive dependent of `failed` as blocked.
    fn block_dependents(&mut self, failed: usize, states: &mut [NodeState]) {
        let mut stack = self.loaded.succs[failed].clone();
        while let Some(i) = stack.pop() {
            if states[i] != NodeState::Blocked {
                states[i] = NodeState::Blocked;
                self.records[i].error = Some(ExecutionError {
                    execution: self.execution,
                    node: Some(self.loaded.nodes[i].id),
                    object: None,
                    grant: Some(self.loaded.nodes[i].grant),
                    kind: ExecutionErrorKind::DependencyFailed,
                    detail: format!("depends on failed {}", self.loaded.nodes[failed].id),
                });
                stack.extend(&self.loaded.succs[i]);
            }
        }
    }
}

/// Run a loaded graph inside one transaction: commit if every node
/// succeeds, otherwise roll back. Always returns a receipt.
pub fn execute(fs: &MeatFs, loaded: &Loaded) -> ExecutionReceipt {
    let execution = fs.new_execution();
    let n = loaded.graph.nodes.len();
    let records = loaded
        .graph
        .nodes
        .iter()
        .zip(&loaded.nodes)
        .map(|(node, r)| NodeRecord {
            node: node.id,
            op: node.op.name(),
            state: NodeState::Pending,
            object: match r.target {
                ResolvedTarget::Object(o) => Some(o),
                ResolvedTarget::New(_) => None,
            },
            grant: r.grant,
            uses: r.uses.iter().map(|u| u.grant).collect(),
            capability: match (&node.op, &r.target) {
                (Op::Invoke(_), ResolvedTarget::Object(o)) => fs.meta(*o),
                _ => None,
            },
            input: None,
            output: None,
            error: None,
        })
        .collect();
    let mut run = Run { loaded, tx: fs.transaction(), execution, records };

    let mut states = vec![NodeState::Pending; n];
    let mut ready = BTreeSet::new();
    for (i, state) in states.iter_mut().enumerate() {
        if loaded.preds[i].is_empty() {
            *state = NodeState::Ready;
            ready.insert(i);
        }
    }
    let mut schedule = Vec::with_capacity(n);
    let mut error = None;
    // Lowest NodeId among ready nodes: a reproducible tie-break, not semantics.
    while let Some(i) = ready.pop_first() {
        states[i] = NodeState::Running;
        schedule.push(loaded.nodes[i].id);
        match run.step(i) {
            Ok((object, output)) => {
                states[i] = NodeState::Succeeded;
                run.records[i].object = object;
                run.records[i].output = Some(output);
                for &s in &loaded.succs[i] {
                    if states[s] == NodeState::Pending
                        && loaded.preds[s].iter().all(|&p| states[p] == NodeState::Succeeded)
                    {
                        states[s] = NodeState::Ready;
                        ready.insert(s);
                    }
                }
            }
            Err(e) => {
                let e = run.failure(i, &e);
                states[i] = NodeState::Failed;
                run.records[i].error = Some(e.clone());
                run.block_dependents(i, &mut states);
                error = Some(e);
                break;
            }
        }
    }
    for (record, state) in run.records.iter_mut().zip(&states) {
        record.state = *state;
    }

    let Run { tx, records, .. } = run;
    let transaction = match error {
        Some(_) => {
            tx.rollback();
            TxOutcome::RolledBack
        }
        None => match tx.commit() {
            Ok(()) => TxOutcome::Committed,
            Err(e) => {
                let (kind, object) = classify(&e);
                error =
                    Some(ExecutionError { execution, node: None, object, grant: None, kind, detail: e.to_string() });
                TxOutcome::RolledBack
            }
        },
    };

    let outputs = match transaction {
        TxOutcome::Committed => loaded
            .graph
            .outputs
            .iter()
            .map(|o| (o.name.clone(), records[o.source.0 as usize].output.clone().expect("committed nodes all ran")))
            .collect(),
        TxOutcome::RolledBack => BTreeMap::new(),
    };
    let events = loaded
        .grants
        .issued(0)
        .and_then(|g| loaded.grants.access(g.id()))
        .map(|a| a.caused_by(Cause { execution, node: NodeId(0) }))
        .map_or_else(Vec::new, |a| fs.execution_events(&a).expect("the set's own grant is live"));
    ExecutionReceipt {
        execution,
        graph: loaded.graph.id,
        principal: loaded.grants.principal(),
        grants: loaded
            .grants
            .iter()
            .map(|g| GrantRecord { grant: g.id(), target: g.target().clone(), rights: g.rights() })
            .collect(),
        nodes: records,
        schedule,
        events,
        transaction,
        outputs,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::builtin::{Echo, Fail, Upper};
    use capability::{Capability, CapabilityContext, Determinism, Fault};
    use meatfs::{Access, EventKind, Seed};
    use meatyaml::{Edge, Graph, Node, WriteOp};

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    const ECHO: &str = include_str!("../../../examples/echo.meat.yaml");
    const BUTCHER: &str = include_str!("../../../examples/butcher.meat.yaml");
    const FAILURE: &str = include_str!("../../../examples/failure.meat.yaml");

    /// Effectful test capability: writes its input to every attached path,
    /// and additionally to the path named by `{ also: "/..." }` if given.
    struct Stamp;
    impl Capability for Stamp {
        type Input = Value;
        type Output = Value;
        fn describe(&self) -> &'static str {
            "write input to attached paths"
        }
        fn meta(&self) -> CapabilityMeta {
            CapabilityMeta::new(Purity::Effectful, Determinism::Deterministic)
        }
        fn invoke(&self, cx: &CapabilityContext<'_>, input: Value) -> Result<Value, Fault> {
            let fx = cx.effects()?;
            for entry in fx.grants() {
                fx.write(&entry.path, input.clone())?;
            }
            if let Some(also) = input.get("also").and_then(Value::as_text) {
                fx.write(&Path::parse(also)?, input.clone())?;
            }
            Ok(input)
        }
    }

    fn root(host: &GrantSet) -> Access<'_> {
        host.access(host.iter().next().unwrap().id()).unwrap()
    }

    fn boot(seed: u64) -> (MeatFs, GrantSet) {
        let (fs, host) = MeatFs::genesis(Seed::fixed(seed));
        let r = root(&host);
        capability::mount(&fs, &r, &p("/tools/echo"), Echo).unwrap();
        capability::mount(&fs, &r, &p("/tools/text/upper"), Upper).unwrap();
        capability::mount(&fs, &r, &p("/tools/test/fail"), Fail).unwrap();
        capability::mount(&fs, &r, &p("/tools/test/stamp"), Stamp).unwrap();
        capability::mount(&fs, &r, &p("/models/mock/infer"), model::MockModel).unwrap();
        capability::mount(&fs, &r, &p("/models/mock/fail"), model::MockFail).unwrap();
        fs.bind(&r, &p("/memory/context"), Value::map([("text", Value::from("boot"))])).unwrap();
        (fs, host)
    }

    fn policy() -> Policy {
        Policy::default()
            .allow(p("/tools"), Rights::INVOKE)
            .allow(p("/models"), Rights::INVOKE)
            .allow(p("/memory"), Rights::READ)
            .allow(p("/state"), Rights::READ | Rights::WRITE)
    }

    fn snapshot(fs: &MeatFs, host: &GrantSet) -> Vec<(Path, ObjectId, Option<Value>)> {
        fs.snapshot(&root(host), &Path::root()).unwrap()
    }

    fn run(fs: &MeatFs, src: &str) -> Result<ExecutionReceipt, LoadError> {
        let program = meatyaml::compile(src).unwrap();
        Ok(execute(fs, &load(fs, &policy(), &program)?))
    }

    fn text(s: &str) -> Value {
        Value::map([("text", Value::from(s))])
    }

    #[test]
    fn butcher_acceptance() {
        let (fs, host) = boot(42);
        let receipt = run(&fs, BUTCHER).unwrap();
        assert!(receipt.succeeded(), "{:?}", receipt.error);
        assert_eq!(receipt.outputs, BTreeMap::from([("upper".into(), text("MEAT")), ("echo".into(), text("MEAT"))]));
        assert_eq!(receipt.schedule, [0, 1, 2, 3].map(NodeId));
        assert!(receipt.nodes.iter().all(|n| n.state == NodeState::Succeeded));

        // Objects created by the execution are recorded and visible.
        let upper = fs.resolve(&p("/state/upper")).unwrap();
        assert_eq!(receipt.nodes[2].object, Some(upper));
        assert_eq!(fs.read(&root(&host), upper).unwrap(), text("MEAT"));
        let written: Vec<_> = receipt
            .events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::Written { .. }))
            .map(|e| (e.cause.unwrap().node, e.object, e.grant))
            .collect();
        assert_eq!(
            written,
            vec![
                (NodeId(2), upper, Some(receipt.nodes[2].grant)),
                (NodeId(3), receipt.nodes[3].object.unwrap(), Some(receipt.nodes[3].grant))
            ]
        );
    }

    #[test]
    fn failure_rolls_back_and_blocks() {
        let (fs, host) = boot(42);
        let before = snapshot(&fs, &host);
        let receipt = run(&fs, FAILURE).unwrap();

        assert_eq!(snapshot(&fs, &host), before, "namespace unchanged");
        assert_eq!(receipt.transaction, TxOutcome::RolledBack);
        assert!(receipt.outputs.is_empty());
        let states: Vec<_> = receipt.nodes.iter().map(|n| n.state).collect();
        assert_eq!(states, [NodeState::Succeeded, NodeState::Failed, NodeState::Blocked]);
        assert_eq!(receipt.nodes[2].error.as_ref().unwrap().kind, ExecutionErrorKind::DependencyFailed);

        let error = receipt.error.unwrap();
        let fail = fs.resolve(&p("/tools/test/fail")).unwrap();
        assert_eq!(
            (error.node, error.object, error.grant, error.kind),
            (Some(NodeId(1)), Some(fail), Some(receipt.nodes[1].grant), ExecutionErrorKind::CapabilityFailed)
        );

        // The evidence survives: A was staged, the failure invoked, A rolled back.
        let a = receipt.nodes[0].object.unwrap();
        let kinds: Vec<_> = receipt.events.iter().map(|e| (e.object, e.kind.clone())).collect();
        assert_eq!(
            kinds,
            vec![(a, EventKind::Staged), (fail, EventKind::Invoked { ok: false }), (a, EventKind::RolledBack)]
        );
    }

    #[test]
    fn loading_is_observational() {
        let (fs, host) = boot(1);
        let before = snapshot(&fs, &host);
        load(&fs, &policy(), &meatyaml::compile(BUTCHER).unwrap()).unwrap();
        let denied = load(&fs, &Policy::default(), &meatyaml::compile(BUTCHER).unwrap());
        assert!(matches!(denied, Err(LoadError::Authority(meatfs::Error::PolicyDenied { .. }))));
        assert_eq!(snapshot(&fs, &host), before);
    }

    #[test]
    fn execution_is_deterministic() {
        let a = run(&boot(7).0, BUTCHER).unwrap();
        let b = run(&boot(7).0, BUTCHER).unwrap();
        assert_eq!(a, b);
        let f1 = run(&boot(7).0, FAILURE).unwrap();
        let f2 = run(&boot(7).0, FAILURE).unwrap();
        assert_eq!(f1, f2);
    }

    #[test]
    fn scheduling_follows_edges_not_node_ids() {
        let (fs, host) = boot(1);
        let write = |id, value: i64| Node {
            id: NodeId(id),
            label: None,
            op: Op::Write(WriteOp { target: p("/state/x"), value: Some(Value::Int(value)) }),
        };
        // n1 must precede n0: the final value is n0's.
        let graph = Graph::new(
            vec![write(0, 0), write(1, 1)],
            vec![Edge { from: NodeId(1), to: NodeId(0), kind: EdgeKind::Order }],
            vec![],
        );
        let program = Program { agent: "a".into(), authority: vec![], graph };
        let receipt = execute(&fs, &load(&fs, &policy(), &program).unwrap());
        assert_eq!(receipt.schedule, vec![NodeId(1), NodeId(0)]);
        let x = fs.resolve(&p("/state/x")).unwrap();
        assert_eq!(fs.read(&root(&host), x).unwrap(), Value::Int(0));
    }

    #[test]
    fn capabilities_get_only_node_scoped_authority() {
        let (fs, host) = boot(1);
        let program = r#"
agent: { name: a }
authority:
  - { path: /tools/test/stamp, rights: [invoke] }
  - { path: /state, rights: [write] }
flow:
  - id: own
    write: { path: /state/b, value: 0 }
  - invoke:
      path: /tools/test/stamp
      input: { n: 1 }
      uses: [{ path: /state/a, rights: [write] }]
"#;
        let receipt = run(&fs, program).unwrap();
        assert!(receipt.succeeded(), "{:?}", receipt.error);
        let a = fs.resolve(&p("/state/a")).unwrap();
        assert_eq!(fs.read(&root(&host), a).unwrap(), Value::map([("n", Value::Int(1))]));

        // The execution holds a grant for /state/b, but the stamp node does not.
        let (fs, host) = boot(1);
        let before = snapshot(&fs, &host);
        let escalate = program.replace("input: { n: 1 }", "input: { also: /state/b }");
        let receipt = run(&fs, &escalate).unwrap();
        let error = receipt.error.unwrap();
        assert_eq!((error.node, error.kind), (Some(NodeId(1)), ExecutionErrorKind::AuthorityDenied));
        assert_eq!(snapshot(&fs, &host), before, "the stamp's own write rolled back too");
    }

    #[test]
    fn pure_capabilities_cannot_be_given_uses() {
        let (fs, _) = boot(1);
        let e = run(
            &fs,
            r#"
agent: { name: a }
authority: [{ path: /tools, rights: [invoke] }, { path: /state, rights: [write] }]
flow:
  - invoke: { path: /tools/echo, input: { text: x }, uses: [{ path: /state/a, rights: [write] }] }
"#,
        );
        assert!(matches!(e, Err(LoadError::InvalidGraph(_))));
    }

    #[test]
    fn escalation_fails_at_policy() {
        let (fs, _) = boot(1);
        let e = run(&fs, include_str!("../../../examples/escalate-declared.meat.yaml")).err().unwrap();
        assert_eq!(
            e,
            LoadError::Authority(meatfs::Error::PolicyDenied { path: p("/memory/context"), rights: Rights::WRITE })
        );
    }

    #[test]
    fn rejects_structurally_invalid_ir() {
        let program = meatyaml::compile(ECHO).unwrap();
        let mut cyclic = program.graph.clone();
        cyclic.edges.push(Edge { from: NodeId(1), to: NodeId(0), kind: EdgeKind::Order });
        assert!(matches!(validate(&cyclic), Err(LoadError::InvalidGraph(_))));

        let mut double_input = program.graph.clone();
        double_input.edges.push(Edge { from: NodeId(0), to: NodeId(1), kind: EdgeKind::Data });
        assert!(matches!(validate(&double_input), Err(LoadError::InvalidGraph(_))));
    }

    #[test]
    fn retired_graphs_cannot_run() {
        let (fs, _) = boot(1);
        let program = meatyaml::compile(ECHO).unwrap();
        let loaded = load(&fs, &policy(), &program).unwrap();
        let grant = loaded.nodes()[0].grant;
        loaded.retire(&fs);
        let again = load(&fs, &policy(), &program).unwrap();
        assert_ne!(again.nodes()[0].grant, grant);
        assert!(execute(&fs, &again).succeeded());
    }

    const THINKER: &str = include_str!("../../../examples/thinker.meat.yaml");
    const MODEL_FAILURE: &str = include_str!("../../../examples/model-failure.meat.yaml");
    const FORK: &str = include_str!("../../../examples/fork.meat.yaml");

    fn message(role: &str, content: &str) -> Value {
        Value::map([("role", Value::from(role)), ("content", Value::from(content))])
    }

    #[test]
    fn model_is_an_ordinary_capability() {
        let (fs, host) = boot(42);
        let receipt = run(&fs, THINKER).unwrap();
        assert!(receipt.succeeded(), "{:?}", receipt.error);

        let answer = Value::map([
            ("message", message("assistant", "MEAT")),
            ("usage", Value::map([("input_tokens", Value::Int(2)), ("output_tokens", Value::Int(1))])),
            ("finish", Value::from("stop")),
        ]);
        assert_eq!(receipt.outputs, BTreeMap::from([("answer".into(), answer.clone())]));
        let stored = fs.resolve(&p("/state/answer")).unwrap();
        assert_eq!(fs.read(&root(&host), stored).unwrap(), answer);

        // Resolved to the model's identity, holding exactly its own invoke grant.
        let model = fs.resolve(&p("/models/mock/infer")).unwrap();
        let infer = &receipt.nodes[0];
        assert_eq!(infer.object, Some(model));
        assert!(infer.uses.is_empty());
        let grant = receipt.grants.iter().find(|g| g.grant == infer.grant).unwrap();
        assert_eq!((&grant.target, grant.rights), (&Target::Object(model), Rights::INVOKE));

        // Provenance: input, declared properties and implementation.
        let meta = infer.capability.clone().unwrap();
        assert_eq!((meta.purity, meta.determinism), (Purity::Effectful, Determinism::Deterministic));
        assert_eq!(meta.invocation.implementation.as_deref(), Some(model::MockModel::IMPLEMENTATION));
        assert_eq!(meta.invocation.revision.as_deref(), Some(model::MockModel::REVISION));
        let messages = infer.input.as_ref().unwrap().get("messages").unwrap();
        assert_eq!(messages, &Value::List(vec![message("system", "uppercase"), message("user", "meat")]));

        // Journaled like any invocation, caused by its node with its grant.
        let invoked: Vec<_> = receipt.events.iter().filter(|e| e.object == model).map(|e| (&e.kind, e.grant)).collect();
        assert_eq!(invoked, vec![(&EventKind::Invoked { ok: true }, Some(infer.grant))]);

        assert_eq!(run(&boot(42).0, THINKER).unwrap(), receipt, "replayable with identical seed and implementation");
    }

    #[test]
    fn model_failure_rolls_back() {
        let (fs, host) = boot(42);
        let before = snapshot(&fs, &host);
        let receipt = run(&fs, MODEL_FAILURE).unwrap();
        assert_eq!(snapshot(&fs, &host), before);
        assert_eq!(receipt.transaction, TxOutcome::RolledBack);
        let states: Vec<_> = receipt.nodes.iter().map(|n| n.state).collect();
        assert_eq!(states, [NodeState::Succeeded, NodeState::Failed, NodeState::Blocked]);
        let error = receipt.error.unwrap();
        let fail = fs.resolve(&p("/models/mock/fail")).unwrap();
        assert_eq!(
            (error.node, error.object, error.grant, error.kind),
            (Some(NodeId(1)), Some(fail), Some(receipt.nodes[1].grant), ExecutionErrorKind::CapabilityFailed)
        );
    }

    #[test]
    fn model_branch_is_independent_of_tool_branch() {
        let (fs, _) = boot(42);
        let program = meatyaml::compile(FORK).unwrap();
        // tool → store_tool and model → store_model, nothing across.
        let edges: Vec<_> = program.graph.edges.iter().map(|e| (e.from.0, e.to.0, e.kind)).collect();
        assert_eq!(edges, vec![(0, 2, EdgeKind::Data), (1, 3, EdgeKind::Data)]);

        let receipt = execute(&fs, &load(&fs, &policy(), &program).unwrap());
        assert!(receipt.succeeded());
        assert_eq!(receipt.outputs["tool"], text("MEAT"));
        assert_eq!(receipt.outputs["model"].get("message"), Some(&message("assistant", "MEAT")));
        assert_eq!(receipt.nodes[0].capability.as_ref().unwrap().purity, Purity::Pure);
        assert_eq!(receipt.nodes[1].capability.as_ref().unwrap().purity, Purity::Effectful);
    }
}

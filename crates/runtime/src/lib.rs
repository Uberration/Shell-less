//! The first MEAT IR backend: a deterministic, transactional Rust interpreter.
//!
//! ```text
//! Program ──load──▶ Loaded ──execute──▶ ExecutionOutcome { outputs, receipt }
//!            │                 │
//!            │                 ├─ one MeatFS transaction per execution
//!            │                 ├─ nodes run when their dependencies succeed; ties → lowest NodeId
//!            │                 ├─ failure blocks dependents, then discards staged MeatFS changes
//!            │                 ├─ every node acts only with the grants attached to it
//!            │                 └─ the receipt keeps payloads only as the host's capture policy allows
//!            │
//!            ├─ validate IR structure and host composition limits
//!            ├─ AuthorityRequest → host Policy → GrantSet   (observational: no state changes)
//!            ├─ project grants per node: target grant + `uses` grants
//!            └─ order declared-effectful invocations conservatively (derived edges)
//! ```
//!
//! # Atomicity boundary
//!
//! Atomicity covers MeatFS state only. Success commits staged MeatFS
//! changes; failure discards them; the audit journal keeps both. Effects
//! outside MeatFS — a request a backend already processed, a counter it
//! already bumped — are not undone. An invocation that completed stays
//! `Succeeded` even when the transaction rolls back, and a commit conflict is
//! reported, never retried: retrying could invoke a model again.
//!
//! # Declared metadata
//!
//! Purity, determinism and implementation identity are what a capability
//! *declares*. They are recorded as such and do not, on their own, justify
//! caching, retries, speculative parallelism or claims of verified replay.
//! Compiled-in capabilities are trusted host code; grants bound what they
//! can do to MeatFS, not what arbitrary Rust can do.

mod compose;

use meatfs::{
    CapabilityMeta, Cause, Event, EventKind, ExecutionId, GrantId, GrantSet, MeatFs, NodeGrant, NodeGrants, NodeId,
    ObjectId, Path, Policy, PrincipalId, Purity, Rights, Target, Transaction, Value,
};
use meatyaml::{EdgeKind, Graph, GraphId, Op, Program};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Host-controlled bounds on composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Deepest nesting of one `compose` expression.
    pub max_depth: usize,
    /// Most constructors in one `compose` expression.
    pub max_exprs: usize,
    /// Largest value one `compose` node may construct (see `compose::size`).
    pub max_value_size: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_depth: 32, max_exprs: 4096, max_value_size: 1 << 20 }
    }
}

/// The host's capture policy and the recorded form of content, shared with
/// MeatFS so receipts and the journal follow one rule.
pub use meatfs::{CapturedValue, ContentCapture};

/// Why a program could not be loaded. Nothing has executed and the
/// namespace is unchanged. The record is made under the host's capture
/// policy: under `Omit` it holds no source text (no paths, no capability
/// messages), only kinds, node ids, rights and opaque identities.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadError {
    /// The IR violates a structural rule.
    InvalidGraph { node: Option<NodeId>, reason: &'static str },
    /// A composition exceeds host limits.
    LimitExceeded { node: NodeId },
    /// Authority resolution failed: policy denial, unbound name, …
    Authority {
        /// The MeatFS error variant, e.g. `PolicyDenied`.
        kind: &'static str,
        /// Content-free summary: kind, rights and opaque identities.
        summary: String,
        /// The full error, retained only under `Inline`.
        detail: Option<String>,
    },
}

impl LoadError {
    fn authority(e: meatfs::Error, capture: ContentCapture) -> LoadError {
        LoadError::Authority { kind: e.kind(), summary: e.redacted(), detail: capture.text(|| e.to_string()) }
    }

    /// Render the record. Shows exactly what the record retained.
    pub fn render(&self) -> String {
        match self {
            LoadError::InvalidGraph { node: Some(node), reason } => {
                format!("load error InvalidGraph node={node}: {reason}")
            }
            LoadError::InvalidGraph { node: None, reason } => format!("load error InvalidGraph: {reason}"),
            LoadError::LimitExceeded { node } => format!("load error LimitExceeded node={node}"),
            LoadError::Authority { detail: Some(detail), .. } => format!("load error Authority: {detail}"),
            LoadError::Authority { summary, .. } => format!("load error Authority: {summary}"),
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for LoadError {}

/// What a node's operation acts on, fixed at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedTarget {
    Object(ObjectId),
    /// An unbound name a write node will create when it executes.
    New(Path),
    /// A `compose` node: it acts on nothing.
    None,
}

/// A node with its statically resolved target and its exact authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedNode {
    pub id: NodeId,
    pub target: ResolvedTarget,
    /// The grant for the node's own operation; `None` for `compose`.
    pub grant: Option<GrantId>,
    /// Grants attached to an invocation: all its capability may use.
    pub uses: Vec<NodeGrant>,
    /// For invocations: the capability's declaration as read at load. The
    /// loader schedules by it and execution requires it to be unchanged.
    pub declared: Option<CapabilityMeta>,
}

/// A resolved execution graph: IR, issued authority, per-node projection.
pub struct Loaded {
    graph: Graph,
    grants: GrantSet,
    nodes: Vec<ResolvedNode>,
    /// Order edges the loader added between declared-effectful invocations.
    derived: Vec<(NodeId, NodeId)>,
    preds: Vec<Vec<usize>>,
    succs: Vec<Vec<usize>>,
    limits: Limits,
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

    pub fn derived_order(&self) -> &[(NodeId, NodeId)] {
        &self.derived
    }

    /// Revoke the graph's authority. It cannot run again.
    pub fn retire(self, fs: &MeatFs) {
        fs.retire(self.grants);
    }
}

type Adjacency = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Validate IR structure, returning predecessor and successor lists.
///
/// Rules: node ids are dense and match their index; every edge joins two
/// existing nodes; a non-compose node has at most one data input, and none
/// if it reads or carries literal data; writes have data; a compose node's
/// data inputs are exactly the distinct sources it selects from; outputs
/// name existing nodes; the graph is acyclic.
fn validate(graph: &Graph) -> Result<Adjacency, LoadError> {
    let invalid = |node: Option<NodeId>, reason: &'static str| Err(LoadError::InvalidGraph { node, reason });
    let n = graph.nodes.len();
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id != NodeId(i as u32) {
            return invalid(Some(NodeId(i as u32)), "node id does not match its index");
        }
    }
    let mut preds = vec![Vec::new(); n];
    let mut succs = vec![Vec::new(); n];
    let mut data_in: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for edge in &graph.edges {
        let (from, to) = (edge.from.0 as usize, edge.to.0 as usize);
        if from >= n || to >= n {
            return invalid(Some(edge.to), "edge endpoint outside the graph");
        }
        if edge.kind == EdgeKind::Data {
            data_in[to].push(edge.from);
        }
        preds[to].push(from);
        succs[from].push(to);
    }
    for (node, inputs) in graph.nodes.iter().zip(&mut data_in) {
        inputs.sort_unstable();
        match &node.op {
            Op::Compose(expr) => {
                if *inputs != expr.sources() {
                    return invalid(Some(node.id), "data inputs do not match the sources it selects");
                }
            }
            _ if inputs.len() > 1 => return invalid(Some(node.id), "more than one data input"),
            Op::Read(_) if !inputs.is_empty() => return invalid(Some(node.id), "read with a data input"),
            op if op.literal().is_some() && !inputs.is_empty() => {
                return invalid(Some(node.id), "both literal data and a data input")
            }
            Op::Write(w) if w.value.is_none() && inputs.is_empty() => {
                return invalid(Some(node.id), "write without data")
            }
            _ => {}
        }
    }
    if let Some(o) = graph.outputs.iter().find(|o| o.source.0 as usize >= n) {
        return invalid(Some(o.source), "output names a missing node");
    }
    if topological(&preds, &succs).len() != n {
        return invalid(None, "graph has a cycle");
    }
    Ok((preds, succs))
}

/// Kahn's algorithm with lowest-index tie-break. Shorter than `n` on a cycle.
fn topological(preds: &[Vec<usize>], succs: &[Vec<usize>]) -> Vec<usize> {
    let mut remaining: Vec<usize> = preds.iter().map(Vec::len).collect();
    let mut ready: BTreeSet<usize> = (0..preds.len()).filter(|&i| remaining[i] == 0).collect();
    let mut order = Vec::with_capacity(preds.len());
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &s in &succs[i] {
            remaining[s] -= 1;
            if remaining[s] == 0 {
                ready.insert(s);
            }
        }
    }
    order
}

fn reaches(succs: &[Vec<usize>], from: usize, to: usize) -> bool {
    let mut stack = vec![from];
    let mut seen = vec![false; succs.len()];
    while let Some(n) = stack.pop() {
        if n == to {
            return true;
        }
        if !std::mem::replace(&mut seen[n], true) {
            stack.extend(&succs[n]);
        }
    }
    false
}

/// Resolve a program into an executable graph under host `policy` and
/// `limits`. A failure is recorded under `capture`.
///
/// Observational: on success or failure, the namespace is unchanged.
pub fn load(
    fs: &MeatFs,
    policy: &Policy,
    limits: Limits,
    capture: ContentCapture,
    program: &Program,
) -> Result<Loaded, LoadError> {
    let (mut preds, mut succs) = validate(&program.graph)?;
    for node in &program.graph.nodes {
        if let Op::Compose(expr) = &node.op {
            if expr.depth() > limits.max_depth || expr.count() > limits.max_exprs {
                return Err(LoadError::LimitExceeded { node: node.id });
            }
        }
    }
    let grants = fs.issue(policy, &program.request()).map_err(|e| LoadError::authority(e, capture))?;

    // `request()` lists, per node, its target (if any) then its uses.
    let mut issued = grants.iter();
    let mut nodes = Vec::with_capacity(program.graph.nodes.len());
    for node in &program.graph.nodes {
        let (target, grant) = match node.op.target() {
            None => (ResolvedTarget::None, None),
            Some(_) => {
                let grant = issued.next().expect("one grant per want");
                let target = match grant.target() {
                    Target::Object(o) => ResolvedTarget::Object(*o),
                    Target::Name(p) => ResolvedTarget::New(p.clone()),
                    Target::Namespace(_) => unreachable!("issue never grants namespaces"),
                };
                (target, Some(grant.id()))
            }
        };
        let uses: Vec<NodeGrant> = node
            .op
            .uses()
            .iter()
            .map(|u| NodeGrant { path: u.path.clone(), grant: issued.next().expect("one grant per want").id() })
            .collect();
        let declared = match (&node.op, &target) {
            (Op::Invoke(_), ResolvedTarget::Object(o)) => fs.meta(*o),
            _ => None,
        };
        if let (Some(meta), false) = (&declared, uses.is_empty()) {
            if meta.purity == Purity::Pure {
                return Err(LoadError::InvalidGraph {
                    node: Some(node.id),
                    reason: "pure capability cannot be given `uses`",
                });
            }
        }
        nodes.push(ResolvedNode { id: node.id, target, grant, uses, declared });
    }
    drop(issued);

    // Effects outside MeatFS are invisible to footprints, so invocations not
    // declared pure — including those whose purity is unknown — are chained
    // along one deterministic topological order of the source graph. Every
    // added edge points forward in that order, so none can form a cycle.
    let effectful: Vec<usize> = topological(&preds, &succs)
        .into_iter()
        .filter(|&i| {
            matches!(program.graph.nodes[i].op, Op::Invoke(_))
                && nodes[i].declared.as_ref().is_none_or(|m| m.purity != Purity::Pure)
        })
        .collect();
    let mut derived = Vec::new();
    for pair in effectful.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        if !reaches(&succs, from, to) {
            preds[to].push(from);
            succs[from].push(to);
            derived.push((NodeId(from as u32), NodeId(to as u32)));
        }
    }
    Ok(Loaded { graph: program.graph.clone(), grants, nodes, derived, preds, succs, limits })
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
    /// A composition selected a key that is not there.
    MissingValue,
    /// A composition selected through a value of the wrong kind.
    WrongValueKind,
    /// A composition selected past the end of a list.
    IndexOutOfRange,
    /// A composition exceeded a host limit.
    LimitExceeded,
    /// An invoked capability no longer declares what it declared at load.
    DeclarationChanged,
}

/// A structured failure: what failed, where, with which authority.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionError {
    pub execution: ExecutionId,
    pub node: Option<NodeId>,
    pub object: Option<ObjectId>,
    pub grant: Option<GrantId>,
    pub kind: ExecutionErrorKind,
    /// Human-readable context, which may echo content. Kept only under
    /// [`ContentCapture::Inline`]; never needed to interpret the failure.
    pub detail: Option<String>,
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.node.map(|n| format!("{n}: ")).unwrap_or_default();
        write!(f, "{node}{:?}", self.kind)?;
        if let Some(detail) = &self.detail {
            write!(f, ": {detail}")?;
        }
        Ok(())
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
        E::DeclarationChanged(o) => (K::DeclarationChanged, Some(*o)),
        E::Conflict(o) => (K::TransactionConflict, Some(*o)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxOutcome {
    Committed,
    RolledBack,
}

/// What became of the MeatFS changes a node staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedChanges {
    None,
    Committed,
    /// Discarded. Says nothing about effects outside MeatFS.
    RolledBack,
}

/// A grant target as a receipt records it: identities as they are, names
/// only as the capture policy allows. The live grant keeps its real target.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordedTarget {
    Object(ObjectId),
    /// An unbound name the grant allowed creating.
    Name(CapturedValue),
    Namespace(CapturedValue),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrantRecord {
    pub grant: GrantId,
    pub target: RecordedTarget,
    pub rights: Rights,
}

/// A graph output as a receipt records it, identified by its position in
/// the graph's outputs; its name is source text and captured like content.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputRecord {
    pub index: usize,
    pub name: CapturedValue,
    pub source: NodeId,
    pub value: CapturedValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeRecord {
    pub node: NodeId,
    pub op: &'static str,
    /// Whether the node's own work completed. Independent of whether its
    /// MeatFS changes were kept; see `staged`.
    pub state: NodeState,
    /// The object touched; for a creating write, the object it created.
    pub object: Option<ObjectId>,
    pub grant: Option<GrantId>,
    pub uses: Vec<GrantId>,
    /// For invocations: execution properties and implementation identity
    /// as *declared* by the invoked object. Not verified.
    pub declared: Option<CapabilityMeta>,
    pub input: Option<CapturedValue>,
    pub output: Option<CapturedValue>,
    pub staged: StagedChanges,
    pub error: Option<ExecutionError>,
}

/// The substrate's audit record of one execution, successful or not.
///
/// Connects execution → node → grant used → object touched → event
/// produced. Payloads appear only as the host's [`ContentCapture`] allows.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    pub execution: ExecutionId,
    pub graph: GraphId,
    pub principal: PrincipalId,
    pub capture: ContentCapture,
    pub grants: Vec<GrantRecord>,
    /// Order edges the loader added between declared-effectful invocations.
    pub derived_order: Vec<(NodeId, NodeId)>,
    /// Every node, by `NodeId`.
    pub nodes: Vec<NodeRecord>,
    /// The order nodes actually ran in.
    pub schedule: Vec<NodeId>,
    /// Audit events caused by this execution, including discarded work.
    pub events: Vec<Event>,
    pub transaction: TxOutcome,
    /// Graph outputs as recorded, in graph order; empty unless the
    /// transaction committed.
    pub outputs: Vec<OutputRecord>,
    /// The failure that rolled the execution back, if any.
    pub error: Option<ExecutionError>,
}

impl ExecutionReceipt {
    pub fn succeeded(&self) -> bool {
        self.transaction == TxOutcome::Committed
    }
}

/// The two products of an execution, kept apart: the actual graph outputs
/// for the caller, and the receipt for the audit trail.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionOutcome {
    /// Graph outputs; empty unless the transaction committed.
    pub outputs: BTreeMap<String, Value>,
    pub receipt: ExecutionReceipt,
}

enum Failure {
    Substrate(meatfs::Error),
    Compose(compose::ComposeFault),
}

struct Run<'a> {
    loaded: &'a Loaded,
    tx: Transaction<'a>,
    execution: ExecutionId,
    capture: ContentCapture,
    inputs: Vec<Option<Value>>,
    outputs: Vec<Option<Value>>,
    objects: Vec<Option<ObjectId>>,
    errors: Vec<Option<ExecutionError>>,
}

impl Run<'_> {
    fn input(&self, i: usize) -> Value {
        let node = &self.loaded.graph.nodes[i];
        match (node.op.literal(), self.loaded.graph.data_input(node.id)) {
            (Some(literal), _) => literal.clone(),
            (None, Some(from)) => self.outputs[from.0 as usize].clone().expect("dependencies succeeded"),
            (None, None) => Value::Null,
        }
    }

    fn step(&mut self, i: usize) -> Result<(Option<ObjectId>, Value), Failure> {
        let (node, resolved, grants) = (&self.loaded.graph.nodes[i], &self.loaded.nodes[i], &self.loaded.grants);
        if let Op::Compose(expr) = &node.op {
            let outputs = &self.outputs;
            let output = |n: NodeId| outputs[n.0 as usize].as_ref().expect("dependencies succeeded");
            let mut budget = self.loaded.limits.max_value_size;
            return compose::evaluate(expr, &output, &mut budget).map(|v| (None, v)).map_err(Failure::Compose);
        }
        let grant = resolved.grant.expect("non-compose nodes hold a grant");
        let access = grants.access(grant).expect("resolved grant is in the set");
        let access = access.caused_by(Cause { execution: self.execution, node: node.id });
        let input = self.input(i);
        if !matches!(node.op, Op::Read(_)) {
            self.inputs[i] = Some(input.clone());
        }
        let result = match (&node.op, &resolved.target) {
            (Op::Read(_), ResolvedTarget::Object(o)) => self.tx.read(&access, *o).map(|v| (Some(*o), v)),
            (Op::Invoke(_), ResolvedTarget::Object(o)) => {
                let attached = NodeGrants::new(grants, &resolved.uses).expect("uses are in the set");
                self.tx.invoke(&access, *o, input, attached, resolved.declared.as_ref()).map(|v| (Some(*o), v))
            }
            (Op::Write(_), ResolvedTarget::Object(o)) => {
                self.tx.write(&access, *o, input.clone()).map(|_| (Some(*o), input))
            }
            (Op::Write(_), ResolvedTarget::New(p)) => {
                self.tx.create(&access, p, input.clone()).map(|o| (Some(o), input))
            }
            (_, ResolvedTarget::New(p)) => Err(meatfs::Error::Unbound(p.clone())),
            (Op::Compose(_), _) | (_, ResolvedTarget::None) => unreachable!("compose returned above"),
        };
        result.map_err(Failure::Substrate)
    }

    /// Structure a failure of node `i`.
    fn failure(&self, i: usize, failure: Failure) -> ExecutionError {
        let resolved = &self.loaded.nodes[i];
        let (kind, object, detail) = match failure {
            Failure::Substrate(e) => {
                let (kind, object) = classify(&e);
                (kind, object, e.to_string())
            }
            Failure::Compose(f) => (f.kind, None, f.detail),
        };
        let object = object.or(match &resolved.target {
            ResolvedTarget::Object(o) => Some(*o),
            _ => None,
        });
        ExecutionError {
            execution: self.execution,
            node: Some(resolved.id),
            object,
            grant: resolved.grant,
            kind,
            detail: self.capture.text(|| detail),
        }
    }

    /// Mark every transitive dependent of `failed` as blocked.
    fn block_dependents(&mut self, failed: usize, states: &mut [NodeState]) {
        let mut stack = self.loaded.succs[failed].clone();
        while let Some(i) = stack.pop() {
            if states[i] != NodeState::Blocked {
                states[i] = NodeState::Blocked;
                self.errors[i] = Some(ExecutionError {
                    execution: self.execution,
                    node: Some(self.loaded.nodes[i].id),
                    object: None,
                    grant: self.loaded.nodes[i].grant,
                    kind: ExecutionErrorKind::DependencyFailed,
                    detail: self.capture.text(|| format!("depends on failed {}", self.loaded.nodes[failed].id)),
                });
                stack.extend(&self.loaded.succs[i]);
            }
        }
    }
}

/// Run a loaded graph inside one MeatFS transaction: commit if every node
/// succeeds, otherwise discard staged changes. Always produces a receipt,
/// retaining payloads only as `capture` allows.
pub fn execute(fs: &MeatFs, loaded: &Loaded, capture: ContentCapture) -> ExecutionOutcome {
    execute_with(fs, loaded, capture, &mut || {})
}

/// [`execute`], calling `before_commit` immediately before the real commit
/// is attempted. Lets the test harness act as a second writer.
pub(crate) fn execute_with(
    fs: &MeatFs,
    loaded: &Loaded,
    capture: ContentCapture,
    before_commit: &mut dyn FnMut(),
) -> ExecutionOutcome {
    let execution = fs.new_execution();
    let n = loaded.graph.nodes.len();
    let mut run = Run {
        loaded,
        tx: fs.transaction(),
        execution,
        capture,
        inputs: vec![None; n],
        outputs: vec![None; n],
        objects: loaded
            .nodes
            .iter()
            .map(|r| match r.target {
                ResolvedTarget::Object(o) => Some(o),
                _ => None,
            })
            .collect(),
        errors: vec![None; n],
    };

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
                if object.is_some() {
                    run.objects[i] = object;
                }
                run.outputs[i] = Some(output);
                for &s in &loaded.succs[i] {
                    let ready_now = loaded.preds[s].iter().all(|&p| states[p] == NodeState::Succeeded);
                    if states[s] == NodeState::Pending && ready_now {
                        states[s] = NodeState::Ready;
                        ready.insert(s);
                    }
                }
            }
            Err(failure) => {
                let e = run.failure(i, failure);
                states[i] = NodeState::Failed;
                run.errors[i] = Some(e.clone());
                run.block_dependents(i, &mut states);
                error = Some(e);
                break;
            }
        }
    }

    let Run { tx, inputs, outputs, objects, errors, .. } = run;
    let transaction = match error {
        Some(_) => {
            tx.rollback();
            TxOutcome::RolledBack
        }
        // Never retried: the graph may already have invoked effectful work.
        None => {
            before_commit();
            match tx.commit() {
                Ok(()) => TxOutcome::Committed,
                Err(e) => {
                    let (kind, object) = classify(&e);
                    let detail = capture.text(|| e.to_string());
                    error = Some(ExecutionError { execution, node: None, object, grant: None, kind, detail });
                    TxOutcome::RolledBack
                }
            }
        }
    };

    let events = loaded
        .grants
        .issued(0)
        .and_then(|g| loaded.grants.access(g.id()))
        .map(|a| a.caused_by(Cause { execution, node: NodeId(0) }))
        .map_or_else(Vec::new, |a| fs.execution_events(&a, capture).expect("the set's own grant is live"));
    let staged_by: BTreeSet<NodeId> =
        events.iter().filter(|e| e.kind == EventKind::Staged).filter_map(|e| e.cause.map(|c| c.node)).collect();

    let graph_outputs: BTreeMap<String, Value> = match transaction {
        TxOutcome::Committed => loaded
            .graph
            .outputs
            .iter()
            .map(|o| (o.name.clone(), outputs[o.source.0 as usize].clone().expect("committed nodes all ran")))
            .collect(),
        TxOutcome::RolledBack => BTreeMap::new(),
    };
    let nodes = (0..n)
        .map(|i| {
            let (node, resolved) = (&loaded.graph.nodes[i], &loaded.nodes[i]);
            NodeRecord {
                node: node.id,
                op: node.op.name(),
                state: states[i],
                object: objects[i],
                grant: resolved.grant,
                uses: resolved.uses.iter().map(|u| u.grant).collect(),
                declared: resolved.declared.clone(),
                input: inputs[i].as_ref().map(|v| capture.value(v)),
                output: outputs[i].as_ref().map(|v| capture.value(v)),
                staged: match (staged_by.contains(&node.id), transaction) {
                    (false, _) => StagedChanges::None,
                    (true, TxOutcome::Committed) => StagedChanges::Committed,
                    (true, TxOutcome::RolledBack) => StagedChanges::RolledBack,
                },
                error: errors[i].clone(),
            }
        })
        .collect();
    let receipt = ExecutionReceipt {
        execution,
        graph: loaded.graph.id,
        principal: loaded.grants.principal(),
        capture,
        grants: loaded
            .grants
            .iter()
            .map(|g| GrantRecord {
                grant: g.id(),
                target: match g.target() {
                    Target::Object(o) => RecordedTarget::Object(*o),
                    Target::Name(p) => RecordedTarget::Name(capture.name(p)),
                    Target::Namespace(p) => RecordedTarget::Namespace(capture.name(p)),
                },
                rights: g.rights(),
            })
            .collect(),
        derived_order: loaded.derived.clone(),
        nodes,
        schedule,
        events,
        transaction,
        outputs: match transaction {
            TxOutcome::Committed => loaded
                .graph
                .outputs
                .iter()
                .enumerate()
                .map(|(index, o)| OutputRecord {
                    index,
                    name: capture.label(&o.name),
                    source: o.source,
                    value: capture.value(&graph_outputs[&o.name]),
                })
                .collect(),
            TxOutcome::RolledBack => Vec::new(),
        },
        error,
    };
    ExecutionOutcome { outputs: graph_outputs, receipt }
}

#[cfg(test)]
mod tests;

use crate::authority::{Access, AuthorityRequest, Cause, Grant, GrantSet, NodeGrant, NodeGrants, Policy, Target};
use crate::id::IdGen;
use crate::{AuthorityDomainId, Error, ExecutionId, GrantId, ObjectId, Path, PrincipalId, Result, Rights, Seed, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};

/// Whether a capability may touch anything beyond its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purity {
    /// `input → output` only: no MeatFS access, no clocks, no randomness, no
    /// external state. Safe to cache, replay, parallelize and compile.
    Pure,
    /// May act on MeatFS through the grants attached to its node.
    Effectful,
}

/// Whether identical inputs (and, for effectful capabilities, identical
/// visible state) always produce identical outputs. Independent of
/// [`Purity`]: a pure function may be nondeterministic in principle, and an
/// effectful one deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Determinism {
    Deterministic,
    Nondeterministic,
}

/// Which implementation produced a result: provenance, not configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InvocationMeta {
    pub implementation: Option<String>,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityMeta {
    pub purity: Purity,
    pub determinism: Determinism,
    pub invocation: InvocationMeta,
}

impl CapabilityMeta {
    pub const fn new(purity: Purity, determinism: Determinism) -> Self {
        CapabilityMeta { purity, determinism, invocation: InvocationMeta { implementation: None, revision: None } }
    }

    pub fn implemented_by(mut self, implementation: impl Into<String>, revision: impl Into<String>) -> Self {
        self.invocation =
            InvocationMeta { implementation: Some(implementation.into()), revision: Some(revision.into()) };
        self
    }
}

/// Why an invocation did not produce an output.
#[derive(Debug, Clone, PartialEq)]
pub enum Fault {
    InvalidInput(String),
    Failed(String),
    /// A substrate operation performed by the capability failed.
    Substrate(Error),
}

impl From<Error> for Fault {
    fn from(e: Error) -> Self {
        Fault::Substrate(e)
    }
}

/// An invocable object: the raw, untyped form of a capability.
pub trait Invoke: Send + Sync {
    fn meta(&self) -> CapabilityMeta;

    /// A description of the object's contract, surfaced by `inspect`.
    fn signature(&self) -> Value {
        Value::Null
    }

    fn invoke(&self, cx: &CallContext<'_>, input: Value) -> std::result::Result<Value, Fault>;
}

/// What an invoked object is given. A pure capability receives no effects
/// at all, so it cannot reach MeatFS even by accident.
pub struct CallContext<'a> {
    pub object: ObjectId,
    effects: Option<Effects<'a>>,
}

impl<'a> CallContext<'a> {
    pub fn effects(&self) -> std::result::Result<&Effects<'a>, Fault> {
        self.effects.as_ref().ok_or(Fault::Substrate(Error::Impure(self.object)))
    }
}

/// MeatFS as seen by an effectful capability: only the grants attached to
/// its node, inside its execution's transaction.
pub struct Effects<'a> {
    tx: &'a Transaction<'a>,
    grants: NodeGrants<'a>,
    cause: Option<Cause>,
}

impl Effects<'_> {
    /// The authority attached to this invocation.
    pub fn grants(&self) -> &[NodeGrant] {
        self.grants.entries
    }

    fn access(&self, path: &Path) -> Result<Access<'_>> {
        let entry = self.grants.entries.iter().find(|e| e.path == *path).ok_or(Error::NotAttached(path.clone()))?;
        let access = self.grants.set.access(entry.grant).expect("NodeGrants only names grants in its set");
        Ok(match self.cause {
            Some(cause) => access.caused_by(cause),
            None => access,
        })
    }

    pub fn read(&self, path: &Path) -> Result<Value> {
        let access = self.access(path)?;
        match access.grant.target() {
            Target::Object(object) => self.tx.read(&access, *object),
            _ => Err(Error::Unbound(path.clone())),
        }
    }

    /// Create-or-replace, as the attached grant allows.
    pub fn write(&self, path: &Path, value: Value) -> Result<ObjectId> {
        let access = self.access(path)?;
        match access.grant.target() {
            Target::Object(object) => self.tx.write(&access, *object, value).map(|_| *object),
            _ => self.tx.create(&access, path, value),
        }
    }
}

enum Body {
    Data { value: Value, version: u64 },
    Capability(Arc<dyn Invoke>),
}

struct Object {
    body: Body,
    names: BTreeSet<Path>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Data,
    Capability,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inspection {
    pub object: ObjectId,
    pub kind: NodeKind,
    /// Data version; 0 for capabilities.
    pub version: u64,
    pub names: Vec<Path>,
    pub meta: Option<CapabilityMeta>,
    pub signature: Value,
}

/// One entry of the audit journal. The journal records attempts and their
/// outcomes; it is never rolled back.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub seq: u64,
    pub object: ObjectId,
    pub principal: PrincipalId,
    /// The grant exercised; `None` for actions of the namespace itself.
    pub grant: Option<GrantId>,
    pub cause: Option<Cause>,
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    Bound(Path),
    Read {
        version: u64,
    },
    /// A write or creation staged in a transaction, not yet visible.
    Staged,
    /// A change became visible state.
    Written {
        version: u64,
    },
    /// A staged change was discarded.
    RolledBack,
    Invoked {
        ok: bool,
    },
}

/// A live stream of events within a grant's target.
pub struct Subscription {
    rx: Receiver<Event>,
}

impl Subscription {
    pub fn try_next(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    pub fn drain(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }
}

fn covers(target: &Target, object: ObjectId, names: &BTreeSet<Path>) -> bool {
    match target {
        Target::Object(id) => *id == object,
        Target::Name(_) => false,
        Target::Namespace(prefix) => names.iter().any(|n| n.starts_with(prefix)),
    }
}

/// Whether `path` can be bound without colliding with, nesting inside, or
/// enclosing an existing name.
fn bindable<V>(names: &BTreeMap<Path, V>, path: &Path) -> Result<()> {
    let taken = path.is_root()
        || names.contains_key(path)
        || std::iter::successors(path.parent(), Path::parent).any(|a| names.contains_key(&a))
        || names.range((Bound::Excluded(path), Bound::Unbounded)).next().is_some_and(|(n, _)| n.starts_with(path));
    if taken {
        Err(Error::AlreadyBound(path.clone()))
    } else {
        Ok(())
    }
}

/// Who performed an action, for the journal.
#[derive(Clone, Copy)]
struct Who {
    principal: PrincipalId,
    grant: Option<GrantId>,
    cause: Option<Cause>,
}

impl Who {
    fn of(access: &Access<'_>) -> Who {
        Who { principal: access.grants.principal, grant: Some(access.grant.id()), cause: access.cause }
    }
}

#[derive(Default)]
struct Journal {
    seq: u64,
    events: Vec<Event>,
    subscribers: Vec<(Target, Sender<Event>)>,
}

struct State {
    ids: IdGen,
    objects: BTreeMap<ObjectId, Object>,
    names: BTreeMap<Path, ObjectId>,
    principals: BTreeMap<String, PrincipalId>,
    /// Grants issued by this domain and not yet retired.
    live: BTreeSet<GrantId>,
}

impl State {
    fn verify_grant(&self, domain: AuthorityDomainId, access: &Access<'_>) -> Result<()> {
        if access.grants.domain == domain && self.live.contains(&access.grant.id()) {
            Ok(())
        } else {
            Err(Error::InvalidGrant(access.grant.id()))
        }
    }

    fn verify(
        &self,
        domain: AuthorityDomainId,
        access: &Access<'_>,
        object: ObjectId,
        needed: Rights,
    ) -> Result<&Object> {
        self.verify_grant(domain, access)?;
        let obj = self.objects.get(&object).ok_or(Error::UnknownObject(object))?;
        if covers(access.grant.target(), object, &obj.names) && access.grant.rights().contains(needed) {
            Ok(obj)
        } else {
            Err(Error::Denied { object, needed })
        }
    }

    /// Authority to bind `path`: a namespace grant over it, or a name grant
    /// for exactly it.
    fn verify_bind(&self, domain: AuthorityDomainId, access: &Access<'_>, path: &Path) -> Result<()> {
        self.verify_grant(domain, access)?;
        let ok = access.grant.rights().contains(Rights::WRITE)
            && match access.grant.target() {
                Target::Namespace(prefix) => path.starts_with(prefix),
                Target::Name(name) => name == path,
                Target::Object(_) => false,
            };
        ok.then_some(()).ok_or(Error::PolicyDenied { path: path.clone(), rights: Rights::WRITE })
    }

    fn verify_namespace(
        &self,
        domain: AuthorityDomainId,
        access: &Access<'_>,
        path: &Path,
        needed: Rights,
    ) -> Result<()> {
        self.verify_grant(domain, access)?;
        match access.grant.target() {
            Target::Namespace(prefix) if path.starts_with(prefix) && access.grant.rights().contains(needed) => Ok(()),
            _ => Err(Error::PolicyDenied { path: path.clone(), rights: needed }),
        }
    }

    fn insert(&mut self, id: ObjectId, path: &Path, body: Body) {
        self.objects.insert(id, Object { body, names: BTreeSet::from([path.clone()]) });
        self.names.insert(path.clone(), id);
    }
}

/// The in-memory MeatFS namespace.
pub struct MeatFs {
    domain: AuthorityDomainId,
    state: RwLock<State>,
    journal: Mutex<Journal>,
}

const NO_NAMES: &BTreeSet<Path> = &BTreeSet::new();

impl MeatFs {
    /// Create a namespace and the one root grant set that governs it.
    ///
    /// There is no other way to obtain namespace authority: the host holds
    /// this set and issues narrower grants from it via [`MeatFs::issue`].
    pub fn genesis(seed: Seed) -> (MeatFs, GrantSet) {
        let domain = AuthorityDomainId::fresh();
        let mut ids = IdGen::new(seed);
        let host = ids.principal();
        let root = Grant::new(ids.grant(), host, Target::Namespace(Path::root()), Rights::ALL);
        let state = State {
            ids,
            objects: BTreeMap::new(),
            names: BTreeMap::new(),
            principals: BTreeMap::from([("host".to_owned(), host)]),
            live: BTreeSet::from([root.id()]),
        };
        let fs = MeatFs { domain, state: RwLock::new(state), journal: Mutex::default() };
        (fs, GrantSet { domain, principal: host, grants: vec![root] })
    }

    pub fn domain(&self) -> AuthorityDomainId {
        self.domain
    }

    fn record(&self, names: &BTreeSet<Path>, object: ObjectId, who: Who, kind: EventKind) {
        let mut journal = self.journal.lock().unwrap();
        journal.seq += 1;
        let event =
            Event { seq: journal.seq, object, principal: who.principal, grant: who.grant, cause: who.cause, kind };
        journal.subscribers.retain(|(target, tx)| !covers(target, object, names) || tx.send(event.clone()).is_ok());
        journal.events.push(event);
    }

    /// Resolve an authority request against host policy into grants.
    ///
    /// Observational and all-or-nothing: nothing in the namespace changes.
    /// Each want receives its own grant, index-aligned with the request. A
    /// bound name yields an object grant; an unbound one, when write is
    /// wanted and the name is bindable, yields a name grant.
    pub fn issue(&self, policy: &Policy, request: &AuthorityRequest) -> Result<GrantSet> {
        let mut st = self.state.write().unwrap();

        let mut targets = Vec::with_capacity(request.wants.len());
        let mut new_names: BTreeMap<Path, ()> = BTreeMap::new();
        for (path, rights) in &request.wants {
            if !policy.permits(path, *rights) {
                return Err(Error::PolicyDenied { path: path.clone(), rights: *rights });
            }
            targets.push(match st.names.get(path) {
                Some(&object) => Target::Object(object),
                None if rights.contains(Rights::WRITE) => {
                    if !new_names.contains_key(path) {
                        bindable(&st.names, path)?;
                        bindable(&new_names, path)?;
                        new_names.insert(path.clone(), ());
                    }
                    Target::Name(path.clone())
                }
                None => return Err(Error::Unbound(path.clone())),
            });
        }

        let principal = match st.principals.get(&request.principal) {
            Some(&id) => id,
            None => {
                let id = st.ids.principal();
                st.principals.insert(request.principal.clone(), id);
                id
            }
        };
        let grants = targets
            .into_iter()
            .zip(&request.wants)
            .map(|(target, (_, rights))| {
                let grant = Grant::new(st.ids.grant(), principal, target, *rights);
                st.live.insert(grant.id());
                grant
            })
            .collect();
        Ok(GrantSet { domain: self.domain, principal, grants })
    }

    /// Revoke every grant in the set.
    pub fn retire(&self, grants: GrantSet) {
        let mut st = self.state.write().unwrap();
        for grant in grants.iter() {
            st.live.remove(&grant.id());
        }
    }

    pub fn new_execution(&self) -> ExecutionId {
        self.state.write().unwrap().ids.execution()
    }

    /// Look up the object a name is bound to. Names confer no authority.
    pub fn resolve(&self, path: &Path) -> Option<ObjectId> {
        self.state.read().unwrap().names.get(path).copied()
    }

    /// A capability's declared metadata. Public contract, not authority.
    pub fn meta(&self, object: ObjectId) -> Option<CapabilityMeta> {
        match &self.state.read().unwrap().objects.get(&object)?.body {
            Body::Capability(c) => Some(c.meta()),
            Body::Data { .. } => None,
        }
    }

    fn bind_body(&self, access: &Access<'_>, path: &Path, body: Body) -> Result<ObjectId> {
        let mut st = self.state.write().unwrap();
        st.verify_bind(self.domain, access, path)?;
        bindable(&st.names, path)?;
        let object = st.ids.object();
        st.insert(object, path, body);
        self.record(&st.objects[&object].names, object, Who::of(access), EventKind::Bound(path.clone()));
        Ok(object)
    }

    /// Bind a new data object at `path`, immediately.
    pub fn bind(&self, access: &Access<'_>, path: &Path, value: Value) -> Result<ObjectId> {
        self.bind_body(access, path, Body::Data { value, version: 1 })
    }

    /// Bind a new capability object at `path`, immediately.
    pub fn mount(&self, access: &Access<'_>, path: &Path, capability: Arc<dyn Invoke>) -> Result<ObjectId> {
        self.bind_body(access, path, Body::Capability(capability))
    }

    pub fn read(&self, access: &Access<'_>, object: ObjectId) -> Result<Value> {
        let st = self.state.read().unwrap();
        let obj = st.verify(self.domain, access, object, Rights::READ)?;
        let Body::Data { value, version } = &obj.body else {
            return Err(Error::Unsupported { object, op: "read" });
        };
        self.record(&obj.names, object, Who::of(access), EventKind::Read { version: *version });
        Ok(value.clone())
    }

    /// Replace a data object's value, immediately.
    pub fn write(&self, access: &Access<'_>, object: ObjectId, value: Value) -> Result<u64> {
        let tx = self.transaction();
        tx.write(access, object, value)?;
        tx.commit()?;
        let st = self.state.read().unwrap();
        match st.objects[&object].body {
            Body::Data { version, .. } => Ok(version),
            Body::Capability(_) => unreachable!("write verified a data object"),
        }
    }

    /// Invoke outside any graph: the capability receives no attached grants.
    pub fn invoke(&self, access: &Access<'_>, object: ObjectId, input: Value) -> Result<Value> {
        let tx = self.transaction();
        match tx.invoke(access, object, input, NodeGrants::empty(access.grants)) {
            Ok(output) => tx.commit().map(|_| output),
            Err(e) => {
                tx.rollback();
                Err(e)
            }
        }
    }

    pub fn inspect(&self, access: &Access<'_>, object: ObjectId) -> Result<Inspection> {
        let st = self.state.read().unwrap();
        let obj = st.verify(self.domain, access, object, Rights::INSPECT)?;
        let names = obj.names.iter().cloned().collect();
        Ok(match &obj.body {
            Body::Data { version, .. } => Inspection {
                object,
                kind: NodeKind::Data,
                version: *version,
                names,
                meta: None,
                signature: Value::Null,
            },
            Body::Capability(c) => Inspection {
                object,
                kind: NodeKind::Capability,
                version: 0,
                names,
                meta: Some(c.meta()),
                signature: c.signature(),
            },
        })
    }

    /// Receive every future event within the presented grant's target.
    pub fn subscribe(&self, access: &Access<'_>) -> Result<Subscription> {
        self.state.read().unwrap().verify_grant(self.domain, access)?;
        let target = access.grant.target().clone();
        if !access.grant.rights().contains(Rights::SUBSCRIBE) {
            return Err(match target {
                Target::Object(object) => Error::Denied { object, needed: Rights::SUBSCRIBE },
                Target::Name(path) | Target::Namespace(path) => Error::PolicyDenied { path, rights: Rights::SUBSCRIBE },
            });
        }
        let (tx, rx) = mpsc::channel();
        self.journal.lock().unwrap().subscribers.push((target, tx));
        Ok(Subscription { rx })
    }

    /// Every bound name beneath `prefix`, in order.
    pub fn walk(&self, access: &Access<'_>, prefix: &Path) -> Result<Vec<(Path, ObjectId, NodeKind)>> {
        let st = self.state.read().unwrap();
        st.verify_namespace(self.domain, access, prefix, Rights::INSPECT)?;
        Ok(st
            .names
            .range(prefix..)
            .take_while(|(name, _)| name.starts_with(prefix))
            .map(|(name, &id)| {
                let kind = match st.objects[&id].body {
                    Body::Data { .. } => NodeKind::Data,
                    Body::Capability(_) => NodeKind::Capability,
                };
                (name.clone(), id, kind)
            })
            .collect())
    }

    /// Every bound name and data value beneath `prefix`: a comparable view of
    /// namespace state.
    pub fn snapshot(&self, access: &Access<'_>, prefix: &Path) -> Result<Vec<(Path, ObjectId, Option<Value>)>> {
        let st = self.state.read().unwrap();
        st.verify_namespace(self.domain, access, prefix, Rights::INSPECT | Rights::READ)?;
        Ok(st
            .names
            .range(prefix..)
            .take_while(|(name, _)| name.starts_with(prefix))
            .map(|(name, &id)| {
                let value = match &st.objects[&id].body {
                    Body::Data { value, .. } => Some(value.clone()),
                    Body::Capability(_) => None,
                };
                (name.clone(), id, value)
            })
            .collect())
    }

    /// The audit journal for every object bound beneath `prefix`.
    pub fn journal(&self, access: &Access<'_>, prefix: &Path) -> Result<Vec<Event>> {
        let st = self.state.read().unwrap();
        st.verify_namespace(self.domain, access, prefix, Rights::INSPECT)?;
        let target = Target::Namespace(prefix.clone());
        let journal = self.journal.lock().unwrap();
        Ok(journal
            .events
            .iter()
            .filter(|e| covers(&target, e.object, st.objects.get(&e.object).map_or(NO_NAMES, |o| &o.names)))
            .cloned()
            .collect())
    }

    /// Every event caused by the presented access's execution. A principal
    /// may always see what its own execution did.
    pub fn execution_events(&self, access: &Access<'_>) -> Result<Vec<Event>> {
        self.state.read().unwrap().verify_grant(self.domain, access)?;
        let Some(cause) = access.cause else {
            return Ok(Vec::new());
        };
        let journal = self.journal.lock().unwrap();
        Ok(journal.events.iter().filter(|e| e.cause.is_some_and(|c| c.execution == cause.execution)).cloned().collect())
    }

    pub fn transaction(&self) -> Transaction<'_> {
        Transaction { fs: self, staging: RefCell::default() }
    }
}

#[derive(Default)]
struct Staging {
    /// Committed version observed by each read.
    reads: BTreeMap<ObjectId, u64>,
    /// Committed version each existing object had when first staged.
    bases: BTreeMap<ObjectId, u64>,
    values: BTreeMap<ObjectId, Value>,
    /// Objects to be bound at commit, with their names.
    creates: BTreeMap<ObjectId, Path>,
    /// Staged objects in first-touch order, with who touched them last.
    order: Vec<ObjectId>,
    who: BTreeMap<ObjectId, Who>,
}

impl Staging {
    fn stage(&mut self, object: ObjectId, value: Value, who: Who) {
        if self.values.insert(object, value).is_none() {
            self.order.push(object);
        }
        self.who.insert(object, who);
    }

    fn created_at(&self, path: &Path) -> Option<ObjectId> {
        self.creates.iter().find(|(_, p)| *p == path).map(|(o, _)| *o)
    }
}

/// An optimistic, all-or-nothing unit of change.
///
/// Reads see the transaction's own staged writes. `commit` fails with
/// [`Error::Conflict`] if anything read or written changed underneath it,
/// or [`Error::InvalidGrant`] if a staged grant was retired, and otherwise
/// applies every staged change atomically. Staging and outcomes are always
/// journaled; only state is transactional.
pub struct Transaction<'a> {
    fs: &'a MeatFs,
    staging: RefCell<Staging>,
}

impl Transaction<'_> {
    pub fn read(&self, access: &Access<'_>, object: ObjectId) -> Result<Value> {
        let st = self.fs.state.read().unwrap();
        let obj = st.verify(self.fs.domain, access, object, Rights::READ)?;
        let Body::Data { value, version } = &obj.body else {
            return Err(Error::Unsupported { object, op: "read" });
        };
        let mut staging = self.staging.borrow_mut();
        let value = match staging.values.get(&object) {
            Some(staged) => staged.clone(),
            None => {
                staging.reads.entry(object).or_insert(*version);
                value.clone()
            }
        };
        self.fs.record(&obj.names, object, Who::of(access), EventKind::Read { version: *version });
        Ok(value)
    }

    /// Stage a replacement value for an existing data object.
    pub fn write(&self, access: &Access<'_>, object: ObjectId, value: Value) -> Result<()> {
        let st = self.fs.state.read().unwrap();
        let obj = st.verify(self.fs.domain, access, object, Rights::WRITE)?;
        let Body::Data { version, .. } = obj.body else {
            return Err(Error::Unsupported { object, op: "write" });
        };
        let mut staging = self.staging.borrow_mut();
        staging.bases.entry(object).or_insert(version);
        staging.stage(object, value, Who::of(access));
        self.fs.record(&obj.names, object, Who::of(access), EventKind::Staged);
        Ok(())
    }

    /// Stage a new data object at an unbound name. Staging the same name
    /// again replaces the staged value. The object's identity is fixed now;
    /// it becomes visible only on commit.
    pub fn create(&self, access: &Access<'_>, path: &Path, value: Value) -> Result<ObjectId> {
        let mut st = self.fs.state.write().unwrap();
        st.verify_bind(self.fs.domain, access, path)?;
        let mut staging = self.staging.borrow_mut();
        let object = match staging.created_at(path) {
            Some(object) => object,
            None => {
                bindable(&st.names, path)?;
                let staged: BTreeMap<Path, ()> = staging.creates.values().map(|p| (p.clone(), ())).collect();
                bindable(&staged, path)?;
                let object = st.ids.object();
                staging.creates.insert(object, path.clone());
                object
            }
        };
        staging.stage(object, value, Who::of(access));
        self.fs.record(NO_NAMES, object, Who::of(access), EventKind::Staged);
        Ok(object)
    }

    /// Invoke a capability. An effectful capability acts inside this
    /// transaction with exactly the `attached` grants.
    pub fn invoke(
        &self,
        access: &Access<'_>,
        object: ObjectId,
        input: Value,
        attached: NodeGrants<'_>,
    ) -> Result<Value> {
        let capability = {
            let st = self.fs.state.read().unwrap();
            match &st.verify(self.fs.domain, access, object, Rights::INVOKE)?.body {
                Body::Capability(c) => Arc::clone(c),
                Body::Data { .. } => return Err(Error::Unsupported { object, op: "invoke" }),
            }
        };
        let effects = match capability.meta().purity {
            Purity::Pure => None,
            Purity::Effectful => Some(Effects { tx: self, grants: attached, cause: access.cause }),
        };
        let result = capability.invoke(&CallContext { object, effects }, input);
        let st = self.fs.state.read().unwrap();
        self.fs.record(&st.objects[&object].names, object, Who::of(access), EventKind::Invoked { ok: result.is_ok() });
        result.map_err(|fault| match fault {
            Fault::InvalidInput(message) => Error::InvalidInput { object, message },
            Fault::Failed(message) => Error::Capability { object, message },
            Fault::Substrate(e) => e,
        })
    }

    fn check(&self, st: &State) -> Result<()> {
        let staging = self.staging.borrow();
        let unchanged = |object: &ObjectId, seen: &u64| match st.objects.get(object).map(|o| &o.body) {
            Some(Body::Data { version, .. }) => version == seen,
            _ => false,
        };
        for (object, seen) in staging.reads.iter().chain(&staging.bases) {
            if !unchanged(object, seen) {
                return Err(Error::Conflict(*object));
            }
        }
        for (object, path) in &staging.creates {
            bindable(&st.names, path).map_err(|_| Error::Conflict(*object))?;
        }
        for who in staging.who.values() {
            let grant = who.grant.expect("staged changes carry a grant");
            if !st.live.contains(&grant) {
                return Err(Error::InvalidGrant(grant));
            }
        }
        Ok(())
    }

    /// Apply every staged change atomically, or none of them.
    pub fn commit(self) -> Result<()> {
        let mut st = self.fs.state.write().unwrap();
        if let Err(e) = self.check(&st) {
            drop(st);
            self.rollback();
            return Err(e);
        }
        let mut staging = self.staging.into_inner();
        for object in std::mem::take(&mut staging.order) {
            let value = staging.values.remove(&object).expect("ordered objects are staged");
            let who = staging.who[&object];
            let version = match staging.creates.get(&object) {
                Some(path) => {
                    st.insert(object, path, Body::Data { value, version: 1 });
                    self.fs.record(&st.objects[&object].names, object, who, EventKind::Bound(path.clone()));
                    1
                }
                None => {
                    let Body::Data { value: slot, version } = &mut st.objects.get_mut(&object).expect("checked").body
                    else {
                        unreachable!("only data objects are staged")
                    };
                    *slot = value;
                    *version += 1;
                    *version
                }
            };
            self.fs.record(&st.objects[&object].names, object, who, EventKind::Written { version });
        }
        Ok(())
    }

    /// Discard every staged change, journaling that it was discarded.
    pub fn rollback(self) {
        let staging = self.staging.into_inner();
        let st = self.fs.state.read().unwrap();
        for object in staging.order {
            let names = st.objects.get(&object).map_or(NO_NAMES, |o| &o.names);
            self.fs.record(names, object, staging.who[&object], EventKind::RolledBack);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    struct Double;
    impl Invoke for Double {
        fn meta(&self) -> CapabilityMeta {
            CapabilityMeta::new(Purity::Pure, Determinism::Deterministic)
        }
        fn invoke(&self, _: &CallContext<'_>, input: Value) -> std::result::Result<Value, Fault> {
            match input {
                Value::Int(i) => Ok(Value::Int(i * 2)),
                other => Err(Fault::InvalidInput(format!("expected int, got {}", other.kind()))),
            }
        }
    }

    /// Declared pure, but tries to touch MeatFS anyway.
    struct Sneaky;
    impl Invoke for Sneaky {
        fn meta(&self) -> CapabilityMeta {
            CapabilityMeta::new(Purity::Pure, Determinism::Deterministic)
        }
        fn invoke(&self, cx: &CallContext<'_>, _: Value) -> std::result::Result<Value, Fault> {
            cx.effects()?;
            Ok(Value::Null)
        }
    }

    /// Effectful: writes its input to every attached path.
    struct Fanout;
    impl Invoke for Fanout {
        fn meta(&self) -> CapabilityMeta {
            CapabilityMeta::new(Purity::Effectful, Determinism::Deterministic)
        }
        fn invoke(&self, cx: &CallContext<'_>, input: Value) -> std::result::Result<Value, Fault> {
            let fx = cx.effects()?;
            for entry in fx.grants() {
                fx.write(&entry.path, input.clone())?;
            }
            match input.as_text() {
                Some(path) => fx.write(&Path::parse(path)?, input.clone()).map(|_| Value::Null).map_err(Fault::from),
                None => Ok(Value::Null),
            }
        }
    }

    fn root(host: &GrantSet) -> Access<'_> {
        host.access(host.iter().next().unwrap().id()).unwrap()
    }

    fn setup() -> (MeatFs, GrantSet) {
        let (fs, host) = MeatFs::genesis(Seed::fixed(1));
        let r = root(&host);
        fs.mount(&r, &p("/tools/double"), Arc::new(Double)).unwrap();
        fs.mount(&r, &p("/tools/sneaky"), Arc::new(Sneaky)).unwrap();
        fs.mount(&r, &p("/tools/fanout"), Arc::new(Fanout)).unwrap();
        fs.bind(&r, &p("/memory/secret"), "s".into()).unwrap();
        (fs, host)
    }

    fn policy() -> Policy {
        Policy::default()
            .allow(p("/tools"), Rights::INVOKE)
            .allow(p("/state"), Rights::READ | Rights::WRITE)
            .allow(p("/memory"), Rights::READ)
    }

    fn access(grants: &GrantSet, i: usize) -> Access<'_> {
        grants.access(grants.issued(i).unwrap().id()).unwrap()
    }

    #[test]
    fn identity_is_not_the_name() {
        let (fs, host) = setup();
        let r = root(&host);
        let id = fs.resolve(&p("/tools/double")).unwrap();
        assert_eq!(fs.inspect(&r, id).unwrap().names, vec![p("/tools/double")]);
        assert_eq!(fs.resolve(&p("/tools/nope")), None);
        for clash in ["/tools/double", "/tools/double/x", "/tools"] {
            assert_eq!(fs.bind(&r, &p(clash), Value::Null), Err(Error::AlreadyBound(p(clash))));
        }
    }

    #[test]
    fn issue_is_observational_and_per_use() {
        let (fs, host) = setup();
        let before = fs.snapshot(&root(&host), &Path::root()).unwrap();
        let request = AuthorityRequest::new("agent")
            .want(p("/tools/double"), Rights::INVOKE)
            .want(p("/state/agent/out"), Rights::WRITE);
        let grants = fs.issue(&policy(), &request).unwrap();
        assert_eq!(fs.snapshot(&root(&host), &Path::root()).unwrap(), before, "issuing changes nothing");

        let double = fs.resolve(&p("/tools/double")).unwrap();
        assert_eq!(grants.issued(0).unwrap().target(), &Target::Object(double));
        assert_eq!(grants.issued(1).unwrap().target(), &Target::Name(p("/state/agent/out")));
        assert_eq!(fs.invoke(&access(&grants, 0), double, Value::Int(21)).unwrap(), Value::Int(42));

        // The invoke grant is no key to anything else.
        let secret = fs.resolve(&p("/memory/secret")).unwrap();
        assert!(matches!(fs.read(&access(&grants, 0), secret), Err(Error::Denied { .. })));
    }

    #[test]
    fn issue_is_all_or_nothing_under_policy() {
        let (fs, _) = setup();
        let escalate = AuthorityRequest::new("agent")
            .want(p("/state/fresh"), Rights::WRITE)
            .want(p("/memory/secret"), Rights::WRITE);
        let e = fs.issue(&policy(), &escalate).unwrap_err();
        assert_eq!(e, Error::PolicyDenied { path: p("/memory/secret"), rights: Rights::WRITE });

        let missing = AuthorityRequest::new("agent").want(p("/tools/nope"), Rights::INVOKE);
        assert_eq!(fs.issue(&policy(), &missing).unwrap_err(), Error::Unbound(p("/tools/nope")));
        let nested = AuthorityRequest::new("agent").want(p("/tools/double/x"), Rights::WRITE);
        assert!(fs.issue(&Policy::default().allow(Path::root(), Rights::WRITE), &nested).is_err());
    }

    #[test]
    fn grants_belong_to_their_live_domain() {
        let (fs, _) = setup();
        let request = AuthorityRequest::new("agent").want(p("/tools/double"), Rights::INVOKE);
        // An identically seeded twin issues byte-identical grant ids; they must
        // still be worthless here, even once `fs` has issued the same ids.
        let (twin, _) = setup();
        let foreign = twin.issue(&policy(), &request).unwrap();
        let ours = fs.issue(&policy(), &request).unwrap();
        assert_eq!(foreign.issued(0).unwrap().id(), ours.issued(0).unwrap().id());
        assert_ne!(foreign.domain(), ours.domain());
        let double = fs.resolve(&p("/tools/double")).unwrap();
        assert!(matches!(fs.invoke(&access(&foreign, 0), double, Value::Int(1)), Err(Error::InvalidGrant(_))));

        let id = ours.issued(0).unwrap().id();
        assert!(fs.invoke(&access(&ours, 0), double, Value::Int(1)).is_ok());
        fs.retire(ours);
        assert!(!fs.state.read().unwrap().live.contains(&id));
    }

    #[test]
    fn pure_capabilities_get_no_effects() {
        let (fs, host) = setup();
        let sneaky = fs.resolve(&p("/tools/sneaky")).unwrap();
        assert_eq!(fs.invoke(&root(&host), sneaky, Value::Null), Err(Error::Impure(sneaky)));
    }

    #[test]
    fn effects_are_limited_to_attached_grants() {
        let (fs, host) = setup();
        let request = AuthorityRequest::new("agent")
            .want(p("/tools/fanout"), Rights::INVOKE)
            .want(p("/state/mine"), Rights::WRITE)
            .want(p("/state/other"), Rights::WRITE);
        let grants = fs.issue(&policy(), &request).unwrap();
        let fanout = fs.resolve(&p("/tools/fanout")).unwrap();
        let attached = [NodeGrant { path: p("/state/mine"), grant: grants.issued(1).unwrap().id() }];
        let node = NodeGrants::new(&grants, &attached).unwrap();

        // Writing what is attached works, inside the caller's transaction.
        let tx = fs.transaction();
        tx.invoke(&access(&grants, 0), fanout, Value::Int(1), node).unwrap();
        assert_eq!(fs.resolve(&p("/state/mine")), None, "staged, not yet visible");
        tx.commit().unwrap();
        let mine = fs.resolve(&p("/state/mine")).unwrap();
        assert_eq!(fs.read(&root(&host), mine).unwrap(), Value::Int(1));

        // The set also holds a grant for /state/other, but it is not attached.
        let tx = fs.transaction();
        let none = NodeGrants::empty(&grants);
        let e = tx.invoke(&access(&grants, 0), fanout, "/state/other".into(), none).unwrap_err();
        assert_eq!(e, Error::NotAttached(p("/state/other")));
        tx.rollback();
        assert_eq!(fs.resolve(&p("/state/other")), None);
    }

    #[test]
    fn events_carry_provenance() {
        let (fs, host) = setup();
        let request = AuthorityRequest::new("agent").want(p("/tools/double"), Rights::INVOKE);
        let grants = fs.issue(&policy(), &request).unwrap();
        let double = fs.resolve(&p("/tools/double")).unwrap();
        let cause = Cause { execution: fs.new_execution(), node: NodeId(3) };
        let a = access(&grants, 0).caused_by(cause);
        fs.invoke(&a, double, Value::Int(1)).unwrap();

        let events = fs.execution_events(&a).unwrap();
        assert_eq!(events.len(), 1);
        let e = &events[0];
        assert_eq!(
            (e.object, e.principal, e.grant, e.cause),
            (double, grants.principal(), Some(a.grant().id()), Some(cause))
        );
        assert_eq!(fs.journal(&root(&host), &p("/tools")).unwrap().len(), 4, "three mounts and one invoke");
    }

    #[test]
    fn subscriptions_follow_the_grant_target() {
        let (fs, host) = setup();
        let r = root(&host);
        let sub = fs.subscribe(&r).unwrap();
        let double = fs.resolve(&p("/tools/double")).unwrap();
        fs.invoke(&r, double, Value::Int(1)).unwrap();
        assert_eq!(sub.drain().len(), 1);
    }

    #[test]
    fn transactions_are_atomic_and_journaled_either_way() {
        let (fs, host) = setup();
        let r = root(&host);
        let n = fs.bind(&r, &p("/state/n"), Value::Int(1)).unwrap();
        let m = fs.bind(&r, &p("/state/m"), Value::Int(0)).unwrap();

        let tx = fs.transaction();
        tx.read(&r, n).unwrap();
        tx.write(&r, m, Value::Int(2)).unwrap();
        tx.create(&r, &p("/state/new"), Value::Int(3)).unwrap();
        assert_eq!(tx.read(&r, m).unwrap(), Value::Int(2), "reads see staged writes");
        fs.write(&r, n, Value::Int(9)).unwrap();
        assert_eq!(tx.commit(), Err(Error::Conflict(n)));
        assert_eq!(fs.read(&r, m).unwrap(), Value::Int(0));
        assert_eq!(fs.resolve(&p("/state/new")), None);
        let kinds: Vec<_> = fs.journal(&r, &p("/state/m")).unwrap().into_iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds[1..],
            [EventKind::Staged, EventKind::Read { version: 1 }, EventKind::RolledBack, EventKind::Read { version: 1 }]
        );

        let tx = fs.transaction();
        tx.write(&r, n, Value::Int(10)).unwrap();
        let new = tx.create(&r, &p("/state/new"), Value::Int(20)).unwrap();
        tx.commit().unwrap();
        assert_eq!(fs.resolve(&p("/state/new")), Some(new));
        assert_eq!(fs.read(&r, new).unwrap(), Value::Int(20));
    }

    #[test]
    fn identical_seeds_give_identical_identities() {
        let (a, _) = setup();
        let (b, _) = setup();
        assert_eq!(a.resolve(&p("/tools/double")), b.resolve(&p("/tools/double")));
    }
}

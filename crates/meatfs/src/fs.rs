use crate::authority::{Access, AuthorityRequest, Cause, Grant, GrantSet, Policy, Target};
use crate::id::IdGen;
use crate::{Error, ExecutionId, GrantId, ObjectId, Path, PrincipalId, Result, Rights, Seed, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};

/// Whether a capability may touch anything beyond its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purity {
    /// `input → output` only: no MeatFS access, no clocks, no randomness, no
    /// external state. Safe to cache, replay, parallelize and compile.
    Pure,
    /// May act on MeatFS through the caller's grants.
    Effectful,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityMeta {
    pub purity: Purity,
}

/// An invocable object: the raw, untyped form of a capability.
///
/// The namespace never holds its lock while invoking, so an effectful
/// implementation may call back into MeatFS through [`CallContext::effects`].
pub trait Invoke: Send + Sync {
    fn meta(&self) -> CapabilityMeta;

    /// A description of the object's contract, surfaced by `inspect`.
    fn signature(&self) -> Value {
        Value::Null
    }

    fn invoke(&self, cx: &CallContext<'_>, input: Value) -> std::result::Result<Value, String>;
}

/// What an invoked object is given. A pure capability receives no effects
/// at all, so it cannot reach MeatFS even by accident.
pub struct CallContext<'a> {
    pub object: ObjectId,
    effects: Option<Effects<'a>>,
}

impl<'a> CallContext<'a> {
    pub fn effects(&self) -> std::result::Result<&Effects<'a>, String> {
        self.effects.as_ref().ok_or_else(|| "pure capability attempted an effect".to_owned())
    }
}

/// The caller's namespace and grants, lent to an effectful capability.
pub struct Effects<'a> {
    pub fs: &'a MeatFs,
    pub grants: &'a GrantSet,
    pub cause: Option<Cause>,
}

impl Effects<'_> {
    pub fn access(&self, id: GrantId) -> Option<Access<'_>> {
        let access = self.grants.access(id)?;
        Some(match self.cause {
            Some(cause) => access.caused_by(cause),
            None => access,
        })
    }
}

enum Body {
    /// `value` is `None` until first written (version 0).
    Data {
        value: Option<Value>,
        version: u64,
    },
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
    /// Data version; 0 for capabilities and unwritten data.
    pub version: u64,
    pub names: Vec<Path>,
    pub meta: Option<CapabilityMeta>,
    pub signature: Value,
}

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
    Read { version: u64 },
    Written { version: u64 },
    Invoked { ok: bool },
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
    /// Grants issued by this namespace and not yet retired.
    live: BTreeSet<GrantId>,
}

impl State {
    fn verify_grant(&self, issuer: u128, access: &Access<'_>) -> Result<()> {
        if access.grants.issuer == issuer && self.live.contains(&access.grant.id()) {
            Ok(())
        } else {
            Err(Error::InvalidGrant(access.grant.id()))
        }
    }

    fn verify(&self, issuer: u128, access: &Access<'_>, object: ObjectId, needed: Rights) -> Result<&Object> {
        self.verify_grant(issuer, access)?;
        let obj = self.objects.get(&object).ok_or(Error::UnknownObject(object))?;
        if covers(access.grant.target(), object, &obj.names) && access.grant.rights().contains(needed) {
            Ok(obj)
        } else {
            Err(Error::Denied { object, needed })
        }
    }

    /// Authority over names (binding, walking), which only a namespace grant carries.
    fn verify_namespace(&self, issuer: u128, access: &Access<'_>, path: &Path, needed: Rights) -> Result<()> {
        self.verify_grant(issuer, access)?;
        match access.grant.target() {
            Target::Namespace(prefix) if path.starts_with(prefix) && access.grant.rights().contains(needed) => Ok(()),
            _ => Err(Error::PolicyDenied { path: path.clone(), rights: needed }),
        }
    }

    fn bind(&mut self, path: &Path, body: Body) -> ObjectId {
        let id = self.ids.object();
        self.objects.insert(id, Object { body, names: BTreeSet::from([path.clone()]) });
        self.names.insert(path.clone(), id);
        id
    }
}

/// The in-memory MeatFS namespace.
pub struct MeatFs {
    /// Identity of this namespace as a grant issuer.
    issuer: u128,
    state: RwLock<State>,
    journal: Mutex<Journal>,
}

impl MeatFs {
    /// Create a namespace and the one root grant set that governs it.
    ///
    /// There is no other way to obtain namespace authority: the host holds
    /// this set and issues narrower grants from it via [`MeatFs::issue`].
    pub fn genesis(seed: Seed) -> (MeatFs, GrantSet) {
        // The issuer token is unique per namespace instance, independent of the
        // seed, so two identically seeded namespaces never honour each
        // other's grants. It never appears in events or receipts.
        static NEXT_ISSUER: AtomicU64 = AtomicU64::new(1);
        let issuer = u128::from(NEXT_ISSUER.fetch_add(1, Ordering::Relaxed));
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
        let fs = MeatFs { issuer, state: RwLock::new(state), journal: Mutex::default() };
        (fs, GrantSet { issuer, principal: host, grants: vec![root] })
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
    /// All-or-nothing: every wanted path must be permitted by `policy` and
    /// either bound or — when write is wanted — bindable, in which case an
    /// empty data object is provisioned. Grants are issued per object, so a
    /// principal receives authority over exactly what it will touch.
    pub fn issue(&self, policy: &Policy, request: &AuthorityRequest) -> Result<GrantSet> {
        let mut st = self.state.write().unwrap();

        // Validate everything before changing anything.
        let mut planned: BTreeMap<Path, ()> = st.names.keys().map(|n| (n.clone(), ())).collect();
        let mut provision = Vec::new();
        for (path, rights) in &request.wants {
            if !policy.permits(path, *rights) {
                return Err(Error::PolicyDenied { path: path.clone(), rights: *rights });
            }
            if !planned.contains_key(path) {
                if !rights.contains(Rights::WRITE) {
                    return Err(Error::Unbound(path.clone()));
                }
                bindable(&planned, path)?;
                planned.insert(path.clone(), ());
                provision.push(path);
            }
        }

        let principal = match st.principals.get(&request.principal) {
            Some(&id) => id,
            None => {
                let id = st.ids.principal();
                st.principals.insert(request.principal.clone(), id);
                id
            }
        };
        for path in provision {
            let object = st.bind(path, Body::Data { value: None, version: 0 });
            let who = Who { principal, grant: None, cause: None };
            self.record(&st.objects[&object].names, object, who, EventKind::Bound(path.clone()));
        }

        let mut rights_of: BTreeMap<ObjectId, Rights> = BTreeMap::new();
        let mut order = Vec::new();
        for (path, rights) in &request.wants {
            let object = st.names[path];
            let held = rights_of.entry(object).or_insert_with(|| {
                order.push(object);
                Rights::NONE
            });
            *held = *held | *rights;
        }
        let grants = order
            .into_iter()
            .map(|object| {
                let grant = Grant::new(st.ids.grant(), principal, Target::Object(object), rights_of[&object]);
                st.live.insert(grant.id());
                grant
            })
            .collect();
        Ok(GrantSet { issuer: self.issuer, principal, grants })
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

    fn bind_body(&self, access: &Access<'_>, path: &Path, body: Body) -> Result<ObjectId> {
        let mut st = self.state.write().unwrap();
        st.verify_namespace(self.issuer, access, path, Rights::WRITE)?;
        bindable(&st.names, path)?;
        let object = st.bind(path, body);
        self.record(&st.objects[&object].names, object, Who::of(access), EventKind::Bound(path.clone()));
        Ok(object)
    }

    /// Bind a new data object at `path`.
    pub fn bind(&self, access: &Access<'_>, path: &Path, value: Value) -> Result<ObjectId> {
        let object = self.bind_body(access, path, Body::Data { value: None, version: 0 })?;
        self.write(access, object, value)?;
        Ok(object)
    }

    /// Bind a new capability object at `path`.
    pub fn mount(&self, access: &Access<'_>, path: &Path, capability: Arc<dyn Invoke>) -> Result<ObjectId> {
        self.bind_body(access, path, Body::Capability(capability))
    }

    pub fn read(&self, access: &Access<'_>, object: ObjectId) -> Result<Value> {
        let st = self.state.read().unwrap();
        let obj = st.verify(self.issuer, access, object, Rights::READ)?;
        let (value, version) = match &obj.body {
            Body::Data { value: Some(value), version } => (value.clone(), *version),
            Body::Data { value: None, .. } => return Err(Error::Empty(object)),
            Body::Capability(_) => return Err(Error::Unsupported { object, op: "read" }),
        };
        self.record(&obj.names, object, Who::of(access), EventKind::Read { version });
        Ok(value)
    }

    pub fn write(&self, access: &Access<'_>, object: ObjectId, value: Value) -> Result<u64> {
        let mut st = self.state.write().unwrap();
        st.verify(self.issuer, access, object, Rights::WRITE)?;
        let obj = st.objects.get_mut(&object).expect("verified");
        let Body::Data { value: slot, version } = &mut obj.body else {
            return Err(Error::Unsupported { object, op: "write" });
        };
        *slot = Some(value);
        *version += 1;
        let version = *version;
        self.record(&obj.names, object, Who::of(access), EventKind::Written { version });
        Ok(version)
    }

    pub fn invoke(&self, access: &Access<'_>, object: ObjectId, input: Value) -> Result<Value> {
        let capability = {
            let st = self.state.read().unwrap();
            match &st.verify(self.issuer, access, object, Rights::INVOKE)?.body {
                Body::Capability(c) => Arc::clone(c),
                Body::Data { .. } => return Err(Error::Unsupported { object, op: "invoke" }),
            }
        };
        let effects = match capability.meta().purity {
            Purity::Pure => None,
            Purity::Effectful => Some(Effects { fs: self, grants: access.grants, cause: access.cause }),
        };
        let result = capability.invoke(&CallContext { object, effects }, input);
        let st = self.state.read().unwrap();
        self.record(&st.objects[&object].names, object, Who::of(access), EventKind::Invoked { ok: result.is_ok() });
        result.map_err(|message| Error::Capability { object, message })
    }

    pub fn inspect(&self, access: &Access<'_>, object: ObjectId) -> Result<Inspection> {
        let st = self.state.read().unwrap();
        let obj = st.verify(self.issuer, access, object, Rights::INSPECT)?;
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
        self.state.read().unwrap().verify_grant(self.issuer, access)?;
        let target = access.grant.target().clone();
        if !access.grant.rights().contains(Rights::SUBSCRIBE) {
            return Err(match target {
                Target::Object(object) => Error::Denied { object, needed: Rights::SUBSCRIBE },
                Target::Namespace(path) => Error::PolicyDenied { path, rights: Rights::SUBSCRIBE },
            });
        }
        let (tx, rx) = mpsc::channel();
        self.journal.lock().unwrap().subscribers.push((target, tx));
        Ok(Subscription { rx })
    }

    /// Every bound name beneath `prefix`, in order.
    pub fn walk(&self, access: &Access<'_>, prefix: &Path) -> Result<Vec<(Path, ObjectId, NodeKind)>> {
        let st = self.state.read().unwrap();
        st.verify_namespace(self.issuer, access, prefix, Rights::INSPECT)?;
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

    /// The audit journal for every object bound beneath `prefix`.
    pub fn journal(&self, access: &Access<'_>, prefix: &Path) -> Result<Vec<Event>> {
        let st = self.state.read().unwrap();
        st.verify_namespace(self.issuer, access, prefix, Rights::INSPECT)?;
        let target = Target::Namespace(prefix.clone());
        let journal = self.journal.lock().unwrap();
        Ok(journal.events.iter().filter(|e| covers(&target, e.object, &st.objects[&e.object].names)).cloned().collect())
    }

    /// Every event caused by the presented access's execution. A principal
    /// may always see what its own execution did.
    pub fn execution_events(&self, access: &Access<'_>) -> Result<Vec<Event>> {
        self.state.read().unwrap().verify_grant(self.issuer, access)?;
        let Some(cause) = access.cause else {
            return Ok(Vec::new());
        };
        let journal = self.journal.lock().unwrap();
        Ok(journal.events.iter().filter(|e| e.cause.is_some_and(|c| c.execution == cause.execution)).cloned().collect())
    }

    pub fn transaction(&self) -> Transaction<'_> {
        Transaction { fs: self, reads: Vec::new(), writes: Vec::new() }
    }
}

/// Optimistic, all-or-nothing batch of writes.
///
/// Reads record the version seen; `commit` fails with [`Error::Conflict`] if
/// any of them changed, or [`Error::InvalidGrant`] if a staged grant was
/// retired meanwhile, and otherwise applies every staged write atomically.
pub struct Transaction<'a> {
    fs: &'a MeatFs,
    reads: Vec<(ObjectId, u64)>,
    writes: Vec<(ObjectId, Value, Who)>,
}

impl Transaction<'_> {
    pub fn read(&mut self, access: &Access<'_>, object: ObjectId) -> Result<Value> {
        if let Some((_, value, _)) = self.writes.iter().rev().find(|(o, _, _)| *o == object) {
            return Ok(value.clone());
        }
        let st = self.fs.state.read().unwrap();
        match &st.verify(self.fs.issuer, access, object, Rights::READ)?.body {
            Body::Data { value: Some(value), version } => {
                self.reads.push((object, *version));
                Ok(value.clone())
            }
            Body::Data { value: None, .. } => Err(Error::Empty(object)),
            Body::Capability(_) => Err(Error::Unsupported { object, op: "read" }),
        }
    }

    pub fn write(&mut self, access: &Access<'_>, object: ObjectId, value: Value) -> Result<()> {
        let st = self.fs.state.read().unwrap();
        if let Body::Capability(_) = st.verify(self.fs.issuer, access, object, Rights::WRITE)?.body {
            return Err(Error::Unsupported { object, op: "write" });
        }
        self.writes.push((object, value, Who::of(access)));
        Ok(())
    }

    pub fn commit(self) -> Result<()> {
        let mut st = self.fs.state.write().unwrap();
        for &(object, seen) in &self.reads {
            match st.objects[&object].body {
                Body::Data { version, .. } if version == seen => {}
                _ => return Err(Error::Conflict(object)),
            }
        }
        for (_, _, who) in &self.writes {
            let grant = who.grant.expect("staged writes carry a grant");
            if !st.live.contains(&grant) {
                return Err(Error::InvalidGrant(grant));
            }
        }
        for (object, value, who) in self.writes {
            let obj = st.objects.get_mut(&object).expect("verified at staging");
            let Body::Data { value: slot, version } = &mut obj.body else { unreachable!("verified at staging") };
            *slot = Some(value);
            *version += 1;
            let version = *version;
            self.fs.record(&obj.names, object, who, EventKind::Written { version });
        }
        Ok(())
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
            CapabilityMeta { purity: Purity::Pure }
        }
        fn invoke(&self, _: &CallContext<'_>, input: Value) -> std::result::Result<Value, String> {
            match input {
                Value::Int(i) => Ok(Value::Int(i * 2)),
                other => Err(format!("expected int, got {}", other.kind())),
            }
        }
    }

    /// Declared pure, but tries to touch MeatFS anyway.
    struct Sneaky;
    impl Invoke for Sneaky {
        fn meta(&self) -> CapabilityMeta {
            CapabilityMeta { purity: Purity::Pure }
        }
        fn invoke(&self, cx: &CallContext<'_>, _: Value) -> std::result::Result<Value, String> {
            cx.effects()?;
            Ok(Value::Null)
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
        fs.bind(&r, &p("/memory/secret"), "s".into()).unwrap();
        (fs, host)
    }

    fn policy() -> Policy {
        Policy::default()
            .allow(p("/tools"), Rights::INVOKE)
            .allow(p("/state"), Rights::READ | Rights::WRITE)
            .allow(p("/memory"), Rights::READ)
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
    fn issue_resolves_policy_into_object_grants() {
        let (fs, _) = setup();
        let request = AuthorityRequest::new("agent")
            .want(p("/tools/double"), Rights::INVOKE)
            .want(p("/state/agent/out"), Rights::WRITE);
        let grants = fs.issue(&policy(), &request).unwrap();
        let double = fs.resolve(&p("/tools/double")).unwrap();
        let out = fs.resolve(&p("/state/agent/out")).expect("provisioned");

        let invoke = grants.for_object(double).unwrap();
        assert_eq!(invoke.rights(), Rights::INVOKE);
        let access = grants.access(invoke.id()).unwrap();
        assert_eq!(fs.invoke(&access, double, Value::Int(21)).unwrap(), Value::Int(42));

        // The invoke grant names one object; it is no key to anything else.
        assert_eq!(fs.write(&access, out, Value::Null), Err(Error::Denied { object: out, needed: Rights::WRITE }));
        let secret = fs.resolve(&p("/memory/secret")).unwrap();
        assert!(matches!(fs.read(&access, secret), Err(Error::Denied { .. })));

        let write = grants.access(grants.for_object(out).unwrap().id()).unwrap();
        assert_eq!(fs.write(&write, out, "x".into()).unwrap(), 1);
        assert_eq!(fs.read(&write, out), Err(Error::Denied { object: out, needed: Rights::READ }));
    }

    #[test]
    fn issue_is_all_or_nothing_under_policy() {
        let (fs, _) = setup();
        let escalate = AuthorityRequest::new("agent")
            .want(p("/state/fresh"), Rights::WRITE)
            .want(p("/memory/secret"), Rights::WRITE);
        let e = fs.issue(&policy(), &escalate).unwrap_err();
        assert_eq!(e, Error::PolicyDenied { path: p("/memory/secret"), rights: Rights::WRITE });
        assert_eq!(fs.resolve(&p("/state/fresh")), None, "nothing provisioned on failure");

        let missing = AuthorityRequest::new("agent").want(p("/tools/nope"), Rights::INVOKE);
        assert_eq!(fs.issue(&policy(), &missing).unwrap_err(), Error::Unbound(p("/tools/nope")));
    }

    #[test]
    fn grants_are_bound_to_their_issuer_and_lifetime() {
        let (fs, _) = setup();
        let (other, _) = MeatFs::genesis(Seed::fixed(2));
        let r2 = other.issue(&Policy::default(), &AuthorityRequest::new("x")).unwrap();
        assert_eq!(r2.iter().count(), 0);

        let request = AuthorityRequest::new("agent").want(p("/tools/double"), Rights::INVOKE);
        // An identically seeded twin issues byte-identical grant ids; they must
        // still be worthless here, even once `fs` has issued the same ids.
        let (twin, _) = setup();
        let foreign = twin.issue(&policy(), &request).unwrap();
        let _same_ids = fs.issue(&policy(), &request).unwrap();
        let double = fs.resolve(&p("/tools/double")).unwrap();
        let access = foreign.access(foreign.iter().next().unwrap().id()).unwrap();
        assert!(matches!(fs.invoke(&access, double, Value::Int(1)), Err(Error::InvalidGrant(_))));

        let grants = fs.issue(&policy(), &request).unwrap();
        let id = grants.iter().next().unwrap().id();
        assert!(fs.invoke(&grants.access(id).unwrap(), double, Value::Int(1)).is_ok());
        fs.retire(grants);
        assert!(!fs.state.read().unwrap().live.contains(&id));
    }

    #[test]
    fn pure_capabilities_get_no_effects() {
        let (fs, host) = setup();
        let sneaky = fs.resolve(&p("/tools/sneaky")).unwrap();
        let e = fs.invoke(&root(&host), sneaky, Value::Null).unwrap_err();
        assert_eq!(e, Error::Capability { object: sneaky, message: "pure capability attempted an effect".into() });
    }

    #[test]
    fn events_carry_provenance() {
        let (fs, host) = setup();
        let request = AuthorityRequest::new("agent").want(p("/tools/double"), Rights::INVOKE);
        let grants = fs.issue(&policy(), &request).unwrap();
        let double = fs.resolve(&p("/tools/double")).unwrap();
        let cause = Cause { execution: fs.new_execution(), node: NodeId(3) };
        let grant = grants.iter().next().unwrap();
        let access = grants.access(grant.id()).unwrap().caused_by(cause);
        fs.invoke(&access, double, Value::Int(1)).unwrap();

        let events = fs.execution_events(&access).unwrap();
        assert_eq!(events.len(), 1);
        let e = &events[0];
        assert_eq!(
            (e.object, e.principal, e.grant, e.cause),
            (double, grants.principal(), Some(grant.id()), Some(cause))
        );
        assert_eq!(fs.journal(&root(&host), &p("/tools")).unwrap().len(), 3, "two mounts and one invoke");
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
    fn transactions_are_atomic_and_detect_conflicts() {
        let (fs, host) = setup();
        let r = root(&host);
        let n = fs.bind(&r, &p("/state/n"), Value::Int(1)).unwrap();
        let m = fs.bind(&r, &p("/state/m"), Value::Int(0)).unwrap();

        let mut tx = fs.transaction();
        tx.read(&r, n).unwrap();
        tx.write(&r, m, Value::Int(2)).unwrap();
        fs.write(&r, n, Value::Int(9)).unwrap();
        assert_eq!(tx.commit(), Err(Error::Conflict(n)));
        assert_eq!(fs.read(&r, m).unwrap(), Value::Int(0));

        let mut tx = fs.transaction();
        tx.write(&r, n, Value::Int(10)).unwrap();
        tx.write(&r, m, Value::Int(20)).unwrap();
        tx.commit().unwrap();
        assert_eq!(fs.read(&r, m).unwrap(), Value::Int(20));
    }

    #[test]
    fn identical_seeds_give_identical_identities() {
        let (a, _) = setup();
        let (b, _) = setup();
        assert_eq!(a.resolve(&p("/tools/double")), b.resolve(&p("/tools/double")));
    }
}

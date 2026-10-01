use crate::{Authority, Error, Path, Result, Rights, Value};
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};

/// An object that can be invoked: the raw, untyped form of a capability.
///
/// Typed capabilities live in the `capability` crate and are adapted to this
/// trait when mounted. The namespace never holds the lock while invoking, so
/// an implementation may call back into `ctx.fs` under `ctx.authority`.
pub trait Invoke: Send + Sync {
    fn invoke(&self, ctx: &CallContext<'_>, input: Value) -> std::result::Result<Value, String>;

    /// A description of the object's contract, surfaced by `inspect`.
    fn signature(&self) -> Value {
        Value::Null
    }
}

/// What an invoked object is given: the namespace and the caller's authority,
/// never more.
pub struct CallContext<'a> {
    pub fs: &'a MeatFs,
    pub authority: &'a Authority,
    pub path: &'a Path,
}

enum Node {
    Dir(BTreeMap<String, Node>),
    Data { value: Value, version: u64 },
    Object(Arc<dyn Invoke>),
}

impl Node {
    fn kind(&self) -> NodeKind {
        match self {
            Node::Dir(_) => NodeKind::Dir,
            Node::Data { .. } => NodeKind::Data,
            Node::Object(_) => NodeKind::Capability,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Dir,
    Data,
    Capability,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inspection {
    pub path: Path,
    pub kind: NodeKind,
    /// Data version; 0 for non-data objects.
    pub version: u64,
    /// Child names, for directories.
    pub children: Vec<String>,
    /// Contract, for capabilities.
    pub signature: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub seq: u64,
    pub path: Path,
    pub principal: String,
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    Written { version: u64 },
    Mounted,
    Invoked { ok: bool },
}

/// A live stream of events under a path prefix.
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

#[derive(Default)]
struct Events {
    seq: u64,
    journal: Vec<Event>,
    subscribers: Vec<(Path, Sender<Event>)>,
}

impl Events {
    fn emit(&mut self, path: &Path, principal: &str, kind: EventKind) {
        self.seq += 1;
        let event = Event { seq: self.seq, path: path.clone(), principal: principal.to_owned(), kind };
        self.subscribers.retain(|(prefix, tx)| !event.path.starts_with(prefix) || tx.send(event.clone()).is_ok());
        self.journal.push(event);
    }
}

/// The in-memory MeatFS namespace.
pub struct MeatFs {
    root: RwLock<Node>,
    events: Mutex<Events>,
}

impl Default for MeatFs {
    fn default() -> Self {
        Self::new()
    }
}

fn check(auth: &Authority, path: &Path, needed: Rights) -> Result<()> {
    if auth.allows(path, needed) {
        Ok(())
    } else {
        Err(Error::Denied { path: path.clone(), needed })
    }
}

fn lookup<'n>(root: &'n Node, path: &Path) -> Result<&'n Node> {
    let mut node = root;
    for seg in path.segments() {
        node = match node {
            Node::Dir(children) => children.get(seg),
            _ => None,
        }
        .ok_or_else(|| Error::NotFound(path.clone()))?;
    }
    Ok(node)
}

/// The directory that should hold `path`, creating intermediate directories.
fn parent_dir<'n>(root: &'n mut Node, path: &Path) -> Result<&'n mut BTreeMap<String, Node>> {
    let parent = path.parent().ok_or_else(|| Error::Unsupported { path: path.clone(), op: "replace root" })?;
    let mut node = root;
    for seg in parent.segments() {
        let Node::Dir(children) = node else {
            return Err(Error::AlreadyExists(parent));
        };
        node = children.entry(seg.clone()).or_insert_with(|| Node::Dir(BTreeMap::new()));
    }
    match node {
        Node::Dir(children) => Ok(children),
        _ => Err(Error::AlreadyExists(parent)),
    }
}

fn version_of(root: &Node, path: &Path) -> u64 {
    match lookup(root, path) {
        Ok(Node::Data { version, .. }) => *version,
        _ => 0,
    }
}

/// Write data at `path`, returning its new version.
fn put(root: &mut Node, path: &Path, value: Value) -> Result<u64> {
    let name = path.name().expect("root rejected by parent_dir").to_owned();
    let dir = parent_dir(root, path)?;
    match dir.get_mut(&name) {
        None => {
            dir.insert(name, Node::Data { value, version: 1 });
            Ok(1)
        }
        Some(Node::Data { value: slot, version }) => {
            *slot = value;
            *version += 1;
            Ok(*version)
        }
        Some(_) => Err(Error::Unsupported { path: path.clone(), op: "write" }),
    }
}

impl MeatFs {
    pub fn new() -> Self {
        Self { root: RwLock::new(Node::Dir(BTreeMap::new())), events: Mutex::default() }
    }

    fn emit(&self, path: &Path, auth: &Authority, kind: EventKind) {
        self.events.lock().unwrap().emit(path, &auth.principal, kind);
    }

    pub fn read(&self, auth: &Authority, path: &Path) -> Result<Value> {
        check(auth, path, Rights::READ)?;
        match lookup(&self.root.read().unwrap(), path)? {
            Node::Data { value, .. } => Ok(value.clone()),
            _ => Err(Error::Unsupported { path: path.clone(), op: "read" }),
        }
    }

    pub fn write(&self, auth: &Authority, path: &Path, value: Value) -> Result<u64> {
        check(auth, path, Rights::WRITE)?;
        let version = put(&mut self.root.write().unwrap(), path, value)?;
        self.emit(path, auth, EventKind::Written { version });
        Ok(version)
    }

    /// Mount an invocable object at `path`.
    pub fn mount(&self, auth: &Authority, path: &Path, object: Arc<dyn Invoke>) -> Result<()> {
        check(auth, path, Rights::WRITE)?;
        {
            let mut root = self.root.write().unwrap();
            let name = path.name().ok_or_else(|| Error::AlreadyExists(path.clone()))?.to_owned();
            let dir = parent_dir(&mut root, path)?;
            if dir.contains_key(&name) {
                return Err(Error::AlreadyExists(path.clone()));
            }
            dir.insert(name, Node::Object(object));
        }
        self.emit(path, auth, EventKind::Mounted);
        Ok(())
    }

    pub fn invoke(&self, auth: &Authority, path: &Path, input: Value) -> Result<Value> {
        check(auth, path, Rights::INVOKE)?;
        let object = match lookup(&self.root.read().unwrap(), path)? {
            Node::Object(object) => Arc::clone(object),
            _ => return Err(Error::Unsupported { path: path.clone(), op: "invoke" }),
        };
        let ctx = CallContext { fs: self, authority: auth, path };
        let result = object.invoke(&ctx, input);
        self.emit(path, auth, EventKind::Invoked { ok: result.is_ok() });
        result.map_err(|message| Error::Capability { path: path.clone(), message })
    }

    pub fn inspect(&self, auth: &Authority, path: &Path) -> Result<Inspection> {
        check(auth, path, Rights::INSPECT)?;
        let root = self.root.read().unwrap();
        let node = lookup(&root, path)?;
        let mut inspection = Inspection {
            path: path.clone(),
            kind: node.kind(),
            version: 0,
            children: Vec::new(),
            signature: Value::Null,
        };
        match node {
            Node::Dir(children) => inspection.children = children.keys().cloned().collect(),
            Node::Data { version, .. } => inspection.version = *version,
            Node::Object(object) => inspection.signature = object.signature(),
        }
        Ok(inspection)
    }

    /// Receive every future event at or beneath `prefix`.
    pub fn subscribe(&self, auth: &Authority, prefix: &Path) -> Result<Subscription> {
        check(auth, prefix, Rights::SUBSCRIBE)?;
        let (tx, rx) = mpsc::channel();
        self.events.lock().unwrap().subscribers.push((prefix.clone(), tx));
        Ok(Subscription { rx })
    }

    /// The audit journal of every event under `prefix`.
    pub fn journal(&self, auth: &Authority, prefix: &Path) -> Result<Vec<Event>> {
        check(auth, prefix, Rights::INSPECT)?;
        let events = self.events.lock().unwrap();
        Ok(events.journal.iter().filter(|e| e.path.starts_with(prefix)).cloned().collect())
    }

    /// Every data/capability path in the namespace, depth first.
    pub fn walk(&self, auth: &Authority, prefix: &Path) -> Result<Vec<(Path, NodeKind)>> {
        check(auth, prefix, Rights::INSPECT)?;
        fn go(node: &Node, path: Path, out: &mut Vec<(Path, NodeKind)>) {
            match node {
                Node::Dir(children) => {
                    for (name, child) in children {
                        go(child, path.join(name).expect("stored names are valid"), out);
                    }
                }
                leaf => out.push((path, leaf.kind())),
            }
        }
        let mut out = Vec::new();
        go(lookup(&self.root.read().unwrap(), prefix)?, prefix.clone(), &mut out);
        Ok(out)
    }

    pub fn transaction<'a>(&'a self, auth: &'a Authority) -> Transaction<'a> {
        Transaction { fs: self, auth, reads: Vec::new(), writes: Vec::new() }
    }
}

/// Optimistic, all-or-nothing batch of writes.
///
/// Reads record the version seen; `commit` fails with [`Error::Conflict`] if
/// any of them changed, and otherwise applies every staged write atomically.
pub struct Transaction<'a> {
    fs: &'a MeatFs,
    auth: &'a Authority,
    reads: Vec<(Path, u64)>,
    writes: Vec<(Path, Value)>,
}

impl Transaction<'_> {
    pub fn read(&mut self, path: &Path) -> Result<Value> {
        if let Some((_, v)) = self.writes.iter().rev().find(|(p, _)| p == path) {
            return Ok(v.clone());
        }
        check(self.auth, path, Rights::READ)?;
        let root = self.fs.root.read().unwrap();
        let value = match lookup(&root, path)? {
            Node::Data { value, .. } => value.clone(),
            _ => return Err(Error::Unsupported { path: path.clone(), op: "read" }),
        };
        self.reads.push((path.clone(), version_of(&root, path)));
        Ok(value)
    }

    pub fn write(&mut self, path: &Path, value: Value) -> Result<()> {
        check(self.auth, path, Rights::WRITE)?;
        self.writes.push((path.clone(), value));
        Ok(())
    }

    pub fn commit(self) -> Result<()> {
        let mut root = self.fs.root.write().unwrap();
        for (path, seen) in &self.reads {
            if version_of(&root, path) != *seen {
                return Err(Error::Conflict(path.clone()));
            }
        }
        // Validate every target before mutating anything.
        for (path, _) in &self.writes {
            if let Ok(node) = lookup(&root, path) {
                if !matches!(node, Node::Data { .. }) {
                    return Err(Error::Unsupported { path: path.clone(), op: "write" });
                }
            }
            let mut ancestor = path.parent();
            while let Some(dir) = ancestor {
                if let Ok(node) = lookup(&root, &dir) {
                    if !matches!(node, Node::Dir(_)) {
                        return Err(Error::AlreadyExists(dir));
                    }
                }
                ancestor = dir.parent();
            }
        }
        let mut committed = Vec::with_capacity(self.writes.len());
        for (path, value) in self.writes {
            let version = put(&mut root, &path, value)?;
            committed.push((path, version));
        }
        drop(root);
        let mut events = self.fs.events.lock().unwrap();
        for (path, version) in committed {
            events.emit(&path, &self.auth.principal, EventKind::Written { version });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    struct Double;
    impl Invoke for Double {
        fn invoke(&self, _: &CallContext<'_>, input: Value) -> std::result::Result<Value, String> {
            match input {
                Value::Int(i) => Ok(Value::Int(i * 2)),
                other => Err(format!("expected int, got {}", other.kind())),
            }
        }
    }

    #[test]
    fn read_write_versions() {
        let fs = MeatFs::new();
        let root = Authority::root("host");
        assert_eq!(fs.write(&root, &p("/state/a"), "x".into()).unwrap(), 1);
        assert_eq!(fs.write(&root, &p("/state/a"), "y".into()).unwrap(), 2);
        assert_eq!(fs.read(&root, &p("/state/a")).unwrap(), Value::from("y"));
        assert_eq!(fs.inspect(&root, &p("/state")).unwrap().children, vec!["a"]);
        assert!(matches!(fs.read(&root, &p("/state")), Err(Error::Unsupported { .. })));
    }

    #[test]
    fn authority_is_enforced() {
        let fs = MeatFs::new();
        let agent = Authority::new("agent").grant(p("/state/agent"), Rights::READ | Rights::WRITE);
        fs.write(&agent, &p("/state/agent/x"), Value::Int(1)).unwrap();
        assert!(matches!(fs.write(&agent, &p("/state/other"), Value::Null), Err(Error::Denied { .. })));
        assert!(matches!(fs.read(&agent, &p("/memory/secret")), Err(Error::Denied { .. })));
    }

    #[test]
    fn invoke_and_audit() {
        let fs = MeatFs::new();
        let root = Authority::root("host");
        fs.mount(&root, &p("/tools/double"), Arc::new(Double)).unwrap();
        let sub = fs.subscribe(&root, &p("/tools")).unwrap();
        assert_eq!(fs.invoke(&root, &p("/tools/double"), Value::Int(21)).unwrap(), Value::Int(42));
        assert!(fs.invoke(&root, &p("/tools/double"), "no".into()).is_err());
        assert!(matches!(fs.write(&root, &p("/tools/double"), Value::Null), Err(Error::Unsupported { .. })));
        let kinds: Vec<_> = sub.drain().into_iter().map(|e| e.kind).collect();
        assert_eq!(kinds, vec![EventKind::Invoked { ok: true }, EventKind::Invoked { ok: false }]);
        assert_eq!(fs.journal(&root, &p("/tools")).unwrap().len(), 3);
    }

    #[test]
    fn transactions_are_atomic_and_detect_conflicts() {
        let fs = MeatFs::new();
        let root = Authority::root("host");
        fs.write(&root, &p("/state/n"), Value::Int(1)).unwrap();

        let mut tx = fs.transaction(&root);
        tx.read(&p("/state/n")).unwrap();
        tx.write(&p("/state/m"), Value::Int(2)).unwrap();
        fs.write(&root, &p("/state/n"), Value::Int(9)).unwrap();
        assert_eq!(tx.commit(), Err(Error::Conflict(p("/state/n"))));
        assert!(fs.read(&root, &p("/state/m")).is_err());

        let mut tx = fs.transaction(&root);
        tx.write(&p("/state/a"), Value::Int(1)).unwrap();
        tx.write(&p("/state/b"), Value::Int(2)).unwrap();
        tx.commit().unwrap();
        assert_eq!(fs.read(&root, &p("/state/b")).unwrap(), Value::Int(2));
    }
}

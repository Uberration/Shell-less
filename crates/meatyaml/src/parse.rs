use crate::{
    Edge, EdgeKind, Error, ErrorKind, Graph, GraphOutput, InvokeOp, Node, Op, Program, ReadOp, Selector, Use,
    ValueExpr, WriteOp,
};
use meatfs::{ContentCapture, NodeId, Path, Rights, Value};
use std::collections::HashMap;
use yaml_rust2::{Yaml, YamlLoader};

/// A compile failure before capture is applied. Internal only: it becomes
/// an [`Error`] record, under the host's capture policy, at the boundary.
struct Raw {
    kind: ErrorKind,
    at: Loc,
    position: Option<(usize, usize)>,
    detail: String,
}

impl Raw {
    fn record(self, capture: ContentCapture) -> Error {
        let location = match capture {
            ContentCapture::Omit => self.at.safe,
            ContentCapture::Inline => self.at.full,
        };
        let detail = self.detail;
        Error { kind: self.kind, location, position: self.position, detail: capture.text(|| detail) }
    }
}

type Result<T> = std::result::Result<T, Raw>;

/// A location within a program, kept in two forms: `full`, and `safe`, in
/// which every source-controlled segment (map keys, unknown operation
/// names, output names) is replaced by its position in the enclosing map,
/// `#i`, so distinct segments stay distinct.
#[derive(Debug, Clone, Default)]
struct Loc {
    full: String,
    safe: String,
}

impl Loc {
    fn push(&self, full: &str, safe: &str, index: bool) -> Loc {
        let sep = if index || self.full.is_empty() { "" } else { "." };
        Loc { full: format!("{}{sep}{full}", self.full), safe: format!("{}{sep}{safe}", self.safe) }
    }

    /// A fixed grammar label.
    fn field(&self, label: &'static str) -> Loc {
        self.push(label, label, false)
    }

    fn index(&self, i: usize) -> Loc {
        let i = format!("[{i}]");
        self.push(&i, &i, true)
    }

    /// A key taken from the source text, at `index` in its map.
    fn name(&self, name: &str, index: usize) -> Loc {
        self.push(name, &format!("#{index}"), false)
    }

    /// An operation or constructor name: fixed if the grammar knows it.
    fn keyword(&self, word: &str, index: usize, known: &[&'static str]) -> Loc {
        match known.iter().find(|k| **k == word) {
            Some(k) => self.field(k),
            None => self.name(word, index),
        }
    }
}

fn err<T>(kind: ErrorKind, at: &Loc, detail: impl Into<String>) -> Result<T> {
    Err(Raw { kind, at: at.clone(), position: None, detail: detail.into() })
}

const OPERATIONS: &[&str] = &["read", "write", "invoke", "compose"];
const CONSTRUCTORS: &[&str] = &["literal", "select", "map", "list"];

/// Compile MEATYAML source into a validated [`Program`]. Error records
/// retain no source text; see [`compile_with`].
pub fn compile(source: &str) -> std::result::Result<Program, Error> {
    compile_with(source, ContentCapture::Omit)
}

/// Compile, recording any error under the host's `capture` policy.
pub fn compile_with(source: &str, capture: ContentCapture) -> std::result::Result<Program, Error> {
    compile_raw(source).map_err(|raw| raw.record(capture))
}

fn compile_raw(source: &str) -> Result<Program> {
    let root = Loc::default();
    let docs = YamlLoader::load_from_str(source).map_err(|e| Raw {
        kind: ErrorKind::Syntax,
        at: Loc::default(),
        position: Some((e.marker().line(), e.marker().col() + 1)),
        detail: e.info().to_owned(),
    })?;
    let [doc] = docs.as_slice() else {
        return err(ErrorKind::Shape, &root, "expected exactly one document");
    };
    let mut top = Map::of(doc, &root)?;

    let agent = {
        let at = root.field("agent");
        let mut agent = Map::of(top.require("agent")?, &at)?;
        let name = text(agent.require("name")?, &at.field("name"))?;
        agent.finish()?;
        name
    };
    let authority = match top.optional("authority") {
        Some(yaml) => compile_uses(yaml, &root.field("authority"))?.into_iter().map(|u| (u.path, u.rights)).collect(),
        None => Vec::new(),
    };
    let steps = match top.require("flow")? {
        Yaml::Array(steps) if !steps.is_empty() => steps,
        _ => return err(ErrorKind::Shape, &root.field("flow"), "expected a non-empty list of steps"),
    };
    let outputs = top.optional("outputs");
    top.finish()?;

    let mut compiler = Compiler { authority, ..Compiler::default() };
    for (i, step) in steps.iter().enumerate() {
        compiler.step(step, &root.field("flow").index(i))?;
    }
    let outputs = match outputs {
        Some(yaml) => compiler.outputs(yaml, &root.field("outputs"))?,
        None => Vec::new(),
    };
    Ok(Program { agent, authority: compiler.authority, graph: Graph::new(compiler.nodes, compiler.edges, outputs) })
}

/// A list of `{ path, rights }`, as used by `authority` and `uses`.
fn compile_uses(yaml: &Yaml, at: &Loc) -> Result<Vec<Use>> {
    let Yaml::Array(entries) = yaml else {
        return err(ErrorKind::Shape, at, "expected a list of { path, rights }");
    };
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let at = at.index(i);
            let mut entry = Map::of(entry, &at)?;
            let path = path(entry.require("path")?, &at.field("path"))?;
            let rights_at = at.field("rights");
            let names: Vec<&str> = match entry.require("rights")? {
                Yaml::String(s) => s.split('+').map(str::trim).collect(),
                Yaml::Array(items) => items.iter().map(|y| y.as_str().unwrap_or("")).collect(),
                _ => return err(ErrorKind::Shape, &rights_at, "expected rights such as [read, write] or `read+write`"),
            };
            entry.finish()?;
            let rights = names.iter().try_fold(Rights::NONE, |acc, name| match Rights::from_name(name) {
                Some(r) => Ok(acc | r),
                None => err(ErrorKind::UnknownRight, &rights_at, format!("unknown right `{name}`")),
            })?;
            Ok(Use { path, rights })
        })
        .collect()
}

/// Whether two footprints touch the same state with at least one write.
fn conflicts(a: &Op, b: &Op) -> bool {
    let touches = |r: Rights| r.contains(Rights::READ) || r.contains(Rights::WRITE);
    a.footprint().iter().any(|(pa, ra)| {
        b.footprint().iter().any(|(pb, rb)| {
            (pa.starts_with(pb) || pb.starts_with(pa))
                && ((ra.contains(Rights::WRITE) && touches(*rb)) || (rb.contains(Rights::WRITE) && touches(*ra)))
        })
    })
}

/// Lowers steps into IR, resolving all source-level references to edges.
#[derive(Default)]
struct Compiler {
    authority: Vec<(Path, Rights)>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    labels: HashMap<String, NodeId>,
}

impl Compiler {
    fn declared(&self, path: &Path) -> Rights {
        self.authority.iter().filter(|(prefix, _)| path.starts_with(prefix)).fold(Rights::NONE, |acc, (_, r)| acc | *r)
    }

    /// Whether `to` is already reachable from `from` along existing edges.
    fn reaches(&self, from: NodeId, to: NodeId) -> bool {
        let mut stack = vec![from];
        let mut seen = vec![false; self.nodes.len() + 1];
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if !std::mem::replace(&mut seen[n.0 as usize], true) {
                stack.extend(self.edges.iter().filter(|e| e.from == n).map(|e| e.to));
            }
        }
        false
    }

    fn step(&mut self, step: &Yaml, at: &Loc) -> Result<()> {
        let mut step_map = Map::of(step, at)?;
        let label = step_map.optional("id").map(|y| text(y, &at.field("id"))).transpose()?;
        let after = step_map.optional("after");
        let [(op_index, op_name, body)] = step_map.entries[..] else {
            return err(ErrorKind::Shape, at, "a step has exactly one operation (plus optional `id` and `after`)");
        };
        let id = NodeId(self.nodes.len() as u32);
        let previous = self.nodes.last().map(|n| n.id);
        let after_at = at.field("after");
        let after = match after {
            None => Vec::new(),
            Some(Yaml::Array(items)) => {
                items.iter().map(|y| self.reference(y, &after_at, previous)).collect::<Result<Vec<_>>>()?
            }
            Some(y) => vec![self.reference(y, &after_at, previous)?],
        };
        let at = at.keyword(op_name, op_index, OPERATIONS);

        let (op, data_sources) = if op_name == "compose" {
            let expr = self.expr(body, &at, previous)?;
            let sources = expr.sources();
            (Op::Compose(expr), sources)
        } else {
            self.operation(op_name, body, &at, previous)?
        };

        for (path, needed) in op.footprint() {
            if !self.declared(path).contains(needed) {
                let detail = format!("{path} needs `{needed}`, which is not declared in `authority`");
                return err(ErrorKind::Undeclared, &at, detail);
            }
        }
        for from in data_sources {
            self.edges.push(Edge { from, to: id, kind: EdgeKind::Data });
        }
        if let Some(label) = &label {
            if self.labels.insert(label.clone(), id).is_some() {
                return err(ErrorKind::DuplicateId, &at, format!("duplicate id `{label}`"));
            }
        }
        self.nodes.push(Node { id, label, op });

        // Order: preserve sequential semantics wherever footprints conflict,
        // plus whatever the source asked for explicitly with `after`.
        let node = &self.nodes[id.0 as usize];
        // Latest first, so edges implied through a later node are skipped.
        let mut required: Vec<NodeId> = self.nodes[..id.0 as usize]
            .iter()
            .filter(|earlier| conflicts(&earlier.op, &node.op))
            .map(|earlier| earlier.id)
            .chain(after)
            .collect();
        required.sort_unstable_by(|a, b| b.cmp(a));
        required.dedup();
        for earlier in required {
            if !self.reaches(earlier, id) {
                self.edges.push(Edge { from: earlier, to: id, kind: EdgeKind::Order });
            }
        }
        Ok(())
    }

    /// A read, write or invoke step: the op and its data source, if any.
    fn operation(&self, op_name: &str, body: &Yaml, at: &Loc, previous: Option<NodeId>) -> Result<(Op, Vec<NodeId>)> {
        if !matches!(op_name, "read" | "write" | "invoke") {
            return err(ErrorKind::UnknownOperation, at, format!("unknown operation `{op_name}`"));
        }
        // Short form `op: /path` takes its data from the previous step.
        let (target, literal, from, uses) = match body {
            Yaml::String(_) => (path(body, at)?, None, previous, Vec::new()),
            _ => {
                let mut body = Map::of(body, at)?;
                let target = path(body.require("path")?, &at.field("path"))?;
                let literal_key: Option<&'static str> = match op_name {
                    "write" => Some("value"),
                    "invoke" => Some("input"),
                    _ => None,
                };
                let mut literal = literal_key.and_then(|k| body.optional(k).map(|y| (k, y)));
                let mut from = body.optional("from");
                // `input: { from: x }` is a reference, not a literal map.
                if let Some((_, Yaml::Hash(h))) = literal {
                    if let [(Yaml::String(k), reference)] = h.iter().collect::<Vec<_>>()[..] {
                        if k == "from" && from.is_none() {
                            (literal, from) = (None, Some(reference));
                        }
                    }
                }
                let uses = match body.optional("uses") {
                    Some(y) if op_name == "invoke" => compile_uses(y, &at.field("uses"))?,
                    Some(_) => return err(ErrorKind::UnknownKey, &at.field("uses"), "only `invoke` takes `uses`"),
                    None => Vec::new(),
                };
                body.finish()?;
                let from = match (literal, from) {
                    (Some((k, _)), Some(_)) => {
                        return err(ErrorKind::ConflictingData, at, format!("give either `{k}` or `from`, not both"))
                    }
                    (Some(_), None) => None,
                    (None, Some(from)) => Some(self.reference(from, &at.field("from"), previous)?),
                    (None, None) => previous,
                };
                let literal = literal.map(|(k, y)| value(y, &at.field(k))).transpose()?;
                (target, literal, from, uses)
            }
        };

        let op = match op_name {
            "read" => Op::Read(ReadOp { target }),
            "invoke" => Op::Invoke(InvokeOp { target, input: literal, uses }),
            _ => {
                if literal.is_none() && from.is_none() {
                    return err(ErrorKind::MissingData, at, "nothing to write: give `value` or `from`");
                }
                Op::Write(WriteOp { target, value: literal })
            }
        };
        // Reads take no input; literals need none.
        let data = match (from, &op) {
            (_, Op::Read(_)) => None,
            (_, op) if op.literal().is_some() => None,
            (from, _) => from,
        };
        Ok((op, data.into_iter().collect()))
    }

    /// One composition expression: a map with exactly one constructor key.
    fn expr(&self, yaml: &Yaml, at: &Loc, previous: Option<NodeId>) -> Result<ValueExpr> {
        let map = Map::of(yaml, at)?;
        let [(index, constructor, body)] = map.entries[..] else {
            return err(ErrorKind::Shape, at, "an expression has exactly one of `literal`, `select`, `map`, `list`");
        };
        let at = at.keyword(constructor, index, CONSTRUCTORS);
        match constructor {
            "literal" => Ok(ValueExpr::Literal(value(body, &at)?)),
            "select" => {
                let mut select = Map::of(body, &at)?;
                let source = self.reference(select.require("from")?, &at.field("from"), previous)?;
                let path_at = at.field("path");
                let path = match select.optional("path") {
                    None => Vec::new(),
                    Some(Yaml::Array(items)) => items
                        .iter()
                        .enumerate()
                        .map(|(i, item)| match item {
                            Yaml::String(k) => Ok(Selector::Key(k.clone())),
                            Yaml::Integer(n) if *n >= 0 => Ok(Selector::Index(*n as usize)),
                            _ => err(
                                ErrorKind::InvalidValue,
                                &path_at.index(i),
                                "expected a key (text) or index (non-negative int)",
                            ),
                        })
                        .collect::<Result<_>>()?,
                    Some(_) => return err(ErrorKind::Shape, &path_at, "expected a list of keys and indexes"),
                };
                select.finish()?;
                Ok(ValueExpr::Select { source, path })
            }
            "map" => {
                let fields = Map::of(body, &at)?;
                let fields = fields
                    .entries
                    .iter()
                    .map(|&(i, k, v)| Ok((k.to_owned(), self.expr(v, &at.name(k, i), previous)?)))
                    .collect::<Result<_>>()?;
                Ok(ValueExpr::Map(fields))
            }
            "list" => match body {
                Yaml::Array(items) => Ok(ValueExpr::List(
                    items
                        .iter()
                        .enumerate()
                        .map(|(i, item)| self.expr(item, &at.index(i), previous))
                        .collect::<Result<_>>()?,
                )),
                _ => err(ErrorKind::Shape, &at, "expected a list of expressions"),
            },
            other => err(ErrorKind::UnknownConstructor, &at, format!("unknown constructor `{other}`")),
        }
    }

    fn reference(&self, from: &Yaml, at: &Loc, previous: Option<NodeId>) -> Result<NodeId> {
        match text(from, at)?.as_str() {
            "previous" => previous.ok_or(()).or_else(|_| err(ErrorKind::UnknownReference, at, "no previous step")),
            label => match self.labels.get(label) {
                Some(&id) => Ok(id),
                None => err(ErrorKind::UnknownReference, at, format!("no earlier step with id `{label}`")),
            },
        }
    }

    fn outputs(&self, yaml: &Yaml, at: &Loc) -> Result<Vec<GraphOutput>> {
        let map = Map::of(yaml, at)?;
        map.entries
            .iter()
            .map(|&(index, name, spec)| {
                let at = at.name(name, index);
                let source = match spec {
                    Yaml::Hash(_) => {
                        let mut spec = Map::of(spec, &at)?;
                        let from = spec.require("from")?;
                        spec.finish()?;
                        from
                    }
                    other => other,
                };
                let source = self.reference(source, &at.field("from"), None)?;
                Ok(GraphOutput { name: name.to_owned(), source })
            })
            .collect()
    }
}

/// A YAML map whose keys must all be consumed. Entries keep their position
/// in the source, which locates source-controlled keys without quoting them.
struct Map<'y> {
    at: Loc,
    entries: Vec<(usize, &'y str, &'y Yaml)>,
}

impl<'y> Map<'y> {
    fn of(yaml: &'y Yaml, at: &Loc) -> Result<Self> {
        let Yaml::Hash(hash) = yaml else {
            return err(ErrorKind::Shape, at, "expected a map");
        };
        let mut entries = Vec::with_capacity(hash.len());
        for (i, (k, v)) in hash.iter().enumerate() {
            match k.as_str() {
                Some(k) => entries.push((i, k, v)),
                None => return err(ErrorKind::Shape, at, "map keys must be strings"),
            };
        }
        Ok(Map { at: at.clone(), entries })
    }

    fn optional(&mut self, key: &str) -> Option<&'y Yaml> {
        let i = self.entries.iter().position(|&(_, k, _)| k == key)?;
        Some(self.entries.remove(i).2)
    }

    /// `key` is always a grammar label, so it may appear in the location.
    fn require(&mut self, key: &'static str) -> Result<&'y Yaml> {
        match self.optional(key) {
            Some(v) => Ok(v),
            None => err(ErrorKind::MissingKey, &self.at.field(key), format!("missing `{key}`")),
        }
    }

    fn finish(self) -> Result<()> {
        match self.entries.first() {
            None => Ok(()),
            Some(&(i, key, _)) => err(ErrorKind::UnknownKey, &self.at.name(key, i), format!("unknown key `{key}`")),
        }
    }
}

fn text(yaml: &Yaml, at: &Loc) -> Result<String> {
    match yaml {
        Yaml::String(s) => Ok(s.clone()),
        _ => err(ErrorKind::Shape, at, "expected text"),
    }
}

fn path(yaml: &Yaml, at: &Loc) -> Result<Path> {
    Path::parse(&text(yaml, at)?).or_else(|e| err(ErrorKind::InvalidPath, at, e.to_string()))
}

fn value(yaml: &Yaml, at: &Loc) -> Result<Value> {
    Ok(match yaml {
        Yaml::Null => Value::Null,
        Yaml::Boolean(b) => Value::Bool(*b),
        Yaml::Integer(i) => Value::Int(*i),
        Yaml::Real(r) => {
            Value::Float(r.parse().or_else(|_| err(ErrorKind::InvalidValue, at, format!("bad float `{r}`")))?)
        }
        Yaml::String(s) => Value::Text(s.clone()),
        Yaml::Array(items) => {
            Value::List(items.iter().enumerate().map(|(i, y)| value(y, &at.index(i))).collect::<Result<_>>()?)
        }
        Yaml::Hash(_) => {
            let map = Map::of(yaml, at)?;
            Value::Map(
                map.entries
                    .iter()
                    .map(|&(i, k, v)| Ok((k.to_owned(), value(v, &at.name(k, i))?)))
                    .collect::<Result<_>>()?,
            )
        }
        _ => return err(ErrorKind::InvalidValue, at, "unsupported yaml value"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The location an error records when the host captures inline.
    fn located(src: &str) -> String {
        compile_with(src, ContentCapture::Inline).unwrap_err().location
    }

    const ECHO: &str = include_str!("../../../examples/echo.meat.yaml");
    const BUTCHER: &str = include_str!("../../../examples/butcher.meat.yaml");

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    fn edges(program: &Program) -> Vec<(u32, u32, EdgeKind)> {
        program.graph.edges.iter().map(|e| (e.from.0, e.to.0, e.kind)).collect()
    }

    #[test]
    fn compiles_to_ir_with_explicit_edges_and_outputs() {
        let program = compile(ECHO).unwrap();
        let g = &program.graph;
        assert_eq!(
            g.nodes.iter().map(|n| n.op.clone()).collect::<Vec<_>>(),
            vec![
                Op::Invoke(InvokeOp {
                    target: p("/tools/echo"),
                    input: Some(Value::map([("text", Value::from("MEAT"))])),
                    uses: vec![],
                }),
                Op::Write(WriteOp { target: p("/state/result"), value: None }),
            ]
        );
        assert_eq!(edges(&program), vec![(0, 1, EdgeKind::Data)]);
        assert_eq!(g.outputs, vec![GraphOutput { name: "result".into(), source: NodeId(1) }]);
        assert_eq!(
            program.request().wants,
            vec![(p("/tools/echo"), Rights::INVOKE), (p("/state/result"), Rights::WRITE)]
        );
    }

    #[test]
    fn butcher_branches_without_spurious_ordering() {
        let program = compile(BUTCHER).unwrap();
        // upper → echo, upper → store_upper, echo → store_echo; no order edges:
        // the four nodes touch disjoint state.
        assert_eq!(edges(&program), vec![(0, 1, EdgeKind::Data), (0, 2, EdgeKind::Data), (1, 3, EdgeKind::Data)]);
        let names: Vec<_> = program.graph.outputs.iter().map(|o| (o.name.as_str(), o.source.0)).collect();
        assert_eq!(names, vec![("upper", 2), ("echo", 3)], "source order");
    }

    #[test]
    fn conflicting_footprints_get_order_edges() {
        let program = compile(
            r#"
agent: { name: a }
authority:
  - { path: /state, rights: [read, write] }
  - { path: /tools, rights: [invoke] }
flow:
  - write: { path: /state/x, value: 1 }
  - read: /state/x
  - write: { path: /state/y, value: 2 }
  - invoke: { path: /tools/t, input: null, uses: [{ path: /state/y, rights: [read] }] }
  - write: { path: /state/x, value: 3 }
  - id: last
    write: { path: /state/z, value: 4 }
    after: [previous]
"#,
        )
        .unwrap();
        assert_eq!(
            edges(&program),
            vec![
                (0, 1, EdgeKind::Order), // read after write of /state/x
                (2, 3, EdgeKind::Order), // invoke reads /state/y via `uses`
                (1, 4, EdgeKind::Order), // overwrite after read (0 → 4 is implied)
                (4, 5, EdgeKind::Order), // explicit `after`
            ]
        );
    }

    #[test]
    fn previous_is_source_sugar_for_an_edge() {
        let explicit = compile(ECHO).unwrap();
        let sugar = compile(&ECHO.replace("from: make_meat", "from: previous")).unwrap();
        let short = compile(
            &ECHO.replace("    write:\n      path: /state/result\n      from: make_meat", "    write: /state/result"),
        )
        .unwrap();
        assert_eq!(explicit.graph, sugar.graph);
        assert_eq!(explicit.graph, short.graph);
    }

    #[test]
    fn graph_identity_is_content_not_labels() {
        let a = compile(ECHO).unwrap().graph.id;
        let relabelled = compile(&ECHO.replace("make_meat", "m")).unwrap().graph.id;
        let changed = compile(&ECHO.replace("text: MEAT", "text: VEG")).unwrap().graph.id;
        let renamed_output = compile(&ECHO.replace("  result:", "  answer:")).unwrap().graph.id;
        assert_eq!(a, relabelled);
        assert_ne!(a, changed);
        assert_ne!(a, renamed_output);
    }

    #[test]
    fn rejects_undeclared_authority() {
        assert_eq!(located(&ECHO.replace("rights: [write]", "rights: [read]")), "flow[1].write");
        let e = compile(&BUTCHER.replace(
            "      input:\n        from: upper",
            "      from: upper\n      uses: [{ path: /memory, rights: [read] }]",
        ))
        .unwrap_err();
        assert_eq!(e.location, "flow[1].invoke");
    }

    #[test]
    fn rejects_malformed_programs() {
        let cases = [
            (ECHO.replace("from: make_meat", "from: nowhere"), "flow[1].write.from"),
            (ECHO.replace("from: make_meat", "from: make_meat\n      value: 1"), "flow[1].write"),
            (ECHO.replace("    write:", "    exec:"), "flow[1].exec"),
            (ECHO.replace("path: /tools/echo\n      input", "path: /tools/../etc\n      input"), "flow[0].invoke.path"),
            (ECHO.replace("  name: echoer", "  name: echoer\n  shell: bash"), "agent.shell"),
            (ECHO.replace("[invoke]", "[root]"), "authority[0].rights"),
            (ECHO.replace("- id: store", "- id: make_meat"), "flow[1].write"),
            (ECHO.replace("  - id: store\n", "  - id: store\n    read: /x\n"), "flow[1]"),
            (ECHO.replace("from: store", "from: ghost"), "outputs.result.from"),
        ];
        for (src, at) in cases {
            assert_eq!(located(&src), at, "{src}");
        }
    }

    const COMPOSER: &str = include_str!("../../../examples/composer.meat.yaml");

    #[test]
    fn composer_shares_one_source_across_branches() {
        let program = compile(COMPOSER).unwrap();
        // source → upper, source → request → infer; nothing else.
        assert_eq!(edges(&program), vec![(0, 1, EdgeKind::Data), (0, 2, EdgeKind::Data), (2, 3, EdgeKind::Data)]);
        let Op::Compose(request) = &program.graph.nodes[2].op else { panic!("request is a compose") };
        assert_eq!(request.sources(), vec![NodeId(0)]);
        assert_eq!(program.request().wants.len(), 2, "compose needs no authority");
    }

    #[test]
    fn literal_is_never_a_reference() {
        let program = compile(
            "agent: { name: a }\nflow:\n  - id: x\n    compose: { literal: 1 }\n  - compose:\n      literal:\n        from: x\n",
        )
        .unwrap();
        assert_eq!(
            program.graph.nodes[1].op,
            Op::Compose(ValueExpr::Literal(Value::map([("from", Value::from("x"))])))
        );
        assert!(program.graph.edges.is_empty());
    }

    #[test]
    fn selectors_distinguish_keys_from_indexes() {
        let src = |sel: &str| {
            format!("agent: {{ name: a }}\nflow:\n  - id: s\n    compose: {{ literal: [1] }}\n  - compose: {{ select: {{ from: s, path: [{sel}] }} }}\n")
        };
        let key = compile(&src("\"0\"")).unwrap();
        let index = compile(&src("0")).unwrap();
        let Op::Compose(ValueExpr::Select { path, .. }) = &key.graph.nodes[1].op else { panic!() };
        assert_eq!(path, &[Selector::Key("0".into())]);
        let Op::Compose(ValueExpr::Select { path, .. }) = &index.graph.nodes[1].op else { panic!() };
        assert_eq!(path, &[Selector::Index(0)]);
        assert_ne!(key.graph.id, index.graph.id);
    }

    #[test]
    fn composition_identity_and_errors() {
        let base = compile(COMPOSER).unwrap().graph.id;
        assert_eq!(compile(&COMPOSER.replace("source", "origin")).unwrap().graph.id, base, "labels are not identity");
        assert_ne!(compile(&COMPOSER.replace("path: [text]", "path: [txt]")).unwrap().graph.id, base);
        assert_ne!(compile(&COMPOSER.replace("literal: uppercase", "literal: lowercase")).unwrap().graph.id, base);

        let cases = [
            (
                COMPOSER.replace("from: source\n                    path", "from: ghost\n                    path"),
                "flow[2].compose.map.messages.list[1].map.content.select.from",
            ),
            (
                COMPOSER.replace(
                    "    compose:\n      literal:\n        text: meat",
                    "    compose:\n      literal: 1\n      list: []",
                ),
                "flow[0].compose",
            ),
            (
                COMPOSER.replace("literal: system", "quote: system"),
                "flow[2].compose.map.messages.list[0].map.role.quote",
            ),
            (
                COMPOSER.replace("path: [text]", "path: [-1]"),
                "flow[2].compose.map.messages.list[1].map.content.select.path[0]",
            ),
        ];
        for (src, at) in cases {
            assert_eq!(located(&src), at, "{src}");
        }
    }

    #[test]
    fn errors_have_stable_kinds_and_content_free_renderings() {
        const SECRET: &str = "zq-secret-5150";
        let cases = [
            (format!("agent: {{ name: a }}\nflow: [ {{ read: /x }}\n  {SECRET}: ]"), ErrorKind::Syntax),
            (format!("agent: {{ name: a }}\n{SECRET}: 1\nflow: [ {{ read: /x }} ]"), ErrorKind::UnknownKey),
            (format!("agent: {{ name: a }}\nflow: [ {{ {SECRET}: /x }} ]"), ErrorKind::UnknownOperation),
            (format!("agent: {{ name: a }}\nflow: [ {{ read: /a/../{SECRET} }} ]"), ErrorKind::InvalidPath),
            (format!("agent: {{ name: a }}\nauthority: [{{ path: /x, rights: [{SECRET}] }}]\nflow: [ {{ read: /x }} ]"), ErrorKind::UnknownRight),
            (format!("agent: {{ name: a }}\nflow: [ {{ read: /{SECRET} }} ]"), ErrorKind::Undeclared),
            (format!("agent: {{ name: a }}\nflow: [ {{ compose: {{ map: {{ {SECRET}: {{ quote: 1 }} }} }} }} ]"), ErrorKind::UnknownConstructor),
            (format!("agent: {{ name: a }}\nflow: [ {{ compose: {{ select: {{ from: {SECRET} }} }} }} ]"), ErrorKind::UnknownReference),
            (format!("agent: {{ name: a }}\nflow: [ {{ compose: {{ literal: 1 }} }} ]\noutputs: {{ {SECRET}: ghost }}"), ErrorKind::UnknownReference),
            (format!("agent: {{ name: a }}\nflow:\n  - {{ id: x, compose: {{ literal: 1 }} }}\n  - {{ compose: {{ map: {{ {SECRET}: {{ select: {{ from: x, path: [-1] }} }} }} }} }}"), ErrorKind::InvalidValue),
        ];
        for (src, kind) in cases {
            // Omit: the record itself holds no source text, in any field.
            let e = compile(&src).unwrap_err();
            assert_eq!(e.kind, kind, "{src}");
            assert_eq!(e.detail, None);
            assert!(!format!("{e:?}").contains(SECRET), "{e:?}");
            // Inline: the host chose to retain it.
            let revealed = compile_with(&src, ContentCapture::Inline).unwrap_err();
            assert!(revealed.detail.is_some());
            let revealing = !matches!(kind, ErrorKind::Syntax);
            assert!(format!("{revealed:?}").contains(SECRET) || !revealing, "{revealed:?}");
        }
        let syntax = compile(&format!("agent: {{ name: a }}\nflow: [ {{ read: /x }}\n  {SECRET}: ]")).unwrap_err();
        assert!(syntax.position.is_some());
    }

    /// Frozen: graph outputs keep source order, and that order is part of
    /// graph identity (a documented pre-release GraphId change).
    #[test]
    fn outputs_keep_source_order_in_identity() {
        let src = |outputs: &str| {
            format!("agent: {{ name: a }}\nflow:\n  - {{ id: x, compose: {{ literal: 1 }} }}\n  - {{ id: y, compose: {{ literal: 2 }} }}\noutputs:\n{outputs}")
        };
        let zy = compile(&src("  zeta: x\n  alpha: y\n")).unwrap();
        let names: Vec<_> = zy.graph.outputs.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, ["zeta", "alpha"], "source order, not sorted");
        let ay = compile(&src("  alpha: y\n  zeta: x\n")).unwrap();
        assert_ne!(zy.graph.id, ay.graph.id);
    }
}

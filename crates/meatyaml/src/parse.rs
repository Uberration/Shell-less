use crate::{
    Edge, EdgeKind, Error, Graph, GraphOutput, InvokeOp, Node, Op, Program, ReadOp, Selector, Use, ValueExpr, WriteOp,
};
use meatfs::{NodeId, Path, Rights, Value};
use std::collections::{BTreeMap, HashMap};
use yaml_rust2::{Yaml, YamlLoader};

type Result<T> = std::result::Result<T, Error>;

fn err<T>(at: impl Into<String>, message: impl Into<String>) -> Result<T> {
    Err(Error { at: at.into(), message: message.into() })
}

/// Compile MEATYAML source into a validated [`Program`].
pub fn compile(source: &str) -> Result<Program> {
    let docs = YamlLoader::load_from_str(source).or_else(|e| err("", format!("yaml: {e}")))?;
    let [doc] = docs.as_slice() else {
        return err("", "expected exactly one document");
    };
    let mut top = Map::of(doc, "")?;

    let agent = {
        let mut agent = Map::of(top.require("agent")?, "agent")?;
        let name = text(agent.require("name")?, "agent.name")?;
        agent.finish()?;
        name
    };
    let authority = match top.optional("authority") {
        Some(yaml) => compile_uses(yaml, "authority")?.into_iter().map(|u| (u.path, u.rights)).collect(),
        None => Vec::new(),
    };
    let steps = match top.require("flow")? {
        Yaml::Array(steps) if !steps.is_empty() => steps,
        _ => return err("flow", "expected a non-empty list of steps"),
    };
    let outputs = top.optional("outputs");
    top.finish()?;

    let mut compiler = Compiler { authority, ..Compiler::default() };
    for (i, step) in steps.iter().enumerate() {
        compiler.step(step, &format!("flow[{i}]"))?;
    }
    let outputs = match outputs {
        Some(yaml) => compiler.outputs(yaml)?,
        None => Vec::new(),
    };
    Ok(Program { agent, authority: compiler.authority, graph: Graph::new(compiler.nodes, compiler.edges, outputs) })
}

/// A list of `{ path, rights }`, as used by `authority` and `uses`.
fn compile_uses(yaml: &Yaml, at: &str) -> Result<Vec<Use>> {
    let Yaml::Array(entries) = yaml else {
        return err(at, "expected a list of { path, rights }");
    };
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let at = format!("{at}[{i}]");
            let mut entry = Map::of(entry, &at)?;
            let path = path(entry.require("path")?, &format!("{at}.path"))?;
            let rights_at = format!("{at}.rights");
            let names: Vec<&str> = match entry.require("rights")? {
                Yaml::String(s) => s.split('+').map(str::trim).collect(),
                Yaml::Array(items) => items.iter().map(|y| y.as_str().unwrap_or("")).collect(),
                _ => return err(rights_at, "expected rights such as [read, write] or `read+write`"),
            };
            entry.finish()?;
            let rights = names.iter().try_fold(Rights::NONE, |acc, name| match Rights::from_name(name) {
                Some(r) => Ok(acc | r),
                None => err(&rights_at, format!("unknown right `{name}`")),
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

    fn step(&mut self, step: &Yaml, at: &str) -> Result<()> {
        let mut step_map = Map::of(step, at)?;
        let label = step_map.optional("id").map(|y| text(y, &format!("{at}.id"))).transpose()?;
        let after = step_map.optional("after");
        let [(op_name, body)] = step_map.entries.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>()[..] else {
            return err(at, "a step has exactly one operation (plus optional `id` and `after`)");
        };
        let id = NodeId(self.nodes.len() as u32);
        let previous = self.nodes.last().map(|n| n.id);
        let after = match after {
            None => Vec::new(),
            Some(Yaml::Array(items)) => {
                items.iter().map(|y| self.reference(y, &format!("{at}.after"), previous)).collect::<Result<Vec<_>>>()?
            }
            Some(y) => vec![self.reference(y, &format!("{at}.after"), previous)?],
        };
        let at = format!("{at}.{op_name}");

        let (op, data_sources) = if op_name == "compose" {
            let expr = self.expr(body, &at, previous)?;
            let sources = expr.sources();
            (Op::Compose(expr), sources)
        } else {
            self.operation(op_name, body, &at, previous)?
        };

        for (path, needed) in op.footprint() {
            if !self.declared(path).contains(needed) {
                return err(&at, format!("{path} needs `{needed}`, which is not declared in `authority`"));
            }
        }
        for from in data_sources {
            self.edges.push(Edge { from, to: id, kind: EdgeKind::Data });
        }
        if let Some(label) = &label {
            if self.labels.insert(label.clone(), id).is_some() {
                return err(at, format!("duplicate id `{label}`"));
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
    fn operation(&self, op_name: &str, body: &Yaml, at: &str, previous: Option<NodeId>) -> Result<(Op, Vec<NodeId>)> {
        let at = at.to_owned();
        // Short form `op: /path` takes its data from the previous step.
        let (target, literal, from, uses) = match body {
            Yaml::String(_) => (path(body, &at)?, None, previous, Vec::new()),
            _ => {
                let mut body = Map::of(body, &at)?;
                let target = path(body.require("path")?, &format!("{at}.path"))?;
                let literal_key = match op_name {
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
                    Some(y) if op_name == "invoke" => compile_uses(y, &format!("{at}.uses"))?,
                    Some(_) => return err(at, "only `invoke` takes `uses`"),
                    None => Vec::new(),
                };
                body.finish()?;
                let from = match (literal, from) {
                    (Some((k, _)), Some(_)) => return err(at, format!("give either `{k}` or `from`, not both")),
                    (Some(_), None) => None,
                    (None, Some(from)) => Some(self.reference(from, &format!("{at}.from"), previous)?),
                    (None, None) => previous,
                };
                let literal = literal.map(|(k, y)| value(y, &format!("{at}.{k}"))).transpose()?;
                (target, literal, from, uses)
            }
        };

        let op = match op_name {
            "read" => Op::Read(ReadOp { target }),
            "invoke" => Op::Invoke(InvokeOp { target, input: literal, uses }),
            "write" => {
                if literal.is_none() && from.is_none() {
                    return err(at, "nothing to write: give `value` or `from`");
                }
                Op::Write(WriteOp { target, value: literal })
            }
            other => return err(at, format!("unknown operation `{other}`")),
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
    fn expr(&self, yaml: &Yaml, at: &str, previous: Option<NodeId>) -> Result<ValueExpr> {
        let map = Map::of(yaml, at)?;
        let [(constructor, body)] = map.entries.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>()[..] else {
            return err(at, "an expression has exactly one of `literal`, `select`, `map`, `list`");
        };
        let at = format!("{at}.{constructor}");
        match constructor {
            "literal" => Ok(ValueExpr::Literal(value(body, &at)?)),
            "select" => {
                let mut select = Map::of(body, &at)?;
                let source = self.reference(select.require("from")?, &format!("{at}.from"), previous)?;
                let path = match select.optional("path") {
                    None => Vec::new(),
                    Some(Yaml::Array(items)) => items
                        .iter()
                        .enumerate()
                        .map(|(i, item)| match item {
                            Yaml::String(k) => Ok(Selector::Key(k.clone())),
                            Yaml::Integer(n) if *n >= 0 => Ok(Selector::Index(*n as usize)),
                            _ => err(format!("{at}.path[{i}]"), "expected a key (text) or index (non-negative int)"),
                        })
                        .collect::<Result<_>>()?,
                    Some(_) => return err(format!("{at}.path"), "expected a list of keys and indexes"),
                };
                select.finish()?;
                Ok(ValueExpr::Select { source, path })
            }
            "map" => {
                let fields = Map::of(body, &at)?;
                let fields = fields
                    .entries
                    .iter()
                    .map(|(k, v)| Ok(((*k).to_owned(), self.expr(v, &format!("{at}.{k}"), previous)?)))
                    .collect::<Result<_>>()?;
                Ok(ValueExpr::Map(fields))
            }
            "list" => match body {
                Yaml::Array(items) => Ok(ValueExpr::List(
                    items
                        .iter()
                        .enumerate()
                        .map(|(i, item)| self.expr(item, &format!("{at}[{i}]"), previous))
                        .collect::<Result<_>>()?,
                )),
                _ => err(at, "expected a list of expressions"),
            },
            other => err(at, format!("unknown constructor `{other}`")),
        }
    }

    fn reference(&self, from: &Yaml, at: &str, previous: Option<NodeId>) -> Result<NodeId> {
        match text(from, at)?.as_str() {
            "previous" => previous.ok_or(()).or_else(|_| err(at, "no previous step")),
            label => match self.labels.get(label) {
                Some(&id) => Ok(id),
                None => err(at, format!("no earlier step with id `{label}`")),
            },
        }
    }

    fn outputs(&self, yaml: &Yaml) -> Result<Vec<GraphOutput>> {
        let map = Map::of(yaml, "outputs")?;
        map.entries
            .iter()
            .map(|(name, spec)| {
                let at = format!("outputs.{name}");
                let source = match spec {
                    Yaml::Hash(_) => {
                        let mut spec = Map::of(spec, &at)?;
                        let from = spec.require("from")?;
                        spec.finish()?;
                        from
                    }
                    other => other,
                };
                let source = self.reference(source, &format!("{at}.from"), None)?;
                Ok(GraphOutput { name: (*name).to_owned(), source })
            })
            .collect()
    }
}

/// A YAML map whose keys must all be consumed.
struct Map<'y> {
    at: String,
    entries: BTreeMap<&'y str, &'y Yaml>,
}

impl<'y> Map<'y> {
    fn of(yaml: &'y Yaml, at: &str) -> Result<Self> {
        let Yaml::Hash(hash) = yaml else {
            return err(at, "expected a map");
        };
        let mut entries = BTreeMap::new();
        for (k, v) in hash {
            match k.as_str() {
                Some(k) => entries.insert(k, v),
                None => return err(at, "map keys must be strings"),
            };
        }
        Ok(Map { at: at.to_owned(), entries })
    }

    fn optional(&mut self, key: &str) -> Option<&'y Yaml> {
        self.entries.remove(key)
    }

    fn require(&mut self, key: &str) -> Result<&'y Yaml> {
        match self.entries.remove(key) {
            Some(v) => Ok(v),
            None => err(&self.at, format!("missing `{key}`")),
        }
    }

    fn finish(self) -> Result<()> {
        match self.entries.keys().next() {
            None => Ok(()),
            Some(key) => err(self.at, format!("unknown key `{key}`")),
        }
    }
}

fn text(yaml: &Yaml, at: &str) -> Result<String> {
    match yaml {
        Yaml::String(s) => Ok(s.clone()),
        _ => err(at, "expected text"),
    }
}

fn path(yaml: &Yaml, at: &str) -> Result<Path> {
    Path::parse(&text(yaml, at)?).or_else(|e| err(at, e.to_string()))
}

fn value(yaml: &Yaml, at: &str) -> Result<Value> {
    Ok(match yaml {
        Yaml::Null => Value::Null,
        Yaml::Boolean(b) => Value::Bool(*b),
        Yaml::Integer(i) => Value::Int(*i),
        Yaml::Real(r) => Value::Float(r.parse().or_else(|_| err(at, format!("bad float `{r}`")))?),
        Yaml::String(s) => Value::Text(s.clone()),
        Yaml::Array(items) => Value::List(items.iter().map(|y| value(y, at)).collect::<Result<_>>()?),
        Yaml::Hash(_) => {
            let map = Map::of(yaml, at)?;
            Value::Map(
                map.entries
                    .iter()
                    .map(|(k, v)| Ok(((*k).to_owned(), value(v, &format!("{at}.{k}"))?)))
                    .collect::<Result<_>>()?,
            )
        }
        _ => return err(at, "unsupported yaml value"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(names, vec![("echo", 3), ("upper", 2)]);
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
        let e = compile(&ECHO.replace("rights: [write]", "rights: [read]")).unwrap_err();
        assert_eq!(e.at, "flow[1].write");
        let e = compile(&BUTCHER.replace(
            "      input:\n        from: upper",
            "      from: upper\n      uses: [{ path: /memory, rights: [read] }]",
        ))
        .unwrap_err();
        assert_eq!(e.at, "flow[1].invoke");
    }

    #[test]
    fn rejects_malformed_programs() {
        let cases = [
            (ECHO.replace("from: make_meat", "from: nowhere"), "flow[1].write.from"),
            (ECHO.replace("from: make_meat", "from: make_meat\n      value: 1"), "flow[1].write"),
            (ECHO.replace("    write:", "    exec:"), "flow[1].exec"),
            (ECHO.replace("path: /tools/echo\n      input", "path: /tools/../etc\n      input"), "flow[0].invoke.path"),
            (ECHO.replace("  name: echoer", "  name: echoer\n  shell: bash"), "agent"),
            (ECHO.replace("[invoke]", "[root]"), "authority[0].rights"),
            (ECHO.replace("- id: store", "- id: make_meat"), "flow[1].write"),
            (ECHO.replace("  - id: store\n", "  - id: store\n    read: /x\n"), "flow[1]"),
            (ECHO.replace("from: store", "from: ghost"), "outputs.result.from"),
        ];
        for (src, at) in cases {
            assert_eq!(compile(&src).unwrap_err().at, at, "{src}");
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
            assert_eq!(compile(&src).unwrap_err().at, at, "{src}");
        }
    }
}

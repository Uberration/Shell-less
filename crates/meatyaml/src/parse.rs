use crate::{Edge, Error, Graph, InvokeOp, Node, Op, Program, ReadOp, WriteOp};
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
        Some(yaml) => compile_authority(yaml)?,
        None => Vec::new(),
    };
    let steps = match top.require("flow")? {
        Yaml::Array(steps) if !steps.is_empty() => steps,
        _ => return err("flow", "expected a non-empty list of steps"),
    };
    top.finish()?;

    let mut compiler = Compiler::default();
    for (i, step) in steps.iter().enumerate() {
        compiler.step(step, &format!("flow[{i}]"), &authority)?;
    }
    Ok(Program { agent, authority, graph: Graph::new(compiler.nodes, compiler.edges) })
}

fn compile_authority(yaml: &Yaml) -> Result<Vec<(Path, Rights)>> {
    let Yaml::Array(entries) = yaml else {
        return err("authority", "expected a list of { path, rights }");
    };
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let at = format!("authority[{i}]");
            let mut entry = Map::of(entry, &at)?;
            let prefix = path(entry.require("path")?, &format!("{at}.path"))?;
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
            Ok((prefix, rights))
        })
        .collect()
}

/// Lowers steps into IR, resolving all source-level references to edges.
#[derive(Default)]
struct Compiler {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    labels: HashMap<String, NodeId>,
}

impl Compiler {
    fn step(&mut self, step: &Yaml, at: &str, authority: &[(Path, Rights)]) -> Result<()> {
        let mut step_map = Map::of(step, at)?;
        let label = step_map.optional("id").map(|y| text(y, &format!("{at}.id"))).transpose()?;
        let [(op_name, body)] = step_map.entries.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>()[..] else {
            return err(at, "a step has exactly one operation (plus an optional `id`)");
        };
        let at = format!("{at}.{op_name}");
        let id = NodeId(self.nodes.len() as u32);
        let previous = self.nodes.last().map(|n| n.id);

        // Short form `op: /path` takes its data from the previous step.
        let (target, literal, from) = match body {
            Yaml::String(_) => (path(body, &at)?, None, previous),
            _ => {
                let mut body = Map::of(body, &at)?;
                let target = path(body.require("path")?, &format!("{at}.path"))?;
                let literal_key = match op_name {
                    "write" => Some("value"),
                    "invoke" => Some("input"),
                    _ => None,
                };
                let literal = literal_key.and_then(|k| body.optional(k).map(|y| (k, y)));
                let from = body.optional("from");
                body.finish()?;
                let from = match (literal, from) {
                    (Some((k, _)), Some(_)) => return err(at, format!("give either `{k}` or `from`, not both")),
                    (Some(_), None) => None,
                    (None, Some(from)) => Some(self.reference(from, &format!("{at}.from"), previous)?),
                    (None, None) => previous,
                };
                let literal = literal.map(|(k, y)| value(y, &format!("{at}.{k}"))).transpose()?;
                (target, literal, from)
            }
        };

        let op = match op_name {
            "read" => Op::Read(ReadOp { target }),
            "invoke" => Op::Invoke(InvokeOp { target, input: literal }),
            "write" => {
                if literal.is_none() && from.is_none() {
                    return err(at, "nothing to write: give `value` or `from`");
                }
                Op::Write(WriteOp { target, value: literal })
            }
            other => return err(at, format!("unknown operation `{other}`")),
        };

        let (target, needed) = (op.target(), op.rights());
        let declared = authority
            .iter()
            .filter(|(prefix, _)| target.starts_with(prefix))
            .fold(Rights::NONE, |acc, (_, r)| acc | *r);
        if !declared.contains(needed) {
            return err(at, format!("{target} needs `{needed}`, which is not declared in `authority`"));
        }

        // Reads take no data input; everything else is fed by `from`, if any.
        if let (Some(from), false, None) = (from, matches!(op, Op::Read(_)), op.literal()) {
            self.edges.push(Edge { from, to: id });
        }
        if let Some(label) = &label {
            if self.labels.insert(label.clone(), id).is_some() {
                return err(at, format!("duplicate id `{label}`"));
            }
        }
        self.nodes.push(Node { id, label, op });
        Ok(())
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

    /// The Milestone 2 acceptance program.
    const ECHO: &str = r#"
agent:
  name: echoer

authority:
  - path: /tools/echo
    rights: [invoke]

  - path: /state/result
    rights: [write]

flow:
  - id: make_meat
    invoke:
      path: /tools/echo
      input:
        text: MEAT

  - id: store
    write:
      path: /state/result
      from: make_meat
"#;

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    #[test]
    fn compiles_to_ir_with_explicit_edges() {
        let program = compile(ECHO).unwrap();
        let g = &program.graph;
        assert_eq!(
            g.nodes.iter().map(|n| n.op.clone()).collect::<Vec<_>>(),
            vec![
                Op::Invoke(InvokeOp {
                    target: p("/tools/echo"),
                    input: Some(Value::map([("text", Value::from("MEAT"))])),
                }),
                Op::Write(WriteOp { target: p("/state/result"), value: None }),
            ]
        );
        assert_eq!(g.edges, vec![Edge { from: NodeId(0), to: NodeId(1) }]);
        assert_eq!(g.input_of(NodeId(1)), Some(NodeId(0)));
        assert_eq!(
            program.request().wants,
            vec![(p("/tools/echo"), Rights::INVOKE), (p("/state/result"), Rights::WRITE)]
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
        assert_eq!(a, relabelled);
        assert_ne!(a, changed);
    }

    #[test]
    fn literal_inputs_take_no_edge() {
        let program = compile(
            r#"
agent: { name: a }
authority:
  - { path: /memory, rights: [read] }
  - { path: /state/a, rights: read+write }
flow:
  - read: /memory/context
  - write: { path: /state/a/x, value: 1 }
  - write: { path: /state/a/y, from: previous }
"#,
        )
        .unwrap();
        assert_eq!(program.graph.edges, vec![Edge { from: NodeId(1), to: NodeId(2) }]);
    }

    #[test]
    fn rejects_undeclared_authority() {
        let e = compile(&ECHO.replace("rights: [write]", "rights: [read]")).unwrap_err();
        assert_eq!(e.at, "flow[1].write");
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
        ];
        for (src, at) in cases {
            assert_eq!(compile(&src).unwrap_err().at, at, "{src}");
        }
    }
}

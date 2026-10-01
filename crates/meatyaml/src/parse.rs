use crate::{Error, Graph, Node, Op, Program, Source};
use meatfs::{Grant, Path, Rights, Value};
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

    let mut grants = Vec::new();
    for section in ["input", "output"] {
        if let Some(yaml) = top.optional(section) {
            grants.extend(compile_grants(yaml, section)?);
        }
    }

    let flow = match top.require("flow")? {
        Yaml::Array(steps) if !steps.is_empty() => steps,
        _ => return err("flow", "expected a non-empty list of steps"),
    };
    top.finish()?;

    let mut graph = Graph::default();
    let mut ids = HashMap::new();
    for (i, step) in flow.iter().enumerate() {
        let at = format!("flow[{i}]");
        let node = compile_step(step, &at, i, &ids)?;
        if let Some(id) = &node.id {
            if ids.insert(id.clone(), i).is_some() {
                return err(at, format!("duplicate id `{id}`"));
            }
        }
        let (path, needed) = (node.op.path(), node.op.rights());
        let covered = grants
            .iter()
            .filter(|g: &&Grant| path.starts_with(&g.prefix))
            .fold(Rights::NONE, |acc, g| acc | g.rights)
            .contains(needed);
        if !covered {
            return err(format!("{at}.{}", node.op.name()), format!("{path} needs `{needed}`, which is not declared"));
        }
        graph.nodes.push(node);
    }

    Ok(Program { agent, grants, graph })
}

fn compile_grants(yaml: &Yaml, section: &str) -> Result<Vec<Grant>> {
    let Yaml::Hash(entries) = yaml else {
        return err(section, "expected a map of path: rights");
    };
    entries
        .iter()
        .map(|(key, rights)| {
            let at = format!("{section}.{}", key.as_str().unwrap_or("?"));
            let prefix = path(key, &at)?;
            let names: Vec<&str> = match rights {
                Yaml::String(s) => s.split('+').map(str::trim).collect(),
                Yaml::Array(items) => items.iter().map(|y| y.as_str().unwrap_or("")).collect(),
                _ => return err(at, "expected rights such as `read`, `read+write` or a list"),
            };
            let rights = names.iter().try_fold(Rights::NONE, |acc, name| match Rights::from_name(name) {
                Some(r) => Ok(acc | r),
                None => err(&at, format!("unknown right `{name}`")),
            })?;
            Ok(Grant { prefix, rights })
        })
        .collect()
}

fn compile_step(step: &Yaml, at: &str, index: usize, ids: &HashMap<String, usize>) -> Result<Node> {
    let Yaml::Hash(entries) = step else {
        return err(at, "expected a step such as `read: /path`");
    };
    let [(op, body)] = entries.iter().collect::<Vec<_>>()[..] else {
        return err(at, "a step has exactly one operation");
    };
    let op_name = op.as_str().unwrap_or("?");
    let at = format!("{at}.{op_name}");
    let previous = index.checked_sub(1).map(Source::Node);

    // Short form: `op: /path`.
    if let Yaml::String(_) = body {
        let path = path(body, &at)?;
        let op = match op_name {
            "read" => Op::Read { path },
            "invoke" => Op::Invoke { path, input: previous.unwrap_or(Source::Literal(Value::Null)) },
            "write" => match previous {
                Some(value) => Op::Write { path, value },
                None => return err(at, "nothing to write: no previous step"),
            },
            other => return err(at, format!("unknown operation `{other}`")),
        };
        return Ok(Node { id: None, op });
    }

    let mut body = Map::of(body, &at)?;
    let path = path(body.require("path")?, &format!("{at}.path"))?;
    let id = body.optional("id").map(|y| text(y, &format!("{at}.id"))).transpose()?;
    let op = match op_name {
        "read" => Op::Read { path },
        "invoke" => {
            let input = data_source(&mut body, "input", &at, previous, ids)?;
            Op::Invoke { path, input: input.unwrap_or(Source::Literal(Value::Null)) }
        }
        "write" => match data_source(&mut body, "value", &at, previous, ids)? {
            Some(value) => Op::Write { path, value },
            None => return err(at, "nothing to write: give `value` or `from`"),
        },
        other => return err(at, format!("unknown operation `{other}`")),
    };
    body.finish()?;
    Ok(Node { id, op })
}

/// Resolve a step's data from a literal `<literal_key>` or a `from` reference,
/// defaulting to the previous step.
fn data_source(
    body: &mut Map<'_>,
    literal_key: &str,
    at: &str,
    previous: Option<Source>,
    ids: &HashMap<String, usize>,
) -> Result<Option<Source>> {
    match (body.optional(literal_key), body.optional("from")) {
        (Some(_), Some(_)) => err(at, format!("give either `{literal_key}` or `from`, not both")),
        (Some(literal), None) => Ok(Some(Source::Literal(value(literal, &format!("{at}.{literal_key}"))?))),
        (None, Some(from)) => {
            let at = format!("{at}.from");
            match text(from, &at)?.as_str() {
                "previous" => previous.map(Some).ok_or(()).or_else(|_| err(at, "no previous step")),
                id => match ids.get(id) {
                    Some(&i) => Ok(Some(Source::Node(i))),
                    None => err(at, format!("no earlier step with id `{id}`")),
                },
            }
        }
        (None, None) => Ok(previous),
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

    const ECHO: &str = r#"
agent:
  name: echoer
input:
  /tools/echo: invoke
output:
  /state/result: write
flow:
  - invoke:
      path: /tools/echo
      input:
        text: "MEAT"
  - write:
      path: /state/result
      from: previous
"#;

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    #[test]
    fn compiles_the_canonical_example() {
        let program = compile(ECHO).unwrap();
        assert_eq!(program.agent, "echoer");
        assert_eq!(
            program.graph.nodes.iter().map(|n| n.op.clone()).collect::<Vec<_>>(),
            vec![
                Op::Invoke {
                    path: p("/tools/echo"),
                    input: Source::Literal(Value::map([("text", Value::from("MEAT"))])),
                },
                Op::Write { path: p("/state/result"), value: Source::Node(0) },
            ]
        );
        let auth = program.authority();
        assert!(auth.allows(&p("/state/result"), Rights::WRITE));
        assert!(!auth.allows(&p("/state/result"), Rights::READ));
    }

    #[test]
    fn short_forms_and_named_references() {
        let program = compile(
            r#"
agent: { name: scout }
input:
  /memory/context: read
  /tools: invoke
output:
  /state/scout: write
flow:
  - read: { path: /memory/context, id: ctx }
  - invoke: /tools/web/search
  - write: /state/scout/report
  - write: { path: /state/scout/context, from: ctx }
"#,
        )
        .unwrap();
        let ops: Vec<_> = program.graph.nodes.into_iter().map(|n| n.op).collect();
        assert_eq!(ops[1], Op::Invoke { path: p("/tools/web/search"), input: Source::Node(0) });
        assert_eq!(ops[2], Op::Write { path: p("/state/scout/report"), value: Source::Node(1) });
        assert_eq!(ops[3], Op::Write { path: p("/state/scout/context"), value: Source::Node(0) });
    }

    #[test]
    fn rejects_undeclared_authority() {
        let e = compile(&ECHO.replace("/state/result: write", "/state/result: read")).unwrap_err();
        assert_eq!(e.at, "flow[1].write");
    }

    #[test]
    fn rejects_malformed_programs() {
        let cases = [
            (ECHO.replace("from: previous", "from: nowhere"), "flow[1].write.from"),
            (ECHO.replace("from: previous", "from: previous\n      value: 1"), "flow[1].write"),
            (ECHO.replace("- write:", "- exec:"), "flow[1].exec"),
            (ECHO.replace("path: /tools/echo", "path: /tools/../etc"), "flow[0].invoke.path"),
            (ECHO.replace("  name: echoer", "  name: echoer\n  shell: bash"), "agent"),
            (ECHO.replace("/tools/echo: invoke", "/tools/echo: root"), "input./tools/echo"),
        ];
        for (src, at) in cases {
            assert_eq!(compile(&src).unwrap_err().at, at, "{src}");
        }
    }
}

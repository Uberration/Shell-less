//! Executes compiled MEATYAML graphs against MeatFS.
//!
//! The runtime holds no authority of its own: every operation runs under the
//! program's declared [`Authority`](meatfs::Authority), so MeatFS enforces at
//! run time exactly what `meatyaml` validated at compile time. Today nodes
//! run sequentially and locally; the graph is the unit later handed to
//! Hydra for placement.

use meatfs::{MeatFs, Path, Value};
use meatyaml::{Op, Program, Source};
use std::fmt;

/// The record of one executed node.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub op: &'static str,
    pub path: Path,
    pub output: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    pub step: usize,
    pub source: meatfs::Error,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "step {}: {}", self.step, self.source)
    }
}

impl std::error::Error for Error {}

/// Run `program` to completion, returning one [`Step`] per graph node.
pub fn execute(fs: &MeatFs, program: &Program) -> Result<Vec<Step>, Error> {
    let auth = program.authority();
    let mut trace: Vec<Step> = Vec::with_capacity(program.graph.nodes.len());
    for (step, node) in program.graph.nodes.iter().enumerate() {
        let resolve = |source: &Source| match source {
            Source::Literal(value) => value.clone(),
            Source::Node(i) => trace[*i].output.clone(),
        };
        let output = match &node.op {
            Op::Read { path } => fs.read(&auth, path),
            Op::Invoke { path, input } => fs.invoke(&auth, path, resolve(input)),
            Op::Write { path, value } => {
                let value = resolve(value);
                fs.write(&auth, path, value.clone()).map(|_| value)
            }
        }
        .map_err(|source| Error { step, source })?;
        trace.push(Step { op: node.op.name(), path: node.op.path().clone(), output });
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::builtin::{Echo, Upper};
    use meatfs::Authority;

    fn p(s: &str) -> Path {
        Path::parse(s).unwrap()
    }

    fn host() -> (MeatFs, Authority) {
        let fs = MeatFs::new();
        let root = Authority::root("host");
        capability::mount(&fs, &root, &p("/tools/echo"), Echo).unwrap();
        capability::mount(&fs, &root, &p("/tools/text/upper"), Upper).unwrap();
        (fs, root)
    }

    #[test]
    fn executes_into_the_filesystem() {
        let (fs, root) = host();
        let program = meatyaml::compile(
            r#"
agent: { name: shouter }
input: { /tools: invoke }
output: { /state/shouter: write }
flow:
  - invoke: { path: /tools/echo, input: { text: "meat" } }
  - invoke: /tools/text/upper
  - write: /state/shouter/result
"#,
        )
        .unwrap();
        let trace = execute(&fs, &program).unwrap();
        assert_eq!(trace.len(), 3);
        let expected = Value::map([("text", Value::from("MEAT"))]);
        assert_eq!(fs.read(&root, &p("/state/shouter/result")).unwrap(), expected);
    }

    #[test]
    fn missing_capability_fails_at_its_step() {
        let (fs, _) = host();
        let program =
            meatyaml::compile("agent: { name: a }\ninput: { /tools: invoke }\nflow:\n  - invoke: /tools/nope\n")
                .unwrap();
        let e = execute(&fs, &program).unwrap_err();
        assert_eq!(e, Error { step: 0, source: meatfs::Error::NotFound(p("/tools/nope")) });
    }
}

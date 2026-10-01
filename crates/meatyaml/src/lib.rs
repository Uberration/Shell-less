//! MEATYAML: source code for executable composition.
//!
//! A MEATYAML program is compiled — not interpreted — into a validated
//! [`Program`]: a typed execution graph plus the exact authority it requires.
//! Validation is static and strict: unknown keys are errors, data references
//! must point backwards, and every path the flow touches must be covered by
//! the program's declared `input`/`output` authority.
//!
//! ```yaml
//! agent:
//!   name: echoer
//! input:
//!   /tools/echo: invoke
//! output:
//!   /state/result: write
//! flow:
//!   - invoke:
//!       path: /tools/echo
//!       input:
//!         text: "MEAT"
//!   - write:
//!       path: /state/result
//!       from: previous
//! ```

mod parse;

use meatfs::{Authority, Grant, Path, Rights, Value};
use std::fmt;

pub use parse::compile;

/// A compiled MEATYAML program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub agent: String,
    /// Declared authority: the only rights the program will ever hold.
    pub grants: Vec<Grant>,
    pub graph: Graph,
}

impl Program {
    /// The authority this program executes under. Nothing ambient is added.
    pub fn authority(&self) -> Authority {
        self.grants.iter().fold(Authority::new(&self.agent), |auth, g| auth.grant(g.prefix.clone(), g.rights))
    }
}

/// Execution graph. Nodes are stored in a valid execution order and every
/// [`Source::Node`] refers to an earlier node, so the graph is acyclic by
/// construction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Optional name other nodes may reference with `from: <id>`.
    pub id: Option<String>,
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Read { path: Path },
    Write { path: Path, value: Source },
    Invoke { path: Path, input: Source },
}

impl Op {
    pub fn path(&self) -> &Path {
        match self {
            Op::Read { path } | Op::Write { path, .. } | Op::Invoke { path, .. } => path,
        }
    }

    pub fn rights(&self) -> Rights {
        match self {
            Op::Read { .. } => Rights::READ,
            Op::Write { .. } => Rights::WRITE,
            Op::Invoke { .. } => Rights::INVOKE,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Op::Read { .. } => "read",
            Op::Write { .. } => "write",
            Op::Invoke { .. } => "invoke",
        }
    }
}

/// Where a node's data comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    Literal(Value),
    /// The output of the node at this index.
    Node(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error {
    /// Location within the program, e.g. `flow[1].write.from`.
    pub at: String,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.at.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.at, self.message)
        }
    }
}

impl std::error::Error for Error {}

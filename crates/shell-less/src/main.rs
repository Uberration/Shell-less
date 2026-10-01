//! `shell-less`: MEATYAML → graph → capability resolution → execution →
//! filesystem-visible result.
//!
//! ```text
//! shell-less check <program.meat.yaml>   compile and validate only
//! shell-less run   <program.meat.yaml>   compile, execute, show the namespace
//! ```

use capability::builtin::{Echo, Upper};
use meatfs::{Authority, EventKind, MeatFs, NodeKind, Path, Value};
use meatyaml::{Op, Program, Source};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, file) = match args.as_slice() {
        [c, f] if c == "run" || c == "check" => (c.as_str(), f),
        _ => {
            eprintln!("usage: shell-less <run|check> <program.meat.yaml>");
            return ExitCode::from(2);
        }
    };
    match drive(command, file) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn drive(command: &str, file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;

    let program = meatyaml::compile(&source)?;
    section("graph");
    print_graph(&program);
    if command == "check" {
        return Ok(());
    }

    let host = Authority::root("host");
    let fs = boot(&host)?;

    section("resolution");
    for node in &program.graph.nodes {
        if let Op::Invoke { path, .. } = &node.op {
            let signature = fs.inspect(&host, path).map(|i| i.signature);
            match signature {
                Ok(sig) => println!("  {path}  {sig}"),
                Err(e) => println!("  {path}  unresolved ({e})"),
            }
        }
    }

    section("execution");
    let trace = runtime::execute(&fs, &program)?;
    for (i, step) in trace.iter().enumerate() {
        println!("  [{i}] {:<6} {}  → {}", step.op, step.path, step.output);
    }

    section("namespace");
    for (path, kind) in fs.walk(&host, &Path::root())? {
        match kind {
            NodeKind::Data => println!("  {path} = {}", fs.read(&host, &path)?),
            NodeKind::Capability => println!("  {path}  <capability>"),
            NodeKind::Dir => println!("  {path}/"),
        }
    }

    section("audit");
    for event in fs.journal(&host, &Path::root())? {
        let what = match event.kind {
            EventKind::Written { version } => format!("write v{version}"),
            EventKind::Mounted => "mount".to_owned(),
            EventKind::Invoked { ok } => format!("invoke {}", if ok { "ok" } else { "failed" }),
        };
        println!("  #{:<3} {:<8} {:<14} {}", event.seq, event.principal, what, event.path);
    }
    Ok(())
}

/// The host namespace: native capabilities and seed memory.
fn boot(host: &Authority) -> meatfs::Result<MeatFs> {
    let fs = MeatFs::new();
    capability::mount(&fs, host, &Path::parse("/tools/echo")?, Echo)?;
    capability::mount(&fs, host, &Path::parse("/tools/text/upper")?, Upper)?;
    fs.write(host, &Path::parse("/memory/context")?, Value::map([("text", Value::from("shell-less boot"))]))?;
    Ok(fs)
}

fn print_graph(program: &Program) {
    println!("  agent {}", program.agent);
    for grant in &program.grants {
        println!("  grant {} {}", grant.prefix, grant.rights);
    }
    for (i, node) in program.graph.nodes.iter().enumerate() {
        let data = match &node.op {
            Op::Read { .. } => String::new(),
            Op::Invoke { input: s, .. } | Op::Write { value: s, .. } => match s {
                Source::Literal(v) => format!(" ← {v}"),
                Source::Node(n) => format!(" ← [{n}]"),
            },
        };
        let id = node.id.as_deref().map(|id| format!(" ({id})")).unwrap_or_default();
        println!("  [{i}] {:<6} {}{data}{id}", node.op.name(), node.op.path());
    }
}

fn section(name: &str) {
    println!("── {name}");
}

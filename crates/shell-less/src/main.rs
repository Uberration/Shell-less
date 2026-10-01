//! `shell-less`: MEATYAML → MEAT IR → authority resolution → resolved
//! execution graph → MeatFS objects → execution receipt.
//!
//! ```text
//! shell-less check <program.meat.yaml>              compile and validate only
//! shell-less run [--seed N] <program.meat.yaml>     load, execute, print the receipt
//! ```

use capability::builtin::{Echo, Upper};
use meatfs::{EventKind, GrantSet, MeatFs, NodeKind, ObjectId, Path, Policy, Rights, Seed, Value};
use meatyaml::Program;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["check", file] => Some(("check", None, file)),
        ["run", file] => Some(("run", None, file)),
        ["run", "--seed", n, file] => n.parse().ok().map(|n| ("run", Some(n), file)),
        _ => None,
    };
    let Some((command, seed, file)) = parsed else {
        eprintln!("usage: shell-less check <program.meat.yaml>\n       shell-less run [--seed N] <program.meat.yaml>");
        return ExitCode::from(2);
    };
    match drive(command, seed, file) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn drive(command: &str, seed: Option<u64>, file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let program = meatyaml::compile(&source)?;
    section("MEAT IR");
    print_ir(&program);
    if command == "check" {
        return Ok(());
    }

    let (fs, host) = boot(seed.map_or_else(Seed::entropy, Seed::fixed))?;
    let loaded = runtime::load(&fs, &policy()?, &program)?;

    section("resolved");
    for (node, binding) in loaded.graph().nodes.iter().zip(loaded.bindings()) {
        let rights = loaded.grants().get(binding.grant).map(|g| g.rights()).unwrap_or_default();
        println!(
            "  {}  {:<6} {}  {}  {} [{rights}]",
            node.id,
            node.op.name(),
            node.op.target(),
            binding.object.short(),
            binding.grant.short()
        );
    }

    let receipt = runtime::execute(&fs, &loaded)?;
    section("receipt");
    println!("  execution  {}", receipt.execution);
    println!("  graph      {}", receipt.graph);
    println!("  principal  {} ({})", receipt.principal, program.agent);
    for g in &receipt.grants {
        println!("  grant      {}  {} [{}]", g.grant.short(), g.object.short(), g.rights);
    }
    for n in &receipt.nodes {
        println!("  node       {}  {:<6} {}  {}  → {}", n.node, n.op, n.object.short(), n.grant.short(), n.output);
    }
    for e in &receipt.events {
        let node = e.cause.map(|c| c.node.to_string()).unwrap_or_default();
        let grant = e.grant.map(|g| g.short()).unwrap_or_default();
        println!("  event      #{:<3} {node}  {}  {grant}  {}", e.seq, e.object.short(), kind(&e.kind));
    }
    println!("  result     {}", receipt.result);

    section("namespace");
    let root = root(&host);
    for (path, object, node_kind) in fs.walk(&root, &Path::root())? {
        match node_kind {
            NodeKind::Data => println!("  {}  {path} = {}", object.short(), read_or_empty(&fs, &host, object)),
            NodeKind::Capability => println!("  {}  {path}  <capability>", object.short()),
        }
    }
    Ok(())
}

/// The host namespace: native capabilities and seed memory.
fn boot(seed: Seed) -> meatfs::Result<(MeatFs, GrantSet)> {
    let (fs, host) = MeatFs::genesis(seed);
    let root = root(&host);
    capability::mount(&fs, &root, &Path::parse("/tools/echo")?, Echo)?;
    capability::mount(&fs, &root, &Path::parse("/tools/text/upper")?, Upper)?;
    fs.bind(&root, &Path::parse("/memory/context")?, Value::map([("text", Value::from("shell-less boot"))]))?;
    Ok((fs, host))
}

/// What agents may request from this host.
fn policy() -> meatfs::Result<Policy> {
    Ok(Policy::default()
        .allow(Path::parse("/tools")?, Rights::INVOKE | Rights::INSPECT)
        .allow(Path::parse("/memory")?, Rights::READ)
        .allow(Path::parse("/state")?, Rights::READ | Rights::WRITE))
}

fn root(host: &GrantSet) -> meatfs::Access<'_> {
    host.access(host.iter().next().expect("genesis issues the root grant").id()).expect("grant is in its set")
}

fn read_or_empty(fs: &MeatFs, host: &GrantSet, object: ObjectId) -> String {
    match fs.read(&root(host), object) {
        Ok(v) => v.to_string(),
        Err(meatfs::Error::Empty(_)) => "<empty>".to_owned(),
        Err(e) => format!("<{e}>"),
    }
}

fn print_ir(program: &Program) {
    let graph = &program.graph;
    println!("  agent  {}", program.agent);
    println!("  graph  {}", graph.id);
    for (path, rights) in &program.authority {
        println!("  declare {path} [{rights}]");
    }
    for node in &graph.nodes {
        let literal = node.op.literal().map(|v| format!(" ← {v}")).unwrap_or_default();
        let label = node.label.as_deref().map(|l| format!("  ({l})")).unwrap_or_default();
        println!("  {}  {:<6} {}{literal}{label}", node.id, node.op.name(), node.op.target());
    }
    for edge in &graph.edges {
        println!("  edge   {} → {}", edge.from, edge.to);
    }
}

fn kind(kind: &EventKind) -> String {
    match kind {
        EventKind::Bound(path) => format!("bound {path}"),
        EventKind::Read { version } => format!("read v{version}"),
        EventKind::Written { version } => format!("write v{version}"),
        EventKind::Invoked { ok } => format!("invoke {}", if *ok { "ok" } else { "failed" }),
    }
}

fn section(name: &str) {
    println!("── {name}");
}

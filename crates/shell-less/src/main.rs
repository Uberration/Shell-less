//! `shell-less`: MEATYAML → MEAT IR → authority resolution → resolved
//! execution graph → transactional execution → execution receipt.
//!
//! ```text
//! shell-less check <program.meat.yaml>              compile and validate only
//! shell-less run [--seed N] <program.meat.yaml>     load, execute, print the receipt
//! ```

use capability::builtin::{Echo, Fail, Upper};
use meatfs::{EventKind, GrantSet, MeatFs, NodeKind, Path, Policy, Rights, Seed, Target, Value};
use meatyaml::{EdgeKind, Program};
use runtime::{ExecutionReceipt, ResolvedTarget};
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
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Returns whether the execution committed.
fn drive(command: &str, seed: Option<u64>, file: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let program = meatyaml::compile(&source)?;
    section("MEAT IR");
    print_ir(&program);
    if command == "check" {
        return Ok(true);
    }

    let (fs, host) = boot(seed.map_or_else(Seed::entropy, Seed::fixed))?;
    let loaded = runtime::load(&fs, &policy()?, &program)?;

    section("resolved");
    for (node, resolved) in loaded.graph().nodes.iter().zip(loaded.nodes()) {
        let target = match &resolved.target {
            ResolvedTarget::Object(o) => o.short(),
            ResolvedTarget::New(_) => "<new>".to_owned(),
        };
        let uses: String = resolved.uses.iter().map(|u| format!("  +{} {}", u.path, u.grant.short())).collect();
        println!(
            "  {}  {:<6} {}  {target}  {}{uses}",
            node.id,
            node.op.name(),
            node.op.target(),
            resolved.grant.short()
        );
    }

    let receipt = runtime::execute(&fs, &loaded);
    section("receipt");
    print_receipt(&program, &receipt);

    section("namespace");
    let root = root(&host);
    for (path, object, kind) in fs.walk(&root, &Path::root())? {
        match kind {
            NodeKind::Data => println!("  {}  {path} = {}", object.short(), fs.read(&root, object)?),
            NodeKind::Capability => println!("  {}  {path}  <capability>", object.short()),
        }
    }
    Ok(receipt.succeeded())
}

/// The host namespace: native capabilities and seed memory.
fn boot(seed: Seed) -> meatfs::Result<(MeatFs, GrantSet)> {
    let (fs, host) = MeatFs::genesis(seed);
    let root = root(&host);
    capability::mount(&fs, &root, &Path::parse("/tools/echo")?, Echo)?;
    capability::mount(&fs, &root, &Path::parse("/tools/text/upper")?, Upper)?;
    capability::mount(&fs, &root, &Path::parse("/tools/test/fail")?, Fail)?;
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
        let uses: String = node.op.uses().iter().map(|u| format!("  +{} [{}]", u.path, u.rights)).collect();
        println!("  {}  {:<6} {}{literal}{uses}{label}", node.id, node.op.name(), node.op.target());
    }
    for edge in &graph.edges {
        let kind = match edge.kind {
            EdgeKind::Data => "data ",
            EdgeKind::Order => "order",
        };
        println!("  {kind}  {} → {}", edge.from, edge.to);
    }
    for output in &graph.outputs {
        println!("  output {} ← {}", output.name, output.source);
    }
}

fn print_receipt(program: &Program, receipt: &ExecutionReceipt) {
    println!("  execution  {}", receipt.execution);
    println!("  graph      {}", receipt.graph);
    println!("  principal  {} ({})", receipt.principal, program.agent);
    for g in &receipt.grants {
        let target = match &g.target {
            Target::Object(o) => o.short(),
            Target::Name(p) => format!("name {p}"),
            Target::Namespace(p) => format!("namespace {p}"),
        };
        println!("  grant      {}  {target} [{}]", g.grant.short(), g.rights);
    }
    let schedule: Vec<_> = receipt.schedule.iter().map(ToString::to_string).collect();
    println!("  schedule   {}", schedule.join(" → "));
    for n in &receipt.nodes {
        let object = n.object.map(|o| o.short()).unwrap_or_else(|| "-".to_owned());
        let detail = match (&n.output, &n.error) {
            (_, Some(e)) => format!("{:?}", e.kind),
            (Some(v), None) => format!("→ {v}"),
            (None, None) => String::new(),
        };
        println!(
            "  node       {}  {:<6} {:<9} {object}  {}  {detail}",
            n.node,
            n.op,
            format!("{:?}", n.state),
            n.grant.short()
        );
    }
    for e in &receipt.events {
        let node = e.cause.map(|c| c.node.to_string()).unwrap_or_default();
        let grant = e.grant.map(|g| g.short()).unwrap_or_default();
        println!("  event      #{:<3} {node}  {}  {grant}  {}", e.seq, e.object.short(), kind(&e.kind));
    }
    println!("  transaction {:?}", receipt.transaction);
    if let Some(e) = &receipt.error {
        println!("  error      {e}");
    }
    match receipt.outputs.get("result") {
        Some(result) if receipt.outputs.len() == 1 => println!("  result     {result}"),
        _ => {
            for (name, value) in &receipt.outputs {
                println!("  output     {name} = {value}");
            }
        }
    }
}

fn kind(kind: &EventKind) -> String {
    match kind {
        EventKind::Bound(path) => format!("bound {path}"),
        EventKind::Read { version } => format!("read v{version}"),
        EventKind::Staged => "staged".to_owned(),
        EventKind::Written { version } => format!("written v{version}"),
        EventKind::RolledBack => "rolled back".to_owned(),
        EventKind::Invoked { ok } => format!("invoke {}", if *ok { "ok" } else { "failed" }),
    }
}

fn section(name: &str) {
    println!("── {name}");
}

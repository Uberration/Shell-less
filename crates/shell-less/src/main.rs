//! `shell-less`: MEATYAML → MEAT IR → authority resolution → resolved
//! execution graph → transactional execution → execution receipt.
//!
//! ```text
//! shell-less check [--capture inline] <program.meat.yaml>
//! shell-less run   [--seed N] [--capture inline] [--show-outputs] <program.meat.yaml>
//! ```
//!
//! The capture policy is chosen before the program is parsed and governs
//! every diagnostic: compile errors, load errors, the IR dump, the receipt
//! and the namespace dump. Under the default (`omit`), none of them prints
//! source text (names, paths, keys, labels, literals) or payloads; they keep
//! kinds, locations, opaque identities, structure and outcomes. Graph
//! results are printed only on `--show-outputs`, which does not reveal
//! diagnostic content.

use capability::builtin::{Echo, Fail, Upper};
use meatfs::{EventKind, GrantSet, MeatFs, NodeKind, Path, Policy, Rights, Seed, Value};
use meatyaml::{EdgeKind, Op, Program, Selector, ValueExpr};
use runtime::{CapturedValue, ContentCapture, ExecutionReceipt, Limits, RecordedTarget, ResolvedTarget};
use std::fmt::Display;
use std::process::ExitCode;

const USAGE: &str = "usage: shell-less check [--capture omit|inline] <program.meat.yaml>
       shell-less run [--seed N] [--capture omit|inline] [--show-outputs] <program.meat.yaml>";

struct Args {
    command: String,
    file: String,
    seed: Option<u64>,
    capture: ContentCapture,
    show_outputs: bool,
}

fn parse(args: &[String]) -> Option<Args> {
    let (command, rest) = args.split_first()?;
    if command != "run" && command != "check" {
        return None;
    }
    let mut parsed = Args {
        command: command.clone(),
        file: String::new(),
        seed: None,
        capture: ContentCapture::Omit,
        show_outputs: false,
    };
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--seed" if command == "run" => parsed.seed = Some(rest.next()?.parse().ok()?),
            "--show-outputs" if command == "run" => parsed.show_outputs = true,
            "--capture" => {
                parsed.capture = match rest.next()?.as_str() {
                    "omit" => ContentCapture::Omit,
                    "inline" => ContentCapture::Inline,
                    _ => return None,
                }
            }
            file if parsed.file.is_empty() && !file.starts_with("--") => parsed.file = file.to_owned(),
            _ => return None,
        }
    }
    (!parsed.file.is_empty()).then_some(parsed)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(args) = parse(&args) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match drive(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(rendered) => {
            eprintln!("error: {rendered}");
            ExitCode::FAILURE
        }
    }
}

/// What diagnostics may show, decided by the host's capture policy.
#[derive(Clone, Copy)]
struct Show(ContentCapture);

impl Show {
    fn reveal(self) -> bool {
        self.0 == ContentCapture::Inline
    }

    fn redact(self, text: impl Display, placeholder: &str) -> String {
        if self.reveal() {
            text.to_string()
        } else {
            placeholder.to_owned()
        }
    }

    /// A namespace path written in, or derived from, the program.
    fn path(self, path: impl Display) -> String {
        self.redact(path, "<path>")
    }

    /// Any other source-controlled string: names, keys, labels.
    fn name(self, name: impl Display) -> String {
        self.redact(name, "<name>")
    }

    fn value(self, value: &Value) -> String {
        self.redact(value, "<omitted>")
    }

    fn captured(self, value: &CapturedValue) -> String {
        match value {
            CapturedValue::Omitted => "<omitted>".to_owned(),
            CapturedValue::Inline(v) => v.to_string(),
        }
    }
}

/// Returns whether the execution committed. Errors come back already
/// rendered under the capture policy.
fn drive(args: &Args) -> Result<bool, String> {
    let show = Show(args.capture);
    // The file name is a host argument, not program content.
    let source = std::fs::read_to_string(&args.file).map_err(|e| format!("{}: {e}", args.file))?;
    let program = meatyaml::compile_with(&source, args.capture).map_err(|e| e.render())?;
    section("MEAT IR");
    print_ir(&program, show);
    if args.command == "check" {
        return Ok(true);
    }

    let host_error = |e: meatfs::Error| format!("host: {}", e.redacted());
    let (fs, host) = boot(args.seed.map_or_else(Seed::entropy, Seed::fixed), args.capture).map_err(host_error)?;
    let policy = policy().map_err(host_error)?;
    let loaded = runtime::load(&fs, &policy, Limits::default(), args.capture, &program).map_err(|e| e.render())?;

    section("resolved");
    for (node, resolved) in loaded.graph().nodes.iter().zip(loaded.nodes()) {
        let target = match &resolved.target {
            ResolvedTarget::Object(o) => o.short(),
            ResolvedTarget::New(_) => "<new>".to_owned(),
            ResolvedTarget::None => "-".to_owned(),
        };
        let grant = resolved.grant.map(|g| g.short()).unwrap_or_else(|| "-".to_owned());
        let uses: String =
            resolved.uses.iter().map(|u| format!("  +{} {}", show.path(&u.path), u.grant.short())).collect();
        println!("  {}  {:<7} {}  {target}  {grant}{uses}", node.id, node.op.name(), describe_target(&node.op, show));
    }
    for (from, to) in loaded.derived_order() {
        println!("  derived order  {from} → {to}");
    }

    let outcome = runtime::execute(&fs, &loaded, args.capture);
    section("receipt");
    print_receipt(&program, &outcome.receipt, show);

    section("namespace");
    let root = root(&host);
    let walk = fs.walk(&root, &Path::root()).map_err(host_error)?;
    for (path, object, kind) in walk {
        match kind {
            NodeKind::Data => {
                let value = fs.read(&root, object).map_err(host_error)?;
                println!("  {}  {} = {}", object.short(), show.path(&path), show.value(&value));
            }
            NodeKind::Capability => println!("  {}  {}  <capability>", object.short(), show.path(&path)),
        }
    }

    // The result channel: explicit, and independent of diagnostic capture.
    if args.show_outputs {
        section("outputs");
        for (name, value) in &outcome.outputs {
            println!("  {name} = {value}");
        }
    }
    Ok(outcome.receipt.succeeded())
}

/// The host namespace: native capabilities, mock models and seed memory.
/// The journal records under the same capture policy as everything else.
fn boot(seed: Seed, capture: ContentCapture) -> meatfs::Result<(MeatFs, GrantSet)> {
    let (fs, host) = MeatFs::genesis(seed);
    let root = root(&host);
    fs.set_journal_capture(&root, capture)?;
    capability::mount(&fs, &root, &Path::parse("/tools/echo")?, Echo)?;
    capability::mount(&fs, &root, &Path::parse("/tools/text/upper")?, Upper)?;
    capability::mount(&fs, &root, &Path::parse("/tools/test/fail")?, Fail)?;
    capability::mount(&fs, &root, &Path::parse("/models/mock/infer")?, model::MockModel)?;
    capability::mount(&fs, &root, &Path::parse("/models/mock/fail")?, model::MockFail)?;
    fs.bind(&root, &Path::parse("/memory/context")?, Value::map([("text", Value::from("shell-less boot"))]))?;
    Ok((fs, host))
}

/// What agents may request from this host.
fn policy() -> meatfs::Result<Policy> {
    Ok(Policy::default()
        .allow(Path::parse("/tools")?, Rights::INVOKE | Rights::INSPECT)
        .allow(Path::parse("/models")?, Rights::INVOKE | Rights::INSPECT)
        .allow(Path::parse("/memory")?, Rights::READ)
        .allow(Path::parse("/state")?, Rights::READ | Rights::WRITE))
}

fn root(host: &GrantSet) -> meatfs::Access<'_> {
    host.access(host.iter().next().expect("genesis issues the root grant").id()).expect("grant is in its set")
}

fn describe_target(op: &Op, show: Show) -> String {
    match op {
        Op::Compose(expr) => {
            let sources: Vec<_> = expr.sources().iter().map(ToString::to_string).collect();
            format!("[{}]", sources.join(", "))
        }
        op => op.target().map(|t| show.path(t)).unwrap_or_default(),
    }
}

/// The shape of a composition: constructors and resolved references are
/// structure; keys, selector keys and literals are source text.
fn shape(expr: &ValueExpr, show: Show) -> String {
    match expr {
        ValueExpr::Literal(v) => format!("literal {}", show.value(v)),
        ValueExpr::Select { source, path } => {
            let path: Vec<String> = path
                .iter()
                .map(|s| match s {
                    Selector::Key(k) => format!(".{}", show.name(k)),
                    Selector::Index(i) => format!("[{i}]"),
                })
                .collect();
            format!("{source}{}", path.concat())
        }
        ValueExpr::Map(fields) => {
            let fields: Vec<_> = fields.iter().map(|(k, e)| format!("{}: {}", show.name(k), shape(e, show))).collect();
            format!("{{{}}}", fields.join(", "))
        }
        ValueExpr::List(items) => format!("[{}]", items.iter().map(|e| shape(e, show)).collect::<Vec<_>>().join(", ")),
    }
}

fn print_ir(program: &Program, show: Show) {
    let graph = &program.graph;
    println!("  agent  {}", show.name(&program.agent));
    println!("  graph  {}", graph.id);
    for (path, rights) in &program.authority {
        println!("  declare {} [{rights}]", show.path(path));
    }
    for node in &graph.nodes {
        let data = match &node.op {
            Op::Compose(expr) => format!(" = {}", shape(expr, show)),
            op => op.literal().map(|v| format!(" ← {}", show.value(v))).unwrap_or_default(),
        };
        let label = node.label.as_deref().map(|l| format!("  ({})", show.name(l))).unwrap_or_default();
        let uses: String = node.op.uses().iter().map(|u| format!("  +{} [{}]", show.path(&u.path), u.rights)).collect();
        let target = node.op.target().map(|t| format!(" {}", show.path(t))).unwrap_or_default();
        println!("  {}  {:<7}{target}{data}{uses}{label}", node.id, node.op.name());
    }
    for edge in &graph.edges {
        let kind = match edge.kind {
            EdgeKind::Data => "data ",
            EdgeKind::Order => "order",
        };
        println!("  {kind}  {} → {}", edge.from, edge.to);
    }
    for output in &graph.outputs {
        println!("  output {} ← {}", show.name(&output.name), output.source);
    }
}

fn print_receipt(program: &Program, receipt: &ExecutionReceipt, show: Show) {
    println!("  execution  {}", receipt.execution);
    println!("  graph      {}", receipt.graph);
    println!("  principal  {} ({})", receipt.principal, show.name(&program.agent));
    println!("  capture    {:?}", receipt.capture);
    for g in &receipt.grants {
        let target = match &g.target {
            RecordedTarget::Object(o) => o.short(),
            RecordedTarget::Name(name) => format!("name {}", show.captured(name)),
            RecordedTarget::Namespace(name) => format!("namespace {}", show.captured(name)),
        };
        println!("  grant      {}  {target} [{}]", g.grant.short(), g.rights);
    }
    for (from, to) in &receipt.derived_order {
        println!("  derived    {from} → {to}");
    }
    let schedule: Vec<_> = receipt.schedule.iter().map(ToString::to_string).collect();
    println!("  schedule   {}", schedule.join(" → "));
    for n in &receipt.nodes {
        let object = n.object.map(|o| o.short()).unwrap_or_else(|| "-".to_owned());
        let grant = n.grant.map(|g| g.short()).unwrap_or_else(|| "-".to_owned());
        let detail = match (&n.output, &n.error) {
            (_, Some(e)) => format!("{:?}", e.kind),
            (Some(v), None) => format!("→ {}", show.captured(v)),
            (None, None) => String::new(),
        };
        let state = format!("{:?}", n.state);
        println!("  node       {}  {:<7} {state:<9} {object}  {grant}  staged:{:?}  {detail}", n.node, n.op, n.staged);
        if let Some(meta) = &n.declared {
            let implementation = match (&meta.invocation.implementation, &meta.invocation.revision) {
                (Some(i), Some(r)) => format!("  {i}@{r}"),
                (Some(i), None) => format!("  {i}"),
                _ => String::new(),
            };
            println!("                 declared {:?} + {:?}{implementation}", meta.purity, meta.determinism);
        }
    }
    for e in &receipt.events {
        let node = e.cause.map(|c| c.node.to_string()).unwrap_or_default();
        let grant = e.grant.map(|g| g.short()).unwrap_or_default();
        println!("  event      #{:<3} {node}  {}  {grant}  {}", e.seq, e.object.short(), kind(&e.kind, show));
    }
    println!("  transaction {:?}", receipt.transaction);
    if let Some(e) = &receipt.error {
        println!("  error      {e}");
    }
    for o in &receipt.outputs {
        println!("  output     #{} {} ← {} = {}", o.index, show.captured(&o.name), o.source, show.captured(&o.value));
    }
}

fn kind(kind: &EventKind, show: Show) -> String {
    match kind {
        EventKind::Bound(name) => format!("bound {}", show.captured(name)),
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

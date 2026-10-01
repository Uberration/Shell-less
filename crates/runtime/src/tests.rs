use super::*;
use capability::builtin::{Echo, Fail, Upper};
use capability::{Capability, CapabilityContext, Determinism, Fault};
use meatfs::{Access, EventKind, Seed};
use meatyaml::{Edge, Graph, Node, WriteOp};

fn p(s: &str) -> Path {
    Path::parse(s).unwrap()
}

const ECHO: &str = include_str!("../../../examples/echo.meat.yaml");
const BUTCHER: &str = include_str!("../../../examples/butcher.meat.yaml");
const FAILURE: &str = include_str!("../../../examples/failure.meat.yaml");

/// Effectful test capability: writes its input to every attached path,
/// and additionally to the path named by `{ also: "/..." }` if given.
struct Stamp;
impl Capability for Stamp {
    type Input = Value;
    type Output = Value;
    fn describe(&self) -> &'static str {
        "write input to attached paths"
    }
    fn meta(&self) -> CapabilityMeta {
        CapabilityMeta::new(Purity::Effectful, Determinism::Deterministic)
    }
    fn invoke(&self, cx: &CapabilityContext<'_>, input: Value) -> Result<Value, Fault> {
        let fx = cx.effects()?;
        for entry in fx.grants() {
            fx.write(&entry.path, input.clone())?;
        }
        if let Some(also) = input.get("also").and_then(Value::as_text) {
            fx.write(&Path::parse(also)?, input.clone())?;
        }
        Ok(input)
    }
}

fn root(host: &GrantSet) -> Access<'_> {
    host.access(host.iter().next().unwrap().id()).unwrap()
}

fn boot(seed: u64) -> (MeatFs, GrantSet) {
    let (fs, host) = MeatFs::genesis(Seed::fixed(seed));
    let r = root(&host);
    capability::mount(&fs, &r, &p("/tools/echo"), Echo).unwrap();
    capability::mount(&fs, &r, &p("/tools/text/upper"), Upper).unwrap();
    capability::mount(&fs, &r, &p("/tools/test/fail"), Fail).unwrap();
    capability::mount(&fs, &r, &p("/tools/test/stamp"), Stamp).unwrap();
    capability::mount(&fs, &r, &p("/models/mock/infer"), model::MockModel).unwrap();
    capability::mount(&fs, &r, &p("/models/mock/fail"), model::MockFail).unwrap();
    fs.bind(&r, &p("/memory/context"), Value::map([("text", Value::from("boot"))])).unwrap();
    (fs, host)
}

fn policy() -> Policy {
    Policy::default()
        .allow(p("/tools"), Rights::INVOKE)
        .allow(p("/models"), Rights::INVOKE)
        .allow(p("/memory"), Rights::READ)
        .allow(p("/state"), Rights::READ | Rights::WRITE)
}

fn snapshot(fs: &MeatFs, host: &GrantSet) -> Vec<(Path, ObjectId, Option<Value>)> {
    fs.snapshot(&root(host), &Path::root()).unwrap()
}

fn outcome(fs: &MeatFs, src: &str, capture: ContentCapture) -> Result<ExecutionOutcome, LoadError> {
    let program = meatyaml::compile(src).unwrap();
    Ok(execute(fs, &load(fs, &policy(), Limits::default(), capture, &program)?, capture))
}

/// Run with inline capture, for tests that inspect payloads in the receipt.
fn run(fs: &MeatFs, src: &str) -> Result<ExecutionReceipt, LoadError> {
    outcome(fs, src, ContentCapture::Inline).map(|o| o.receipt)
}

fn inline(captured: &Option<CapturedValue>) -> &Value {
    match captured {
        Some(CapturedValue::Inline(v)) => v,
        other => panic!("expected an inline value, got {other:?}"),
    }
}

fn inlined(receipt: &ExecutionReceipt) -> BTreeMap<String, Value> {
    receipt
        .outputs
        .iter()
        .map(|o| {
            let name = inline(&Some(o.name.clone())).as_text().unwrap().to_owned();
            (name, inline(&Some(o.value.clone())).clone())
        })
        .collect()
}

fn text(s: &str) -> Value {
    Value::map([("text", Value::from(s))])
}

#[test]
fn butcher_acceptance() {
    let (fs, host) = boot(42);
    let receipt = run(&fs, BUTCHER).unwrap();
    assert!(receipt.succeeded(), "{:?}", receipt.error);
    assert_eq!(inlined(&receipt), BTreeMap::from([("upper".into(), text("MEAT")), ("echo".into(), text("MEAT"))]));
    assert_eq!(receipt.schedule, [0, 1, 2, 3].map(NodeId));
    assert!(receipt.nodes.iter().all(|n| n.state == NodeState::Succeeded));

    // Objects created by the execution are recorded and visible.
    let upper = fs.resolve(&p("/state/upper")).unwrap();
    assert_eq!(receipt.nodes[2].object, Some(upper));
    assert_eq!(fs.read(&root(&host), upper).unwrap(), text("MEAT"));
    let written: Vec<_> = receipt
        .events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::Written { .. }))
        .map(|e| (e.cause.unwrap().node, e.object, e.grant))
        .collect();
    assert_eq!(
        written,
        vec![
            (NodeId(2), upper, receipt.nodes[2].grant),
            (NodeId(3), receipt.nodes[3].object.unwrap(), receipt.nodes[3].grant)
        ]
    );
}

#[test]
fn failure_rolls_back_and_blocks() {
    let (fs, host) = boot(42);
    let before = snapshot(&fs, &host);
    let receipt = run(&fs, FAILURE).unwrap();

    assert_eq!(snapshot(&fs, &host), before, "namespace unchanged");
    assert_eq!(receipt.transaction, TxOutcome::RolledBack);
    assert!(receipt.outputs.is_empty());
    let states: Vec<_> = receipt.nodes.iter().map(|n| n.state).collect();
    assert_eq!(states, [NodeState::Succeeded, NodeState::Failed, NodeState::Blocked]);
    assert_eq!(receipt.nodes[2].error.as_ref().unwrap().kind, ExecutionErrorKind::DependencyFailed);

    let error = receipt.error.unwrap();
    let fail = fs.resolve(&p("/tools/test/fail")).unwrap();
    assert_eq!(
        (error.node, error.object, error.grant, error.kind),
        (Some(NodeId(1)), Some(fail), receipt.nodes[1].grant, ExecutionErrorKind::CapabilityFailed)
    );

    // The evidence survives: A was staged, the failure invoked, A rolled back.
    let a = receipt.nodes[0].object.unwrap();
    let kinds: Vec<_> = receipt.events.iter().map(|e| (e.object, e.kind.clone())).collect();
    assert_eq!(
        kinds,
        vec![(a, EventKind::Staged), (fail, EventKind::Invoked { ok: false }), (a, EventKind::RolledBack)]
    );
}

#[test]
fn loading_is_observational() {
    let (fs, host) = boot(1);
    let before = snapshot(&fs, &host);
    load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(BUTCHER).unwrap()).unwrap();
    let denied =
        load(&fs, &Policy::default(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(BUTCHER).unwrap());
    assert!(matches!(denied, Err(LoadError::Authority { kind: "PolicyDenied", detail: None, .. })));
    assert_eq!(snapshot(&fs, &host), before);
}

#[test]
fn execution_is_deterministic() {
    let a = run(&boot(7).0, BUTCHER).unwrap();
    let b = run(&boot(7).0, BUTCHER).unwrap();
    assert_eq!(a, b);
    let f1 = run(&boot(7).0, FAILURE).unwrap();
    let f2 = run(&boot(7).0, FAILURE).unwrap();
    assert_eq!(f1, f2);
}

#[test]
fn scheduling_follows_edges_not_node_ids() {
    let (fs, host) = boot(1);
    let write = |id, value: i64| Node {
        id: NodeId(id),
        label: None,
        op: Op::Write(WriteOp { target: p("/state/x"), value: Some(Value::Int(value)) }),
    };
    // n1 must precede n0: the final value is n0's.
    let graph = Graph::new(
        vec![write(0, 0), write(1, 1)],
        vec![Edge { from: NodeId(1), to: NodeId(0), kind: EdgeKind::Order }],
        vec![],
    );
    let program = Program { agent: "a".into(), authority: vec![], graph };
    let receipt = execute(
        &fs,
        &load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap(),
        ContentCapture::Inline,
    )
    .receipt;
    assert_eq!(receipt.schedule, vec![NodeId(1), NodeId(0)]);
    let x = fs.resolve(&p("/state/x")).unwrap();
    assert_eq!(fs.read(&root(&host), x).unwrap(), Value::Int(0));
}

#[test]
fn capabilities_get_only_node_scoped_authority() {
    let (fs, host) = boot(1);
    let program = r#"
agent: { name: a }
authority:
  - { path: /tools/test/stamp, rights: [invoke] }
  - { path: /state, rights: [write] }
flow:
  - id: own
    write: { path: /state/b, value: 0 }
  - invoke:
      path: /tools/test/stamp
      input: { n: 1 }
      uses: [{ path: /state/a, rights: [write] }]
"#;
    let receipt = run(&fs, program).unwrap();
    assert!(receipt.succeeded(), "{:?}", receipt.error);
    let a = fs.resolve(&p("/state/a")).unwrap();
    assert_eq!(fs.read(&root(&host), a).unwrap(), Value::map([("n", Value::Int(1))]));

    // The execution holds a grant for /state/b, but the stamp node does not.
    let (fs, host) = boot(1);
    let before = snapshot(&fs, &host);
    let escalate = program.replace("input: { n: 1 }", "input: { also: /state/b }");
    let receipt = run(&fs, &escalate).unwrap();
    let error = receipt.error.unwrap();
    assert_eq!((error.node, error.kind), (Some(NodeId(1)), ExecutionErrorKind::AuthorityDenied));
    assert_eq!(snapshot(&fs, &host), before, "the stamp's own write rolled back too");
}

#[test]
fn pure_capabilities_cannot_be_given_uses() {
    let (fs, _) = boot(1);
    let e = run(
        &fs,
        r#"
agent: { name: a }
authority: [{ path: /tools, rights: [invoke] }, { path: /state, rights: [write] }]
flow:
  - invoke: { path: /tools/echo, input: { text: x }, uses: [{ path: /state/a, rights: [write] }] }
"#,
    );
    assert!(matches!(e, Err(LoadError::InvalidGraph { .. })));
}

#[test]
fn escalation_fails_at_policy() {
    let (fs, _) = boot(1);
    let e = run(&fs, include_str!("../../../examples/escalate-declared.meat.yaml")).err().unwrap();
    let LoadError::Authority { kind, summary, detail } = e else { panic!("expected an authority failure") };
    assert_eq!((kind, summary.as_str()), ("PolicyDenied", "PolicyDenied path=<omitted> rights=write"));
    assert!(detail.unwrap().starts_with("/memory/context:"), "run() captures inline");
}

#[test]
fn rejects_structurally_invalid_ir() {
    let program = meatyaml::compile(ECHO).unwrap();
    let mut cyclic = program.graph.clone();
    cyclic.edges.push(Edge { from: NodeId(1), to: NodeId(0), kind: EdgeKind::Order });
    assert!(matches!(validate(&cyclic), Err(LoadError::InvalidGraph { .. })));

    let mut double_input = program.graph.clone();
    double_input.edges.push(Edge { from: NodeId(0), to: NodeId(1), kind: EdgeKind::Data });
    assert!(matches!(validate(&double_input), Err(LoadError::InvalidGraph { .. })));
}

#[test]
fn retired_graphs_cannot_run() {
    let (fs, _) = boot(1);
    let program = meatyaml::compile(ECHO).unwrap();
    let loaded = load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap();
    let grant = loaded.nodes()[0].grant;
    loaded.retire(&fs);
    let again = load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap();
    assert_ne!(again.nodes()[0].grant, grant);
    assert!(execute(&fs, &again, ContentCapture::Inline).receipt.succeeded());
}

const THINKER: &str = include_str!("../../../examples/thinker.meat.yaml");
const MODEL_FAILURE: &str = include_str!("../../../examples/model-failure.meat.yaml");
const FORK: &str = include_str!("../../../examples/fork.meat.yaml");

fn message(role: &str, content: &str) -> Value {
    Value::map([("role", Value::from(role)), ("content", Value::from(content))])
}

#[test]
fn model_is_an_ordinary_capability() {
    let (fs, host) = boot(42);
    let receipt = run(&fs, THINKER).unwrap();
    assert!(receipt.succeeded(), "{:?}", receipt.error);

    let answer = Value::map([
        ("message", message("assistant", "MEAT")),
        ("usage", Value::map([("input_tokens", Value::Int(2)), ("output_tokens", Value::Int(1))])),
        ("finish", Value::from("stop")),
    ]);
    assert_eq!(inlined(&receipt), BTreeMap::from([("answer".into(), answer.clone())]));
    let stored = fs.resolve(&p("/state/answer")).unwrap();
    assert_eq!(fs.read(&root(&host), stored).unwrap(), answer);

    // Resolved to the model's identity, holding exactly its own invoke grant.
    let model = fs.resolve(&p("/models/mock/infer")).unwrap();
    let infer = &receipt.nodes[0];
    assert_eq!(infer.object, Some(model));
    assert!(infer.uses.is_empty());
    let grant = receipt.grants.iter().find(|g| Some(g.grant) == infer.grant).unwrap();
    assert_eq!((&grant.target, grant.rights), (&RecordedTarget::Object(model), Rights::INVOKE));

    // Provenance: input, declared properties and implementation.
    let meta = infer.declared.clone().unwrap();
    assert_eq!((meta.purity, meta.determinism), (Purity::Effectful, Determinism::Deterministic));
    assert_eq!(meta.invocation.implementation.as_deref(), Some(model::MockModel::IMPLEMENTATION));
    assert_eq!(meta.invocation.revision.as_deref(), Some(model::MockModel::REVISION));
    let messages = inline(&infer.input).get("messages").unwrap();
    assert_eq!(messages, &Value::List(vec![message("system", "uppercase"), message("user", "meat")]));

    // Journaled like any invocation, caused by its node with its grant.
    let invoked: Vec<_> = receipt.events.iter().filter(|e| e.object == model).map(|e| (&e.kind, e.grant)).collect();
    assert_eq!(invoked, vec![(&EventKind::Invoked { ok: true }, infer.grant)]);

    assert_eq!(run(&boot(42).0, THINKER).unwrap(), receipt, "replayable with identical seed and implementation");
}

#[test]
fn model_failure_rolls_back() {
    let (fs, host) = boot(42);
    let before = snapshot(&fs, &host);
    let receipt = run(&fs, MODEL_FAILURE).unwrap();
    assert_eq!(snapshot(&fs, &host), before);
    assert_eq!(receipt.transaction, TxOutcome::RolledBack);
    let states: Vec<_> = receipt.nodes.iter().map(|n| n.state).collect();
    assert_eq!(states, [NodeState::Succeeded, NodeState::Failed, NodeState::Blocked]);
    let error = receipt.error.unwrap();
    let fail = fs.resolve(&p("/models/mock/fail")).unwrap();
    assert_eq!(
        (error.node, error.object, error.grant, error.kind),
        (Some(NodeId(1)), Some(fail), receipt.nodes[1].grant, ExecutionErrorKind::CapabilityFailed)
    );
}

#[test]
fn model_branch_is_independent_of_tool_branch() {
    let (fs, _) = boot(42);
    let program = meatyaml::compile(FORK).unwrap();
    // tool → store_tool and model → store_model, nothing across.
    let edges: Vec<_> = program.graph.edges.iter().map(|e| (e.from.0, e.to.0, e.kind)).collect();
    assert_eq!(edges, vec![(0, 2, EdgeKind::Data), (1, 3, EdgeKind::Data)]);

    let receipt = execute(
        &fs,
        &load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap(),
        ContentCapture::Inline,
    )
    .receipt;
    assert!(receipt.succeeded());
    assert_eq!(inlined(&receipt)["tool"], text("MEAT"));
    assert_eq!(inlined(&receipt)["model"].get("message"), Some(&message("assistant", "MEAT")));
    assert_eq!(receipt.nodes[0].declared.as_ref().unwrap().purity, Purity::Pure);
    assert_eq!(receipt.nodes[1].declared.as_ref().unwrap().purity, Purity::Effectful);
}

// ── M4.1: composition, content-safe receipts, transaction boundary ──────────

const COMPOSER: &str = include_str!("../../../examples/composer.meat.yaml");
const SENTINEL: &str = "zq-sentinel-7731";

/// Effectful outside MeatFS: bumps a host-owned counter no transaction owns.
struct Counter(std::sync::Arc<std::sync::atomic::AtomicI64>);
impl Capability for Counter {
    type Input = Value;
    type Output = Value;
    fn describe(&self) -> &'static str {
        "increment a host counter"
    }
    fn meta(&self) -> CapabilityMeta {
        CapabilityMeta::new(Purity::Effectful, Determinism::Nondeterministic)
    }
    fn invoke(&self, _: &CapabilityContext<'_>, _: Value) -> Result<Value, Fault> {
        Ok(Value::Int(self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1))
    }
}

/// Fails with an error message that echoes its input.
struct Leaky;
impl Capability for Leaky {
    type Input = Value;
    type Output = Value;
    fn describe(&self) -> &'static str {
        "fail, quoting the input"
    }
    fn meta(&self) -> CapabilityMeta {
        CapabilityMeta::new(Purity::Pure, Determinism::Deterministic)
    }
    fn invoke(&self, _: &CapabilityContext<'_>, input: Value) -> Result<Value, Fault> {
        Err(Fault::Failed(format!("refusing {input}")))
    }
}

fn counter(fs: &MeatFs, host: &GrantSet, path: &str) -> std::sync::Arc<std::sync::atomic::AtomicI64> {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
    capability::mount(fs, &root(host), &p(path), Counter(count.clone())).unwrap();
    count
}

fn count(c: &std::sync::Arc<std::sync::atomic::AtomicI64>) -> i64 {
    c.load(std::sync::atomic::Ordering::SeqCst)
}

#[test]
fn composer_feeds_one_source_to_both_branches() {
    let (fs, _) = boot(42);
    let outcome = outcome(&fs, COMPOSER, ContentCapture::default()).unwrap();
    let receipt = &outcome.receipt;
    assert!(receipt.succeeded(), "{:?}", receipt.error);
    assert_eq!(outcome.outputs["upper"], text("MEAT"));
    assert_eq!(outcome.outputs["answer"].get("message"), Some(&message("assistant", "MEAT")));

    // Compose nodes hold no authority and touch no object.
    for i in [0, 2] {
        assert_eq!((receipt.nodes[i].op, receipt.nodes[i].grant, receipt.nodes[i].object), ("compose", None, None));
    }
    assert_eq!(receipt.grants.len(), 2);
    assert!(receipt.derived_order.is_empty(), "only one declared-effectful invocation");
    assert_eq!(receipt.schedule, [0, 1, 2, 3].map(NodeId));
}

#[test]
fn literal_maps_stay_literal_at_runtime() {
    let (fs, _) = boot(1);
    let src = "agent: { name: a }\nflow:\n  - id: x\n    compose: { literal: 1 }\n  - id: y\n    compose:\n      literal:\n        from: x\noutputs: { y: y }\n";
    let outcome = outcome(&fs, src, ContentCapture::Omit).unwrap();
    assert_eq!(outcome.outputs["y"], Value::map([("from", Value::from("x"))]));
}

#[test]
fn composition_failures_are_structured_and_roll_back() {
    let cases = [
        ("path: [nope]", ExecutionErrorKind::MissingValue),
        ("path: [text, 0]", ExecutionErrorKind::WrongValueKind),
        ("path: [list, 5]", ExecutionErrorKind::IndexOutOfRange),
    ];
    for (selector, kind) in cases {
        let (fs, host) = boot(1);
        let before = snapshot(&fs, &host);
        let src = format!(
            r#"
agent: {{ name: a }}
authority: [{{ path: /state, rights: [write] }}, {{ path: /tools, rights: [invoke] }}]
flow:
  - id: a
    write: {{ path: /state/a, value: 1 }}
  - id: source
    compose: {{ literal: {{ text: meat, list: [1] }} }}
  - id: pick
    after: a
    compose: {{ select: {{ from: source, {selector} }} }}
  - invoke: {{ path: /tools/echo, from: pick }}
"#
        );
        let receipt = outcome(&fs, &src, ContentCapture::Omit).unwrap().receipt;
        let states: Vec<_> = receipt.nodes.iter().map(|n| n.state).collect();
        assert_eq!(states, [NodeState::Succeeded, NodeState::Succeeded, NodeState::Failed, NodeState::Blocked]);
        let error = receipt.error.unwrap();
        assert_eq!((error.node, error.object, error.grant, error.kind), (Some(NodeId(2)), None, None, kind));
        assert_eq!(receipt.nodes[0].staged, StagedChanges::RolledBack);
        assert_eq!(snapshot(&fs, &host), before);
    }
}

#[test]
fn composition_is_bounded_by_host_limits() {
    let (fs, _) = boot(1);
    let program = meatyaml::compile(COMPOSER).unwrap();
    let shallow = Limits { max_depth: 2, ..Limits::default() };
    assert!(matches!(
        load(&fs, &policy(), shallow, ContentCapture::Omit, &program),
        Err(LoadError::LimitExceeded { .. })
    ));
    let few = Limits { max_exprs: 3, ..Limits::default() };
    assert!(matches!(load(&fs, &policy(), few, ContentCapture::Omit, &program), Err(LoadError::LimitExceeded { .. })));

    let tiny = Limits { max_value_size: 8, ..Limits::default() };
    let receipt =
        execute(&fs, &load(&fs, &policy(), tiny, ContentCapture::Omit, &program).unwrap(), ContentCapture::Omit)
            .receipt;
    assert_eq!(receipt.error.unwrap().kind, ExecutionErrorKind::LimitExceeded);
}

/// A program that carries the sentinel through a prompt, a model output and
/// a state write, then fails in a capability whose error quotes its input.
fn sentinel_program(fail: bool) -> String {
    let failing = if fail { "  - invoke: { path: /tools/test/leaky, from: infer }\n" } else { "" };
    format!(
        r#"
agent: {{ name: secretive }}
authority:
  - {{ path: /models/mock/infer, rights: [invoke] }}
  - {{ path: /tools/test/leaky, rights: [invoke] }}
  - {{ path: /state/secret, rights: [write] }}
flow:
  - id: source
    compose: {{ literal: {{ text: {SENTINEL} }} }}
  - id: request
    compose:
      map:
        messages:
          list:
            - map: {{ role: {{ literal: system }}, content: {{ literal: uppercase }} }}
            - map: {{ role: {{ literal: user }}, content: {{ select: {{ from: source, path: [text] }} }} }}
  - id: infer
    invoke: {{ path: /models/mock/infer, from: request }}
  - id: store
    write: {{ path: /state/secret, from: infer }}
{failing}outputs:
  answer: store
"#
    )
}

fn leaks(haystack: &str) -> bool {
    haystack.contains(SENTINEL) || haystack.contains(&SENTINEL.to_uppercase())
}

#[test]
fn default_receipts_omit_content() {
    for fail in [false, true] {
        let (fs, host) = boot(5);
        capability::mount(&fs, &root(&host), &p("/tools/test/leaky"), Leaky).unwrap();
        let outcome = outcome(&fs, &sentinel_program(fail), ContentCapture::default()).unwrap();
        let receipt = &outcome.receipt;
        assert_eq!(receipt.capture, ContentCapture::Omit);
        assert!(!leaks(&format!("{receipt:?}")), "receipt leaks content (fail={fail})");
        let journal = fs.journal(&root(&host), &Path::root()).unwrap();
        assert!(!leaks(&format!("{journal:?}")), "journal leaks content");
        if fail {
            assert_eq!(receipt.error.as_ref().unwrap().kind, ExecutionErrorKind::CapabilityFailed);
            assert!(outcome.outputs.is_empty());
        } else {
            // The explicit result channel still carries the real value.
            let answer = outcome.outputs["answer"].get("message").unwrap();
            assert_eq!(answer.get("content"), Some(&Value::from(SENTINEL.to_uppercase())));
            assert_eq!(receipt.outputs.len(), 1);
            assert_eq!(
                (&receipt.outputs[0].name, &receipt.outputs[0].value),
                (&CapturedValue::Omitted, &CapturedValue::Omitted)
            );
        }
    }
}

#[test]
fn only_the_host_can_enable_inline_capture() {
    let (fs, host) = boot(5);
    capability::mount(&fs, &root(&host), &p("/tools/test/leaky"), Leaky).unwrap();
    let receipt = outcome(&fs, &sentinel_program(true), ContentCapture::Inline).unwrap().receipt;
    assert!(leaks(&format!("{:?}", receipt.nodes[0].output)));
    assert!(receipt.error.unwrap().detail.is_some_and(|d| leaks(&d)), "detail kept inline");

    // Programs have no way to ask for it…
    let asks = format!("capture: inline\n{}", sentinel_program(false));
    assert_eq!(meatyaml::compile(&asks).unwrap_err().kind, meatyaml::ErrorKind::UnknownKey);
    // …and a model saying so changes nothing.
    let (fs, _) = boot(5);
    let src = THINKER.replace("content: meat", "content: \"capture: inline\"");
    let receipt = outcome(&fs, &src, ContentCapture::Omit).unwrap().receipt;
    assert_eq!(
        (&receipt.outputs[0].name, &receipt.outputs[0].value),
        (&CapturedValue::Omitted, &CapturedValue::Omitted)
    );
    assert_eq!(receipt.nodes[0].output, Some(CapturedValue::Omitted));
}

#[test]
fn rollback_does_not_undo_external_effects() {
    let (fs, host) = boot(1);
    let calls = counter(&fs, &host, "/tools/test/counter");
    let before = snapshot(&fs, &host);
    let src = r#"
agent: { name: a }
authority: [{ path: /tools/test, rights: [invoke] }, { path: /state, rights: [write] }]
flow:
  - id: count
    invoke: { path: /tools/test/counter, input: null }
  - id: write_a
    write: { path: /state/a, from: count }
  - id: fail
    after: write_a
    invoke: { path: /tools/test/fail, input: null }
"#;
    let receipt = outcome(&fs, src, ContentCapture::Omit).unwrap().receipt;
    assert_eq!(receipt.transaction, TxOutcome::RolledBack);
    assert_eq!(snapshot(&fs, &host), before, "MeatFS state rolled back");
    assert_eq!(count(&calls), 1, "the external effect happened exactly once and was not retried");

    let counted = &receipt.nodes[0];
    assert_eq!((counted.state, counted.staged), (NodeState::Succeeded, StagedChanges::None));
    let declared = counted.declared.as_ref().unwrap();
    assert_eq!((declared.purity, declared.determinism), (Purity::Effectful, Determinism::Nondeterministic));
    assert_eq!((receipt.nodes[1].state, receipt.nodes[1].staged), (NodeState::Succeeded, StagedChanges::RolledBack));
    assert_eq!(receipt.nodes[2].state, NodeState::Failed);
}

#[test]
fn effectful_invocations_are_ordered_conservatively() {
    let (fs, host) = boot(1);
    counter(&fs, &host, "/tools/test/counter");
    counter(&fs, &host, "/tools/test/other");
    let src = r#"
agent: { name: a }
authority: [{ path: /tools, rights: [invoke] }]
flow:
  - invoke: { path: /tools/test/counter, input: null }
  - invoke: { path: /tools/echo, input: { text: x } }
  - invoke: { path: /tools/test/other, input: null }
"#;
    let program = meatyaml::compile(src).unwrap();
    assert!(program.graph.edges.is_empty(), "disjoint MeatFS footprints: no IR edges");
    let loaded = load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap();
    // The two declared-effectful invocations are chained; the pure one is not.
    assert_eq!(loaded.derived_order(), &[(NodeId(0), NodeId(2))]);
    let receipt = execute(&fs, &loaded, ContentCapture::Omit).receipt;
    assert_eq!(receipt.derived_order, vec![(NodeId(0), NodeId(2))]);
}

#[test]
fn omitted_receipts_are_deterministic() {
    let a = outcome(&boot(9).0, COMPOSER, ContentCapture::Omit).unwrap();
    let b = outcome(&boot(9).0, COMPOSER, ContentCapture::Omit).unwrap();
    assert_eq!(a, b);
}

// ── M4.1 closeout: real commit conflict, ordering, pinning, load diagnostics ──

#[test]
fn commit_conflict_with_a_real_second_writer() {
    let (fs, host) = boot(1);
    let calls = counter(&fs, &host, "/tools/test/counter");
    let shared = fs.bind(&root(&host), &p("/state/shared"), Value::from("initial")).unwrap();
    let src = r#"
agent: { name: a }
authority: [{ path: /tools/test/counter, rights: [invoke] }, { path: /state, rights: [write] }]
flow:
  - id: count
    invoke: { path: /tools/test/counter, input: null }
  - write: { path: /state/shared, from: count }
  - write: { path: /state/other, value: staged }
outputs: { count: count }
"#;
    let loaded =
        load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(src).unwrap()).unwrap();

    // Immediately before the execution commits, the host commits a different
    // value to /state/shared through its own, genuine MeatFS transaction.
    let mut second_writer_ran = false;
    let outcome = execute_with(&fs, &loaded, ContentCapture::Omit, &mut || {
        let tx = fs.transaction();
        tx.write(&root(&host), shared, Value::from("second writer")).unwrap();
        tx.commit().unwrap();
        second_writer_ran = true;
    });
    assert!(second_writer_ran);
    let receipt = &outcome.receipt;

    let error = receipt.error.as_ref().unwrap();
    assert_eq!((error.kind, error.node, error.object), (ExecutionErrorKind::TransactionConflict, None, Some(shared)));
    assert_eq!(receipt.transaction, TxOutcome::RolledBack);
    assert!(!receipt.succeeded());
    assert!(outcome.outputs.is_empty() && receipt.outputs.is_empty(), "no speculative outputs");

    // Every node finished; their MeatFS changes did not survive.
    assert!(receipt.nodes.iter().all(|n| n.state == NodeState::Succeeded));
    let staged: Vec<_> = receipt.nodes.iter().map(|n| n.staged).collect();
    assert_eq!(staged, [StagedChanges::None, StagedChanges::RolledBack, StagedChanges::RolledBack]);
    assert_eq!(fs.read(&root(&host), shared).unwrap(), Value::from("second writer"), "competing commit preserved");
    assert_eq!(fs.resolve(&p("/state/other")), None, "failed execution's creation discarded");

    // Exactly one invocation; nothing was retried.
    assert_eq!(count(&calls), 1);
    assert_eq!(receipt.schedule.len(), 3);
    let counter_id = fs.resolve(&p("/tools/test/counter")).unwrap();
    let invocations =
        receipt.events.iter().filter(|e| e.object == counter_id && matches!(e.kind, EventKind::Invoked { .. }));
    assert_eq!(invocations.count(), 1);
}

#[test]
fn unknown_purity_takes_the_conservative_path() {
    let (fs, host) = boot(1);
    counter(&fs, &host, "/tools/test/counter");
    // A data object at an invocable path: no declaration, so purity unknown.
    fs.bind(&root(&host), &p("/tools/test/opaque"), Value::Null).unwrap();
    let src = r#"
agent: { name: a }
authority: [{ path: /tools, rights: [invoke] }]
flow:
  - invoke: { path: /tools/test/opaque, input: null }
  - invoke: { path: /tools/echo, input: { text: x } }
  - invoke: { path: /tools/test/counter, input: null }
"#;
    let loaded =
        load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(src).unwrap()).unwrap();
    assert_eq!(loaded.nodes()[0].declared, None);
    assert_eq!(loaded.derived_order(), &[(NodeId(0), NodeId(2))], "unknown is chained; the pure echo is not");
}

#[test]
fn explicit_order_survives_against_node_id_order() {
    let (fs, host) = boot(1);
    // Three effectful invocations sharing one clock: each returns its call order.
    let clock = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
    for name in ["a", "b", "c"] {
        capability::mount(&fs, &root(&host), &p(&format!("/tools/clock/{name}")), Counter(clock.clone())).unwrap();
    }
    let invoke = |id, name: &str| Node {
        id: NodeId(id),
        label: None,
        op: Op::Invoke(meatyaml::InvokeOp {
            target: p(&format!("/tools/clock/{name}")),
            input: Some(Value::Null),
            uses: vec![],
        }),
    };
    // n2 must precede n0; n1 is unconstrained by the source.
    let graph = Graph::new(
        vec![invoke(0, "a"), invoke(1, "b"), invoke(2, "c")],
        vec![Edge { from: NodeId(2), to: NodeId(0), kind: EdgeKind::Order }],
        vec![],
    );
    let program = Program { agent: "a".into(), authority: vec![], graph };
    let loaded = load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).unwrap();
    // Topological order n1, n2, n0: chaining adds n1 → n2; n2 → n0 already holds.
    assert_eq!(loaded.derived_order(), &[(NodeId(1), NodeId(2))]);
    let receipt = execute(&fs, &loaded, ContentCapture::Inline).receipt;
    assert_eq!(receipt.schedule, [1, 2, 0].map(NodeId));
    let ticks: Vec<_> = receipt.nodes.iter().map(|n| inline(&n.output).clone()).collect();
    assert_eq!(ticks, [3, 1, 2].map(Value::Int), "n0 ran last despite the lowest id");
}

/// Declares Pure until told otherwise; counts its invocations.
struct Fickle {
    effectful: std::sync::Arc<std::sync::atomic::AtomicBool>,
    calls: std::sync::Arc<std::sync::atomic::AtomicI64>,
}
impl Capability for Fickle {
    type Input = Value;
    type Output = Value;
    fn describe(&self) -> &'static str {
        "changes its declaration on request"
    }
    fn meta(&self) -> CapabilityMeta {
        let purity =
            if self.effectful.load(std::sync::atomic::Ordering::SeqCst) { Purity::Effectful } else { Purity::Pure };
        CapabilityMeta::new(purity, Determinism::Deterministic)
    }
    fn invoke(&self, _: &CapabilityContext<'_>, input: Value) -> Result<Value, Fault> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(input)
    }
}

#[test]
fn execution_uses_the_declaration_scheduling_used() {
    let (fs, host) = boot(1);
    let effectful = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
    let fickle = Fickle { effectful: effectful.clone(), calls: calls.clone() };
    capability::mount(&fs, &root(&host), &p("/tools/test/fickle"), fickle).unwrap();
    let src = "agent: { name: a }\nauthority: [{ path: /tools, rights: [invoke] }]\nflow:\n  - invoke: { path: /tools/test/fickle, input: 1 }\n";
    let loaded =
        load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(src).unwrap()).unwrap();
    assert_eq!(loaded.nodes()[0].declared.as_ref().unwrap().purity, Purity::Pure);

    effectful.store(true, std::sync::atomic::Ordering::SeqCst);
    let receipt = execute(&fs, &loaded, ContentCapture::Omit).receipt;
    assert_eq!(receipt.error.unwrap().kind, ExecutionErrorKind::DeclarationChanged);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0, "refused before running");
    assert_eq!(
        receipt.nodes[0].declared.as_ref().unwrap().purity,
        Purity::Pure,
        "receipt shows the pinned declaration"
    );
}

#[test]
fn load_errors_render_without_source_text() {
    let (fs, _) = boot(1);
    let src = format!(
        "agent: {{ name: a }}\nauthority: [{{ path: /memory/{SENTINEL}, rights: [write] }}]\nflow:\n  - write: {{ path: /memory/{SENTINEL}, value: 1 }}\n"
    );
    let program = meatyaml::compile(&src).unwrap();
    let omitted = load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &program).err().unwrap();
    assert!(matches!(omitted, LoadError::Authority { kind: "PolicyDenied", detail: None, .. }));
    assert!(!leaks(&format!("{omitted:?}")), "the record itself holds no path: {omitted:?}");
    let inline = load(&fs, &policy(), Limits::default(), ContentCapture::Inline, &program).err().unwrap();
    assert!(leaks(&inline.render()));
}

// ── Record boundary: names in receipts and the journal are content ──────────

/// A write to a previously unbound path; the sentinel appears only there.
fn name_program(suffix: &str) -> String {
    format!(
        "agent: {{ name: a }}\nauthority: [{{ path: /state, rights: [write] }}]\nflow:\n  - id: store\n    write: {{ path: /state/{SENTINEL}{suffix}, value: 1 }}\noutputs: {{ result: store }}\n"
    )
}

fn name_run(fs: &MeatFs, suffix: &str, capture: ContentCapture) -> (Loaded, ExecutionOutcome) {
    let program = meatyaml::compile(&name_program(suffix)).unwrap();
    let loaded = load(fs, &policy(), Limits::default(), capture, &program).unwrap();
    let outcome = execute(fs, &loaded, capture);
    (loaded, outcome)
}

fn bound_event(events: &[Event], object: ObjectId) -> Event {
    events.iter().find(|e| e.object == object && matches!(e.kind, EventKind::Bound(_))).unwrap().clone()
}

/// Every public field of a receipt, inspected structurally: nothing inline.
fn assert_retains_no_content(receipt: &ExecutionReceipt) {
    assert_eq!(receipt.capture, ContentCapture::Omit);
    for g in &receipt.grants {
        assert!(!matches!(
            g.target,
            RecordedTarget::Name(CapturedValue::Inline(_)) | RecordedTarget::Namespace(CapturedValue::Inline(_))
        ));
    }
    for n in &receipt.nodes {
        assert!(
            !matches!(n.input, Some(CapturedValue::Inline(_))) && !matches!(n.output, Some(CapturedValue::Inline(_)))
        );
        assert!(n.error.as_ref().is_none_or(|e| e.detail.is_none()));
    }
    for e in &receipt.events {
        assert!(!matches!(e.kind, EventKind::Bound(CapturedValue::Inline(_))));
    }
    for o in &receipt.outputs {
        assert_eq!((&o.name, &o.value), (&CapturedValue::Omitted, &CapturedValue::Omitted));
    }
    assert!(receipt.error.as_ref().is_none_or(|e| e.detail.is_none()));
    assert!(!leaks(&format!("{receipt:?}")));
}

#[test]
fn name_grant_record_omits_the_path_while_the_live_grant_authorizes() {
    let (fs, host) = boot(1);
    let (loaded, outcome) = name_run(&fs, "", ContentCapture::Omit);
    let path = p(&format!("/state/{SENTINEL}"));

    // Operational: the live grant keeps the real name and authorizes the write.
    assert_eq!(loaded.grants().issued(0).unwrap().target(), &Target::Name(path.clone()));
    assert!(outcome.receipt.succeeded());
    let created = fs.resolve(&path).expect("the write created the name");
    assert_eq!(fs.read(&root(&host), created).unwrap(), Value::Int(1));

    // Recorded: the receipt's grant target holds no copy of the name.
    let grant = &outcome.receipt.grants[0];
    assert_eq!(grant.target, RecordedTarget::Name(CapturedValue::Omitted));
    assert_eq!(grant.grant, loaded.grants().issued(0).unwrap().id());
}

#[test]
fn binding_event_records_identity_and_provenance_without_the_path() {
    let (fs, host) = boot(1);
    let (_, outcome) = name_run(&fs, "", ContentCapture::Omit);
    let receipt = &outcome.receipt;
    let created = fs.resolve(&p(&format!("/state/{SENTINEL}"))).unwrap();

    for events in
        [fs.journal(&root(&host), &Path::root()).unwrap(), fs.journal_retained(&root(&host), &Path::root()).unwrap()]
    {
        let bound = bound_event(&events, created);
        assert_eq!(bound.kind, EventKind::Bound(CapturedValue::Omitted));
        assert_eq!(bound.cause.unwrap(), Cause { execution: receipt.execution, node: NodeId(0) });
        assert_eq!((bound.principal, bound.grant), (receipt.principal, Some(receipt.grants[0].grant)));
        assert!(!leaks(&format!("{events:?}")), "not even the raw history retained it");
    }
}

#[test]
fn omitted_receipts_retain_no_content_in_any_field() {
    let (fs, host) = boot(1);
    capability::mount(&fs, &root(&host), &p("/tools/test/leaky"), Leaky).unwrap();
    for src in [name_program(""), sentinel_program(false), sentinel_program(true)] {
        let receipt = outcome(&fs, &src, ContentCapture::Omit).unwrap().receipt;
        assert_retains_no_content(&receipt);
    }
}

#[test]
fn inline_capture_retains_names_and_never_reconstructs_omitted_ones() {
    let (fs, host) = boot(1);
    // First run: journal and receipt under the default, Omit.
    let (_, first) = name_run(&fs, "-first", ContentCapture::Omit);
    let first_object = fs.resolve(&p(&format!("/state/{SENTINEL}-first"))).unwrap();

    // The host opts in: journal and receipt inline.
    fs.set_journal_capture(&root(&host), ContentCapture::Inline).unwrap();
    let (_, second) = name_run(&fs, "-second", ContentCapture::Inline);
    let second_path = format!("/state/{SENTINEL}-second");
    let second_object = fs.resolve(&p(&second_path)).unwrap();
    let receipt = &second.receipt;
    assert_eq!(
        receipt.grants[0].target,
        RecordedTarget::Name(CapturedValue::Inline(Value::from(second_path.as_str())))
    );
    assert_eq!(
        bound_event(&receipt.events, second_object).kind,
        EventKind::Bound(CapturedValue::Inline(Value::from(second_path.as_str())))
    );
    assert_eq!(receipt.outputs[0].name, CapturedValue::Inline(Value::from("result")));

    // Neither the later policy nor an inline view brings back the first name.
    let retained = fs.journal_retained(&root(&host), &Path::root()).unwrap();
    assert_eq!(bound_event(&retained, first_object).kind, EventKind::Bound(CapturedValue::Omitted));
    assert_retains_no_content(&first.receipt);
}

#[test]
fn omitted_views_hide_inline_history_and_results_stay_exact() {
    let (fs, host) = boot(1);
    fs.set_journal_capture(&root(&host), ContentCapture::Inline).unwrap();
    name_run(&fs, "-kept", ContentCapture::Inline);
    fs.set_journal_capture(&root(&host), ContentCapture::Omit).unwrap();
    let (_, later) = name_run(&fs, "-later", ContentCapture::Omit);

    // The omitted view shows no content, including the earlier inline record.
    let omitted = fs.journal(&root(&host), &Path::root()).unwrap();
    assert!(omitted.iter().all(|e| !matches!(e.kind, EventKind::Bound(CapturedValue::Inline(_)))));
    assert!(!leaks(&format!("{omitted:?}")));
    // The raw history shows exactly what each record retained.
    let retained = fs.journal_retained(&root(&host), &Path::root()).unwrap();
    let kept = fs.resolve(&p(&format!("/state/{SENTINEL}-kept"))).unwrap();
    let dropped = fs.resolve(&p(&format!("/state/{SENTINEL}-later"))).unwrap();
    assert!(matches!(bound_event(&retained, kept).kind, EventKind::Bound(CapturedValue::Inline(_))));
    assert_eq!(bound_event(&retained, dropped).kind, EventKind::Bound(CapturedValue::Omitted));
    // The receipt for the later run is omitted; its real result is exact.
    assert_retains_no_content(&later.receipt);
    assert_eq!(later.outputs, BTreeMap::from([("result".to_owned(), Value::Int(1))]));
}

// ── M5: real inference is an ordinary node ──────────────────────────────────

const STORY: &str = r#"
agent: { name: storyteller }
authority:
  - { path: /models/local/stories/infer, rights: [invoke] }
  - { path: /state/story, rights: [write] }
flow:
  - id: request
    compose:
      map:
        messages:
          list:
            - map: { role: { literal: user }, content: { literal: hello world } }
        parameters:
          map: { max_tokens: { literal: 6 } }
  - id: infer
    invoke: { path: /models/local/stories/infer, from: request }
  - id: store
    write: { path: /state/story, from: infer }
outputs:
  story: store
"#;

fn with_local_model(seed: u64) -> (MeatFs, GrantSet) {
    let (fs, host) = boot(seed);
    let model = model::local::fixture::model(model::local::LocalLimits::default());
    capability::mount(&fs, &root(&host), &p("/models/local/stories/infer"), model).unwrap();
    (fs, host)
}

#[test]
fn local_inference_feeds_a_transactional_write_and_an_output() {
    let (fs, host) = with_local_model(3);
    let run = outcome(&fs, STORY, ContentCapture::Omit).unwrap();
    let receipt = &run.receipt;
    assert!(receipt.succeeded(), "{:?}", receipt.error);

    let story = &run.outputs["story"];
    assert_eq!(story.get("usage").unwrap().get("input_tokens"), Some(&Value::Int(3)));
    assert!(matches!(story.get("finish"), Some(Value::Text(f)) if f == "length" || f == "stop"));
    let stored = fs.resolve(&p("/state/story")).unwrap();
    assert_eq!(&fs.read(&root(&host), stored).unwrap(), story, "committed through the ordinary write");

    // The node holds exactly its invoke grant on the model's identity.
    let model_id = fs.resolve(&p("/models/local/stories/infer")).unwrap();
    let infer = &receipt.nodes[1];
    assert_eq!((infer.object, infer.uses.len()), (Some(model_id), 0));
    let grant = receipt.grants.iter().find(|g| Some(g.grant) == infer.grant).unwrap();
    assert_eq!((&grant.target, grant.rights), (&RecordedTarget::Object(model_id), Rights::INVOKE));
    let declared = infer.declared.as_ref().unwrap();
    assert_eq!((declared.purity, declared.determinism), (Purity::Effectful, Determinism::Nondeterministic));
    assert_eq!(declared.invocation.implementation.as_deref(), Some(model::local::IMPLEMENTATION));
    assert_retains_no_content(receipt);

    // Repeatable in this environment: same seed, same artifacts, same receipt.
    let (fs2, _) = with_local_model(3);
    assert_eq!(outcome(&fs2, STORY, ContentCapture::Omit).unwrap(), run);
}

#[test]
fn local_inference_failures_use_the_existing_machinery() {
    let (fs, host) = with_local_model(3);
    let before = snapshot(&fs, &host);
    // A system message is outside the supported completion subset.
    let src = STORY.replace("{ role: { literal: user }", "{ role: { literal: system }");
    let receipt = outcome(&fs, &src, ContentCapture::Omit).unwrap().receipt;
    assert_eq!(receipt.transaction, TxOutcome::RolledBack);
    let error = receipt.error.unwrap();
    let model_id = fs.resolve(&p("/models/local/stories/infer")).unwrap();
    assert_eq!(
        (error.node, error.object, error.kind),
        (Some(NodeId(1)), Some(model_id), ExecutionErrorKind::InvalidInput)
    );
    assert_eq!(receipt.nodes[2].state, NodeState::Blocked);
    assert_eq!(snapshot(&fs, &host), before);
    let invocations = receipt.events.iter().filter(|e| e.object == model_id).count();
    assert_eq!(invocations, 1, "invoked once, never retried");
}

#[test]
fn local_inferences_are_ordered_conservatively() {
    let (fs, _) = with_local_model(3);
    let src = r#"
agent: { name: a }
authority: [{ path: /models/local/stories/infer, rights: [invoke] }]
flow:
  - invoke: { path: /models/local/stories/infer, input: { messages: [{ role: user, content: a }], parameters: { max_tokens: 2 } } }
  - invoke: { path: /models/local/stories/infer, input: { messages: [{ role: user, content: b }], parameters: { max_tokens: 2 } } }
"#;
    let loaded =
        load(&fs, &policy(), Limits::default(), ContentCapture::Omit, &meatyaml::compile(src).unwrap()).unwrap();
    assert_eq!(loaded.derived_order(), &[(NodeId(0), NodeId(1))], "declared effectful: chained");
}

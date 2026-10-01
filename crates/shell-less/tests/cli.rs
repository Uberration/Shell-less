//! End-to-end checks against the real binary: diagnostics follow the host's
//! capture policy, results flow only through `--show-outputs`, and every
//! example exits as expected.

use std::process::Command;

const SENTINEL: &str = "zq-sentinel-4417";

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Output {
    fn all(&self) -> String {
        format!("{}{}", self.stdout, self.stderr)
    }
}

fn shell_less(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_shell-less")).args(args).output().unwrap();
    Output {
        code: out.status.code().expect("exited normally"),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Write `source` to a scratch file and run `command` on it.
fn on_source(name: &str, source: &str, command: &str, flags: &[&str]) -> Output {
    let file = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&file, source).unwrap();
    let file = file.to_str().unwrap();
    let mut args = vec![command];
    if command == "run" {
        args.extend(["--seed", "1"]);
    }
    args.extend(flags);
    args.push(file);
    shell_less(&args)
}

fn leaks(text: &str) -> bool {
    text.contains(SENTINEL) || text.contains(&SENTINEL.to_uppercase())
}

/// Programs that fail to compile, with the sentinel in the offending text.
fn compile_failures() -> Vec<(&'static str, String)> {
    let s = SENTINEL;
    vec![
        ("malformed", format!("agent: {{ name: a }}\nflow: [ {{ read: /x }}\n  {s}: ]\n")),
        ("invalid-path", format!("agent: {{ name: a }}\nflow: [ {{ read: /a/../{s} }} ]\n")),
        ("unknown-key", format!("agent: {{ name: a }}\n{s}: 1\nflow: [ {{ read: /x }} ]\n")),
        ("unknown-op", format!("agent: {{ name: a }}\nflow: [ {{ {s}: /x }} ]\n")),
        ("unknown-right", format!("agent: {{ name: a }}\nauthority: [{{ path: /x, rights: [{s}] }}]\nflow: [ {{ read: /x }} ]\n")),
        ("undeclared", format!("agent: {{ name: a }}\nflow: [ {{ read: /state/{s} }} ]\n")),
        (
            "invalid-value",
            format!("agent: {{ name: a }}\nflow:\n  - {{ id: x, compose: {{ literal: 1 }} }}\n  - {{ compose: {{ map: {{ {s}: {{ select: {{ from: x, path: [-1] }} }} }} }} }}\n"),
        ),
        ("bad-reference", format!("agent: {{ name: a }}\nflow: [ {{ compose: {{ select: {{ from: {s} }} }} }} ]\n")),
    ]
}

/// A valid program whose structure is full of the sentinel: agent name,
/// label, map key, selector key, written path, output name.
fn valid_program() -> String {
    let s = SENTINEL;
    format!(
        r#"
agent: {{ name: {s} }}
authority: [{{ path: /state, rights: [write] }}]
flow:
  - id: {s}
    compose: {{ literal: {{ {s}: {s} }} }}
  - id: pick
    compose:
      map:
        {s}:
          select: {{ from: {s}, path: [{s}] }}
  - write: {{ path: /state/{s}, from: pick }}
outputs:
  {s}: pick
"#
    )
}

#[test]
fn compile_errors_omit_source_text_in_check_and_run() {
    for (name, source) in compile_failures() {
        for command in ["check", "run"] {
            let out = on_source(&format!("{name}.meat.yaml"), &source, command, &[]);
            assert_eq!(out.code, 1, "{name} {command}: {}", out.all());
            assert!(out.stderr.contains("compile error kind="), "{name} {command}: {}", out.stderr);
            assert!(!leaks(&out.stdout) && !leaks(&out.stderr), "{name} {command} leaks: {}", out.all());
        }
    }
}

#[test]
fn compile_errors_locate_syntax_problems() {
    let (_, malformed) = &compile_failures()[0];
    let out = on_source("located.meat.yaml", malformed, "check", &[]);
    assert!(out.stderr.contains("kind=Syntax") && out.stderr.contains("line=3"), "{}", out.stderr);
}

#[test]
fn inline_capture_reveals_compile_diagnostics() {
    let (_, unknown_key) = compile_failures().into_iter().find(|(n, _)| *n == "unknown-key").unwrap();
    let out = on_source("inline-key.meat.yaml", &unknown_key, "check", &["--capture", "inline"]);
    assert!(leaks(&out.stderr), "{}", out.stderr);
}

#[test]
fn load_errors_omit_source_text() {
    let s = SENTINEL;
    let denied = format!(
        "agent: {{ name: a }}\nauthority: [{{ path: /memory/{s}, rights: [write] }}]\nflow: [ {{ write: {{ path: /memory/{s}, value: 1 }} }} ]\n"
    );
    let out = on_source("denied.meat.yaml", &denied, "run", &[]);
    assert_eq!(out.code, 1);
    assert!(out.stderr.contains("load error Authority: PolicyDenied"), "{}", out.stderr);
    assert!(!leaks(&out.all()), "{}", out.all());
    let out = on_source("denied-inline.meat.yaml", &denied, "run", &["--capture", "inline"]);
    assert!(leaks(&out.stderr), "{}", out.stderr);
}

#[test]
fn valid_structure_is_omitted_from_dumps() {
    for command in ["check", "run"] {
        let out = on_source(&format!("valid-{command}.meat.yaml"), &valid_program(), command, &[]);
        assert_eq!(out.code, 0, "{}", out.all());
        assert!(!leaks(&out.all()), "{command} leaks: {}", out.all());
        let out = on_source(
            &format!("valid-{command}-inline.meat.yaml"),
            &valid_program(),
            command,
            &["--capture", "inline"],
        );
        assert!(leaks(&out.stdout), "{}", out.stdout);
    }
}

#[test]
fn show_outputs_opens_only_the_result_channel() {
    let out = on_source("shown.meat.yaml", &valid_program(), "run", &["--show-outputs"]);
    assert_eq!(out.code, 0, "{}", out.all());
    let (diagnostics, outputs) = out.stdout.split_once("── outputs").unwrap();
    assert!(leaks(outputs), "{outputs}");
    assert!(!leaks(diagnostics) && !leaks(&out.stderr), "{}", out.all());
}

#[test]
fn model_error_text_stays_out_of_diagnostics() {
    // The mock model's error message quotes an unknown instruction.
    let s = SENTINEL;
    let program = format!(
        "agent: {{ name: a }}\nauthority: [{{ path: /models, rights: [invoke] }}]\nflow:\n  - invoke:\n      path: /models/mock/infer\n      input: {{ messages: [{{ role: system, content: {s} }}, {{ role: user, content: x }}] }}\n"
    );
    let out = on_source("model-error.meat.yaml", &program, "run", &[]);
    assert_eq!(out.code, 1);
    assert!(out.stdout.contains("InvalidInput"), "{}", out.stdout);
    assert!(!leaks(&out.all()), "{}", out.all());
}

#[test]
fn examples_exit_as_expected() {
    let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples");
    let table = [
        ("butcher", "run", 0),
        ("composer", "run", 0),
        ("echo", "run", 0),
        ("fork", "run", 0),
        ("scout", "run", 0),
        ("thinker", "run", 0),
        ("failure", "run", 1),
        ("model-failure", "run", 1),
        ("escalate-declared", "run", 1),
        ("escalate", "check", 1),
        ("escalate", "run", 1),
        ("story", "check", 0),
        ("story", "run", 1), // no host artifacts: nothing mounted
    ];
    let mut seen: Vec<String> = std::fs::read_dir(examples)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|f| f.strip_suffix(".meat.yaml").map(str::to_owned))
        .collect();
    seen.sort();
    let mut listed: Vec<String> = table.iter().map(|(n, _, _)| (*n).to_owned()).collect();
    listed.sort();
    listed.dedup();
    assert_eq!(seen, listed, "every example has an expected exit code");

    for (name, command, expected) in table {
        let file = format!("{examples}/{name}.meat.yaml");
        let mut args = vec![command];
        if command == "run" {
            args.extend(["--seed", "1"]);
        }
        args.push(&file);
        let out = shell_less(&args);
        assert_eq!(out.code, expected, "{command} {name}: {}", out.all());
    }
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(shell_less(&[]).code, 2);
    assert_eq!(shell_less(&["run", "--capture", "everything", "x"]).code, 2);
}

// ── M5: host-supplied model artifacts ───────────────────────────────────────

const STORY: &str = r#"
agent: { name: storyteller }
authority:
  - { path: /models/local/stories/infer, rights: [invoke] }
  - { path: /state/story, rights: [write] }
flow:
  - id: infer
    invoke:
      path: /models/local/stories/infer
      input: { messages: [{ role: user, content: hello world }], parameters: { max_tokens: 5 } }
  - id: store
    write: { path: /state/story, from: infer }
outputs:
  story: store
"#;

/// Fixture artifacts written to scratch files; returns their paths.
fn fixture_artifacts(tag: &str, checkpoint: &[u8]) -> (String, String) {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let (c, t) = (dir.join(format!("{tag}-checkpoint.bin")), dir.join(format!("{tag}-tokenizer.bin")));
    std::fs::write(&c, checkpoint).unwrap();
    std::fs::write(&t, model::local::fixture::tokenizer()).unwrap();
    (c.to_str().unwrap().to_owned(), t.to_str().unwrap().to_owned())
}

fn good_checkpoint() -> Vec<u8> {
    model::local::fixture::checkpoint(model::local::fixture::tiny_spec())
}

#[test]
fn host_mounts_validated_local_model() {
    let (c, t) = fixture_artifacts("good", &good_checkpoint());
    let out = on_source("story.meat.yaml", STORY, "run", &["--checkpoint", &c, "--tokenizer", &t, "--show-outputs"]);
    assert_eq!(out.code, 0, "{}", out.all());
    assert!(
        out.stdout
            .contains("declared Effectful + Nondeterministic  shell-less/llama2c-v0-f32-scalar@shell-less-legacy-v0-completion/2;checkpoint=fnv1a64:"),
        "{}",
        out.stdout
    );
    let outputs = out.stdout.split("── outputs").nth(1).unwrap();
    assert!(outputs.contains(r#""input_tokens":3"#), "{outputs}");
    assert!(!out.all().contains(&c) && !out.all().contains(&t), "no host path in diagnostics or metadata");
}

#[test]
fn startup_failures_omit_host_paths() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(SENTINEL);
    let missing = dir.join("stories15M.bin");
    let missing = missing.to_str().unwrap();
    let out = on_source("story-missing.meat.yaml", STORY, "run", &["--checkpoint", missing, "--tokenizer", missing]);
    assert_eq!(out.code, 1);
    assert!(out.stderr.contains("Checkpoint: io error (NotFound) (file <path>)"), "{}", out.stderr);
    assert!(!leaks(&out.all()), "{}", out.all());
    let out = on_source(
        "story-missing-inline.meat.yaml",
        STORY,
        "run",
        &["--checkpoint", missing, "--tokenizer", missing, "--capture", "inline"],
    );
    assert!(leaks(&out.stderr), "{}", out.stderr);
}

#[test]
fn invalid_artifacts_fail_before_anything_is_mounted() {
    let truncated = good_checkpoint();
    let (c, t) = fixture_artifacts("truncated", &truncated[..truncated.len() - 4]);
    let out = on_source("story-truncated.meat.yaml", STORY, "run", &["--checkpoint", &c, "--tokenizer", &t]);
    assert_eq!(out.code, 1);
    assert!(out.stderr.contains("Checkpoint: truncated"), "{}", out.stderr);
    assert!(!out.stdout.contains("── receipt"), "nothing ran: {}", out.stdout);

    // Without artifacts there is nothing at the model's path to invoke.
    let out = on_source("story-unmounted.meat.yaml", STORY, "run", &[]);
    assert_eq!(out.code, 1);
    assert!(out.stderr.contains("load error Authority: Unbound"), "{}", out.stderr);
}

#[test]
fn artifacts_come_as_a_pair() {
    let (c, _) = fixture_artifacts("lonely", &good_checkpoint());
    let out = on_source("story-lonely.meat.yaml", STORY, "run", &["--checkpoint", &c]);
    assert_eq!(out.code, 2);
}

/// The trained-model acceptance run through a graph: the committed write and
/// explicit output. Needs `SHELL_LESS_CHECKPOINT` and `SHELL_LESS_TOKENIZER`;
/// run with `--ignored --nocapture`. Missing artifacts fail the test.
#[test]
#[ignore]
fn trained_model_through_a_graph() {
    let checkpoint = std::env::var("SHELL_LESS_CHECKPOINT").expect("set SHELL_LESS_CHECKPOINT");
    let tokenizer = std::env::var("SHELL_LESS_TOKENIZER").expect("set SHELL_LESS_TOKENIZER");
    let story = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/story.meat.yaml");
    let args = ["run", "--seed", "1", "--checkpoint", &checkpoint, "--tokenizer", &tokenizer, "--show-outputs", story];
    let out = shell_less(&args);
    eprintln!("{}", out.all());
    assert_eq!(out.code, 0);
    assert!(out.stdout.contains("transaction Committed"));
    let outputs = out.stdout.split("── outputs").nth(1).expect("outputs section");
    assert!(outputs.contains(r#""input_tokens":5"#), "{outputs}");
}

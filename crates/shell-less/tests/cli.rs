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

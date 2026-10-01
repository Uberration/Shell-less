//! End-to-end: default CLI diagnostics never print payloads.

use std::process::Command;

const SENTINEL: &str = "zq-sentinel-4417";

fn program(system: &str) -> String {
    format!(
        r#"
agent: {{ name: secretive }}
authority:
  - {{ path: /models/mock/infer, rights: [invoke] }}
  - {{ path: /state/secret, rights: [write] }}
flow:
  - id: source
    compose: {{ literal: {{ text: {SENTINEL} }} }}
  - id: request
    compose:
      map:
        messages:
          list:
            - map: {{ role: {{ literal: system }}, content: {{ literal: "{system}" }} }}
            - map: {{ role: {{ literal: user }}, content: {{ select: {{ from: source, path: [text] }} }} }}
  - id: infer
    invoke: {{ path: /models/mock/infer, from: request }}
  - id: store
    write: {{ path: /state/secret, from: infer }}
outputs:
  answer: store
"#
    )
}

fn run(name: &str, source: &str, flags: &[&str]) -> String {
    let file = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&file, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_shell-less"))
        .args(["run", "--seed", "1"])
        .args(flags)
        .arg(&file)
        .output()
        .unwrap();
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn leaks(output: &str) -> bool {
    output.contains(SENTINEL) || output.contains(&SENTINEL.to_uppercase())
}

#[test]
fn default_diagnostics_omit_payloads() {
    let ok = run("ok.meat.yaml", &program("uppercase"), &[]);
    assert!(ok.contains("transaction Committed"), "{ok}");
    assert!(!leaks(&ok), "{ok}");

    // The mock model's error message quotes the unknown instruction.
    let failed = run("fail.meat.yaml", &program(SENTINEL), &[]);
    assert!(failed.contains("InvalidInput"), "{failed}");
    assert!(!leaks(&failed), "{failed}");
}

#[test]
fn results_and_inline_capture_are_explicit() {
    let shown = run("shown.meat.yaml", &program("uppercase"), &["--show-outputs"]);
    let outputs = shown.split("── outputs").nth(1).unwrap();
    assert!(outputs.contains(&SENTINEL.to_uppercase()), "{shown}");
    assert!(!leaks(shown.split("── outputs").next().unwrap()), "only the outputs section shows results");

    let inline = run("inline.meat.yaml", &program(SENTINEL), &["--capture", "inline"]);
    assert!(leaks(&inline), "{inline}");
}

//! Deterministic model-shaped capabilities. Not intelligent; exact.

use crate::{FinishReason, InferRequest, InferResponse, Message, Role, Usage};
use capability::{Capability, CapabilityContext, CapabilityMeta, Determinism, Fault, Purity};

/// Whitespace-separated words stand in for tokens.
fn tokens(text: &str) -> i64 {
    text.split_whitespace().count() as i64
}

/// A mock model: the last system message names a transformation
/// (`uppercase`, `lowercase`, `reverse`, or none for echo), applied to the
/// last user message. `max_tokens` truncates by words.
///
/// Inference is declared effectful even here: real models depend on
/// weights, samplers, hardware and caches, and the contract does not
/// pretend otherwise. This implementation is, however, deterministic.
pub struct MockModel;

impl MockModel {
    pub const IMPLEMENTATION: &'static str = "mock-transform";
    pub const REVISION: &'static str = "1";
}

impl Capability for MockModel {
    type Input = InferRequest;
    type Output = InferResponse;

    fn describe(&self) -> &'static str {
        "deterministic mock model: system names a transform applied to the last user message"
    }

    fn meta(&self) -> CapabilityMeta {
        CapabilityMeta::new(Purity::Effectful, Determinism::Deterministic)
            .implemented_by(Self::IMPLEMENTATION, Self::REVISION)
    }

    fn invoke(&self, _: &CapabilityContext<'_>, request: InferRequest) -> Result<InferResponse, Fault> {
        let last = |role| request.messages.iter().rev().find(|m| m.role == role).map(|m| m.content.as_str());
        let prompt = last(Role::User).ok_or_else(|| Fault::InvalidInput("no user message".to_owned()))?;
        let transformed = match last(Role::System) {
            None => prompt.to_owned(),
            Some("uppercase") => prompt.to_uppercase(),
            Some("lowercase") => prompt.to_lowercase(),
            Some("reverse") => prompt.chars().rev().collect(),
            Some(other) => return Err(Fault::InvalidInput(format!("unknown mock instruction `{other}`"))),
        };
        let (content, finish) = match request.parameters.max_tokens {
            Some(max) if tokens(&transformed) > max => {
                let kept: Vec<_> = transformed.split_whitespace().take(max.max(0) as usize).collect();
                (kept.join(" "), FinishReason::Length)
            }
            _ => (transformed, FinishReason::Stop),
        };
        let input_tokens = request.messages.iter().map(|m| tokens(&m.content)).sum();
        let usage = Usage { input_tokens, output_tokens: tokens(&content) };
        Ok(InferResponse { message: Message::new(Role::Assistant, content), usage: Some(usage), finish })
    }
}

/// A mock model that always fails, deterministically.
pub struct MockFail;

impl Capability for MockFail {
    type Input = InferRequest;
    type Output = InferResponse;

    fn describe(&self) -> &'static str {
        "deterministic mock model that always fails"
    }

    fn meta(&self) -> CapabilityMeta {
        CapabilityMeta::new(Purity::Effectful, Determinism::Deterministic).implemented_by("mock-fail", "1")
    }

    fn invoke(&self, _: &CapabilityContext<'_>, _: InferRequest) -> Result<InferResponse, Fault> {
        Err(Fault::Failed("mock model failure".to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use meatfs::{MeatFs, Path, Seed, Value};

    fn request(system: Option<&str>, user: &str, max_tokens: Option<i64>) -> Value {
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(Value::map([("role", Value::from("system")), ("content", Value::from(system))]));
        }
        messages.push(Value::map([("role", Value::from("user")), ("content", Value::from(user))]));
        let mut fields = vec![("messages", Value::List(messages))];
        if let Some(max) = max_tokens {
            fields.push(("parameters", Value::map([("max_tokens", Value::Int(max))])));
        }
        Value::map(fields)
    }

    fn infer(input: Value) -> meatfs::Result<Value> {
        let (fs, host) = MeatFs::genesis(Seed::fixed(0));
        let root = host.access(host.iter().next().unwrap().id()).unwrap();
        let model = capability::mount(&fs, &root, &Path::parse("/models/mock/infer").unwrap(), MockModel).unwrap();
        fs.invoke(&root, model, input)
    }

    #[test]
    fn transforms_deterministically() {
        let out = infer(request(Some("uppercase"), "meat", None)).unwrap();
        assert_eq!(out.get("message").unwrap().to_string(), r#"{"content":"MEAT","role":"assistant"}"#);
        assert_eq!(out.get("finish"), Some(&Value::from("stop")));
        assert_eq!(out.get("usage").unwrap().to_string(), r#"{"input_tokens":2,"output_tokens":1}"#);
        assert_eq!(infer(request(Some("uppercase"), "meat", None)).unwrap(), out);
    }

    #[test]
    fn max_tokens_truncates() {
        let out = infer(request(None, "one two three", Some(2))).unwrap();
        assert_eq!(out.get("message").unwrap().get("content"), Some(&Value::from("one two")));
        assert_eq!(out.get("finish"), Some(&Value::from("length")));
    }

    #[test]
    fn rejects_malformed_requests() {
        assert!(matches!(infer(request(Some("summon"), "x", None)), Err(meatfs::Error::InvalidInput { .. })));
        assert!(matches!(
            infer(Value::map([("messages", Value::List(vec![]))])),
            Err(meatfs::Error::InvalidInput { .. })
        ));
        let bad_role = Value::map([(
            "messages",
            Value::List(vec![Value::map([("role", Value::from("oracle")), ("content", Value::from("x"))])]),
        )]);
        assert!(matches!(infer(bad_role), Err(meatfs::Error::InvalidInput { .. })));
    }
}

//! The model capability contract: models are ordinary typed capabilities.
//!
//! This crate owns the shape of inference — [`InferRequest`] →
//! [`InferResponse`] — deterministic mock implementations, and one local
//! reference backend ([`local`]). It has no orchestration, memory, agent
//! loop, provider, transport or prompt templating, and the runtime knows
//! nothing about it: a model is mounted at a path such as
//! `/models/local/stories/infer` and invoked like any capability.

pub mod local;
mod mock;

pub use mock::{MockFail, MockModel};

use capability::{Fields, FromValue, IntoValue, Result};
use meatfs::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl FromValue for Role {
    fn from_value(value: Value) -> Result<Self> {
        match value.as_text() {
            Some("system") => Ok(Role::System),
            Some("user") => Ok(Role::User),
            Some("assistant") => Ok(Role::Assistant),
            _ => Err(format!("expected role system|user|assistant, got {value}")),
        }
    }
    fn schema() -> Value {
        Value::from("system|user|assistant")
    }
}

impl IntoValue for Role {
    fn into_value(self) -> Value {
        Value::from(self.as_str())
    }
    fn schema() -> Value {
        Value::from("system|user|assistant")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message { role, content: content.into() }
    }
}

impl FromValue for Message {
    fn from_value(value: Value) -> Result<Self> {
        let mut fields = Fields::of(value)?;
        let message = Message { role: fields.take("role")?, content: fields.take("content")? };
        fields.finish()?;
        Ok(message)
    }
    fn schema() -> Value {
        Value::map([("role", <Role as FromValue>::schema()), ("content", Value::from("text"))])
    }
}

impl IntoValue for Message {
    fn into_value(self) -> Value {
        Value::map([("role", self.role.into_value()), ("content", Value::Text(self.content))])
    }
    fn schema() -> Value {
        <Message as FromValue>::schema()
    }
}

/// Sampling and length controls. Deliberately tiny.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferParameters {
    /// Upper bound on generated tokens.
    pub max_tokens: Option<i64>,
}

impl FromValue for InferParameters {
    fn from_value(value: Value) -> Result<Self> {
        let mut fields = Fields::of(value)?;
        let parameters = InferParameters { max_tokens: fields.take_optional("max_tokens")? };
        fields.finish()?;
        Ok(parameters)
    }
    fn schema() -> Value {
        Value::map([("max_tokens", Value::from("int?"))])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferRequest {
    pub messages: Vec<Message>,
    pub parameters: InferParameters,
}

impl FromValue for InferRequest {
    fn from_value(value: Value) -> Result<Self> {
        let mut fields = Fields::of(value)?;
        let request = InferRequest {
            messages: fields.take("messages")?,
            parameters: fields.take_optional("parameters")?.unwrap_or_default(),
        };
        fields.finish()?;
        Ok(request)
    }
    fn schema() -> Value {
        Value::map([
            ("messages", <Vec<Message> as FromValue>::schema()),
            ("parameters", <InferParameters as FromValue>::schema()),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

impl IntoValue for Usage {
    fn into_value(self) -> Value {
        Value::map([("input_tokens", Value::Int(self.input_tokens)), ("output_tokens", Value::Int(self.output_tokens))])
    }
    fn schema() -> Value {
        Value::map([("input_tokens", Value::from("int")), ("output_tokens", Value::from("int"))])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model finished naturally.
    Stop,
    /// Generation hit `max_tokens`.
    Length,
}

impl IntoValue for FinishReason {
    fn into_value(self) -> Value {
        Value::from(match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        })
    }
    fn schema() -> Value {
        Value::from("stop|length")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferResponse {
    pub message: Message,
    pub usage: Option<Usage>,
    pub finish: FinishReason,
}

impl IntoValue for InferResponse {
    fn into_value(self) -> Value {
        Value::map([
            ("message", self.message.into_value()),
            ("usage", self.usage.map_or(Value::Null, IntoValue::into_value)),
            ("finish", self.finish.into_value()),
        ])
    }
    fn schema() -> Value {
        Value::map([
            ("message", <Message as IntoValue>::schema()),
            ("usage", <Usage as IntoValue>::schema()),
            ("finish", <FinishReason as IntoValue>::schema()),
        ])
    }
}

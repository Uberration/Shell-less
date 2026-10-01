//! The first native capabilities. Pure, deterministic, no I/O.

use crate::{Capability, CapabilityContext, Fields, FromValue, IntoValue, Result};
use meatfs::Value;

/// `{ text: text }`
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub text: String,
}

impl Text {
    pub fn schema_value() -> Value {
        Value::map([("text", Value::from("text"))])
    }
}

impl FromValue for Text {
    fn from_value(value: Value) -> Result<Self> {
        let mut fields = Fields::of(value)?;
        let text = fields.take("text")?;
        fields.finish()?;
        Ok(Text { text })
    }
    fn schema() -> Value {
        Text::schema_value()
    }
}

impl IntoValue for Text {
    fn into_value(self) -> Value {
        Value::map([("text", Value::Text(self.text))])
    }
    fn schema() -> Value {
        Text::schema_value()
    }
}

/// Returns its input unchanged.
pub struct Echo;

impl Capability for Echo {
    type Input = Text;
    type Output = Text;

    fn describe(&self) -> &'static str {
        "return the input text unchanged"
    }

    fn invoke(&self, _: &CapabilityContext<'_>, input: Text) -> Result<Text> {
        Ok(input)
    }
}

/// Uppercases its input.
pub struct Upper;

impl Capability for Upper {
    type Input = Text;
    type Output = Text;

    fn describe(&self) -> &'static str {
        "uppercase the input text"
    }

    fn invoke(&self, _: &CapabilityContext<'_>, input: Text) -> Result<Text> {
        Ok(Text { text: input.text.to_uppercase() })
    }
}

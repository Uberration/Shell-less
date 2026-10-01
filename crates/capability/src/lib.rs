//! Typed capabilities: the only way an agent acts on the world.
//!
//! A capability has a typed input, a typed output and a published contract.
//! It is mounted into MeatFS at a path and invoked through the namespace, so
//! authority checks and auditing apply uniformly. There is deliberately no
//! "run this string" capability.

pub mod builtin;

use meatfs::{Access, Invoke, MeatFs, ObjectId, Path, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

pub use meatfs::{CallContext as CapabilityContext, CapabilityMeta, Fault, Purity};

pub type Result<T, E = String> = std::result::Result<T, E>;

/// Conversion from the untyped [`Value`] crossing the namespace boundary.
pub trait FromValue: Sized {
    fn from_value(value: Value) -> Result<Self>;
    /// Type description published by `inspect`.
    fn schema() -> Value;
}

/// Conversion into the untyped [`Value`] crossing the namespace boundary.
pub trait IntoValue {
    fn into_value(self) -> Value;
    fn schema() -> Value;
}

pub trait Capability: Send + Sync + 'static {
    type Input: FromValue;
    type Output: IntoValue;

    /// One line stating what this capability does.
    fn describe(&self) -> &'static str;

    /// Declared explicitly by every capability; there is no default.
    fn meta(&self) -> CapabilityMeta;

    fn invoke(&self, ctx: &CapabilityContext<'_>, input: Self::Input) -> Result<Self::Output, Fault>;
}

/// Adapts a typed [`Capability`] into an invocable MeatFS object.
struct Typed<C>(C);

impl<C: Capability> Invoke for Typed<C> {
    fn meta(&self) -> CapabilityMeta {
        self.0.meta()
    }

    fn invoke(&self, ctx: &CapabilityContext<'_>, input: Value) -> Result<Value, Fault> {
        let input = C::Input::from_value(input).map_err(Fault::InvalidInput)?;
        self.0.invoke(ctx, input).map(IntoValue::into_value)
    }

    fn signature(&self) -> Value {
        Value::map([
            ("describe", Value::from(self.0.describe())),
            ("pure", Value::Bool(self.0.meta().purity == Purity::Pure)),
            ("input", C::Input::schema()),
            ("output", C::Output::schema()),
        ])
    }
}

/// Mount a typed capability at `path`, returning its identity.
pub fn mount<C: Capability>(fs: &MeatFs, access: &Access<'_>, path: &Path, capability: C) -> meatfs::Result<ObjectId> {
    fs.mount(access, path, Arc::new(Typed(capability)))
}

/// Field access over a map-shaped input, for hand-written [`FromValue`] impls.
pub struct Fields(BTreeMap<String, Value>);

impl Fields {
    pub fn of(value: Value) -> Result<Self> {
        match value {
            Value::Map(map) => Ok(Fields(map)),
            other => Err(format!("expected map, got {}", other.kind())),
        }
    }

    pub fn take<T: FromValue>(&mut self, key: &str) -> Result<T> {
        let value = self.0.remove(key).ok_or_else(|| format!("missing field `{key}`"))?;
        T::from_value(value).map_err(|e| format!("field `{key}`: {e}"))
    }

    /// Reject unknown fields, so typos never silently pass.
    pub fn finish(self) -> Result<()> {
        match self.0.keys().next() {
            None => Ok(()),
            Some(key) => Err(format!("unknown field `{key}`")),
        }
    }
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, $name:literal) => {
        impl FromValue for $ty {
            fn from_value(value: Value) -> Result<Self> {
                match value {
                    Value::$variant(v) => Ok(v),
                    other => Err(format!("expected {}, got {}", $name, other.kind())),
                }
            }
            fn schema() -> Value {
                Value::from($name)
            }
        }
        impl IntoValue for $ty {
            fn into_value(self) -> Value {
                Value::$variant(self)
            }
            fn schema() -> Value {
                Value::from($name)
            }
        }
    };
}

scalar!(String, Text, "text");
scalar!(i64, Int, "int");
scalar!(bool, Bool, "bool");

impl FromValue for Value {
    fn from_value(value: Value) -> Result<Self> {
        Ok(value)
    }
    fn schema() -> Value {
        Value::from("any")
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
    fn schema() -> Value {
        Value::from("any")
    }
}

#[cfg(test)]
mod tests {
    use super::builtin::{Echo, Text};
    use super::*;
    use meatfs::Seed;

    fn echo() -> (MeatFs, meatfs::GrantSet, ObjectId) {
        let (fs, host) = MeatFs::genesis(Seed::fixed(0));
        let root = host.access(host.iter().next().unwrap().id()).unwrap();
        let id = mount(&fs, &root, &Path::parse("/tools/echo").unwrap(), Echo).unwrap();
        (fs, host, id)
    }

    #[test]
    fn typed_round_trip_and_signature() {
        let (fs, host, echo) = echo();
        let root = host.access(host.iter().next().unwrap().id()).unwrap();
        let out = fs.invoke(&root, echo, Value::map([("text", Value::from("MEAT"))])).unwrap();
        assert_eq!(out, Value::map([("text", Value::from("MEAT"))]));

        let inspection = fs.inspect(&root, echo).unwrap();
        assert_eq!(inspection.meta, Some(CapabilityMeta { purity: Purity::Pure }));
        assert_eq!(inspection.signature.get("input"), Some(&Text::schema_value()));
    }

    #[test]
    fn rejects_ill_typed_input() {
        let (fs, host, echo) = echo();
        let root = host.access(host.iter().next().unwrap().id()).unwrap();
        for bad in [
            Value::from("bare"),
            Value::map([("text", Value::Int(1))]),
            Value::map([("text", Value::from("a")), ("extra", Value::Null)]),
        ] {
            assert!(matches!(fs.invoke(&root, echo, bad), Err(meatfs::Error::InvalidInput { .. })));
        }
    }
}

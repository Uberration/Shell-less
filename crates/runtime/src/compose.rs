//! Evaluation of `compose` nodes: pure, bounded value construction.

use crate::ExecutionErrorKind;
use meatfs::Value;
use meatyaml::{Selector, ValueExpr};

/// Approximate in-memory size of a value, in bytes-ish units: one per
/// value plus the length of every string and key.
pub(crate) fn size(value: &Value) -> usize {
    match value {
        Value::Text(s) => 1 + s.len(),
        Value::List(items) => 1 + items.iter().map(size).sum::<usize>(),
        Value::Map(fields) => 1 + fields.iter().map(|(k, v)| k.len() + size(v)).sum::<usize>(),
        _ => 1,
    }
}

/// Why a composition failed. `detail` may quote program text, never payload
/// values; it is still only kept under inline capture.
pub(crate) struct ComposeFault {
    pub kind: ExecutionErrorKind,
    pub detail: String,
}

fn fault(kind: ExecutionErrorKind, detail: impl Into<String>) -> ComposeFault {
    ComposeFault { kind, detail: detail.into() }
}

fn spend(budget: &mut usize, cost: usize) -> Result<(), ComposeFault> {
    *budget = budget
        .checked_sub(cost)
        .ok_or_else(|| fault(ExecutionErrorKind::LimitExceeded, "constructed value too large"))?;
    Ok(())
}

/// Build the value `expr` describes. `output(source)` yields the output of a
/// node this expression depends on; the scheduler guarantees it exists.
/// Every byte placed into the result is charged against `budget` before it
/// is copied.
pub(crate) fn evaluate<'v>(
    expr: &ValueExpr,
    output: &dyn Fn(meatfs::NodeId) -> &'v Value,
    budget: &mut usize,
) -> Result<Value, ComposeFault> {
    match expr {
        ValueExpr::Literal(value) => {
            spend(budget, size(value))?;
            Ok(value.clone())
        }
        ValueExpr::Select { source, path } => {
            let mut value = output(*source);
            for (depth, selector) in path.iter().enumerate() {
                value = match (selector, value) {
                    (Selector::Key(key), Value::Map(fields)) => fields.get(key).ok_or_else(|| {
                        fault(ExecutionErrorKind::MissingValue, format!("{source} path[{depth}]: no key `{key}`"))
                    })?,
                    (Selector::Index(index), Value::List(items)) => items.get(*index).ok_or_else(|| {
                        fault(ExecutionErrorKind::IndexOutOfRange, format!("{source} path[{depth}]: no index {index}"))
                    })?,
                    (Selector::Key(_), other) => {
                        return Err(fault(
                            ExecutionErrorKind::WrongValueKind,
                            format!("{source} path[{depth}]: expected map, found {}", other.kind()),
                        ))
                    }
                    (Selector::Index(_), other) => {
                        return Err(fault(
                            ExecutionErrorKind::WrongValueKind,
                            format!("{source} path[{depth}]: expected list, found {}", other.kind()),
                        ))
                    }
                };
            }
            spend(budget, size(value))?;
            Ok(value.clone())
        }
        ValueExpr::Map(fields) => {
            spend(budget, 1)?;
            let mut out = std::collections::BTreeMap::new();
            for (key, expr) in fields {
                spend(budget, key.len())?;
                out.insert(key.clone(), evaluate(expr, output, budget)?);
            }
            Ok(Value::Map(out))
        }
        ValueExpr::List(items) => {
            spend(budget, 1)?;
            items.iter().map(|e| evaluate(e, output, budget)).collect::<Result<_, _>>().map(Value::List)
        }
    }
}

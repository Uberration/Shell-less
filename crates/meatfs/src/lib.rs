//! MeatFS: the canonical semantic namespace of Shell-less.
//!
//! Everything an agent can touch — memory, state, tools, models, jobs — is an
//! object at a path. Objects expose a subset of typed operations
//! (`read`, `write`, `invoke`, `subscribe`, `inspect`), and every operation is
//! checked against an explicit [`Authority`]. OS mounts, 9P, remote transports
//! and shared memory are adapters over this model, not the model itself.

mod error;
mod fs;
mod path;
mod rights;
mod value;

pub use error::{Error, Result};
pub use fs::{CallContext, Event, EventKind, Inspection, Invoke, MeatFs, NodeKind, Subscription, Transaction};
pub use path::Path;
pub use rights::{Authority, Grant, Rights};
pub use value::Value;

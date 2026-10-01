//! MeatFS: the canonical semantic namespace of Shell-less.
//!
//! Objects have immutable identity ([`ObjectId`]); paths are names bound to
//! them. Objects expose a subset of typed operations (`read`, `write`,
//! `invoke`, `subscribe`, `inspect`), and every operation presents a grant
//! issued by the namespace itself — knowing a path confers nothing. State
//! changes made on behalf of a graph go through a [`Transaction`]; the audit
//! journal records attempts and outcomes independently of whether state
//! commits. OS mounts, 9P, remote transports and shared memory are adapters
//! over this model, not the model itself.

mod authority;
mod error;
mod fs;
mod id;
mod path;
mod rights;
mod value;

pub use authority::{
    Access, AuthorityRequest, Cause, Constraints, Grant, GrantSet, NodeGrant, NodeGrants, Policy, Target,
};
pub use error::{Error, Result};
pub use fs::{
    CallContext, CapabilityMeta, Effects, Event, EventKind, Fault, Inspection, Invoke, MeatFs, NodeKind, Purity,
    Subscription, Transaction,
};
pub use id::{AuthorityDomainId, ExecutionId, GrantId, NodeId, ObjectId, PrincipalId, Seed};
pub use path::Path;
pub use rights::Rights;
pub use value::Value;

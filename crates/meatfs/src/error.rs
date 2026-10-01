use crate::{GrantId, ObjectId, Path, Rights};
use std::fmt;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidPath(String),
    /// No object is bound at this name.
    Unbound(Path),
    /// The name is taken, or would nest inside / around an existing name.
    AlreadyBound(Path),
    UnknownObject(ObjectId),
    /// The object does not support this operation.
    Unsupported {
        object: ObjectId,
        op: &'static str,
    },
    /// The presented grant does not cover the operation.
    Denied {
        object: ObjectId,
        needed: Rights,
    },
    /// The grant was not issued by this authority domain, or was retired.
    InvalidGrant(GrantId),
    /// Host policy forbids requesting this authority.
    PolicyDenied {
        path: Path,
        rights: Rights,
    },
    /// A capability asked for authority not attached to its node.
    NotAttached(Path),
    /// A capability declared pure attempted an effect.
    Impure(ObjectId),
    /// A capability rejected its input.
    InvalidInput {
        object: ObjectId,
        message: String,
    },
    /// A capability failed while executing.
    Capability {
        object: ObjectId,
        message: String,
    },
    /// A transaction's precondition no longer holds.
    Conflict(ObjectId),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidPath(p) => write!(f, "invalid path `{p}`"),
            Error::Unbound(p) => write!(f, "{p}: nothing bound"),
            Error::AlreadyBound(p) => write!(f, "{p}: name conflicts with an existing binding"),
            Error::UnknownObject(o) => write!(f, "{o}: no such object"),
            Error::Unsupported { object, op } => write!(f, "{object}: does not support `{op}`"),
            Error::Denied { object, needed } => write!(f, "{object}: denied, requires {needed}"),
            Error::InvalidGrant(g) => write!(f, "{g}: not a live grant of this authority domain"),
            Error::PolicyDenied { path, rights } => write!(f, "{path}: policy forbids requesting {rights}"),
            Error::NotAttached(p) => write!(f, "{p}: no authority attached to this node"),
            Error::Impure(o) => write!(f, "{o}: pure capability attempted an effect"),
            Error::InvalidInput { object, message } => write!(f, "{object}: invalid input: {message}"),
            Error::Capability { object, message } => write!(f, "{object}: {message}"),
            Error::Conflict(o) => write!(f, "{o}: changed during transaction"),
        }
    }
}

impl std::error::Error for Error {}

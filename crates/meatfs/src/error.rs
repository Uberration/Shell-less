use crate::{Path, Rights};
use std::fmt;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidPath(String),
    NotFound(Path),
    AlreadyExists(Path),
    /// The operation is not supported by the object at this path.
    Unsupported {
        path: Path,
        op: &'static str,
    },
    Denied {
        path: Path,
        needed: Rights,
    },
    /// A capability rejected its input or failed while executing.
    Capability {
        path: Path,
        message: String,
    },
    /// A transaction's precondition no longer holds.
    Conflict(Path),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidPath(p) => write!(f, "invalid path `{p}`"),
            Error::NotFound(p) => write!(f, "{p}: not found"),
            Error::AlreadyExists(p) => write!(f, "{p}: already exists"),
            Error::Unsupported { path, op } => write!(f, "{path}: does not support `{op}`"),
            Error::Denied { path, needed } => write!(f, "{path}: denied, requires {needed}"),
            Error::Capability { path, message } => write!(f, "{path}: {message}"),
            Error::Conflict(p) => write!(f, "{p}: changed during transaction"),
        }
    }
}

impl std::error::Error for Error {}

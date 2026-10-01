//! The authority subsystem.
//!
//! Two things are kept deliberately distinct:
//!
//! * [`Policy`] — path-prefix rules describing what *may be requested*. Plain
//!   data, freely constructed by the host.
//! * [`Grant`] — an authority instance *actually issued* by a namespace. It
//!   cannot be constructed, cloned or mutated outside this crate, and the
//!   issuing namespace keeps a registry of live grants, so a grant is only
//!   honoured by the namespace that issued it and only until retired.
//!
//! Knowing a pathname confers nothing: every operation presents an [`Access`]
//! naming one grant from the holder's [`GrantSet`].

use crate::{ExecutionId, GrantId, NodeId, ObjectId, Path, PrincipalId, Rights};

/// What a grant points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// One object, by identity, wherever it is bound.
    Object(ObjectId),
    /// Every object bound beneath a prefix, plus the right to bind new names
    /// there. Issued only to the host at genesis.
    Namespace(Path),
}

/// Restrictions on how a grant may be exercised. Empty for now; this is
/// where expiry, use counts and execution pinning will live.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Constraints {}

/// An issued authority instance. Unforgeable outside `meatfs`.
#[derive(Debug, PartialEq, Eq)]
pub struct Grant {
    id: GrantId,
    subject: PrincipalId,
    target: Target,
    rights: Rights,
    constraints: Constraints,
}

impl Grant {
    pub(crate) fn new(id: GrantId, subject: PrincipalId, target: Target, rights: Rights) -> Grant {
        Grant { id, subject, target, rights, constraints: Constraints::default() }
    }

    pub fn id(&self) -> GrantId {
        self.id
    }

    pub fn subject(&self) -> PrincipalId {
        self.subject
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn rights(&self) -> Rights {
        self.rights
    }

    pub fn constraints(&self) -> &Constraints {
        &self.constraints
    }
}

/// Every grant issued to one principal by one request.
#[derive(Debug)]
pub struct GrantSet {
    pub(crate) issuer: u128,
    pub(crate) principal: PrincipalId,
    pub(crate) grants: Vec<Grant>,
}

impl GrantSet {
    pub fn principal(&self) -> PrincipalId {
        self.principal
    }

    pub fn iter(&self) -> impl Iterator<Item = &Grant> {
        self.grants.iter()
    }

    pub fn get(&self, id: GrantId) -> Option<&Grant> {
        self.grants.iter().find(|g| g.id == id)
    }

    /// The grant issued for a specific object.
    pub fn for_object(&self, object: ObjectId) -> Option<&Grant> {
        self.grants.iter().find(|g| g.target == Target::Object(object))
    }

    /// Present one grant from this set for an operation.
    pub fn access(&self, id: GrantId) -> Option<Access<'_>> {
        Some(Access { grants: self, grant: self.get(id)?, cause: None })
    }
}

/// Why an operation happened: which node of which execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cause {
    pub execution: ExecutionId,
    pub node: NodeId,
}

/// A grant presented for one operation, with its provenance.
#[derive(Clone, Copy)]
pub struct Access<'a> {
    pub(crate) grants: &'a GrantSet,
    pub(crate) grant: &'a Grant,
    pub(crate) cause: Option<Cause>,
}

impl<'a> Access<'a> {
    pub fn caused_by(mut self, cause: Cause) -> Self {
        self.cause = Some(cause);
        self
    }

    pub fn grant(&self) -> &'a Grant {
        self.grant
    }

    pub fn cause(&self) -> Option<Cause> {
        self.cause
    }
}

/// Host policy: what principals may request. Not itself authority.
#[derive(Debug, Clone, Default)]
pub struct Policy {
    rules: Vec<(Path, Rights)>,
}

impl Policy {
    pub fn allow(mut self, prefix: Path, rights: Rights) -> Self {
        self.rules.push((prefix, rights));
        self
    }

    pub fn permits(&self, path: &Path, needed: Rights) -> bool {
        self.rules
            .iter()
            .filter(|(prefix, _)| path.starts_with(prefix))
            .fold(Rights::NONE, |acc, (_, r)| acc | *r)
            .contains(needed)
    }
}

/// The concrete authority a principal asks to be issued.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthorityRequest {
    pub principal: String,
    /// Exact names and the rights wanted on each, merged per path.
    pub wants: Vec<(Path, Rights)>,
}

impl AuthorityRequest {
    pub fn new(principal: impl Into<String>) -> Self {
        Self { principal: principal.into(), wants: Vec::new() }
    }

    pub fn want(mut self, path: Path, rights: Rights) -> Self {
        match self.wants.iter_mut().find(|(p, _)| *p == path) {
            Some((_, r)) => *r = *r | rights,
            None => self.wants.push((path, rights)),
        }
        self
    }
}

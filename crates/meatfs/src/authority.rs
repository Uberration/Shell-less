//! The authority subsystem.
//!
//! Two things are kept deliberately distinct:
//!
//! * [`Policy`] — path-prefix rules describing what *may be requested*. Plain
//!   data, freely constructed by the host.
//! * [`Grant`] — an authority instance *actually issued* by a namespace. It
//!   cannot be constructed, cloned or mutated outside this crate. The issuing
//!   namespace — its [`AuthorityDomainId`] — keeps a registry of live grants,
//!   so a grant is honoured only by the live domain that issued it, and only
//!   until retired.
//!
//! Knowing a pathname confers nothing: every operation presents an [`Access`]
//! naming one grant from the holder's [`GrantSet`]. Invoked capabilities see
//! only a [`NodeGrants`] projection — the grants attached to their node.

use crate::{AuthorityDomainId, ExecutionId, GrantId, NodeId, ObjectId, Path, PrincipalId, Rights};

/// What a grant points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// One existing object, by identity, wherever it is bound.
    Object(ObjectId),
    /// One currently unbound name: the right to bind a new data object there.
    Name(Path),
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

/// Every grant issued to one principal by one request, index-aligned with
/// the request's `wants`.
#[derive(Debug)]
pub struct GrantSet {
    pub(crate) domain: AuthorityDomainId,
    pub(crate) principal: PrincipalId,
    pub(crate) grants: Vec<Grant>,
}

impl GrantSet {
    pub fn domain(&self) -> AuthorityDomainId {
        self.domain
    }

    pub fn principal(&self) -> PrincipalId {
        self.principal
    }

    pub fn iter(&self) -> impl Iterator<Item = &Grant> {
        self.grants.iter()
    }

    /// The grant issued for the request's `index`-th want.
    pub fn issued(&self, index: usize) -> Option<&Grant> {
        self.grants.get(index)
    }

    pub fn get(&self, id: GrantId) -> Option<&Grant> {
        self.grants.iter().find(|g| g.id == id)
    }

    /// Present one grant from this set for an operation.
    pub fn access(&self, id: GrantId) -> Option<Access<'_>> {
        Some(Access { grants: self, grant: self.get(id)?, cause: None })
    }
}

/// A grant attached to a node under the name the node requested it by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeGrant {
    pub path: Path,
    pub grant: GrantId,
}

/// The authority attached to one node: a projection of a [`GrantSet`].
///
/// Construction only ever narrows — every entry must name a grant in the set.
#[derive(Clone, Copy)]
pub struct NodeGrants<'a> {
    pub(crate) set: &'a GrantSet,
    pub(crate) entries: &'a [NodeGrant],
}

impl<'a> NodeGrants<'a> {
    pub fn new(set: &'a GrantSet, entries: &'a [NodeGrant]) -> Option<Self> {
        entries.iter().all(|e| set.get(e.grant).is_some()).then_some(NodeGrants { set, entries })
    }

    pub fn empty(set: &'a GrantSet) -> Self {
        NodeGrants { set, entries: &[] }
    }

    pub fn entries(&self) -> &'a [NodeGrant] {
        self.entries
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

    pub fn grants(&self) -> &'a GrantSet {
        self.grants
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

/// The concrete authority a principal asks to be issued: one entry per use,
/// so each use receives its own grant with exactly the rights it needs.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthorityRequest {
    pub principal: String,
    pub wants: Vec<(Path, Rights)>,
}

impl AuthorityRequest {
    pub fn new(principal: impl Into<String>) -> Self {
        Self { principal: principal.into(), wants: Vec::new() }
    }

    pub fn want(mut self, path: Path, rights: Rights) -> Self {
        self.wants.push((path, rights));
        self
    }
}

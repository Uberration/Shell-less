use crate::Path;
use std::fmt;
use std::ops::BitOr;

/// The operations an authority may perform on an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rights(u8);

impl Rights {
    pub const NONE: Rights = Rights(0);
    pub const READ: Rights = Rights(1);
    pub const WRITE: Rights = Rights(1 << 1);
    pub const INVOKE: Rights = Rights(1 << 2);
    pub const SUBSCRIBE: Rights = Rights(1 << 3);
    pub const INSPECT: Rights = Rights(1 << 4);
    pub const ALL: Rights = Rights(0b1_1111);

    const NAMES: [(Rights, &'static str); 5] = [
        (Rights::READ, "read"),
        (Rights::WRITE, "write"),
        (Rights::INVOKE, "invoke"),
        (Rights::SUBSCRIBE, "subscribe"),
        (Rights::INSPECT, "inspect"),
    ];

    pub fn contains(self, other: Rights) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn from_name(name: &str) -> Option<Rights> {
        Self::NAMES.iter().find(|(_, n)| *n == name).map(|(r, _)| *r)
    }
}

impl BitOr for Rights {
    type Output = Rights;
    fn bitor(self, rhs: Rights) -> Rights {
        Rights(self.0 | rhs.0)
    }
}

impl fmt::Display for Rights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<_> = Self::NAMES.iter().filter(|(r, _)| self.contains(*r)).map(|(_, n)| *n).collect();
        if names.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&names.join("+"))
        }
    }
}

/// Rights over a subtree of the namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grant {
    pub prefix: Path,
    pub rights: Rights,
}

/// The explicit authority carried by every operation.
///
/// There is no ambient authority: an empty `Authority` can do nothing.
#[derive(Debug, Clone, Default)]
pub struct Authority {
    pub principal: String,
    grants: Vec<Grant>,
}

impl Authority {
    pub fn new(principal: impl Into<String>) -> Self {
        Self { principal: principal.into(), grants: Vec::new() }
    }

    /// Full authority over the whole namespace. For the host, never for agents.
    pub fn root(principal: impl Into<String>) -> Self {
        Self::new(principal).grant(Path::root(), Rights::ALL)
    }

    pub fn grant(mut self, prefix: Path, rights: Rights) -> Self {
        self.grants.push(Grant { prefix, rights });
        self
    }

    pub fn grants(&self) -> &[Grant] {
        &self.grants
    }

    pub fn allows(&self, path: &Path, needed: Rights) -> bool {
        let held =
            self.grants.iter().filter(|g| path.starts_with(&g.prefix)).fold(Rights::NONE, |acc, g| acc | g.rights);
        held.contains(needed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grants_cover_subtrees_only() {
        let auth = Authority::new("scout")
            .grant(Path::parse("/state/scout").unwrap(), Rights::WRITE)
            .grant(Path::parse("/memory").unwrap(), Rights::READ);
        assert!(auth.allows(&Path::parse("/state/scout/report").unwrap(), Rights::WRITE));
        assert!(!auth.allows(&Path::parse("/state/other").unwrap(), Rights::WRITE));
        assert!(!auth.allows(&Path::parse("/memory/x").unwrap(), Rights::WRITE));
        assert!(!Authority::new("nobody").allows(&Path::root(), Rights::READ));
    }
}

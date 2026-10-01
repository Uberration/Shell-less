use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hasher};

macro_rules! id {
    ($(#[$doc:meta])* $name:ident, $tag:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u128);

        impl $name {
            pub fn as_u128(self) -> u128 {
                self.0
            }

            /// The leading 32 bits, for human display only.
            pub fn short(self) -> String {
                format!(concat!($tag, ":{:08x}"), (self.0 >> 96) as u32)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($tag, ":{:032x}"), self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }
    };
}

id!(
    /// Immutable identity of an object. Paths are only names bound to it.
    ObjectId,
    "obj"
);
id!(
    /// Identity of an authority holder.
    PrincipalId,
    "prn"
);
id!(
    /// Identity of one issued grant.
    GrantId,
    "grt"
);
id!(
    /// Identity of one graph execution.
    ExecutionId,
    "exe"
);

/// A node within an execution graph. Assigned by the compiler, stable for a
/// given graph, and carried by events for provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Source of identifier entropy for a namespace.
///
/// `Seed::fixed` makes every identifier — and therefore every event and
/// receipt — reproducible. `Seed::entropy` is for namespaces that should not
/// collide with any other.
#[derive(Debug, Clone, Copy)]
pub struct Seed(u64);

impl Seed {
    pub fn fixed(seed: u64) -> Seed {
        Seed(seed)
    }

    pub fn entropy() -> Seed {
        Seed(RandomState::new().build_hasher().finish())
    }
}

/// SplitMix64: tiny, fast, and good enough for locally unique identifiers.
pub(crate) struct IdGen(u64);

impl IdGen {
    pub(crate) fn new(seed: Seed) -> Self {
        IdGen(seed.0)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_u128(&mut self) -> u128 {
        (u128::from(self.next_u64()) << 64) | u128::from(self.next_u64())
    }

    pub(crate) fn object(&mut self) -> ObjectId {
        ObjectId(self.next_u128())
    }

    pub(crate) fn principal(&mut self) -> PrincipalId {
        PrincipalId(self.next_u128())
    }

    pub(crate) fn grant(&mut self) -> GrantId {
        GrantId(self.next_u128())
    }

    pub(crate) fn execution(&mut self) -> ExecutionId {
        ExecutionId(self.next_u128())
    }
}

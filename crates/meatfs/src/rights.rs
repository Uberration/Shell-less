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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_and_names() {
        let rw = Rights::READ | Rights::WRITE;
        assert!(rw.contains(Rights::READ) && !rw.contains(Rights::INVOKE));
        assert_eq!(rw.to_string(), "read+write");
        assert_eq!(Rights::from_name("invoke"), Some(Rights::INVOKE));
    }
}

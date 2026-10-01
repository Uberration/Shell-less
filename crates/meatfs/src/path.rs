use crate::{Error, Result};
use std::fmt;
use std::str::FromStr;

/// An absolute, normalized MeatFS path such as `/tools/echo`.
///
/// Segments are restricted to `[A-Za-z0-9._-]` and may not be `.` or `..`,
/// so a path can never be used to escape its namespace.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Path {
    segments: Vec<String>,
}

impl Path {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn parse(s: &str) -> Result<Self> {
        let invalid = || Error::InvalidPath(s.to_owned());
        let rest = s.strip_prefix('/').ok_or_else(invalid)?;
        if rest.is_empty() {
            return Ok(Self::root());
        }
        let segments = rest
            .split('/')
            .map(|seg| {
                let ok = !seg.is_empty()
                    && seg != "."
                    && seg != ".."
                    && seg.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
                ok.then(|| seg.to_owned()).ok_or_else(invalid)
            })
            .collect::<Result<_>>()?;
        Ok(Self { segments })
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn parent(&self) -> Option<Path> {
        let (_, init) = self.segments.split_last()?;
        Some(Path { segments: init.to_vec() })
    }

    pub fn name(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    pub fn join(&self, segment: &str) -> Result<Path> {
        let child = Path::parse(&format!("/{segment}"))?;
        let mut segments = self.segments.clone();
        segments.extend(child.segments);
        Ok(Path { segments })
    }

    /// Whether `self` is `other` or lies beneath it.
    pub fn starts_with(&self, other: &Path) -> bool {
        self.segments.starts_with(&other.segments)
    }
}

impl FromStr for Path {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Path::parse(s)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.segments.is_empty() {
            return f.write_str("/");
        }
        for seg in &self.segments {
            write!(f, "/{seg}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_displays() {
        let p = Path::parse("/tools/echo").unwrap();
        assert_eq!(p.to_string(), "/tools/echo");
        assert_eq!(p.parent().unwrap().to_string(), "/tools");
        assert_eq!(Path::parse("/").unwrap(), Path::root());
    }

    #[test]
    fn rejects_escapes_and_junk() {
        for bad in ["", "tools", "/a/../b", "/a//b", "/a/", "/a b", "/a;rm"] {
            assert!(Path::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prefix() {
        let a = Path::parse("/state/scout/report").unwrap();
        assert!(a.starts_with(&Path::parse("/state").unwrap()));
        assert!(!a.starts_with(&Path::parse("/sta").unwrap()));
        assert!(a.starts_with(&Path::root()));
    }
}

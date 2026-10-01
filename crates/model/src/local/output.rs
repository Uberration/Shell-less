//! Assembling generated token bytes into bounded, valid UTF-8 text.
//!
//! Token pieces are raw bytes; a character may span several byte-fallback
//! tokens. Bytes are therefore assembled across token boundaries, and a
//! sequence is repaired only once it is known to be invalid:
//!
//! * a complete character is appended if it fits the byte budget;
//! * an invalid sequence becomes one U+FFFD (3 bytes), if that fits;
//! * an incomplete sequence is held until the next token completes it, or
//!   generation ends, when it becomes one U+FFFD, if that fits.
//!
//! The budget applies to the text returned, after repair. When a character
//! or replacement does not fit, the text stops there — never mid-character —
//! and generation ends with finish `length`.

/// Text under a hard byte budget, with at most one pending partial sequence.
pub(crate) struct Output {
    text: String,
    /// Bytes of an incomplete UTF-8 sequence awaiting the next token, plus
    /// at most one token piece while it is being consumed.
    pending: Vec<u8>,
    limit: usize,
}

const REPLACEMENT: char = '\u{FFFD}';

/// Append `ch` if it fits; `false` means the budget is spent.
fn put(text: &mut String, limit: usize, ch: char) -> bool {
    if text.len() + ch.len_utf8() > limit {
        return false;
    }
    text.push(ch);
    true
}

impl Output {
    /// Bytes this assembler may allocate: the full text budget (reserved up
    /// front, never grown) and scratch for one piece plus a partial sequence.
    pub fn bytes(limit: usize, max_piece: usize) -> Option<usize> {
        limit.checked_add(max_piece)?.checked_add(3)
    }

    pub fn new(limit: usize, max_piece: usize) -> Output {
        Output { text: String::with_capacity(limit), pending: Vec::with_capacity(max_piece + 3), limit }
    }

    /// Append one token's bytes. Returns `false` once the budget is spent;
    /// the text then holds the bounded prefix and must not grow further.
    pub fn push(&mut self, piece: &[u8]) -> bool {
        self.pending.extend_from_slice(piece);
        loop {
            let (valid, invalid) = match std::str::from_utf8(&self.pending) {
                Ok(s) => (s.len(), None),
                Err(e) => (e.valid_up_to(), Some(e.error_len())),
            };
            let complete = std::str::from_utf8(&self.pending[..valid]).expect("validated prefix");
            for ch in complete.chars() {
                if !put(&mut self.text, self.limit, ch) {
                    self.pending.clear();
                    return false;
                }
            }
            match invalid {
                // Everything consumed.
                None => {
                    self.pending.clear();
                    return true;
                }
                // An invalid sequence of `n` bytes: repaired now.
                Some(Some(n)) => {
                    if !put(&mut self.text, self.limit, REPLACEMENT) {
                        self.pending.clear();
                        return false;
                    }
                    self.pending.drain(..valid + n);
                }
                // An incomplete sequence at the end: wait for more bytes.
                Some(None) => {
                    self.pending.drain(..valid);
                    return true;
                }
            }
        }
    }

    /// End of generation. A held incomplete sequence becomes one U+FFFD if
    /// it fits. Returns the text and whether it ended within the budget.
    pub fn finish(mut self) -> (String, bool) {
        let fits = self.pending.is_empty() || put(&mut self.text, self.limit, REPLACEMENT);
        (self.text, fits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assemble(limit: usize, pieces: &[&[u8]]) -> (String, bool) {
        let mut out = Output::new(limit, 32);
        for piece in pieces {
            if !out.push(piece) {
                let (text, _) = out.finish();
                assert!(text.len() <= limit);
                return (text, false);
            }
        }
        let (text, fits) = out.finish();
        assert!(text.len() <= limit);
        (text, fits)
    }

    #[test]
    fn replacement_never_exceeds_a_small_budget() {
        // One invalid byte would become three bytes of U+FFFD.
        assert_eq!(assemble(1, &[&[0xFF]]), (String::new(), false));
        assert_eq!(assemble(2, &[&[0xFF]]), (String::new(), false));
        assert_eq!(assemble(3, &[&[0xFF]]), ("\u{FFFD}".to_owned(), true));
        // After a fitting prefix, the replacement is what does not fit.
        assert_eq!(assemble(4, &[b"ab", &[0xFF]]), ("ab".to_owned(), false));
    }

    #[test]
    fn characters_split_across_tokens_survive() {
        assert_eq!(assemble(16, &[&[0xC3], &[0xA9]]), ("é".to_owned(), true));
        assert_eq!(assemble(16, &[&[0xE6], &[0x97], &[0xA5]]), ("日".to_owned(), true));
        assert_eq!(assemble(16, &[b"a", &[0xF0, 0x9F], &[0x99, 0x82], b"b"]), ("a🙂b".to_owned(), true));
    }

    #[test]
    fn characters_are_never_split_by_the_budget() {
        assert_eq!(assemble(2, &[b"a", "é".as_bytes()]), ("a".to_owned(), false));
        assert_eq!(assemble(3, &[b"a", "é".as_bytes()]), ("aé".to_owned(), true));
        assert_eq!(assemble(4, &[&[0xE6], &[0x97], &[0xA5], b"xy"]), ("日x".to_owned(), false));
    }

    #[test]
    fn incomplete_endings_are_repaired_within_the_budget() {
        // A dangling lead byte at the end becomes one replacement, if it fits.
        assert_eq!(assemble(16, &[b"a", &[0xE6, 0x97]]), ("a\u{FFFD}".to_owned(), true));
        assert_eq!(assemble(3, &[b"a", &[0xE6, 0x97]]), ("a".to_owned(), false));
        // A lead byte followed by a non-continuation byte is invalid at once.
        assert_eq!(assemble(16, &[&[0xE6], b"a"]), ("\u{FFFD}a".to_owned(), true));
    }

    #[test]
    fn allocation_stays_within_the_declared_bytes() {
        let mut out = Output::new(8, 4);
        for _ in 0..20 {
            out.push(&[0xFF]);
        }
        assert!(out.text.capacity() + out.pending.capacity() <= Output::bytes(8, 4).unwrap());
        let (text, fits) = out.finish();
        assert_eq!((text.as_str(), fits), ("\u{FFFD}\u{FFFD}", true));
    }
}

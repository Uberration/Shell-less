//! The llama2.c `tokenizer.bin` export of the Llama 2 sentencepiece model,
//! with upstream `run.c` encode/decode semantics.
//!
//! File: `max_token_length: i32`, then for each of `vocab_size` tokens (the
//! count comes from the checkpoint, not the file): `score: f32`,
//! `len: i32`, `len` bytes. Little-endian. Ids 0–2 are the special pieces
//! `<unk>`, `\n<s>\n`, `\n</s>\n`; ids 3–258 are the byte pieces
//! `<0x00>`–`<0xFF>`; the space piece `" "` must exist (dummy prefix).
//!
//! Encoding follows `run.c` `encode`: optional BOS, a dummy `" "` prefix for
//! non-empty text, one vocabulary lookup per UTF-8 code point, byte
//! fallback (`byte + 3`) for code points not in the vocabulary, then
//! repeated merging of the adjacent pair whose concatenation has the
//! highest score (ties: leftmost).
//!
//! Profile behaviour that differs from upstream:
//!
//! * Ordinary vocabulary lookup and merging never turn literal
//!   special-token spellings (`"\n<s>\n"`) or byte-piece spellings
//!   (`"<0x41>"`) into those control or byte tokens; upstream does when
//!   the intermediate pieces exist. Actual UTF-8 byte fallback is
//!   unchanged and still produces byte-token ids.
//! * Decoding preserves every byte, including `<0x00>`; upstream's C
//!   strings drop a NUL.
//! * Prompts containing NUL are not accepted (enforced by the capability).

use super::checkpoint::read_exact;
use super::{Artifact, LocalError, LocalLimits};
use std::collections::HashMap;
use std::io::Read;

pub(crate) const UNK: u32 = 0;
pub(crate) const BOS: u32 = 1;
pub(crate) const EOS: u32 = 2;
/// First byte piece; byte `b` is token `b + 3`.
const BYTE_BASE: u32 = 3;
const FIRST_TEXT_PIECE: u32 = BYTE_BASE + 256;
const SPECIAL_PIECES: [&[u8]; 3] = [b"<unk>", b"\n<s>\n", b"\n</s>\n"];
/// Upstream merges only pairs scoring above this.
const MERGE_FLOOR: f32 = -1e10;
/// Sanity bound on the declared longest piece.
const MAX_TOKEN_LENGTH: usize = 1024;

pub(crate) struct Tokenizer {
    pieces: Vec<Vec<u8>>,
    scores: Vec<f32>,
    /// Text pieces only (ids ≥ 259): what vocabulary lookup and merging may
    /// produce. Byte pieces come only from byte fallback.
    lookup: HashMap<Vec<u8>, u32>,
    space: u32,
    /// The declared longest piece: an upper bound on any decoded piece.
    max_piece: usize,
}

fn invalid(reason: &'static str, token: Option<usize>) -> LocalError {
    LocalError::InvalidTokenizer { reason, token }
}

impl Tokenizer {
    /// Read and validate a tokenizer for a model of `vocab_size` tokens.
    pub fn read(
        reader: &mut dyn Read,
        len: u64,
        vocab_size: usize,
        limits: &LocalLimits,
    ) -> Result<(Tokenizer, u64), LocalError> {
        if len > limits.max_tokenizer_bytes {
            return Err(LocalError::TooLarge { artifact: Artifact::Tokenizer });
        }
        if vocab_size < FIRST_TEXT_PIECE as usize + 1 {
            return Err(invalid("vocabulary too small for special, byte and text pieces", None));
        }
        let mut bytes = vec![0u8; len as usize];
        read_exact(reader, &mut bytes, Artifact::Tokenizer)?;
        let mut probe = [0u8; 1];
        if reader.read(&mut probe).map_err(|e| LocalError::Io { artifact: Artifact::Tokenizer, kind: e.kind() })? != 0 {
            return Err(LocalError::TrailingBytes { artifact: Artifact::Tokenizer });
        }
        let mut hash = super::checkpoint::Fnv::new();
        hash.update(&bytes);

        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], LocalError> {
            let end = at.checked_add(n).filter(|&e| e <= bytes.len());
            let end = end.ok_or(LocalError::Truncated { artifact: Artifact::Tokenizer })?;
            let slice = &bytes[at..end];
            at = end;
            Ok(slice)
        };
        let word = |b: &[u8]| i32::from_le_bytes(b.try_into().expect("4 bytes"));
        let max_len = word(take(4)?);
        if max_len <= 0 || max_len as usize > MAX_TOKEN_LENGTH {
            return Err(invalid("max_token_length out of range", None));
        }
        let max_len = max_len as usize;

        // Each entry needs at least 8 bytes, so a vocabulary larger than the
        // file allows is rejected before anything is reserved for it.
        if vocab_size > bytes.len() / 8 {
            return Err(LocalError::Truncated { artifact: Artifact::Tokenizer });
        }
        let mut pieces = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let score = f32::from_le_bytes(take(4)?.try_into().expect("4 bytes"));
            if !score.is_finite() {
                return Err(invalid("non-finite score", Some(id)));
            }
            let n = word(take(4)?);
            if n < 0 || n as usize > max_len {
                return Err(invalid("piece length outside 0..=max_token_length", Some(id)));
            }
            pieces.push(take(n as usize)?.to_vec());
            scores.push(score);
        }
        if at != bytes.len() {
            return Err(LocalError::TrailingBytes { artifact: Artifact::Tokenizer });
        }

        for (id, expected) in SPECIAL_PIECES.iter().enumerate() {
            if pieces[id] != *expected {
                return Err(invalid("special pieces 0-2 are not <unk>, BOS, EOS", Some(id)));
            }
        }
        for byte in 0..=255u8 {
            let id = (BYTE_BASE + u32::from(byte)) as usize;
            if pieces[id] != format!("<0x{byte:02X}>").as_bytes() {
                return Err(invalid("byte pieces 3-258 are not <0x00>..<0xFF>", Some(id)));
            }
        }
        let mut lookup = HashMap::with_capacity(vocab_size);
        for (id, piece) in pieces.iter().enumerate().skip(FIRST_TEXT_PIECE as usize) {
            if std::str::from_utf8(piece).is_err() {
                return Err(invalid("text piece is not UTF-8", Some(id)));
            }
            if lookup.insert(piece.clone(), id as u32).is_some() {
                return Err(invalid("duplicate text piece", Some(id)));
            }
        }
        let space = *lookup.get(b" ".as_slice()).ok_or(invalid("no space piece for the dummy prefix", None))?;
        Ok((Tokenizer { pieces, scores, lookup, space, max_piece: max_len }, hash.finish()))
    }

    /// Encode `text` with a leading BOS. `text` must not contain NUL, which
    /// upstream's C strings cannot represent.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        debug_assert!(!text.contains('\0'));
        let mut tokens = vec![BOS];
        if !text.is_empty() {
            tokens.push(self.space);
        }
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let encoded = c.encode_utf8(&mut buf).as_bytes();
            match self.lookup.get(encoded) {
                Some(&id) => tokens.push(id),
                None => tokens.extend(encoded.iter().map(|&b| BYTE_BASE + u32::from(b))),
            }
        }
        let mut pair = Vec::new();
        loop {
            let mut best: Option<(f32, u32, usize)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                pair.clear();
                pair.extend_from_slice(&self.pieces[tokens[i] as usize]);
                pair.extend_from_slice(&self.pieces[tokens[i + 1] as usize]);
                if let Some(&id) = self.lookup.get(&pair) {
                    let score = self.scores[id as usize];
                    if score > best.map_or(MERGE_FLOOR, |(s, _, _)| s) {
                        best = Some((score, id, i));
                    }
                }
            }
            let Some((_, id, i)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        tokens
    }

    pub fn max_piece(&self) -> usize {
        self.max_piece
    }

    /// The bytes `token` contributes after `prev`, per upstream `decode`: a
    /// byte piece is its raw byte, and a leading space is dropped after BOS.
    pub fn decode(&self, prev: u32, token: u32) -> &[u8] {
        let piece = &self.pieces[token as usize];
        if (BYTE_BASE..FIRST_TEXT_PIECE).contains(&token) {
            let byte = (token - BYTE_BASE) as usize;
            return &BYTE_TABLE[byte..=byte];
        }
        if prev == BOS && piece.first() == Some(&b' ') {
            &piece[1..]
        } else {
            piece
        }
    }
}

/// Every byte value, for returning a byte piece as a one-byte slice.
static BYTE_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = i as u8;
        i += 1;
    }
    table
};

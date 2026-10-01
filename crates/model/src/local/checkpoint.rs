//! The llama2.c legacy "v0" float32 checkpoint, decoded little-endian.
//!
//! Layout (upstream `run.c` `read_checkpoint` / `memory_map_weights` and
//! `export.py` `legacy_export`, karpathy/llama2.c @ 350e04f):
//!
//! ```text
//! header   7 × i32: dim, hidden_dim, n_layers, n_heads, n_kv_heads,
//!                   vocab_size (negative ⇒ unshared classifier), seq_len
//! f32      token_embedding   [vocab, dim]
//!          rms_att           [layers, dim]
//!          wq                [layers, dim, n_heads·head]
//!          wk, wv            [layers, dim, n_kv_heads·head] each
//!          wo                [layers, n_heads·head, dim]
//!          rms_ffn           [layers, dim]
//!          w1                [layers, hidden, dim]
//!          w2                [layers, dim, hidden]
//!          w3                [layers, hidden, dim]
//!          rms_final         [dim]
//!          freq_cos, freq_sin [seq_len, head/2] each (RoPE tables)
//!          wcls              [vocab, dim]   only when unshared
//! ```
//!
//! The file must be exactly that long. Other llama2.c exports (which begin
//! with the `ak42` magic) and GGUF are rejected, not guessed at.

use super::{Artifact, LocalError, LocalLimits};
use std::io::Read;
use std::ops::Range;

pub(crate) const HEADER_BYTES: usize = 28;
/// `export.py` `version_export` writes this magic ("ak42") first.
const MAGIC_LLAMA2C_VERSIONED: u32 = 0x616b_3432;
const MAGIC_GGUF: u32 = 0x4655_4747;

/// RoPE base the profile requires; the file's tables are checked against it.
pub(crate) const ROPE_THETA: f64 = 10_000.0;
const ROPE_TABLE_TOLERANCE: f64 = 1e-3;

/// Model hyperparameters from a validated header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    /// Whether the classifier reuses the token embedding table.
    pub shared_classifier: bool,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }

    fn parse(header: &[u8; HEADER_BYTES]) -> Result<Config, LocalError> {
        let word = |i: usize| i32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        match word(0) as u32 {
            MAGIC_LLAMA2C_VERSIONED => {
                return Err(unsupported("llama2.c versioned export (v1/v2); only legacy v0 float32 is supported"))
            }
            MAGIC_GGUF => return Err(unsupported("GGUF")),
            _ => {}
        }
        let positive = |i: usize, field: &'static str| match word(i) {
            n if n > 0 => Ok(n as usize),
            _ => Err(LocalError::InvalidHeader { field, reason: "must be positive" }),
        };
        let raw_vocab = word(5);
        if raw_vocab == 0 || raw_vocab == i32::MIN {
            return Err(LocalError::InvalidHeader {
                field: "vocab_size",
                reason: "must be non-zero and representable",
            });
        }
        let config = Config {
            dim: positive(0, "dim")?,
            hidden_dim: positive(1, "hidden_dim")?,
            n_layers: positive(2, "n_layers")?,
            n_heads: positive(3, "n_heads")?,
            n_kv_heads: positive(4, "n_kv_heads")?,
            vocab_size: raw_vocab.unsigned_abs() as usize,
            seq_len: positive(6, "seq_len")?,
            shared_classifier: raw_vocab > 0,
        };
        let invalid = |field, reason| Err(LocalError::InvalidHeader { field, reason });
        if !config.dim.is_multiple_of(config.n_heads) {
            return invalid("dim", "not divisible by n_heads");
        }
        if !config.head_size().is_multiple_of(2) {
            return invalid("dim", "head size must be even for rotary embedding");
        }
        if config.n_kv_heads > config.n_heads || !config.n_heads.is_multiple_of(config.n_kv_heads) {
            return invalid("n_kv_heads", "must divide n_heads");
        }
        Ok(config)
    }
}

fn unsupported(reason: &'static str) -> LocalError {
    LocalError::Unsupported { artifact: Artifact::Checkpoint, reason }
}

/// Every tensor as a range into one contiguous buffer.
pub(crate) struct Weights {
    pub data: Vec<f32>,
    pub token_embedding: Range<usize>,
    pub rms_att: Range<usize>,
    pub wq: Range<usize>,
    pub wk: Range<usize>,
    pub wv: Range<usize>,
    pub wo: Range<usize>,
    pub rms_ffn: Range<usize>,
    pub w1: Range<usize>,
    pub w2: Range<usize>,
    pub w3: Range<usize>,
    pub rms_final: Range<usize>,
    pub freq_cos: Range<usize>,
    pub freq_sin: Range<usize>,
    /// The token embedding when the classifier is shared.
    pub wcls: Range<usize>,
}

/// Allocates consecutive ranges with checked arithmetic.
struct Layout(usize);

impl Layout {
    fn take(&mut self, parts: &[usize]) -> Result<Range<usize>, LocalError> {
        let len = parts.iter().try_fold(1usize, |acc, &p| acc.checked_mul(p)).ok_or_else(too_big)?;
        let start = self.0;
        self.0 = start.checked_add(len).ok_or_else(too_big)?;
        Ok(start..self.0)
    }
}

fn too_big() -> LocalError {
    LocalError::TooLarge { artifact: Artifact::Checkpoint }
}

fn layout(c: &Config) -> Result<(Weights, usize), LocalError> {
    let (l, d, h, kv) = (c.n_layers, c.dim, c.hidden_dim, c.kv_dim());
    let mut at = Layout(0);
    let token_embedding = at.take(&[c.vocab_size, d])?;
    let rms_att = at.take(&[l, d])?;
    let wq = at.take(&[l, d, d])?;
    let wk = at.take(&[l, d, kv])?;
    let wv = at.take(&[l, d, kv])?;
    let wo = at.take(&[l, d, d])?;
    let rms_ffn = at.take(&[l, d])?;
    let w1 = at.take(&[l, h, d])?;
    let w2 = at.take(&[l, d, h])?;
    let w3 = at.take(&[l, h, d])?;
    let rms_final = at.take(&[d])?;
    let freq_cos = at.take(&[c.seq_len, c.head_size() / 2])?;
    let freq_sin = at.take(&[c.seq_len, c.head_size() / 2])?;
    let wcls = if c.shared_classifier { token_embedding.clone() } else { at.take(&[c.vocab_size, d])? };
    let weights = Weights {
        data: Vec::new(),
        token_embedding,
        rms_att,
        wq,
        wk,
        wv,
        wo,
        rms_ffn,
        w1,
        w2,
        w3,
        rms_final,
        freq_cos,
        freq_sin,
        wcls,
    };
    Ok((weights, at.0))
}

/// FNV-1a, 64-bit: a non-cryptographic identity for provenance metadata.
pub(crate) struct Fnv(u64);

impl Fnv {
    pub fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) fn read_exact(reader: &mut dyn Read, buf: &mut [u8], artifact: Artifact) -> Result<(), LocalError> {
    reader.read_exact(buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => LocalError::Truncated { artifact },
        kind => LocalError::Io { artifact, kind },
    })
}

/// Read and validate a checkpoint of `len` bytes. Nothing is allocated from
/// header-provided sizes until the header, the complete layout and the host
/// limit have been checked against the actual length.
pub(crate) fn read(
    reader: &mut dyn Read,
    len: u64,
    limits: &LocalLimits,
) -> Result<(Config, Weights, u64), LocalError> {
    if len > limits.max_checkpoint_bytes {
        return Err(LocalError::TooLarge { artifact: Artifact::Checkpoint });
    }
    let mut header = [0u8; HEADER_BYTES];
    read_exact(reader, &mut header, Artifact::Checkpoint)?;
    let config = Config::parse(&header)?;
    let (mut weights, floats) = layout(&config)?;
    let expected = floats.checked_mul(4).and_then(|b| b.checked_add(HEADER_BYTES)).ok_or_else(too_big)? as u64;
    if len < expected {
        return Err(LocalError::Truncated { artifact: Artifact::Checkpoint });
    }
    if len > expected {
        return Err(LocalError::TrailingBytes { artifact: Artifact::Checkpoint });
    }

    let mut hash = Fnv::new();
    hash.update(&header);
    let mut data = Vec::with_capacity(floats);
    let mut chunk = vec![0u8; 1 << 16];
    while data.len() < floats {
        let n = ((floats - data.len()) * 4).min(chunk.len());
        read_exact(reader, &mut chunk[..n], Artifact::Checkpoint)?;
        hash.update(&chunk[..n]);
        data.extend(chunk[..n].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))));
    }
    let mut probe = [0u8; 1];
    if reader.read(&mut probe).map_err(|e| LocalError::Io { artifact: Artifact::Checkpoint, kind: e.kind() })? != 0 {
        return Err(LocalError::TrailingBytes { artifact: Artifact::Checkpoint });
    }
    weights.data = data;
    validate_values(&config, &weights)?;
    Ok((config, weights, hash.finish()))
}

fn validate_values(config: &Config, w: &Weights) -> Result<(), LocalError> {
    let tensors = [
        ("token_embedding", &w.token_embedding),
        ("rms_att", &w.rms_att),
        ("wq", &w.wq),
        ("wk", &w.wk),
        ("wv", &w.wv),
        ("wo", &w.wo),
        ("rms_ffn", &w.rms_ffn),
        ("w1", &w.w1),
        ("w2", &w.w2),
        ("w3", &w.w3),
        ("rms_final", &w.rms_final),
        ("freq_cos", &w.freq_cos),
        ("freq_sin", &w.freq_sin),
        ("wcls", &w.wcls),
    ];
    for (tensor, range) in tensors {
        if !w.data[range.clone()].iter().all(|v| v.is_finite()) {
            return Err(LocalError::NonFinite { tensor });
        }
    }
    // The legacy layout carries RoPE tables. The forward pass computes RoPE
    // as upstream run.c does; the tables must agree with that profile
    // (theta 10000, interleaved pairs), or the checkpoint is not this model.
    let half = config.head_size() / 2;
    for pos in 0..config.seq_len {
        for j in 0..half {
            let freq = 1.0 / ROPE_THETA.powf((2 * j) as f64 / config.head_size() as f64);
            let angle = pos as f64 * freq;
            let cos = f64::from(w.data[w.freq_cos.start + pos * half + j]);
            let sin = f64::from(w.data[w.freq_sin.start + pos * half + j]);
            if (cos - angle.cos()).abs() > ROPE_TABLE_TOLERANCE || (sin - angle.sin()).abs() > ROPE_TABLE_TOLERANCE {
                return Err(unsupported("rotary tables do not match theta-10000 interleaved RoPE"));
            }
        }
    }
    Ok(())
}

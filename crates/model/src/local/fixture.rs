//! Deterministic fixture artifacts in the supported profile, for tests.
//!
//! Weights come from integer arithmetic only (SplitMix64, 24-bit mantissa
//! values scaled by powers of two), so their bytes are identical on every
//! platform. RoPE tables are computed in f64; they are validated, not used.
//! These artifacts exercise mechanics and numerics. They are not a trained
//! model and prove nothing about compatibility with trained weights.

use super::checkpoint::ROPE_THETA;

#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub shared_classifier: bool,
    pub seed: u64,
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Exactly representable: a multiple of 2^-24 in [-0.5, 0.5), times `scale`
    /// (a power of two keeps it exact).
    fn uniform(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / 16_777_216.0 - 0.5) * scale
    }
}

/// A checkpoint in the legacy v0 float32 layout.
pub fn checkpoint(spec: Spec) -> Vec<u8> {
    let Spec { dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared_classifier, seed } = spec;
    let head = dim / n_heads;
    let kv_dim = dim * n_kv_heads / n_heads;
    let vocab_field = if shared_classifier { vocab_size as i32 } else { -(vocab_size as i32) };
    let mut out = Vec::new();
    for v in
        [dim as i32, hidden_dim as i32, n_layers as i32, n_heads as i32, n_kv_heads as i32, vocab_field, seq_len as i32]
    {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut rng = SplitMix(seed);
    let mut put = |n: usize, scale: f32, offset: f32, out: &mut Vec<u8>| {
        for _ in 0..n {
            out.extend_from_slice(&(offset + rng.uniform(scale)).to_le_bytes());
        }
    };
    put(vocab_size * dim, 2.0, 0.0, &mut out); // token embedding
    put(n_layers * dim, 0.5, 1.0, &mut out); // rms_att
    put(n_layers * dim * dim, 1.0, 0.0, &mut out); // wq
    put(n_layers * dim * kv_dim, 1.0, 0.0, &mut out); // wk
    put(n_layers * dim * kv_dim, 1.0, 0.0, &mut out); // wv
    put(n_layers * dim * dim, 0.5, 0.0, &mut out); // wo
    put(n_layers * dim, 0.5, 1.0, &mut out); // rms_ffn
    put(n_layers * hidden_dim * dim, 1.0, 0.0, &mut out); // w1
    put(n_layers * dim * hidden_dim, 0.5, 0.0, &mut out); // w2
    put(n_layers * hidden_dim * dim, 1.0, 0.0, &mut out); // w3
    put(dim, 0.5, 1.0, &mut out); // rms_final
    for f in [f64::cos, f64::sin] {
        for pos in 0..seq_len {
            for j in 0..head / 2 {
                let freq = 1.0 / ROPE_THETA.powf((2 * j) as f64 / head as f64);
                out.extend_from_slice(&(f(pos as f64 * freq) as f32).to_le_bytes());
            }
        }
    }
    if !shared_classifier {
        put(vocab_size * dim, 2.0, 0.0, &mut out);
    }
    out
}

/// Text pieces of the fixture tokenizer, with scores, after the 3 special
/// and 256 byte pieces. Ids start at 259 in this order.
pub const TEXT_PIECES: &[(&str, f32)] = &[
    (" ", -1.0),      // 259
    ("h", -2.0),      // 260
    ("e", -2.0),      // 261
    ("l", -2.0),      // 262
    ("o", -2.0),      // 263
    ("w", -2.0),      // 264
    ("r", -2.0),      // 265
    ("d", -2.0),      // 266
    ("he", -3.0),     // 267
    ("ll", -3.5),     // 268
    ("hell", -4.0),   // 269
    ("hello", -4.5),  // 270
    (" hello", -5.0), // 271
    (" w", -3.0),     // 272
    ("or", -3.0),     // 273
    (" wor", -4.0),   // 274
    (" world", -5.5), // 275
    ("ld", -3.2),     // 276
    ("é", -2.0),      // 277
    ("\n", -2.0),     // 278
    ("  ", -6.0),     // 279
    ("<", -2.0),      // 280
    ("s", -2.0),      // 281
    (">", -2.0),      // 282
    ("<s", -3.0),     // 283
    ("<s>", -3.0),    // 284
    ("\n<s>", -3.0),  // 285
    ("x", -2.0),      // 286
    ("0", -2.0),      // 287
    ("4", -2.0),      // 288
    ("1", -2.0),      // 289
    ("<0", -3.0),     // 290
    ("<0x", -3.0),    // 291
    ("<0x4", -3.0),   // 292
    ("<0x41", -3.0),  // 293
    ("a", -2.0),      // 294
    (" a", -2.5),     // 295
];

pub const TOKENIZER_VOCAB: usize = 259 + TEXT_PIECES.len();

/// A model over the fixture tokenizer that predicts `winner` after every
/// position: embeddings are constant, attention and FFN outputs are zero,
/// and only the classifier row for `winner` is non-zero. For exercising
/// stop tokens, limits and decoding deterministically.
pub fn constant_prediction(winner: usize) -> Vec<u8> {
    let (dim, hidden, layers, heads, vocab, seq) = (8usize, 8usize, 1usize, 2usize, TOKENIZER_VOCAB, 32usize);
    let head = dim / heads;
    let mut out = Vec::new();
    for v in [dim as i32, hidden as i32, layers as i32, heads as i32, heads as i32, -(vocab as i32), seq as i32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let put = |n: usize, value: f32, out: &mut Vec<u8>| {
        for _ in 0..n {
            out.extend_from_slice(&value.to_le_bytes());
        }
    };
    put(vocab * dim, 0.5, &mut out); // token embedding: every row the same
    put(layers * dim, 1.0, &mut out); // rms_att
    put(layers * dim * dim * 4, 0.0, &mut out); // wq, wk, wv, wo
    put(layers * dim, 1.0, &mut out); // rms_ffn
    put(layers * hidden * dim * 3, 0.0, &mut out); // w1, w2, w3
    put(dim, 1.0, &mut out); // rms_final
    for f in [f64::cos, f64::sin] {
        for pos in 0..seq {
            for j in 0..head / 2 {
                let freq = 1.0 / ROPE_THETA.powf((2 * j) as f64 / head as f64);
                out.extend_from_slice(&(f(pos as f64 * freq) as f32).to_le_bytes());
            }
        }
    }
    for row in 0..vocab {
        put(dim, if row == winner { 1.0 } else { 0.0 }, &mut out);
    }
    out
}

/// A tokenizer in the `tokenizer.bin` export layout.
pub fn tokenizer() -> Vec<u8> {
    let mut pieces: Vec<(Vec<u8>, f32)> =
        vec![(b"<unk>".to_vec(), 0.0), (b"\n<s>\n".to_vec(), 0.0), (b"\n</s>\n".to_vec(), 0.0)];
    pieces.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>").into_bytes(), 0.0)));
    pieces.extend(TEXT_PIECES.iter().map(|(p, s)| (p.as_bytes().to_vec(), *s)));
    let max = pieces.iter().map(|(p, _)| p.len()).max().unwrap_or(0) as i32;
    let mut out = max.to_le_bytes().to_vec();
    for (piece, score) in pieces {
        out.extend_from_slice(&score.to_le_bytes());
        out.extend_from_slice(&(piece.len() as i32).to_le_bytes());
        out.extend_from_slice(&piece);
    }
    out
}

/// A small model over the fixture tokenizer, for capability-level tests.
pub fn tiny_spec() -> Spec {
    Spec {
        dim: 16,
        hidden_dim: 24,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: TOKENIZER_VOCAB,
        seq_len: 32,
        shared_classifier: true,
        seed: 7,
    }
}

/// The fixture model, loaded with the given limits.
pub fn model(limits: super::LocalLimits) -> super::LocalModel {
    super::LocalModel::from_bytes(&checkpoint(tiny_spec()), &tokenizer(), limits).expect("fixture artifacts are valid")
}

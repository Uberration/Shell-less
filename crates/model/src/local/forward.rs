//! The Llama 2 forward pass, scalar f32, mirroring upstream `run.c`
//! `forward` operation for operation and in the same accumulation order.
//!
//! Per position: token embedding → per layer [RMSNorm → Q,K,V projections →
//! interleaved RoPE (theta 10000) → causal multi-head attention over the KV
//! cache (grouped K/V heads) → output projection + residual → RMSNorm →
//! SwiGLU FFN + residual] → final RMSNorm → classifier.

use super::checkpoint::{Config, Weights};

/// Per-invocation execution state. Never shared between invocations.
pub(crate) struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    /// Positions this state can hold.
    context: usize,
}

impl State {
    /// Bytes the state needs for `context` positions, or `None` on overflow.
    pub fn bytes(config: &Config, context: usize) -> Option<usize> {
        let kv = config.n_layers.checked_mul(context)?.checked_mul(config.kv_dim())?.checked_mul(2)?;
        let floats = [
            config.dim.checked_mul(4)?, // x, xb, xb2, q
            config.hidden_dim.checked_mul(2)?,
            config.n_heads.checked_mul(context)?,
            config.vocab_size,
            kv,
        ]
        .iter()
        .try_fold(0usize, |acc, &n| acc.checked_add(n))?;
        floats.checked_mul(4)
    }

    /// Fresh, zeroed state for `context` positions.
    pub fn new(config: &Config, context: usize) -> State {
        let kv = config.n_layers * context * config.kv_dim();
        State {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * context],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; kv],
            value_cache: vec![0.0; kv],
            context,
        }
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    for ((o, w), v) in out.iter_mut().zip(weight).zip(x) {
        *o = w * (ss * v);
    }
}

fn rmsnorm_in_place(x: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v = w * (ss * *v);
    }
}

fn softmax(x: &mut [f32]) {
    let mut max = x[0];
    for &v in &x[1..] {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `out[d] = W[d, n] · x[n]`, row-major W, accumulated left to right.
fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * n..(i + 1) * n];
        let mut val = 0.0f32;
        for j in 0..n {
            val += row[j] * x[j];
        }
        *o = val;
    }
}

fn rotate(vec: &mut [f32], i: usize, fcr: f32, fci: f32) {
    let (v0, v1) = (vec[i], vec[i + 1]);
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
}

/// Run one position. `token < vocab_size` and `pos < state.context`; the
/// caller guarantees both. Returns the logits for the next token.
pub(crate) fn forward<'s>(c: &Config, w: &Weights, s: &'s mut State, token: usize, pos: usize) -> &'s [f32] {
    let (dim, kv_dim, hidden, head_size) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
    let kv_mul = c.n_heads / c.n_kv_heads;
    let ctx = s.context;
    let at = |range: &std::ops::Range<usize>, offset: usize, len: usize| &w.data[range.start + offset..][..len];

    s.x.copy_from_slice(at(&w.token_embedding, token * dim, dim));

    for l in 0..c.n_layers {
        rmsnorm(&mut s.xb, &s.x, at(&w.rms_att, l * dim, dim));

        let loff = l * ctx * kv_dim;
        let here = loff + pos * kv_dim;
        matmul(&mut s.q, &s.xb, at(&w.wq, l * dim * dim, dim * dim));
        matmul(&mut s.key_cache[here..here + kv_dim], &s.xb, at(&w.wk, l * dim * kv_dim, dim * kv_dim));
        matmul(&mut s.value_cache[here..here + kv_dim], &s.xb, at(&w.wv, l * dim * kv_dim, dim * kv_dim));

        for i in (0..dim).step_by(2) {
            let head_dim = i % head_size;
            let freq = 1.0f32 / 10000.0f32.powf(head_dim as f32 / head_size as f32);
            let val = pos as f32 * freq;
            let (fcr, fci) = (val.cos(), val.sin());
            rotate(&mut s.q, i, fcr, fci);
            if i < kv_dim {
                rotate(&mut s.key_cache[here..here + kv_dim], i, fcr, fci);
            }
        }

        for h in 0..c.n_heads {
            let q = &s.q[h * head_size..][..head_size];
            let att = &mut s.att[h * ctx..][..pos + 1];
            let kv_head = (h / kv_mul) * head_size;
            for (t, score_out) in att.iter_mut().enumerate() {
                let k = &s.key_cache[loff + t * kv_dim + kv_head..][..head_size];
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                *score_out = score;
            }
            softmax(att);
            let xb = &mut s.xb[h * head_size..][..head_size];
            xb.fill(0.0);
            for (t, &a) in att.iter().enumerate() {
                let v = &s.value_cache[loff + t * kv_dim + kv_head..][..head_size];
                for i in 0..head_size {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(&mut s.xb2, &s.xb, at(&w.wo, l * dim * dim, dim * dim));
        for (x, d) in s.x.iter_mut().zip(&s.xb2) {
            *x += d;
        }

        rmsnorm(&mut s.xb, &s.x, at(&w.rms_ffn, l * dim, dim));
        matmul(&mut s.hb, &s.xb, at(&w.w1, l * dim * hidden, dim * hidden));
        matmul(&mut s.hb2, &s.xb, at(&w.w3, l * dim * hidden, dim * hidden));
        for (h1, h3) in s.hb.iter_mut().zip(&s.hb2) {
            let mut val = *h1;
            val *= 1.0f32 / (1.0f32 + (-val).exp());
            val *= h3;
            *h1 = val;
        }
        matmul(&mut s.xb, &s.hb, at(&w.w2, l * dim * hidden, dim * hidden));
        for (x, d) in s.x.iter_mut().zip(&s.xb) {
            *x += d;
        }
    }

    rmsnorm_in_place(&mut s.x, at(&w.rms_final, 0, dim));
    matmul(&mut s.logits, &s.x, at(&w.wcls, 0, c.vocab_size * dim));
    &s.logits
}

//! A local CPU reference backend: real inference as an ordinary capability.
//!
//! # The Shell-less legacy-v0 completion profile
//!
//! One profile, versioned as [`PROFILE`]. Its file formats come from
//! karpathy/llama2.c at commit 350e04fe35433e6d2941dce5a1f53308f87058eb
//! (`run.c` sha256 9c4f2d5c…658bd). Its generation semantics are Shell-less's
//! own, and are listed below where they differ from upstream:
//!
//! * checkpoint: the legacy v0 float32 layout (see [`checkpoint`]), decoded
//!   little-endian; Llama 2 architecture with RMSNorm, interleaved RoPE
//!   (theta 10000), grouped K/V heads and a SwiGLU FFN. The stored RoPE
//!   tables must match that RoPE. Upstream skips them; checking them is a
//!   strict profile restriction.
//! * tokenizer: the matching `tokenizer.bin` export (see [`tokenizer`]).
//!
//! Acceptance target: `stories15M.bin` from the karpathy/tinyllamas model
//! series, with the upstream 32,000-piece `tokenizer.bin` (sha256
//! 50a52ef8…ce361). Matching vocabulary sizes do not prove a tokenizer
//! belongs to a checkpoint; the host is responsible for supplying the pair.
//!
//! # Boundary
//!
//! The host supplies both files, which are read and validated once, then
//! held immutable and mounted. Requests cannot name files, trigger reads or
//! select weights. Each invocation gets fresh execution state, KV cache
//! included.
//!
//! # Requests
//!
//! Completion, not chat: exactly one `user` message, used verbatim as the
//! prefix (NUL is not accepted); `max_tokens` is required. The prompt tokens
//! (BOS and dummy prefix included) plus `max_tokens` must fit
//! min(`seq_len`, host context limit). That reserves one position more than
//! strictly needed, and is kept as a tested boundary.
//!
//! # Generation
//!
//! Greedy: all logits must be finite (otherwise the invocation fails), then
//! the highest wins, lowest token id on ties. After each selection:
//!
//! | selected | outcome |
//! |---|---|
//! | BOS (1) | finish `stop`, no text; upstream `generate`'s delimiter |
//! | EOS (2) | finish `stop`, no text; a Shell-less extension (upstream `generate` does not stop on EOS) |
//! | UNK (0) | the invocation fails: an unexpected unknown token is not a sequence end |
//! | other | its bytes are appended to the output |
//!
//! Generation also ends with finish `length` after `max_tokens` selections,
//! or when the next character cannot fit the output byte limit.
//!
//! # Output
//!
//! Token bytes are assembled across token boundaries into UTF-8 (see
//! [`output`]). Invalid sequences, and an incomplete sequence left at the
//! end, become U+FFFD. The output byte limit applies to the returned text
//! after that repair, and the text is never cut mid-character.
//!
//! `usage.input_tokens` counts prompt tokens. `usage.output_tokens` counts
//! generated token selections, including a stop token that produced no text
//! and a token whose text no longer fitted: both were generation work.
//!
//! # Determinism
//!
//! Declared `Effectful` and `Nondeterministic`. Greedy decoding uses no
//! randomness, but f32 `exp`, `powf`, `sin` and `cos` are not specified
//! to be bit-identical across platforms or toolchains, so the substrate is
//! given no general determinism guarantee.
//!
//! # Revision
//!
//! The declared revision is `PROFILE;checkpoint=…;tokenizer=…`: the profile
//! version plus FNV-1a-64 fingerprints of both artifacts. The fingerprints are
//! non-cryptographic and serve reporting only — not authorization,
//! integrity or attestation.

pub(crate) mod checkpoint;
mod forward;
mod output;
mod tokenizer;

#[cfg(any(test, feature = "fixtures"))]
pub mod fixture;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

pub use checkpoint::Config;

use crate::{FinishReason, InferRequest, InferResponse, Message, Role, Usage};
use capability::{Capability, CapabilityContext, CapabilityMeta, Determinism, Fault, Purity};
use checkpoint::Weights;
use forward::State;
use output::Output;
use std::fmt;
use std::io::Read;
use std::sync::Arc;
use tokenizer::{Tokenizer, BOS, EOS, UNK};

pub const IMPLEMENTATION: &str = "shell-less/llama2c-v0-f32-scalar";
/// The completion profile's version: bumped whenever tokenization, stopping,
/// decoding or usage semantics change, independently of the artifacts.
pub const PROFILE: &str = "shell-less-legacy-v0-completion/2";

/// Host-owned bounds: on what is loaded, and on every invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalLimits {
    pub max_checkpoint_bytes: u64,
    pub max_tokenizer_bytes: u64,
    pub max_prompt_bytes: usize,
    /// Prompt tokens plus requested new tokens, also capped by `seq_len`.
    pub max_context_tokens: usize,
    pub max_new_tokens: usize,
    pub max_output_bytes: usize,
    /// Per-invocation execution state, KV cache included.
    pub max_working_bytes: usize,
}

impl Default for LocalLimits {
    fn default() -> Self {
        LocalLimits {
            max_checkpoint_bytes: 512 << 20,
            max_tokenizer_bytes: 8 << 20,
            max_prompt_bytes: 2048,
            max_context_tokens: 256,
            max_new_tokens: 256,
            max_output_bytes: 16 << 10,
            max_working_bytes: 64 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artifact {
    Checkpoint,
    Tokenizer,
}

/// Why artifacts could not be loaded. Never contains a host path or file
/// content, so it is safe on any diagnostic surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalError {
    Io { artifact: Artifact, kind: std::io::ErrorKind },
    TooLarge { artifact: Artifact },
    Truncated { artifact: Artifact },
    TrailingBytes { artifact: Artifact },
    Unsupported { artifact: Artifact, reason: &'static str },
    InvalidHeader { field: &'static str, reason: &'static str },
    NonFinite { tensor: &'static str },
    InvalidTokenizer { reason: &'static str, token: Option<usize> },
}

impl LocalError {
    /// Which artifact the failure concerns.
    pub fn artifact(&self) -> Artifact {
        match self {
            LocalError::Io { artifact, .. }
            | LocalError::TooLarge { artifact }
            | LocalError::Truncated { artifact }
            | LocalError::TrailingBytes { artifact }
            | LocalError::Unsupported { artifact, .. } => *artifact,
            LocalError::InvalidHeader { .. } | LocalError::NonFinite { .. } => Artifact::Checkpoint,
            LocalError::InvalidTokenizer { .. } => Artifact::Tokenizer,
        }
    }
}

impl fmt::Display for LocalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalError::Io { artifact, kind } => write!(f, "{artifact:?}: io error ({kind:?})"),
            LocalError::TooLarge { artifact } => write!(f, "{artifact:?}: exceeds the host limit"),
            LocalError::Truncated { artifact } => write!(f, "{artifact:?}: truncated"),
            LocalError::TrailingBytes { artifact } => {
                write!(f, "{artifact:?}: trailing bytes after the expected layout")
            }
            LocalError::Unsupported { artifact, reason } => write!(f, "{artifact:?}: unsupported: {reason}"),
            LocalError::InvalidHeader { field, reason } => write!(f, "Checkpoint: invalid header: {field} {reason}"),
            LocalError::NonFinite { tensor } => write!(f, "Checkpoint: non-finite value in {tensor}"),
            LocalError::InvalidTokenizer { reason, token: Some(id) } => write!(f, "Tokenizer: {reason} (token {id})"),
            LocalError::InvalidTokenizer { reason, token: None } => write!(f, "Tokenizer: {reason}"),
        }
    }
}

impl std::error::Error for LocalError {}

struct Model {
    config: Config,
    weights: Weights,
    tokenizer: Tokenizer,
}

/// A validated checkpoint and tokenizer, ready to mount as a capability.
#[derive(Clone)]
pub struct LocalModel {
    model: Arc<Model>,
    limits: LocalLimits,
    meta: CapabilityMeta,
}

impl LocalModel {
    /// Read and validate both host files. Nothing is mounted on failure.
    pub fn load(
        checkpoint: &std::path::Path,
        tokenizer: &std::path::Path,
        limits: LocalLimits,
    ) -> Result<LocalModel, LocalError> {
        let open = |path: &std::path::Path, artifact| {
            let file = std::fs::File::open(path).map_err(|e| LocalError::Io { artifact, kind: e.kind() })?;
            let len = file.metadata().map_err(|e| LocalError::Io { artifact, kind: e.kind() })?.len();
            Ok::<_, LocalError>((std::io::BufReader::new(file), len))
        };
        let (mut ckpt, ckpt_len) = open(checkpoint, Artifact::Checkpoint)?;
        let (mut tok, tok_len) = open(tokenizer, Artifact::Tokenizer)?;
        Self::from_readers(&mut ckpt, ckpt_len, &mut tok, tok_len, limits)
    }

    /// Validate in-memory artifacts.
    pub fn from_bytes(checkpoint: &[u8], tokenizer: &[u8], limits: LocalLimits) -> Result<LocalModel, LocalError> {
        let (mut c, mut t) = (checkpoint, tokenizer);
        Self::from_readers(&mut c, checkpoint.len() as u64, &mut t, tokenizer.len() as u64, limits)
    }

    fn from_readers(
        checkpoint: &mut dyn Read,
        checkpoint_len: u64,
        tokenizer: &mut dyn Read,
        tokenizer_len: u64,
        limits: LocalLimits,
    ) -> Result<LocalModel, LocalError> {
        let (config, weights, checkpoint_hash) = checkpoint::read(checkpoint, checkpoint_len, &limits)?;
        let (tokenizer, tokenizer_hash) = Tokenizer::read(tokenizer, tokenizer_len, config.vocab_size, &limits)?;
        let revision =
            format!("{PROFILE};checkpoint=fnv1a64:{checkpoint_hash:016x};tokenizer=fnv1a64:{tokenizer_hash:016x}");
        let meta = CapabilityMeta::new(Purity::Effectful, Determinism::Nondeterministic)
            .implemented_by(IMPLEMENTATION, revision);
        Ok(LocalModel { model: Arc::new(Model { config, weights, tokenizer }), limits, meta })
    }

    pub fn config(&self) -> Config {
        self.model.config
    }

    /// How a prompt is tokenized, BOS included. For acceptance reports.
    pub fn prompt_tokens(&self, prompt: &str) -> Vec<u32> {
        self.model.tokenizer.encode(prompt)
    }

    /// Logits for each position of `tokens`, from fresh state. For checking
    /// numerical behaviour against an external reference.
    pub fn logits(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let m = &self.model;
        let mut state = State::new(&m.config, tokens.len().max(1));
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| forward::forward(&m.config, &m.weights, &mut state, t as usize, pos).to_vec())
            .collect()
    }
}

/// Greedy selection over logits that are all finite: a non-finite logit
/// (NaN or ±inf, from finite weights through overflowing arithmetic) fails
/// the invocation before anything is selected.
fn select(logits: &[f32]) -> Result<u32, Fault> {
    argmax(logits).ok_or_else(|| Fault::Failed("non-finite logits".to_owned()))
}

/// Greedy choice: the highest logit, lowest id on ties. `None` if any logit
/// is not finite.
fn argmax(logits: &[f32]) -> Option<u32> {
    if !logits.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    Some(best as u32)
}

fn invalid(reason: &str) -> Fault {
    Fault::InvalidInput(reason.to_owned())
}

impl Capability for LocalModel {
    type Input = InferRequest;
    type Output = InferResponse;

    fn describe(&self) -> &'static str {
        "greedy text completion with a local llama2.c v0 float32 checkpoint"
    }

    fn meta(&self) -> CapabilityMeta {
        self.meta.clone()
    }

    fn invoke(&self, _: &CapabilityContext<'_>, request: InferRequest) -> Result<InferResponse, Fault> {
        let (m, limits) = (&self.model, &self.limits);
        let prompt = match request.messages.as_slice() {
            [Message { role: Role::User, content }] => content,
            _ => return Err(invalid("completion takes exactly one user message")),
        };
        let max_new = match request.parameters.max_tokens {
            Some(n) if n >= 1 && n as u64 <= limits.max_new_tokens as u64 => n as usize,
            Some(_) => return Err(invalid("max_tokens outside the host limit")),
            None => return Err(invalid("max_tokens is required")),
        };
        if prompt.len() > limits.max_prompt_bytes {
            return Err(invalid("prompt exceeds the host byte limit"));
        }
        if prompt.contains('\0') {
            return Err(invalid("prompt contains NUL"));
        }

        // Prompt encoding is bounded by the prompt byte limit checked above.
        let prompt_tokens = m.tokenizer.encode(prompt);
        let capacity = m.config.seq_len.min(limits.max_context_tokens);
        let context = prompt_tokens.len().checked_add(max_new).filter(|&n| n <= capacity);
        let context = context.ok_or_else(|| invalid("prompt plus max_tokens exceeds the context limit"))?;
        // Everything the generation allocates: execution state (KV cache
        // included), the output text and its decoding scratch, token ids.
        let working = State::bytes(&m.config, context)
            .zip(Output::bytes(limits.max_output_bytes, m.tokenizer.max_piece()))
            .and_then(|(state, output)| state.checked_add(output))
            .and_then(|n| n.checked_add(prompt_tokens.len().checked_mul(4)?));
        match working {
            Some(bytes) if bytes <= limits.max_working_bytes => {}
            _ => return Err(invalid("request exceeds the working-memory limit")),
        }

        // Fresh state: nothing from an earlier invocation is reachable.
        let mut state = State::new(&m.config, context);
        let mut output = Output::new(limits.max_output_bytes, m.tokenizer.max_piece());
        let mut logits = &[][..];
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            logits = forward::forward(&m.config, &m.weights, &mut state, token as usize, pos);
        }
        let mut prev = *prompt_tokens.last().expect("BOS is always present");
        let mut next = select(logits)?;
        let mut selections = 1usize;
        let mut finish = loop {
            match next {
                BOS | EOS => break FinishReason::Stop,
                UNK => return Err(Fault::Failed("generation selected UNK".to_owned())),
                _ => {}
            }
            if !output.push(m.tokenizer.decode(prev, next)) {
                break FinishReason::Length;
            }
            if selections == max_new {
                break FinishReason::Length;
            }
            let pos = prompt_tokens.len() + selections - 1;
            logits = forward::forward(&m.config, &m.weights, &mut state, next as usize, pos);
            prev = next;
            next = select(logits)?;
            selections += 1;
        };
        let (text, within_budget) = output.finish();
        if !within_budget {
            finish = FinishReason::Length;
        }

        Ok(InferResponse {
            message: Message::new(Role::Assistant, text),
            usage: Some(Usage { input_tokens: prompt_tokens.len() as i64, output_tokens: selections as i64 }),
            finish,
        })
    }
}

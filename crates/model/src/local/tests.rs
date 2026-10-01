//! Tests for the local backend. Expected values come from the external
//! reference described in `crates/model/reference/README.md`; running these
//! tests needs nothing but Cargo.

use super::fixture::{self, Spec};
use super::*;

/// Numeric fixture A: shared classifier, grouped K/V heads (4 query, 2 KV).
pub(super) const SPEC_A: Spec = Spec {
    dim: 16,
    hidden_dim: 24,
    n_layers: 2,
    n_heads: 4,
    n_kv_heads: 2,
    vocab_size: 32,
    seq_len: 8,
    shared_classifier: true,
    seed: 1,
};
pub(super) const TOKENS_A: [u32; 5] = [1, 7, 7, 30, 2];

/// Numeric fixture B: unshared classifier (negative vocab field), plain MHA.
pub(super) const SPEC_B: Spec = Spec {
    dim: 12,
    hidden_dim: 20,
    n_layers: 1,
    n_heads: 2,
    n_kv_heads: 2,
    vocab_size: 24,
    seq_len: 6,
    shared_classifier: false,
    seed: 2,
};
pub(super) const TOKENS_B: [u32; 4] = [0, 5, 23, 5];

/// Writes the reference inputs to `$SHELL_LESS_REFERENCE_DIR`. Used only to
/// regenerate provenance; see the reference README.
#[test]
#[ignore]
fn write_reference_inputs() {
    let dir =
        std::path::PathBuf::from(std::env::var("SHELL_LESS_REFERENCE_DIR").expect("set SHELL_LESS_REFERENCE_DIR"));
    std::fs::write(dir.join("fixture_a.bin"), fixture::checkpoint(SPEC_A)).unwrap();
    std::fs::write(dir.join("fixture_b.bin"), fixture::checkpoint(SPEC_B)).unwrap();
    std::fs::write(dir.join("fixture_tokenizer.bin"), fixture::tokenizer()).unwrap();
}

use super::checkpoint;
use super::forward::{self, State};
use super::reference::{FIXTURE_ENCODINGS, LOGITS_A, LOGITS_B, REAL_ENCODINGS};
use super::tokenizer::Tokenizer;
use crate::{InferRequest, InferResponse};
use capability::IntoValue;
use meatfs::{MeatFs, Path, Seed, Value};

/// Our logits for `tokens` at each position, from one fresh state, so every
/// position after the first attends over cached keys and values.
fn our_logits(spec: Spec, tokens: &[u32]) -> Vec<Vec<f32>> {
    let bytes = fixture::checkpoint(spec);
    let (config, weights, _) =
        checkpoint::read(&mut bytes.as_slice(), bytes.len() as u64, &LocalLimits::default()).unwrap();
    let mut state = State::new(&config, tokens.len());
    tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| forward::forward(&config, &weights, &mut state, t as usize, pos).to_vec())
        .collect()
}

/// Tolerance against the reference: |ours − ref| ≤ 1e-5 · max(1, |ref|).
/// The scalar code follows run.c's operation order, so on the reference
/// platform the observed difference is 0; the tolerance allows for libm
/// differences in exp/powf/sin/cos elsewhere.
fn assert_matches_reference(ours: &[Vec<f32>], reference: &[&[u32]]) {
    assert_eq!(ours.len(), reference.len());
    let mut worst = 0.0f32;
    for (pos, (row, expected)) in ours.iter().zip(reference).enumerate() {
        assert_eq!(row.len(), expected.len());
        for (i, (&a, &bits)) in row.iter().zip(expected.iter()).enumerate() {
            let b = f32::from_bits(bits);
            let diff = (a - b).abs();
            assert!(diff <= 1e-5 * b.abs().max(1.0), "position {pos}, logit {i}: {a} vs reference {b}");
            worst = worst.max(diff);
        }
    }
    eprintln!("largest absolute difference from reference: {worst:e}");
}

#[test]
fn logits_match_upstream_with_shared_classifier_and_grouped_kv() {
    assert_matches_reference(&our_logits(SPEC_A, &TOKENS_A), LOGITS_A);
}

#[test]
fn logits_match_upstream_with_unshared_classifier() {
    assert_matches_reference(&our_logits(SPEC_B, &TOKENS_B), LOGITS_B);
}

fn fixture_tokenizer() -> Tokenizer {
    let bytes = fixture::tokenizer();
    Tokenizer::read(&mut bytes.as_slice(), bytes.len() as u64, fixture::TOKENIZER_VOCAB, &LocalLimits::default())
        .unwrap()
        .0
}

/// Where this tokenizer deliberately differs from upstream: text never
/// becomes a special piece or a byte piece.
const DEVIATIONS: &[(&str, &[u32])] = &[
    ("\n<s>\n", &[1, 259, 285, 278]), // upstream merges into BOS: [1, 259, 1]
    ("<0x41>", &[1, 259, 293, 282]),  // upstream merges into byte piece 0x41: [1, 259, 68]
];

#[test]
fn encoding_matches_upstream_except_documented_deviations() {
    let tokenizer = fixture_tokenizer();
    for (text, upstream) in FIXTURE_ENCODINGS {
        let ours = tokenizer.encode(text);
        match DEVIATIONS.iter().find(|(t, _)| t == text) {
            Some((_, expected)) => {
                assert_eq!(ours, *expected, "{text:?}");
                assert_ne!(ours, *upstream, "the deviation is real for {text:?}");
            }
            None => assert_eq!(ours, *upstream, "{text:?}"),
        }
    }
    // Spot checks of the cases the table covers.
    assert_eq!(tokenizer.encode(""), [1], "no dummy prefix for empty text");
    assert_eq!(tokenizer.encode("日"), [1, 259, 0xE6 + 3, 0x97 + 3, 0xA5 + 3], "byte fallback");
    assert_eq!(tokenizer.encode("hello  world"), [1, 271, 259, 275], "whitespace runs");
}

#[test]
fn decoding_matches_upstream() {
    let t = fixture_tokenizer();
    assert_eq!(t.decode(1, 271), b"hello", "leading space dropped after BOS");
    assert_eq!(t.decode(271, 275), b" world");
    assert_eq!(t.decode(1, 259), b"");
    assert_eq!(t.decode(263, 233), [0xE6], "byte piece is its raw byte");
    assert_eq!(t.decode(5, 258), [0xFF]);
    assert_eq!(t.decode(1, 277), "é".as_bytes());
    // Upstream returns a C string, so <0x00> decodes to "" there; the byte
    // itself is returned here.
    assert_eq!(t.decode(5, 3), [0x00]);
    let bytes: Vec<u8> = [233u32, 154, 168].iter().flat_map(|&id| t.decode(259, id).to_vec()).collect();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), "日");
}

/// The real 32,000-piece `tokenizer.bin` against upstream encodings.
/// Needs `SHELL_LESS_TOKENIZER`; run with `--ignored`.
#[test]
#[ignore]
fn real_tokenizer_matches_upstream() {
    let path = std::env::var("SHELL_LESS_TOKENIZER").expect("set SHELL_LESS_TOKENIZER");
    let bytes = std::fs::read(path).unwrap();
    let (tokenizer, _) =
        Tokenizer::read(&mut bytes.as_slice(), bytes.len() as u64, 32000, &LocalLimits::default()).unwrap();
    for (text, upstream) in REAL_ENCODINGS {
        assert_eq!(tokenizer.encode(text), *upstream, "{text:?}");
    }
}

// ── Validation ──────────────────────────────────────────────────────────────

fn load(checkpoint: &[u8], tokenizer: &[u8]) -> Result<LocalModel, LocalError> {
    LocalModel::from_bytes(checkpoint, tokenizer, LocalLimits::default())
}

fn header_with(field: usize, value: i32) -> Vec<u8> {
    let mut bytes = fixture::checkpoint(fixture::tiny_spec());
    bytes[field * 4..field * 4 + 4].copy_from_slice(&value.to_le_bytes());
    bytes
}

#[test]
fn checkpoints_are_validated_before_use() {
    let good = fixture::checkpoint(fixture::tiny_spec());
    let tok = fixture::tokenizer();
    assert!(load(&good, &tok).is_ok());

    let truncated = &good[..good.len() - 1];
    assert_eq!(load(truncated, &tok).err(), Some(LocalError::Truncated { artifact: Artifact::Checkpoint }));
    let mut trailing = good.clone();
    trailing.push(0);
    assert_eq!(load(&trailing, &tok).err(), Some(LocalError::TrailingBytes { artifact: Artifact::Checkpoint }));
    assert_eq!(load(&good[..10], &tok).err(), Some(LocalError::Truncated { artifact: Artifact::Checkpoint }));

    let header_errors = [
        (header_with(0, 15), "dim"),       // not divisible by n_heads
        (header_with(0, 12), "dim"),       // head size 3 is odd
        (header_with(4, 3), "n_kv_heads"), // does not divide n_heads
        (header_with(4, 8), "n_kv_heads"), // more KV heads than heads
        (header_with(2, 0), "n_layers"),   // zero
        (header_with(5, i32::MIN), "vocab_size"),
    ];
    for (bytes, field) in header_errors {
        assert!(matches!(load(&bytes, &tok), Err(LocalError::InvalidHeader { field: f, .. }) if f == field), "{field}");
    }

    // Other formats are rejected by their magic, not misread as dimensions.
    let mut versioned = good.clone();
    versioned[..4].copy_from_slice(&0x616b_3432u32.to_le_bytes());
    assert!(matches!(load(&versioned, &tok), Err(LocalError::Unsupported { .. })));
    let mut gguf = good.clone();
    gguf[..4].copy_from_slice(b"GGUF");
    assert!(matches!(load(&gguf, &tok), Err(LocalError::Unsupported { .. })));

    // Values: non-finite weights, and RoPE tables from another profile.
    let mut nan = good.clone();
    let at = checkpoint::HEADER_BYTES + 4 * 5;
    nan[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(load(&nan, &tok).err(), Some(LocalError::NonFinite { tensor: "token_embedding" }));
    let mut rope = good.clone();
    let spec = fixture::tiny_spec();
    let tables = spec.seq_len * (spec.dim / spec.n_heads); // cos and sin
    let sin_at = good.len() - 4 * (tables / 2) + 4 * 3; // inside freq_sin, shared classifier
    rope[sin_at..sin_at + 4].copy_from_slice(&0.9f32.to_le_bytes());
    assert!(matches!(load(&rope, &tok), Err(LocalError::Unsupported { artifact: Artifact::Checkpoint, .. })));
}

#[test]
fn sizes_are_checked_before_allocation() {
    let tok = fixture::tokenizer();
    // A header claiming an enormous model in a tiny file: refused from the
    // header and length alone.
    let mut huge = header_with(0, 1 << 30);
    huge[4..8].copy_from_slice(&(1i32 << 30).to_le_bytes());
    assert!(matches!(load(&huge, &tok), Err(LocalError::Truncated { .. } | LocalError::TooLarge { .. })));
    // A host byte limit below the real size.
    let good = fixture::checkpoint(fixture::tiny_spec());
    let small = LocalLimits { max_checkpoint_bytes: 1024, ..LocalLimits::default() };
    let e = LocalModel::from_bytes(&good, &tok, small).err();
    assert_eq!(e, Some(LocalError::TooLarge { artifact: Artifact::Checkpoint }));
    let small = LocalLimits { max_tokenizer_bytes: 100, ..LocalLimits::default() };
    let e = LocalModel::from_bytes(&good, &tok, small).err();
    assert_eq!(e, Some(LocalError::TooLarge { artifact: Artifact::Tokenizer }));
}

#[test]
fn tokenizers_are_validated_against_the_export_profile() {
    let ckpt = fixture::checkpoint(fixture::tiny_spec());
    let good = fixture::tokenizer();
    let entry = |id: usize| {
        // Offset of entry `id`: 4-byte header, then (score, len, bytes) each.
        let mut at = 4;
        for _ in 0..id {
            let len = i32::from_le_bytes(good[at + 4..at + 8].try_into().unwrap()) as usize;
            at += 8 + len;
        }
        at
    };
    let reason = |bytes: &[u8]| match load(&ckpt, bytes) {
        Err(LocalError::InvalidTokenizer { reason, .. }) => reason,
        other => panic!("expected an invalid tokenizer, got {:?}", other.err()),
    };

    assert_eq!(
        load(&ckpt, &good[..good.len() - 1]).err(),
        Some(LocalError::Truncated { artifact: Artifact::Tokenizer })
    );
    let mut trailing = good.clone();
    trailing.push(b'x');
    assert_eq!(load(&ckpt, &trailing).err(), Some(LocalError::TrailingBytes { artifact: Artifact::Tokenizer }));

    let mut bad_special = good.clone();
    bad_special[entry(1) + 8 + 2] = b'x'; // "\n<s>\n" → "\n<x>\n"
    assert_eq!(reason(&bad_special), "special pieces 0-2 are not <unk>, BOS, EOS");
    let mut bad_byte = good.clone();
    bad_byte[entry(3 + 0x41) + 8 + 4] = b'2'; // "<0x41>" → "<0x42>"
    assert_eq!(reason(&bad_byte), "byte pieces 3-258 are not <0x00>..<0xFF>");
    let mut long_piece = good.clone();
    long_piece[..4].copy_from_slice(&3i32.to_le_bytes()); // max length 3, but "<unk>" is 5
    assert_eq!(reason(&long_piece), "piece length outside 0..=max_token_length");
    let mut nan_score = good.clone();
    let at = entry(260);
    nan_score[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(reason(&nan_score), "non-finite score");

    // A checkpoint whose vocabulary is larger than the tokenizer's entries.
    let bigger = fixture::checkpoint(Spec { vocab_size: fixture::TOKENIZER_VOCAB + 4, ..fixture::tiny_spec() });
    assert_eq!(load(&bigger, &good).err(), Some(LocalError::Truncated { artifact: Artifact::Tokenizer }));
}

#[test]
fn load_errors_never_name_host_paths() {
    let secret = std::env::temp_dir().join("zq-hostpath-secret").join("missing.bin");
    let e = LocalModel::load(&secret, &secret, LocalLimits::default()).err().unwrap();
    assert_eq!(e, LocalError::Io { artifact: Artifact::Checkpoint, kind: std::io::ErrorKind::NotFound });
    assert!(!format!("{e} {e:?}").contains("zq-hostpath-secret"));
}

// ── Invocation ──────────────────────────────────────────────────────────────

fn mounted(model: LocalModel) -> (MeatFs, meatfs::GrantSet, meatfs::ObjectId) {
    let (fs, host) = MeatFs::genesis(Seed::fixed(0));
    let root = host.access(host.iter().next().unwrap().id()).unwrap();
    let id = capability::mount(&fs, &root, &Path::parse("/models/local/stories/infer").unwrap(), model).unwrap();
    (fs, host, id)
}

fn request(messages: &[(&str, &str)], max_tokens: Option<i64>) -> Value {
    let messages = messages
        .iter()
        .map(|(role, content)| Value::map([("role", Value::from(*role)), ("content", Value::from(*content))]))
        .collect();
    let mut fields = vec![("messages", Value::List(messages))];
    if let Some(n) = max_tokens {
        fields.push(("parameters", Value::map([("max_tokens", Value::Int(n))])));
    }
    Value::map(fields)
}

fn infer(fs: &MeatFs, host: &meatfs::GrantSet, id: meatfs::ObjectId, request: Value) -> meatfs::Result<Value> {
    fs.invoke(&host.access(host.iter().next().unwrap().id()).unwrap(), id, request)
}

fn response(text: &str, input_tokens: i64, output_tokens: i64, finish: FinishReason) -> Value {
    InferResponse {
        message: Message::new(Role::Assistant, text),
        usage: Some(Usage { input_tokens, output_tokens }),
        finish,
    }
    .into_value()
}

fn with_winner(checkpoint: Vec<u8>, limits: LocalLimits) -> (MeatFs, meatfs::GrantSet, meatfs::ObjectId) {
    mounted(LocalModel::from_bytes(&checkpoint, &fixture::tokenizer(), limits).unwrap())
}

#[test]
fn stopping_and_limits_are_explicit() {
    let with = |winner: usize, limits: LocalLimits| with_winner(fixture::constant_prediction(winner), limits);
    let limits = LocalLimits::default();

    // BOS (the profile's delimiter) and EOS (a documented extension) stop
    // without text. The selection still counts as generation work.
    for stop in [1usize, 2] {
        let (fs, host, id) = with(stop, limits);
        let out = infer(&fs, &host, id, request(&[("user", "a")], Some(8))).unwrap();
        assert_eq!(out, response("", 2, 1, FinishReason::Stop), "stop token {stop}");
    }

    // A text token repeats until max_tokens: finish length. Prompt "a" is
    // [BOS, " a"]; each "a" follows " a" or "a", so no space is stripped.
    let (fs, host, id) = with(294, limits);
    let out = infer(&fs, &host, id, request(&[("user", "a")], Some(4))).unwrap();
    assert_eq!(out, response("aaaa", 2, 4, FinishReason::Length));
    // The output byte limit also ends generation with finish length; the
    // fourth selection was made, so it is counted, though its text did not fit.
    let (fs, host, id) = with(294, LocalLimits { max_output_bytes: 3, ..limits });
    let out = infer(&fs, &host, id, request(&[("user", "a")], Some(10))).unwrap();
    assert_eq!(out, response("aaa", 2, 4, FinishReason::Length));
    // Empty prompt: BOS only, and the first piece follows BOS.
    let (fs, host, id) = with(295, limits);
    let out = infer(&fs, &host, id, request(&[("user", "")], Some(2))).unwrap();
    assert_eq!(out, response("a a", 1, 2, FinishReason::Length));
    // Bytes that never form UTF-8 are repaired: invalid, then incomplete.
    let (fs, host, id) = with(0xE6 + 3, limits);
    let out = infer(&fs, &host, id, request(&[("user", "a")], Some(2))).unwrap();
    assert_eq!(out, response("\u{FFFD}\u{FFFD}", 2, 2, FinishReason::Length));
}

#[test]
fn unk_and_non_finite_logits_fail_distinctly() {
    let limits = LocalLimits::default();
    // A selected UNK is a generation failure, not a successful stop.
    let (fs, host, id) = with_winner(fixture::constant_prediction(0), limits);
    let e = infer(&fs, &host, id, request(&[("user", "a")], Some(8))).unwrap_err();
    assert_eq!(e, meatfs::Error::Capability { object: id, message: "generation selected UNK".into() });
    // Finite weights, overflowing arithmetic: the logit is +inf. Caught
    // before selection, and reported as a numerical failure.
    let (fs, host, id) = with_winner(fixture::classifier_row(294, f32::MAX), limits);
    let e = infer(&fs, &host, id, request(&[("user", "a")], Some(8))).unwrap_err();
    assert_eq!(e, meatfs::Error::Capability { object: id, message: "non-finite logits".into() });
    // The same overflow onto UNK's row is still numerical, not "UNK".
    let (fs, host, id) = with_winner(fixture::classifier_row(0, f32::MAX), limits);
    let e = infer(&fs, &host, id, request(&[("user", "a")], Some(8))).unwrap_err();
    assert_eq!(e, meatfs::Error::Capability { object: id, message: "non-finite logits".into() });
    assert_eq!(argmax(&[f32::NAN, 1.0]), None, "a NaN at index 0 is never the winner");
    assert_eq!(argmax(&[1.0, 3.0, 3.0]), Some(1), "ties: lowest id");
}

#[test]
fn the_byte_limit_holds_after_utf8_repair() {
    // Every selection is byte 0xFF, which is never UTF-8: each becomes a
    // three-byte U+FFFD, so small budgets must not be overrun.
    let invalid = || fixture::constant_prediction(0xFF + 3);
    for (budget, text, selections) in [(1, "", 1), (2, "", 1), (3, "\u{FFFD}", 2), (7, "\u{FFFD}\u{FFFD}", 3)] {
        let limits = LocalLimits { max_output_bytes: budget, ..LocalLimits::default() };
        let (fs, host, id) = with_winner(invalid(), limits);
        let out = infer(&fs, &host, id, request(&[("user", "a")], Some(8))).unwrap();
        assert_eq!(out, response(text, 2, selections, FinishReason::Length), "budget {budget}");
        assert!(text.len() <= budget);
    }
}

#[test]
fn working_memory_covers_the_output_buffer() {
    let base = LocalLimits { max_working_bytes: 256 << 10, ..LocalLimits::default() };
    let (fs, host, id) = mounted(fixture::model(base));
    assert!(infer(&fs, &host, id, request(&[("user", "hello")], Some(4))).is_ok());
    // The same request, but an output budget that the working-memory limit
    // cannot hold: rejected before anything is allocated.
    let (fs, host, id) = mounted(fixture::model(LocalLimits { max_output_bytes: 1 << 20, ..base }));
    let e = infer(&fs, &host, id, request(&[("user", "hello")], Some(4))).unwrap_err();
    assert!(matches!(e, meatfs::Error::InvalidInput { .. }));
}

#[test]
fn requests_outside_the_supported_subset_are_rejected() {
    let limits =
        LocalLimits { max_prompt_bytes: 64, max_context_tokens: 16, max_new_tokens: 8, ..LocalLimits::default() };
    let (fs, host, id) = mounted(fixture::model(limits));
    let rejected = |request: Value| {
        assert!(matches!(infer(&fs, &host, id, request.clone()), Err(meatfs::Error::InvalidInput { .. })), "{request}");
    };
    rejected(request(&[("system", "be brief"), ("user", "hello")], Some(4)));
    rejected(request(&[("system", "hello")], Some(4)));
    rejected(request(&[("user", "hello"), ("assistant", "hi"), ("user", "hello")], Some(4)));
    rejected(request(&[("user", "hello")], None));
    rejected(request(&[("user", "hello")], Some(0)));
    rejected(request(&[("user", "hello")], Some(9)));
    rejected(request(&[("user", &"a".repeat(65))], Some(1)));
    rejected(request(&[("user", "nul\0byte")], Some(1)));
    // 1 BOS + 1 prefix + 8 " a" tokens + 8 new > 16: rejected, not truncated.
    rejected(request(&[("user", "a a a a a a a a")], Some(8)));
    let tiny = LocalLimits { max_working_bytes: 64, ..limits };
    let (fs, host, id) = mounted(fixture::model(tiny));
    assert!(matches!(
        infer(&fs, &host, id, request(&[("user", "hello")], Some(1))),
        Err(meatfs::Error::InvalidInput { .. })
    ));
    // Unsupported sampling settings cannot even be expressed.
    let mut sampled = request(&[("user", "hello")], Some(1));
    if let Value::Map(fields) = &mut sampled {
        fields.insert("parameters".into(), Value::map([("max_tokens", Value::Int(1)), ("temperature", Value::Int(1))]));
    }
    let (fs, host, id) = mounted(fixture::model(limits));
    assert!(matches!(infer(&fs, &host, id, sampled), Err(meatfs::Error::InvalidInput { .. })));
}

#[test]
fn invocations_share_no_execution_state() {
    let (fs, host, id) = mounted(fixture::model(LocalLimits::default()));
    let a = || request(&[("user", "hello world")], Some(12));
    let b = request(&[("user", "a a é 日 q")], Some(12));
    let first = infer(&fs, &host, id, a()).unwrap();
    let other = infer(&fs, &host, id, b).unwrap();
    let again = infer(&fs, &host, id, a()).unwrap();
    assert_eq!(first, again, "B left nothing behind for A");
    assert_ne!(first, other);
    // And a separately loaded copy agrees: the result depends on the
    // request and the weights only.
    let (fs2, host2, id2) = mounted(fixture::model(LocalLimits::default()));
    assert_eq!(infer(&fs2, &host2, id2, a()).unwrap(), first);
    let usage = first.get("usage").unwrap();
    assert_eq!(usage.get("input_tokens"), Some(&Value::Int(3)));
}

#[test]
fn metadata_is_declared_conservatively_and_names_no_path() {
    let model = fixture::model(LocalLimits::default());
    let meta = model.meta();
    assert_eq!((meta.purity, meta.determinism), (Purity::Effectful, Determinism::Nondeterministic));
    assert_eq!(meta.invocation.implementation.as_deref(), Some(IMPLEMENTATION));
    let revision = meta.invocation.revision.unwrap();
    let prefix = format!("{PROFILE};checkpoint=fnv1a64:");
    assert!(revision.starts_with(&prefix) && revision.contains(";tokenizer=fnv1a64:"), "{revision}");
    assert_eq!(model.config().vocab_size, fixture::TOKENIZER_VOCAB);
    let _: InferRequest = capability::FromValue::from_value(request(&[("user", "x")], Some(1))).unwrap();
}

/// The trained-model acceptance run. Needs host-provisioned artifacts:
/// `SHELL_LESS_CHECKPOINT` (stories15M.bin) and `SHELL_LESS_TOKENIZER`
/// (the matching tokenizer.bin). Run with `--ignored --nocapture`. Missing
/// artifacts fail the test; they never make it pass vacuously.
#[test]
#[ignore]
fn trained_stories15m_generates_a_continuation() {
    let checkpoint = std::env::var("SHELL_LESS_CHECKPOINT").expect("set SHELL_LESS_CHECKPOINT");
    let tokenizer = std::env::var("SHELL_LESS_TOKENIZER").expect("set SHELL_LESS_TOKENIZER");
    let model = LocalModel::load(checkpoint.as_ref(), tokenizer.as_ref(), LocalLimits::default()).unwrap();
    let c = model.config();
    assert_eq!((c.dim, c.n_layers, c.n_heads, c.vocab_size, c.seq_len), (288, 6, 6, 32000, 256), "stories15M");
    let prompt = "Once upon a time";
    let prompt_ids = model.prompt_tokens(prompt);
    assert_eq!(prompt_ids, [1, 9038, 2501, 263, 931], "upstream encoding of the prompt");
    let revision = model.meta().invocation.revision.unwrap();
    let (fs, host, id) = mounted(model);
    let out = infer(&fs, &host, id, request(&[("user", prompt)], Some(48))).unwrap();
    let text = out.get("message").unwrap().get("content").unwrap().as_text().unwrap().to_owned();
    eprintln!("revision: {revision}");
    eprintln!("prompt token ids: {prompt_ids:?}");
    eprintln!("continuation: {text:?}");
    eprintln!("usage: {}", out.get("usage").unwrap());
    eprintln!("finish: {}", out.get("finish").unwrap());
    assert!(!text.trim().is_empty());
    assert_eq!(out.get("usage").unwrap().get("input_tokens"), Some(&Value::Int(5)));
}

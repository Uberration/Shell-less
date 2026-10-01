# Reference values for the local backend

The expected logits and token ids in `crates/model/src/local/reference.rs`
come from upstream llama2.c, run through the two harnesses in this directory.
Running the Cargo tests needs none of this: no C compiler, no Python, no
downloads, no model files.

## Upstream reference

* Repository: karpathy/llama2.c, commit
  `350e04fe35433e6d2941dce5a1f53308f87058eb` (2024-05-29).
* `run.c` sha256 `9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd`.
* `tokenizer.bin` sha256 `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361`
  (the 32,000-piece Llama 2 tokenizer export).
* The harnesses `#include "run.c"` with `TESTING` defined and call its own
  `build_transformer`/`forward` and `build_tokenizer`/`encode`/`decode`.
* Built with GCC 13.3 on x86-64: `gcc -O2 -I<checkout> -o logits logits.c -lm`
  (likewise `tokenize.c`). No `-ffast-math`, no OpenMP, no `-march`.

## Procedure

```sh
# 1. Write the fixture inputs (deterministic; see local/fixture.rs).
SHELL_LESS_REFERENCE_DIR=$DIR cargo test -p model write_reference_inputs -- --ignored

# 2. Logits at every position. Positions after the first attend over cached
#    keys and values.
./logits $DIR/fixture_a.bin 1 7 7 30 2   # shared classifier, 4 heads / 2 KV heads
./logits $DIR/fixture_b.bin 0 5 23 5     # unshared classifier (negative vocab field)

# 3. Encodings (BOS, no EOS); strings are passed hex-encoded.
./tokenize $DIR/fixture_tokenizer.bin 296 <hex>...
./tokenize tokenizer.bin 32000 <hex>...
```

`reference.rs` was generated from that output and records the inputs.

## Tolerance

Logits are compared with |ours − reference| ≤ 1e-5 · max(1, |reference|).
The Rust forward pass follows `run.c`'s operation and accumulation order, so
on the reference platform the observed difference is exactly 0 at every
position of both fixtures. The tolerance allows for `expf`/`powf`/`sinf`/
`cosf` differing in the last bits on other platforms.

## Deliberate tokenizer differences

1. Text never becomes a special piece (ids 0–2) or a byte piece (3–258).
   Upstream merges literal `"\n<s>\n"` into BOS and `"<0x41>"` into the
   byte piece for `A` when the intermediate pieces exist. Both cases appear
   in the fixture table. They do not occur with the real 32,000-piece vocabulary:
   all 14 real encodings match upstream exactly.
2. A byte piece decodes to its byte, including `<0x00>`. Upstream returns
   C strings, so `<0x00>` decodes to an empty string there.
3. Prompts containing NUL are rejected; upstream's C strings end at NUL.

## Trained-model acceptance

The real run uses `stories15M.bin` from the karpathy/tinyllamas model series
with the `tokenizer.bin` above. Neither is checked in. With both provisioned:

```sh
SHELL_LESS_CHECKPOINT=stories15M.bin SHELL_LESS_TOKENIZER=tokenizer.bin \
  cargo test -p model trained_stories15m -- --ignored --nocapture
SHELL_LESS_TOKENIZER=tokenizer.bin cargo test -p model real_tokenizer -- --ignored

cargo run -p shell-less -- run --checkpoint stories15M.bin --tokenizer tokenizer.bin \
  --show-outputs story.meat.yaml
```

To compare a continuation with upstream, run `run.c` with temperature 0 (greedy):
`./run stories15M.bin -t 0 -n <steps> -i "Once upon a time"`. Upstream
prints the prompt as well; Shell-less returns only the new text.

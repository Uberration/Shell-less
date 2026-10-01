# Shell-less

Shell-less is a Rust-only agent substrate in the MEAT ecosystem. It does not use a shell as an architectural primitive.

```text
MEAT filesystem / execution substrate
 ├── agents
 ├── models
 ├── memory
 ├── tools          (typed capabilities, never `exec(string)`)
 ├── state
 ├── events
 └── orchestration
```

The model is just another component attached to the substrate.

## Laws

- Rust only at the control/software layer.
- **AI is filesystems.** MeatFS is the universal semantic interface. OS mounts, 9P and remote transports are adapters.
- **MEATYAML is software.** A program compiles to a validated typed graph and then executes. It is not config for an interpreter.
- Assembly is the execution substrate. Hot paths collapse into generated native code (Speck / MEATASM).
- No shell, Python, Node, SaaS or vendor framework as foundation.
- FUSE / 9P / remote transports are interface options over MeatFS, never the object model.
- Speck / MEATASM sits beneath the runtime as a compilation target, not handwritten assembly everywhere.
- Hydra stays outside Shell-less; later it becomes the distributed placement layer for its jobs.

## Pipeline

```text
MEATYAML ─compile─▶ MEAT IR ─load─▶ AuthorityRequest ─policy─▶ GrantSet ─▶ per-node grants ─execute─▶ ExecutionReceipt
                    nodes, data/order edges,        (observational:                       (one transaction,
                    outputs                          nothing changes)                      commit or roll back)
```

The IR is the contract. The Rust interpreter in `runtime` is its first backend. Speck/MEATASM and Hydra come later, as further consumers of the same IR.

| crate | role |
|---|---|
| `meatfs` | Namespace. `ObjectId` is identity; paths are names. `Policy` (what may be requested) vs `Grant` (what was issued; unforgeable, valid only in its live `AuthorityDomainId`, retirable). Grant targets: an object, an unbound name (for creation), or, for the host only, a namespace. `Transaction` stages reads, writes and creations atomically. The journal records attempts and outcomes and is never rolled back. |
| `capability` | Typed `Capability` trait with mandatory `CapabilityMeta { purity }` and structured `Fault`s. Pure capabilities receive no effects handle; effectful ones see only the grants attached to their node. |
| `meatyaml` | Compiles to MEAT IR: `Graph { id, nodes, edges, outputs }`. `Data` edges carry values; `Order` edges are derived wherever node footprints (target + `uses`) conflict, or requested with `after:`. `previous`, labels and source order never reach the IR. |
| `runtime` | `load` validates IR, issues one grant per use and projects grants per node. `execute` runs nodes as their dependencies succeed (ties → lowest `NodeId`), tracks `NodeState`, blocks dependents of a failure, commits or rolls back, and always returns an `ExecutionReceipt` with outputs, schedule, node records, events and a structured `ExecutionError`. |
| `model` | The model contract: `InferRequest`/`InferResponse`, `Message`, `Role`, `InferParameters`, `FinishReason`, `Usage`, the deterministic `MockModel`/`MockFail`, and `model::local`, a scalar f32 CPU reference backend for one profile (llama2.c legacy v0 float32 checkpoint + its `tokenizer.bin`). No providers, transport, loops or templating. The runtime does not depend on it. |
| `shell-less` | Demonstration binary; prints the receipt the runtime produced. |

```sh
cargo run -p shell-less -- run --seed 42 examples/butcher.meat.yaml     # branching graph, two outputs
cargo run -p shell-less -- run --seed 42 examples/failure.meat.yaml     # A staged → failure → B blocked → rollback
cargo run -p shell-less -- run --seed 42 --show-outputs examples/composer.meat.yaml   # one source → tool + model
cargo run -p shell-less -- run --checkpoint stories15M.bin --tokenizer tokenizer.bin \
  --show-outputs examples/story.meat.yaml                                 # host-provisioned artifacts
cargo run -p shell-less -- run --seed 42 examples/thinker.meat.yaml    # /models/mock/infer → /state/answer
cargo run -p shell-less -- run --seed 42 examples/model-failure.meat.yaml
cargo run -p shell-less -- run --seed 42 examples/fork.meat.yaml        # model branch beside a tool branch
cargo run -p shell-less -- run --seed 42 examples/echo.meat.yaml
cargo run -p shell-less -- check examples/escalate.meat.yaml            # rejected by the compiler: undeclared
cargo run -p shell-less -- run examples/escalate-declared.meat.yaml     # rejected by host policy at load
```

### Validation

```sh
./scripts/validate.sh   # fmt --check, clippy -D warnings, tests, locked build; stops at the first failure
```

### Semantics

- **Ordering.** Only edges order execution. `NodeId` breaks ties among ready nodes for reproducibility and carries no meaning.
- **Write is create-or-replace.** Creating a new name happens when the node executes, inside the transaction. Loading never changes the namespace.
- **Atomicity.** Each execution is one transaction. If any node fails, every staged change is discarded. The audit journal still keeps the attempt (`staged`, `invoke failed`, `rolled back`).
- **Authority.** It is checked at compile time (declared `authority`), at load time (host `Policy`, all-or-nothing) and at every operation (the presented grant). A capability can never reach grants attached to another node.
- **Determinism.** The same graph, seed and initial state give an identical receipt.
- **Capability properties.** `Purity` (may it touch state?) and `Determinism` (same inputs, same output?) are independent. `InvocationMeta { implementation, revision }` records which implementation produced a result, and receipts record it for every invocation.
- **Composition.** `compose` builds a value from `literal`, `select` (key/index path into an earlier output), `map` and `list`. Each expression has exactly one constructor. It holds no grants, reads no state, and has no interpolation or evaluation. Every selected source is a data edge. Missing keys, wrong kinds, out-of-range indexes and host limit breaches are structured failures.
- **Content capture.** The host chooses `ContentCapture::Omit` (the default) or `Inline` before a program is parsed. Under it, content means payloads and source-controlled text: names, paths, map and selector keys, labels, output names and literals. A path is content even though it is also an address.
  - *Operational state* keeps the names it needs to act: paths, namespace bindings and live grant targets (a name grant still names its path).
  - *Records* retain content only as capture allows, decided when the record is made. Records are compile errors, load errors, receipts (recorded grant targets, node inputs and outputs, positional output records, error details, events) and journal events. An `Omit` record holds no copy of the text; it names source-controlled segments by position (`#i`) or by existing identities.
  - The journal records under a host-set policy (`set_journal_capture`, default `Omit`). `journal()` is an omitted view of all history. `journal_retained()` is the privileged raw history: it shows exactly what each record kept, and never reconstructs what one omitted.
  - Real results go to the caller through `ExecutionOutcome.outputs` (CLI: `--show-outputs`), which is never redacted. Programs and capabilities cannot raise capture. Omission is a promise about text, not a claim that structure, identities or outcomes reveal nothing.
- **Atomicity boundary.** Only MeatFS state is transactional. Effects outside MeatFS are not undone. A completed invocation stays `Succeeded`, and its node's `staged` field reports whether its MeatFS changes were kept. A commit conflict fails the whole execution (`TransactionConflict`, no node, no outputs), even if every node finished. A competing writer's commit is preserved, and nothing is retried.
- **Resolved execution plan.** The plan is the source IR, plus the capability declarations read at load, plus loader-derived ordering. Invocations not declared pure, including those of unknown purity, are chained along one deterministic topological order of the source graph (`derived_order` in the receipt). Every added edge points forward, so none can create a cycle. Declarations are pinned at load: if a capability declares something else at execution time, the call is refused (`DeclarationChanged`) before it runs. This checks declared metadata against the plan. It is not proof that a capability's behaviour is immutable; compiled-in capabilities remain trusted host code. `GraphId` identifies the source IR only.
- **Declared ≠ verified.** Purity, determinism and implementation identity are recorded as `declared`. Nothing yet verifies them, and they authorize no caching, retries or speculative parallelism. Compiled-in capabilities are trusted host code; grants bound their MeatFS access, not arbitrary Rust.
- **Local inference** (the Shell-less legacy-v0 completion profile). The host passes `--checkpoint` and `--tokenizer`. Both are read and validated in full (header, layout and length, limits, finite values, RoPE tables, tokenizer export layout) before the model is mounted at `/models/local/stories/infer`; a failure mounts nothing. Programs reach the model only through an invoke grant and cannot name, reopen or replace files. A request is a completion: exactly one user message, a required `max_tokens`, greedy decoding over finite logits, and host limits on prompt bytes, context, new tokens, output bytes (enforced after UTF-8 repair) and working memory. BOS and EOS end generation; a selected UNK fails it. Each invocation gets a fresh KV cache. The model is declared `Effectful` + `Nondeterministic`: greedy decoding uses no randomness, but f32 libm results may differ across platforms. Its logits match upstream `run.c` bit for bit on the reference platform. Provenance and procedure: `crates/model/reference/README.md`.
- **Models are capabilities.** A model is mounted at a path like `/models/<name>/infer` and invoked like anything else. Inference is declared `Effectful`, even for the deterministic mock, because real inference depends on weights, samplers, hardware and caches.

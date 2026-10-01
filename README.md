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
| `model` | The model contract: `InferRequest`/`InferResponse`, `Message`, `Role`, `InferParameters`, `FinishReason`, `Usage`, and the deterministic `MockModel`/`MockFail`. No providers, transport, loops or templating. The runtime does not depend on it. |
| `shell-less` | Demonstration binary; prints the receipt the runtime produced. |

```sh
cargo run -p shell-less -- run --seed 42 examples/butcher.meat.yaml     # branching graph, two outputs
cargo run -p shell-less -- run --seed 42 examples/failure.meat.yaml     # A staged → failure → B blocked → rollback
cargo run -p shell-less -- run --seed 42 examples/thinker.meat.yaml    # /models/mock/infer → /state/answer
cargo run -p shell-less -- run --seed 42 examples/model-failure.meat.yaml
cargo run -p shell-less -- run --seed 42 examples/fork.meat.yaml        # model branch beside a tool branch
cargo run -p shell-less -- run --seed 42 examples/echo.meat.yaml
cargo run -p shell-less -- check examples/escalate.meat.yaml            # rejected by the compiler: undeclared
cargo run -p shell-less -- run examples/escalate-declared.meat.yaml     # rejected by host policy at load
```

### Semantics

- **Ordering.** Only edges order execution. `NodeId` breaks ties among ready nodes for reproducibility and carries no meaning.
- **Write is create-or-replace.** Creating a new name happens when the node executes, inside the transaction. Loading never changes the namespace.
- **Atomicity.** Each execution is one transaction. If any node fails, every staged change is discarded. The audit journal still keeps the attempt (`staged`, `invoke failed`, `rolled back`).
- **Authority.** It is checked at compile time (declared `authority`), at load time (host `Policy`, all-or-nothing) and at every operation (the presented grant). A capability can never reach grants attached to another node.
- **Determinism.** The same graph, seed and initial state give an identical receipt.
- **Capability properties.** `Purity` (may it touch state?) and `Determinism` (same inputs, same output?) are independent. `InvocationMeta { implementation, revision }` records which implementation produced a result, and receipts record it for every invocation.
- **Models are capabilities.** A model is mounted at a path like `/models/<name>/infer` and invoked like anything else. Inference is declared `Effectful`, even for the deterministic mock, because real inference depends on weights, samplers, hardware and caches.

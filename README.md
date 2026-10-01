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
MEATYAML ─compile─▶ MEAT IR ─load─▶ AuthorityRequest ─policy─▶ GrantSet ─▶ resolved graph ─execute─▶ ExecutionReceipt
                    (nodes + edges)                                        (ObjectId + GrantId per node)
```

The IR is the contract. The Rust interpreter in `runtime` is its first backend. Speck/MEATASM and Hydra come later, as further consumers of the same IR.

| crate | role |
|---|---|
| `meatfs` | Namespace. `ObjectId` is identity; paths are names bound to it. Authority subsystem: `Policy` (what may be requested) vs `Grant` (what was issued: unforgeable, issuer-bound, retirable). Every op presents an `Access`. Journal events carry object, principal, grant, execution and node. |
| `capability` | Typed `Capability` trait with mandatory `CapabilityMeta { purity }`. Pure capabilities receive no effects handle at all. |
| `meatyaml` | Compiles MEATYAML into MEAT IR: `Graph { id, nodes, edges }`. `from: previous` and labels are source sugar resolved into `Edge`s. `GraphId` is a hash of canonical IR content. |
| `runtime` | `load` validates the IR, fixes a deterministic order, issues grants and binds each node to `ObjectId` + `GrantId`. `execute` runs it with no name lookups and returns an `ExecutionReceipt`. |
| `shell-less` | Demonstration binary; prints the receipt the runtime produced. |

```sh
cargo run -p shell-less -- run --seed 42 examples/echo.meat.yaml       # acceptance program
cargo run -p shell-less -- run --seed 42 examples/scout.meat.yaml
cargo run -p shell-less -- check examples/escalate.meat.yaml            # rejected by the compiler: undeclared
cargo run -p shell-less -- run examples/escalate-declared.meat.yaml     # rejected by host policy at issue
```

Authority is checked in three places:

1. The compiler rejects any step that the declared `authority` does not cover.
2. The host `Policy` refuses to issue anything outside it. Issuing is all-or-nothing.
3. MeatFS refuses any operation whose presented grant does not cover that object and right.

Knowing a pathname confers nothing. Identical graph, seed and initial state give an identical receipt.

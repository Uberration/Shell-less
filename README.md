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

## Milestone one

```text
MEATYAML → parse → typed graph → capability resolution → execution → filesystem-visible result
```

| crate | role |
|---|---|
| `meatfs` | in-memory namespace: data and capability objects, `read`/`write`/`invoke`/`subscribe`/`inspect`, explicit `Authority` on every op, event journal (audit), optimistic transactions |
| `capability` | typed `Capability` trait (`Input: FromValue`, `Output: IntoValue`), published signatures, built-ins |
| `meatyaml` | compiles MEATYAML into a `Program`: graph + declared authority, statically checked for least privilege |
| `runtime` | executes a graph under the program's authority alone |
| `shell-less` | the demonstration binary |

```sh
cargo run -p shell-less -- run   examples/echo.meat.yaml
cargo run -p shell-less -- run   examples/scout.meat.yaml
cargo run -p shell-less -- check examples/escalate.meat.yaml   # rejected: undeclared write
```

Authority is enforced twice. `meatyaml` refuses any flow step that its `input`/`output` declarations do not cover. `runtime` then executes with exactly those grants, so MeatFS refuses anything else at run time. No ambient authority exists.

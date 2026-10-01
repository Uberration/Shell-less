#!/usr/bin/env bash
# The validation gate. Each command's own exit status decides; nothing is
# filtered through a pipe, and the first failure stops the run.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --locked -p shell-less --bin shell-less
echo "validate: all checks passed"

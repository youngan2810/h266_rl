# Reproducibility and source management

## Upstream source pin

- Repository: `https://github.com/fraunhoferhhi/vvenc.git`
- Required checkout: `9428ea8636ae7f443ecde89999d16b2dfc421524` (VVenC v1.14.0)
- License: [The Clear BSD License](https://github.com/fraunhoferhhi/vvenc/blob/9428ea8636ae7f443ecde89999d16b2dfc421524/LICENSE.txt), copyright (c) Fraunhofer-Gesellschaft & The VVenC Authors. Preserve copyright/disclaimer if redistributing upstream source/binaries. No patent licenses are granted by this license.

The repository contains **research bridge files and a pinned overlay**, not a fork, submodule, binary, or wholesale copy of VVenC. This avoids a duplicate upstream history and keeps project code reviewable. The overlay is deliberately strict: it rejects a dirty tree, a mismatching Git HEAD, missing source anchors or a pre-existing bridge header. Apply it only to a clean clone; do not run twice.

## Recreate source

```bash
set -eu
# Set REPO_ROOT to the location of this repository.
REPO_ROOT="$(pwd)/h266_rl"
PIN=9428ea8636ae7f443ecde89999d16b2dfc421524

git clone https://github.com/fraunhoferhhi/vvenc.git vvenc_stock
git -C vvenc_stock checkout --detach "$PIN"
cp -a vvenc_stock vvenc_qp_rl
python3 "$REPO_ROOT/scripts/apply_to_vvenc.py" --vvenc "$(pwd)/vvenc_qp_rl"

git -C vvenc_stock rev-parse HEAD
git -C vvenc_qp_rl diff --stat
git -C vvenc_qp_rl status --short
```

Compile stock and patched builds with identical toolchains, build type and codec parameters. Build commands and smoke instructions are in [README](../README.md).

## Expected paths and dependencies

The overlay's header path resolves relative to its own script; there are **no session-specific absolute paths in the scripts**. Python dependencies: NumPy; C++ tests: g++ with C++17; encoder build: CMake and a suitable compiler; decoder check: a separately obtained and built `vvdecapp`. Its tested exact commit **has not yet been recorded**. Record compiler/CMake versions and actual `vvdecapp` SHA/version when performing acceptance testing.

## Data policy

Do not commit BDD100K media, raw or reconstructed videos, `.266` bitstreams, trained weights, Kaggle outputs, build trees, credentials, private traces or run logs. Use the included synthetic YUV generator for smoke tests and write logs under `runs/` (ignored). A public repo must not be treated as a data publication channel.

## Research scope

Stage 3 changes only QP control and instrumentation. Stage 4 (RL environment and reward), Stage 5 (PPO training), and Stage 6 (full evaluation) are out of scope and have **not been implemented** here.

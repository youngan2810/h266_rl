# H.266/VVC Native Reinforcement Learning for Task-Aware Video Coding

**Stage 3 source-code backup — State Extraction & QP Control (pre-PPO)**

> **Research status:** The standalone C++/Python bridge and its 10 unit tests have been verified locally. **Integration with a full VVenC build, bitstream equivalence, and VVdeC decoding have not yet been validated.** This repository must not be described as a codec-conformance-tested or PPO-ready implementation.

## Project overview

The project investigates **VVC-native reinforcement learning for task-aware H.266/VVC video compression**, initially targeting car detection. It takes conceptual inspiration from **[RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html)**, but **does not transfer x264/H.264 features or block-level PPO architecture directly** to VVenC.

The Stage 3 scope is **QP control only**: expose a causal VVenC-derived picture observation, request a discrete QP delta from an external deterministic action provider, inject that delta into the encoder's QP/lambda path, and log applied values for subsequent verification. **No PPO training, reward integration, partition control, or MTS control is included.**

## Source and reproducibility pin

| Component | Selection |
| --- | --- |
| Video encoder | [Fraunhofer HHI VVenC](https://github.com/fraunhoferhhi/vvenc) v1.14.0 |
| Exact VVenC commit | `9428ea8636ae7f443ecde89999d16b2dfc421524` |
| VVC decoder | [Fraunhofer HHI VVdeC](https://github.com/fraunhoferhhi/vvdec) (pin exact tested version when available) |
| Rate control pilot | Positive target bitrate, one pass with look-ahead |
| Native perceptual QPA | Remains enabled for the intended pilot configuration |
| Initial action space | Picture-level `ΔQP ∈ {-2,-1,0,+1,+2}` |
| Spatial action | Experimental CTU/QG map; **not validated** |
| Observation | `vvrl.v1` S0: 8 logical groups / 11 float32 components |
| Initial provider | Zero, constant, stdin dummy; no PPO |

VVenC is **not vendored** into this repository. The included fail-closed overlay applies research modifications only to a **clean checkout of the exact upstream commit**. This is preferable to committing a modified copy of the complete encoder, and preserves VVenC's upstream history and license notices. The upstream [Clear BSD License](https://github.com/fraunhoferhhi/vvenc/blob/9428ea8636ae7f443ecde89999d16b2dfc421524/LICENSE.txt) remains applicable to that code; it does **not** grant patent rights.

## Stage 3 architecture

```text
Input video (YUV) ──> VVenC preprocessing / native rate control
                                     │
                                     ▼
                     VVC-native State Extractor (S0, causal)
                                     │   observation JSON / vvrl.v1
                                     ▼
                  Deterministic Action Provider (or Python stdin)
                                     │   picture ΔQP (-2...+2)
                                     ▼
                  QP + lambda injection, then native QPA/RDO
                                     │
                                     ▼
                    VVenC bitstream (.266) + JSONL telemetry
                                     │
                                     ▼
                       VVdeC validation (not yet run)
```

The **8 S0 groups** are target BPP, normalized cumulative budget error, native proposed QP, I/P/B slice type, temporal layer, GOP-relative position, previous completed-picture QP, and previous completed-picture actual/target bit ratio. One-hot type and a history-valid flag yield **11 float32 values** in `scripts/state_schema.py`. S1 spatial/temporal activity remains invalid/masked pending a configuration-specific availability audit. No current-frame RDO/residual outcomes are used as pre-action observations.

**Important:** Picture `qp_requested`, `qp_accepted`, final slice QP and effective CU QPs are distinct. The source contains optional QPA and quantization-group behavior that may change QP again. The Stage 3 telemetry alone is **not proof of bitstream-signaled QP conformance**.

## Repository contents

```text
h266_rl/
├── .gitignore                      # Secrets, datasets, media, binaries, caches
├── README.md                       # Research overview and usage
├── configs/pilot.env.example       # Optional dummy-mode environment example
├── src/VVRLControl.h               # Header-only State/Action/JSONL bridge
├── scripts/
│   ├── apply_to_vvenc.py           # Source-pinned, fail-closed VVenC overlay
│   ├── state_schema.py             # Observation validation and float32 mapping
│   ├── run_interactive_dummy.py    # Python stdin action provider
│   ├── run_smoke.py                # Designed A/B/C/D/E codec smoke tests
│   ├── validate_log.py             # Offline JSONL causal/format assertions
│   └── generate_yuv420.py          # Small synthetic video for smoke test
├── tests/
│   ├── test_control.cpp            # Standalone C++ protocol driver
│   └── test_schema.py              # 10 Python/C++ unit tests
└── docs/
    ├── ARCHITECTURE.md             # Source hook details and causal boundaries
    ├── VALIDATION.md               # Evidence-backed test status
    └── REPRODUCIBILITY.md          # Fresh checkout, build, test, licenses
```

## Setup and dependencies

Install `git`, `cmake`, a **C++17 compiler**, `python3`, and `numpy`. For bitstream validation, build the upstream **VVdeC** decoder separately. The source is intended to run on Linux/Kaggle; **CPU encoding** is the expected VVenC workload (Kaggle GPU does not make VVenC GPU-accelerated).

```bash
# From the parent directory of the research repository
REPO_ROOT="$(pwd)/h266_rl"
VVENC_PIN=9428ea8636ae7f443ecde89999d16b2dfc421524

git clone https://github.com/fraunhoferhhi/vvenc.git vvenc_stock
git -C vvenc_stock checkout --detach "$VVENC_PIN"
cp -a vvenc_stock vvenc_qp_rl
python3 "$REPO_ROOT/scripts/apply_to_vvenc.py" --vvenc "$(pwd)/vvenc_qp_rl"

git -C vvenc_qp_rl diff --stat
cmake -S vvenc_stock -B vvenc_stock/build -DCMAKE_BUILD_TYPE=Release
cmake --build vvenc_stock/build -j 2
cmake -S vvenc_qp_rl -B vvenc_qp_rl/build -DCMAKE_BUILD_TYPE=Release
cmake --build vvenc_qp_rl/build -j 2
find vvenc_stock/build vvenc_qp_rl/build -type f -name vvencapp
```

The overlay refuses a dirty checkout, an incorrect commit, missing anchors, or a pre-existing `VVRLControl.h`. It modifies the four upstream `.cpp` files documented in [Architecture](docs/ARCHITECTURE.md) and installs the header. **Passing these checks does not guarantee the full encoder compiles.**

## Tests

Run **standalone tests** (do not require VVenC binaries):

```bash
python3 -m pip install numpy
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
```

Run the **actual codec smoke suite**, after building both encoders and providing a working `vvdecapp` binary:

```bash
python3 scripts/run_smoke.py \
  --stock /absolute/path/to/stock/vvencapp \
  --patched /absolute/path/to/patched/vvencapp \
  --decoder /absolute/path/to/vvdecapp \
  --work ./runs/vvrl_smoke --frames 12 --bitrate 300000
```

The script creates a small synthetic 480×320 YUV420 clip (not BDD100K) and schedules:

- **A:** stock versus `ΔQP=0` bitstream SHA256 identity.
- **B:** fixed `ΔQP=+1/-1`, with logs and actual bitrate/PSNR readouts.
- **C:** experimental checkerboard CTU/QG action-map and post-coding CTU summaries.
- **D:** VVdeC decoding, frame-byte count and reconstructed PSNR-Y.
- **E:** causal JSONL observations, 11-dimensional finite float32 schema and POC-linked history.

**These are test definitions, not successful end-to-end test results.** `--decoder` must be specified for D. Store encoder logs, `.266` bitstreams and reconstructed videos outside Git (e.g., `runs/`). Review [Validation](docs/VALIDATION.md) before relying on any results.

### Dummy provider

For an independent `ΔQP=0` run, set `VVCRL_MODE=zero`; for constant `ΔQP`, set `VVCRL_MODE=constant` and `VVCRL_DELTA` to an integer in `[-2,2]`. The C++ bridge supports `stdin`/`stdin-map` with the prefixed `VVCRL_OBS` protocol. Example interactive driver:

```bash
mkdir -p runs
python3 scripts/run_interactive_dummy.py \
  --encoder /absolute/path/to/patched/vvencapp \
  --action alternate --log ./runs/interactive.jsonl -- \
  --input /absolute/path/to/clip.yuv --size 480x320 --fps 30 --frames 12 \
  --output ./runs/interactive.266 --bitrate 300000 --passes 1 \
  --preset medium --qpa 1 --threads 0 --mtprofile 0 \
  --additional 'GOPSize=32 LookAhead=1 MaxParallelFrames=0'
python3 scripts/validate_log.py ./runs/interactive.jsonl
```

This pilot reserves process `stdin` for actions: **the input YUV must be a file**, not standard input. Use serial encoding for the initial causality check. The actual codec build and command-line compatibility must still be validated on the pinned checkout.

## Implementation status and roadmap

| Stage | Status | Evidence / next gate |
| --- | --- | --- |
| **Stage 3** — State extraction / QP injection | **Source prepared; integration unvalidated** | 10/10 standalone tests; require full VVenC build and smoke A/B/D/E |
| Experimental CTU/QG spatial QP map | **Implemented in overlay, unvalidated** | Requires map/QG signaling and VVdeC checks |
| **Stage 4** — RL Environment & Reward | Planned, **not implemented** | Not in this commit |
| **Stage 5** — PPO training | Planned, **not implemented** | No model or training results |
| **Stage 6** — Evaluation & ablation | Planned, **not implemented** | No benchmark claims |
| Future — Partition & MTS control | Future research, **not implemented** | Explicitly out of scope |

## Privacy, data and licensing

**No BDD100K videos/labels, model weights, decoded frames, training results, access credentials, Kaggle outputs, or VVenC/VVdeC binaries are included.** The test generator creates its own synthetic clip locally. Keep datasets and credentials outside the repository and inspect `git status` before pushing any future changes.

**Caution:** The original VVenC repository is maintained under its Clear BSD License; this backup contains *only project-side source/overlay instructions*, not a vendored VVenC tree. A standalone redistribution license for the new bridge code has **not** been selected, so do not assume the same license covers the project-authored files.

## References

- [RL-RC-DoT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html)
- [Fraunhofer HHI VVenC](https://github.com/fraunhoferhhi/vvenc)
- [Fraunhofer HHI VVdeC](https://github.com/fraunhoferhhi/vvdec)

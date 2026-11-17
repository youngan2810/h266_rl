# Stage 3 validation record — 2026-11-17

Evidence classifications: **PASS (executed locally)**, **SOURCE-CHECKED (static source/anchor inspection)**, **IMPLEMENTED/UNVERIFIED (code exists, no actual VVenC run)**, **NOT IMPLEMENTED**.

| Check | Status | Evidence / limitation |
| --- | --- | --- |
| Exact target VVenC commit/source anchors | SOURCE-CHECKED | Official upstream commit `9428ea8...`; overlay's four source anchors inspected |
| Standalone `VVRLControl.h` C++ compilation | PASS | Invoked by unit tests via `g++ -std=c++17` |
| Observation schema, finite values, float32 `(11,)` | PASS | Unit test suite |
| Dummy zero/constant modes | PASS | Header-only driver; **not** actual VVenC encode |
| Interactive ACT POC validation / map shape | PASS | Header-only driver; no codec run |
| Python syntax check | PASS | `python -m compileall -q scripts tests` |
| Full VVenC overlay application on real checkout | IMPLEMENTED/UNVERIFIED | `scripts/apply_to_vvenc.py` prepared; full upstream checkout unavailable locally |
| Patched VVenC build | IMPLEMENTED/UNVERIFIED | No native VVenC build log |
| Test A: no-op bit-identical vs stock | IMPLEMENTED/UNVERIFIED | `run_smoke.py` defines test; no actual `.266` evidence |
| Test B: constant ΔQP changes coding behavior | IMPLEMENTED/UNVERIFIED | QP/lambda path proposed; not measured on encoder |
| Test C: spatial QP map/QG conformance | IMPLEMENTED/UNVERIFIED | Experimental CTU map code; not decoder-validated |
| Test D: VVdeC decoding | IMPLEMENTED/UNVERIFIED | Decoder not executed; no frame-count/PSNR proof |
| Test E: full encoder state causality | IMPLEMENTED/UNVERIFIED | Offline validator exists; no actual encoder JSONL |
| Gymnasium RL environment + reward | NOT IMPLEMENTED | Stage 4 |
| PPO training and BD-Rate study | NOT IMPLEMENTED | Stages 5–6 |

## Reproduced local unit-test output

Environment at time of test: Python 3.13.5; g++ 14.2.0; source workspace `/mnt/data/vvrl_stage1` before publication, then reorganized to this repository with only source-relative include/import path adjustments.

```text
Ran 10 tests in 9.701s
OK
```

**After repository normalization:** Python 3.13.5 and g++ 14.2.0, `python -m unittest discover -s tests -v` completed **10/10 PASS** in 9.498s; `python -m compileall -q scripts tests` completed successfully. The C++ and Python tests exercised the same implementations after path-only relocation. A clean test suite **does not imply** bitstream identity or VVC conformance.

## Exit gates before Stage 4

1. `vvenc_stock` and `vvenc_qp_rl` compile with the exact pin, same flags and toolchain.
2. Test A: `ΔQP=0` produces the exact same bitstream hash as unmodified VVenC under the same settings.
3. Test B: positive/negative requested ΔQP changes actual coding QP/lambda as expected; record actual bits/PSNR, not a presumed monotonic outcome.
4. Test D: real VVdeC decode succeeds with correct frame count and dimensions.
5. Test E: encoder-sourced observations/feedback are causal and schema-valid.
6. Test C remains a **separate experimental gate**; do not call spatial map production-ready before QG/decoder checks.

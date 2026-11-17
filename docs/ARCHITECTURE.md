# Stage 3 architecture and instrumentation (source-pinned)

**Scope:** Picture-level QP control and an explicitly experimental CTU/QG action-map path. No partition/MTS controller, PPO, or downstream reward integration.

## Verified source boundary

Upstream VVenC **v1.14.0**, `9428ea8636ae7f443ecde89999d16b2dfc421524`:

1. `source/App/vvencapp/vvencapp.cpp`: `main()` calls `vvenc_encode()`.
2. `source/Lib/EncoderLib/EncLib.cpp`: `EncLib::encodePicture()` processes input/AU scheduling.
3. `source/Lib/EncoderLib/EncGOP.cpp`: `EncGOP::xEncodePicture()` calls `RateCtrl::initRateControlPic()` when target bitrate > 0, then `EncPicture::compressPicture()`.
4. `source/Lib/EncoderLib/EncSlice.cpp`: `EncSlice::xInitSliceLambdaQP()` initializes `slice->sliceQp` and quantization lambdas; QPA may adapt the proposal.
5. `source/Lib/EncoderLib/EncCu.cpp`: `EncCu::xCompressCU()` performs native CTU/QG QP adaptation when enabled; this overlay has an optional post-QPA offset hook.
6. `source/Lib/EncoderLib/RateCtrl.cpp`: `RateCtrl::updateAfterPicEncRC()` updates history *after* encoding feedback is available.
7. `source/Lib/EncoderLib/CABACWriter.cpp`: `CABACWriter::cu_qp_delta()` codes predicted CU QP deltas, subject to VVC QG logic.
8. `source/Lib/CommonLib/TrQuant.cpp`: `TrQuant::transformNxN()` invokes quantization using `QpParam`.

The overlay source anchors have been checked against the pinned upstream commit. A full patched VVenC compilation has **not** been performed.

## Implemented bridge interface

`src/VVRLControl.h` is header-only and copied into `source/Lib/EncoderLib/VVRLControl.h` by `scripts/apply_to_vvenc.py`. It contains `vvrl::Observation`, `vvrl::Controller`, fail-fast validation and JSONL events. `scripts/state_schema.py` parses observation records into 11 finite float32 components. Available pilot modes: `off`, `zero`, `constant`, `stdin`, `map-checker`, `map-file`, `stdin-map`.

The overlay modifies exactly these upstream files:

| File | Hook | Status |
| --- | --- | --- |
| `EncGOP.cpp` | Pre-picture observation, requested/accepted picture ΔQP, initial lambda scaling; coding CU QP summary after compression | Prepared; not build-tested |
| `EncSlice.cpp` | Log post-initialization slice QP | Prepared; not build-tested |
| `RateCtrl.cpp` | Update previous-QP/bit-ratio history on completed-picture RC feedback | Prepared; not build-tested |
| `EncCu.cpp` | Optional post-native-QPA CTU/QG offset with lambda update | Experimental; no conformance validation |

## Causality and rate control

Only native rate-control data, GOP metadata and *completed-picture* feedback enter the pre-action S0. Current-picture residual energy, coded bits, RDO candidate scores, reconstructed PSNR and YOLO outputs **do not** enter this State. Coding order may differ from POC order. Source readouts and log causal validation do not by themselves prove all encoder events are correctly synchronized under parallel settings; the pilot assumes `--threads 0` / `MaxParallelFrames=0`.

Picture action contract: native QP and lambda are observed **before** RL intervention, a bounded integer ΔQP is requested, QP is clipped to the bit-depth-dependent valid range, and picture initial lambda is scaled by `2^(ΔQP_effective/3)`. Native Rate Control and QPA remain in use. A post-QPA slice/CU QP can differ from accepted picture QP.

The CTU/QG branch is research-only. `PPS.useDQP`, current quantization group, residual and entropy signaling must be checked using a real encoder/decoder. A per-CTU average of CU QPs is **not** a decoder-confirmed QP trace.

## Log schema

`vvrl.v1` JSONL event kinds: `observation`, `picture_action`, `slice_qp`, `coding_qp`, `coding_ctu_qp`, `ctu_map_action`, `feedback`. Observation metadata includes POC, CTU raster grid, source geometry, valid flags and rate/QP context. The map modes use CTU raster indices only as a *proposal* to QG logic, not a guarantee that every CU QP is individually signaled.

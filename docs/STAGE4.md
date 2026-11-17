# Stage 4 — Reward and RL interface (WIP)

**DO NOT MERGE OR LABEL PPO-READY BEFORE THE VALIDATION GATES BELOW PASS.**

## Existing Stage 3 boundary

Pinned VVenC v1.14.0 at `9428ea8636ae7f443ecde89999d16b2dfc421524`; native one-pass rate control and QPA remain enabled. `scripts/apply_to_vvenc.py` is unchanged. No partition/MTS control has been added. Stage 3 remains subject to unverified full-build and bitstream conformance gates. Existing observations are still `vvrl.v1`, shape `(11,)`, `float32`; categorical picture actions map `[0,1,2,3,4]` to `[-2,-1,0,1,2]`.

## Pipeline actually represented

```
Real 480x320 YUV420p sequence
  -> VVenC v1.14.0 patched with Stage 3 stdin bridge
  -> POC-specific discrete ΔQP and resulting .266 + real per-picture bit counts
  -> real VVdeC output (strict YUV frame count / pixel format validation)
  -> raw/decoded RGB (same POC and image geometry)
  -> same local COCO YOLOv5n detector + fixed preprocessing/settings
  -> pseudo-label car precision, running bitrate and reward
  -> next pre-action VVenC state as emitted in the same full-clip replay.
```

**Execution caveat:** `FullSequenceReplay.run()` encodes **all frames in a continuous sequence** on every invocation. All actions chosen so far are replayed; unknown future actions default to ΔQP=0. This is **not** a live online single-encode `step()`! It preserves reference-frame operation *inside* each full-clip encoding but incurs O(T²) work, and earlier reconstructed frames can depend on future coding-order references / their QPs. For future PPO, a persistent AU-level synchronous bridge with proper POC-keyed delayed rewards and real decoder output is required.

The 5-return Gymnasium adapter `VVCReplayEnv` exists only for integration exploration. Its declaration does **not** mean the environment is suitable for PPO; neither RL training nor PPO network has been implemented.

## Paper reward and deliberate interpretation choices

RL-RC-DoT (CVPR 2025) states `r_rate = -|log(R_current / R_target)|`, `r = r_rate + λr_task`. Appendix 7.7 uses `λ=20` and Sec. 4.4 uses YOLOv5n car detection precision as the task measure. This code defines `R_current(t)=fps*sum(actualTotalBits[POC<=t])/(t+1)` from encoder feedback. This is an *explicit presentation-prefix interpretation*: the paper does not supply a unique executable implementation for every detail of current-average bitrate.

`r_task`: the same frozen local YOLOv5n checkpoint detects COCO class 2 (`car`) on source and on VVdeC-reconstructed RGB at identical resolution and settings. `precision_proxy` uses one-to-one greedy confidence ordered matching at configurable IoU threshold 0.5. These threshold/matching details **are not asserted to reproduce the original authors' evaluator**. Neither real BDD annotations nor test videos are used at inference.

**Empty detections:** precision is undefined if raw or recon lacks detections; code raises `InvalidReward` instead of returning a fabricated zero. A scientifically justified, predeclared policy and population statistics are needed before PPO. Model input normalization and frame matching must be reviewed with real output. Explicit `detector_confidence=0.25`, NMS IoU `.45`, match IoU `.5` are starting conventions, **not verified values from the paper**.

## How to run (Kaggle or machine with VVenC/VVdeC and YOLOv5n)

```bash
# From project root:
python -m pip install -r requirements-stage4.txt
PYTHONPATH=src:. python -m unittest discover -s tests -v

python scripts/prepare_bdd_video.py \
  --input /path/to/BDD100K_train_clip.mov \
  --output /path/to/work/train_clip_480x320.yuv --fps 30 --frames 8

# Construct VVenC v1.14.0 from exact commit and apply Stage 3:
# python scripts/apply_to_vvenc.py --vvenc /path/to/clean/vvenc
# cmake -S /path/to/clean/vvenc -B /path/to/clean/vvenc/build -DCMAKE_BUILD_TYPE=Release
# cmake --build /path/to/clean/vvenc/build -j 2

PYTHONPATH=src:. python scripts/run_stage4_smoke.py \
  --encoder /path/to/patched/vvencapp \
  --decoder /path/to/vvdecapp \
  --yuv /path/to/work/train_clip_480x320.yuv \
  --yolov5-repo /path/to/yolov5 --weights /path/to/yolov5n.pt \
  --workdir /path/to/work/stage4 --fps 30 --bitrate 300000 --steps 1
```

**Logs**: `stage4_rewards.jsonl`, `rollout_XXXX/control.jsonl`, actual `.266`, actual VVdeC YUV and codec stderr logs. All are local, intentionally gitignored. Settings and SHA of YOLO weights must be recorded for the experimental report. No metrics are claimed here.

## Verification checklist

- [x] Deterministic math unit tests for rate reward, car precision and invalid inputs (run locally).
- [x] Strict YUV420 file/shape unit tests (run locally).
- [x] Stage 3 standalone Python/C++ tests (run locally).
- [ ] Compiled patched VVenC (not available in this environment).
- [ ] Stage 3 noop/constant/spatial action codec smoke (not available).
- [ ] VVdeC bitstream conformance + frame alignment (not available).
- [ ] Frozen YOLOv5n source / weight integrity on representative BDD clips (not available).
- [ ] Real end-to-end single-sequence reward logs and performance timings (not available).
- [ ] Valid synchronized online step semantics (NOT IMPLEMENTED).
- [ ] Scientifically resolved empty-detection and matching policy for PPO (NOT FINALIZED).

Stage 4 is **WIP** and must remain on the feature branch. No generated experiment result or dataset is bundled with the repository.

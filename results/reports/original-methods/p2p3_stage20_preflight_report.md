# P2-to-P3 Distillation Stage20 Preflight

Date: 2026-08-05 (Asia/Shanghai)

## Protocol

- D0-D3 use BDD100K, native YOLO26n student, COCO initialization, 640 input, batch16/nbs64, seed0, AMP, the 200-epoch schedule, and a resumable pause after epoch20.
- D1 uses mapped common-level KD, D2 adds the P2-to-P3 bridge and continuous small-object mask, and D3 adds dual-branch consensus.
- The CUDA preflight used physical GPU0, real frozen weights, and one deterministic batch containing 16 images and 768 mixed-size targets.

## Results

| Group | Extra loss | Peak allocated | Peak reserved | Finite gradients | Teacher gradient |
|---|---:|---:|---:|---|---|
| D0 | n/a | 2.271 GiB | 2.480 GiB | yes | n/a |
| D1 | 6.195974 | 2.351 GiB | 2.557 GiB | yes | no |
| D2 | 0.165322 | 2.413 GiB | 2.705 GiB | yes | no |
| D3 | 0.128062 | 2.413 GiB | 2.705 GiB | yes | no |

The three native detection-loss components were exactly identical across D0-D3 before adding distillation. No OOM, CPU fallback, NaN, Inf, missing trainable gradient, or teacher gradient occurred. D1 retains the frozen upstream-equivalent loss definition and `dis=6`; its batch distillation component remained of the same order as the detection components, so it was not changed using validation accuracy.

Machine-readable evidence: `results/p2p3-stage20-preflight.json`.

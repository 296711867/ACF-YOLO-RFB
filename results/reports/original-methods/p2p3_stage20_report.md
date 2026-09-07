# P2-to-P3 Small-Object Distillation Stage20 Report

Date: 2026-08-06 (Asia/Shanghai)

## Protocol

All groups used BDD100K 70k/10k, native YOLO26n students, identical COCO initialization, 640 input, batch16/nbs64, seed0, AMP, and the same 200-epoch schedule. Each run saved complete optimizer, scaler, and EMA state and paused after epoch20. D1 used common P3/P4/P5 feature KD, D2 added the P2-to-P3 bridge and continuous small-object mask, and D3 added dual-branch consensus.

Frozen implementation and run commit: `c927ca25cbd9b7cef724ded8b8572f005a4c5570`. Frozen teacher SHA256: `f1a8ad980cdaf949a12ac98514b3ccc583e1026fe4b09aa664e8d2589bc2826c`.

## Results

| Group | all AP | small AP | small AR | medium AP | large AP | small AP delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| D0 native control | 0.226392 | 0.073483 | 0.205715 | 0.252215 | 0.482080 | reference | retain control |
| D1 common KD | 0.217824 | 0.068147 | 0.194927 | 0.242745 | 0.477988 | -7.26% | stop |
| D2 bridge + small mask | 0.226810 | 0.074534 | 0.207550 | 0.252766 | 0.481962 | +1.43% | stop |
| D3 D2 + consensus | 0.226618 | 0.074529 | 0.206599 | 0.251753 | 0.484899 | +1.42% | stop |

D1 degraded all AP by 3.78%, small AP by 7.26%, and small AR by 5.24%. D2 and D3 kept all/medium/large metrics effectively neutral and produced positive small AP/AR changes, but their small AP gains were only 1.43% and 1.42%, below the preregistered 5% Stage20 gate. D3 did not improve on D2, so dual-branch consensus has no supported incremental benefit at this stage.

All four runs completed without OOM, CPU fallback, NaN, Inf, automatic fitness recovery, or checkpoint-state failure. Engineering feasibility therefore passed, but the accuracy hypothesis failed its frozen gate. D1-D3 must not resume to Stage40, run to epoch200, or trigger loss-weight tuning or additional variants.

Machine-readable summary and hashes: `results/p2p3-stage20/summary.json` and `results/p2p3-stage20/manifest.json`. Per-group full 10k size metrics and ignored raw prediction JSONL files are stored in the same directory. Runtime matrix evidence remains under `runs/detect/phase4-distill-stage20/`.

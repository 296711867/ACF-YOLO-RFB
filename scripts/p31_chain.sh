#!/bin/bash
# P31 chain (v2): L0+RFB x K1 output distillation -> resume-to-200 -> terminal eval -> timeofday -> 3 bootstraps -> commit.
# v2 fix (2026-10-14): v1 relied on run_experiment exiting after stage_stop_epoch(20) and then
# mislabeled 20-epoch artifacts as ep200. Now: training resumes until results.csv epoch == 200;
# eval/bootstrap steps only run at epoch 200. 20-epoch artifacts were renamed to *-ep20.
set -euo pipefail
cd /home/<user>/code/github-project/<internal-project>/yolo
export PYTHONPATH=upstream:.
PY=/home/<user>/miniconda3/envs/yolo26-sdf/bin/python
GIT="git --git-dir=.project.git --work-tree=."
RUN=runs/detect/phase31-kd-rfb/bdd100k-p31-kd-k1-rfb-640-e200-b16-seed0
OUT=results/p31-kd-rfb
CFG=configs/train/bdd100k-p31-kd-k1-rfb-640-e200-b16-seed0.yaml

echo "[1/4] training to epoch 200 (GPU0)"
if [ ! -f "$RUN/results.csv" ]; then
  $PY scripts/run_experiment.py --config "$CFG"
else
  EPOCH=$(tail -1 "$RUN/results.csv" | cut -d, -f1)
  if [ "$EPOCH" != "200" ]; then
    echo "  at epoch $EPOCH, resuming to 200"
    $PY scripts/resume_experiment.py --run-dir "$RUN"
  else
    echo "  training already complete (200), skipping"
  fi
fi
EPOCH=$(tail -1 "$RUN/results.csv" | cut -d, -f1)
if [ "$EPOCH" != "200" ]; then
  echo "ERROR: training not complete (epoch $EPOCH) - aborting before eval"; exit 1
fi

echo "[2/4] terminal 10k size evaluation + timeofday slices (epoch 200)"
$PY scripts/evaluate_object_sizes.py \
  --model "$RUN/weights/best.pt" \
  --data configs/data/bdd100k.yaml \
  --output "$OUT/k1-rfb-size-val-ep200.json" --device 0
$PY scripts/evaluate_timeofday_slices.py \
  --attributes data/BDD100K/label_source_2025/val_scalabel \
  --predictions "$OUT/k1-rfb-size-val-ep200.predictions.jsonl" \
  --output "$OUT/k1-rfb-timeofday-ep200.json"

echo "[3/4] three paired bootstraps (1000x, seed 260805)"
$PY scripts/bootstrap_bdd_paired_ap.py \
  --baseline results/kd-arms/c-b16-size-val-ep200.predictions.jsonl \
  --candidate "$OUT/k1-rfb-size-val-ep200.predictions.jsonl" \
  --output "$OUT/bootstrap-k1-rfb-vs-c-b16-ep200.json" \
  --replicates 1000 --seed 260805 --workers 8
$PY scripts/bootstrap_bdd_paired_ap.py \
  --baseline results/p29-context-screen/rfb-size-val-ep200.predictions.jsonl \
  --candidate "$OUT/k1-rfb-size-val-ep200.predictions.jsonl" \
  --output "$OUT/bootstrap-k1-rfb-vs-l0-rfb-ep200.json" \
  --replicates 1000 --seed 260805 --workers 8
$PY scripts/bootstrap_bdd_paired_ap.py \
  --baseline results/kd-arms/k1-size-val-ep200.predictions.jsonl \
  --candidate "$OUT/k1-rfb-size-val-ep200.predictions.jsonl" \
  --output "$OUT/bootstrap-k1-rfb-vs-k1-ep200.json" \
  --replicates 1000 --seed 260805 --workers 8

echo "[4/4] commit evidence"
$GIT add "$OUT"
$GIT -c user.name="YOLO26-SDF Project" -c user.email="yolo26-sdf@localhost" \
  commit -m "P31 K1-on-RFB ep200 terminal evidence + three bootstraps (auto chain v2)" || echo "  nothing to commit"
echo "P31 chain done."

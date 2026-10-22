#!/bin/bash
# P29b RFB-3S recovery: run AFTER the machine reboots from the CUDA Error 804
# driver/library mismatch (2026-10-11). Idempotent - safe to re-run.
# Context: seed1 training finished (ep200, best.pt intact) but the chain died
# before its terminal evaluation because NEW cuda processes could not init.
# seed2 never started. See P29 protocol §7 and SESSION_HANDOFF.
set -euo pipefail
cd /home/<user>/code/github-project/<internal-project>/yolo
export PYTHONPATH=upstream:.
PY=/home/<user>/miniconda3/envs/yolo26-sdf/bin/python
GIT="git --git-dir=.project.git --work-tree=."

echo "[0/4] CUDA self-check"
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA still broken - do not proceed'; x=torch.zeros(1).cuda(); print('CUDA OK:', x.device)"

echo "[1/4] seed1 terminal evaluation (training already done)"
if [ ! -f results/p29-context-screen/bootstrap-rfb-s1-vs-l0-s1-ep200.json ]; then
  $PY scripts/evaluate_object_sizes.py \
    --model runs/detect/phase29b-rfb-3seed/bdd100k-p29-rfb-s1-640-e200-b32-seed1/weights/best.pt \
    --data configs/data/bdd100k.yaml \
    --output results/p29-context-screen/rfb-s1-size-val-ep200.json --device 1
  $PY scripts/evaluate_timeofday_slices.py \
    --attributes data/BDD100K/label_source_2025/val_scalabel \
    --predictions results/p29-context-screen/rfb-s1-size-val-ep200.predictions.jsonl \
    --output results/p29-context-screen/rfb-s1-timeofday-ep200.json
  $PY scripts/bootstrap_bdd_paired_ap.py \
    --baseline results/acfr-stage20/l0-s1-size-val-ep200-regen.predictions.jsonl \
    --candidate results/p29-context-screen/rfb-s1-size-val-ep200.predictions.jsonl \
    --output results/p29-context-screen/bootstrap-rfb-s1-vs-l0-s1-ep200.json \
    --replicates 1000 --seed 260805 --workers 8
  $GIT add results/p29-context-screen/
  $GIT -c user.name="YOLO26-SDF Project" -c user.email="yolo26-sdf@localhost" \
    commit -m "P29b RFB seed1 ep200 terminal evidence + per-seed bootstrap (recovered after CUDA-804 reboot)"
else
  echo "  seed1 evaluation already present, skipping"
fi

echo "[2/4] seed2 training (sequential relay on GPU1, device-field override per protocol 7.4)"
sed 's/device: "0"/device: "1"/' configs/train/bdd100k-p29-rfb-s2-640-e200-b32-seed2.yaml > /tmp/p29b_rfb_s2_gpu1.yaml
S2_RUN=runs/detect/phase29b-rfb-3seed/bdd100k-p29-rfb-s2-640-e200-b32-seed2
if [ ! -f "$S2_RUN/results.csv" ]; then
  $PY scripts/run_experiment.py --config /tmp/p29b_rfb_s2_gpu1.yaml
elif [ "$(tail -1 "$S2_RUN/results.csv" | cut -d, -f1)" != "200" ]; then
  echo "  seed2 partially trained, resuming from checkpoint"
  $PY scripts/resume_experiment.py --run-dir "$S2_RUN"
else
  echo "  seed2 training already complete, skipping"
fi

echo "[3/4] seed2 terminal evaluation + per-seed bootstrap"
$PY scripts/evaluate_object_sizes.py \
  --model runs/detect/phase29b-rfb-3seed/bdd100k-p29-rfb-s2-640-e200-b32-seed2/weights/best.pt \
  --data configs/data/bdd100k.yaml \
  --output results/p29-context-screen/rfb-s2-size-val-ep200.json --device 1
$PY scripts/evaluate_timeofday_slices.py \
  --attributes data/BDD100K/label_source_2025/val_scalabel \
  --predictions results/p29-context-screen/rfb-s2-size-val-ep200.predictions.jsonl \
  --output results/p29-context-screen/rfb-s2-timeofday-ep200.json
$PY scripts/bootstrap_bdd_paired_ap.py \
  --baseline results/acfr-stage20/l0-s2-size-val-ep200-regen.predictions.jsonl \
  --candidate results/p29-context-screen/rfb-s2-size-val-ep200.predictions.jsonl \
  --output results/p29-context-screen/bootstrap-rfb-s2-vs-l0-s2-ep200.json \
  --replicates 1000 --seed 260805 --workers 8

echo "[4/4] 3-seed pooled adoption bootstrap + commit"
$PY scripts/bootstrap_multiseed_paired.py \
  --pair results/lightweight-unit-screen/l0-teacher-reeval-val-ep200.predictions.jsonl=results/p29-context-screen/rfb-size-val-ep200.predictions.jsonl \
  --pair results/acfr-stage20/l0-s1-size-val-ep200-regen.predictions.jsonl=results/p29-context-screen/rfb-s1-size-val-ep200.predictions.jsonl \
  --pair results/acfr-stage20/l0-s2-size-val-ep200-regen.predictions.jsonl=results/p29-context-screen/rfb-s2-size-val-ep200.predictions.jsonl \
  --output results/p29-context-screen/bootstrap-rfb-3seed-pooled-ep200.json \
  --replicates 1000 --seed 260805 --workers 8
$GIT add results/p29-context-screen/
$GIT -c user.name="YOLO26-SDF Project" -c user.email="yolo26-sdf@localhost" \
  commit -m "P29b RFB seed2 terminal evidence + per-seed bootstrap + 3-seed pooled adoption bootstrap (recovered chain)"
echo "DONE - pooled verdict JSON: results/p29-context-screen/bootstrap-rfb-3seed-pooled-ep200.json"
# Self-cleanup: remove the @reboot cron entry that launched this script.
(crontab -l 2>/dev/null | grep -v 'p29b_recover_after_reboot' || true) | crontab - || true
echo "reboot cron entry removed"

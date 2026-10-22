#!/bin/bash
# P33 chain: KITTI zero-shot migration eval on GPU1.
# mapped-label check -> 7 frozen BDD models infer+eval on trainval (7481) ->
# summary README -> commit. No training, no selection, testing set untouched.
set -euo pipefail
cd /home/<user>/code/github-project/<internal-project>/yolo
export PYTHONPATH=upstream:.
PY=/home/<user>/miniconda3/envs/yolo26-sdf/bin/python
GIT="git --git-dir=.project.git --work-tree=."
OUT=results/kitti-zeroshot
CFG=configs/data/kitti-trainval-bdd10.yaml
mkdir -p "$OUT"

echo "[1/3] mapped-label view check"
if [ ! -d data/KITTI/bdd10/labels/val ] || [ "$(ls data/KITTI/bdd10/labels/val | wc -l)" != "7481" ]; then
  $PY scripts/prepare_kitti_bdd10_view.py
fi
N=$(ls data/KITTI/bdd10/labels/val | wc -l)
if [ "$N" != "7481" ]; then
  echo "ERROR: mapped labels incomplete ($N != 7481) - aborting"; exit 1
fi

declare -A MODELS=(
  [p2]=runs/detect/phase1/bdd100k-yolo26n-p2-640-e200-seed0
  [acr]=runs/detect/phase18b-acr-stage20/bdd100k-yolo26n-p2-acr-r1-candidate-640-e200-stop20-b32-seed0
  [acfr]=runs/detect/phase19a-acfr-stage20/bdd100k-yolo26n-p2-acr-acfr-a1-640-e200-stop20-b32-seed0
  [rfb]=runs/detect/phase29-context-screen/bdd100k-p29-rfb-640-e200-stop20-b32-seed0
  [k1]=runs/detect/phase22-kd-arms/bdd100k-kd-k1-output-640-e200-stop20-b16-seed0
  [c-b16]=runs/detect/phase23-confound-control/bdd100k-l0-control-640-e200-stop20-b16-seed0
  [k1-rfb]=runs/detect/phase31-kd-rfb/bdd100k-p31-kd-k1-rfb-640-e200-b16-seed0
)

echo "[2/3] zero-shot infer+eval on KITTI trainval (GPU1)"
for name in p2 acr acfr rfb k1 c-b16 k1-rfb; do
  run=${MODELS[$name]}
  weights="$run/weights/best.pt"
  if [ ! -f "$weights" ]; then
    echo "ERROR: missing weights for $name: $weights"; exit 1
  fi
  if [ -f "$OUT/$name-size-val-kitti.json" ]; then
    echo "  $name already evaluated, skipping"
    continue
  fi
  echo "  evaluating $name ($run)"
  $PY scripts/evaluate_object_sizes.py \
    --model "$weights" \
    --data "$CFG" \
    --output "$OUT/$name-size-val-kitti.json" \
    --device 1
done

echo "[3/3] summary + commit"
$PY - "$OUT" <<'EOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for name in ["p2", "acr", "acfr", "rfb", "k1", "c-b16", "k1-rfb"]:
    data = json.loads((out / f"{name}-size-val-kitti.json").read_text(encoding="utf-8"))
    m = data["metrics"]
    rows.append((name, m["all"]["AP50_95"], m["small"]["AP50_95"], m["medium"]["AP50_95"], m["large"]["AP50_95"]))

lines = [
    "# P33 KITTI zero-shot migration (trainval, 7481 images, BDD-10 mapped classes)",
    "",
    "Protocol: docs/protocols/P33_KITTI_ZEROSHOT_PROTOCOL.md (frozen 2026-10-17).",
    "Frozen BDD best.pt weights, no tuning; same eval code/params as BDD 10k terminal eval",
    "(640/b32/conf0.001/iou0.7/maxDet300). Mapped classes: person/car/bike (Car->car,",
    "Pedestrian->person, Cyclist->bike; Van/Person_sitting/Tram/Misc/DontCare dropped;",
    "BDD-only classes rider/train/light/sign have no KITTI GT). All 7481 labeled trainval",
    "images evaluated; KITTI testing split (7518, unlabeled) untouched.",
    "",
    "| model | all AP50-95 | small | medium | large |",
    "|---|---|---|---|---|",
]
for name, a, s, med, lg in rows:
    lines.append(f"| {name} | {a:.6f} | {s:.6f} | {med:.6f} | {lg:.6f} |")
(out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
EOF

$GIT add "$OUT"
$GIT -c user.name="YOLO26-SDF Project" -c user.email="yolo26-sdf@localhost" \
  commit -m "P33 KITTI zero-shot migration evidence: 7 frozen BDD models, trainval 7481 (auto chain)" || echo "  nothing to commit"
echo "P33 chain done."

#!/bin/bash
# P32 chain: VisDrone zero-shot migration eval on GPU1.
# mapped-label check -> 7 frozen BDD models infer+eval on official val (548) ->
# summary README -> commit. No training, no selection, test-dev untouched.
set -euo pipefail
cd /home/<user>/code/github-project/<internal-project>/yolo
export PYTHONPATH=upstream:.
PY=/home/<user>/miniconda3/envs/yolo26-sdf/bin/python
GIT="git --git-dir=.project.git --work-tree=."
OUT=results/visdrone-zeroshot
CFG=configs/data/visdrone-val-bdd10.yaml
mkdir -p "$OUT"

echo "[1/3] mapped-label view check"
if [ ! -d data/VisDrone/bdd10/labels/val ] || [ "$(ls data/VisDrone/bdd10/labels/val | wc -l)" != "548" ]; then
  $PY scripts/prepare_visdrone_bdd10_view.py
fi
N=$(ls data/VisDrone/bdd10/labels/val | wc -l)
if [ "$N" != "548" ]; then
  echo "ERROR: mapped labels incomplete ($N != 548) - aborting"; exit 1
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

echo "[2/3] zero-shot infer+eval on VisDrone official val (GPU1)"
for name in p2 acr acfr rfb k1 c-b16 k1-rfb; do
  run=${MODELS[$name]}
  weights="$run/weights/best.pt"
  if [ ! -f "$weights" ]; then
    echo "ERROR: missing weights for $name: $weights"; exit 1
  fi
  if [ -f "$OUT/$name-size-val-visdrone.json" ]; then
    echo "  $name already evaluated, skipping"
    continue
  fi
  echo "  evaluating $name ($run)"
  $PY scripts/evaluate_object_sizes.py \
    --model "$weights" \
    --data "$CFG" \
    --output "$OUT/$name-size-val-visdrone.json" \
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
    data = json.loads((out / f"{name}-size-val-visdrone.json").read_text(encoding="utf-8"))
    m = data["metrics"]
    rows.append((name, m["all"]["AP50_95"], m["small"]["AP50_95"], m["medium"]["AP50_95"], m["large"]["AP50_95"]))

lines = [
    "# P32 VisDrone zero-shot migration (official val, 548 images, BDD-10 mapped classes)",
    "",
    "Protocol: docs/protocols/P32_VISDRONE_ZEROSHOT_PROTOCOL.md (frozen 2026-10-17).",
    "Frozen BDD best.pt weights, no tuning; same eval code/params as BDD 10k terminal eval",
    "(640/b32/conf0.001/iou0.7/maxDet300). Mapped classes: person/car/truck/bus/motor/bike",
    "(pedestrian+people merged into person; van/tricycle/awning-tricycle dropped; BDD-only",
    "classes rider/train/light/sign have no VisDrone GT). Domain gap (aerial, denser small",
    "objects) is expected and reported as-is.",
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
  commit -m "P32 VisDrone zero-shot migration evidence: 7 frozen BDD models, official val 548 (auto chain)" || echo "  nothing to commit"
echo "P32 chain done."

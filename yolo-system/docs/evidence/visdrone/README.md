# P32 VisDrone zero-shot migration (official val, 548 images, BDD-10 mapped classes)

Protocol: docs/protocols/P32_VISDRONE_ZEROSHOT_PROTOCOL.md (frozen 2026-10-17).
Frozen BDD best.pt weights, no tuning; same eval code/params as BDD 10k terminal eval
(640/b32/conf0.001/iou0.7/maxDet300). Mapped classes: person/car/truck/bus/motor/bike
(pedestrian+people merged into person; van/tricycle/awning-tricycle dropped; BDD-only
classes rider/train/light/sign have no VisDrone GT). Domain gap (aerial, denser small
objects) is expected and reported as-is.

| model | all AP50-95 | small | medium | large |
|---|---|---|---|---|
| p2 | 0.043338 | 0.026511 | 0.066857 | 0.180800 |
| acr | 0.044838 | 0.030749 | 0.071944 | 0.165656 |
| acfr | 0.044785 | 0.028664 | 0.068186 | 0.158618 |
| rfb | 0.047573 | 0.030432 | 0.072501 | 0.151880 |
| k1 | 0.047961 | 0.031868 | 0.073728 | 0.169482 |
| c-b16 | 0.044323 | 0.029508 | 0.071474 | 0.156373 |
| k1-rfb | 0.050566 | 0.032648 | 0.076180 | 0.200170 |

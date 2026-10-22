# P34 RTTS zero-shot migration (full set, 4322 foggy traffic images, BDD-10 mapped classes)

Protocol: docs/protocols/P34_RTTS_ZEROSHOT_PROTOCOL.md (frozen 2026-10-17).
Frozen BDD best.pt weights, no tuning; same eval code/params as BDD 10k terminal eval
(640/b32/conf0.001/iou0.7/maxDet300). Mapped classes: person/car/bus/motor/bike
(all five RTTS classes direct-mapped, zero drops; BDD-only classes rider/train/light/sign
have no RTTS GT). All 4322 images evaluated; bundled ImageSets not consulted (no split
selection). Fog domain: contrast degradation expected and reported as-is.

| model | all AP50-95 | small | medium | large |
|---|---|---|---|---|
| p2 | 0.241399 | 0.122354 | 0.286470 | 0.344150 |
| acr | 0.250215 | 0.129348 | 0.295957 | 0.351011 |
| acfr | 0.249657 | 0.132512 | 0.294762 | 0.345165 |
| rfb | 0.255173 | 0.137077 | 0.302373 | 0.356449 |
| k1 | 0.270843 | 0.142805 | 0.314683 | 0.375820 |
| c-b16 | 0.258352 | 0.140541 | 0.303556 | 0.350599 |
| k1-rfb | 0.271792 | 0.145275 | 0.315515 | 0.372156 |

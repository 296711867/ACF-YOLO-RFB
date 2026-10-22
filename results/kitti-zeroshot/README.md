# P33 KITTI zero-shot migration (trainval, 7481 images, BDD-10 mapped classes)

Protocol: docs/protocols/P33_KITTI_ZEROSHOT_PROTOCOL.md (frozen 2026-10-17).
Frozen BDD best.pt weights, no tuning; same eval code/params as BDD 10k terminal eval
(640/b32/conf0.001/iou0.7/maxDet300). Mapped classes: person/car/bike (Car->car,
Pedestrian->person, Cyclist->bike; Van/Person_sitting/Tram/Misc/DontCare dropped;
BDD-only classes rider/train/light/sign have no KITTI GT). All 7481 labeled trainval
images evaluated; KITTI testing split (7518, unlabeled) untouched.

| model | all AP50-95 | small | medium | large |
|---|---|---|---|---|
| p2 | 0.245863 | 0.135138 | 0.248548 | 0.341093 |
| acr | 0.244832 | 0.127348 | 0.251281 | 0.334547 |
| acfr | 0.245521 | 0.143960 | 0.251700 | 0.335301 |
| rfb | 0.245954 | 0.143698 | 0.253628 | 0.335450 |
| k1 | 0.250306 | 0.141416 | 0.255629 | 0.346326 |
| c-b16 | 0.246147 | 0.137071 | 0.252949 | 0.340004 |
| k1-rfb | 0.251543 | 0.133944 | 0.257031 | 0.347050 |

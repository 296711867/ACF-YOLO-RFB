# Layer cost audit — L0 (ACR+ACFR teacher architecture)

- Model YAML: `configs/models/yolo26n-p2-acr-acfr.yaml`
- Input: 640x640
- Total params: 2,498,756
- Total MACs (conv+linear): 5.597 G
- Ultralytics reference: params=2,498,756, GFLOPs=11.580

## Per-layer table

| layer | module | path | out C | out HxW | params | MACs (M) | MACs share |
|---:|---|---|---:|---|---:|---:|---:|
| 0 | Conv | stride 2 | 16 | 320x320 | 464 | 44.24 | 0.8% |
| 1 | Conv | P2 (protected) | 32 | 160x160 | 4,672 | 117.96 | 2.1% |
| 2 | C3k2 | P2 (protected) | 64 | 160x160 | 6,640 | 163.84 | 2.9% |
| 3 | Conv | P3 (protected) | 64 | 80x80 | 36,992 | 235.93 | 4.2% |
| 4 | C3k2 | P3 (protected) | 128 | 80x80 | 26,080 | 163.84 | 2.9% |
| 5 | Conv | P4 high-level | 128 | 40x40 | 147,712 | 235.93 | 4.2% |
| 6 | C3k2 | P4 high-level | 128 | 40x40 | 87,040 | 137.63 | 2.5% |
| 7 | Conv | P5 high-level | 256 | 20x20 | 295,424 | 117.96 | 2.1% |
| 8 | C3k2 | P5 high-level | 256 | 20x20 | 346,112 | 137.63 | 2.5% |
| 9 | SPPF | P5 high-level | 256 | 20x20 | 164,608 | 65.54 | 1.2% |
| 10 | C2PSA | P5 high-level | 256 | 20x20 | 249,728 | 98.76 | 1.8% |
| 11 | Upsample | P4 high-level | 256 | 40x40 | 0 | 0.00 | 0.0% |
| 12 | Concat | P4 high-level | 384 | 40x40 | 0 | 0.00 | 0.0% |
| 13 | C3k2 | P4 high-level | 128 | 40x40 | 119,808 | 190.05 | 3.4% |
| 14 | Upsample | P3 (protected) | 128 | 80x80 | 0 | 0.00 | 0.0% |
| 15 | Concat | P3 (protected) | 256 | 80x80 | 0 | 0.00 | 0.0% |
| 16 | C3k2 | P3 (protected) | 64 | 80x80 | 34,304 | 216.27 | 3.9% |
| 17 | Upsample | P2 (protected) | 64 | 160x160 | 0 | 0.00 | 0.0% |
| 18 | ACFRConcat | P2 (protected) | 128 | 160x160 | 8,132 | 201.52 | 3.6% |
| 19 | C3k2 | P2 (protected) | 64 | 160x160 | 26,112 | 655.36 | 11.7% |
| 20 | Conv | P3 (protected) | 32 | 80x80 | 18,496 | 117.96 | 2.1% |
| 21 | Concat | P3 (protected) | 96 | 80x80 | 0 | 0.00 | 0.0% |
| 22 | C3k2 | P3 (protected) | 64 | 80x80 | 24,064 | 150.73 | 2.7% |
| 23 | Conv | P4 high-level | 64 | 40x40 | 36,992 | 58.98 | 1.1% |
| 24 | Concat | P4 high-level | 192 | 40x40 | 0 | 0.00 | 0.0% |
| 25 | C3k2 | P4 high-level | 128 | 40x40 | 95,232 | 150.73 | 2.7% |
| 26 | Conv | P5 high-level | 128 | 20x20 | 147,712 | 58.98 | 1.1% |
| 27 | Concat | P5 high-level | 384 | 20x20 | 0 | 0.00 | 0.0% |
| 28 | C3k2 | P5 high-level | 192 | 20x20 | 279,744 | 110.94 | 2.0% |
| 29 | Detect | Detect head (immutable) | - | - | 342,688 | 2165.98 | 38.7% |

## Unit-swap projection on high-level plain Conv layers

| layer | path | C1->C2 | out HxW | orig params | DWConv params | GhostConv params | orig MACs (M) | DWConv MACs (M) | GhostConv MACs (M) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 5 | P4 high-level | 128->128 | 40x40 | 147,712 | 1,408 | 75,584 | 235.93 | 1.84 | 120.52 |
| 7 | P5 high-level | 128->256 | 20x20 | 295,424 | 2,816 | 151,168 | 117.96 | 0.92 | 60.26 |
| 23 | P4 high-level | 64->64 | 40x40 | 36,992 | 704 | 19,360 | 58.98 | 0.92 | 30.77 |
| 26 | P5 high-level | 128->128 | 20x20 | 147,712 | 1,408 | 75,584 | 58.98 | 0.46 | 30.13 |

| unit | projected params | projected GMACs | params drop | GMACs drop | Stage20 gate (params >=12%, GMACs >=8%) |
|---|---:|---:|---:|---:|---|
| DWConv | 1,877,252 | 5.129 | -24.9% | -8.4% | PASS |
| GhostConv | 2,192,612 | 5.367 | -12.3% | -4.1% | FAIL |

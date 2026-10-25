# DXF comparison

ref: `dataset/clear_sum_dataset/vse_dxf/клапан.dxf`  
gen: `docs/architecture/experiments/klapan_recheck/B_after_fixes.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.752 (P 0.717 / R 0.789) |
| geometry F1 @0.25 mm | 0.931 (P 0.892 / R 0.973) |
| geometry F1 @0.5 mm | 0.937 (P 0.900 / R 0.977) |
| geometry F1 @1.0 mm | 0.943 (P 0.909 / R 0.979) |
| primitive recall / precision | 0.956 / 0.702 |
| layer class accuracy | 0.554 |
| dimension recall | 0.786 (28 ref, 25 gen) |
| dimension recall by value (notation ignored) | 0.857 |
| dimension type accuracy | 0.750 |
| text exact / mean CER | 0.348 / 0.480 |
| text exact / mean CER (matched by line centre) | 0.348 / 0.468 |
| hatch area ref / gen | 2756.7 / 1663.4 |

Missing dimensions: ⌀22, R7, 4, R2, ⌀32, 1,5x45°

Extra dimensions: 1,54, 4+1, 32

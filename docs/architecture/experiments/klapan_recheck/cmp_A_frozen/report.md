# DXF comparison

ref: `dataset/clear_sum_dataset/vse_dxf/клапан.dxf`  
gen: `runs/klapan_png2dxf/out/frozen_blind/klapan_variant_A.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.952 (P 0.946 / R 0.958) |
| geometry F1 @0.25 mm | 0.980 (P 0.974 / R 0.987) |
| geometry F1 @0.5 mm | 0.985 (P 0.979 / R 0.991) |
| geometry F1 @1.0 mm | 0.987 (P 0.981 / R 0.993) |
| primitive recall / precision | 0.947 / 0.931 |
| layer class accuracy | 0.795 |
| dimension recall | 0.964 (28 ref, 28 gen) |
| dimension recall by value (notation ignored) | 1.000 |
| dimension type accuracy | 0.929 |
| text exact / mean CER | 0.696 / 0.282 |
| text exact / mean CER (matched by line centre) | 0.841 / 0.124 |
| hatch area ref / gen | 2756.7 / 2694.9 |

Missing dimensions: 4

Extra dimensions: 4+1^

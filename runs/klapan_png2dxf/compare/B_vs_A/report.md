# DXF comparison

ref: `runs/klapan_png2dxf/out/frozen_blind/klapan_variant_A.dxf`  
gen: `runs/klapan_png2dxf/out/frozen_blind/klapan_variant_B.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.781 (P 0.740 / R 0.827) |
| geometry F1 @0.25 mm | 0.919 (P 0.874 / R 0.970) |
| geometry F1 @0.5 mm | 0.930 (P 0.886 / R 0.980) |
| geometry F1 @1.0 mm | 0.939 (P 0.900 / R 0.983) |
| primitive recall / precision | 0.936 / 0.681 |
| layer class accuracy | 0.568 |
| dimension recall | 0.821 (28 ref, 26 gen) |
| dimension type accuracy | 0.821 |
| text exact / mean CER | 0.312 / 0.521 |
| text exact / mean CER (matched by line centre) | 0.406 / 0.413 |
| hatch area ref / gen | 2694.9 / 1188.6 |

Missing dimensions: ⌀22, R7, 4+1^, ⌀32, 1,5x45°

Extra dimensions: 1,5, 4+1, 32

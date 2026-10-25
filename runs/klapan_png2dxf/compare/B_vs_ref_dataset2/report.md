# DXF comparison

ref: `dataset/dataset2/клапан.dxf`  
gen: `runs/klapan_png2dxf/out/frozen_blind/klapan_variant_B.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.312 (P 0.295 / R 0.331) |
| geometry F1 @0.25 mm | 0.661 (P 0.621 / R 0.707) |
| geometry F1 @0.5 mm | 0.929 (P 0.879 / R 0.985) |
| geometry F1 @1.0 mm | 0.937 (P 0.892 / R 0.987) |
| primitive recall / precision | 0.947 / 0.696 |
| layer class accuracy | 0.561 |
| dimension recall | 0.786 (28 ref, 26 gen) |
| dimension type accuracy | 0.750 |
| text exact / mean CER | 0.348 / 0.480 |
| hatch area ref / gen | 2756.7 / 1188.6 |

Missing dimensions: ⌀22, R7, М12х1,5-Н7, 4, ⌀32, 1,5x45°

Extra dimensions: 1,5, 4+1, M12x1,5-H7, 32

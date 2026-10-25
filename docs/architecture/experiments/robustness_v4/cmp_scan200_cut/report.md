# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\клапан.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness_v4\scan200_cut.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.031 (P 0.031 / R 0.031) |
| geometry F1 @0.25 mm | 0.093 (P 0.094 / R 0.092) |
| geometry F1 @0.5 mm | 0.190 (P 0.190 / R 0.191) |
| geometry F1 @1.0 mm | 0.315 (P 0.310 / R 0.320) |
| primitive recall / precision | 0.073 / 0.137 |
| layer class accuracy | 0.621 |
| dimension recall | 0.143 (28 ref, 7 gen) |
| dimension recall by value (notation ignored) | 0.179 |
| dimension type accuracy | 0.143 |
| text exact / mean CER | 0.145 / 0.692 |
| text exact / mean CER (matched by line centre) | 0.145 / 0.714 |
| hatch area ref / gen | 2756.7 / 1129.1 |

Missing dimensions: ⌀135, ⌀160, ⌀150±0,5, ⌀22, ⌀32, ⌀28, ⌀42, 10, 60°, 4, 5,5, 20, 25, 46, 14, М12х1,5-Н7, 1,6x45°, 4, 1,5, R2, 30, 15, ⌀32, 1,5x45°

Extra dimensions: 13,3, .14, (n.5)

# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\клапан.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness\scan200_border.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.061 (P 0.157 / R 0.038) |
| geometry F1 @0.25 mm | 0.269 (P 0.680 / R 0.167) |
| geometry F1 @0.5 mm | 0.401 (P 0.959 / R 0.254) |
| geometry F1 @1.0 mm | 0.414 (P 0.961 / R 0.264) |
| primitive recall / precision | 0.015 / 0.897 |
| layer class accuracy | 0.493 |
| dimension recall | 0.214 (28 ref, 16 gen) |
| dimension recall by value (notation ignored) | 0.286 |
| dimension type accuracy | 0.179 |
| text exact / mean CER | 0.188 / 0.521 |
| text exact / mean CER (matched by line centre) | 0.203 / 0.508 |
| hatch area ref / gen | 2756.7 / 239.7 |

Missing dimensions: ⌀160, ⌀150±0,5, ⌀22, ⌀32, ⌀42, 10, 60°, 20, 25, 46, 14, М12х1,5-Н7, 1,6x45°, 4, R3, R2, 1,5, R2, 30, 15, 1,5x45°, R3

Extra dimensions: 1,52, 3,32, 7,37, 7,73, 11,49, 12,55, 15., 150±0,5, 19,71, 24,62

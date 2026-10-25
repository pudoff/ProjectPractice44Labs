# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\клапан.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness_v4\scan200_border.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.172 (P 0.163 / R 0.181) |
| geometry F1 @0.25 mm | 0.570 (P 0.536 / R 0.609) |
| geometry F1 @0.5 mm | 0.792 (P 0.737 / R 0.856) |
| geometry F1 @1.0 mm | 0.827 (P 0.763 / R 0.904) |
| primitive recall / precision | 0.524 / 0.636 |
| layer class accuracy | 0.548 |
| dimension recall | 0.214 (28 ref, 9 gen) |
| dimension recall by value (notation ignored) | 0.214 |
| dimension type accuracy | 0.179 |
| text exact / mean CER | 0.188 / 0.521 |
| text exact / mean CER (matched by line centre) | 0.203 / 0.508 |
| hatch area ref / gen | 2756.7 / 1038.8 |

Missing dimensions: ⌀135, ⌀160, ⌀150±0,5, ⌀22, ⌀32, ⌀28, ⌀42, 10, 60°, 4, 5,5, 20, 25, 14, М12х1,5-Н7, R2, 1,5, R2, 15, ⌀32, 1,5x45°, R3

Extra dimensions: 43°, 25°, 35°

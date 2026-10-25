# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\клапан.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness\scan200_cut.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.009 (P 0.029 / R 0.005) |
| geometry F1 @0.25 mm | 0.018 (P 0.057 / R 0.011) |
| geometry F1 @0.5 mm | 0.085 (P 0.265 / R 0.051) |
| geometry F1 @1.0 mm | 0.157 (P 0.476 / R 0.094) |
| primitive recall / precision | 0.000 / 0.202 |
| layer class accuracy | 0.525 |
| dimension recall | 0.036 (28 ref, 12 gen) |
| dimension recall by value (notation ignored) | 0.071 |
| dimension type accuracy | 0.036 |
| text exact / mean CER | 0.145 / 0.692 |
| text exact / mean CER (matched by line centre) | 0.145 / 0.714 |
| hatch area ref / gen | 2756.7 / 263.4 |

Missing dimensions: ⌀135, ⌀160, ⌀150±0,5, ⌀22, ⌀32, ⌀28, ⌀42, 10, 60°, 5,5, 20, 25, 46, R7, 14, М12х1,5-Н7, 1,6x45°, 4, R3, R2, 1,5, R2, 30, 15, ⌀32, 1,5x45°, R3

Extra dimensions: 5,55, .14, 12,79, 15,46, 21,11, 22,48, 28,9, 29, 43,46, 139,61, 165,5

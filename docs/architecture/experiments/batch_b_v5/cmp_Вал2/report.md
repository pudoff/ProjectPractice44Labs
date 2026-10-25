# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Вал2.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\Вал2.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.854 (P 0.834 / R 0.875) |
| geometry F1 @0.25 mm | 0.886 (P 0.865 / R 0.908) |
| geometry F1 @0.5 mm | 0.894 (P 0.874 / R 0.916) |
| geometry F1 @1.0 mm | 0.909 (P 0.891 / R 0.927) |
| primitive recall / precision | 0.845 / 0.663 |
| layer class accuracy | 0.431 |
| dimension recall | 0.607 (28 ref, 23 gen) |
| dimension recall by value (notation ignored) | 0.643 |
| dimension type accuracy | 0.607 |
| text exact / mean CER | 0.250 / 0.602 |
| text exact / mean CER (matched by line centre) | 0.250 / 0.611 |
| hatch area ref / gen | 2271.3 / 1673.1 |

Missing dimensions: 44, 10, R1, 10, 1x45, 28, 26, 58, R4, 8H9, 8js9

Extra dimensions: 0,8, T9, 1,6, 26,98, 28-0.014, 120°

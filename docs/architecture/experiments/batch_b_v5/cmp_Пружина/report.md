# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Пружина.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\Пружина.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.883 (P 0.872 / R 0.894) |
| geometry F1 @0.25 mm | 0.964 (P 0.957 / R 0.970) |
| geometry F1 @0.5 mm | 0.968 (P 0.961 / R 0.974) |
| geometry F1 @1.0 mm | 0.971 (P 0.966 / R 0.976) |
| primitive recall / precision | 0.874 / 0.705 |
| layer class accuracy | 0.530 |
| dimension recall | 0.000 (8 ref, 7 gen) |
| dimension recall by value (notation ignored) | 0.125 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.257 / 0.521 |
| text exact / mean CER (matched by line centre) | 0.257 / 0.478 |
| hatch area ref / gen | 24.6 / 0 |

Missing dimensions: L, L, L, L, ⌀14, ⌀12, 6, ⌀4*

Extra dimensions: 7,46, L2, 19,96, ⌀20,02, 12, 35, 50,04

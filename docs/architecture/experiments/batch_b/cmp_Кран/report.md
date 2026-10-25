# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Кран.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Кран.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.109 (P 0.101 / R 0.118) |
| geometry F1 @0.25 mm | 0.143 (P 0.134 / R 0.154) |
| geometry F1 @0.5 mm | 0.217 (P 0.202 / R 0.233) |
| geometry F1 @1.0 mm | 0.309 (P 0.291 / R 0.329) |
| primitive recall / precision | 0.024 / 0.030 |
| layer class accuracy | 0.667 |
| dimension recall | 0.062 (16 ref, 14 gen) |
| dimension recall by value (notation ignored) | 0.125 |
| dimension type accuracy | 0.062 |
| text exact / mean CER | 0.000 / 0.983 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.983 |
| hatch area ref / gen | 1852.8 / 1163.8 |

Missing dimensions: 15, 45, 80, 54, 46, 50, 24, 32, 32, 7, 24, 94, 47, 11, 45

Extra dimensions: 11,68, 15,33, ⌀24,81, ⌀25,01, ⌀33,07, ⌀33,98, 46,28, ⌀47,82, 49,92, ⌀57,35, 82,42, ⌀99,81, 50°

# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Сфера1.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Сфера1.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.697 (P 0.719 / R 0.677) |
| geometry F1 @0.25 mm | 0.977 (P 0.966 / R 0.987) |
| geometry F1 @0.5 mm | 0.980 (P 0.968 / R 0.993) |
| geometry F1 @1.0 mm | 0.984 (P 0.973 / R 0.995) |
| primitive recall / precision | 0.936 / 0.854 |
| layer class accuracy | 0.367 |
| dimension recall | 0.750 (8 ref, 7 gen) |
| dimension recall by value (notation ignored) | 0.875 |
| dimension type accuracy | 0.750 |
| text exact / mean CER | 0.389 / 0.377 |
| text exact / mean CER (matched by line centre) | 0.389 / 0.322 |
| hatch area ref / gen | 0 / 0 |

Missing dimensions: ⌀110, ⌀76

Extra dimensions: 110

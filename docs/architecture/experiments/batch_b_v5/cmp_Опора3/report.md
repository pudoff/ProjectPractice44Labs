# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Опора3.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\Опора3.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.902 (P 0.872 / R 0.934) |
| geometry F1 @0.25 mm | 0.935 (P 0.902 / R 0.970) |
| geometry F1 @0.5 mm | 0.946 (P 0.913 / R 0.980) |
| geometry F1 @1.0 mm | 0.956 (P 0.925 / R 0.988) |
| primitive recall / precision | 0.935 / 0.813 |
| layer class accuracy | 0.704 |
| dimension recall | 0.526 (19 ref, 13 gen) |
| dimension recall by value (notation ignored) | 0.632 |
| dimension type accuracy | 0.526 |
| text exact / mean CER | 0.500 / 0.439 |
| text exact / mean CER (matched by line centre) | 0.500 / 0.439 |
| hatch area ref / gen | 2214.6 / 2091.6 |

Missing dimensions: 32, 4отв. 14, 62, 102, 134, 6, 96, 30°, 30°

Extra dimensions: ⌀6, 13,9, ⌀96

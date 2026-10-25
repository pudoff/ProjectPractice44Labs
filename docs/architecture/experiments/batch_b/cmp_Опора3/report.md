# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Опора3.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Опора3.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.105 (P 0.098 / R 0.113) |
| geometry F1 @0.25 mm | 0.134 (P 0.125 / R 0.145) |
| geometry F1 @0.5 mm | 0.211 (P 0.196 / R 0.228) |
| geometry F1 @1.0 mm | 0.318 (P 0.295 / R 0.344) |
| primitive recall / precision | 0.053 / 0.037 |
| layer class accuracy | 0.526 |
| dimension recall | 0.053 (19 ref, 13 gen) |
| dimension recall by value (notation ignored) | 0.105 |
| dimension type accuracy | 0.053 |
| text exact / mean CER | 0.038 / 0.942 |
| text exact / mean CER (matched by line centre) | 0.038 / 0.942 |
| hatch area ref / gen | 2214.6 / 1237.3 |

Missing dimensions: 22, 36, 43, 57, 80, 32, 10, 72, 34, 15, 4отв. 14, 62, 102, 134, 6, 96, 30°, 30°

Extra dimensions: ⌀6, 10,62, 14,45, 15,92, 22,74, 36,1, 37,2, 44,44, 58,83, 76,44, 82,42, ⌀99,14

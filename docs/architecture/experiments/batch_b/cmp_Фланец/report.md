# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Фланец.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Фланец.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.618 (P 0.611 / R 0.624) |
| geometry F1 @0.25 mm | 0.929 (P 0.907 / R 0.953) |
| geometry F1 @0.5 mm | 0.939 (P 0.917 / R 0.962) |
| geometry F1 @1.0 mm | 0.951 (P 0.931 / R 0.973) |
| primitive recall / precision | 0.864 / 0.750 |
| layer class accuracy | 0.560 |
| dimension recall | 0.556 (18 ref, 14 gen) |
| dimension recall by value (notation ignored) | 0.778 |
| dimension type accuracy | 0.500 |
| text exact / mean CER | 0.423 / 0.461 |
| text exact / mean CER (matched by line centre) | 0.423 / 0.461 |
| hatch area ref / gen | 2899.0 / 1904.6 |

Missing dimensions: 12, 110, 34,5, 85, 45, 60, 65, 50

Extra dimensions: ⌀65, ⌀110, 60°, 45°

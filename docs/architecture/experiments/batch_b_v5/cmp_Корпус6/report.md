# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Корпус6.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\Корпус6.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.932 (P 0.927 / R 0.937) |
| geometry F1 @0.25 mm | 0.955 (P 0.949 / R 0.962) |
| geometry F1 @0.5 mm | 0.966 (P 0.959 / R 0.974) |
| geometry F1 @1.0 mm | 0.976 (P 0.969 / R 0.984) |
| primitive recall / precision | 0.937 / 0.737 |
| layer class accuracy | 0.639 |
| dimension recall | 0.850 (20 ref, 20 gen) |
| dimension recall by value (notation ignored) | 1.000 |
| dimension type accuracy | 0.850 |
| text exact / mean CER | 0.600 / 0.312 |
| text exact / mean CER (matched by line centre) | 0.600 / 0.312 |
| hatch area ref / gen | 1581.5 / 1355.5 |

Missing dimensions: 7, 50, 20

Extra dimensions: ⌀7, ⌀20, ⌀50

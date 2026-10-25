# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Корпус6.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Корпус6.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.116 (P 0.113 / R 0.119) |
| geometry F1 @0.25 mm | 0.150 (P 0.145 / R 0.154) |
| geometry F1 @0.5 mm | 0.231 (P 0.224 / R 0.238) |
| geometry F1 @1.0 mm | 0.322 (P 0.312 / R 0.333) |
| primitive recall / precision | 0.044 / 0.047 |
| layer class accuracy | 0.429 |
| dimension recall | 0.150 (20 ref, 20 gen) |
| dimension recall by value (notation ignored) | 0.200 |
| dimension type accuracy | 0.150 |
| text exact / mean CER | 0.000 / 0.975 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.975 |
| hatch area ref / gen | 1581.5 / 1475.6 |

Missing dimensions: 7, 40, 50, 80, 60, 20, 10, 50, 55, 35, 45, 100, 35, 55, 18, 30, 26

Extra dimensions: ⌀7, 10,63, 18,58, ⌀20,66, 27,6, 30,97, 35,92, 37,15, 42,46, 46,51, ⌀51,64, 53,09, 56,63, 58,39, 63,7, 84,84, 103,11

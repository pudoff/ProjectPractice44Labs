# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\фланец3.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\фланец3.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.703 (P 0.705 / R 0.701) |
| geometry F1 @0.25 mm | 0.909 (P 0.914 / R 0.905) |
| geometry F1 @0.5 mm | 0.914 (P 0.918 / R 0.909) |
| geometry F1 @1.0 mm | 0.917 (P 0.923 / R 0.912) |
| primitive recall / precision | 0.610 / 0.622 |
| layer class accuracy | 0.089 |
| dimension recall | 0.636 (22 ref, 18 gen) |
| dimension recall by value (notation ignored) | 0.682 |
| dimension type accuracy | 0.636 |
| text exact / mean CER | 0.136 / 0.665 |
| text exact / mean CER (matched by line centre) | 0.136 / 0.658 |
| hatch area ref / gen | 979.3 / 903.5 |

Missing dimensions: 2, 1, 45°, 34, 6x170=1020, 4x130=520, M8-7H, (1,6x45°)

Extra dimensions: 0,97, (1,6x45), 20 omb., 42°

# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\корпус клапана.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\корпус клапана.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.128 (P 0.116 / R 0.142) |
| geometry F1 @0.25 mm | 0.180 (P 0.164 / R 0.200) |
| geometry F1 @0.5 mm | 0.269 (P 0.245 / R 0.298) |
| geometry F1 @1.0 mm | 0.383 (P 0.350 / R 0.422) |
| primitive recall / precision | 0.048 / 0.032 |
| layer class accuracy | 0.556 |
| dimension recall | 0.033 (30 ref, 24 gen) |
| dimension recall by value (notation ignored) | 0.033 |
| dimension type accuracy | 0.033 |
| text exact / mean CER | 0.000 / 0.958 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.978 |
| hatch area ref / gen | 1970.3 / 1058.4 |

Missing dimensions: М45х2, ⌀42, 12,5, ⌀11,5, ⌀17,5, ⌀31, ⌀34, 2отв.⌀7, 33,5, R10, R5, R12,5, 23, 18, 7, 19,5, 30, 51, 75, ∅48, 48, 20, 27, 56, 43,5, R2,5, 32, R1, R2

Extra dimensions: 7, 7,43, 12,23, 18,51, 19,1, 19,92, 21,22, 24,44, 28,65, 30,57, 32,64, 33,23, 34,51, 36,1, 44,59, 44,84, 47,77, 49,15, 52,26, 57,74, 79,62, R5, R10

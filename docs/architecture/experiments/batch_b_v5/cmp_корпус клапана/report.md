# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\корпус клапана.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\корпус клапана.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.808 (P 0.767 / R 0.853) |
| geometry F1 @0.25 mm | 0.907 (P 0.863 / R 0.957) |
| geometry F1 @0.5 mm | 0.922 (P 0.884 / R 0.963) |
| geometry F1 @1.0 mm | 0.938 (P 0.906 / R 0.972) |
| primitive recall / precision | 0.904 / 0.685 |
| layer class accuracy | 0.565 |
| dimension recall | 0.667 (30 ref, 23 gen) |
| dimension recall by value (notation ignored) | 0.700 |
| dimension type accuracy | 0.667 |
| text exact / mean CER | 0.565 / 0.282 |
| text exact / mean CER (matched by line centre) | 0.565 / 0.271 |
| hatch area ref / gen | 1970.3 / 1211.0 |

Missing dimensions: 12,5, ⌀11,5, ⌀17,5, 2отв.⌀7, R10, R12,5, ∅48, R2,5, R1, R2

Extra dimensions: 6,98, ⌀115, 17,5

# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Вал2.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b\Вал2.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.102 (P 0.094 / R 0.113) |
| geometry F1 @0.25 mm | 0.137 (P 0.126 / R 0.150) |
| geometry F1 @0.5 mm | 0.204 (P 0.187 / R 0.223) |
| geometry F1 @1.0 mm | 0.281 (P 0.259 / R 0.307) |
| primitive recall / precision | 0.026 / 0.021 |
| layer class accuracy | 0.750 |
| dimension recall | 0.179 (28 ref, 24 gen) |
| dimension recall by value (notation ignored) | 0.179 |
| dimension type accuracy | 0.179 |
| text exact / mean CER | 0.029 / 0.963 |
| text exact / mean CER (matched by line centre) | 0.029 / 0.940 |
| hatch area ref / gen | 2271.3 / 33.2 |

Missing dimensions: 70, 44, 10, 7, 50k6, 5, R1, 10, 1x45, 1x45, 28, 26, 32, 235, 32h8, 40k6, 36±0,2, R4, 8H9, 35, 3паза 12js9, 8js9, 24

Extra dimensions: 1x45, 0,8, 5, 7,27, ⌀8,48, T9, 12,76, 24,77, 28,9-0.014, 33,07, 33,82, 36,17, 38,04, 41,32, 51,65, 74,32, 249,17, R1, R6

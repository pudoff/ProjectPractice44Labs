# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_003.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness\synthetic_003.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.017 (P 0.019 / R 0.016) |
| geometry F1 @0.25 mm | 0.067 (P 0.072 / R 0.062) |
| geometry F1 @0.5 mm | 0.153 (P 0.163 / R 0.144) |
| geometry F1 @1.0 mm | 0.344 (P 0.361 / R 0.329) |
| primitive recall / precision | 0.064 / 0.140 |
| layer class accuracy | 0.220 |
| dimension recall | 0.000 (0 ref, 1 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.032 / 0.927 |
| text exact / mean CER (matched by line centre) | 0.032 / 0.927 |
| hatch area ref / gen | 0 / 11.2 |

Missing dimensions: 

Extra dimensions: 24,29

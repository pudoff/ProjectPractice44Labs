# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_001.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness\synthetic_001.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.074 (P 0.084 / R 0.066) |
| geometry F1 @0.25 mm | 0.280 (P 0.315 / R 0.252) |
| geometry F1 @0.5 mm | 0.553 (P 0.615 / R 0.503) |
| geometry F1 @1.0 mm | 0.882 (P 0.962 / R 0.814) |
| primitive recall / precision | 0.278 / 0.613 |
| layer class accuracy | 0.247 |
| dimension recall | 0.000 (0 ref, 0 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.000 / 0.950 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.950 |
| hatch area ref / gen | 0 / 21.4 |

Missing dimensions: 

Extra dimensions: 

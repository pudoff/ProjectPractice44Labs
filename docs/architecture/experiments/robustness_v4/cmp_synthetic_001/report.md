# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_001.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness_v4\synthetic_001.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.093 (P 0.097 / R 0.089) |
| geometry F1 @0.25 mm | 0.358 (P 0.369 / R 0.347) |
| geometry F1 @0.5 mm | 0.648 (P 0.662 / R 0.634) |
| geometry F1 @1.0 mm | 0.955 (P 0.966 / R 0.944) |
| primitive recall / precision | 0.533 / 0.629 |
| layer class accuracy | 0.254 |
| dimension recall | 0.000 (0 ref, 0 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.000 / 0.950 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.950 |
| hatch area ref / gen | 0 / 23.2 |

Missing dimensions: 

Extra dimensions: 

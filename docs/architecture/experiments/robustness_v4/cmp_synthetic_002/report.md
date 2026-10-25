# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_002.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness_v4\synthetic_002.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.028 (P 0.029 / R 0.027) |
| geometry F1 @0.25 mm | 0.083 (P 0.085 / R 0.080) |
| geometry F1 @0.5 mm | 0.171 (P 0.176 / R 0.167) |
| geometry F1 @1.0 mm | 0.376 (P 0.381 / R 0.371) |
| primitive recall / precision | 0.000 / 0.082 |
| layer class accuracy | 0.424 |
| dimension recall | 0.000 (0 ref, 0 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.000 / 0.963 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.963 |
| hatch area ref / gen | 0 / 19.5 |

Missing dimensions: 

Extra dimensions: 

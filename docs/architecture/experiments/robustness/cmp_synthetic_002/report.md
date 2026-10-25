# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_002.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness\synthetic_002.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.019 (P 0.020 / R 0.017) |
| geometry F1 @0.25 mm | 0.074 (P 0.080 / R 0.069) |
| geometry F1 @0.5 mm | 0.156 (P 0.168 / R 0.146) |
| geometry F1 @1.0 mm | 0.367 (P 0.385 / R 0.351) |
| primitive recall / precision | 0.000 / 0.090 |
| layer class accuracy | 0.491 |
| dimension recall | 0.000 (0 ref, 0 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.000 / 0.963 |
| text exact / mean CER (matched by line centre) | 0.000 / 0.963 |
| hatch area ref / gen | 0 / 11.4 |

Missing dimensions: 

Extra dimensions: 

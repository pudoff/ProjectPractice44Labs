# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\data\dxf\drawing_003.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\robustness_v4\synthetic_003.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.012 (P 0.012 / R 0.011) |
| geometry F1 @0.25 mm | 0.071 (P 0.074 / R 0.070) |
| geometry F1 @0.5 mm | 0.172 (P 0.176 / R 0.167) |
| geometry F1 @1.0 mm | 0.368 (P 0.374 / R 0.363) |
| primitive recall / precision | 0.064 / 0.125 |
| layer class accuracy | 0.214 |
| dimension recall | 0.000 (0 ref, 0 gen) |
| dimension recall by value (notation ignored) | 0.000 |
| dimension type accuracy | 0.000 |
| text exact / mean CER | 0.032 / 0.927 |
| text exact / mean CER (matched by line centre) | 0.032 / 0.927 |
| hatch area ref / gen | 0 / 16.0 |

Missing dimensions: 

Extra dimensions: 

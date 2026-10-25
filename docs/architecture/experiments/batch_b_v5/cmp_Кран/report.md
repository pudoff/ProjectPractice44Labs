# DXF comparison

ref: `D:\pp\ProjectPractice44Labs\dataset\clear_sum_dataset\vse_dxf\Кран.dxf`  
gen: `D:\pp\ProjectPractice44Labs\docs\architecture\experiments\batch_b_v5\Кран.dxf`  
tolerance: 0.5 mm

| metric | value |
|---|---|
| geometry F1 @0.1 mm | 0.904 (P 0.874 / R 0.935) |
| geometry F1 @0.25 mm | 0.946 (P 0.915 / R 0.978) |
| geometry F1 @0.5 mm | 0.955 (P 0.927 / R 0.984) |
| geometry F1 @1.0 mm | 0.968 (P 0.943 / R 0.993) |
| primitive recall / precision | 0.964 / 0.722 |
| layer class accuracy | 0.726 |
| dimension recall | 0.375 (16 ref, 14 gen) |
| dimension recall by value (notation ignored) | 0.875 |
| dimension type accuracy | 0.375 |
| text exact / mean CER | 0.367 / 0.509 |
| text exact / mean CER (matched by line centre) | 0.367 / 0.509 |
| hatch area ref / gen | 1852.8 / 1427.6 |

Missing dimensions: 45, 54, 46, 50, 24, 32, 32, 7, 24, 94

Extra dimensions: ⌀24, ⌀24, ⌀32, ⌀32, ⌀45, ⌀54, ⌀94, 50°

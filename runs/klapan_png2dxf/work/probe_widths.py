import cv2, numpy as np
from meas import ink, to_px, SX
d = cv2.distanceTransform(ink(), cv2.DIST_L2, 5)
def w(x, y):
    px, py = to_px(x, y); r = 8
    win = d[int(py)-r:int(py)+r+1, int(px)-r:int(px)+r+1]
    return 2*win.max()/SX
pts = {"left face x0": (54.30, 152), "rim face": (100.28, 120), "tangent line 95.8": (95.81, 120),
 "thread major": (75, 166.40), "drill minor": (75, 165.29), "thread end": (66.29, 158), "axis dash": (64.5, 160.41),
 "hatch": None, "dim line 46": (80, 99.12), "ext line 160": (120, 240.40), "frame inner": (20, 150), "sheet edge": (0.0, 150),
 "titleblock thick": (230, 30), "title thin row": (240, 50), "bore 135": (110, 227.93), "callout circle": (94.06, 232.87),
 "detail contour": (185.9, 250), "detail break": (166.5, 239)}
with open("probe_widths.txt", "w", encoding="utf-8") as f:
    for k, v in pts.items():
        if v: f.write(f"{k:20s} {w(*v):.3f}\n")

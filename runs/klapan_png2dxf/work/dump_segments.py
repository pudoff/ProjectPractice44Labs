"""Vectorize the sheet once (pipeline_b/vectorize.py) and cache primitives for measurements."""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline_b"))
from meas import CAL, ink  # noqa: E402
from vectorize import vectorize  # noqa: E402

t = time.time()
prims, _ = vectorize(ink(), CAL)
pickle.dump(prims, open("prims.pkl", "wb"))
print(len(prims), "primitives", round(time.time() - t, 1), "s")


def query(x0, y0, x1, y1, min_len=0.3):
    rows = []
    for p in prims:
        if p[0] == "line":
            a, b = p[1], p[2]
            if all(x0 <= v[0] <= x1 and y0 <= v[1] <= y1 for v in (a, b)):
                L = ((a - b) ** 2).sum() ** 0.5
                if L >= min_len:
                    rows.append(f"L ({a[0]:.2f},{a[1]:.2f})-({b[0]:.2f},{b[1]:.2f}) len={L:.2f}")
        else:
            c, r = p[1], p[2]
            if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
                rows.append(f"A c=({c[0]:.2f},{c[1]:.2f}) r={r:.2f} a={p[3]:.0f}..{p[4]:.0f} sweep={p[5]:.0f}")
    return rows

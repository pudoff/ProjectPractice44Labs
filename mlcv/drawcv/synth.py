"""Synthetic control sheet with exactly known geometry: PNG / JPEG / PDF renders, a simulated scan (rotation,
blur, noise, JPEG), the reference DXF and truth.json with the expected contours and defects.

    python -m drawcv.synth OUT_DIR [--dpi 300]

Content (A4 portrait, GOST frame and title block):
  plate      - 80 x 50 outline with an R8 rounded corner, holes 16 and 10, centre lines (dash-dot);
  section    - closed 40 x 20 rectangle hatched at 45 deg (the supported simple hatch case);
  broken     - 50 x 30 rectangle with a 1.5 mm break in its top edge (must be reported as a gap);
  bowtie     - closed four-segment outline crossing itself (must be reported as self-intersection);
  texts      - labels next to and between contours, 2.5 / 3.5 / 5 mm high, a dimension with arrows."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SHEET = (210.0, 297.0)
THICK, THIN = 0.7, 0.25

PLATE = {"x0": 40.0, "y0": 190.0, "w": 80.0, "h": 50.0, "r": 8.0}
HOLES = [((60.0, 215.0), 8.0), ((100.0, 215.0), 5.0)]
SECTION = (130.0, 200.0, 40.0, 20.0)
BROKEN = {"x0": 40.0, "y0": 110.0, "w": 50.0, "h": 30.0, "gap": (63.0, 64.5)}
BOWTIE = [(120.0, 110.0), (170.0, 140.0), (170.0, 110.0), (120.0, 140.0)]
TEXTS = [("PLATE 80x50", (40.0, 250.0), 5.0), ("R8", (110.0, 243.0), 3.5), ("A-A", (140.0, 225.0), 5.0),
         ("BROKEN", (45.0, 145.0), 3.5), ("0,5", (95.0, 150.0), 2.5), ("SELF CROSS", (125.0, 145.0), 3.5)]


def plate_segments():
    """Plate outline as lines and the rounded corner arc (top right), counter-clockwise."""
    x0, y0, w, h, r = PLATE["x0"], PLATE["y0"], PLATE["w"], PLATE["h"], PLATE["r"]
    x1, y1 = x0 + w, y0 + h
    return [("line", (x0, y0), (x1, y0)), ("line", (x1, y0), (x1, y1 - r)),
            ("arc", (x1 - r, y1 - r), r, 0.0, 90.0), ("line", (x1 - r, y1), (x0, y1)), ("line", (x0, y1), (x0, y0))]


def broken_segments():
    x0, y0, w, h = BROKEN["x0"], BROKEN["y0"], BROKEN["w"], BROKEN["h"]
    g0, g1 = BROKEN["gap"]
    return [("line", (x0, y0), (x0 + w, y0)), ("line", (x0 + w, y0), (x0 + w, y0 + h)),
            ("line", (x0 + w, y0 + h), (g1, y0 + h)), ("line", (g0, y0 + h), (x0, y0 + h)),
            ("line", (x0, y0 + h), (x0, y0))]


def section_segments():
    x, y, w, h = SECTION
    return [("line", (x, y), (x + w, y)), ("line", (x + w, y), (x + w, y + h)),
            ("line", (x + w, y + h), (x, y + h)), ("line", (x, y + h), (x, y))]


def hatch_lines(spacing=2.5):
    """45 degree lines y = x - a clipped to the section rectangle, perpendicular spacing `spacing`."""
    x, y, w, h = SECTION
    out = []
    step = spacing * math.sqrt(2)
    a = x - (y + h) + step
    while a < x + w - y:
        cand = [(y + a, y), (y + h + a, y + h), (x, x - a), (x + w, x + w - a)]
        pts = sorted({(round(px, 9), round(py, 9)) for px, py in cand
                      if x - 1e-9 <= px <= x + w + 1e-9 and y - 1e-9 <= py <= y + h + 1e-9})
        if len(pts) >= 2 and math.dist(pts[0], pts[-1]) > 0.5:
            out.append(("line", pts[0], pts[-1]))
        a += step
    return out


def bowtie_segments():
    p = BOWTIE
    return [("line", p[k], p[(k + 1) % 4]) for k in range(4)]


def frame_segments():
    w, h = SHEET
    trim = [("line", (0, 0), (w, 0)), ("line", (w, 0), (w, h)), ("line", (w, h), (0, h)), ("line", (0, h), (0, 0))]
    fx0, fy0, fx1, fy1 = 20.0, 5.0, w - 5.0, h - 5.0
    frame = [("line", (fx0, fy0), (fx1, fy0)), ("line", (fx1, fy0), (fx1, fy1)), ("line", (fx1, fy1), (fx0, fy1)),
             ("line", (fx0, fy1), (fx0, fy0))]
    tb = [("line", (fx1 - 185, fy0), (fx1 - 185, fy0 + 55)), ("line", (fx1 - 185, fy0 + 55), (fx1, fy0 + 55)),
          ("line", (fx1 - 120, fy0), (fx1 - 120, fy0 + 55)), ("line", (fx1 - 120, fy0 + 25), (fx1, fy0 + 25))]
    return trim, frame + tb


def axis_segments():
    out = []
    for (cx, cy), r in HOLES:
        out.append(("axis", (cx - r - 4, cy), (cx + r + 4, cy)))
        out.append(("axis", (cx, cy - r - 4), (cx, cy + r + 4)))
    return out


def dimension():
    """Horizontal dimension 80 above the plate: extension lines, dimension line with arrows, text."""
    x0, x1, y = PLATE["x0"], PLATE["x0"] + PLATE["w"], PLATE["y0"] + PLATE["h"] + 12
    return {"lines": [("line", (x0, PLATE["y0"] + PLATE["h"] + 1), (x0, y + 2)),
                      ("line", (x1, PLATE["y0"] + PLATE["h"] + 1), (x1, y + 2)), ("line", (x0, y), (x1, y))],
            "arrows": [((x0, y), 0.0), ((x1, y), 180.0)], "text": ("80", (78.0, y + 1.5), 3.5)}


class Canvas:
    def __init__(self, dpi):
        self.k = dpi / 25.4
        self.w, self.h = int(round(SHEET[0] * self.k)), int(round(SHEET[1] * self.k))
        self.img = np.full((self.h, self.w), 255, np.uint8)

    def px(self, p):
        return (int(round(p[0] * self.k * 16)), int(round((SHEET[1] - p[1]) * self.k * 16)))

    def thick(self, mm):
        return max(1, int(round(mm * self.k)))

    def line(self, a, b, mm):
        cv2.line(self.img, self.px(a), self.px(b), 0, self.thick(mm), cv2.LINE_AA, shift=4)

    def arc(self, c, r, a0, a1, mm):
        n = max(8, int(abs(a1 - a0) / 2))
        pts = [(c[0] + r * math.cos(math.radians(a)), c[1] + r * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
        cv2.polylines(self.img, [np.array([self.px(p) for p in pts], np.int32).reshape(-1, 1, 2)], False, 0,
                      self.thick(mm), cv2.LINE_AA, shift=4)

    def dashed(self, a, b, mm, pattern=(12.0, 1.5, 1.0, 1.5)):
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        t, k = 0.0, 0
        while t < L:
            seg = pattern[k % len(pattern)]
            if k % 2 == 0:
                self.line(a + d * t, a + d * min(t + seg, L), mm)
            t += seg
            k += 1

    def centre_line(self, a, b, mm, centre_dash=6.0, gap=1.5, dot=0.8):
        """Dash-dot centre line as drawn by CAD (GOST 2.303): a dash centred on the middle, then gap - dot - gap
        - dash outwards on both sides."""
        a, b = np.array(a, float), np.array(b, float)
        m = (a + b) / 2
        d = (b - a) / np.linalg.norm(b - a)
        half = np.linalg.norm(b - a) / 2
        self.line(m - d * centre_dash / 2, m + d * centre_dash / 2, mm)
        for sgn in (-1, 1):
            t = centre_dash / 2 + gap
            while t < half:
                self.line(m + sgn * d * t, m + sgn * d * min(t + dot, half), mm)
                t += dot + gap
                if t < half:
                    self.line(m + sgn * d * t, m + sgn * d * min(t + 2 * centre_dash, half), mm)
                    t += 2 * centre_dash + gap

    def arrow(self, tip, direction_deg, length=3.0, width=1.0):
        d = np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))])
        n = np.array([-d[1], d[0]])
        tip = np.array(tip)
        pts = [tip, tip + d * length + n * width / 2, tip + d * length - n * width / 2]
        cv2.fillPoly(self.img, [np.array([self.px(p) for p in pts], np.int32).reshape(-1, 1, 2)], 0, cv2.LINE_AA, shift=4)

    def text(self, s, at, height_mm):
        scale = height_mm * self.k / 22.0
        x, y = self.px(at)
        cv2.putText(self.img, s, (x // 16, y // 16), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, int(round(scale * 2))),
                    cv2.LINE_AA)


def draw(item, canvas, mm):
    if item[0] == "line":
        canvas.line(item[1], item[2], mm)
    elif item[0] == "arc":
        canvas.arc(item[1], item[2], item[3], item[4], mm)


def render(dpi=300):
    cv_ = Canvas(dpi)
    trim, frame = frame_segments()
    for s in trim:
        draw(s, cv_, THIN)
    for s in frame:
        draw(s, cv_, THICK)
    for s in plate_segments() + broken_segments() + section_segments() + bowtie_segments():
        draw(s, cv_, THICK)
    for (c, r) in HOLES:
        cv_.arc(c, r, 0, 360, THICK)
    for s in hatch_lines():
        draw(s, cv_, THIN)
    for _, a, b in axis_segments():
        cv_.centre_line(a, b, THIN)
    dim = dimension()
    for s in dim["lines"]:
        draw(s, cv_, THIN)
    for tip, ang in dim["arrows"]:
        cv_.arrow(tip, ang)
    for s, at, hgt in TEXTS + [dim["text"]]:
        cv_.text(s, at, hgt)
    cv_.text("CONTROL SHEET", (115.0, 20.0), 5.0)
    return cv_.img


def simulate_scan(img, dpi, angle_deg=1.2, seed=7):
    rng = np.random.default_rng(seed)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    out = cv2.GaussianBlur(out, (3, 3), 0.8)
    yy, xx = np.mgrid[0:h, 0:w]
    shade = (12 * (xx / w) + 8 * (yy / h)).astype(np.float32)
    noisy = out.astype(np.float32) - shade + rng.normal(0, 6, out.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def write_scaled_pdf(img, dpi, path, sx, sy=None):
    """PDF of the sheet size with the drawing printed at sx (x) / sy (y) of its size, centred - like a document
    printed with "fit to page" or by a driver with its own margins."""
    from PIL import Image

    sy = sx if sy is None else sy
    h, w = img.shape
    scaled = cv2.resize(img, (int(round(w * sx)), int(round(h * sy))), interpolation=cv2.INTER_AREA)
    page = np.full((h, w), 255, np.uint8)
    sh, sw = scaled.shape
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    if sh <= h and sw <= w:
        page[y0:y0 + sh, x0:x0 + sw] = scaled
    else:
        page = scaled[-y0:-y0 + h, -x0:-x0 + w] if y0 < 0 and x0 < 0 else cv2.resize(scaled, (w, h))
    Image.fromarray(page).save(path, resolution=dpi)


def write_dxf(path):
    import ezdxf

    doc = ezdxf.new("R2007", setup=True)
    doc.units = 6
    for name, lt in (("MAIN", "CONTINUOUS"), ("THIN", "CONTINUOUS"), ("AXIS", "CENTER"), ("DEFPOINTS", "CONTINUOUS")):
        if name not in doc.layers:
            doc.layers.add(name, linetype=lt)
    msp = doc.modelspace()

    def add(item, layer):
        if item[0] in ("line", "axis"):
            msp.add_line(item[1], item[2], dxfattribs={"layer": layer})
        else:
            msp.add_arc(item[1], item[2], item[3], item[4], dxfattribs={"layer": layer})

    trim, frame = frame_segments()
    for s in trim:
        add(s, "DEFPOINTS")
    for s in frame + plate_segments() + broken_segments() + section_segments() + bowtie_segments():
        add(s, "MAIN")
    for c, r in HOLES:
        msp.add_circle(c, r, dxfattribs={"layer": "MAIN"})
    for s in axis_segments():
        add(s, "AXIS")
    hatch = msp.add_hatch(dxfattribs={"layer": "THIN"})
    x, y, w, h = SECTION
    hatch.paths.add_polyline_path([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], is_closed=True)
    hatch.set_pattern_fill("ANSI31", scale=2.5 / 3.175)
    dim = dimension()
    msp.add_linear_dim(base=(0, dim["lines"][2][1][1]), p1=dim["lines"][0][1], p2=dim["lines"][1][1],
                       dxfattribs={"layer": "THIN"}).render()
    for s, at, hgt in TEXTS:
        msp.add_text(s, height=hgt, dxfattribs={"layer": "THIN", "insert": at})
    doc.saveas(path)


def truth():
    y0, h = BROKEN["y0"], BROKEN["h"]
    return {
        "sheet_mm": list(SHEET),
        "circles": [{"center": list(c), "radius": r} for c, r in HOLES],
        "arcs": [{"center": [PLATE["x0"] + PLATE["w"] - PLATE["r"], PLATE["y0"] + PLATE["h"] - PLATE["r"]],
                  "radius": PLATE["r"], "start_angle": 0.0, "end_angle": 90.0}],
        "closed_profiles": [
            {"name": "plate", "area_mm2": PLATE["w"] * PLATE["h"] - (1 - math.pi / 4) * PLATE["r"] ** 2,
             "bbox": [PLATE["x0"], PLATE["y0"], PLATE["x0"] + PLATE["w"], PLATE["y0"] + PLATE["h"]]},
            {"name": "section", "area_mm2": SECTION[2] * SECTION[3],
             "bbox": [SECTION[0], SECTION[1], SECTION[0] + SECTION[2], SECTION[1] + SECTION[3]]}],
        "gaps": [{"name": "broken", "between": [[BROKEN["gap"][0], y0 + h], [BROKEN["gap"][1], y0 + h]],
                  "distance_mm": BROKEN["gap"][1] - BROKEN["gap"][0]}],
        "self_intersections": [{"name": "bowtie", "at": [145.0, 125.0]}],
        "hatch": {"bbox": [SECTION[0], SECTION[1], SECTION[0] + SECTION[2], SECTION[1] + SECTION[3]], "angle_deg": 45.0,
                  "spacing_mm": 2.5},
        "axes": [{"start": list(a), "end": list(b)} for _, a, b in axis_segments()],
    }


def make_all(out_dir, dpi=300):
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = render(dpi)
    cv2.imencode(".png", img)[1].tofile(str(out / "control.png"))
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out / "control.jpg"))
    Image.fromarray(img).save(out / "control.pdf", resolution=dpi)
    scan = simulate_scan(img, dpi)
    cv2.imencode(".jpg", scan, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tofile(str(out / "control_scan.jpg"))
    write_dxf(str(out / "control.dxf"))
    (out / "truth.json").write_text(json.dumps(truth(), indent=1), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dpi", type=float, default=300)
    a = ap.parse_args()
    print(make_all(a.out, a.dpi))


if __name__ == "__main__":
    main()

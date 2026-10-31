"""Text detection + recognition (RapidOCR / PP-OCRv5 'eslav' recogniser: Cyrillic + Latin + digits).
Two passes (upright and rotated 90° CW) cover horizontal and vertical texts; results are merged."""
import math
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from raster import to_mm


@dataclass
class TextItem:
    text: str
    score: float
    poly: np.ndarray          # 4x2 mm, reading order: top-left, top-right, bottom-right, bottom-left
    angle: float              # reading direction, degrees (0 = left->right, 90 = bottom->top)
    height: float             # box height across the reading direction, mm

    @property
    def center(self):
        return self.poly.mean(axis=0)

    def bbox(self):
        return (*self.poly.min(axis=0), *self.poly.max(axis=0))


_engine = None


def engine():
    global _engine
    if _engine is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        if os.getenv("OCR_MODEL_PATH") and not os.getenv("MODEL_DIR"):
            # Existing backend profile: only the recognizer path is configured.
            # This image does not include the separate service/ prototype.
            params = {"Rec.lang_type": LangRec.ESLAV, "Rec.ocr_version": OCRVersion.PPOCRV5,
                      "Rec.model_type": ModelType.MOBILE, "Det.limit_side_len": 6000,
                      "Det.limit_type": "max", "Global.text_score": 0.3,
                      "Global.log_level": "warning",
                      "Rec.model_path": os.environ["OCR_MODEL_PATH"]}
        else:
            # Offline profile: all assets must pass validation, with no fallback
            # to the backend profile if an asset is missing or corrupted.
            from service.models import rapidocr_params
            params = rapidocr_params()
            params.update({
                "Rec.lang_type": LangRec.ESLAV,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
            })
        _engine = RapidOCR(params=params)
    return _engine


def _items(result, back, cal, f):
    out = []
    if result.boxes is None:
        return out
    for box, txt, sc in zip(result.boxes, result.txts, result.scores):
        b = back(np.asarray(box, float)) / f
        xs, ys = to_mm(cal, b[:, 0], b[:, 1])
        poly = np.column_stack([xs, ys])
        d = poly[1] - poly[0]
        ang = math.degrees(math.atan2(d[1], d[0]))
        hgt = float(np.linalg.norm(poly[3] - poly[0]))
        txt = txt.strip()
        if txt:
            out.append(TextItem(txt, float(sc), poly, ang, hgt))
    return out


def run_ocr(gray, cal, scale=0.5):
    img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = img.shape
    eng = engine()
    items = _items(eng(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)), lambda b: b, cal, scale)
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    back = lambda b: np.column_stack([b[:, 1], H - 1 - b[:, 0]])   # rotated (u, v) -> original (x, y)
    items_r = _items(eng(cv2.cvtColor(rot, cv2.COLOR_GRAY2BGR)), back, cal, scale)
    return merge(items, items_r)


def ocr_window(gray, cal, box, scales=(0.35, 0.25, 0.5)):
    """Targeted OCR of a small sheet window (mm box) - used for labels missed by the full-page pass."""
    from raster import to_px
    x0, y0, x1, y1 = box
    px0, py1 = to_px(cal, x0, y0)
    px1, py0 = to_px(cal, x1, y1)
    px0, py0 = max(int(px0), 0), max(int(py0), 0)
    crop = gray[py0:int(py1), px0:int(px1)]
    if crop.size == 0:
        return []
    out = []
    for f in scales:
        img = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        res = engine()(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        out += _items(res, lambda b: (b - 20) + [px0 * f, py0 * f], cal, f)
    return out


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / ua if ua > 0 else 0, inter / small if small > 0 else 0


def _orientation_ok(it):
    """A reading whose direction matches the long side of its box."""
    x0, y0, x1, y1 = it.bbox()
    horizontal_box = (x1 - x0) >= (y1 - y0)
    horizontal_read = abs(math.sin(math.radians(it.angle))) < 0.5
    return bool(horizontal_box == horizontal_read) or len(it.text) <= 2


def merge(a, b):
    """Union of two passes; overlapping detections keep the better (orientation-consistent, higher score)."""
    pool = sorted(a + b, key=lambda t: (-_orientation_ok(t), -t.score * min(len(t.text), 6)))
    kept = []
    for it in pool:
        if any(_iou(it, k)[1] > 0.5 for k in kept):
            continue
        kept.append(it)
    return kept


DIAM = "⌀"
_DIAM_LEAD = re.compile(r"^[ØøΦφ$⌀]\s*")


def normalize(txt):
    """Typical OCR confusions on ESKD fonts."""
    t = txt.replace("×", "x").replace("Х", "x") if re.search(r"\d[×Хx]\d", txt) else txt.replace("×", "x")
    t = _DIAM_LEAD.sub(DIAM, t)
    t = t.replace("°", "°").replace("º", "°")
    return t


_NUM = re.compile(r"(\d+(?:[.,]\d+)?)")


def parse_dim_text(txt):
    """-> (prefix, value or None, rest).  '⌀150±0,5' -> ('⌀', 150.0, '±0,5')."""
    t = normalize(txt).strip().lstrip("/").strip()
    prefix = ""
    if t.startswith(DIAM):
        prefix, t = DIAM, t[1:]
    elif t[:1] in ("R", "r") and len(t) > 1 and t[1].isdigit():
        prefix, t = "R", t[1:]
    elif t[:1] in ("M", "М") and len(t) > 1 and t[1].isdigit():
        prefix, t = "M", t[1:]
    m = _NUM.match(t)
    if not m:
        return prefix, None, t
    v = float(m.group(1).replace(",", "."))
    return prefix, v, t[m.end():]

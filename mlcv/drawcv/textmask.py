"""Text regions and their removal from the working binary image.

Providers:
  rapidocr   - text detector of RapidOCR (DBNet, ONNX, offline); only detection, recognition belongs to OCR;
  file       - text polygons from the OCR module (ocr.json, coordinates in source pixels);
  components - fallback without models: clusters of small glyph-sized connected components;
  none       - no masking.
Only connected components lying (almost) entirely inside a text polygon are erased, so a line crossing a label
survives. Erased pixels are kept as a separate mask for review; the source image is never changed."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

_engine = None
INSIDE_FRACTION = 0.98
MAX_TEXT_HEIGHT_MM = 15.0


def rapidocr_engine():
    global _engine
    if _engine is None:
        params = {"Det.limit_side_len": 5000, "Det.limit_type": "max", "Global.log_level": "warning"}
        config = os.getenv("DRAWCV_RAPIDOCR_CONFIG")
        if config:
            params.update(json.loads(Path(config).read_text(encoding="utf-8")))
            for key in ("Det.model_path", "Cls.model_path", "Rec.model_path", "Rec.rec_keys_path"):
                if key not in params or not Path(params[key]).is_file():
                    raise FileNotFoundError(f"Local RapidOCR asset missing: {key}")
        elif os.getenv("DRAWCV_OFFLINE") == "1":
            raise RuntimeError("Offline drawcv requires DRAWCV_RAPIDOCR_CONFIG with local model paths")
        from rapidocr import RapidOCR, ModelType, OCRVersion
        # JSON stores strings; RapidOCR's Python params require Enum instances.
        enum_fields = {"model_type": ModelType, "ocr_version": OCRVersion}
        for key, value in list(params.items()):
            enum = enum_fields.get(key.rsplit(".", 1)[-1])
            if enum is not None:
                params[key] = enum(value)
        _engine = RapidOCR(params=params)
    return _engine


def detect_rapidocr(gray):
    res = rapidocr_engine()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), use_det=True, use_cls=False, use_rec=False)
    if res.boxes is None:
        return []
    scores = res.scores if res.scores is not None else [None] * len(res.boxes)
    return [{"poly_work_px": np.asarray(b, float), "score": float(s) if s is not None else None}
            for b, s in zip(res.boxes, scores)]


def load_ocr_file(path, source_to_work):
    """Accepts {"texts"|"candidates"|"regions": [...]} or a bare list; each item has "poly_source_px" / "poly_px"
    (list of [x, y]) or "box_px" ([x0, y0, x1, y1]) in source pixels."""
    data = json.loads(open(path, encoding="utf-8").read())
    items = data if isinstance(data, list) else next((data[k] for k in ("regions", "texts", "candidates") if k in data), [])
    out = []
    for it in items:
        poly = it.get("poly_source_px") or it.get("poly_px")
        if poly is None and it.get("box_px"):
            x0, y0, x1, y1 = it["box_px"]
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if poly is None:
            continue
        out.append({"poly_work_px": source_to_work.apply(poly), "score": it.get("score"), "ocr_id": it.get("id"),
                    "text": it.get("text") or it.get("value")})
    return out


def detect_components(bw, px_per_mm):
    """Glyph-sized components grouped into lines by a small closing; returns rotated rectangles."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    hmm, wmm = h / px_per_mm, w / px_per_mm
    fill = area / np.maximum(w * h, 1)
    glyph = (hmm > 1.2) & (hmm < 10) & (wmm < 8) & (np.maximum(w, h) / np.maximum(np.minimum(w, h), 1) < 6) & (fill > 0.12)
    mask = np.isin(lab, np.where(glyph)[0] + 1).astype(np.uint8)
    k = max(3, int(1.2 * px_per_mm))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if min(rw, rh) / px_per_mm < 1.2:
            continue
        out.append({"poly_work_px": cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(float), "score": None})
    return out


def find_text_regions(prep, page_gray_work, params, px_per_mm):
    provider = params.text_provider
    if provider == "auto":
        provider = "file" if params.ocr_json else "rapidocr"
    regions, note = [], None
    if provider == "file":
        regions = load_ocr_file(params.ocr_json, prep.source_to_work)
    elif provider == "rapidocr":
        try:
            regions = detect_rapidocr(page_gray_work)
        except Exception as exc:          # model missing / runtime error: fall back, but say so
            if os.getenv("DRAWCV_STRICT_MODELS") == "1":
                raise
            note = f"rapidocr unavailable ({type(exc).__name__}: {exc}); components fallback used"
            provider = "components"
    if provider == "components":
        regions = detect_components(prep.bw, px_per_mm)
    kept = []
    for r in regions:
        rect = cv2.minAreaRect(r["poly_work_px"].astype(np.float32))
        if min(rect[1]) / px_per_mm > MAX_TEXT_HEIGHT_MM:
            continue
        kept.append(r)
    for i, r in enumerate(kept):
        r["id"] = f"textreg_{i}"
        r["provider"] = provider
    return kept, provider, note


def erase_text(bw, regions, px_per_mm, margin_mm):
    """Erase components that lie inside text polygons (expanded by margin). Returns (clean, erased_mask)."""
    h, w = bw.shape
    region_mask = np.zeros((h, w), np.uint8)
    m = max(1, int(round(margin_mm * px_per_mm)))
    for r in regions:
        cv2.fillPoly(region_mask, [np.round(r["poly_work_px"]).astype(np.int32)], 1)
    region_mask = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * m + 1, 2 * m + 1)))
    n, lab = cv2.connectedComponents(bw, connectivity=8)
    total = np.bincount(lab.ravel(), minlength=n)
    inside = np.bincount(lab[region_mask > 0], minlength=n)
    frac = inside / np.maximum(total, 1)
    erase_ids = np.where(frac >= INSIDE_FRACTION)[0]
    erase_ids = erase_ids[erase_ids > 0]
    erased = np.isin(lab, erase_ids)
    clean = bw.copy()
    clean[erased] = 0
    erase_set = set(erase_ids.tolist())
    for r in regions:
        poly = np.round(r["poly_work_px"]).astype(np.int32)
        x0, y0 = np.clip(poly.min(axis=0), 0, [w - 1, h - 1])
        x1, y1 = np.clip(poly.max(axis=0) + 1, 1, [w, h])
        rm = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(rm, [poly - [x0, y0]], 1)
        sub = lab[y0:y1, x0:x1]
        ids = np.unique(sub[(rm > 0) & (sub > 0)])
        r["erased_components"] = sum(int(i) in erase_set for i in ids)
        r["kept_components"] = int(len(ids) - r["erased_components"])
    return clean, erased, region_mask > 0

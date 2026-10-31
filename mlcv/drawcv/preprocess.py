"""Page preparation: manual 90-degree rotation, skew correction, resampling to the working resolution,
background/contrast normalisation, binarisation and removal of tiny specks.

The source image is never modified; every geometric step is recorded in one source_px -> work_px transform,
so any object found on the working image can be mapped back to the original page."""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibrate import frame_hypotheses
from .transform import Transform

ANALYSIS_SIDE = 3000                 # enough for the title block lines of an A1 sheet (3.6 px/mm)


@dataclass
class Prepared:
    gray: np.ndarray                 # working grayscale after normalisation
    bw: np.ndarray                   # working binary, ink = 1
    source_to_work: Transform
    skew_deg: float
    resample: float
    px_per_mm_estimate: float | None
    specks_removed: int
    speck_mask: np.ndarray | None
    notes: list = field(default_factory=list)

    def info(self):
        return {"skew_deg": round(self.skew_deg, 4), "resample": round(self.resample, 6),
                "px_per_mm_estimate": self.px_per_mm_estimate, "specks_removed": self.specks_removed,
                "work_size_px": [int(self.bw.shape[1]), int(self.bw.shape[0])], "notes": self.notes}


def binarize(gray):
    """Otsu on a lightly blurred image; ink = 1."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw.astype(np.uint8)


def analysis_copy(gray, f):
    """Small copy for global analysis. A min filter first keeps thin dark lines (a 2 px frame line of a 1200 dpi
    sheet vanishes under plain area averaging)."""
    if f >= 1:
        return gray
    k = max(1, int(np.ceil(1 / f)))
    return cv2.resize(cv2.erode(gray, np.ones((k, k), np.uint8)), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def estimate_skew(gray, max_deg):
    """Dominant deviation of long near-horizontal/vertical lines from the axes (degrees, image y down)."""
    h, w = gray.shape
    f = ANALYSIS_SIDE / max(h, w)
    small = analysis_copy(gray, f)
    bw = binarize(small)
    lines = cv2.HoughLinesP(bw * 255, 1, np.pi / 3600, threshold=150, minLineLength=small.shape[1] // 4, maxLineGap=4)
    if lines is None:
        return 0.0, 0
    angs, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) < max_deg:
            angs.append(a)
        elif abs(abs(a) - 90) < max_deg:
            angs.append(a - 90 * np.sign(a))
        else:
            continue
        weights.append(math.hypot(x2 - x1, y2 - y1))
    if not angs:
        return 0.0, 0
    order = np.argsort(angs)
    cum = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(angs)[order][np.searchsorted(cum, cum[-1] / 2)])
    return median, len(angs)


def rotation_transform(w, h, angle_deg):
    """Rotation about the image centre by angle_deg (OpenCV sign: positive = counter-clockwise on screen),
    with the output canvas enlarged to keep every corner. Returns (Transform, out_w, out_h)."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    rc = corners @ m[:, :2].T + m[:, 2]
    x0, y0 = rc.min(axis=0)
    x1, y1 = rc.max(axis=0)
    m[:, 2] -= (x0, y0)
    return Transform.from_affine(m), int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))


def quarter_turn(gray, rotate_deg):
    """Exact lossless rotation by a multiple of 90 degrees (counter-clockwise)."""
    k = (int(rotate_deg) // 90) % 4
    h, w = gray.shape
    if k == 0:
        return gray, Transform()
    codes = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}
    mats = {1: [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]],
            2: [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]],
            3: [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]}
    return cv2.rotate(gray, codes[k]), Transform(mats[k])


def normalise_background(gray, px_per_mm):
    """Divide by a large-scale background estimate (uneven lighting of scans) and stretch to full range.
    Thin strokes and small text are preserved: the background kernel is several mm wide."""
    k = max(15, int(round(3.0 * px_per_mm)) | 1)
    small_f = 0.25
    small = cv2.resize(gray, None, fx=small_f, fy=small_f, interpolation=cv2.INTER_AREA)
    ks = max(3, int(k * small_f) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (ks, ks), 0), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255)
    lo, hi = np.percentile(norm[::4, ::4], (0.5, 99.5))
    if hi - lo < 10:
        return gray
    return np.clip((norm - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def resolution_estimate(gray, page_dpi, params):
    """px/mm of the image before resampling: stated dpi (PDF render / file metadata) or a frame hypothesis on a
    small copy. Returns (value or None, how)."""
    if page_dpi:
        return page_dpi / 25.4, "stated dpi"
    h, w = gray.shape
    f = ANALYSIS_SIDE / max(h, w)
    small = analysis_copy(gray, f)
    hyps = frame_hypotheses(binarize(small), params.sheet)
    good = [hy for hy in hyps if hy["xy_agreement"] < 0.02 or hy.get("title_block_verified")]
    if good:
        return float(np.mean(good[0]["px_per_mm"])) / min(f, 1.0), f"frame ({good[0]['method']})"
    return None, "unknown"


def prepare(page, params):
    notes = []
    gray, t = quarter_turn(page.gray, params.rotate_deg)
    if params.rotate_deg % 360:
        notes.append(f"manual rotation {params.rotate_deg} deg")

    skew, n_lines = estimate_skew(gray, params.max_skew_deg)
    if abs(skew) < params.min_skew_deg:
        skew = 0.0
    if skew:
        notes.append(f"skew {skew:.3f} deg corrected from {n_lines} long lines")

    est, how = resolution_estimate(gray, page.dpi, params)
    resample = 1.0
    if est and est > params.work_px_per_mm * 1.05:
        resample = params.work_px_per_mm / est
    notes.append(f"resolution {est:.3f} px/mm ({how}); resample x{resample:.4f}" if est else
                 "resolution unknown before calibration; image kept at its resolution")

    h, w = gray.shape
    if resample != 1.0:
        gray = cv2.resize(gray, (int(round(w * resample)), int(round(h * resample))), interpolation=cv2.INTER_AREA)
        sx, sy = gray.shape[1] / w, gray.shape[0] / h
        t = t.then(Transform([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]]))
    if skew:
        hh, ww = gray.shape
        # rotating by +skew (counter-clockwise in OpenCV's y-down sense) removes a clockwise tilt of -skew
        rt, ow, oh = rotation_transform(ww, hh, skew)
        gray = cv2.warpAffine(gray, rt.m[:2], (ow, oh), flags=cv2.INTER_LINEAR, borderValue=255)
        t = t.then(rt)

    px_per_mm = est * resample if est else None
    if params.contrast == "normalize":
        gray = normalise_background(gray, px_per_mm or 10.0)
    elif params.contrast == "clahe":
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bw = binarize(gray)

    removed, speck_mask = 0, None
    if params.despeckle_mm2 > 0 and px_per_mm:
        min_area = params.despeckle_mm2 * px_per_mm ** 2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
        if len(small):
            speck_mask = np.isin(lab, small)
            bw[speck_mask] = 0
            removed = int(len(small))
    return Prepared(gray=gray, bw=bw, source_to_work=t, skew_deg=float(skew), resample=float(resample),
                    px_per_mm_estimate=px_per_mm, specks_removed=removed, speck_mask=speck_mask, notes=notes)

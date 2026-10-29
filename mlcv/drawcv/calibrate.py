"""px <-> sheet mm calibration of the working image.

Methods, in order of preference:
  manual       - px/mm (and optionally the origin) given by the user;
  sheet_trim   - outermost long lines are the sheet trim line;
  inner_frame  - outermost long lines are the GOST 2.301 frame (margins 20/5/5/5 mm) - scans cut at the trim;
  mixed_trim_frame - trim line on some sides, frame on the others (paper edge cut on one side);
  pdf_page     - no frame found, the PDF page has a standard sheet size: px/mm from the render dpi; the scale
                 stays "unknown" because PDFs printed "fit to page" scale the content by an unknown percentage;
  image_size   - nothing found: the image is assumed to be the whole sheet, scale_status = unknown.
The trim/frame hypothesis is chosen by agreement of the x and y scales and verified by the 185 x 55 mm title
block, which also fixes the sheet format (all A formats have the same aspect ratio). Without a verified title block
a disagreement above 0.3 % makes the scale unknown instead of silently wrong; with it, x and y keep separate scales
and the anisotropy is reported."""
import numpy as np

from .transform import Transform

SHEETS = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (210, 297)}
FRAME_MARGINS = (20.0, 5.0, 5.0, 5.0)        # left, bottom, right, top
TITLE_BLOCK_MM = (185.0, 55.0)               # GOST 2.104 main title block: the same size on every format
TITLE_BLOCK_OK = 0.9
MAX_ANISOTROPY = 0.02                         # printed PDFs: x and y may be scaled differently (seen 0.65-1 %)
AGREEMENT_TOL = 0.003         # correct hypotheses agree to 0.003-0.03 %; wrong role mixes reach 0.5 %


def long_lines(bw, frac=0.5):
    h, w = bw.shape
    rows = np.where(bw.sum(axis=1) > frac * w)[0]
    cols = np.where(bw.sum(axis=0) > frac * h)[0]
    return rows, cols


def _stroke_centre(idx, end):
    """Centre of the contiguous run of long-line rows/cols at the start (end=0) or end (end=-1):
    a thick frame line is measured at its middle, not at its outer edge."""
    seq = list(idx) if end == 0 else list(idx)[::-1]
    run = [seq[0]]
    for v in seq[1:]:
        if abs(v - run[-1]) != 1:
            break
        run.append(v)
    return float(np.mean(run))


def guess_sheet(w_px, h_px, px_per_mm=None):
    """With a known resolution the format is the one whose size fits; otherwise A3 landscape / A4 portrait,
    the most common archive formats."""
    if px_per_mm:
        w_mm, h_mm = sorted((w_px / px_per_mm, h_px / px_per_mm), reverse=True)
        best = min(SHEETS, key=lambda s: abs(max(SHEETS[s]) - w_mm) + abs(min(SHEETS[s]) - h_mm))
        return best
    return "A3" if w_px > h_px else "A4"


def oriented(sheet, landscape):
    a, b = SHEETS[sheet]
    return (max(a, b), min(a, b)) if landscape else (min(a, b), max(a, b))


def standard_sheet_of(size_mm, tol=3.0):
    w, h = size_mm
    for name, (a, b) in SHEETS.items():
        if (abs(w - a) < tol and abs(h - b) < tol) or (abs(w - b) < tol and abs(h - a) < tol):
            return name
    return None


def _runs(idx):
    """Contiguous runs of row/col indices -> list of (centre, first, last)."""
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for v in idx[1:]:
        if v != prev + 1:
            out.append(((start + prev) / 2.0, start, prev))
            start = v
        prev = v
    out.append(((start + prev) / 2.0, start, prev))
    return out


def frame_hypotheses(bw, sheet=None, px_per_mm_hint=None):
    """Candidate calibrations from the outermost long lines, best first.

    Each side is solved independently: its line is the outermost or the next long line, and it is either the
    trim line or the GOST 2.301 frame (margins 20/5/5/5 mm). Scans often keep the trim line on some sides only
    (cut paper edge), so all combinations are scored by the agreement of the x and y scales."""
    h, w = bw.shape
    rows, cols = long_lines(bw)
    rr, cc = _runs(rows), _runs(cols)
    if len(rr) < 2 or len(cc) < 2:
        return []
    sides = {"left": cc[:2], "right": cc[::-1][:2], "top": rr[:2], "bottom": rr[::-1][:2]}
    if cc[-1][0] - cc[0][0] < 0.75 * w or rr[-1][0] - rr[0][0] < 0.75 * h:
        return []
    landscape = (cc[-1][0] - cc[0][0]) > (rr[-1][0] - rr[0][0])
    thickest = max(run[2] - run[1] + 1 for runs in sides.values() for run in runs)

    def weight_mismatch(run, role):
        """The frame is drawn with the main (thick) line, the trim line is thin."""
        t = run[2] - run[1] + 1
        return int((role == "frame" and t < 0.5 * thickest) or (role == "trim" and t >= 0.75 * thickest and thickest >= 3))

    names = [sheet] if sheet else list(SHEETS)
    ml, mb, mr, mt = FRAME_MARGINS
    hyps = []
    for name in names:
        sw, sh = oriented(name, landscape)
        for li, lrole in [(k, r) for k in range(len(sides["left"])) for r in ("trim", "frame")]:
            for ri, rrole in [(k, r) for k in range(len(sides["right"])) for r in ("trim", "frame")]:
                xl = 0.0 if lrole == "trim" else ml
                xr = sw if rrole == "trim" else sw - mr
                cl, cr = sides["left"][li][0], sides["right"][ri][0]
                if cr <= cl:
                    continue
                sx = (cr - cl) / (xr - xl)
                for ti, trole in [(k, r) for k in range(len(sides["top"])) for r in ("trim", "frame")]:
                    for bi, brole in [(k, r) for k in range(len(sides["bottom"])) for r in ("trim", "frame")]:
                        yt = sh if trole == "trim" else sh - mt
                        yb = 0.0 if brole == "trim" else mb
                        rt, rb = sides["top"][ti][0], sides["bottom"][bi][0]
                        if rb <= rt:
                            continue
                        sy = (rb - rt) / (yt - yb)
                        agreement = abs(sx / sy - 1)
                        inner = li + ri + ti + bi
                        hint_err = abs(sx / px_per_mm_hint - 1) if px_per_mm_hint else 0.0
                        hint_err = hint_err if hint_err > 0.03 else 0.0      # the hint only selects the format
                        roles = {"left": lrole, "right": rrole, "top": trole, "bottom": brole}
                        # the partner line of each side (frame for a trim line and vice versa) must exist
                        # wherever it falls inside the image
                        missing = (weight_mismatch(sides["left"][li], lrole) + weight_mismatch(sides["right"][ri], rrole)
                                   + weight_mismatch(sides["top"][ti], trole) + weight_mismatch(sides["bottom"][bi], brole))
                        for pos, step, runs, size in ((cl, ml * sx * (1 if lrole == "trim" else -1), cc, w),
                                                      (cr, mr * sx * (-1 if rrole == "trim" else 1), cc, w),
                                                      (rt, mt * sy * (1 if trole == "trim" else -1), rr, h),
                                                      (rb, mb * sy * (-1 if brole == "trim" else 1), rr, h)):
                            q = pos + step
                            if 0 <= q < size and not any(abs(run[0] - q) <= 0.6 * sx for run in runs):
                                missing += 1
                        method = ("sheet_trim" if set(roles.values()) == {"trim"} else
                                  "inner_frame" if set(roles.values()) == {"frame"} else "mixed_trim_frame")
                        hyps.append({"method": method, "sheet": name, "sheet_mm": [sw, sh], "px_per_mm": [sx, sy],
                                     "origin_px": [cl - xl * sx, rb + yb * sy], "xy_agreement": agreement,
                                     "frame_px": [cl, rt, cr, rb], "side_roles": roles,
                                     "partner_lines_missing": missing,
                                     "_score": agreement + 0.002 * inner + 0.01 * missing + hint_err})
    # All A formats share the aspect ratio, so the frame alone fixes the scale only up to the format. The title
    # block is 185 x 55 mm on every format: a hypothesis whose predicted title-block lines are inked is verified.
    for hy in hyps:
        hy["title_block"] = (title_block_score(bw, hy)
                             if hy["xy_agreement"] <= MAX_ANISOTROPY else 0.0)
        # an anisotropic fit needs a second, independent check: a wrong side role (5 mm) can be compensated by a
        # wrong x scale so that the title block still lines up; then a partner line is missing
        hy["title_block_verified"] = bool(hy["title_block"] >= TITLE_BLOCK_OK) and bool(
            hy["xy_agreement"] <= AGREEMENT_TOL or hy["partner_lines_missing"] == 0)
    verified = [hy for hy in hyps if hy["title_block_verified"]]
    if verified:
        best = min(verified, key=lambda hy: hy["_score"])
        return [best] + sorted((hy for hy in hyps if hy is not best), key=lambda hy: hy["_score"])
    if not sheet:
        # without a title block keep the guessed format (or the one matching the resolution hint)
        guess = guess_sheet(cc[-1][0] - cc[0][0], rr[-1][0] - rr[0][0], px_per_mm_hint)
        hyps = [hy for hy in hyps if hy["sheet"] == guess] if not px_per_mm_hint else hyps
    hyps.sort(key=lambda hy: hy["_score"])
    return hyps


def title_block_score(bw, hy):
    """Share of the predicted left and top edges of the title block (at the bottom-right of the frame) on ink; the
    weaker of the two edges counts, so a hypothesis right in y only (wrong x roles) is not confirmed."""
    h, w = bw.shape
    sw, sh = hy["sheet_mm"]
    sx, sy = hy["px_per_mm"]
    ox, oy = hy["origin_px"]
    _, mb, mr, _ = FRAME_MARGINS
    tw, th = TITLE_BLOCK_MM
    x_left, y_top = sw - mr - tw, mb + th
    ys = np.arange(mb + 2, y_top - 2, 0.5)
    xs = np.arange(x_left + 2, sw - mr - 2, 0.5)
    win = max(1, int(round(0.6 * min(sx, sy))))
    hits, hits_h = [], []
    cx = int(round(ox + x_left * sx))
    for y in ys:
        r = int(round(oy - y * sy))
        hits.append(0 <= r < h and 0 <= cx - win and cx + win < w and bool(bw[r, cx - win:cx + win + 1].any()))
    ry = int(round(oy - y_top * sy))
    for x in xs:
        c = int(round(ox + x * sx))
        hits_h.append(0 <= c < w and 0 <= ry - win and ry + win < h and bool(bw[ry - win:ry + win + 1, c].any()))
    return float(min(np.mean(hits), np.mean(hits_h))) if hits and hits_h else 0.0


def to_sheet_transform(px_per_mm, origin_px):
    sx, sy = px_per_mm
    ox, oy = origin_px
    return Transform([[1 / sx, 0, -ox / sx], [0, -1 / sy, oy / sy], [0, 0, 1]])


def calibrate(bw, params, pdf_page_mm=None, pdf_origin_work_px=None, work_px_per_mm_known=None):
    """Calibration dict for the working binary image (ink = 1)."""
    h, w = bw.shape
    result = {"manual": False, "view_scale": None,
              "view_scale_note": "view scale (1:2, 2:1 ...) is read from the title block by OCR, not by this module"}
    hyps = frame_hypotheses(bw, params.sheet, work_px_per_mm_known)
    frame = hyps[0] if hyps else None

    if params.manual_px_per_mm:
        k = float(params.manual_px_per_mm)
        origin = list(params.manual_origin_px) if params.manual_origin_px else (frame["origin_px"] if frame else [0.0, h - 1.0])
        sheet = params.sheet or (frame["sheet"] if frame else guess_sheet(w, h, k))
        result.update(method="manual", manual=True, sheet=sheet, sheet_mm=list(oriented(sheet, w > h)),
                      px_per_mm=[k, k], origin_px=origin, xy_agreement=0.0, scale_status="known")
    elif frame and (frame["xy_agreement"] <= AGREEMENT_TOL or frame.get("title_block_verified")):
        # the drawn frame wins over the PDF page size: PDFs are often printed "fit to page" (scaled by 0.5-3 %);
        # with the title block confirming the side roles, x and y keep their own scales (anisotropic print)
        result.update({k: v for k, v in frame.items() if not k.startswith("_")})
        result["scale_status"] = "known"
        if frame["xy_agreement"] > AGREEMENT_TOL:
            result["anisotropy"] = round(frame["px_per_mm"][0] / frame["px_per_mm"][1] - 1, 6)
        if pdf_page_mm is not None and work_px_per_mm_known:
            result["pdf_page_check"] = {"page_mm": [round(v, 3) for v in pdf_page_mm],
                                        "page_px_per_mm": work_px_per_mm_known,
                                        "content_scale_vs_page": round(work_px_per_mm_known / frame["px_per_mm"][0], 6)}
    elif pdf_page_mm is not None and standard_sheet_of(pdf_page_mm) and work_px_per_mm_known:
        sheet = standard_sheet_of(pdf_page_mm)
        k = work_px_per_mm_known
        # unverified: a PDF printed "fit to page" scales the content by an unknown percentage
        result.update(method="pdf_page", sheet=sheet, sheet_mm=[round(v, 3) for v in pdf_page_mm],
                      px_per_mm=[k, k], origin_px=list(pdf_origin_work_px), xy_agreement=0.0, scale_status="unknown")
        if frame:
            result["rejected_frame"] = {k_: v for k_, v in frame.items() if not k_.startswith("_")}
    else:
        sheet = params.sheet or guess_sheet(w, h, work_px_per_mm_known)
        sw, sh = oriented(sheet, w > h)
        result.update(method="image_size", sheet=sheet, sheet_mm=[sw, sh], px_per_mm=[(w - 1) / sw, (h - 1) / sh],
                      origin_px=[0.0, h - 1.0], xy_agreement=abs(((w - 1) / sw) / ((h - 1) / sh) - 1),
                      scale_status="unknown")
        if frame:
            result["rejected_frame"] = {k: v for k, v in frame.items() if not k.startswith("_")}
    if work_px_per_mm_known and result["method"] in ("sheet_trim", "inner_frame", "mixed_trim_frame", "image_size"):
        result["resolution_check"] = {"stated_px_per_mm": work_px_per_mm_known,
                                      "relative_error": abs(result["px_per_mm"][0] / work_px_per_mm_known - 1)}
    result["px_per_mm"] = [float(v) for v in result["px_per_mm"]]
    result["origin_px"] = [float(v) for v in result["origin_px"]]
    return result

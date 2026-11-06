"""One detected contour - one reference contour.

Every detected contour can take at most one reference contour and the other way round, so extra copies of a contour
lower precision and one big contour cannot cover several reference contours. The assignment maximises the number of
pairs with IoU >= iou_min, then their total IoU (Hungarian algorithm)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MIN = 0.9


def iou(a, b):
    try:
        if not a.intersects(b):
            return 0.0
        inter = a.intersection(b).area
        union = a.union(b).area
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def iou_matrix(reference, detected):
    m = np.zeros((len(reference), len(detected)))
    for i, r in enumerate(reference):
        rx0, ry0, rx1, ry1 = r.bounds
        for j, d in enumerate(detected):
            dx0, dy0, dx1, dy1 = d.bounds
            if rx0 <= dx1 and dx0 <= rx1 and ry0 <= dy1 and dy0 <= ry1:
                m[i, j] = iou(r, d)
    return m


def match_contours(reference, detected, iou_min=IOU_MIN):
    """reference, detected: lists of shapely polygons. Returns counts, precision/recall and the pairs."""
    m = iou_matrix(reference, detected) if reference and detected else np.zeros((len(reference), len(detected)))
    pairs = []
    if m.size:
        allowed = m >= iou_min
        # a pair is always worth more than any IoU gain, so the number of pairs is maximal first
        weight = np.where(allowed, m + min(m.shape) + 1, 0.0)
        for i, j in zip(*linear_sum_assignment(weight, maximize=True)):
            if allowed[i, j]:
                pairs.append((int(i), int(j), round(float(m[i, j]), 4)))
    used_ref = {i for i, _, _ in pairs}
    used_det = {j for _, j, _ in pairs}
    extra = [j for j in range(len(detected)) if j not in used_det]
    # an unmatched detected contour that fits an already matched reference contour is a copy of it
    copies = [j for j in extra if m.size and (m[:, j] >= iou_min).any()]
    tp = len(pairs)
    precision = tp / len(detected) if detected else None
    recall = tp / len(reference) if reference else None
    f1 = 2 * tp / (len(detected) + len(reference)) if detected or reference else None
    return {"reference_closed": len(reference), "detected_closed": len(detected), "found": tp,
            "missed": len(reference) - tp, "extra": len(extra), "copies": len(copies),
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4), "f1": None if f1 is None else round(f1, 4),
            # the former metric: any detected contour may cover any number of reference contours
            "found_many_to_one": int((m >= iou_min).any(axis=1).sum()) if m.size else 0,
            "pairs": pairs, "missed_ids": [i for i in range(len(reference)) if i not in used_ref], "extra_ids": extra}

"""Dimension reconstruction from arrowheads + extension lines + OCR text."""
import math
import re

import numpy as np

from ocr import DIAM, parse_dim_text
from postproc import angle, length


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def ink_cover(bw, cal, to_px, a, b, step=0.2):
    n = max(2, int(np.linalg.norm(b - a) / step))
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    px, py = to_px(cal, pts[:, 0], pts[:, 1])
    px = np.clip(np.round(px).astype(int), 0, bw.shape[1] - 1)
    py = np.clip(np.round(py).astype(int), 0, bw.shape[0] - 1)
    return float(bw[py, px].mean())


def pair_arrows(arrows, bw, cal, to_px):
    cands = []
    for i, a in enumerate(arrows):
        for j, b in enumerate(arrows):
            if j <= i or a.dir @ b.dir > -0.985:
                continue
            u = b.tip - a.tip
            L = np.linalg.norm(u)
            if L < 0.5 or L > 450:
                continue
            uh = u / L
            if abs(uh[0] * a.dir[1] - uh[1] * a.dir[0]) > 0.03:
                continue
            inside = a.dir @ uh < 0 < b.dir @ uh
            outside = a.dir @ uh > 0 > b.dir @ uh
            if inside:
                if not (a.crossed and b.crossed):
                    continue
                if ink_cover(bw, cal, to_px, a.tip + uh * (a.length + 0.3), b.tip - uh * (b.length + 0.3)) < 0.75 \
                        and L > a.length + b.length + 1:
                    continue
            elif outside:
                if L > 30:
                    continue
            else:
                continue
            cands.append((L, i, j, "inside" if inside else "outside"))
    cands.sort()
    used = set()
    pairs = []
    for L, i, j, kind in cands:
        if i in used or j in used:
            continue
        # no other arrow tip strictly between them on the same line
        a, b = arrows[i], arrows[j]
        uh = _unit(b.tip - a.tip)
        between = False
        for k, c in enumerate(arrows):
            if k in (i, j):
                continue
            s = (c.tip - a.tip) @ uh
            off = abs((c.tip - a.tip) @ np.array([-uh[1], uh[0]]))
            if 0.3 < s < L - 0.3 and off < 0.3 and kind == "inside":
                between = True
                break
        if between:
            continue
        used |= {i, j}
        pairs.append((i, j, kind))
    return pairs


def ext_line(tip, uh, lines, skip, max_len=200.0):
    """Extension line through `tip` perpendicular to the dimension direction `uh`.
    Returns (defpoint, overshoot_end, [(line index, s_from, s_to)] covered) or None."""
    n = np.array([-uh[1], uh[0]])
    best = None
    for k, e in enumerate(lines):
        if k in skip or e["type"] != "line" or length(e) < 1.0:
            continue
        d = _unit(e["q"] - e["p"])
        if abs(d @ uh) > 0.03:
            continue
        if abs((tip - e["p"]) @ np.array([-d[1], d[0]])) > 0.25:
            continue
        s0, s1 = sorted(((e["p"] - tip) @ n, (e["q"] - tip) @ n))
        if s0 - 0.6 > 0 or s1 + 0.6 < 0:
            continue
        best = best or []
        best.append((k, s0, s1))
    if not best:
        return None
    # direction towards the object = side with the longer extent
    lo = min(b[1] for b in best)
    hi = max(b[2] for b in best)
    side = 1.0 if hi >= -lo else -1.0
    # walk contiguous collinear pieces from the tip towards the object
    pieces = sorted([(k, s0 * side, s1 * side) if side > 0 else (k, -s1, -s0) for k, s0, s1 in best],
                    key=lambda t: t[1])
    reach = 0.0
    covered = []
    for k, a, b in pieces:
        if a > reach + 1.0:
            break
        covered.append((k, a, b))
        reach = max(reach, b)
        if reach > max_len:
            break
    over = max(0.0, -min(a for _, a, _ in covered))
    defpt = tip + n * side * reach
    return defpt, tip - n * side * over, covered, n * side


def axis_symmetric(p1, p2, axes):
    for ax in axes:
        d = _unit(ax["q"] - ax["p"])
        nrm = np.array([-d[1], d[0]])
        o1, o2 = (p1 - ax["p"]) @ nrm, (p2 - ax["p"]) @ nrm
        if o1 * o2 < 0 and abs(o1 + o2) < 0.4 and abs(abs(o1) - abs(o2)) < 0.4:
            return True
    return False


def pick_text(texts, used, p1, p2, uh, max_perp=11.0, extra=40.0, check_orient=True):
    n = np.array([-uh[1], uh[0]])
    L = np.linalg.norm(p2 - p1)
    best, bs = None, 1e9
    dim_ang = math.degrees(math.atan2(uh[1], uh[0])) % 180
    for k, t in enumerate(texts):
        if k in used or not re.search(r"\d", t.text):
            continue
        c = t.center
        s = (c - p1) @ uh
        h = abs((c - p1) @ n)
        if not (-extra <= s <= L + extra) or not (0.3 <= h <= max_perp):
            continue
        ta = t.angle % 180
        if check_orient and min(abs(ta - dim_ang), 180 - abs(ta - dim_ang)) > 25 and len(t.text) > 1:
            continue
        outside = max(0.0, -s, s - L)
        score = h + 0.5 * outside
        _, v, _ = parse_dim_text(t.text)
        digits = re.match(r"^[06$Ø]?(\d+(?:[.,]\d+)?)", t.text.strip().lstrip("/"))
        vals = [x for x in (v, float(digits.group(1).replace(",", ".")) if digits else None) if x is not None]
        if vals and min(abs(x - L) for x in vals) > max(0.4, 0.02 * L) and view_scale(L, vals) is None:
            continue
        if score < bs:
            best, bs = k, score
    return best


SCALES = (2.0, 2.5, 4.0, 5.0, 10.0, 0.5, 0.4, 0.25, 0.2, 0.1)


def view_scale(L, vals):
    """Detail views drawn at a standard enlargement (A(5:1)): measured/value is a GOST 2.302 scale."""
    for v in vals:
        if v and v > 0:
            for k in SCALES:
                if abs(L / v - k) < 0.02 * k:
                    return k
    return None


def dim_text(raw, L, symmetric):
    """-> (dxf text override or '<>', value used for snapping or None, view scale)."""
    prefix, v, rest = parse_dim_text(raw)
    t = raw.strip().lstrip("/")
    if v is None and re.match(r"^[+\-±]\s*\d", t):
        return fmt(round(L, 1)) + t.replace("±", "%%p"), None, 1.0
    k = view_scale(L, [v]) if v is not None and abs(v - L) > max(0.4, 0.02 * L) else None
    if k:
        pre = {"⌀": "%%c", "R": "R", "M": "M"}.get(prefix, "")
        return f"{pre}{fmt(v)}{rest}", None, k
    if v is not None and not prefix:
        digits = re.match(r"^([06$Ø])(\d+(?:[.,]\d+)?)", t)
        if digits and abs(float(digits.group(2).replace(",", ".")) - L) < max(0.4, 0.02 * L):
            prefix, v, rest = DIAM, float(digits.group(2).replace(",", ".")), t[digits.end():]
    if v is not None and symmetric and not prefix:
        prefix = DIAM
    if v is None or abs(v - L) > max(0.4, 0.02 * L):
        return t.replace(DIAM, "%%c").replace("°", "%%d").replace("±", "%%p"), None, 1.0
    pre = {"⌀": "%%c", "R": "R", "M": "M"}.get(prefix, "")
    rest = rest.replace("°", "%%d").replace("±", "%%p").replace("×", "x")
    return f"{pre}{fmt(v)}{rest}", v, 1.0


def fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def build(arrows, lines, texts, axes, bw, cal, to_px):
    used_lines = {}
    used_texts = set()
    out = []
    for i, j, kind in pair_arrows(arrows, bw, cal, to_px):
        a, b = arrows[i], arrows[j]
        uh = _unit(b.tip - a.tip)
        L = float(np.linalg.norm(b.tip - a.tip))
        e1 = ext_line(a.tip, uh, lines, set())
        e2 = ext_line(b.tip, uh, lines, set())
        p1 = e1[0] if e1 else a.tip
        p2 = e2[0] if e2 else b.tip
        for e in (e1, e2):
            if e:
                for k, s0, s1 in e[2]:
                    used_lines.setdefault(k, []).append((e[1], e[3]))
        k = pick_text(texts, used_texts, a.tip, b.tip, uh, check_orient=(kind == "inside"))
        sym = axis_symmetric(p1, p2, axes)
        if k is not None:
            used_texts.add(k)
            txt, val, vscale = dim_text(texts[k].text, L, sym)
            loc = texts[k].center
        else:
            txt, val, loc, vscale = ("%%c<>" if sym else "<>"), None, None, 1.0
        a.used = b.used = True
        out.append({"kind": "linear", "p1": p1, "p2": p2, "base": a.tip, "angle": math.degrees(math.atan2(uh[1], uh[0])),
                    "text": txt, "value": val, "loc": loc, "L": L, "ext": (e1, e2), "arrows": (i, j),
                    "scale": vscale})
    return out, used_lines, used_texts


def radius_dims(arrows, arcs, texts, used_texts, ocr_window=None):
    out = []
    for i, a in enumerate(arrows):
        if a.used:
            continue
        best = None
        for e in arcs:
            r = np.linalg.norm(a.tip - e["c"])
            if abs(r - e["r"]) > max(0.35, 0.08 * e["r"]) or e["r"] < 0.8:
                continue
            radial = _unit(a.tip - e["c"])
            if abs(radial[0] * a.dir[1] - radial[1] * a.dir[0]) > 0.35:
                continue
            if best is None or abs(r - e["r"]) < abs(best[0] - best[1]["r"]):
                best = (r, e)
        if best is None:
            continue
        e = best[1]
        k, kd = None, 1e9
        for t_i, t in enumerate(texts):
            if t_i in used_texts or not t.text.strip().upper().startswith("R"):
                continue
            d = np.linalg.norm(t.center - a.base)
            if d < kd and d < 40:
                k, kd = t_i, d
        if k is None and ocr_window is not None:
            b = a.base - a.dir * 6
            for it in ocr_window((b[0] - 14, b[1] - 9, b[0] + 14, b[1] + 9)):
                if re.match(r"^[RrЯ]\s*\d", it.text):
                    it.text = "R" + it.text[1:].strip()
                    texts.append(it)
                    k = len(texts) - 1
                    break
        if k is None:
            continue
        used_texts.add(k)
        _, v, rest = parse_dim_text(texts[k].text)
        txt = "R" + fmt(v) + rest if v is not None else "<>"
        loc = texts[k].center
        a.used = True
        ang = math.degrees(math.atan2(a.tip[1] - e["c"][1], a.tip[0] - e["c"][0]))
        out.append({"kind": "radius", "c": e["c"], "r": e["r"], "angle": ang, "text": txt, "loc": loc, "arrow": i})
    return out


def angular_dims(arrows, arcs, texts, used_texts):
    out = []
    free = [i for i, a in enumerate(arrows) if not a.used]
    for e in arcs:
        if e["r"] < 5:
            continue
        on = [i for i in free if abs(np.linalg.norm(arrows[i].tip - e["c"]) - e["r"]) < 0.5
              and abs(_unit(arrows[i].tip - e["c"]) @ arrows[i].dir) < 0.3]
        if len(on) != 2:
            continue
        a, b = arrows[on[0]], arrows[on[1]]
        mid = (a.tip + b.tip) / 2
        k, kd = None, 1e9
        for t_i, t in enumerate(texts):
            if t_i in used_texts or "°" not in t.text and not re.fullmatch(r"\W*\d+\W*", t.text):
                continue
            d = np.linalg.norm(t.center - mid)
            if d < kd and d < 20:
                k, kd = t_i, d
        a1 = math.degrees(math.atan2(a.tip[1] - e["c"][1], a.tip[0] - e["c"][0]))
        a2 = math.degrees(math.atan2(b.tip[1] - e["c"][1], b.tip[0] - e["c"][0]))
        meas = (a2 - a1) % 360
        meas = min(meas, 360 - meas)
        std = 5 * round(meas / 5)
        if k is None:
            continue
        txt, loc = (f"{std}%%d" if abs(std - meas) < 1.5 else "<>"), None
        if k is not None:
            used_texts.add(k)
            m = re.search(r"\d+", texts[k].text)
            if m and abs(float(m.group(0)) - meas) < 2:
                txt = m.group(0) + "%%d"
            loc = texts[k].center
        a.used = b.used = True
        free = [i for i in free if i not in on]
        out.append({"kind": "angular", "c": e["c"], "r": e["r"], "t1": a.tip, "t2": b.tip, "text": txt, "loc": loc})
    return out


def leaders(arrows, lines, used_lines):
    out = []
    for i, a in enumerate(arrows):
        if a.used:
            continue
        pts = [a.tip]
        cur, back = a.base, -a.dir
        for _ in range(3):
            nxt = None
            for k, e in enumerate(lines):
                if e["type"] != "line" or k in used_lines:
                    continue
                for s, t in (("p", "q"), ("q", "p")):
                    if np.linalg.norm(e[s] - cur) < 0.8 and (len(pts) > 1 or _unit(e[t] - e[s]) @ back > 0.95):
                        nxt = (k, e[t])
                        break
                if nxt:
                    break
            if not nxt:
                break
            used_lines[nxt[0]] = [("leader",)]
            pts.append(nxt[1])
            cur = nxt[1]
        if len(pts) >= 2:
            a.used = True
            out.append({"kind": "leader", "pts": pts})
    return out

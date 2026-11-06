"""Quality on three independent working sheets, per kind, with the causes of the errors.

    python mlcv/eval/three_sheets.py [--seed 2026] [--out mlcv/eval/three_sheets]

The sheets are chosen by a rule, not by hand: real sheets with a reference DXF, from families of parts never used to
develop, tune or measure the module (development sheets, sheets of earlier reports, random sets of seeds 44, 7, 101),
whose reference has lines, circles and arcs, one per family, from a list shuffled with a fixed seed. A sheet whose
reference does not fit the page (fewer than 20 primitives inside the frame after registration, or only a piece of the
sheet) is skipped and listed in the report. Written per sheet: overlay.jpg (found geometry over the source page), settings.json (parameters, time,
models), metrics.json."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import ezdxf
from ezdxf import bbox
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mlcv" / "eval"))

import evaluate as ev  # noqa: E402
from contour_match import iou  # noqa: E402
from drawcv import Params, __version__  # noqa: E402
from drawcv.ingest import load_page  # noqa: E402
from drawcv.transform import Transform  # noqa: E402

CAUSES = {"text": "текст", "hatch": "штриховка", "break": "разрыв", "scale": "неверный масштаб",
          "kind": "другой вид примитива", "excluded": "отброшен модулем", "short": "короткий обломок",
          "copy": "копия контура", "frame": "рамка и штамп", "dims": "графика размеров", "lost": "не найден", "other": "прочее"}
MIN_BODY = 20
MIN_COVERAGE = 0.25
# Parts already measured in earlier reports: not independent any more.
MEASURED = {"Вал", "Вал12", "Вал4", "Вал9", "Корпус7", "Кран", "Опора6", "Опора8", "Цилиндр", "втулка", "втулка4", "гайка", "дно",
            "звездочка", "корпус клапана", "корпус13", "ниппель", "ниппель шаровой2", "перегородка", "стойка", "стойка2",
            "стойка3", "фланец4"}
KINDS = ("line", "circle", "arc")
KIND_RU = {"line": "отрезки", "circle": "окружности", "arc": "дуги", "contour": "контуры"}


def family(name):
    """Sheets of one assembly or type of part count as one family."""
    number = re.match(r"\d{5}", name)
    if number:
        return number.group()
    return re.sub(r"[\d_.\-\s]+$", "", re.sub(r"\s[-–].*$", "", name)).strip().lower() or name.lower()


def candidates(seed, tuned_seeds=(44, 7, 101), tuned_n=8):
    """Suitable sheets in a seeded random order, one per family (a skipped sheet uses its family up too)."""
    import random

    tuned = set(ev.DEV_SHEETS) | MEASURED
    for s in tuned_seeds:
        tuned |= {name for name, _ in ev.random_sheets(tuned_n, s)}
    families = {family(name) for name in tuned}
    names = sorted(p.stem for p in ev.PNG.glob("*.png") if p.stem not in tuned and (ev.DXF / f"{p.stem}.dxf").exists())
    random.Random(seed).shuffle(names)
    for name in names:
        if family(name) in families:
            continue
        try:
            geo = ev.reference(ev.DXF / f"{name}.dxf")[0]
            msp = ezdxf.readfile(str(ev.DXF / f"{name}.dxf")).modelspace()
            ext = bbox.extents(e for e in ev.iter_flat(msp) if e.dxftype() in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE"))
        except Exception:
            continue
        if not ext.has_data or ext.size.x > 1250 or ext.size.y > 900:
            continue
        counts = {k: sum(g["kind"] == k and g["cls"] == "MAIN" for g in geo) for k in KINDS}
        if counts["line"] < 10 or counts["circle"] < 1 or counts["arc"] < 2:
            continue
        families.add(family(name))
        yield name


# ------------------------------------------------------------------ cause of each error

class Places:
    """Where text, hatch and gaps of this sheet are, in reference coordinates."""

    def __init__(self, cv, t):
        polys = [Polygon(np.array(r["poly_mm"]) + t).buffer(0.5) for r in cv["text_regions"] if len(r["poly_mm"]) >= 3]
        self.text = prep(unary_union(polys)) if polys else None
        hulls = []
        for g in cv["hatch_groups"]:
            pts = []
            for i in g["line_ids"]:
                geo = cv["objects"][i]["geometry_mm"]
                pts += [np.array(geo["start"]) + t, np.array(geo["end"]) + t]
            hulls.append(MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(1.0))
        self.hatch = prep(unary_union(hulls)) if hulls else None
        self.gaps = [np.array(d["at_mm"]) + t for d in cv["defects"] if d["code"] in ("gap", "open_end")]
        self.excluded = []
        for oid, o in cv["objects"].items():
            if o["status"] not in ("ok", "converted"):
                self.excluded.append(ev.shift(_prim(o), t))
        self.ex_pts = ev.samples_of(self.excluded)[0] if self.excluded else None
        self.roles = {oid: o["role"] for oid, o in cv["objects"].items()}

    def share(self, region, pts):
        if region is None:
            return 0.0
        return float(np.mean([region.contains(Point(float(x), float(y))) for x, y in pts[:: max(1, len(pts) // 60)]]))


def _prim(o):
    g = o["geometry_mm"]
    if o["kind"] == "line":
        return ev._prim("line", p=g["start"], q=g["end"])
    if o["kind"] == "circle":
        return ev._prim("circle", c=g["center"], r=g["radius"])
    return ev._prim("arc", c=g["center"], r=g["radius"], a0=g["start_angle"], a1=g["end_angle"])


def miss_cause(e, row, places, det_tree, scale_bad):
    if scale_bad:
        return "scale"
    pts = ev.sample(e, 0.25)
    if places.share(places.text, pts) >= 0.3:
        return "text"
    if places.share(places.hatch, pts) >= 0.3:
        return "hatch"
    if row["covered"]:
        return "kind"
    if det_tree is not None:
        covered = float(np.mean(det_tree.query(pts)[0] <= ev.TOL))
        if covered >= 0.2:
            return "break"
    if places.ex_pts is not None and len(places.ex_pts):
        from scipy.spatial import cKDTree
        if float(np.mean(cKDTree(places.ex_pts).query(pts)[0] <= ev.TOL)) >= 0.5:
            return "excluded"
    return "lost"


def extra_cause(d, row, places, scale_bad, annot_tree):
    if scale_bad:
        return "scale"
    if places.roles.get(d["id"]) in ("frame", "title_block"):
        return "frame"
    pts = ev.sample(d, 0.25)
    if places.share(places.text, pts) >= 0.5:
        return "text"
    if places.share(places.hatch, pts) >= 0.5:
        return "hatch"
    if annot_tree is not None and float(np.mean(annot_tree.query(pts)[0] <= ev.TOL)) >= 0.3:
        return "dims"                                    # part of a dimension or extension line, merged or longer
    if row["short"]:
        return "short"
    return "other"


def contour_causes(a, places, scale_bad):
    """Missed reference contours and extra detected contours with causes."""
    res = {"missed": {}, "extra": {}}
    det = a["contours"]["detected"]
    ref = a["contours"]["reference"]
    for i in a["contours"]["missed_ids"]:
        if scale_bad:
            cause = "scale"
        else:
            boundary = ref[i].boundary
            if any(boundary.distance(Point(*g)) <= 1.5 for g in places.gaps):
                cause = "break"
            elif places.share(places.text, np.array(boundary.coords)) >= 0.2:
                cause = "text"
            elif places.share(places.hatch, np.array(boundary.coords)) >= 0.5:
                cause = "hatch"
            else:
                cause = "lost"
        res["missed"][cause] = res["missed"].get(cause, 0) + 1
    matched_refs = [ref[i] for i, _, _ in a["contours"]["pairs"]]
    for j in a["contours"]["extra_ids"]:
        poly = det[j]
        if any(iou(poly, r) >= 0.9 for r in matched_refs):
            cause = "copy"
        elif places.hatch is not None and places.share(places.hatch, np.array(poly.exterior.coords)) >= 0.8:
            cause = "hatch"
        elif places.share(places.text, np.array(poly.exterior.coords)) >= 0.5:
            cause = "text"
        else:
            cause = "other"
        res["extra"][cause] = res["extra"].get(cause, 0) + 1
    return res


def reference_coverage(a):
    """Bounding area of the reference body against that of the found part geometry: a reference holding only a piece of
    the sheet cannot judge the module."""
    ref = [ev.sample(e, 1.0) for e, b in zip(a["geo"], a["body"]) if b]
    det = [ev.sample(d, 1.0) for d in a["det"] if d["role"] == "part"]
    if not ref or not det:
        return 0.0

    def area(parts):
        pts = np.vstack(parts)
        return float(np.prod(pts.max(axis=0) - pts.min(axis=0)))

    return area(ref) / max(area(det), 1e-6)


def stat(vals):
    vals = [abs(v) for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": round(float(np.mean(vals)), 3), "p95": round(float(np.percentile(vals, 95)), 3),
            "max": round(float(np.max(vals)), 3)}


def analyse(a):
    """Per-kind table of one sheet."""
    cv, geo, det, rows, det_rows, body, t = (a["cv"], a["geo"], a["det"], a["rows"], a["det_rows"], a["body"], a["translation"])
    m = a["metrics"]
    scale = m["registration"]["residual_scale"]
    scale_bad = cv["scale_status"] != "known" or (scale is not None and abs(scale - 1) > 0.002)
    places = Places(cv, t)
    det_pts = ev.samples_of(det)[0]
    from scipy.spatial import cKDTree
    det_tree = cKDTree(det_pts) if len(det_pts) else None
    annot_pts = ev.samples_of(a["annot"])[0]
    annot_tree = cKDTree(annot_pts) if len(annot_pts) else None
    kinds = {}
    for kind in KINDS:
        idx = [i for i, (e, b) in enumerate(zip(geo, body)) if e["kind"] == kind and b]
        found = [i for i in idx if rows[i]["found"]]
        causes = {}
        for i in idx:
            if not rows[i]["found"]:
                c = miss_cause(geo[i], rows[i], places, det_tree, scale_bad)
                causes[c] = causes.get(c, 0) + 1
        dets = [(d, r) for d, r in zip(det, det_rows) if d["kind"] == kind]
        extra = [(d, r) for d, r in dets if r["category"] == "extra"]
        extra_causes = {}
        for d, r in extra:
            c = extra_cause(d, r, places, scale_bad, annot_tree)
            extra_causes[c] = extra_causes.get(c, 0) + 1
        errors = {}
        if kind == "line":
            errors["конец"] = stat([rows[i].get("end_err") for i in found])
        else:
            errors["центр"] = stat([rows[i].get("center_err") for i in found])
            errors["радиус"] = stat([rows[i].get("radius_err") for i in found])
        kinds[kind] = {"reference": len(idx), "found": len(found), "missed": len(idx) - len(found), "missed_causes": causes,
                       "detected": len(dets), "on_geometry": sum(r["category"] == "geometry" for _, r in dets),
                       "not_part_graphics": sum(r["category"] in ("dimension_graphics", "hatch_pattern", "text_strokes") for _, r in dets),
                       "extra": len(extra), "extra_causes": extra_causes, "duplicates": sum(r["duplicate"] for _, r in dets),
                       "errors_mm": errors}
    c = m["contours"]
    causes = contour_causes(a, places, scale_bad)
    kinds["contour"] = {"reference": c["reference_closed"], "found": c["found"], "missed": c["missed"],
                        "missed_causes": causes["missed"], "detected": c["detected_closed"], "extra": c["extra"],
                        "extra_causes": causes["extra"], "copies": c["copies"], "precision": c["precision"], "recall": c["recall"]}
    return {"scale": {"status": cv["scale_status"], "method": cv["calibration"]["method"], "residual": scale,
                      "px_per_mm": cv["calibration"]["px_per_mm"], "error_percent": None if scale is None else round(abs(scale - 1) * 100, 3)},
            "kinds": kinds, "backend": m["backend"], "closure": cv["summary"]["closure"], "defects": cv["summary"]["defects"]}


# ------------------------------------------------------------------ overlay on the source page

def overlay(a, source, path, max_side=3000):
    """Green: found and matched. Red: found, extra. Blue: reference missed by the module."""
    page = load_page(source)
    f = min(1.0, max_side / max(page.gray.shape))
    base = cv2.resize(page.gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else page.gray
    img = cv2.cvtColor((base * 0.35 + 165).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    to_src = Transform(a["cv"]["transforms"]["sheet_mm_to_source_px"])
    t = a["translation"]

    def draw(e, color, thick, shift_back=True):
        pts = ev.sample(e, 0.3) - (t if shift_back else 0)
        px = to_src.apply(pts) * f
        cv2.polylines(img, [np.round(px * 16).astype(np.int32).reshape(-1, 1, 2)], e["kind"] == "circle", color, thick, cv2.LINE_AA, shift=4)

    for e, r, b in zip(a["geo"], a["rows"], a["body"]):
        if b and not r["found"]:
            draw(e, (220, 80, 0), 2)
    colors = {"geometry": (40, 150, 40), "extra": (20, 20, 230), "dimension_graphics": (200, 170, 0),
              "hatch_pattern": (0, 170, 170), "text_strokes": (150, 150, 150)}
    for d, r in zip(a["det"], a["det_rows"]):
        draw(d, colors[r["category"]], 1)
    y = 10
    for name, c in (("found on reference", colors["geometry"]), ("extra", colors["extra"]), ("reference missed", (220, 80, 0)),
                    ("dimension graphics", colors["dimension_graphics"]), ("hatch", colors["hatch_pattern"])):
        cv2.rectangle(img, (6, y - 4), (230, y + 18), (255, 255, 255), -1)
        cv2.line(img, (10, y + 7), (40, y + 7), c, 3)
        cv2.putText(img, name, (48, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(str(path))


# ------------------------------------------------------------------ report

def causes_text(c):
    return ", ".join(f"{CAUSES.get(k, k)} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) or "—"


def err_text(e):
    return "; ".join(f"{k} {v['mean']:.3f} / {v['p95']:.3f} / {v['max']:.3f}" for k, v in e.items() if v) or "—"


def sheet_section(name, r):
    a = r["analysis"]
    s = a["scale"]
    k = a["kinds"]
    lines = [f"### {name}", "",
             f"Масштаб: {s['status']}, способ `{s['method']}`, отличие от эталона {s['error_percent']} %. "
             f"Время: {r['time_s']} с (весь модуль), модель: {r['models']}. Наложение: [{r['dir']}/overlay.jpg]({r['dir']}/overlay.jpg), настройки: [{r['dir']}/settings.json]({r['dir']}/settings.json).", "",
             "| вид | эталон | найдено | пропущено | причины пропусков | выдано модулем | лишние | причины лишних | ошибка координат, мм: ср. / 95 % / макс. |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        v = k[kind]
        lines.append(f"| {KIND_RU[kind]} | {v['reference']} | {v['found']} | {v['missed']} | {causes_text(v['missed_causes'])} | {v['detected']} | {v['extra']} | "
                     f"{causes_text(v['extra_causes'])} | {err_text(v['errors_mm'])} |")
    v = k["contour"]
    lines.append(f"| контуры (1:1) | {v['reference']} | {v['found']} | {v['missed']} | {causes_text(v['missed_causes'])} | {v['detected']} | {v['extra']} | "
                 f"{causes_text(v['extra_causes'])} | — |")
    b = a["backend"]
    lines += ["", f"Выдано модулем сверх геометрии детали (графика размеров, штриховка, штрихи текста): "
                  f"{sum(k[x]['not_part_graphics'] for x in KINDS)}. Бэкенд принял {b['accepted']} из {b['closed']} замкнутых контуров. "
                  f"Стыков соединено {a['closure']['joined']}, отказов {a['closure']['refused']}.", ""]
    return "\n".join(lines)


def totals(results):
    keys = ["scale", "text", "hatch", "break", "kind", "excluded", "frame", "dims", "short", "copy", "lost", "other"]
    rows = ["| причина | пропуски: отрезки | окружности | дуги | контуры | лишние: отрезки | окружности | дуги | контуры |",
            "|---|---|---|---|---|---|---|---|---|"]
    for cause in keys:
        vals = []
        for field in ("missed_causes", "extra_causes"):
            for kind in (*KINDS, "contour"):
                vals.append(sum(r["analysis"]["kinds"][kind][field].get(cause, 0) for r in results))
        if any(vals):
            rows.append(f"| {CAUSES[cause]} | " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "mlcv/eval/three_sheets"))
    ap.add_argument("--sheets", nargs="*", help="names instead of the rule")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "mlcv/eval/out_three_sheets"
    results, skipped = [], []
    for name in args.sheets or candidates(args.seed):
        if len(results) == args.n:
            break
        k = len(results) + 1
        case = {"case": f"{name}.png", "sheet": name, "format": "png", "input": ev.PNG / f"{name}.png", "reference": ev.DXF / f"{name}.dxf"}
        t0 = time.perf_counter()
        a = ev.analyse_case(case, work)
        wall = time.perf_counter() - t0
        body_count = a["metrics"]["reference"]["body_primitives"]
        coverage = reference_coverage(a)
        if not args.sheets and body_count < MIN_BODY:
            skipped.append(f"{name} (в рамке после совмещения только {body_count} примитивов эталона)")
            continue
        if not args.sheets and coverage < MIN_COVERAGE:
            skipped.append(f"{name} (эталон занимает {coverage:.0%} площади найденной геометрии: в DXF только часть листа)")
            continue
        sheet_dir = out / f"sheet{k}"
        sheet_dir.mkdir(exist_ok=True)
        overlay(a, case["input"], sheet_dir / "overlay.jpg")
        models = a["cv"]["stats"]["models"]
        (sheet_dir / "settings.json").write_text(json.dumps({
            "sheet": name, "module": {"name": "drawcv", "version": __version__}, "params": Params().to_dict(),
            "text_provider": a["cv"]["stats"]["text_provider"], "models": models, "tolerances_mm": a["cv"]["stats"]["tolerances_mm"],
            "calibration": a["cv"]["calibration"], "timings_s": a["cv"]["timings_s"], "evaluation_wall_s": round(wall, 2),
            "source_size_px": a["cv"]["source"]["size_px"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        analysis = analyse(a)
        (sheet_dir / "metrics.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
        results.append({"name": name, "dir": sheet_dir.name, "analysis": analysis, "time_s": a["cv"]["timings_s"]["total"],
                        "models": ", ".join(f"{m['name']} (`{m.get('sha256', '')[:12]}`)" for m in models) or "нет"})
        print(name, a["metrics"]["found"]["body"], "/", a["metrics"]["reference"]["body_primitives"], flush=True)
    body = ["# Качество на трёх независимых листах", "",
            f"Совместный проект с 44labs. Листы выбраны правилом (`three_sheets.py`, seed {args.seed}): не участвовали в разработке и настройке, "
            "семейства деталей не встречались в прежних отчётах, у эталона есть отрезки, окружности и дуги. Эталон — DXF листа. "
            + (f"Пропущены как несравнимые: {'; '.join(skipped)}. " if skipped else "Несравнимых листов не встретилось. "), "",
            "Найдено: эталонный примитив на ≥ 80 % длины покрыт примитивом того же вида в пределах 0,5 мм. Контуры сравниваются один к одному при IoU ≥ 0,9.", "",
            "## Сводка причин по трём листам", "", totals(results), "", "## По листам", ""]
    for r in results:
        body.append(sheet_section(r["name"], r))
    (out / "REPORT.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()

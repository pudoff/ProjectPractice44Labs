"""Does the whole chain work with a perfect OCR reader? CV -> mlocr.to_drawing -> backend assemble_graph -> DxfExporter.

    python mlcv/eval/ocr_link_check.py DIR [DIR ...]     # DIR is an evaluate.py output (cv.json + metrics.json)
    OCR_RAW=mlocr/evaluation/out/raw python mlcv/eval/ocr_link_check.py DIR ...     # with real OCR readings

By default the readings come from the linear DIMENSION entities of the reference DXF: value and text position are exact,
so every dimension that fails to link fails because of the CV points or the linking rules, not because of OCR. With OCR_RAW
the saved readings of mlocr/evaluation/collect_observations.py are used. It also tells how many measured points of the
reference coincide with a point of cv.json (within 0.25 and 0.5 mm)."""
import collections
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import ezdxf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "mlcv"), str(ROOT / "mlcv" / "eval")]
DXF = ROOT / "dataset/clear_sum_dataset/vse_dxf"
_tmp = Path(tempfile.mkdtemp(prefix="drawcv-link-check-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{(_tmp / 'check.sqlite3').as_posix()}")   # as backend tests: no real database
os.environ.setdefault("STORAGE_ROOT", str(_tmp / "storage"))
os.environ.setdefault("REQUIRE_LOCAL_MODELS", "false")


def reference_dimensions(path):
    doc = ezdxf.readfile(str(path))
    out = []
    for e in doc.modelspace().query("DIMENSION"):
        if (e.dimtype & 7) not in (0, 1):                 # rotated / aligned linear
            continue
        try:
            p1, p2 = np.array(e.dxf.defpoint2)[:2], np.array(e.dxf.defpoint3)[:2]
            value = float(e.get_measurement())
            style = e.get_acad_dstyle(doc.dimstyles.get(e.dxf.dimstyle))
            factor = float(style.get("dimlfac", 1) or 1)
            mid = np.array(e.dxf.text_midpoint)[:2]
        except Exception:
            continue
        if value > 0:
            printed = str(e.dxf.get("text", "") or "")
            diameter = any(k in printed for k in ("%%c", "%%C", "Ø", "⌀", "∅", "\\U+2205", "\\U+2300"))
            angle = math.radians(float(e.dxf.get("angle", 0.0))) if (e.dimtype & 7) == 0 else math.atan2(*(p2 - p1)[::-1])
            out.append({"p1": p1, "p2": p2, "nominal": round(value * factor, 3), "factor": factor, "text_at": mid,
                        "diameter": diameter, "u": np.array([math.cos(angle), math.sin(angle)])})
    return out


def observations(cv, dims, translation):
    from mlocr.src.normalization import enrich

    t = np.array(cv["transforms"]["sheet_mm_to_source_px"], float)
    px_per_mm = abs(t[0, 0])
    texts = []
    for k, d in enumerate(dims):
        centre = d["text_at"] - translation
        x, y = t[:2, :2] @ centre + t[:2, 2]
        size = max(3.0, 1.8 * len(f"{d['nominal']:g}")) * px_per_mm      # half length of the box, along the text
        along = t[:2, :2] @ d["u"]
        along = along / np.linalg.norm(along) * size
        across = np.array([-along[1], along[0]]) / size * 1.5 * px_per_mm
        value = ("Ø" if d["diameter"] else "") + f"{d['nominal']:g}".replace(".", ",")
        texts.append(enrich({"id": f"t{k}", "text": value, "score": 1.0, "purpose": "dimension_or_scale_candidate",
                             "poly_source_px": [[float(v) for v in np.array([x, y]) + a * along + b * across]
                                                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]}))
    return {"status": "ok", "source": cv["raster_identity"], "texts": texts, "issues": [], "title_block": {}}


def real_observations(path, cv):
    """Saved readings, normalised by the current rules, on the raster of this cv.json."""
    from mlocr.src.normalization import enrich

    raw = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    for source in raw["texts"]:
        item = copy.deepcopy(source)
        item["text"] = item.get("raw_text", item["text"])
        had_recovery = any(c.get("rule") == "diameter_visible_in_raster" for c in source.get("corrections", []))
        enrich(item)
        if had_recovery and not item["text"].startswith("Ø"):       # pixels are not read again here
            item["text"] = source["text"]
            item["notation"] = source["notation"]
        texts.append(item)
    return {"status": "ok", "source": cv["raster_identity"], "texts": texts, "issues": [], "title_block": raw.get("title_block", {})}


def check_case(case_dir):
    from app.services.dxf import DxfExporter
    from app.services.recognition import assemble_graph
    from mlocr.src.adapter import to_drawing

    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((case_dir / "cv.json").read_text(encoding="utf-8"))
    if not cv.get("raster_identity"):
        return None                                      # PDF inputs: the backend always passes a prepared PNG
    translation = np.array(metrics["registration"]["translation_mm"])
    dims = reference_dimensions(DXF / f"{metrics['sheet']}.dxf")
    res = {"case": metrics["case"], "reference_dims": len(dims), "points": len(cv["points"])}
    if not dims:
        return res
    pts = np.array([p["xy"] for p in cv["points"]]) + translation
    ends = np.array([p for d in dims for p in (d["p1"], d["p2"])])
    dist = np.min(np.linalg.norm(ends[:, None] - pts[None], axis=2), axis=1)
    res.update(ends_within_025=round(float(np.mean(dist <= 0.25)), 3), ends_within_05=round(float(np.mean(dist <= 0.5)), 3))
    factors = {round(d["factor"], 6) for d in dims}
    cv = copy.deepcopy(cv)
    raw_dir = os.getenv("OCR_RAW")
    if raw_dir:
        raw_path = Path(raw_dir) / f"{metrics['sheet']}.json"
        if not raw_path.exists():
            return None
        observed = real_observations(raw_path, cv)
    else:
        cv.setdefault("calibration", {})["view_scale"] = 1 / max(factors)   # one scale, as to_drawing assumes in the MVP
        observed = observations(cv, dims, translation)
    ocr = to_drawing(observed, cv)
    res["view_scale"] = cv.get("calibration", {}).get("view_scale")
    res["view_scale_basis"] = cv.get("calibration", {}).get("view_scale_basis")
    res["limit_hit"] = any(i["code"] == "dimension_link_limit" for i in ocr["issues"])
    links = ocr.get("dimension_links", [])
    res["links"] = len(links)
    res["status"] = dict(collections.Counter(link["status"] for link in links))
    res["search"] = dict(collections.Counter(link.get("dimension_line_search") or link.get("decision_note") or "linked" for link in links if not link.get("confirmed")))
    res["dimensions"] = len(ocr["dimensions"])
    correct = 0
    by_id = {p["id"]: p for p in cv["points"]}
    for d in ocr["dimensions"]:
        a, b = by_id[d["p1_ref"]]["xy"], by_id[d["p2_ref"]]["xy"]
        a, b = np.array(a) + translation, np.array(b) + translation
        for ref in dims:
            # a dimension measures along its direction: the feet may lie anywhere on the extension lines
            u = ref["u"]
            direct = max(abs((a - ref["p1"]) @ u), abs((b - ref["p2"]) @ u))
            swapped = max(abs((a - ref["p2"]) @ u), abs((b - ref["p1"]) @ u))
            parallel = abs(u @ np.array([math.cos(math.radians(d["angle"])), math.sin(math.radians(d["angle"]))])) > 0.99
            if parallel and min(direct, swapped) <= 0.5:
                correct += 1
                break
    res["dimensions_on_reference_points"] = correct
    graph = assemble_graph(cv, ocr, "check")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.dxf"
        issues = DxfExporter().export(graph, path)
        res["dxf_written"] = path.exists()
        if path.exists():
            res["dxf_dimensions"] = len(ezdxf.readfile(path).modelspace().query("DIMENSION"))
    codes = collections.Counter(i.code for i in issues if i.severity == "error")
    res["errors"] = dict(codes)
    return res


def main():
    rows, total = [], collections.Counter()
    for root in sys.argv[1:]:
        for case in sorted(Path(root).glob("*/metrics.json")):
            if json.loads(case.read_text(encoding="utf-8")).get("sheet") == "control":
                continue
            r = check_case(case.parent)
            if r is None or not r["reference_dims"]:
                continue
            rows.append(r)
            total.update(sheets=1, reference_dims=r["reference_dims"], dimensions=r["dimensions"],
                         on_reference=r["dimensions_on_reference_points"], limit_hit=r["limit_hit"],
                         dxf_written=r["dxf_written"], dxf_dimensions=r.get("dxf_dimensions", 0))
            print(f"{r['case'][:34]:34} dims {r['reference_dims']:3} linked {r['dimensions']:3} on-ref {r['dimensions_on_reference_points']:3} "
                  f"pts {r['points']:4} ends<=.25 {r['ends_within_025']:.0%} <=.5 {r['ends_within_05']:.0%} limit {r['limit_hit']} {r['status']} dxf {r['dxf_written']} {r['errors']}", flush=True)
    print(dict(total))
    # sheets whose reference fits the page: elsewhere a right dimension has nothing to be compared with
    fit = [r for r in rows if r["ends_within_05"] >= 0.2]
    print("reference fits:", {"sheets": len(fit), "reference_dims": sum(r["reference_dims"] for r in fit),
                              "linked": sum(r["dimensions"] for r in fit), "on_reference": sum(r["dimensions_on_reference_points"] for r in fit)})
    print(dict(sum((collections.Counter(r.get("search", {})) for r in rows), collections.Counter())))
    if os.getenv("OCR_LINK_JSON"):
        Path(os.environ["OCR_LINK_JSON"]).write_text(json.dumps({"total": total, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()

"""Does the whole chain work with a perfect OCR reader? CV -> mlocr.to_drawing -> backend assemble_graph -> DxfExporter.

    python mlcv/eval/ocr_link_check.py DIR [DIR ...]     # DIR is an evaluate.py output (cv.json + metrics.json)

The readings come from the linear DIMENSION entities of the reference DXF: value and text position are exact, so every
dimension that fails to link fails because of the CV points or the linking rules, not because of OCR. It also tells
how many measured points of the reference coincide with a point of cv.json (within 0.25 and 0.5 mm)."""
import collections
import copy
import json
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
            out.append({"p1": p1, "p2": p2, "nominal": round(value * factor, 3), "factor": factor, "text_at": mid})
    return out


def observations(cv, dims, translation):
    from mlocr.src.normalization import enrich

    t = np.array(cv["transforms"]["sheet_mm_to_source_px"], float)
    px_per_mm = abs(t[0, 0])
    texts = []
    for k, d in enumerate(dims):
        centre = d["text_at"] - translation
        x, y = t[:2, :2] @ centre + t[:2, 2]
        w, h = 6 * px_per_mm, 1.5 * px_per_mm
        value = f"{d['nominal']:g}".replace(".", ",")
        texts.append(enrich({"id": f"t{k}", "text": value, "score": 1.0, "purpose": "dimension_or_scale_candidate",
                             "poly_source_px": [[x - w, y - h], [x + w, y - h], [x + w, y + h], [x - w, y + h]]}))
    return {"status": "ok", "source": cv["raster_identity"], "texts": texts, "issues": [], "title_block": {}}


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
    cv.setdefault("calibration", {})["view_scale"] = 1 / max(factors)   # one scale, as to_drawing assumes in the MVP
    ocr = to_drawing(observations(cv, dims, translation), cv)
    res["limit_hit"] = any(i["code"] == "dimension_link_limit" for i in ocr["issues"])
    links = ocr.get("dimension_links", [])
    res["links"] = len(links)
    res["status"] = dict(collections.Counter(link["status"] for link in links))
    res["dimensions"] = len(ocr["dimensions"])
    correct = 0
    by_id = {p["id"]: p for p in cv["points"]}
    for d in ocr["dimensions"]:
        a, b = by_id[d["p1_ref"]]["xy"], by_id[d["p2_ref"]]["xy"]
        a, b = np.array(a) + translation, np.array(b) + translation
        for ref in dims:
            direct = max(np.linalg.norm(a - ref["p1"]), np.linalg.norm(b - ref["p2"]))
            swapped = max(np.linalg.norm(a - ref["p2"]), np.linalg.norm(b - ref["p1"]))
            if min(direct, swapped) <= 0.5:
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
    if os.getenv("OCR_LINK_JSON"):
        Path(os.environ["OCR_LINK_JSON"]).write_text(json.dumps({"total": total, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()

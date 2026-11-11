"""Dimension-link cases: ideal readings and real OCR readings vs the reference DXF.

Per sheet (a case dir with cv.json + metrics.json from mlcv/eval/evaluate.py, plus an OCR
observations JSON from mlocr) this runs to_drawing twice — with ideal readings taken from the
reference DIMENSION entities and with the real OCR readings — scores each link against the
reference and dumps the full decision record (label id, feet, part segment, dimension and
extension lines, view scale, reason).

    python -m mlocr.evaluation.dim_link_cases CASE_DIR ... [--ocr OCR_JSON ...]
"""
import argparse
import collections
import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "mlcv"), str(ROOT / "mlcv" / "eval")]
_tmp = Path(tempfile.mkdtemp(prefix="dimlink-cases-"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{(_tmp / 'check.sqlite3').as_posix()}")
os.environ.setdefault("STORAGE_ROOT", str(_tmp / "storage"))
os.environ.setdefault("REQUIRE_LOCAL_MODELS", "false")

import ezdxf  # noqa: E402

import ocr_link_check as base  # noqa: E402  mlcv/eval, on sys.path
from mlocr.evaluation.matching import match_dimensions, real_observations as checked_observations  # noqa: E402

DXF = ROOT / "dataset/clear_sum_dataset/vse_dxf"


def real_observations(ocr_path, cv, payload):
    return checked_observations(payload, cv)


def on_reference(dimension, points_by_id, dims, translation):
    """Single-candidate helper; aggregate scoring must use match_dimensions()."""
    return bool(match_dimensions([dimension], points_by_id, dims, translation))


def link_record(link, adapter_dimensions, calibration=None):
    record = {"text_id": link["text_id"], "recognized_text": link["recognized_text"],
              "nominal_mm": link["nominal_mm"], "status": link["status"], "basis": link.get("basis"),
              "reason": link.get("dimension_line_search") or link.get("decision_note"),
              "candidate_count": link.get("candidate_count")}
    chosen = link.get("confirmed")
    if chosen:
        evidence = chosen["evidence"]
        record.update({"view_scale": chosen["view_scale"], "scale_basis": (calibration or {}).get('view_scale_basis', 'provided_calibration'),
                       "dimension_line": evidence.get("dimension_line"),
                       "dimension_row": evidence.get("dimension_row"),
                       "extension_lines": evidence.get("extension_lines"),
                       "feet": [{"xy": f["xy"], "part_segment": f.get("on"), "anchor": f.get("anchor")}
                                for f in evidence.get("feet", [])]})
        dim = next((d for d in adapter_dimensions if d["id"] == "dim-" + link["text_id"]), None)
        if dim:
            record.update({"p1_ref": dim["p1_ref"], "p2_ref": dim["p2_ref"], "display": dim["display"],
                           "nominal_from_text": dim["nominal_mm"]})
    return record


def check_case(case_dir, ocr_path):
    from app.services.dxf import DxfExporter
    from app.services.recognition import assemble_graph
    from mlocr.src.adapter import to_drawing

    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((case_dir / "cv.json").read_text(encoding="utf-8"))
    translation = np.array(metrics["registration"]["translation_mm"])
    dims = base.reference_dimensions(DXF / f"{metrics['sheet']}.dxf")
    factors = {round(d["factor"], 6) for d in dims}
    payload = json.loads(Path(ocr_path).read_text(encoding="utf-8")) if ocr_path else None

    out = {"case": metrics["case"], "sheet_dxf": metrics["sheet"], "reference_dims": len(dims),
           "ideal_reference_scale": (1 / next(iter(factors))) if len(factors) == 1 else None,
           "modes": {}}
    if not dims:
        return out
    points = np.array([p["xy"] for p in cv.get("points", [])])
    ends = np.array([p for d in dims for p in (d["p1"], d["p2"])])
    if len(points) and len(ends):
        dist = np.min(np.linalg.norm(ends[:, None] - (points[None] + translation), axis=2), axis=1)
        out["ends_within_05"] = round(float(np.mean(dist <= 0.5)), 3)
    for mode in ("ideal", "real"):
        if mode == "real" and payload is None:
            continue
        cv_work = copy.deepcopy(cv)
        try:
            if mode == 'ideal':
                if len(factors) != 1:
                    raise ValueError('Ideal single-view evaluation requires exactly one reference scale')
                cv_work.setdefault('calibration', {}).update(view_scale=1 / next(iter(factors)),
                                                             view_scale_basis='reference_dxf_oracle')
            observed = (base.observations(cv_work, dims, translation) if mode == "ideal"
                        else real_observations(ocr_path, cv_work, payload))
            ocr = to_drawing(observed, cv_work)
        except Exception as exc:
            out["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        points_by_id = {p["id"]: p for p in cv_work["points"]}
        links = ocr.get("dimension_links", [])
        dimensions = ocr.get('dimensions', [])
        records = [link_record(link, dimensions, cv_work.get('calibration')) for link in links]
        matches = match_dimensions(dimensions, points_by_id, dims, translation)
        correct = len(matches)
        graph = assemble_graph(cv_work, ocr, "check")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.dxf"
            issues = DxfExporter().export(graph, path)
            dxf_dimensions = len(ezdxf.readfile(path).modelspace().query("DIMENSION")) if path.exists() else 0
        out["modes"][mode] = {
            "links_total": len(links),
            "status": dict(collections.Counter(link["status"] for link in links)),
            "search": dict(collections.Counter((link.get("dimension_line_search") or link.get("decision_note") or "linked")
                                               for link in links if not link.get("confirmed"))),
            "dimensions": len(ocr.get("dimensions", [])),
            "dimensions_on_reference": correct,
            "wrong_or_extra_dimensions": len(dimensions) - correct,
            "missed_reference_dimensions": len(dims) - correct,
            "precision": correct / len(dimensions) if dimensions else None,
            "recall": correct / len(dims),
            "matched_nominals_correct": sum(math.isclose(dimensions[i]['nominal_mm'], dims[j]['nominal'], abs_tol=.01)
                                             for i, j in matches),
            "calibration": copy.deepcopy(cv_work.get('calibration', {})),
            "reference_assisted": mode == 'ideal',
            "ocr_status": observed.get('status'), "ocr_issues": observed.get('issues', []),
            "dxf_dimensions": dxf_dimensions,
            "export_errors": dict(collections.Counter(i.code for i in issues if i.severity == "error")),
            "records": records,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", type=Path, help="case dirs with cv.json + metrics.json")
    parser.add_argument("--ocr", nargs="*", default=[], help="OCR observations JSON per case dir (same order)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.ocr and len(args.ocr) != len(args.cases):
        parser.error('--ocr must provide exactly one OCR file for every case')
    results = []
    for index, case_dir in enumerate(args.cases):
        ocr_path = args.ocr[index] if index < len(args.ocr) else None
        result = check_case(case_dir, ocr_path)
        results.append(result)
        for mode, res in result["modes"].items():
            if "error" in res:
                print(f"{result['case'][:30]:30} {mode:5} ERROR {res['error']}", flush=True)
                continue
            print(f"{result['case'][:30]:30} {mode:5} dims {result['reference_dims']:3} linked {res['dimensions']:3} "
                  f"on-ref {res['dimensions_on_reference']:3} dxf {res['dxf_dimensions']:3} {res['status']} {res['search']}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    if not any(result['modes'] for result in results):
        parser.error('No reference dimensions found; no quality evaluation was performed')
    return 1 if any('error' in mode for result in results for mode in result['modes'].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

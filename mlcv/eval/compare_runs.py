"""Row-by-row comparison of two evaluation runs, both scored by the current rules.

    python mlcv/eval/compare_runs.py BASE_DIR NEW_DIR [BASE_DIR NEW_DIR ...] [--json OUT.json]

Contours and backend acceptance are recomputed from cv.json, so an older run is rescored with the one-to-one matching."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as ev  # noqa: E402

COUNTERS = ("body_found", "body_total", "extra", "contour_pairs", "contour_reference", "contours_closed", "contour_copies",
            "found_many_to_one", "backend_accepted", "ambiguous_points", "gaps", "self_intersections", "hatches",
            "points", "over_ocr_limit", "graph_rejected")


def score(run_dir):
    rows = {}
    for mp in sorted(Path(run_dir).glob("*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        cv = json.loads((mp.parent / "cv.json").read_text(encoding="utf-8"))
        ref = Path(run_dir) / "synth/control.dxf" if m["sheet"] == "control" else ev.DXF / f"{m['sheet']}.dxf"
        geo = ev.reference(ref)[0]
        t = np.array(m["registration"]["translation_mm"])
        W, H = cv["calibration"]["sheet_mm"]
        frame = (20 + t[0], 5 + t[1], W - 5 + t[0], H - 5 + t[1])
        title = (W - 5 - 185 + t[0], 5 + t[1], W - 5 + t[0], 60 + t[1])
        c, _ = ev.contour_metrics(cv, geo, [ev.in_sheet_body(e, frame, title) for e in geo], t)
        back = ev.backend_check(cv)
        rows[m["case"]] = {
            "body_found": m["found"]["body"], "body_total": m["reference"]["body_primitives"],
            "extra": m["detected"]["categories"].get("extra", 0), "contour_pairs": c["found"], "contour_reference": c["reference_closed"],
            "contours_closed": c["detected_closed"], "contour_copies": c["copies"], "found_many_to_one": c["found_many_to_one"],
            "backend_accepted": back["accepted"], "ambiguous_points": back["ambiguous_points"],
            "gaps": m["defects"].get("gap", 0), "self_intersections": m["defects"].get("self_intersection", 0),
            "hatches": cv["summary"]["hatches_resolved"], "points": len(cv["points"]), "over_ocr_limit": int(len(cv["points"]) > 400),
            "graph_rejected": int(not back["graph_valid"]), "time_s": m["timings_s"]["total"]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    total = {"base": dict.fromkeys(COUNTERS, 0), "new": dict.fromkeys(COUNTERS, 0)}
    rows = []
    for base_dir, new_dir in zip(a.dirs[0::2], a.dirs[1::2]):
        base, new = score(base_dir), score(new_dir)
        for case in base:
            if case not in new:
                continue
            x, y = base[case], new[case]
            rows.append({"case": case, "base": x, "new": y})
            for key in COUNTERS:
                total["base"][key] += x[key]
                total["new"][key] += y[key]
            print(f"{case[:32]:32} тело {x['body_found']:3}->{y['body_found']:3}/{x['body_total']:3} контуры 1:1 {x['contour_pairs']:3}->{y['contour_pairs']:3}/{x['contour_reference']:3} "
                  f"замкн. {x['contours_closed']:3}->{y['contours_closed']:3} бэкенд {x['backend_accepted']:3}->{y['backend_accepted']:3} "
                  f"неоднозн. {x['ambiguous_points']:3}->{y['ambiguous_points']:3} разрывы {x['gaps']:3}->{y['gaps']:3}")
    print(total)
    if a.json:
        Path(a.json).write_text(json.dumps({"inputs": len(rows), "total": total, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

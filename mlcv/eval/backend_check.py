"""What the real backend accepts from a cv.json: its DrawingGraph model and GeometryResolver, not copies of them.

    python mlcv/eval/backend_check.py DIR [DIR ...]      # every DIR/*/cv.json, table to stdout
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

OCR_MAX_POINTS = 400                     # mlocr/src/dimension_links.py refuses to link dimensions above this


def check(cv):
    """Counts of accepted and rejected closed contours and points. Uses the graph fields the backend reads."""
    from app.domain.drawing import DrawingGraph
    from app.services.geometry import GeometryError, GeometryResolver

    out = {"graph_valid": True, "closed": 0, "accepted": 0, "rejected": {}, "points": len(cv.get("points", [])),
           "ambiguous_points": 0}
    try:
        graph = DrawingGraph(coordinate_space=cv["coordinate_space"], scale_status=cv["scale_status"],
                             primitives=cv["primitives"], points=cv["points"], contours=cv["contours"],
                             objects=cv["objects"], hatches=[])
    except Exception as exc:
        out.update(graph_valid=False, error=str(exc).splitlines()[1] if "\n" in str(exc) else str(exc))
        return out
    resolver = GeometryResolver(graph)
    rejected = collections.Counter()
    for contour in graph.contours:
        if not contour.closed:
            continue
        out["closed"] += 1
        try:
            resolver.contour(contour.id)
            out["accepted"] += 1
        except GeometryError as exc:
            rejected[exc.code] += 1
    for point in graph.points:
        try:
            resolver.point(point.id)
        except GeometryError:
            out["ambiguous_points"] += 1
    out["rejected"] = dict(rejected)
    return out


def main():
    total = collections.Counter()
    for root in sys.argv[1:]:
        for path in sorted(Path(root).glob("*/cv.json")):
            r = check(json.loads(path.read_text(encoding="utf-8")))
            print(f"{path.parent.name[:44]:44} graph {'ok' if r['graph_valid'] else 'REJECTED'} closed {r['accepted']}/{r['closed']} "
                  f"{r['rejected'] or ''} points {r['points']} ambiguous {r['ambiguous_points']}")
            total.update(sheets=1, closed=r["closed"], accepted=r["accepted"], ambiguous=r["ambiguous_points"],
                         points=r["points"], graph_rejected=not r["graph_valid"],
                         over_ocr_limit=r["points"] > OCR_MAX_POINTS)
    print(dict(total))


if __name__ == "__main__":
    main()

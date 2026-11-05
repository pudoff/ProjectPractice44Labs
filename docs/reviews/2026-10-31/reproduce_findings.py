"""Small diagnostic inputs for the review. Does not change production code or datasets."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
WORK = ROOT / ".local-backups" / "review-20261031"
WORK.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "mlcv")]

import ezdxf
from app.domain.drawing import DrawingGraph
from app.services.dxf import DxfExporter
from app.services.image import ImageService
from app.services.recognition import assemble_graph
from drawcv.ingest import load_page
from drawcv.topology import build


def export_case(name, data):
    graph = DrawingGraph.model_validate({"coordinate_space": "part_mm", **data})
    # Every invocation gets a new path: an old DXF must not mask a rejected export.
    target = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=WORK)) / "result.dxf"
    issues = DxfExporter().export(graph, target)
    doc = ezdxf.readfile(target) if target.exists() else None
    return {"saved": target.exists(), "issues": [i.model_dump() for i in issues]}, doc


def main():
    result = {}
    line = {"id": "edge", "kind": "line", "start": [0, 0], "end": [45.95, 0]}
    value, doc = export_case("dimension_tolerance", {"primitives": [line], "dimensions": [
        {"id": "dim", "p1": [0, 0], "p2": [45.95, 0], "base": [20, -5], "nominal_mm": 46}]})
    value["actual_measurement"] = doc.modelspace().query("DIMENSION")[0].get_measurement()
    result["dimension_tolerance"] = value

    value, doc = export_case("layers", {"primitives": [{**line, "layer": "AXIS"}],
        "hatches": [{"id": "h", "outer": [[0, 0], [10, 0], [10, 10], [0, 10]]}]})
    value["axis_linetype"] = doc.layers.get("AXIS").dxf.linetype
    value["hatch_layer"] = doc.modelspace().query("HATCH")[0].dxf.layer
    result["layers"] = value

    bad_cases = {
        "zero_length": {"primitives": [{"id": "zero", "kind": "line", "start": [0, 0], "end": [0, 0]}]},
        "self_intersection": {"primitives": [{"id": "bowtie", "kind": "polyline", "closed": True,
            "vertices": [[0, 0], [10, 10], [0, 10], [10, 0]]}]},
        "outside_hatch_hole": {"hatches": [{"id": "h", "outer": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "holes": [[[20, 20], [21, 20], [21, 21], [20, 21]]]}]},
    }
    for name, data in bad_cases.items():
        result[name], _ = export_case(name, data)

    cv = {"status": "ok", "scale_status": "known", "coordinate_space": "sheet_mm",
          "primitives": [line], "points": [{"id": "pt_0", "xy": [0, 0]}],
          "contours": [{"id": "contour_0", "closed": True}],
          "objects": {"edge": {"role": "axis", "linetype": "CENTER"}},
          "calibration": {"sheet": "A4"}, "transforms": {"source_px_to_sheet_mm": np.eye(3).tolist()}}
    graph = assemble_graph(cv, {"status": "ok", "title_block": {"material": "Steel"}}, "test")
    result["discarded_metadata"] = {"graph_fields": sorted(graph.model_dump()),
        "lost_from_graph": [k for k in ("points", "contours", "objects", "calibration", "transforms", "title_block")
                            if k not in graph.model_dump()]}

    source = WORK / "transparent.png"
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    for x in range(5, 15):
        image.putpixel((x, 10), (0, 0, 0, 255))
    image.save(source)
    prepared = ImageService().prepare(source, WORK / "image_run")
    result["transparency"] = {"backend_background": list(Image.open(prepared).getpixel((0, 0))),
        "direct_cv_background": int(load_page(source).gray[0, 0]),
        "backend_unique_gray_values": np.unique(np.asarray(Image.open(prepared).convert("L"))).tolist()}

    def seg(id_, p, q):
        return {"id": id_, "kind": "line", "p": np.array(p, float), "q": np.array(q, float), "role": "part"}
    prims = [seg("a", [0, 0], [10, 0]), seg("b", [10, 0], [10.05, 10.04]),
             seg("c", [10, 10], [0, 10]), seg("d", [0, 10], [0, 0])]
    topology = build(prims, [], 0.3, 3.0)
    closed = [c for c in topology["contours"] if c["closed"]]
    result["topology_gap"] = {"closed_contours": len(closed),
        "closure_error_mm": closed[0]["closure_error_mm"],
        "remaining_gap_mm": float(np.linalg.norm(prims[1]["q"] - prims[2]["p"]))}

    (OUT / "reproductions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

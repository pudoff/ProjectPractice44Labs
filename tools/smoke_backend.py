"""HTTP smoke against the real API/worker/PostgreSQL stack (no mocked recognizers).

Three generated formats exercise recognition; manual graphs separately exercise
semantic export. This does not measure recognition accuracy or CAD GUI support.
"""
import argparse
import io
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid

import ezdxf
from PIL import Image
from drawcv.synth import render


def request(base, path, data=None, content_type=None):
    headers = {"Content-Type": content_type} if content_type else {}
    with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=30) as response:
        return response.read()


def post_json(base, path, value):
    return json.loads(request(base, path, json.dumps(value).encode("utf-8"), "application/json"))


def assert_preserved(source, accepted, path="graph"):
    """All CV fields must survive; schema-added defaults are allowed."""
    if isinstance(source, dict):
        for key, value in source.items():
            assert key in accepted, f"{path}/{key} was dropped"
            assert_preserved(value, accepted[key], f"{path}/{key}")
    elif isinstance(source, list):
        assert len(source) == len(accepted), f"{path}: changed number of items"
        for index, (a, b) in enumerate(zip(source, accepted)):
            assert_preserved(a, b, f"{path}/{index}")
    else:
        assert source == accepted, f"{path}: {source!r} != {accepted!r}"


def wait_api(base, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request(base, "/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("API did not become ready")


def wait_job(base, job_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = json.loads(request(base, f"/api/v1/jobs/{job_id}"))
        if status["state"] not in ("queued", "running"):
            return status
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not finish")


def drawing(width):
    return {
        "schema_version": "0.2", "coordinate_space": "part_mm", "export_profile": "editable",
        "provenance": {"manual": True, "purpose": "export smoke, not automatic OCR"},
        "primitives": [
            {"id": "plate", "kind": "polyline", "closed": True, "vertices": [[0, 0], [width, 0], [width, 20], [0, 20]]},
            {"id": "hole", "kind": "circle", "center": [15, 10], "radius": 3},
            {"id": "axis", "kind": "line", "start": [-5, 10], "end": [width + 5, 10], "layer": "AXIS"}],
        "points": [{"id": "a", "xy": [0, 0], "refs": [{"object_id": "plate", "anchor": "vertex_0"}]},
                   {"id": "b", "xy": [46, 0], "refs": [{"object_id": "plate", "anchor": "vertex_1"}]}],
        "contours": [{"id": "outer", "closed": True, "object_ids": ["plate"]},
                     {"id": "inner", "closed": True, "object_ids": ["hole"]}],
        "dimensions": [{"id": "width", "p1_ref": "a", "p2_ref": "b", "base": [0, -5], "nominal_mm": width}],
        "hatches": [{"id": "section", "outer_ref": "outer", "hole_refs": ["inner"]}],
        "texts": [{"id": "material", "value": "Сталь 20", "insert": [0, 25], "height": 3}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wait_api(args.base)
    inputs, results = [], []
    image = Image.fromarray(render(150)).convert("RGB")
    for suffix, format_ in (("png", "PNG"), ("jpg", "JPEG"), ("pdf", "PDF")):
        path = args.out / f"control.{suffix}"
        image.save(path, format=format_, **({"resolution": 150} if suffix == "pdf" else {"dpi": (150, 150)}))
        inputs.append(path)
    for path in inputs:
        boundary = "practice44-" + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                'Content-Type: application/octet-stream\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        job = json.loads(request(args.base, "/api/v1/jobs", body, f"multipart/form-data; boundary={boundary}"))
        status = wait_job(args.base, job["id"])
        assert status["state"] == "needs_review", status  # OCR has not been delivered.
        prefix = f"/api/v1/jobs/{job['id']}/artifacts/"
        cv = json.loads(request(args.base, prefix + "cv.json"))
        assert cv["status"] == "ok", cv
        assert cv["module"]["name"] == "drawcv"
        graph = json.loads(request(args.base, prefix + "drawing.json"))
        report = json.loads(request(args.base, prefix + "report.json"))
        for field in ("points", "contours", "objects", "calibration", "transforms"):
            assert_preserved(cv[field], graph[field], field)
        assert any(i["code"] == "ocr_error" for i in report["issues"])
        assert not any(i["code"] == "contour_kind_mismatch" for i in report["issues"]), report["issues"]
        (args.out / f"{path.suffix[1:]}-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"input": path.name, "job_id": job["id"], "state": status["state"], "cv_module": cv["module"],
                        "primitives": len(graph["primitives"]), "points": len(graph["points"]), "contours": len(graph["contours"]),
                        "scale_status": graph["scale_status"], "issues": sorted({i["code"] for i in report["issues"]})})
        print(f"Recognized {path.name}: {len(graph['primitives'])} primitives", flush=True)
    job_id = results[0]["job_id"]
    for width in (46, 47):
        submitted = drawing(width)
        (args.out / f"manual-{width}.json").write_text(json.dumps(submitted, ensure_ascii=False, indent=2), encoding="utf-8")
        post_json(args.base, f"/api/v1/jobs/{job_id}/drawing", submitted)
        status = wait_job(args.base, job_id)
        assert status["state"] == "succeeded", status
        raw = request(args.base, f"/api/v1/jobs/{job_id}/artifacts/result.dxf")
        path = args.out / f"manual-{width}.dxf"
        path.write_bytes(raw)
        doc = ezdxf.readfile(path)
        assert doc.modelspace().query("DIMENSION")[0].get_measurement() == width
        assert doc.modelspace().query("HATCH")[0].dxf.layer == "HATCH"
        assert len(doc.modelspace().query("HATCH")[0].paths.paths) == 2
        assert doc.layers.get("AXIS").dxf.linetype == "CENTER"
    old = request(args.base, f"/api/v1/jobs/{job_id}/artifacts/result.dxf?run_number=2")
    assert ezdxf.read(io.StringIO(old.decode("utf-8"), newline=None)).modelspace().query("DIMENSION")[0].get_measurement() == 46
    summary = {"recognition": results, "manual_export": {"measurements": [46, 47], "old_run_preserved": True,
               "native_dimensions": True, "native_hatch_hole": True, "axis_linetype": "CENTER"}, "cad_gui_checked": False}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

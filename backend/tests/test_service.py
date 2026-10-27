import io
import asyncio
import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:////tmp/drawing-service-test.sqlite3"
os.environ["STORAGE_ROOT"] = "/tmp/drawing-service-test-storage"

import ezdxf
import httpx
import numpy as np
import pytest
from app.api import app
from app.db import Base, engine
from app.worker import claim_run, process_run, recover_interrupted_runs
from app.drawing import DrawingGraph
from app.exporter import export_dxf
from app.pipeline import assemble_graph


def request(method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


def test_upload_review_and_manual_graph_export(monkeypatch):
    monkeypatch.setattr("app.pipeline.run_builtin_recognition", lambda image: (
        {"status": "error", "scale_status": "unknown", "error": "CV unavailable"},
        {"status": "error", "error": "OCR unavailable"}))
    image = b"\x89PNG\r\n\x1a\n" + bytes.fromhex("0000000d4948445200000001000000010802000000907753de0000000c49444154789c63606060000000040001a3f4c60000000049454e44ae426082")
    response = request("POST", "/api/v1/jobs", files={"file": ("plate.png", image, "image/png")})
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]
    process_run(claim_run())
    assert request("GET", f"/api/v1/jobs/{job_id}").json()["state"] == "needs_review"
    graph = {"schema_version": "0.1", "coordinate_space": "part_mm", "primitives": [{"id": "p", "kind": "polyline", "vertices": [[0, 0], [46, 0], [46, 20], [0, 20]], "closed": True}], "texts": [{"id": "t", "value": "Сталь 20", "insert": [0, 25], "height": 3}], "dimensions": [{"id": "d", "p1": [0, 0], "p2": [46, 0], "base": [23, -7], "nominal_mm": 46}], "hatches": [{"id": "h", "outer": [[0, 0], [46, 0], [46, 20], [0, 20]], "holes": [[[10, 5], [20, 5], [20, 15], [10, 15]]]}]}
    response = request("POST", f"/api/v1/jobs/{job_id}/drawing", json=graph)
    assert response.status_code == 202, response.text
    process_run(claim_run())
    status = request("GET", f"/api/v1/jobs/{job_id}").json()
    assert status["state"] == "succeeded", status
    result = request("GET", f"/api/v1/jobs/{job_id}/artifacts/result.dxf")
    assert result.status_code == 200
    doc = ezdxf.read(io.StringIO(result.text))
    types = {entity.dxftype() for entity in doc.modelspace()}
    assert {"LWPOLYLINE", "TEXT", "DIMENSION", "HATCH"} <= types
    dimension = doc.modelspace().query("DIMENSION")[0]
    assert dimension.dxf.text == "<>"
    assert dimension.get_geometry_block() is not None
    assert len(doc.modelspace().query("HATCH")[0].paths.paths) == 2
    assert request("GET", f"/api/v1/jobs/{job_id}/artifacts/preview.png").status_code == 200
    assert request("POST", f"/api/v1/jobs/{job_id}/rerun").status_code == 202


def test_validation_and_event_idempotency(monkeypatch):
    bad = request("POST", "/api/v1/jobs", files={"file": ("x.pdf", b"not pdf")})
    assert bad.status_code == 415
    monkeypatch.setattr("app.api.API_KEY", "test-secret")
    auth = {"x-api-key": "test-secret"}
    data = b"%PDF-1.4\n"
    fields = {"channel": "vk", "conversation_id": "123", "event_id": "456"}
    first = request("POST", "/api/v1/jobs", data=fields, files={"file": ("x.pdf", data)}, headers=auth)
    second = request("POST", "/api/v1/jobs", data=fields, files={"file": ("x.pdf", data)}, headers=auth)
    assert first.json()["id"] == second.json()["id"]
    job_id = first.json()["id"]
    assert request("GET", f"/api/v1/jobs/{job_id}", headers=auth).status_code == 404
    bot_headers = {**auth, "x-channel": "vk", "x-conversation-id": "123"}
    assert request("GET", f"/api/v1/jobs/{job_id}", headers=bot_headers).status_code == 200
    assert request("POST", f"/api/v1/jobs/{job_id}/drawing", headers=bot_headers, json={"coordinate_space": "part_mm", "primitives": [{"id": "x", "kind": "spline"}]}).status_code == 422


def test_scale_and_dimension_conflict_are_explicit(tmp_path):
    unknown = DrawingGraph(coordinate_space="part_mm", scale_status="unknown")
    target = tmp_path / "unknown.dxf"
    issues = export_dxf(unknown, target)
    assert not target.exists()
    assert any(issue.code == "unknown_scale" for issue in issues)

    graph = DrawingGraph.model_validate({"coordinate_space": "part_mm", "primitives": [{"id": "edge", "kind": "line", "start": [0, 0], "end": [45, 0]}], "dimensions": [{"id": "width", "p1": [0, 0], "p2": [45, 0], "base": [20, -5], "nominal_mm": 46}]})
    target = tmp_path / "mismatch.dxf"
    issues = export_dxf(graph, target)
    assert target.exists()
    assert any(issue.code == "dimension_mismatch" for issue in issues)
    dim = ezdxf.readfile(target).modelspace().query("DIMENSION")[0]
    assert dim.dxf.text == "<>"

    scaled = DrawingGraph.model_validate({"coordinate_space": "sheet_mm", "primitives": [{"id": "edge", "kind": "line", "start": [0, 0], "end": [50, 0]}], "dimensions": [{"id": "width", "p1": [0, 0], "p2": [50, 0], "base": [25, -5], "nominal_mm": 10, "view_scale": 5}]})
    assert not export_dxf(scaled, tmp_path / "scaled.dxf")


def test_recognition_issues_keep_partial_candidates():
    graph = assemble_graph(
        {"status": "ok", "scale_status": "known", "primitives": [{"id": "line", "kind": "line", "start": [0, 0], "end": [5, 0]}, {"id": "spline", "kind": "spline"}]},
        {"status": "error", "error": "OCR timeout"}, "abc")
    assert len(graph.primitives) == 1
    assert {issue.code for issue in graph.issues} == {"unsupported_object", "ocr_error"}


def test_interrupted_run_can_be_retried():
    image = b"\x89PNG\r\n\x1a\n"
    response = request("POST", "/api/v1/jobs", files={"file": ("x.png", image, "image/png")})
    job_id = response.json()["id"]
    assert claim_run()
    recover_interrupted_runs()
    status = request("GET", f"/api/v1/jobs/{job_id}").json()
    assert status["state"] == "failed"
    assert status["runs"][0]["error_code"] == "worker_interrupted"
    assert request("POST", f"/api/v1/jobs/{job_id}/rerun").json()["runs"][-1]["number"] == 2


def test_builtin_observations_export_scaled_dimension(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs" / "klapan_png2dxf" / "pipeline_b"))
    from pipeline import observations
    from ocr import TextItem

    line = {"type": "line", "p": np.array([0.0, 0.0]), "q": np.array([50.0, 0.0])}
    dim = {"p1": np.array([0.0, 0.0]), "p2": np.array([50.0, 0.0]), "base": np.array([25.0, -5.0]),
           "angle": 0.0, "raw_text": "10", "scale": 5.0, "ext": ((1,), (1,))}
    text = TextItem("10", 0.99, np.array([[20.0, 3.0], [30.0, 3.0], [30.0, 0.0], [20.0, 0.0]]), 0, 3)
    cv, ocr = observations([line], [], [], [], [dim], [text], {0}, {"calibrated_on": "sheet trim line"}, "A3", None)
    graph = assemble_graph(cv, ocr, "abc")
    assert graph.scale_status == "known"
    assert graph.coordinate_space == "sheet_mm"
    assert graph.dimensions[0].view_scale == 5
    assert not export_dxf(graph, tmp_path / "observed.dxf")

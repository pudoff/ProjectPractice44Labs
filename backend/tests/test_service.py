import io
import os
import uuid
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/drawing-service-test.sqlite3"
os.environ["STORAGE_ROOT"] = "/tmp/drawing-service-test-storage"

import ezdxf
import httpx
import numpy as np
import pytest
import pymupdf
from sqlalchemy import event

from app.api import app
from app.domain.drawing import DrawingGraph
from app.services.dxf import DxfExporter
from app.infra.db import Base, engine
from app.tasks.job import claim_run, process_run, recover_interrupted_runs
from app.services.image import ImageService
from app.services.recognition import assemble_graph


@event.listens_for(engine.sync_engine, "connect")
def sqlite_uuid(dbapi_connection, _record):
    dbapi_connection.create_function("uuid_generate_v7", 0, lambda: str(uuid.uuid4()))


async def request(method, path, **kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio", {"use_uvloop": True}


@pytest.fixture(autouse=True)
async def clean_db(anyio_backend):
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_upload_review_and_manual_graph_export(monkeypatch):
    monkeypatch.setattr("app.services.recognition.run_builtin_recognition", lambda image: (
        {"status": "error", "scale_status": "unknown", "error": "CV unavailable"},
        {"status": "error", "error": "OCR unavailable"}))
    image = b"\x89PNG\r\n\x1a\n" + bytes.fromhex("0000000d4948445200000001000000010802000000907753de0000000c49444154789c63606060000000040001a3f4c60000000049454e44ae426082")
    response = await request("POST", "/api/v1/jobs", files={"file": ("plate.png", image, "image/png")})
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]
    await process_run(await claim_run())
    assert (await request("GET", f"/api/v1/jobs/{job_id}")).json()["state"] == "needs_review"
    graph = {"schema_version": "0.1", "coordinate_space": "part_mm", "primitives": [{"id": "p", "kind": "polyline", "vertices": [[0, 0], [46, 0], [46, 20], [0, 20]], "closed": True}], "texts": [{"id": "t", "value": "Сталь 20", "insert": [0, 25], "height": 3}], "dimensions": [{"id": "d", "p1": [0, 0], "p2": [46, 0], "base": [23, -7], "nominal_mm": 46}], "hatches": [{"id": "h", "outer": [[0, 0], [46, 0], [46, 20], [0, 20]], "holes": [[[10, 5], [20, 5], [20, 15], [10, 15]]]}]}
    response = await request("POST", f"/api/v1/jobs/{job_id}/drawing", json=graph)
    assert response.status_code == 202, response.text
    await process_run(await claim_run())
    status = (await request("GET", f"/api/v1/jobs/{job_id}")).json()
    assert status["state"] == "succeeded", status
    assert "result.dxf" in status["runs"][-1]["artifacts"]
    from app.infra.storage import run_dir
    assert (run_dir(job_id, status["runs"][-1]["id"]) / "result.dxf").is_file()
    result = await request("GET", f"/api/v1/jobs/{job_id}/artifacts/result.dxf")
    assert result.status_code == 200
    doc = ezdxf.read(io.StringIO(result.text))
    types = {entity.dxftype() for entity in doc.modelspace()}
    assert {"LWPOLYLINE", "TEXT", "DIMENSION", "HATCH"} <= types
    assert doc.modelspace().query("DIMENSION")[0].dxf.text == "<>"
    assert len(doc.modelspace().query("HATCH")[0].paths.paths) == 2
    assert (await request("GET", f"/api/v1/jobs/{job_id}/artifacts/preview.png")).status_code == 200
    assert (await request("POST", f"/api/v1/jobs/{job_id}/rerun")).status_code == 202


@pytest.mark.anyio
async def test_validation_and_event_idempotency():
    assert (await request("POST", "/api/v1/jobs", files={"file": ("x.pdf", b"not pdf")})).status_code == 415
    data = b"%PDF-1.4\n"
    fields = {"channel": "vk", "conversation_id": "123", "event_id": "456"}
    first = await request("POST", "/api/v1/jobs", data=fields, files={"file": ("x.pdf", data)})
    second = await request("POST", "/api/v1/jobs", data=fields, files={"file": ("x.pdf", data)})
    assert first.status_code == 201, first.text
    assert first.json()["id"] == second.json()["id"]
    job_id = first.json()["id"]
    assert (await request("GET", f"/api/v1/jobs/{job_id}")).status_code == 200
    assert (await request("POST", f"/api/v1/jobs/{job_id}/drawing", json={"coordinate_space": "part_mm", "primitives": [{"id": "x", "kind": "spline"}]})).status_code == 422


def test_scale_and_dimension_conflict_are_explicit(tmp_path):
    unknown = DrawingGraph(coordinate_space="part_mm", scale_status="unknown")
    target = tmp_path / "unknown.dxf"
    issues = DxfExporter().export(unknown, target)
    assert not target.exists()
    assert any(issue.code == "unknown_scale" for issue in issues)
    graph = DrawingGraph.model_validate({"coordinate_space": "part_mm", "primitives": [{"id": "edge", "kind": "line", "start": [0, 0], "end": [45, 0]}], "dimensions": [{"id": "width", "p1": [0, 0], "p2": [45, 0], "base": [20, -5], "nominal_mm": 46}]})
    target = tmp_path / "mismatch.dxf"
    issues = DxfExporter().export(graph, target)
    assert target.exists()
    assert any(issue.code == "dimension_mismatch" for issue in issues)
    assert ezdxf.readfile(target).modelspace().query("DIMENSION")[0].dxf.text == "<>"


def test_recognition_issues_keep_partial_candidates():
    graph = assemble_graph({"status": "ok", "scale_status": "known", "primitives": [{"id": "line", "kind": "line", "start": [0, 0], "end": [5, 0]}, {"id": "spline", "kind": "spline"}]}, {"status": "error", "error": "OCR timeout"}, "abc")
    assert len(graph.primitives) == 1
    assert {issue.code for issue in graph.issues} == {"unsupported_object", "ocr_error"}


def test_primitive_rejects_geometry_of_another_kind():
    with pytest.raises(ValueError, match="unrelated geometry fields"):
        DrawingGraph.model_validate({"coordinate_space": "part_mm", "primitives": [
            {"id": "line", "kind": "line", "start": [0, 0], "end": [1, 0], "radius": 3}
        ]})


def test_pdf_preparation_and_page_limit(tmp_path):
    source = tmp_path / "drawing.pdf"
    document = pymupdf.open()
    document.new_page(width=72, height=72)
    document.save(source)
    document.close()
    image = ImageService().prepare(source, tmp_path / "run")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    document = pymupdf.open(source)
    document.new_page()
    document.save(tmp_path / "two-pages.pdf")
    document.close()
    with pytest.raises(ValueError, match="одностраничный"):
        ImageService().prepare(tmp_path / "two-pages.pdf", tmp_path / "run")


@pytest.mark.anyio
async def test_interrupted_run_can_be_retried():
    response = await request("POST", "/api/v1/jobs", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n", "image/png")})
    job_id = response.json()["id"]
    assert await claim_run()
    await recover_interrupted_runs()
    status = (await request("GET", f"/api/v1/jobs/{job_id}")).json()
    assert status["state"] == "failed"
    assert status["runs"][0]["error_code"] == "worker_interrupted"
    assert (await request("POST", f"/api/v1/jobs/{job_id}/rerun")).json()["runs"][-1]["number"] == 2


def test_builtin_observations_export_scaled_dimension(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs" / "klapan_png2dxf" / "pipeline_b"))
    from pipeline import observations
    from ocr import TextItem
    line = {"type": "line", "p": np.array([0.0, 0.0]), "q": np.array([50.0, 0.0])}
    dim = {"p1": np.array([0.0, 0.0]), "p2": np.array([50.0, 0.0]), "base": np.array([25.0, -5.0]), "angle": 0.0, "raw_text": "10", "scale": 5.0, "ext": ((1,), (1,))}
    text = TextItem("10", 0.99, np.array([[20.0, 3.0], [30.0, 3.0], [30.0, 0.0], [20.0, 0.0]]), 0, 3)
    cv, ocr = observations([line], [], [], [], [dim], [text], {0}, {"calibrated_on": "sheet trim line"}, "A3", None)
    graph = assemble_graph(cv, ocr, "abc")
    assert graph.scale_status == "known"
    assert graph.coordinate_space == "sheet_mm"
    assert graph.dimensions[0].view_scale == 5
    assert not DxfExporter().export(graph, tmp_path / "observed.dxf")


def test_unsupported_dimension_keeps_supported_geometry():
    graph = assemble_graph(
        {"status": "ok", "scale_status": "known", "primitives": [
            {"id": "edge", "kind": "line", "start": [0, 0], "end": [10, 0]}]},
        {"status": "ok", "dimensions": [{"id": "radius", "kind": "radius"}]}, "hash")
    assert len(graph.primitives) == 1
    assert not graph.dimensions
    assert graph.issues[0].code == "unsupported_object"


def test_unresolved_dimension_is_reported_without_fabricated_measurement(tmp_path):
    graph = DrawingGraph.model_validate({
        "coordinate_space": "part_mm",
        "primitives": [{"id": "edge", "kind": "line", "start": [0, 0], "end": [10, 0]}],
        "dimensions": [{"id": "dim", "link_status": "unresolved"}],
    })
    target = tmp_path / "unresolved.dxf"
    issues = DxfExporter().export(graph, target)
    assert any(issue.code == "unresolved_dimension_link" for issue in issues)
    assert not ezdxf.readfile(target).modelspace().query("DIMENSION")

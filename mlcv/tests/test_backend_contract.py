"""cv.json against the backend contract (backend/app/drawing.py, branch feat/fastapi-dxf-backend): the backend
builds DrawingGraph from cv["primitives"], cv["hatches"], cv["issues"] with extra="forbid" models, so the
module may not put any other field into those lists. The models below mirror that file."""
import shutil
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from drawcv.adapter import run_cv

Point = tuple[float, float]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Issue(Strict):
    code: str
    severity: Literal["warning", "error"]
    message: str
    object_id: str | None = None


class Primitive(Strict):
    id: str
    kind: Literal["line", "circle", "arc", "polyline"]
    layer: Literal["MAIN", "THIN", "AXIS"] = "MAIN"
    start: Point | None = None
    end: Point | None = None
    center: Point | None = None
    radius: float | None = Field(default=None, gt=0)
    start_angle: float | None = None
    end_angle: float | None = None
    vertices: list[Point] | None = None
    closed: bool = False


class Hatch(Strict):
    id: str
    outer: list[Point] = Field(min_length=3)
    holes: list[list[Point]] = Field(default_factory=list)
    pattern: str = "ANSI31"
    scale: float = Field(default=1, gt=0)
    angle: float = 0


def check_contract(cv):
    assert cv["schema_version"] == "0.1" and cv["status"] == "ok"
    assert cv["coordinate_space"] in ("part_mm", "sheet_mm")
    assert cv["scale_status"] in ("known", "unknown")
    ids = [Primitive.model_validate(p).id for p in cv["primitives"]]
    ids += [Hatch.model_validate(h).id for h in cv["hatches"]]
    assert len(ids) == len(set(ids))
    for i in cv["issues"]:
        Issue.model_validate(i)
    for p in cv["primitives"]:
        if p["kind"] == "line":
            assert p["start"] and p["end"]
        if p["kind"] in ("circle", "arc"):
            assert p["center"] and p["radius"] > 0
        if p["kind"] == "arc":
            assert p["start_angle"] is not None and p["end_angle"] is not None


def test_adapter_on_backend_layout(synth_dir, tmp_path):
    """The backend calls run_cv(<run>/pages/page-001.png); review images go to <run>/cv/."""
    page = tmp_path / "jobs" / "j1" / "runs" / "0001" / "pages" / "page-001.png"
    page.parent.mkdir(parents=True)
    shutil.copy(synth_dir / "control.png", page)
    cv = run_cv(page)
    check_contract(cv)
    assert (page.parent.parent / "cv" / "overlay.png").exists()
    assert (page.parent.parent / "cv" / "cv.json").exists()


def test_adapter_error_is_returned(tmp_path):
    cv = run_cv(tmp_path / "missing.png")
    assert cv["status"] == "error" and cv["error"]
    assert cv["primitives"] == [] and cv["issues"] == []


@pytest.mark.parametrize("name", ["control.png", "control.pdf", "control_scan.jpg"])
def test_contract_all_inputs(results, name):
    cv, _ = results(name)
    check_contract(cv)


def test_points_for_ocr_links(results):
    """OCR links dimensions to measurable points by id: every visible primitive end/centre has a point."""
    cv, _ = results("control.png")
    pts = {p["id"]: p for p in cv["points"]}
    assert len(pts) == len(cv["points"])
    refs = {(r["object_id"], r["anchor"]) for p in cv["points"] for r in p["refs"]}
    for prim in cv["primitives"]:
        if prim["kind"] == "line":
            assert (prim["id"], "start") in refs and (prim["id"], "end") in refs
        else:
            assert (prim["id"], "center") in refs

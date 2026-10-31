import importlib
import json
import os
from pathlib import Path

from app.config import get_settings
from app.domain.drawing import DrawingGraph, Issue
from app.infra.storage import atomic_json
from app.infra.cv import run_builtin_recognition
from app.services.image import ImageService


def call_adapter(target: str, image: Path) -> dict:
    module_name, separator, function_name = target.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Adapter must be module:function")
    result = getattr(importlib.import_module(module_name), function_name)(image)
    if not isinstance(result, dict):
        raise ValueError("Adapter must return dict")
    return result


def assemble_graph(cv: dict, ocr: dict, source_sha256: str) -> DrawingGraph:
    issues = [Issue.model_validate(i) for i in cv.get("issues", []) + ocr.get("issues", [])]
    if cv.get("status") != "ok":
        issues.append(Issue(code="cv_error", severity="error", message=cv.get("error", "CV-адаптер недоступен")))
    if ocr.get("status") != "ok":
        issues.append(Issue(code="ocr_error", severity="error", message=ocr.get("error", "OCR-адаптер недоступен")))
    primitives = []
    for candidate in cv.get("primitives", []):
        if candidate.get("kind") in {"line", "circle", "arc", "polyline"}:
            metadata = cv.get("objects", {}).get(candidate.get("id"), {})
            primitives.append({**candidate, "linetype": candidate.get("linetype", metadata.get("linetype", "BYLAYER")).upper()})
        else:
            issues.append(Issue(code="unsupported_object", severity="warning", message=f"Неподдержанный CV-объект: {candidate.get('kind')}", object_id=candidate.get("id")))
    dimensions = []
    for candidate in ocr.get("dimensions", []):
        if candidate.get("kind", "linear") == "linear":
            dimensions.append(candidate)
        else:
            issues.append(Issue(code="unsupported_object", severity="warning",
                                message=f"Неподдержанный размер: {candidate.get('kind')}",
                                object_id=candidate.get("id")))
    scale_status = cv.get("scale_status", "unknown")
    if scale_status == "unknown":
        issues.append(Issue(code="unknown_scale", severity="error", message="Масштаб детали не определён"))
    hatch_links = {g["hatch_id"]: g for g in cv.get("hatch_groups", []) if g.get("hatch_id") and g.get("contour_id")}
    hatches = []
    for candidate in cv.get("hatches", []):
        group = hatch_links.get(candidate.get("id"))
        hatches.append({**candidate, **({"outer_ref": group["contour_id"],
                                       "hole_refs": group.get("hole_contour_ids", [])} if group else {})})
    return DrawingGraph(
        coordinate_space=cv.get("coordinate_space", "part_mm"), source_sha256=source_sha256, scale_status=scale_status,
        primitives=primitives, hatches=hatches, texts=ocr.get("texts", []), dimensions=dimensions, issues=issues,
        points=cv.get("points", []), contours=cv.get("contours", []), objects=cv.get("objects", {}),
        calibration=cv.get("calibration", {}), transforms=cv.get("transforms", {}), title_block=ocr.get("title_block", {}),
        provenance={"cv": {key: cv[key] for key in ("module", "source", "params", "preprocess", "stats", "summary", "timings_s", "text_regions", "defects", "hatch_groups") if key in cv},
                    "ocr": {key: ocr[key] for key in ("module", "source", "params", "timings_s") if key in ocr}},
    )


def recognize(source: Path, run_path: Path, source_sha256: str) -> DrawingGraph:
    image = ImageService().prepare(source, run_path)
    settings = get_settings()
    if settings.recognition_profile == "legacy":
        cv, ocr = run_builtin_recognition(image)
    else:
        def invoke(target: str | None) -> dict:
            if not target:
                return {"schema_version": "0.1", "status": "not_configured", "error": "Адаптер не настроен", "candidates": []}
            try:
                return call_adapter(target, image)
            except Exception as exc:
                return {"schema_version": "0.1", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        cv, ocr = invoke(settings.cv_adapter), invoke(settings.ocr_adapter)
    atomic_json(run_path / "cv.json", cv)
    atomic_json(run_path / "ocr.json", ocr)
    graph = assemble_graph(cv, ocr, source_sha256)
    graph.provenance["input"] = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
    if os.getenv("DRAWCV_MODEL_MANIFEST"):
        graph.provenance["models"] = json.loads(Path(os.environ["DRAWCV_MODEL_MANIFEST"]).read_text(encoding="utf-8"))
    return graph

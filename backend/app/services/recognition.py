import importlib
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
            primitives.append(candidate)
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
    return DrawingGraph(coordinate_space=cv.get("coordinate_space", "part_mm"), source_sha256=source_sha256, scale_status=scale_status,
                        primitives=primitives, hatches=cv.get("hatches", []), texts=ocr.get("texts", []),
                        dimensions=dimensions, issues=issues)


def recognize(source: Path, run_path: Path, source_sha256: str) -> DrawingGraph:
    image = ImageService().prepare(source, run_path)
    settings = get_settings()
    if not settings.cv_adapter and not settings.ocr_adapter:
        cv, ocr = run_builtin_recognition(image)
    else:
        def invoke(target: str | None) -> dict:
            if not target:
                return {"schema_version": "0.1", "status": "not_configured", "candidates": []}
            try:
                return call_adapter(target, image)
            except Exception as exc:
                return {"schema_version": "0.1", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        cv, ocr = invoke(settings.cv_adapter), invoke(settings.ocr_adapter)
    atomic_json(run_path / "cv.json", cv)
    atomic_json(run_path / "ocr.json", ocr)
    return assemble_graph(cv, ocr, source_sha256)

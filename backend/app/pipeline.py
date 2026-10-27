import importlib
import importlib.util
import os
import sys
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from .drawing import DrawingGraph, Issue
from .storage import atomic_json


def prepare_image(source: Path, run_path: Path) -> Path:
    target = run_path / "pages" / "page-001.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".pdf":
        pdf = pdfium.PdfDocument(str(source))
        if len(pdf) != 1:
            raise ValueError("Поддерживается только одностраничный PDF")
        pdf[0].render(scale=300 / 72).to_pil().convert("RGB").save(target)
    else:
        with Image.open(source) as image:
            image.convert("RGB").save(target)
    return target


def run_cv(image: Path) -> dict:
    return _invoke("CV_ADAPTER", image)


def run_ocr(image: Path) -> dict:
    return _invoke("OCR_ADAPTER", image)


def _invoke(setting: str, image: Path) -> dict:
    target = os.getenv(setting)
    if not target:
        return {"schema_version": "0.1", "status": "not_configured", "candidates": []}
    module_name, separator, function_name = target.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"{setting} must be module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    result = function(image)
    if not isinstance(result, dict):
        raise ValueError(f"{setting} must return dict")
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
    scale_status = cv.get("scale_status", "unknown")
    if scale_status == "unknown":
        issues.append(Issue(code="unknown_scale", severity="error", message="Масштаб детали не определён"))
    return DrawingGraph(coordinate_space=cv.get("coordinate_space", "part_mm"), source_sha256=source_sha256, scale_status=scale_status,
                        primitives=primitives, hatches=cv.get("hatches", []), texts=ocr.get("texts", []),
                        dimensions=ocr.get("dimensions", []), issues=issues)


def recognize(source: Path, run_path: Path, source_sha256: str) -> DrawingGraph:
    image = prepare_image(source, run_path)
    if not os.getenv("CV_ADAPTER") and not os.getenv("OCR_ADAPTER"):
        cv, ocr = run_builtin_recognition(image)
    else:
        try:
            cv = run_cv(image)
        except Exception as exc:
            cv = {"schema_version": "0.1", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        try:
            ocr = run_ocr(image)
        except Exception as exc:
            ocr = {"schema_version": "0.1", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    atomic_json(run_path / "cv.json", cv)
    atomic_json(run_path / "ocr.json", ocr)
    return assemble_graph(cv, ocr, source_sha256)


def run_builtin_recognition(image: Path) -> tuple[dict, dict]:
    root = Path(__file__).resolve().parents[2]
    if not (root / "runs").exists():
        root = Path(__file__).resolve().parents[1]
    folder = root / "runs" / "klapan_png2dxf" / "pipeline_b"
    if not (folder / "pipeline.py").is_file():
        raise RuntimeError("Встроенный модуль CV/OCR отсутствует")
    name = "drawing_prototype_pipeline"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder / "pipeline.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].run(str(image), None, observations_only=True)

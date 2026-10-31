"""The shared OCR entry point must preserve both deployment contracts."""
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "runs" / "klapan_png2dxf" / "pipeline_b"
sys.path.insert(0, str(PIPELINE))
ocr = importlib.import_module("ocr")


@pytest.fixture
def fake_rapidocr(monkeypatch):
    module = types.ModuleType("rapidocr")
    module.LangRec = types.SimpleNamespace(ESLAV="eslav")
    module.ModelType = types.SimpleNamespace(MOBILE="mobile")
    module.OCRVersion = types.SimpleNamespace(PPOCRV5="PP-OCRv5")
    module.RapidOCR = Mock()
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    monkeypatch.setattr(ocr, "_engine", None)
    monkeypatch.delenv("OCR_MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    return module.RapidOCR


def test_backend_recognizer_path_without_service_package(fake_rapidocr, monkeypatch):
    monkeypatch.setenv("OCR_MODEL_PATH", "/backend/recognizer.onnx")
    # The backend image deliberately does not copy the prototype's service/.
    with patch.dict(sys.modules, {"service.models": None}):
        first = ocr.engine()
        assert ocr.engine() is first
    fake_rapidocr.assert_called_once()
    params = fake_rapidocr.call_args.kwargs["params"]
    assert params["Rec.model_path"] == "/backend/recognizer.onnx"
    assert params["Rec.lang_type"] == "eslav"
    assert "Det.model_path" not in params


def test_offline_assets_take_precedence_over_backend_path(fake_rapidocr, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "/offline/models")
    monkeypatch.setenv("OCR_MODEL_PATH", "/backend/recognizer.onnx")
    paths = {"Det.model_path": "/offline/models/det.onnx",
             "Rec.model_path": "/offline/models/rec_eslav.onnx",
             "Cls.model_path": "/offline/models/cls.onnx",
             "Rec.rec_keys_path": "/offline/models/dict.txt"}
    with patch("service.models.rapidocr_params", return_value=paths.copy()) as check:
        ocr.engine()
    check.assert_called_once_with()
    params = fake_rapidocr.call_args.kwargs["params"]
    for name, path in paths.items():
        assert params[name] == path


@pytest.mark.parametrize("error", [FileNotFoundError("missing asset"), ValueError("bad hash")])
def test_offline_asset_error_never_falls_back(fake_rapidocr, monkeypatch, error):
    monkeypatch.setenv("MODEL_DIR", "/offline/models")
    monkeypatch.setenv("OCR_MODEL_PATH", "/backend/recognizer.onnx")
    with patch("service.models.rapidocr_params", side_effect=error):
        with pytest.raises(type(error), match=str(error)):
            ocr.engine()
    fake_rapidocr.assert_not_called()
    assert ocr._engine is None


def test_default_prototype_requires_local_assets(fake_rapidocr):
    with patch("service.models.rapidocr_params", side_effect=FileNotFoundError("missing asset")):
        with pytest.raises(FileNotFoundError, match="missing asset"):
            ocr.engine()
    fake_rapidocr.assert_not_called()


@pytest.mark.parametrize("on_recalibration", [False, True])
def test_strict_pipeline_propagates_ocr_errors(monkeypatch, tmp_path, on_recalibration):
    import numpy as np
    from runs.klapan_png2dxf.pipeline_b import pipeline

    gray = np.full((100, 100), 255, dtype=np.uint8)
    monkeypatch.setattr(pipeline, "load_gray", lambda path: gray)
    monkeypatch.setattr(pipeline, "binarize", lambda image: 255 - image)
    monkeypatch.setattr(pipeline, "estimate_skew", lambda image: 0)
    monkeypatch.setattr(pipeline, "calibrate", lambda image, sheet: {"sheet": "A4"})
    failure = RuntimeError("OCR inference failed")
    responses = [[types.SimpleNamespace(text="Формат A3")], failure] if on_recalibration else [failure]
    monkeypatch.setattr(pipeline, "run_ocr", Mock(side_effect=responses))
    target = tmp_path / "result.dxf"
    with pytest.raises(RuntimeError, match="OCR inference failed"):
        pipeline.run("input.png", target, strict_ocr=True)
    assert not target.exists()


def test_backend_observations_preserve_ocr_error(monkeypatch, tmp_path):
    import cv2
    import numpy as np
    from runs.klapan_png2dxf.pipeline_b import pipeline

    source = tmp_path / "input.png"
    image = np.full((200, 300), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (260, 160), 0, 2)
    source.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    monkeypatch.setattr(pipeline, "run_ocr", Mock(side_effect=RuntimeError("OCR unavailable")))
    cv, texts = pipeline.run(source, tmp_path / "unused.dxf", sheet="A4", observations_only=True)
    assert cv["status"] == "ok"
    assert texts["status"] == "error"
    assert "OCR unavailable" in texts["error"]
    assert not (tmp_path / "unused.dxf").exists()

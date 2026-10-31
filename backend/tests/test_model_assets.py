import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import get_settings
from app.infra.model_assets import configure_local_models
from service import models


def test_strict_assets_validate_dictionary_and_generate_local_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(get_settings(), "require_local_models", True)
    monkeypatch.setattr(get_settings(), "storage_root", tmp_path / "storage")
    payload = b"test-model"
    checksum = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(models, "MODEL_FILES", {key: (f"{key}.onnx", checksum) for key in ("det", "rec", "cls")})
    monkeypatch.setattr(models, "CV_MODEL_FILES", {"cv_det": ("cv_det.onnx", checksum)})
    for filename, _ in {**models.MODEL_FILES, **models.CV_MODEL_FILES}.values():
        (tmp_path / filename).write_bytes(payload)
    dictionary = tmp_path / models.DICT_NAME
    dictionary.write_text("abc", encoding="utf-8")
    assets = models.paths(check=True, include_cv=True)
    manifest = {key: {"filename": path.name, "bytes": path.stat().st_size, "sha256": models.digest(path)} for key, path in assets.items()}
    (tmp_path / "manifest.local.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("DRAWCV_RAPIDOCR_CONFIG", "DRAWCV_OFFLINE", "DRAWCV_STRICT_MODELS", "DRAWCV_MODEL_MANIFEST"):
        monkeypatch.setenv(name, "")
    configure_local_models()
    config = json.loads((tmp_path / "storage/.runtime/rapidocr.json").read_text(encoding="utf-8"))
    assert config["Det.model_path"] == str(tmp_path / "cv_det.onnx")
    assert config["Det.ocr_version"] == "PP-OCRv6"
    assert config["Rec.rec_keys_path"] == str(dictionary)
    dictionary.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        configure_local_models()


def test_offline_detector_does_not_fall_back_or_download(monkeypatch):
    from drawcv import textmask
    monkeypatch.setattr(textmask, "_engine", None)
    monkeypatch.setenv("DRAWCV_OFFLINE", "1")
    monkeypatch.delenv("DRAWCV_RAPIDOCR_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="requires DRAWCV_RAPIDOCR_CONFIG"):
        textmask.rapidocr_engine()


def test_json_model_config_is_converted_to_rapidocr_enums(tmp_path, monkeypatch):
    import rapidocr
    from drawcv import textmask
    asset = tmp_path / "asset"
    asset.write_bytes(b"test")
    config = {key: str(asset) for key in ("Det.model_path", "Cls.model_path", "Rec.model_path", "Rec.rec_keys_path")}
    config.update({"Det.ocr_version": "PP-OCRv5", "Rec.model_type": "mobile"})
    path = tmp_path / "params.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("DRAWCV_RAPIDOCR_CONFIG", str(path))
    monkeypatch.setattr(textmask, "_engine", None)
    observed = {}
    def create(*, params):
        observed.update(params)
        return object()
    monkeypatch.setattr(rapidocr, "RapidOCR", create)
    textmask.rapidocr_engine()
    assert observed["Det.ocr_version"] is rapidocr.OCRVersion.PPOCRV5
    assert observed["Rec.model_type"] is rapidocr.ModelType.MOBILE

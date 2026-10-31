"""Prepare drawcv's model configuration without downloading anything at runtime."""
import json
import os

from app.config import get_settings
from app.infra.storage import atomic_json


def configure_local_models():
    settings = get_settings()
    if not settings.require_local_models:
        return
    from service.models import digest, paths, rapidocr_params
    from service.settings import model_dir
    assets = paths(check=True, include_cv=True)
    manifest_path = model_dir() / "manifest.local.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {key: {"filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size}
              for key, path in assets.items()}
    if recorded != actual:
        raise ValueError("Local model manifest does not match assets (including dictionary)")
    config = settings.storage_root / ".runtime" / "rapidocr.json"
    params = rapidocr_params()
    params.update({"Det.model_path": str(assets["cv_det"]), "Det.ocr_version": "PP-OCRv6",
                   "Det.model_type": "small", "Det.limit_side_len": 5000})
    atomic_json(config, params)
    snapshot = settings.storage_root / ".runtime" / "models.json"
    atomic_json(snapshot, actual)
    os.environ["DRAWCV_MODEL_MANIFEST"] = str(snapshot.resolve())
    os.environ["DRAWCV_RAPIDOCR_CONFIG"] = str(config.resolve())
    os.environ["DRAWCV_OFFLINE"] = "1"
    os.environ["DRAWCV_STRICT_MODELS"] = "1"


if __name__ == "__main__":
    configure_local_models()

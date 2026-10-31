"""Fail-closed OCR model selection. No fallback to network/cache/site-packages."""
import hashlib
from pathlib import Path

from .settings import model_dir

# Upstream RapidOCR v3.9.2 onnx model catalog (ModelScope release).
MODEL_FILES = {
    "det": ("det.onnx", "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    "rec": ("rec_eslav.onnx", "08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab"),
    "cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
}
DICT_NAME = "ppocrv5_eslav_dict.txt"

def digest(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()

def paths(*, check: bool = True) -> dict[str, Path]:
    root = model_dir()
    result = {k: root / fn for k, (fn, _) in MODEL_FILES.items()}
    result["dict"] = root / DICT_NAME
    if check:
        for k, path in result.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Offline OCR asset missing: {path}")
            if k in MODEL_FILES and digest(path) != MODEL_FILES[k][1]:
                raise ValueError(f"OCR SHA256 mismatch: {path}")
    return result

def rapidocr_params() -> dict:
    p = paths(check=True)
    return {
        "Det.model_path": str(p["det"]),
        "Cls.model_path": str(p["cls"]),
        "Det.ocr_version": "PP-OCRv5",
        "Det.model_type": "mobile",
        "Rec.model_path": str(p["rec"]),
        "Rec.rec_keys_path": str(p["dict"]),
        "Rec.lang_type": "eslav",
        "Rec.ocr_version": "PP-OCRv5",
        "Rec.model_type": "mobile",
        "Global.use_cls": False,
        "Global.model_root_dir": str(model_dir()),
        "Det.limit_side_len": 6000,
        "Det.limit_type": "max",
        "Global.text_score": 0.3,
        "Global.log_level": "warning",
    }

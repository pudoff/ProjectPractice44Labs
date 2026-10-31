import importlib.util
import sys
from pathlib import Path

from app.config import get_settings


def run_builtin_recognition(image: Path) -> tuple[dict, dict]:
    path = get_settings().builtin_pipeline_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Встроенный модуль CV/OCR отсутствует: {path}")
    name = "drawing_prototype_pipeline"
    if name not in sys.modules:
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].run(str(image), None, observations_only=True)

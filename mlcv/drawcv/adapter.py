"""Backend entry point (backend/app/pipeline.py: CV_ADAPTER=drawcv.adapter:run_cv).

The backend passes the prepared page image <run>/pages/page-001.png and saves the returned dict as cv.json.
Review images are written next to it into <run>/cv/. Parameters can be overridden with DRAWCV_PARAMS
(path to a JSON file). Any failure is returned as status=error instead of raising, as the backend expects."""
import json
import os
import traceback
from pathlib import Path

from . import Params, __version__, run


def run_cv(image):
    image = Path(image)
    try:
        params = Params()
        if os.getenv("DRAWCV_PARAMS"):
            params = Params.from_dict(json.loads(Path(os.environ["DRAWCV_PARAMS"]).read_text(encoding="utf-8")))
        out_dir = image.parent.parent / "cv" if image.parent.name == "pages" else image.parent / f"{image.stem}_cv"
        return run(image, out_dir, params)
    except Exception as exc:
        return {"schema_version": "0.1", "status": "error", "module": {"name": "drawcv", "version": __version__},
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=5),
                "primitives": [], "hatches": [], "issues": []}

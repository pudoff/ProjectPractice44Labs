"""drawcv - OpenCV module of the drawing digitisation service: page preparation, primitives, contours, defects.

Stable call for other modules:
    from drawcv import run, Params
    cv = run("sheet.pdf", out_dir="out/sheet", params=Params())     # dict, also written to out/sheet/cv.json
Backend adapter: CV_ADAPTER=drawcv.adapter:run_cv"""
__version__ = "0.1.0"

from .params import Params  # noqa: E402
from .pipeline import run  # noqa: E402

__all__ = ["Params", "run", "__version__"]

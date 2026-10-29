"""Command line: python -m drawcv INPUT [INPUT ...] --out DIR [options]

Each input (PNG, JPEG or PDF page) gets DIR/<name>/ with cv.json, overlay.png, geometry.png, prep_compare.png
and params.json (the exact parameters of the run)."""
import argparse
import json
import sys
from pathlib import Path

from . import Params, __version__, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drawcv", description="OpenCV primitives/contours from a drawing page")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page", type=int, default=1, help="PDF page number (1-based)")
    ap.add_argument("--params", help="JSON file with parameters (see drawcv/params.py)")
    ap.add_argument("--sheet", choices=["A0", "A1", "A2", "A3", "A4"])
    ap.add_argument("--px-per-mm", type=float, help="manual calibration of the working image")
    ap.add_argument("--origin-px", help="manual sheet origin in working pixels, 'x,y'")
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--text-provider", choices=["auto", "rapidocr", "components", "file", "none"])
    ap.add_argument("--ocr-json", help="OCR result with text polygons in source pixels")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--version", action="version", version=f"drawcv {__version__}")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.params).read_text(encoding="utf-8")) if a.params else {}
    params = Params.from_dict(data)
    if a.sheet:
        params.sheet = a.sheet
    if a.px_per_mm:
        params.manual_px_per_mm = a.px_per_mm
    if a.origin_px:
        params.manual_origin_px = tuple(float(v) for v in a.origin_px.split(","))
    if a.rotate is not None:
        params.rotate_deg = a.rotate
    if a.text_provider:
        params.text_provider = a.text_provider
    if a.ocr_json:
        params.ocr_json = a.ocr_json
    if a.no_images:
        params.save_debug_images = False

    out = Path(a.out)
    status = 0
    for src in a.inputs:
        target = out / Path(src).stem
        target.mkdir(parents=True, exist_ok=True)
        (target / "params.json").write_text(json.dumps(params.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            cv = run(src, target, params, page_index=a.page - 1)
        except Exception as exc:
            status = 1
            print(f"{src}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        s = cv["summary"]
        print(f"{src}: {cv['calibration']['method']} {cv['scale_status']}, {s['exported_primitives']} primitives, "
              f"{s['closed_contours']} closed / {s['open_chains']} open contours, defects {s['defects']}, "
              f"{cv['timings_s']['total']} s -> {target}")
    return status


if __name__ == "__main__":
    sys.exit(main())

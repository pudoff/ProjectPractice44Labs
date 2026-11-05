"""Raw OCR readings of dataset sheets, saved for repeatable scoring without running the models again.

Runs where PaddleOCR is installed, i.e. in the service image (see mlocr/evaluation/README.md):
    python collect_observations.py --png DIR --out DIR NAME [NAME ...]"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from mlocr.src.cli import recognize_file

    for name in a.names:
        target = a.out / f"{name}.json"
        if target.exists():
            continue
        t0 = time.perf_counter()
        try:
            payload = recognize_file(a.png / f"{name}.png")
        except Exception as exc:
            print(name, "ERROR", type(exc).__name__, exc, flush=True)
            continue
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(name, len(payload["texts"]), round(time.perf_counter() - t0, 1), flush=True)


if __name__ == "__main__":
    sys.exit(main())

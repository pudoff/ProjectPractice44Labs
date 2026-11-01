"""Minimal offline OCR CLI for image crops and rendered PDF pages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from .semantics import classify_purpose, parse_engineering_notation
from .pdf_text import build_title_block, extract_pdf_text_page, merge_pdf_and_ocr
from .dimension_links import propose_dimension_links


def load_pages(path: Path, page_number: int | None, dpi: int) -> list[tuple[str, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = pymupdf.open(path)
        indexes = [page_number - 1] if page_number else list(range(len(doc)))
        if any(i < 0 or i >= len(doc) for i in indexes):
            raise ValueError(f"Page number out of range: {page_number}")
        scale = dpi / 72
        pages = []
        for i in indexes:
            pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append((f"page-{i + 1}", np.asarray(image)))
        return pages
    with Image.open(path) as image:
        return [(path.stem, np.asarray(image.convert("RGB")))]


def _box_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    ax0, ay0 = poly_a.min(axis=0)
    ax1, ay1 = poly_a.max(axis=0)
    bx0, by0 = poly_b.min(axis=0)
    bx1, by1 = poly_b.max(axis=0)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def _read_pass(engine, image: np.ndarray, pass_angle: int) -> list[dict]:
    height, _ = image.shape[:2]
    if pass_angle == 90:
        passed_image = np.ascontiguousarray(np.rot90(image, k=3))
    else:
        passed_image = image
    records = []
    for batch in engine.predict(passed_image):
        data = batch.json if hasattr(batch, "json") else batch
        if isinstance(data, dict):
            data = data.get("res", data)
        if not isinstance(data, dict):
            continue
        texts, scores = data.get("rec_texts", []), data.get("rec_scores", [])
        polys = data.get("dt_polys", data.get("rec_polys", []))
        for index, raw in enumerate(texts):
            text = str(raw).strip()
            if not text or index >= len(polys):
                continue
            polygon = np.asarray(polys[index], dtype=float)
            if pass_angle == 90:
                # Inverse of np.rot90(..., k=3), i.e. clockwise image rotation.
                polygon = np.column_stack((polygon[:, 1], height - 1 - polygon[:, 0]))
            score = float(scores[index]) if index < len(scores) else 0.0
            records.append({"text": text, "score": score, "polygon": polygon,
                            "pass_angle_deg": pass_angle})
    return records


def _merge_passes(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for record in records:
        duplicate = next((item for item in merged if _box_iou(item["polygon"], record["polygon"]) >= 0.45), None)
        alternative = {"text": record["text"], "score": record["score"],
                       "pass_angle_deg": record["pass_angle_deg"]}
        if duplicate is None:
            record["alternatives"] = [alternative]
            merged.append(record)
            continue
        if not any(a["text"] == record["text"] and a["pass_angle_deg"] == record["pass_angle_deg"]
                   for a in duplicate["alternatives"]):
            duplicate["alternatives"].append(alternative)
        if record["score"] > duplicate["score"]:
            alternatives = duplicate["alternatives"]
            duplicate.update(record)
            duplicate["alternatives"] = alternatives
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PaddleOCR on an image or PDF")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ocr.json"))
    parser.add_argument("--page", type=int, help="1-based page number for PDF")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rasterization resolution")
    parser.add_argument("--crop-origin", type=int, nargs=2, metavar=("X", "Y"),
                        help="Top-left pixel of an image crop within its source page")
    parser.add_argument("--source-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Full source-page pixel size when input is a crop")
    parser.add_argument("--cv-json", type=Path,
                        help="Optional CV JSON from mlcv; adds review-only dimension-link proposals")
    parser.add_argument("--det-model-dir", type=Path, required=True, help="Local PP-OCRv5 detection model directory")
    parser.add_argument("--rec-model-dir", type=Path, required=True, help="Local PP-OCRv5 East Slavic recognition model directory")
    args = parser.parse_args()
    if bool(args.crop_origin) != bool(args.source_size):
        parser.error("--crop-origin and --source-size must be provided together")

    for model_dir in (args.det_model_dir, args.rec_model_dir):
        if not model_dir.is_dir():
            parser.error(f"Local model directory does not exist: {model_dir}")
        if not any((model_dir / name).is_file() for name in ("inference.pdmodel", "inference.json")) or not (model_dir / "inference.pdiparams").is_file():
            parser.error(f"No complete local Paddle inference model found in: {model_dir}")

    from paddleocr import PaddleOCR

    # Explicit model paths prevent PaddleOCR from falling back to remote defaults.
    engine = PaddleOCR(device="cpu", text_detection_model_name="PP-OCRv5_mobile_det",
                       text_detection_model_dir=str(args.det_model_dir),
                       text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
                       text_recognition_model_dir=str(args.rec_model_dir),
                       use_doc_orientation_classify=False, use_doc_unwarping=False,
                       use_textline_orientation=False)
    result_pages = []
    for page_id, image in load_pages(args.input, args.page, args.dpi):
        height, width = image.shape[:2]
        observations = _merge_passes(_read_pass(engine, image, 0) + _read_pass(engine, image, 90))
        ocr_items = []
        source_width, source_height = args.source_size or (width, height)
        for item in observations:
            polygon = item["polygon"]
            if args.crop_origin:
                polygon += np.asarray(args.crop_origin, dtype=float)
            edges = np.roll(polygon, -1, axis=0) - polygon
            delta = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
            angle = float((np.degrees(np.arctan2(delta[1], delta[0])) + 90.0) % 180.0 - 90.0)
            polygon_list = polygon.tolist()
            ocr_items.append({"id": f"{page_id}-ocr-{len(ocr_items) + 1:04d}",
                               "page": page_id,
                               "text": item["text"], "score": item["score"],
                               "alternatives": item["alternatives"],
                               "selection_basis": "highest_recognition_confidence",
                               "angle_deg": angle,
                               "purpose": classify_purpose(item["text"], polygon_list, source_width, source_height),
                               "notation": parse_engineering_notation(item["text"]),
                               "polygon_px": polygon_list, "poly_source_px": polygon_list})
        if args.input.suffix.lower() == ".pdf":
            page_index = int(page_id.split("-")[-1]) - 1
            pdf_items = extract_pdf_text_page(args.input, page_index, args.dpi, width, height)
            page_items = merge_pdf_and_ocr(pdf_items, ocr_items)
        else:
            page_items = ocr_items
        result_pages.append({"page": page_id, "width_px": source_width, "height_px": source_height, "texts": page_items})

    payload = {"input": str(args.input), "model": "PaddleOCR PP-OCRv5 mobile + eslav mobile recognition", "language": "ru",
               "dpi": args.dpi if args.input.suffix.lower() == ".pdf" else None,
               "source_region_origin_px": args.crop_origin,
               "texts": [item for page in result_pages for item in page["texts"]],
               "pages": result_pages, "title_block": build_title_block(result_pages)}
    if args.cv_json:
        cv_data = json.loads(args.cv_json.read_text(encoding="utf-8"))
        payload["dimension_links"] = [link for page in result_pages
                                      for link in propose_dimension_links(page, cv_data)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

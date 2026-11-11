"""Build a human review sheet from OCR runs on mlocr/qa_samples."""
import argparse
import csv
import json
import re
from pathlib import Path


REVIEW_KINDS = {"diameter", "diameter_candidate", "radius", "thread", "linear_tolerance"}
SYMBOLS = re.compile(r"[mмтMØø⌀∅φΦФф$]|(?:^|\D)[08](?:\D|$)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("mlocr/results/qa_samples"))
    parser.add_argument("--images", type=Path, default=Path("mlocr/qa_samples"))
    parser.add_argument("--output", type=Path, default=Path("mlocr/qa_samples/qa_review.csv"))
    args = parser.parse_args()

    fields = ["image", "page", "text_id", "raw_text", "normalized_text", "kind", "nominal",
              "confidence", "bbox_xyxy_px", "corrections", "needs_review", "ground_truth", "qa_note"]
    rows = []
    for result_path in sorted(args.input.glob("*.ocr.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        image_name = Path(result["input"]).name
        for item in result.get("texts", []):
            raw = item.get("raw_text", "")
            notation = item.get("notation", {})
            corrections = item.get("corrections", [])
            if (notation.get("kind") not in REVIEW_KINDS
                    and not corrections
                    and not SYMBOLS.search(raw)):
                continue
            points = item.get("poly_source_px", [])
            bbox = ""
            if points:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                bbox = f"{min(xs):.1f},{min(ys):.1f},{max(xs):.1f},{max(ys):.1f}"
            rows.append({
                "image": image_name,
                "page": item.get("page", "page-1"),
                "text_id": item.get("id", ""),
                "raw_text": raw,
                "normalized_text": item.get("text", ""),
                "kind": notation.get("kind", "unknown"),
                "nominal": notation.get("nominal"),
                "confidence": item.get("score"),
                "bbox_xyxy_px": bbox,
                "corrections": json.dumps(corrections, ensure_ascii=False),
                "needs_review": item.get("needs_review", False),
                "ground_truth": "",
                "qa_note": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"review_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

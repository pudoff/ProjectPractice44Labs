"""Reapply current text and pixel-based OCR corrections to the reviewed QA worksheet."""
import argparse
import csv
import json
from pathlib import Path

from mlocr.src.glyphs import recover_diameter
from mlocr.src.inputs import assert_same_page, iter_pages, page_identity
from mlocr.src.normalization import enrich


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("mlocr/results/qa_samples"))
    parser.add_argument("--images", type=Path, default=Path("mlocr/qa_samples"))
    parser.add_argument("--worksheet", type=Path, default=Path("mlocr/qa_samples/qa_review.csv"))
    parser.add_argument('--output', type=Path, required=True, help='New worksheet; the baseline is never overwritten')
    args = parser.parse_args()
    if args.output.resolve() == args.worksheet.resolve() or args.output.exists():
        parser.error('--output must be a new file distinct from the input worksheet')

    with args.worksheet.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('QA worksheet is empty')
    by_key = {(row["image"], row["text_id"]): row for row in rows}

    result_paths = sorted(args.results.glob('*.ocr.json'))
    if not result_paths:
        parser.error('No *.ocr.json files found; worksheet was not modified')
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        image_name = Path(payload["input"].replace('\\', '/')).name
        image_path = args.images / image_name
        _, image = next(iter_pages(image_path, 1, 300))
        assert_same_page(payload, {'raster_identity': page_identity(image)})
        for observation in payload.get("texts", []):
            key = (image_name, observation["id"])
            row = by_key.get(key)
            if row is None:
                continue
            record = enrich({
                "text": observation.get("raw_text", observation["text"]),
                "needs_review": observation.get("needs_review", False),
                "issues": observation.get("issues", []),
            })
            recover_diameter(record, image, observation["poly_source_px"])
            row["normalized_after"] = record["text"]
            row["corrections_after"] = json.dumps(record["corrections"], ensure_ascii=False)
            row["review_after"] = str(bool(record.get("needs_review"))).lower()
        del image

    fields = list(rows[0])
    for name in ("normalized_after", "corrections_after", "review_after"):
        if name not in fields:
            fields.append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    changed = [row for row in rows if row["normalized_text"] != row["normalized_after"]]
    print(f"rows={len(rows)} changed_after={len(changed)}")
    for row in changed:
        print(f"{row['image']} | {row['text_id']} | {row['normalized_text']} -> {row['normalized_after']}")


if __name__ == "__main__":
    main()

"""Match the reviewed QA crops to a fresh OCR run and score task 1 corrections."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from mlocr.src.glyphs import recover_diameter
from mlocr.src.normalization import enrich


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def bounds(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union else 0.0


def main():
    qa_path = ROOT / "qa_samples/qa_review.csv"
    results_dir = ROOT / "results/qa_samples/task1_final"
    output_dir = ROOT / "results/qa_samples"
    with qa_path.open(encoding="utf-8-sig", newline="") as stream:
        qa_rows = list(csv.DictReader(stream))
    results = {}
    for path in results_dir.glob("*.ocr.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        image_path = ROOT / "qa_samples" / Path(payload["input"]).name
        Image.MAX_IMAGE_PIXELS = 300_000_000
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        for item in payload.get("texts", []):
            raw = item.get("raw_text", item["text"])
            record = enrich({"text": raw, "raw_text": raw,
                             "needs_review": item.get("needs_review", False),
                             "issues": item.get("issues", [])})
            recover_diameter(record, image, item["poly_source_px"])
            item.update(text=record["text"], corrections=record["corrections"],
                        notation=record["notation"], needs_review=record.get("needs_review", False),
                        issues=record.get("issues", []))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        del image
        results[Path(payload["input"]).name] = payload

    comparison = []
    split_source_count = 0
    for row in qa_rows:
        page = results[row["image"]]
        target = tuple(map(float, row["bbox_xyxy_px"].split(",")))
        texts = page["texts"]
        split = [item for item in texts if item.get("split_from") and
                 iou(target, bounds(item["split_from"]["polygon"])) >= .8]
        if split:
            chosen = sorted(split, key=lambda item: bounds(item["poly_source_px"])[0])
            predicted = " ".join(item["text"] for item in chosen)
            method = "gridline_split"
            overlap = iou(target, bounds(chosen[0]["split_from"]["polygon"]))
            split_source_count += 1
        else:
            candidates = [(iou(target, bounds(item["poly_source_px"])), item)
                          for item in texts if item.get("poly_source_px")]
            overlap, item = max(candidates, key=lambda pair: pair[0], default=(0.0, None))
            chosen = [item] if item and overlap >= .15 else []
            predicted = item["text"] if chosen else ""
            method = "polygon_iou" if chosen else "unmatched"

        corrections = [change for item in chosen for change in item.get("corrections", [])]
        issues = list(dict.fromkeys(issue["code"] for item in chosen for issue in item.get("issues", [])))
        scores = [item["score"] for item in chosen]
        angles = []
        for item in chosen:
            angle = item.get("selected_pass_angle_deg")
            if angle is None and item.get("alternatives"):
                angle = max(item["alternatives"], key=lambda alt: alt["score"]).get("pass_angle_deg")
            if angle is not None:
                angles.append(angle)

        row["normalized_after"] = predicted
        row["corrections_after"] = json.dumps(corrections, ensure_ascii=False)
        row["review_after"] = str(any(item.get("needs_review") for item in chosen)).lower()
        row["selected_read_after"] = json.dumps([item.get("raw_text", item["text"]) for item in chosen], ensure_ascii=False)
        row["selected_pass_angle_deg"] = json.dumps(angles)
        row["alternatives_after"] = json.dumps([item.get("alternatives", []) for item in chosen], ensure_ascii=False)
        truth = row.get("ground_truth", "").strip()
        comparison.append({
            "image": row["image"], "page": row["page"], "text_id": row["text_id"],
            "initial_raw": row["raw_text"], "initial_after_rules": row["normalized_text"],
            "final_after_rules": predicted, "ground_truth": truth,
            "exact_match": str(predicted.strip() == truth).lower() if truth else "unscored_no_ground_truth",
            "match_method": method, "bbox_iou": round(overlap, 3),
            "confidence_min": round(min(scores), 3) if scores else "",
            "selected_pass_angle_deg": json.dumps(angles), "issues": json.dumps(issues),
            "needs_review": row["review_after"], "qa_note": row["qa_note"],
        })

    fields = list(qa_rows[0])
    for field in ("selected_read_after", "selected_pass_angle_deg", "alternatives_after"):
        if field not in fields:
            fields.append(field)
    with qa_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(qa_rows)
    comparison_path = output_dir / "task1_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    labelled = [row for row in comparison if row["ground_truth"]]
    diameter = [row for row in labelled if "Ø" in row["ground_truth"]]
    diameter_true = [row for row in diameter if "Ø" in row["final_after_rules"]]
    diameter_missed = [row for row in diameter if "Ø" not in row["final_after_rules"]]
    diameter_false = [row for row in labelled if "Ø" not in row["ground_truth"] and
                      "Ø" in row["final_after_rules"]]
    qa_by_key = {(row["image"], row["text_id"]): row for row in qa_rows}
    diameter_recovered = [row for row in diameter_true if any(
        change.get("rule") in {"engineering_diameter_glyph", "diameter_visible_in_raster"}
        for change in json.loads(qa_by_key[(row["image"], row["text_id"])]["corrections_after"]))]
    m_rules = [row for row, qa in zip(comparison, qa_rows) if "cyrillic_italic_t" in qa["corrections_after"]
               and row["ground_truth"]]
    orientation_conflicts = [row for row in comparison if "ocr_reading_conflict" in row["issues"]]
    orientation_unflagged = [row for row in orientation_conflicts if row["needs_review"] != "true"]
    pass_failures = {name: [page["pass_failures"] for page in payload["pages"] if page.get("pass_failures")]
                     for name, payload in results.items()}
    digit_rows = [row for row in labelled if "0" in row["initial_raw"] + row["ground_truth"] or
                  "8" in row["initial_raw"] + row["ground_truth"]]
    split_observations = sum(bool(item.get("split_from")) for payload in results.values()
                             for item in payload["texts"])
    metrics = {
        "sheets": len(results), "qa_rows": len(comparison),
        "matched_rows": sum(row["match_method"] != "unmatched" for row in comparison),
        "labelled_rows": len(labelled), "exact_text_rows": sum(row["exact_match"] == "true" for row in labelled),
        "exact_text_accuracy": round(sum(row["exact_match"] == "true" for row in labelled) / len(labelled), 4),
        "diameter_ground_truth_rows": len(diameter), "diameter_correct": len(diameter_true),
        "diameter_corrected_by_symbol_rule": len(diameter_recovered),
        "diameter_missed": len(diameter_missed), "false_diameter_additions": len(diameter_false),
        "labelled_rows_with_0_or_8": len(digit_rows),
        "exact_rows_with_0_or_8": sum(row["exact_match"] == "true" for row in digit_rows),
        "labelled_italic_t_corrections": len(m_rules),
        "italic_t_exact_rows": sum(row["exact_match"] == "true" for row in m_rules),
        "italic_t_wrong_rows": sum(row["exact_match"] != "true" for row in m_rules),
        "orientation_conflict_rows": len(orientation_conflicts),
        "orientation_conflicts_not_flagged": len(orientation_unflagged),
        "gridline_split_observations": split_observations,
        "qa_rows_matched_to_split_regions": split_source_count,
        "orientation_pass_failures": pass_failures,
    }
    (output_dir / "task1_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"comparison={comparison_path}")


if __name__ == "__main__":
    main()

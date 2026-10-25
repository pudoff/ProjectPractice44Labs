"""Run pipeline B on dataset sheets (not the test sheet) and compare with their reference DXFs."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
PIPE = ROOT / "runs/klapan_png2dxf/pipeline_b/pipeline.py"
CMP = ROOT / "tools/compare_dxf.py"
OUT = Path(__file__).resolve().parent / next((a[2:] for a in sys.argv[1:] if a.startswith("--batch_")), "batch_b")
SHEETS = [a for a in sys.argv[1:] if not a.startswith("--")] or ["корпус клапана", "Корпус6", "Кран", "Фланец", "Опора3", "Вал2", "фланец3", "Сфера1", "Пружина"]


def metrics(md):
    rows = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "|" in line[2:]:
            k, v = [c.strip() for c in line.strip("|").split("|")[:2]]
            rows[k] = v
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    res = []
    for name in SHEETS:
        png = ROOT / "dataset/clear_sum_dataset/vse_png" / f"{name}.png"
        ref = ROOT / "dataset/clear_sum_dataset/vse_dxf" / f"{name}.dxf"
        dxf = OUT / f"{name}.dxf"
        t = time.time()
        p = subprocess.run([PY, str(PIPE), str(png), str(dxf), "--report", str(OUT / f"{name}.run.json")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        el = round(time.time() - t, 1)
        row = {"sheet": name, "wall_s": el, "ok": p.returncode == 0}
        if p.returncode != 0:
            row["error"] = p.stderr.strip().splitlines()[-1][:200] if p.stderr else "?"
            res.append(row)
            continue
        run = json.load(open(OUT / f"{name}.run.json", encoding="utf-8"))
        row.update({"sheet_fmt": run["calibration"]["sheet"], "fmt_ocr": run["format_from_ocr"],
                    "calibrated_on": run["calibration"]["calibrated_on"], "counts": run["counts"],
                    "dims_with_value": run["dims_with_value"], "title": run["title_fields"]})
        c = subprocess.run([PY, str(CMP), str(ref), str(dxf), "--out", str(OUT / f"cmp_{name}"), "--align", "geom"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if c.returncode == 0:
            row["cmp"] = metrics(OUT / f"cmp_{name}" / "report.md")
        else:
            row["cmp_error"] = c.stderr.strip().splitlines()[-1][:200]
        res.append(row)
        print(name, el, "s", row.get("cmp", {}).get("geometry F1 @0.5 mm"), flush=True)
    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""Re-run compare_dxf.py (geom alignment) for a batch folder produced by run_b_batch.py and print a table."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CMP = ROOT / "tools/compare_dxf.py"
KEYS = ["geometry F1 @0.25 mm", "geometry F1 @0.5 mm", "geometry F1 @1.0 mm", "primitive recall / precision", "dimension recall by value (notation ignored)",
        "dimension recall", "text exact / mean CER (matched by line centre)", "layer class accuracy", "hatch area ref / gen"]


def metrics(md):
    rows = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "|" in line[2:]:
            k, v = [c.strip() for c in line.strip("|").split("|")[:2]]
            rows[k] = v
    return rows


def main(folder):
    out = HERE / folder
    s = json.load(open(out / "summary.json", encoding="utf-8"))
    for row in s:
        n = row["sheet"]
        ref = ROOT / "dataset/clear_sum_dataset/vse_dxf" / f"{n}.dxf"
        subprocess.run([sys.executable, str(CMP), str(ref), str(out / f"{n}.dxf"), "--out", str(out / f"cmp_{n}"),
                        "--align", "geom"], capture_output=True)
        row["cmp"] = metrics(out / f"cmp_{n}" / "report.md")
    json.dump(s, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = ["| лист | время, с | откалибровано по | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 3)]
    for r in s:
        run = json.load(open(out / f"{r['sheet']}.run.json", encoding="utf-8"))
        cal = run["calibration"]["calibrated_on"]
        lines.append(f"| {r['sheet']} | {r['wall_s']} | {cal} | " + " | ".join(r["cmp"].get(k, "") for k in KEYS) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])

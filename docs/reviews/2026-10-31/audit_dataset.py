"""Read-only inventory of the updated CAD dataset; outputs review evidence only."""
import collections
import hashlib
import json
from pathlib import Path

import ezdxf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    base = ROOT / "dataset" / "clear_sum_dataset"
    dxfs = sorted((base / "vse_dxf").glob("*.dxf"))
    pngs = sorted((base / "vse_png").glob("*.png"))
    dnames = {p.stem.casefold(): p.name for p in dxfs}
    pnames = {p.stem.casefold(): p.name for p in pngs}
    hashes = {"dxf": collections.defaultdict(list), "png": collections.defaultdict(list)}
    records, errors = [], []
    for kind, paths in (("dxf", dxfs), ("png", pngs)):
        for i, path in enumerate(paths, 1):
            hashes[kind][digest(path)].append(path.name)
            if kind == "dxf":
                try:
                    doc = ezdxf.readfile(path)
                    model = collections.Counter(e.dxftype() for e in doc.modelspace())
                    layouts = collections.Counter(e.dxftype() for layout in doc.layouts for e in layout)
                    blocks = collections.Counter(e.dxftype() for block in doc.blocks for e in block)
                    records.append({"name": path.name, "version": doc.dxfversion,
                                    "units": int(doc.units), "modelspace": dict(model),
                                    "layout_dimensions": layouts["DIMENSION"],
                                    "block_dimensions": blocks["DIMENSION"]})
                except Exception as exc:
                    errors.append({"name": path.name, "error": f"{type(exc).__name__}: {exc}"})
            if i % 100 == 0:
                print(f"Read {i}/{len(paths)} {kind}", flush=True)
    duplicates = {kind: [names for names in groups.values() if len(names) > 1]
                  for kind, groups in hashes.items()}
    summary = {
        "dxf_files": len(dxfs), "png_files": len(pngs),
        "same_stem_pairs_casefold": len(dnames.keys() & pnames.keys()),
        "dxf_without_png": [dnames[k] for k in sorted(dnames.keys() - pnames.keys())],
        "png_without_dxf": [pnames[k] for k in sorted(pnames.keys() - dnames.keys())],
        "dxf_read_ok": len(records), "dxf_read_errors": errors,
        "dxf_with_modelspace_dimensions": sum(r["modelspace"].get("DIMENSION", 0) > 0 for r in records),
        "dxf_with_dimensions_anywhere": sum(r["layout_dimensions"] + r["block_dimensions"] > 0 for r in records),
        "dxf_with_modelspace_hatch": sum(r["modelspace"].get("HATCH", 0) > 0 for r in records),
        "units_histogram": dict(collections.Counter(r["units"] for r in records)),
        "versions": dict(collections.Counter(r["version"] for r in records)),
        "exact_duplicate_groups": {k: len(v) for k, v in duplicates.items()},
        "exact_duplicate_extra_files": {k: sum(len(g) - 1 for g in v) for k, v in duplicates.items()},
    }
    (OUT / "dataset-audit.json").write_text(json.dumps({"summary": summary,
        "duplicates": duplicates, "dxf_inventory": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

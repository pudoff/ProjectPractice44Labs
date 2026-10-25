import collections
import json
import sys
from pathlib import Path

import ezdxf

ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = {"клапан.dxf", "клапан.pdf"}
OUT = Path(__file__).with_name("dataset_index.json")


def summarize(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    types = collections.Counter(e.dxftype() for e in msp)
    layers = collections.Counter(e.dxf.layer for e in msp)
    styles = sorted({(s.dxf.name, s.dxf.font) for s in doc.styles})
    dims = collections.Counter()
    for d in msp.query("DIMENSION"):
        dims[d.dxf.dimtype & 7] += 1
    hatch = [h.dxf.pattern_name for h in msp.query("HATCH")]
    return {
        "version": doc.dxfversion,
        "codepage": doc.header.get("$DWGCODEPAGE"),
        "insunits": doc.header.get("$INSUNITS"),
        "types": dict(types),
        "layers": dict(layers),
        "linetypes": sorted(lt.dxf.name for lt in doc.linetypes),
        "styles": styles,
        "dimtypes": dict(dims),
        "hatch_patterns": dict(collections.Counter(hatch)),
    }


def main():
    files = sorted((ROOT / "dataset/clear_sum_dataset/vse_dxf").glob("*.dxf"))
    files += sorted((ROOT / "dataset/data/dxf").glob("*.dxf"))[:5]
    res = {}
    for f in files:
        if f.name in FORBIDDEN:
            continue
        try:
            res[str(f.relative_to(ROOT))] = summarize(f)
        except Exception as ex:
            res[str(f.relative_to(ROOT))] = {"error": str(ex)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    agg = collections.Counter()
    lay = collections.Counter()
    fonts = collections.Counter()
    pats = collections.Counter()
    ver = collections.Counter()
    for k, v in res.items():
        if "error" in v:
            continue
        ver[v["version"]] += 1
        agg.update(v["types"])
        lay.update(v["layers"].keys())
        fonts.update(f for _, f in v["styles"])
        pats.update(v["hatch_patterns"])
    with open(OUT.with_suffix(".md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Индекс датасета ({len(res)} DXF, без эталона клапана)\n\n")
        fh.write(f"Версии: {dict(ver)}\n\nТипы примитивов: {dict(agg.most_common())}\n\n")
        fh.write(f"Слои (в скольких файлах): {dict(lay.most_common(40))}\n\n")
        fh.write(f"Шрифты стилей: {dict(fonts.most_common(15))}\n\nШтриховки: {dict(pats.most_common())}\n")
    print("ok", len(res))


if __name__ == "__main__":
    main()

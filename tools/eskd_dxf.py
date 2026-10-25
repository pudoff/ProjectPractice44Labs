"""ESKD-flavoured DXF document factory + verifier (ezdxf).

Import from a build script:

    import sys; sys.path.insert(0, r"tools")
    from eskd_dxf import new_drawing, verify, SHEETS

    doc = new_drawing()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "MAIN"})
    d = msp.add_linear_dim(base=(0, 10), p1=(0, 0), p2=(10, 0), dimstyle="ESKD",
                           text="%%c<>", dxfattribs={"layer": "DIMS"})
    d.render()                       # ALWAYS render, otherwise there is no *D block
    doc.saveas(out_path)             # R2007 => file is written as UTF-8
    print(verify(out_path))

Layer contract (task spec: MAIN/THIN/AXIS/DIMS/TEXT):
    MAIN  - visible contour, Continuous, 0.50 mm
    THIN  - thin lines, hatch, breaks, leaders; hidden lines = THIN + linetype DASHED
    AXIS  - centre/axis lines, linetype CENTER
    DIMS  - DIMENSION, LEADER, roughness/tolerance symbols
    TEXT  - notes, title block text, view labels
"""
import ezdxf
from ezdxf.lldxf.const import DXFValueError

SHEETS = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (210, 297)}

# GOST 2.303 sized patterns in mm (dash > 0, gap < 0, 0 = dot). Names kept standard so
# LibreCAD/AutoCAD recognise them; definitions are embedded so AutoCAD uses these sizes.
LINETYPES = {
    "CENTER": ("Axis ____ _ ____ _ (GOST 2.303)", [24.0, 18.0, -2.5, 1.0, -2.5]),
    "DASHED": ("Hidden __ __ __ (GOST 2.303)", [6.0, 4.5, -1.5]),
    "PHANTOM": ("Phantom ____ _ _ ____ (GOST 2.303)", [30.0, 20.0, -2.5, 1.0, -2.5, 1.0, -2.5]),
}

LAYERS = {
    # name: (ACI color, linetype, lineweight in 1/100 mm)
    "MAIN": (7, "Continuous", 50),
    "THIN": (3, "Continuous", 25),
    "AXIS": (1, "CENTER", 25),
    "DIMS": (4, "Continuous", 25),
    "TEXT": (7, "Continuous", 25),
}


def new_drawing(font="isocpeur.ttf", text_height=3.5, oblique=15.0, width=1.0,
                dim_text_height=3.5, arrow=3.0):
    """Create an R2007 (AC1021) mm drawing with ESKD layers, text style and dimstyle.

    font: a font that ships with AutoCAD AND contains Cyrillic. isocpeur.ttf (ISO, upright,
    oblique applied via style) is the safe default; GOST fonts (gost_b.ttf, GOST_AU.SHX)
    are NOT installed on a stock AutoCAD and would be substituted.
    """
    doc = ezdxf.new("R2007", setup=False, units=4)
    h = doc.header
    h["$INSUNITS"] = 4          # millimetres
    h["$MEASUREMENT"] = 1       # metric
    h["$LUNITS"] = 2            # decimal
    doc.encoding = "cp1251"     # ezdxf rewrites $DWGCODEPAGE from doc.encoding on save; R2007+ stays UTF-8
    h["$DWGCODEPAGE"] = "ANSI_1251"
    h["$LWDISPLAY"] = 1
    h["$LTSCALE"] = 1.0
    h["$CELTSCALE"] = 1.0
    h["$DIMDSEP"] = ord(",")

    for name, (desc, pattern) in LINETYPES.items():
        if name in doc.linetypes:
            doc.linetypes.remove(name)
        doc.linetypes.add(name, pattern=pattern, description=desc)

    for name, (color, lt, lw) in LAYERS.items():
        doc.layers.add(name, color=color, linetype=lt, lineweight=lw)

    for sname in ("Standard", "ESKD"):
        st = doc.styles.get(sname) if sname in doc.styles else doc.styles.add(sname, font=font)
        st.dxf.font = font
        st.dxf.width = width
        st.dxf.oblique = oblique
        st.dxf.height = 0.0     # variable height; set per entity
    h["$TEXTSTYLE"] = "ESKD"

    ds = doc.dimstyles.add("ESKD")
    ds.dxf.dimtxsty = "ESKD"
    ds.dxf.dimtxt = dim_text_height
    ds.dxf.dimasz = arrow
    ds.dxf.dimexe = 2.0          # extension beyond dim line
    ds.dxf.dimexo = 0.0          # offset from object
    ds.dxf.dimgap = 1.0
    ds.dxf.dimtad = 1            # text above line
    ds.dxf.dimtih = 0            # text aligned with dim line
    ds.dxf.dimtoh = 0
    ds.dxf.dimdsep = ord(",")    # decimal comma (GOST)
    ds.dxf.dimzin = 8            # suppress trailing zeros: 5,5 not 5,50
    ds.dxf.dimadec = 0
    ds.dxf.dimdec = 2
    ds.dxf.dimlfac = 1.0
    ds.dxf.dimscale = 1.0
    ds.dxf.dimlwd = 25
    ds.dxf.dimlwe = 25
    ds.dxf.dimclrd = 256         # BYLAYER
    ds.dxf.dimclre = 256
    ds.dxf.dimclrt = 256
    h["$DIMSTYLE"] = "ESKD"
    # LibreCAD regenerates DIMENSION graphics from the header $DIM* variables, not from the
    # DIMSTYLE table (verified: dim text came out 2.5 high with dimtxt=5 in the style only).
    for var in ("dimtxt", "dimasz", "dimexe", "dimexo", "dimgap", "dimtad", "dimtih", "dimtoh",
                "dimdsep", "dimzin", "dimdec", "dimlfac", "dimscale"):
        h["$" + var.upper()] = ds.dxf.get(var)
    h["$DIMTXSTY"] = "ESKD"
    return doc


def verify(path, expect_version="AC1021"):
    """Re-read a written DXF and return a list of problems ([] == OK)."""
    problems = []
    raw = open(path, "rb").read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as ex:
        problems.append(f"file is not valid UTF-8: {ex}")
    if b"\\U+04" in raw:
        problems.append("Cyrillic written as \\U+ escapes (expected raw UTF-8 for R2007+)")
    doc = ezdxf.readfile(path)
    if doc.dxfversion != expect_version:
        problems.append(f"version {doc.dxfversion} != {expect_version}")
    if doc.header.get("$DWGCODEPAGE") != "ANSI_1251":
        problems.append(f"$DWGCODEPAGE is {doc.header.get('$DWGCODEPAGE')!r} (set doc.encoding='cp1251')")
    if doc.header.get("$INSUNITS") != 4:
        problems.append("$INSUNITS is not 4 (mm)")
    auditor = doc.audit()
    for err in auditor.errors:
        problems.append(f"audit error: {err.message}")
    msp = doc.modelspace()
    allowed = set(LAYERS) | {"0", "Defpoints", "DEFPOINTS"}
    for e in msp:
        if e.dxf.layer not in allowed:
            problems.append(f"{e.dxftype()} on unexpected layer {e.dxf.layer!r}")
        if e.dxftype() == "DIMENSION" and e.get_geometry_block() is None:
            problems.append("DIMENSION without rendered block (call .render())")
        if e.dxftype() in ("TEXT", "MTEXT"):
            s = e.dxf.text if e.dxftype() == "TEXT" else e.text
            if "�" in s or "?" * 3 in s:
                problems.append(f"suspicious text {s!r}")
    for st in doc.styles:
        if not st.dxf.font:
            problems.append(f"text style {st.dxf.name} has no font")
    return problems


if __name__ == "__main__":
    import sys
    for p in verify(sys.argv[1]) or ["OK"]:
        print(p)

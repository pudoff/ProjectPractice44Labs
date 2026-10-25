"""Variant A: parametric reconstruction of 'Клапан предохранительный' (A3, 1:1) as an ESKD DXF.

Geometry is built from the dimension values written on the sheet and drafting rules
(tangent fillets, 60° cone, symmetry about the axis). The raster was used only to anchor
each view (one measured point per view) and to read topology/placement of annotations.
Coordinates: sheet mm, origin = lower-left corner of the A3 sheet trim line.
"""
import math
import sys
from pathlib import Path

import numpy as np
from ezdxf.math import Vec2, Vec3
from ezdxf.render.arrows import ARROWS
from ezdxf.enums import TextEntityAlignment as TA

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(RUN / "work"))
from eskd_dxf import new_drawing, verify  # noqa: E402
from geom import ang, fillet_ll, unit  # noqa: E402

OUT = RUN / "out" / "klapan_variant_A.dxf"

doc = new_drawing(font="isocpeur.ttf", oblique=15.0, dim_text_height=5.0, arrow=3.0)
doc.linetypes.remove("CENTER")
doc.linetypes.add("CENTER", pattern=[10.0, 6.0, -1.5, 1.0, -1.5], description="Axis ____ _ ____ _ (GOST 2.303)")
msp = doc.modelspace()

# ---------------------------------------------------------------- primitives

def line(p, q, layer="MAIN", **kw):
    return msp.add_line(tuple(map(float, p)), tuple(map(float, q)), dxfattribs={"layer": layer, **kw})


def arc(c, r, a0, a1, layer="MAIN"):
    """CCW arc from a0 to a1 (degrees)."""
    return msp.add_arc(tuple(map(float, c)), float(r), float(a0) % 360, float(a1) % 360, dxfattribs={"layer": layer})


def circle(c, r, layer="THIN"):
    return msp.add_circle(tuple(map(float, c)), float(r), dxfattribs={"layer": layer})


def rect(x0, y0, x1, y1, layer="MAIN"):
    for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        line(p, q, layer)


def text(s, x, y, h, rot=0.0, layer="TEXT", align=None, width=None):
    t = msp.add_text(s, height=h, rotation=rot, dxfattribs={"layer": layer, "style": "ESKD"})
    if width:
        t.dxf.width = width
    if align:
        t.set_placement((x, y), align=align)
    else:
        t.set_placement((x, y))
    return t


def bulge_center(p, q, b):
    p, q = np.asarray(p, float), np.asarray(q, float)
    theta = 4 * math.atan(b)
    chord = np.linalg.norm(q - p)
    r = chord / (2 * math.sin(abs(theta) / 2))
    mid = (p + q) / 2
    n = np.array([-(q - p)[1], (q - p)[0]]) / chord
    h = r * math.cos(theta / 2)
    c = mid + n * h * (1 if b > 0 else -1) * (1 if abs(theta) <= math.pi else -1)
    return c, r


def bulge(c, p, q, ccw):
    a0, a1 = ang(c, p), ang(c, q)
    sweep = (a1 - a0) % 360 if ccw else -((a0 - a1) % 360)
    return math.tan(math.radians(sweep) / 4)


def draw_poly(verts, layer="MAIN", closed=True):
    """verts: [(x, y, bulge_to_next)] -> LINE/ARC entities."""
    n = len(verts)
    for i in range(n if closed else n - 1):
        x0, y0, b = verts[i]
        x1, y1, _ = verts[(i + 1) % n]
        if abs(b) < 1e-9:
            if math.hypot(x1 - x0, y1 - y0) > 1e-6:
                line((x0, y0), (x1, y1), layer)
        else:
            c, r = bulge_center((x0, y0), (x1, y1), b)
            a0, a1 = ang(c, (x0, y0)), ang(c, (x1, y1))
            if b > 0:
                arc(c, r, a0, a1, layer)
            else:
                arc(c, r, a1, a0, layer)


def hatch(verts, phase, spacing=2.5, layer="THIN"):
    h = msp.add_hatch(dxfattribs={"layer": layer})
    h.set_pattern_fill("ANSI31", scale=spacing / 3.175)
    h.pattern.lines[0].base_point = Vec2(0.0, phase)
    h.paths.add_polyline_path([(x, y, b) for x, y, b in verts], is_closed=True)
    return h


def mirror_profile(upper, ya):
    """upper: [(x, y, b)] from a point on the axis outwards and back to the left face.
    Returns the closed loop upper + mirrored lower (bulges stay valid)."""
    lower = [(x, 2 * ya - y, 0.0) for x, y, _ in reversed(upper)]
    bul = [b for _, _, b in upper]
    loop = list(upper[:-1]) + [(upper[-1][0], upper[-1][1], 0.0)]
    n = len(upper) - 1
    for k in range(n):
        x, y, _ = lower[k]
        loop.append((x, y, bul[n - 1 - k]))
    return loop


# ============================================================== sheet, frame, side columns
SW, SH = 420.0, 297.0
rect(0, 0, SW, SH, "THIN")                      # sheet trim line
rect(20, 5, 415, 292, "MAIN")                   # inner frame
rect(20, 278, 90, 292, "MAIN")                  # top-left box (rotated designation, GOST 2.104 gr.26)

# left additional columns (GOST 2.104 gr.19-23 and Перв.примен./Справ.N)
for x in (8.0, 13.0):
    line((x, 5), (x, 150), "MAIN")
    line((x, 173), (x, 292), "MAIN")
line((8.0, 150), (8.0, 173), "MAIN")
for y in (5, 30, 65, 90, 115, 150):
    line((8, y), (20, y), "MAIN")
for y in (173, 232, 292):
    line((8, y), (20, y), "MAIN")
line((0, 150), (8, 150), "MAIN")
line((0, 173), (8, 173), "MAIN")
line((0, 292), (8, 292), "MAIN")
line((0, 5), (8, 5), "MAIN")
SIDE_H = 3.0
for s, y0, y1 in (("Инв.N подл.", 5, 30), ("Подпись и дата", 30, 65), ("Взам.инв.N", 65, 90),
                  ("Инв.N дубл.", 90, 115), ("Подп. и дата", 115, 150),
                  ("Справ. N", 173, 232), ("Первичная примен.", 232, 292)):
    text(s, 11.9, (y0 + y1) / 2, SIDE_H, rot=90, align=TA.CENTER)
text("Согласов.", 5.5, 161.5, 3.5, rot=90, align=TA.CENTER)

# ============================================================== title block (GOST 2.104 form 1, 185 x 55)
TB_X0, TB_X1, TB_Y0, TB_Y1 = 230.0, 415.0, 5.0, 60.0
rect(TB_X0, TB_Y0, TB_X1, TB_Y1, "MAIN")
for x in (247, 270, 285, 295):
    line((x, TB_Y0), (x, TB_Y1), "MAIN")
line((237, 35), (237, 60), "MAIN")
for y in (35, 40):
    line((TB_X0, y), (295, y), "MAIN")
for y in (10, 15, 20, 25, 30, 45, 50, 55):
    line((TB_X0, y), (295, y), "THIN")
line((295, 45), (TB_X1, 45), "MAIN")
line((295, 20), (TB_X1, 20), "MAIN")
line((365, TB_Y0), (365, 45), "MAIN")
line((365, 40), (TB_X1, 40), "MAIN")
line((365, 25), (TB_X1, 25), "MAIN")
for x in (380, 397):
    line((x, 25), (x, 45), "MAIN")
for x in (370, 375):
    line((x, 25), (x, 40), "THIN")
line((385, 20), (385, 25), "MAIN")

TB_SMALL = 3.5
for s, x in (("Изм.", 230.3), ("Лист", 237.3), ("N° докум.", 248.5), ("Подпись", 270.3), ("Дата", 285.3)):
    text(s, x, 35.6, TB_SMALL)
for s, y in (("Разраб.", 30.6), ("Провер.", 25.6), ("Рук. гр.", 20.6), ("Нач. бюро", 15.6), ("Нормок.", 10.6), ("Утв.", 5.6)):
    text(s, 230.4, y, TB_SMALL)
text("Литер.", 365.6, 41.0, TB_SMALL)
text("Масса", 382.2, 41.0, TB_SMALL)
text("Масшт.", 399.2, 41.0, TB_SMALL)
text("Лист", 366.8, 21.0, TB_SMALL)
text("Листов  1", 387.8, 21.0, TB_SMALL)
text("А", 372.5, 33.5, 7.0, align=TA.MIDDLE_CENTER)
text("0,385", 388.5, 31.0, 3.5, align=TA.CENTER)
text("1:1", 405.8, 31.0, 5.0, align=TA.CENTER)
text("Клапан", 325.1, 31.7, 5.0, align=TA.CENTER)
text("предохранительный", 330.0, 23.5, 5.0, align=TA.CENTER)
text("Ал-9  ГОСТ2685-75", 296.9, 10.9, 5.0)
text("Копировал:", 295.2, 1.1, 3.5)
text("Формат  А3", 373.9, 1.1, 3.5)

# ============================================================== technical requirements
NOTES = [
    (248.0, "1.  Требования  предъявляемые  к  отливке,"),
    (238.6, "по  ТТМ  65-78."),
    (248.0, "2.  Неуказанные  литейные  радиусы  3...5мм."),
    (248.0, "3.  Неуказанные  предельные  отклонения"),
    (238.6, "размеров  на  механическую  обработку:"),
    (238.6, "H14,  h14,  %%pIT15/2."),
    (248.0, "4.  Покрытие:  лак  ЛБС-2  ГОСТ901-78  IIА"),
    (238.6, "на  необрабатываемых  поверхностях."),
    (248.0, "5.  Маркировать  шрифтом  ПО-12"),
    (238.6, "ГОСТ  2930-62  в  литье.  выпуклость  букв  1,5мм."),
    (248.0, "6.  Допускается  использовать  отливку"),
    (238.6, "детали  0130-04-124."),
]
for i, (x, s) in enumerate(NOTES):
    text(s, x, 198.1 - 8.535 * i, 5.0)

# ============================================================== main view: section of the casting
# Anchors measured on the raster: part axis and the left face of the boss.
YA = 160.41          # axis of revolution
X0 = 54.30           # left face (datum of the 10 and 46 dimensions)
# Axial positions, all from dimensions
XF42 = X0 + 10       # face of the ⌀42 boss                    (10)
XR = X0 + 46         # right face of the rim                    (46)
XS = XR - 5.5        # step ⌀160/⌀150                            (5,5)
XL = XS - 4          # left face of the rim                     (4)
XH = XR - 14         # face of the hub (⌀22)                    (14)
XT = XH - 20         # end of full thread M12x1,5               (20)
XD = XH - 25         # end of the drilled hole                  (25)
XC = XH - 1.6        # chamfer 1,6x45°
# Radii (half of the diameters)
R160, R150, R135 = 80.0, 75.0, 67.5
R42, R32, R28, R22 = 21.0, 16.0, 14.0, 11.0
R_THR, R_DRL = 6.0, 4.9   # M12 major / drilled minor (minor drawn ≈0,82d as on the sheet)
WALL, CONE = 4.0, 60.0    # wall 4(+1), 60° cone
# fillets: R2, R3, R7 dimensioned; others = "неуказанные литейные радиусы 3...5" (values matched on raster)
R_BEAD, R_GRV, R_FACE, R_HUB, R_RIM_OUT, R_RIM_IN = 2.0, 3.0, 3.0, 7.0, 3.5, 3.0
GROOVE_R = 1.5            # undercut of the rim (labelled R2 in view A, drawn as a 1,5 semicircle)


def Y(r):
    return YA + r


def upper_profile():
    d_out = unit(CONE)
    p_out = np.array([XF42, Y(R42)])                               # cone outer line passes the ⌀42 edge
    p_in = p_out + np.array([WALL / math.sin(math.radians(CONE)), 0])
    tip = (XD - R_DRL / math.tan(math.radians(60)), YA)            # 120° drill point
    c7, t7a, t7b = fillet_ll((XH, Y(R22)), (1, 0), p_in, d_out, R_HUB, (XH - 8, Y(R22 + 6)))
    c3i, t3a, t3b = fillet_ll(p_in, d_out, (XR, Y(R135)), (1, 0), R_RIM_IN, (XR - 2, Y(R135 - 3)))
    cro, tro_a, tro_b = fillet_ll((XL, YA), (0, 1), p_out, d_out, R_RIM_OUT, (XL - 2, Y(R135)))
    cfo, tfo_a, tfo_b = fillet_ll(p_out, d_out, (XF42, YA), (0, 1), R_FACE, (XF42 + 3, Y(R42 - 1)))
    cg = np.array([X0 + 7, Y(R28 + R_GRV)])                        # R3 groove -> ⌀42 face
    cb = np.array([X0 + R_BEAD, Y(R28)])                           # R2 bead corner (top = ⌀32)
    cs = np.array([cb[0] + math.sqrt((R_BEAD + R_GRV) ** 2 - R_GRV ** 2), Y(R28 + R_GRV)])
    ts = cb + (cs - cb) * R_BEAD / (R_BEAD + R_GRV)                # tangent point bead/concave
    cgr = np.array([XS + GROOVE_R, Y(R150)])
    v = [
        (*tip, 0), (XD, Y(R_DRL), 0), (XC, Y(R_DRL), 0), (XH, Y(R_DRL + 1.6), 0), (XH, Y(R22), 0),
        (*t7a, bulge(c7, t7a, t7b, ccw=False)), (*t7b, 0),
        (*t3a, bulge(c3i, t3a, t3b, ccw=False)), (*t3b, 0),
        (XR, Y(R135), 0), (XR, Y(R150), 0),
        (cgr[0] + GROOVE_R, Y(R150), -1.0), (XS, Y(R150), 0),
        (XS, Y(R160), 0), (XL, Y(R160), 0),
        (*tro_a, bulge(cro, tro_a, tro_b, ccw=False)), (*tro_b, 0),
        (*tfo_a, bulge(cfo, tfo_a, tfo_b, ccw=True)), (*tfo_b, 0),
        (XF42, cg[1], bulge(cg, (XF42, cg[1]), (cg[0], Y(R28)), ccw=False)),
        (cg[0], Y(R28), 0),
        (cs[0], Y(R28), bulge(cs, (cs[0], Y(R28)), ts, ccw=False)),
        (*ts, bulge(cb, ts, (X0, Y(R28)), ccw=True)),
        (X0, Y(R28), 0),
    ]
    return v


SECTION = mirror_profile(upper_profile(), YA)
draw_poly(SECTION, "MAIN")
hatch(SECTION, phase=1.42)

line((XT, Y(-R_THR)), (XT, Y(R_THR)), "MAIN")                  # thread end
line((XD, Y(-R_DRL)), (XD, Y(R_DRL)), "MAIN")                  # bottom of the drilled hole
line((XC, Y(-R_DRL)), (XC, Y(R_DRL)), "MAIN")                  # chamfer edge
for s in (1, -1):
    line((XT, Y(s * R_THR)), (XH, Y(s * R_THR)), "THIN")        # thread major diameter
line((XR, Y(-R135)), (XR, Y(R135)), "MAIN")                    # rim face seen inside the bore
X_TAN = XF42 + WALL / math.sin(math.radians(CONE)) + (R135 - R42) / math.tan(math.radians(CONE))
line((X_TAN, Y(-64.3)), (X_TAN, Y(64.3)), "THIN")              # transition line of the inner fillet
line((51.4, YA), (102.5, YA), "AXIS")

# ============================================================== dimensions (real DIMENSION entities)

def dim_lin(p1, p2, base, angle=0.0, text="<>", loc=None, override=None):
    d = msp.add_linear_dim(base=base, p1=p1, p2=p2, angle=angle, text=text, location=loc, dimstyle="ESKD",
                           override=override or {}, dxfattribs={"layer": "DIMS"})
    d.render()
    return d


def dim_rad(c, r, angle, loc=None, text="R<>", override=None):
    d = msp.add_radius_dim(center=c, radius=r, angle=angle, location=loc, text=text, dimstyle="ESKD",
                           override=override or {}, dxfattribs={"layer": "DIMS"})
    # ezdxf derives the arc point from `location`; the sheet has the leader going from the arc point
    # (at `angle`) through the centre to the shelf, so the arc point is set explicitly.
    d.dimension.dxf.defpoint4 = Vec3(c[0] + r * math.cos(math.radians(angle)), c[1] + r * math.sin(math.radians(angle)), 0)
    d.render()
    return d


def dim_ang(center, p1, p2, base_r, text="<>", loc=None):
    """Angular dim between rays center->p1 and center->p2 (CCW from p1 to p2)."""
    a1, a2 = ang(center, p1), ang(center, p2)
    mid = math.radians((a1 + ((a2 - a1) % 360) / 2))
    d = msp.add_angular_dim_cra(center=center, radius=base_r, start_angle=a1, end_angle=a2, distance=0,
                                location=loc, text=text, dimstyle="ESKD", dxfattribs={"layer": "DIMS"})
    d.render()
    return d


ARROWS.create_block(doc.blocks, ARROWS.none)
HORIZ = {"dimtih": 1, "dimtoh": 1}
OUTSIDE = {"dimtix": 0, "dimsoxd": 0, "dimtofl": 1}
# axial dimensions
dim_lin((X0, Y(-R28)), (XF42, Y(-R42)), base=(0, 112.91))                                    # 10
dim_lin((X0, Y(-R28)), (XR, Y(-R135)), base=(0, 99.10), loc=(71.1, 102.6))                    # 46
dim_lin((XD, Y(R_DRL)), (XH, Y(R22)), base=(0, 195.28), loc=(68.5, 198.7))                    # 25
dim_lin((XT, Y(-R_THR)), (XH, Y(-R22)), base=(0, 130.41), loc=(76.0, 133.9))                  # 20
dim_lin((XH, Y(-R22)), (XR, Y(-R135)), base=(0, 130.41), loc=(92.0, 133.9))                   # 14
dim_lin((XL, Y(-R160)), (XS, Y(-R160)), base=(0, 76.39), loc=(82.3, 79.9), override=OUTSIDE)  # 4
dim_lin((XS, Y(-R160)), (XR, Y(-R150)), base=(0, 67.58), loc=(112.0, 71.9), override=OUTSIDE)  # 5,5
dim_lin((XC, Y(R_DRL)), (XH, Y(R_DRL + 1.6)), base=(0, 188.98), text="<>x45%%d",
        loc=(113.8, 192.9), override=OUTSIDE)                                                # 1,6x45°
line((XS, 65.58), (XS, 46.70), "THIN")       # extension of the ⌀150 step down to the runout/roughness marks
# diameters (linear, vertical)
dim_lin((XF42, Y(-R42)), (XF42, Y(R42)), base=(31.84, 0), angle=90, text="%%c<>", loc=(28.3, 162.1))
dim_lin((X0 + R_BEAD, Y(-R32)), (X0 + R_BEAD, Y(R32)), base=(39.58, 0), angle=90, text="%%c<>", loc=(36.1, 162.1))
dim_lin((X0 + 6, Y(-R28)), (X0 + 6, Y(R28)), base=(47.92, 0), angle=90, text="%%c<>", loc=(44.5, 160.6))
dim_lin((XR, Y(-R135)), (XR, Y(R135)), base=(133.20, 0), angle=90, text="%%c<>", loc=(129.7, 150.8))
dim_lin((XR, Y(-R150)), (XR, Y(R150)), base=(143.24, 0), angle=90, text="%%c<>%%p0,5", loc=(139.2, 152.4))
dim_lin((XS, Y(-R160)), (XS, Y(R160)), base=(151.84, 0), angle=90, text="%%c<>", loc=(148.3, 154.9))
dim_lin((XH, Y(-R_THR)), (XH, Y(R_THR)), base=(110.17, 0), angle=90, text="M12x1,5-H7", loc=(106.0, 128.75))
dim_lin((XH, Y(-R22)), (XH, Y(R22)), base=(121.11, 0), angle=90, text="%%c<>", loc=(118.0, 158.6),
        override={"dimse2": 1, "dimsd2": 1, "dimsah": 1, "dimblk1": "", "dimblk2": ARROWS.none})  # ⌀22 (half)
# radii
dim_rad((75.27, Y(-R22 - R_HUB)), R_HUB, 118.0, loc=(76.5, 140.6), text="R7", override=HORIZ)              # R7

# ============================================================== annotation symbols (DIMS layer)

def solid_tri(a, b, c, layer="DIMS"):
    h = msp.add_hatch(color=256, dxfattribs={"layer": layer})
    h.set_solid_fill(color=256)
    h.paths.add_polyline_path([tuple(a), tuple(b), tuple(c)], is_closed=True)
    return h


def arrow(tip, toward, length=3.0, width=1.0):
    """Filled arrowhead with the tip at `tip`, body pointing to `toward`."""
    tip, toward = np.asarray(tip, float), np.asarray(toward, float)
    d = (toward - tip) / np.linalg.norm(toward - tip)
    n = np.array([-d[1], d[0]])
    base = tip + d * length
    solid_tri(tip, base + n * width / 2, base - n * width / 2)


def leader(points, layer="DIMS"):
    """LEADER entity: arrowhead at the first vertex (dimstyle ESKD)."""
    return msp.add_leader([tuple(map(float, p)) for p in points], dimstyle="ESKD",
                          dxfattribs={"layer": layer})


ROUGH_SHORT = np.array([-2.85, 5.0])
ROUGH_LONG = np.array([5.70, 9.85])
rb = doc.blocks.new("ESKD_ROUGH")
rb.add_line((0, 0), tuple(ROUGH_SHORT), dxfattribs={"layer": "0"})
rb.add_line((0, 0), tuple(ROUGH_LONG), dxfattribs={"layer": "0"})


def roughness(apex, value, rot=0.0, text_at=None, text_rot=None, h=5.0):
    """GOST 2.309 roughness sign (block) + value text."""
    msp.add_blockref("ESKD_ROUGH", tuple(map(float, apex)), dxfattribs={"layer": "DIMS", "rotation": rot})
    if value:
        if text_at is None:
            text_at = (apex[0] - 4.85, apex[1] + 6.35)
        text(value, text_at[0], text_at[1], h, rot=text_rot if text_rot is not None else rot, layer="DIMS")


# angle 60° between the bore line extension and the cone, wall thickness 4(+1)
V60 = np.array([XF42 + (R135 - R42) / math.tan(math.radians(CONE)), Y(R135)])
line((70.70, Y(R135)), (XL, Y(R135)), "THIN")
dim_ang(V60, V60 + unit(180), V60 + unit(240), 20.4, text="60%%d", loc=(77.5, 218.6))
p_o = np.array([76.47, 202.49])
p_i = p_o + unit(-30) * WALL
d4 = msp.add_linear_dim(base=tuple(p_o), p1=tuple(p_o), p2=tuple(p_i), angle=-30.0, text=r"<>{\H0.7x;\S+1^ ;}",
                         location=(64.8, 216.0), dimstyle="ESKD",
                         override={"dimtix": 0, "dimtofl": 1, **HORIZ}, dxfattribs={"layer": "DIMS"})
d4.render()
line((60.5, 209.44), (64.61, 209.44), "DIMS")
line((64.61, 209.44), (74.10, 203.90), "DIMS")
# radii R2, R3 (two arrows from one shelf)
dim_rad((X0 + R_BEAD, Y(R28)), R_BEAD, 116.7, loc=(48.9, 183.9), text="R2", override=HORIZ)
dim_rad((61.3, Y(-R28 - R_GRV)), R_GRV, 46.0, loc=(48.4, 130.2), text="R3", override=HORIZ)
leader([(64.46, 139.25), (48.83, 130.10)])
for a, b in (((43.25, 189.88), (47.94, 191.47)), ((44.67, 190.32), (49.50, 189.08)), ((44.56, 187.44), (46.27, 189.79))):
    line(a, b, "DIMS")                       # stray strokes next to "R2" on the original sheet
# base mark (open triangle) with leader to the left face
for a, b in (((28.57, 190.52), (41.24, 190.52)), ((41.24, 190.52), (34.90, 201.60)), ((34.90, 201.60), (28.57, 190.52))):
    line(a, b, "DIMS")
leader([(X0, 170.60), (34.90, 201.60)])
# chamfer 1x45° (two leaders, shelf, text) and its roughness
leader([(XS, Y(R160)), (108.9, 258.60), (141.65, 258.60)])
leader([(XR, Y(R150)), (108.9, 258.60)])
text("1x45%%d", 112.75, 259.6, 5.0, layer="DIMS")
# view A callout
circle((96.0, 235.21), 3.75, "THIN")
line((76.3, 240.07), (86.8, 240.07), "THIN")
line((86.8, 240.07), (92.86, 237.20), "THIN")
text("А", 76.3, 241.0, 10.0)
# view B arrow
line((41.92, 91.91), (47.84, 95.30), "THIN")
arrow((50.44, 96.80), (41.92, 91.91))
text("В", 38.6, 96.95, 10.0)
# datum Б on the thread axis
rect(105.15, 174.2, 115.2, 184.3, "DIMS")
text("Б", 110.18, 176.8, 5.0, layer="DIMS", align=TA.CENTER)
line((110.17, 174.2), (110.17, 171.33), "DIMS")
solid_tri((107.75, Y(R_THR)), (112.6, Y(R_THR)), (110.17, 171.33))
# runout tolerance  ↗ 0,2 Б
rect(107.53, 56.63, 132.0, 63.62, "DIMS")
line((114.45, 56.63), (114.45, 63.62), "DIMS")
line((125.39, 56.63), (125.39, 63.62), "DIMS")
line((108.87, 57.64), (111.6, 60.62), "DIMS")
arrow((112.96, 62.11), (108.87, 57.64), length=2.0, width=0.8)
text("0,2", 115.5, 57.85, 5.0, layer="DIMS")
text("Б", 127.5, 57.85, 5.0, layer="DIMS")
leader([(XS, 59.83), (107.53, 59.83)])
# roughness marks of the main view
roughness((115.0, Y(R150)), "6,3")
roughness((130.2, Y(R160)), "6,3")
roughness((136.43, 258.60), "6,3")
roughness((XH, 178.2), "12,5", rot=-90, text_at=(97.4, 175.6), text_rot=90)
roughness((XS, 50.40), "6,3", rot=-90, text_at=(105.5, 47.4), text_rot=90)

# ============================================================== freehand break lines (splines through raster-traced points)
from scipy.interpolate import splev, splprep  # noqa: E402
from meas import trace_curve  # noqa: E402


def freehand(guide, n=None, closed=False, band=1.0, step=1.0, smooth=0.02):
    pts = trace_curve(guide, band=band, step=step)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.3:
            keep.append(i)
    pts = pts[keep]
    if closed:
        pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smooth * len(pts), per=closed)
    n = n or max(8, len(pts) // 2)
    u = np.linspace(0, 1, n + (0 if closed else 1), endpoint=not closed)
    fit = np.column_stack(splev(u, tck))
    dense = np.column_stack(splev(np.linspace(0, 1, 400), tck))
    if not closed:
        fit[0], fit[-1] = guide[0], guide[-1]
        dense[0], dense[-1] = guide[0], guide[-1]
    return fit, dense


def add_freehand(fit, closed=False, layer="THIN"):
    pts = np.vstack([fit, fit[:1]]) if closed else fit
    msp.add_spline(fit_points=[tuple(map(float, p)) for p in pts], dxfattribs={"layer": layer})


def split_dense(dense, y):
    """(segment index, point) of the first crossing of the horizontal y."""
    for i, (a, b) in enumerate(zip(dense[:-1], dense[1:])):
        if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]:
            t = (y - a[1]) / (b[1] - a[1])
            return i, a + (b - a) * t
    raise ValueError(y)


# ============================================================== view A (5:1) - groove of the rim
K = 5.0
DA = np.array([185.86, 236.27])                 # image of the corner (XS, ⌀150) in the detail (measured)


def A5(x, r):
    return tuple(DA + K * (np.array([x, Y(r)]) - np.array([XS, Y(R150)])))


CH = 1.0                                         # chamfers 1x45°
XA_L, XA_R = A5(XL, 0)[0], A5(XR, 0)[0]
BRK_A = [(XA_L, 243.5), (166.69, 240.6), (167.4, 237.8), (168.1, 235.1), (168.96, 232.5), (170.0, 230.0),
         (171.5, 227.86), (173.2, 226.0), (175.48, 224.75), (178.17, 224.04), (181.43, 223.6), (184.83, 223.4),
         (188.09, 223.25), (191.2, 222.9), (194.46, 222.54), (197.44, 221.91), (200.27, 221.2), (203.1, 220.5),
         (205.8, 219.65), (208.49, 218.74), (211.18, 217.8), (XA_R, 217.04)]          # read on a 1 mm grid crop
brk_fit = np.array(BRK_A)
_tck, _ = splprep([brk_fit[:, 0], brk_fit[:, 1]], s=0)
brk = np.column_stack(splev(np.linspace(0, 1, 300), _tck))
add_freehand(brk_fit)
gA = np.array(A5(XS + GROOVE_R, R150))
detail = [A5(XL, R160), A5(XS - CH, R160), A5(XS, R160 - CH), A5(XS, R150),
          A5(XS + 2 * GROOVE_R, R150), A5(XR - CH, R150), A5(XR, R150 - CH), (XA_R, 217.04)]
for a, b in zip(detail[:-1], detail[1:]):
    if a == A5(XS, R150):
        arc(gA, GROOVE_R * K, 180, 360)
    else:
        line(a, b, "MAIN")
line(A5(XL, R160), (XA_L, 243.5), "MAIN")
loopA = [(*p, 0.0) for p in detail]
loopA[3] = (*A5(XS, R150), 1.0)
loopA += [(float(x), float(y), 0.0) for x, y in brk[::-1][1:]]
hatch(loopA, phase=3.32)
text("А", 185.6, 268.05, 10.0)
text("(5:1)", 193.3, 268.6, 5.0)
d = msp.add_linear_dim(base=(232.98, 0), p1=A5(XS + GROOVE_R, R150 - GROOVE_R), p2=A5(XR - CH, R150), angle=90,
                       text="1,5", location=(228.9, 247.4), dimstyle="ESKD",
                       override={"dimlfac": 1 / K, **OUTSIDE}, dxfattribs={"layer": "DIMS"})
d.render()
dim_rad(tuple(gA), GROOVE_R * K, 222.0, loc=(210.3, 249.6), text="R2", override={"dimlfac": 1 / K, **HORIZ})
roughness((202.75, 243.6), "6,3", rot=37.5, text_at=(195.5, 245.2), text_rot=52.0)

# ============================================================== "вариант": hub variant (same hub, ⌀32 with chamfer)
VX, VY = 186.80, 129.99                       # left face / axis of the variant view (measured)
text("вариант", 169.5, 172.2, 5.0)


def V(dx, r):
    return (VX + dx, VY + r)


CHV = 1.5
vbrk_fit, vbrk = freehand([(VX + 10, 151.2), (199.08, 154.87), (199.08, 153.17), (201.75, 152.85), (201.98, 151.9),
                           (200.95, 148.3), (201.0, 146.7), (202.0, 144.15), (200.1, 140.45), (201.4, 138.0),
                           (203.5, 136.1), (205.9, 133.0), (204.0, 130.0), (201.75, 127.2), (201.4, 124.2),
                           (202.6, 121.4), (204.8, 118.4), (202.1, 116.4), (200.8, 111.9), (201.4, 109.0),
                           (202.7, 106.8), (199.6, 104.3), (VX + 10, 108.7)], n=26, band=0.8, step=0.7)
add_freehand(vbrk_fit)
for s_ in (1, -1):
    line(V(0, s_ * (R32 - CHV)), V(CHV, s_ * R32), "MAIN")              # chamfer 1,5x45°
    line(V(CHV, s_ * R32), V(7, s_ * R32), "MAIN")
    arc(V(7, s_ * (R32 + R_GRV)), R_GRV, 270 if s_ > 0 else 0, 0 if s_ > 0 else 90)
    line(V(10, s_ * (R32 + R_GRV)), (VX + 10, 151.2 if s_ > 0 else 108.7), "MAIN")
line(V(0, -(R32 - CHV)), V(0, R32 - CHV), "MAIN")
tipv = V(7 - R_DRL / math.tan(math.radians(60)), 0)
for s_ in (1, -1):
    line(tipv, V(7, s_ * R_DRL), "MAIN")
    xe = split_dense(vbrk, VY + s_ * R_DRL)[1][0]
    line(V(7, s_ * R_DRL), (xe, VY + s_ * R_DRL), "MAIN")
    xt = split_dense(vbrk, VY + s_ * R_THR)[1][0]
    line(V(12, s_ * R_THR), (xt, VY + s_ * R_THR), "THIN")
line(V(7, -R_DRL), V(7, R_DRL), "MAIN")
line(V(12, -R_THR), V(12, R_THR), "MAIN")
line((VX - 1.3, VY), (VX + 22.1, VY), "AXIS")
i_top, p_top = split_dense(vbrk, VY + R_DRL)
i_bot, p_bot = split_dense(vbrk, VY - R_DRL)
loopV = [(*V(10, R32 + R_GRV), bulge(V(7, R32 + R_GRV), V(10, R32 + R_GRV), V(7, R32), ccw=False)),
         (*V(7, R32), 0), (*V(CHV, R32), 0), (*V(0, R32 - CHV), 0), (*V(0, -(R32 - CHV)), 0), (*V(CHV, -R32), 0),
         (*V(7, -R32), bulge(V(7, -(R32 + R_GRV)), V(7, -R32), V(10, -(R32 + R_GRV)), ccw=False)),
         (*V(10, -(R32 + R_GRV)), 0)]
loopV += [(float(x), float(y), 0) for x, y in vbrk[::-1][:len(vbrk) - 1 - i_bot]]
loopV += [(float(p_bot[0]), float(p_bot[1]), 0), (*V(7, -R_DRL), 0), (*tipv, 0), (*V(7, R_DRL), 0),
          (float(p_top[0]), float(p_top[1]), 0)]
loopV += [(float(x), float(y), 0) for x, y in vbrk[: i_top + 1][::-1]]
hatch(loopV, phase=0.15)
dim_lin(V(CHV, -R32), V(CHV, R32), base=(171.74, 0), angle=90, text="%%c<>", loc=(168.2, 132.8))
dim_lin(V(0, R32 - CHV), V(CHV, R32), base=(183.27, 0), angle=90, text="1,5x45%%d", loc=(195.2, 161.7),
        override={**OUTSIDE, **HORIZ})
line((183.27, 149.0), (183.27, 157.77), "DIMS")
line((183.27, 157.77), (206.3, 157.77), "DIMS")
dim_rad(V(7, -(R32 + R_GRV)), R_GRV, 48.5, loc=(179.2, 94.6), text="R3", override=HORIZ)
roughness((170.96, VY - R32), "12,5", rot=180, text_at=(168.4, 103.4), text_rot=0)
roughness((183.9, 99.26), "12,5", rot=-131.3, text_at=(187.5, 90.0), text_rot=52.0)

# ============================================================== view B (rotated): marking area
text("В", 41.0, 61.0, 10.0)
circle((56.02, 63.87), 3.99, "THIN")
line((50.14, 64.87), (52.1, 63.0), "THIN")
line((54.14, 64.87), (52.1, 63.0), "THIN")
vb_fit, _ = freehand([(33.1, 49.6), (36.5, 51.25), (42.25, 53.35), (46.5, 52.75), (51.0, 51.25), (55.35, 50.3),
                      (60.0, 51.0), (65.0, 52.5), (69.5, 53.25), (71.5, 52.5), (72.1, 50.0), (72.1, 45.0),
                      (72.5, 39.0), (74.25, 30.0), (73.25, 27.5), (70.0, 26.75), (66.5, 27.0), (62.0, 27.5),
                      (57.0, 27.75), (52.0, 27.25), (46.0, 26.0), (40.5, 24.75), (38.0, 25.25), (36.0, 27.0),
                      (34.5, 30.0), (33.5, 35.0), (33.0, 42.5)], n=24, closed=True, band=1.0, step=1.0)
add_freehand(vb_fit, closed=True, layer="MAIN")
rect(36.60, 31.93, 36.60 + 30, 31.93 + 15, "THIN")
dim_lin((36.60, 31.93), (36.60, 31.93 + 15), base=(28.81, 0), angle=90, loc=(25.9, 38.3))
dim_lin((36.60, 31.93), (36.60 + 30, 31.93), base=(0, 16.56), loc=(49.4, 20.4))
msp.add_circle((63.35, 37.15), 0.35, dxfattribs={"layer": "DIMS"})
line((63.35, 37.15), (78.2, 23.4), "DIMS")
circle((82.32, 17.54), 5.63, "DIMS")
text("п.5", 82.3, 15.3, 5.0, layer="DIMS", align=TA.CENTER)
text("М", 76.0, 25.6, 5.0, rot=-48.0, layer="DIMS")

# ============================================================== general roughness  √ (√) in the upper right corner
for a, b in (((379.69, 271.02), (375.35, 278.36)), ((379.69, 271.02), (388.25, 285.72))):
    line(a, b, "DIMS")
circle((379.69, 274.90), 2.0, "DIMS")
roughness((395.61, 271.03), "")
arc((400.11, 276.11), 10.06, 145, 215, "DIMS")
arc((393.82, 276.11), 10.09, 325, 35, "DIMS")

# ============================================================== save
doc.header["$EXTMIN"] = (0, 0, 0)
doc.header["$EXTMAX"] = (SW, SH, 0)
OUT.parent.mkdir(exist_ok=True)
doc.saveas(OUT)
problems = verify(str(OUT))
print("verify:", "OK" if not problems else problems)

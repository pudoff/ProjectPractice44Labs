"""Semantic 2D reconstruction, manually interpreted from the root valve PNG.

This deliberately does not trace raster lettering or dimension graphics.
The JSON is a reviewed recognition result, not the output of an automatic ML
model. It is the reproducible input to a native DXF entity generator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(tempfile.gettempdir()) / '44labs-valve-semantic'
WORK.mkdir(exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(WORK / 'cache'))
try:
    import ezdxf
except ImportError:
    # Reuse already installed dependencies in this workspace session.
    sys.path.append(str(Path(tempfile.gettempdir()) /
        '44labs-bolt-artifacts-bad6e25064e94550a32bab05af03086d/pdf-dxf-deps'))
    import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Vec3
from ezdxf import path as ezpath

LAYERS = {
    'PART_MAIN': (7, 35, 'CONTINUOUS'),
    'PART_EDGES': (7, 25, 'CONTINUOUS'),
    'PART_DETAIL_A': (7, 35, 'CONTINUOUS'),
    'PART_VARIANT': (7, 35, 'CONTINUOUS'),
    'PART_VIEW_B': (7, 35, 'CONTINUOUS'),
    'SECTION_HATCH': (8, 13, 'CONTINUOUS'),
    'THREAD': (5, 13, 'CONTINUOUS'),
    'AXIS': (1, 13, 'CENTER'),
    'DIMENSIONS': (3, 18, 'CONTINUOUS'),
    'TEXT': (2, 18, 'CONTINUOUS'),
    'SYMBOLS': (6, 18, 'CONTINUOUS'),
    'BREAK_LINES': (8, 18, 'CONTINUOUS'),
    'FRAME': (8, 18, 'CONTINUOUS'),
}


class Outline:
    """Connected exact segments, with circular arcs as polyline bulges."""
    def __init__(self, start):
        self.points = [[*start, 0.0]]

    def line(self, point):
        self.points.append([*point, 0.0])
        return self

    def arc(self, center, end, ccw=True):
        start = self.points[-1][:2]
        r1 = math.dist(start, center)
        r2 = math.dist(end, center)
        assert abs(r1-r2) < 1e-7, (start, center, end, r1, r2)
        a = math.atan2(start[1]-center[1], start[0]-center[0])
        b = math.atan2(end[1]-center[1], end[0]-center[0])
        sweep = (b-a) % (2*math.pi) if ccw else -((a-b) % (2*math.pi))
        self.points[-1][2] = math.tan(sweep/4)
        return self.line(end)


def build(spec):
    p = spec['parameters']
    estimated = spec['raster_estimates']
    doc = ezdxf.new('R2007', setup=True, units=4)
    for key,value in {'$INSUNITS': 4, '$MEASUREMENT': 1, '$LUNITS': 2,
        '$LUPREC': 3, '$DIMASSOC': 1, '$LWDISPLAY': True,
        '$LIMMIN': (0, 0), '$LIMMAX': (420, 297)}.items():
        doc.header[key]=value
    for name, (color, weight, linetype) in LAYERS.items():
        doc.layers.new(name, dxfattribs=dict(color=color, lineweight=weight, linetype=linetype))
    doc.styles.new('DRAWING_TEXT', dxfattribs={'font': 'arial.ttf'})
    doc.styles.new('DRAWING_ITALIC', dxfattribs={'font': 'ariali.ttf'})
    doc.dimstyles.new('VALVE_NATIVE', dxfattribs={
        'dimtxt': 4.3, 'dimasz': 2.0, 'dimexo': .5, 'dimexe': 1.5,
        'dimtad': 1, 'dimgap': .8, 'dimdec': 1, 'dimzin': 8,
        'dimdsep': 44, 'dimtxsty': 'DRAWING_ITALIC', 'dimscale': 1,
        'dimlfac': 1, 'dimclrd': 0, 'dimclre': 0, 'dimclrt': 0,
        'dimtih': 0, 'dimtoh': 0, 'dimtmove': 2,
    })
    doc.appids.new('VALVE_SEMANTIC')
    m = doc.modelspace()
    records = []
    ids = Counter()
    profiles = []
    dimensions = {}

    # Affine mapping calibrated from 46 and 160 mm in the supplied PNG.
    fx = 46 / (2380.131348511-1299.752759857) * (9925/1888)
    fy = 160 / (5110.473579921-1351.5) * (9925/1888)
    def uv(q):
        return ((q[0]-23/(9925/1888))*fx, (1331.3966750629722-q[1])*fy)
    origin = uv((247.2476786509, 614.6199556116))
    def main(q):
        return (origin[0]+q[0], origin[1]+q[1])

    def tag(e, key, role):
        ids[key]+=1
        if ids[key]>1:
            key=f'{key}.{ids[key]}'
        e.set_xdata('VALVE_SEMANTIC', [(1000, key), (1000, role)])
        records.append({'id': key, 'role':role, 'handle': e.dxf.handle, 'type': e.dxftype(), 'layer': e.dxf.layer})
        return e

    def line(a, b, layer='PART_EDGES'):
        return m.add_line(a, b, dxfattribs={'layer': layer})

    def poly(points, layer, key, closed=True, transform=lambda q: q, hatch=False):
        vertices = [(*transform(q[:2]), q[2] if len(q)>2 else 0) for q in points]
        e = tag(m.add_lwpolyline(vertices, format='xyb', close=closed,
                dxfattribs={'layer': layer}), key, 'contour')
        if closed:
            profiles.append(e)
        if hatch:
            h = tag(m.add_hatch(dxfattribs={'layer': 'SECTION_HATCH'}), key+'.hatch', 'section')
            h.set_pattern_fill('ANSI31', color=256, scale=.8)
            boundary = h.paths.add_polyline_path(vertices, is_closed=True)
            h.associate(boundary, [e])
        return e

    def text(s, at, height=4.3, layer='TEXT', rotation=0, italic=True):
        return m.add_text(s, dxfattribs={'layer': layer, 'height': height,
            'style': 'DRAWING_ITALIC' if italic else 'DRAWING_TEXT', 'rotation': rotation}).set_placement(at)

    def leader(points, annotation=None, arrow=True):
        e = m.add_leader(points, dimstyle='VALVE_NATIVE', override={'dimasz': 1.5},
            dxfattribs={'layer': 'SYMBOLS', 'has_arrowhead': int(arrow)})
        if annotation is not None:
            e.dxf.annotation_handle = annotation.dxf.handle
        return e

    def register_dim(d, key, expected=None, view_scale=1):
        # Keep the genuine generated block: no deleting, exploding, tracing or
        # replacing rendered text. <> is substituted from the measurement.
        d.render()
        e = tag(d.dimension, key, 'dimension')
        block = doc.blocks[e.dxf.geometry]
        for item in block:
            item.dxf.layer = 'DIMENSIONS'
            item.dxf.color = 256  # BYLAYER, also for CADs exposing dimension blocks.
        value = float(e.get_measurement()) / view_scale
        dimensions[key] = {'entity': e, 'value': value, 'expected': expected,
                           'view_scale': view_scale}
        if expected is not None:
            assert abs(value-expected) < 1e-7, (key, value, expected)
        return e

    def linear(key, a, b, base, angle=0, prefix='', suffix='', scale=1,
               location=None, tolerance=None, expected=None, overrides=None):
        style = {'dimlfac': 1/scale, 'dimpost': prefix+'<>'+suffix}
        if overrides:
            style.update(overrides)
        d = m.add_linear_dim(base=base, p1=a, p2=b, angle=angle,
            location=location, text='<>', dimstyle='VALVE_NATIVE', override=style,
            dxfattribs={'layer': 'DIMENSIONS'})
        if tolerance:
            d.set_tolerance(upper=tolerance[0], lower=tolerance[1], hfactor=.65, dec=1)
        return register_dim(d, key, expected, scale)

    def radius(key, center, r, location, scale=1):
        d = m.add_radius_dim(center=center, radius=r, location=location,
            text='<>', dimstyle='VALVE_NATIVE', override={'dimlfac': 1/scale,
            'dimtoh': 1, 'dimtih': 1, 'dimpost': 'R<>'}, dxfattribs={'layer': 'DIMENSIONS'})
        return register_dim(d, key, r/scale, scale)

    width, shoulder, neck = p['overall_width'], p['shoulder_x'], p['neck_x']
    rmax, rrear, rrim = p['outer_diameter']/2, p['rear_diameter']/2, p['inner_rim_diameter']/2
    rn, rb = p['nose_base_diameter']/2, p['boss_diameter']/2
    theta = math.radians(p['cone_angle_deg'])
    slope = math.tan(theta)
    groove_left = width-p['groove_offset_from_back']
    rim_left = groove_left-p['rim_width']
    cone_b = rrim-slope*rim_left
    wall = p['wall_thickness']
    inner_b = cone_b-wall/math.cos(theta)
    rneck, rnose, rbowl = p['neck_radius'], p['nose_radius'], p['bowl_radius']
    # Nominal circle/line intersections and tangent fillets, not pixel samples.
    vertex_y = slope*neck+cone_b
    t = rneck*math.tan((math.pi/2-theta)/2)
    neck_center = (neck+rneck, vertex_y-t)
    neck_tangent = (neck+t*math.cos(theta), vertex_y+t*math.sin(theta))
    bowl_center = ((rb+rbowl-inner_b+rbowl/math.cos(theta))/slope, rb+rbowl)
    bowl_tangent = (bowl_center[0]-rbowl*math.sin(theta), bowl_center[1]+rbowl*math.cos(theta))
    ri = estimated['inner_rim_fillet_radius']
    rim_vertex = ((rrim-inner_b)/slope, rrim)
    ti = ri*math.tan(theta/2)
    rim_tangent = (rim_vertex[0]-ti*math.cos(theta), rrim-ti*math.sin(theta))
    rim_center = (rim_vertex[0]+ti, rrim-ri)
    rg = p['groove_radius']
    groove_center = (groove_left+rg, rrear+rg-p['groove_depth'])
    groove_end = (groove_center[0]+math.sqrt(rg**2-(rrear-groove_center[1])**2), rrear)
    minor = estimated['bore_minor_radius']
    chamfer = p['thread_chamfer']
    # Two tangent circles reproduce the rounded nose. Their unlabelled blend
    # is an interpreted construction; it remains explicit in the generator.
    nose2_x = rnose+rnose*math.sqrt(3)
    upper = Outline((0, rn))
    upper.arc((rnose, rn), (rnose+rnose*math.sqrt(3)/2, rn+rnose/2), False)
    upper.arc((nose2_x, rn+rnose), (nose2_x, rn), True)
    upper.line((neck-rneck, rn)).arc((neck-rneck, rn+rneck), (neck, rn+rneck), True)
    upper.line((neck, neck_center[1])).arc(neck_center, neck_tangent, False)
    upper.line((rim_left, rrim)).line((rim_left, rmax))
    upper.line((groove_left-p['rim_chamfer'], rmax))
    upper.line((groove_left, rmax-p['rim_chamfer']))
    upper.line((groove_left, groove_center[1])).arc(groove_center, groove_end, True)
    upper.line((width, rrear)).line((width, rrim)).line((rim_center[0], rrim))
    upper.arc(rim_center, rim_tangent, True).line(bowl_tangent)
    upper.arc(bowl_center, (bowl_center[0], rb), True).line((shoulder, rb))
    upper.line((shoulder, minor+chamfer)).line((shoulder-chamfer, minor))
    upper.line((p['drill_end_x'], minor)).line((estimated['drill_tip_x'], 0))
    contour = copy.deepcopy(upper.points)
    for j in range(len(upper.points)-1, 0, -1):
        contour[-1][2] = upper.points[j-1][2]
        q = upper.points[j-1]
        contour.append([q[0], -q[1], 0])
    main_profile = poly(contour, 'PART_MAIN', 'main.material_section', transform=main, hatch=True)
    tag(line(main((width, -rrim)), main((width, rrim))), 'main.rear_projection', 'visible_edge')
    tag(line(main((shoulder, -minor-chamfer)), main((shoulder, minor+chamfer))), 'main.shoulder_projection', 'visible_edge')
    for sign in (-1, 1):
        line(main((p['thread_end_x'], sign*p['thread_nominal_diameter']/2)),
             main((shoulder, sign*p['thread_nominal_diameter']/2)), 'THREAD')
    rt=p['thread_nominal_diameter']/2
    line(main((p['thread_end_x'], -rt)), main((p['thread_end_x'], rt)), 'THREAD')
    line(main((-5, 0)), main((width+5, 0)), 'AXIS')

    # Main dimensions, numeric values read from exact definition points.
    for key, diameter, x, base_x in [
        ('diameter160', p['outer_diameter'], rim_left, 683),
        ('diameter150', p['rear_diameter'], width, 645),
        ('diameter135', p['inner_rim_diameter'], width, 600),
        ('diameter42_reference', 42, neck, 147),
        ('diameter32', p['nose_diameter'], rnose, 181),
        ('diameter28', p['nose_base_diameter'], 0, 219),
        ('diameter22', p['boss_diameter'], shoulder, 546),
    ]:
        linear(key, main((x, -diameter/2)), main((x, diameter/2)), uv((base_x, 615)),
            angle=90, prefix='%%c', expected=diameter,
            tolerance=(.5, .5) if key=='diameter150' else None)
    linear('thread_M12', main((shoulder, -rt)), main((shoulder, rt)), uv((497, 615)),
        angle=90, prefix='M', suffix='×1,5-H7', location=uv((480, 770)), expected=p['thread_nominal_diameter'])
    for key, x1, x2, by, expected in [
        ('depth25', p['drill_end_x'], shoulder, 459, shoulder-p['drill_end_x']),
        ('depth20', p['thread_end_x'], shoulder, 750, shoulder-p['thread_end_x']),
        ('back_shoulder', shoulder, width, 750, width-shoulder),
        ('neck10', 0, neck, 827, neck),
        ('overall_width', 0, width, 889, width),
        ('rim4', rim_left, groove_left, 990, p['rim_width']),
        ('groove_offset', groove_left, width, 1030, p['groove_offset_from_back']),
    ]:
        y1,y2 = {
            'depth25':(minor, rb), 'depth20':(-rt,-rb),
            'back_shoulder':(-rb,-rb), 'neck10':(-rn,-rn-rneck),
            'overall_width':(-rn,-rn), 'rim4':(-rmax,-rmax),
            'groove_offset':(-rrear,-rrear),
        }[key]
        location={'rim4':uv((376,975)),'groove_offset':uv((504,1014))}.get(key)
        linear(key, main((x1, y1)), main((x2, y2)), uv((300, by)),
               expected=expected, location=location)
    linear('thread_chamfer', main((shoulder-chamfer, minor)), main((shoulder, minor+chamfer)),
           uv((530, 487)), suffix='×45%%d', location=uv((515, 465)), expected=chamfer)
    linear('rim_chamfer', main((groove_left-p['rim_chamfer'], rmax)),
           main((groove_left, rmax-p['rim_chamfer'])), uv((555, 175)),
           suffix='×45%%d', location=uv((555, 155)), expected=p['rim_chamfer'])
    wall_point = (24, slope*24+cone_b)
    wall_other = (wall_point[0]+wall*math.sin(theta), wall_point[1]-wall*math.cos(theta))
    d = m.add_aligned_dim(main(wall_point), main(wall_other), distance=0,
        text='<>', dimstyle='VALVE_NATIVE', dxfattribs={'layer': 'DIMENSIONS'},
        override={'dimtmove': 2, 'dimtxt': 3.5})
    d.set_location(uv((294, 391)), leader=True, relative=False)
    d.set_tolerance(upper=1, lower=0, hfactor=.65, dec=0)
    register_dim(d, 'wall4', wall)
    angle_center = main((rim_left, rrim))
    d = m.add_angular_dim_3p(base=uv((326, 342)),center=angle_center,
        p1=(angle_center[0]-20, angle_center[1]),
        p2=(angle_center[0]-20*math.cos(theta), angle_center[1]-20*math.sin(theta)),
        text='<>', dimstyle='VALVE_NATIVE', dxfattribs={'layer': 'DIMENSIONS'})
    register_dim(d, 'cone_angle', p['cone_angle_deg'])
    radius('nose_R2', main((rnose, rn)), rnose, uv((189, 498)))
    radius('neck_R3', main((neck-rneck, -rn-rneck)), rneck, uv((190, 742)))
    radius('bowl_R7', main((bowl_center[0], -bowl_center[1])), rbowl, uv((346, 706)))
    leader([main((neck_tangent[0], -neck_tangent[1])), uv((226, 742)), uv((181, 742))])

    # Enlarged groove: true R2 and depth 1.5, shown at 5:1.
    sc = p['detail_a_scale']
    a_start = uv((835, 265))
    ac = (a_start[0]+rg*sc, a_start[1])
    ae = (ac[0]+math.sqrt(rg**2-(rg-p['groove_depth'])**2)*sc,
          ac[1]-(rg-p['groove_depth'])*sc)
    ap = Outline(uv((746, 163))).line(uv((813, 163))).line(uv((835, 185))).line(a_start)
    ap.arc(ac, ae, True).line((uv((937, 276))[0], ae[1])).line(uv((959, 299))).line(uv((959, 363)))
    for q in [(920,349),(879,338),(831,331),(796,326),(775,316),(760,292),(750,266),(746,240)]:
        ap.line(uv(q))
    poly(ap.points, 'PART_DETAIL_A', 'detail_a.material_section', hatch=True)
    ab = (ac[0], ac[1]-rg*sc)
    linear('detail_a_depth', ab, ae, uv((1046, 290)), angle=90, scale=sc,
           location=uv((1030, 235)), expected=p['groove_depth'])
    radius('detail_a_R2', ac, rg*sc, uv((1010, 386)), sc)

    # Alternative section; irregular cut edges are explicitly a schematic break.
    vo = uv((839, 751))
    def variant(q): return (vo[0]+q[0], vo[1]+q[1])
    vp = Outline((0, 14.5)).line((1.5, 16)).line((7, 16))
    vp.arc((7, 19), (10, 19), True)
    for q in [(10,23),(12,24),(13,23),(15,23),(14,19),(13.5,17),(14.5,14),(15,13),
              (12.8,10),(14,7),(15,6),(8,6),(8,4.8),(3,4.8),(.2,0),
              (3,-4.8),(8,-4.8),(8,-6),(15,-6),(16,-7),(19,-12),
              (17,-14),(15,-17),(14,-20),(15,-23),(12,-23),(12,-25),(10,-21),(10,-19)]:
        vp.line(q)
    vp.arc((7,-19), (7,-16), True).line((1.5,-16)).line((0,-14.5))
    poly(vp.points, 'PART_VARIANT', 'variant.material_section', transform=variant, hatch=True)
    line(variant((-5,0)), variant((23,0)), 'AXIS')
    line(variant((8,-6)), variant((8,6)), 'THREAD')
    linear('variant_diameter32', variant((1.5,-16)), variant((1.5,16)), uv((773,751)),
           angle=90, prefix='%%c', expected=32)
    linear('variant_chamfer', variant((0,14.5)), variant((1.5,16)), uv((862,625)),
           suffix='×45%%d', location=uv((865,606)), expected=1.5)
    radius('variant_R3', variant((7,-19)), 3, uv((789,908)))
    # View B rectangle is a separate closed editable contour, not a group/INSERT.
    vb = uv((168.3, 1189.3))
    bw,bh = p['view_b_width'],p['view_b_height']
    bpoints=[(vb[0],vb[1]),(vb[0]+bw,vb[1]),(vb[0]+bw,vb[1]+bh),(vb[0],vb[1]+bh)]
    poly(bpoints, 'PART_VIEW_B', 'view_b.rectangle')
    linear('view_b_30', bpoints[0], bpoints[1], uv((235,1258)), expected=bw)
    linear('view_b_15', bpoints[0], bpoints[3], uv((133.5,1150)), angle=90, expected=bh)
    breakpts=[uv(q) for q in [(151,1122),(170,1107),(192,1095),(224,1103),(251,1110),
        (280,1104),(309,1095),(325,1098),(329,1129),(331,1161),(335,1193),
        (339,1210),(330,1221),(297,1221),(261,1218),(226,1221),(191,1229),
        (173,1225),(158,1204),(151,1168),(151,1122)]]
    m.add_cad_spline_control_frame(breakpts, dxfattribs={'layer':'BREAK_LINES'})
    t=text('п.5', uv((359,1265)), 4.5)
    m.add_circle(uv((372,1258)), 6.2, dxfattribs={'layer':'SYMBOLS'})
    leader([uv((287,1165)),uv((358,1236))], t)

    # These transcriptions were visually verified in the original PNG.
    notes = spec['texts']
    for i,(s,box,italic) in enumerate(notes):
        x1,y1,x2,y2=box
        h=5.1 if i<12 else (4.5 if i in (12,13,14,36) else 2.6)
        if i==37: h=7.5
        baseline=min(y2-3,1327)
        e=text(s,uv((x1,baseline)),h,italic=italic)
        e.set_placement(uv((x1,baseline)),uv((x2,baseline)),align=TextEntityAlignment.FIT)

    # Every small numeral is TEXT; symbols remain annotation geometry.
    def rough(q, value, rotation=0):
        at=uv(q); ang=math.radians(rotation)
        def tr(x,y):return(at[0]+x*math.cos(ang)-y*math.sin(ang),at[1]+x*math.sin(ang)+y*math.cos(ang))
        poly([tr(-2.2,3.8),tr(0,0),tr(4.4,7.6)],'SYMBOLS','roughness.symbol',False)
        if abs(rotation)==180:
            text(value,tr(1.6,9),3.8,'TEXT',0)
        else:
            text(value,tr(-1.6,5.2),3.8,'TEXT',rotation)
    for q in [(519,279),(588,257),(616,175),(449,1117)]: rough(q,'6,3')
    rough((893,242),'6,3')
    leader([a_start,uv((893,242))],arrow=False)
    rough((392,532),'12,5',90)
    rough((772,822),'12,5',180)
    rough((828,879),'12,5',-60)
    text('А',uv((346,256)),8)
    leader([uv((424,271)),uv((391,258)),uv((347,258))],arrow=False)
    m.add_circle(main((groove_left+1,75.5)),4.2,dxfattribs={'layer':'SYMBOLS'})
    text('В',uv((174,900)),8)
    leader([uv((221,903)),uv((192,919))])
    text('В',uv((184,1059)),8)
    circle_center=uv((253,1052))
    m.add_circle(circle_center,4.2,dxfattribs={'layer':'SYMBOLS'})
    leader([uv((244,1052)),uv((251,1042))],arrow=False)
    text('Б',uv((488,544)),5)
    poly([uv(q) for q in [(475,508),(520,508),(520,554),(475,554)]], 'SYMBOLS','datum_B.frame')
    m.add_solid([uv((487,588)),uv((507,588)),uv((497,567))],dxfattribs={'layer':'SYMBOLS'})
    line(uv((497,567)),uv((497,554)),'SYMBOLS')
    # Geometrical tolerance is textual annotation with an editable frame. It
    # is not falsely encoded as a linear size or as a parametric constraint.
    for x in (485,516,566,594):line(uv((x,1048)),uv((x,1080)),'SYMBOLS')
    line(uv((485,1048)),uv((594,1048)),'SYMBOLS')
    line(uv((485,1080)),uv((594,1080)),'SYMBOLS')
    leader([uv((507,1055)),uv((492,1074))])
    text('0,2',uv((519,1074)),4.3)
    text('Б',uv((574,1074)),4.3)
    leader([main((groove_left,-79)),uv((460,1064)),uv((485,1064))],arrow=False)
    poly([uv(q) for q in [(132,475),(159,426),(187,475)]],
         'SYMBOLS','source.triangular_symbol')
    # General surface requirement, with no unrecognised numerical lettering.
    poly([uv(q) for q in [(1687,85),(1703,113),(1735,58)]],'SYMBOLS','general_surface',False)
    m.add_circle(uv((1704,99)),1.9,dxfattribs={'layer':'SYMBOLS'})
    text('(',uv((1744,113)),11)
    poly([uv(q) for q in [(1760,95),(1773,116),(1802,70)]],'SYMBOLS','general_surface.alt',False)
    text(')',uv((1802,113)),11)
    # Sheet and title table are independent native linework, on FRAME only.
    poly([(0,0),(420,0),(420,297),(0,297)],'FRAME','sheet.outside')
    poly([(20,5),(415,5),(415,292),(20,292)],'FRAME','sheet.border')
    for x in [8,13]:line((x,5),(x,292),'FRAME')
    for y in [30,70,115,170,230]:line((8,y),(20,y),'FRAME')
    for s,y in [('Инв. № подл.',8),('Подпись и дата',32),('Взам. инв. №',73),
                ('Инв. № дубл.',118),('Справ. №',174),('Первичн. примен.',235)]:
        text(s,(11,y),2.4,rotation=90,italic=False)
    # Reproduce the table geometry from its interpreted coordinates.
    xs=[1032,1064,1109,1212,1278,1325,1638,1704,1780,1865]
    for y in [1065,1088,1110,1133,1155,1177,1199,1222,1244,1266,1289,1311]:
        line(uv((1032,y)),uv((1325,y)),'FRAME')
    for x in xs[:6]:line(uv((x,1065)),uv((x,1311)),'FRAME')
    for y in [1065,1133,1311]:line(uv((1325,y)),uv((1865,y)),'FRAME')
    line(uv((1638,1133)),uv((1638,1311)),'FRAME')
    line(uv((1865,1065)),uv((1865,1311)),'FRAME')
    for y in [1155,1222,1244]:line(uv((1638,y)),uv((1865,y)),'FRAME')
    for x in [1704,1780]:line(uv((x,1133)),uv((x,1222)),'FRAME')
    line(uv((1325,1244)),uv((1638,1244)),'FRAME')

    doc.set_modelspace_vport(height=310,center=(210,148.5))
    # Modelspace is the default edit target; no INSERT wraps the whole sheet.
    doc.header['$TILEMODE']=1
    page=doc.layouts.get('Layout1')
    page.page_setup(size=(420,297),margins=(0,0,0,0),units='mm')
    page.add_viewport(center=(210,148.5),size=(420,297),view_center_point=(210,148.5),
                      view_height=297,status=2)
    return doc, records, dimensions, profiles


def validate(doc, dims, profiles):
    m=doc.modelspace()
    audit=doc.audit()
    assert not audit.errors and not audit.fixes
    assert not list(m.query('IMAGE INSERT'))
    assert not [e for e in m if e.dxf.layer=='0']
    assert len(profiles)>=4
    from shapely.geometry import Polygon
    profile_results=[]
    for e in profiles:
        assert e.closed
        verts=list(ezpath.make_path(e).flattening(.01))
        polygon=Polygon([(v.x,v.y) for v in verts])
        assert polygon.is_valid and polygon.area>0, (e.dxf.layer,polygon.is_valid)
        profile_results.append({'handle':e.dxf.handle,'layer':e.dxf.layer,
            'closed':True,'self_intersections':False,'area_mm2':polygon.area})
    for key,entry in dims.items():
        e=entry['entity']
        assert e.dxf.text=='<>', (key,e.dxf.text)
        block=doc.blocks[e.dxf.geometry]
        assert len(block.query('TEXT MTEXT'))>0
        assert not len(block.query('HATCH LWPOLYLINE'))
        assert all(i.dxf.layer=='DIMENSIONS' for i in block)
        if key=='cone_angle':
            rendered=' '.join(i.dxf.text if i.dxftype()=='TEXT' else i.text
                              for i in block.query('TEXT MTEXT'))
            assert '60' in rendered and '300' not in rendered
    for e in m.query('HATCH'):
        assert e.dxf.associative==1 and e.dxf.solid_fill==0
        for boundary in e.paths:
            assert len(boundary.source_boundary_objects)==1
            source=doc.entitydb[boundary.source_boundary_objects[0]]
            assert source.dxftype()=='LWPOLYLINE' and source.closed
            assert e.dxf.handle in source.get_reactors()
    # Exercise a real native dimension edit, without a numeric text override.
    probe=dims['overall_width']['entity'].copy()
    m.add_entity(probe)
    probe.dxf.defpoint3 += Vec3(1,0,0)
    probe.render()
    assert abs(probe.get_measurement()-dims['overall_width']['value']-1)<1e-7
    measured_text=[x.dxf.text if x.dxftype()=='TEXT' else x.text
                   for x in doc.blocks[probe.dxf.geometry].query('TEXT MTEXT')]
    assert str(int(round(probe.get_measurement()))) in ' '.join(measured_text)
    probe_block=probe.dxf.geometry
    assert probe_block != dims['overall_width']['entity'].dxf.geometry
    m.delete_entity(probe)
    doc.blocks.delete_block(probe_block,safe=False)
    return profile_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=Path(__file__).with_name('valve_semantic.json'))
    parser.add_argument('--output',type=Path,default=ROOT/'клапан_codex_v2.dxf')
    parser.add_argument('--validation-dir',type=Path,default=ROOT/'docs/validation/valve-v2')
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    doc,records,dims,profiles=build(spec)
    checked_profiles=validate(doc,dims,profiles)
    # Regenerate from changed semantic input: geometry AND relevant sizes move.
    changed=copy.deepcopy(spec)
    changed['parameters']['overall_width']+=1
    changed_doc,_,changed_dims,changed_profiles=build(changed)
    assert abs(changed_dims['overall_width']['value']-dims['overall_width']['value']-1)<1e-7
    assert abs(changed_dims['back_shoulder']['value']-dims['back_shoulder']['value']-1)<1e-7
    assert max(v[0] for v in changed_profiles[0])>max(v[0] for v in profiles[0])+.99
    args.output.parent.mkdir(parents=True,exist_ok=True)
    doc.saveas(args.output)
    check=ezdxf.readfile(args.output)
    audit=check.audit()
    assert not audit.errors and not audit.fixes
    reloaded_dims={key:{**value,'entity':check.entitydb[value['entity'].dxf.handle]}
                   for key,value in dims.items()}
    checked_profiles=validate(check,reloaded_dims,[check.entitydb[e.dxf.handle] for e in profiles])
    by_layer=defaultdict(Counter)
    for e in check.modelspace():by_layer[e.dxf.layer][e.dxftype()]+=1
    out=args.validation_dir
    out.mkdir(parents=True,exist_ok=True)
    from ezdxf.addons.drawing import Frontend,RenderContext,layout,config
    from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend
    def preview(name, entities=None, color=False):
        backend=PyMuPdfBackend()
        cfg=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
            color_policy=config.ColorPolicy.COLOR if color else config.ColorPolicy.BLACK)
        front=Frontend(RenderContext(check),backend,config=cfg)
        included=None if entities is None else {e.dxf.handle for e in entities}
        front.draw_layout(check.modelspace(),finalize=True,
            filter_func=None if included is None else lambda e:e.dxf.handle in included)
        page=layout.Page(420,297,layout.Units.mm) if included is None else layout.Page(
            0,0,layout.Units.mm,margins=layout.Margins.all(5))
        (out/name).write_bytes(backend.get_pixmap_bytes(page,fmt='png',dpi=180))
    preview('preview.png')
    preview('layers.png',color=True)
    preview('part-only.png',[e for e in check.modelspace() if e.dxf.layer.startswith('PART_') or e.dxf.layer=='THREAD'])
    report={
        'source':spec['source'],'source_sha256':hashlib.sha256((ROOT/spec['source']).read_bytes()).hexdigest(),
        'output':args.output.name,'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'dxf_version':check.dxfversion,'units':'mm','ground_truth_dxf_used':False,
        'recognition_mode':spec['recognition_mode'],'status':'needs_engineering_review',
        'working_layer_count':len(by_layer),'layers':dict(by_layer),'entities':records,
        'dimensions':{k:{'handle':v['entity'].dxf.handle,'measurement':v['value'],
            'expected':v['expected'],'text':v['entity'].dxf.text,'view_scale':v['view_scale']}
            for k,v in dims.items()},'profiles':checked_profiles,
        'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),
        'native_dimension_edit_test':f"passed: {dims['overall_width']['value']:g} -> {dims['overall_width']['value']+1:g}, checked regenerated text",
        'semantic_regeneration_test':'passed: width +1 changes contour, overall and shoulder dimensions',
        'cad_application_test':'not_run','review':spec['review'],
    }
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'layers':dict(by_layer),
                     'dimensions':len(dims),'profiles_checked':len(checked_profiles)},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()

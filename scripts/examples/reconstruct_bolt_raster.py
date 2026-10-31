"""Assisted reconstruction of bolt1_1 raster PDF; not a general ML converter.

Only the supplied raster PDF is read. Labels/regions were checked visually.
Geometry is reconstructed in mm, with editable DIMENSION and TEXT entities.
Residual drafting symbols are traced as vector hatches (no embedded bitmap).
"""
from pathlib import Path
import os
import sys
import math
import json
import hashlib
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ['TEMP']) / '44labs-bolt1-1-work'
WORK.mkdir(exist_ok=True)
os.environ['XDG_CACHE_HOME'] = str(WORK / 'cache')
# Dependencies retained from the earlier conversion experiment.
DEPS = Path(os.environ['TEMP']) / '44labs-bolt-artifacts-bad6e25064e94550a32bab05af03086d' / 'pdf-dxf-deps'
sys.path.append(str(DEPS))
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from PIL import Image
from pypdf import PdfReader
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.addons.drawing import Frontend, RenderContext, layout, config
from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

SOURCE = ROOT / 'болт1_1.pdf'
OUTPUT = ROOT / 'болт1_1.dxf'
reader = PdfReader(SOURCE)
assert len(reader.pages) == 1
page = reader.pages[0]
assert len(page.images) == 1 and not page.extract_text().strip()
im = np.array(page.images[0].image.convert('L'))
assert im.shape == (1965, 2779)
ink = (im < 180).astype(np.uint8) * 255
remaining = ink.copy()

def edge_column(x, y1, y2):
    a = 255 - im[y1:y2, x-3:x+4].astype(float)
    weights = a.sum(axis=0)
    return float(np.dot(np.arange(x-3, x+4), weights) / weights.sum())

under_px = edge_column(486, 440, 470)
end_px = edge_column(1670, 445, 520)
SCALE = 180.0 / (end_px - under_px)
ORIGIN = (6.0, 1960.0)

def xy(p):
    return ((float(p[0])-ORIGIN[0])*SCALE,
            (ORIGIN[1]-float(p[1]))*SCALE)

def px(p):
    return (float(p[0])/SCALE+ORIGIN[0], ORIGIN[1]-float(p[1])/SCALE)

doc = ezdxf.new('R2007', setup=True, units=4)
doc.header['$INSUNITS'] = 4
doc.header['$MEASUREMENT'] = 1
doc.header['$LUNITS'] = 2
doc.header['$LUPREC'] = 4
doc.header['$LWDISPLAY'] = False
doc.header['$LIMMIN'] = (0, 0)
doc.header['$LIMMAX'] = (420, 297)
doc.appids.new('RASTER_RECONSTRUCTION')
for name, weight in [('MAIN',25), ('THIN',15), ('AXIS',15), ('DIMS',15), ('TEXT',15)]:
    doc.layers.new(name, dxfattribs={'color':7, 'lineweight':weight})
doc.styles.new('RASTER_TEXT', dxfattribs={'font':'arial.ttf'})
doc.styles.new('RASTER_ITALIC', dxfattribs={'font':'ariali.ttf'})
doc.linetypes.new('RASTER_CENTER',dxfattribs={'description':'Center line, mm',
                                            'pattern':[8.0,5.0,-1.0,1.0,-1.0]})
doc.dimstyles.new('RASTER_MM', dxfattribs={
    'dimtxt':5.0, 'dimasz':2.5, 'dimexo':0.0, 'dimexe':1.5,
    'dimgap':1.0, 'dimtad':1, 'dimdec':3, 'dimzin':8,
    'dimdsep':44, 'dimtxsty':'RASTER_ITALIC', 'dimscale':1.0,
    'dimlfac':1.0,
})
msp = doc.modelspace()
labels = []
dimension_report = []

def line(a, b, layer='MAIN', target=None):
    return (msp if target is None else target).add_line(a, b, dxfattribs={'layer':layer})

def extract_lines(mask, target, layer='THIN', min_length=50):
    """Extract both directions before erasing, retaining grid intersections."""
    erase_all = np.zeros_like(mask)
    for horizontal in (True,False):
        k = max(20,min_length-12)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,1) if horizontal else (1,k))
        opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        n,cc,stats,centroids=cv2.connectedComponentsWithStats(opened,8)
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if (horizontal and (w<min_length or h>8)) or (not horizontal and (h<min_length or w>8)):
                continue
            if horizontal:
                v=float(centroids[i][1]); a,b=(x,v),(x+w-1,v)
            else:
                v=float(centroids[i][0]); a,b=(v,y),(v,y+h-1)
            line(xy(a),xy(b),layer,target)
            cv2.line(erase_all,tuple(np.round(a).astype(int)),tuple(np.round(b).astype(int)),255,3)
    result=mask.copy()
    result[erase_all>0]=0
    return result

def trace(mask, target, layer='THIN'):
    """Preserve thin symbols as filled vector outlines, including holes."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    hierarchy = hierarchy[0]
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue
        # Single-pixel branches have zero filled area. Keep a hairline outline
        # as well as fills so they do not disappear from dimension graphics.
        points=cv2.approxPolyDP(contour,0.35,True).reshape(-1,2)
        if len(points)>=2:
            target.add_lwpolyline([xy(v) for v in points],close=len(points)>2,
                                 dxfattribs={'layer':layer,'lineweight':0})
        if cv2.contourArea(contour)<0.8:
            continue
        hatch = target.add_hatch(color=7, dxfattribs={'layer':layer})
        def boundary(c, flags):
            p = cv2.approxPolyDP(c, 0.22, True).reshape(-1,2)
            if len(p) >= 3:
                hatch.paths.add_polyline_path([xy(v) for v in p], is_closed=True, flags=flags)
        boundary(contour, 1)
        child = hierarchy[i][2]
        while child != -1:
            boundary(contours[child], 0)
            child = hierarchy[child][0]
        if not hatch.paths:
            target.delete_entity(hatch)

def text_region(text, box, target=None, rotation=0, italic=True):
    """Replace a visually transcribed label; fit its visible raster bounds."""
    x1,y1,x2,y2 = box
    ys,xs = np.nonzero(ink[y1:y2,x1:x2])
    if len(xs) == 0:
        raise ValueError(('empty label',text,box))
    left,top = x1+int(xs.min()), y1+int(ys.min())
    right,bottom = x1+int(xs.max())+1, y1+int(ys.max())+1
    remaining[y1:y2,x1:x2] = 0
    target = msp if target is None else target
    if rotation == 90:
        start,finish = xy((right,bottom)),xy((right,top))
        height = (right-left)*SCALE
    elif rotation == 0:
        start,finish = xy((left,bottom)),xy((right,bottom))
        height = (bottom-top)*SCALE*(0.82 if len(text)>35 else 1.0)
    else:
        # Only the small 30-degree label uses an oblique baseline.
        start = xy((left,bottom-18))
        length = math.hypot(right-left,bottom-top)*SCALE*0.9
        finish = (start[0]+length*math.cos(math.radians(rotation)),
                  start[1]+length*math.sin(math.radians(rotation)))
        height = 4.8
    e = target.add_text(text, dxfattribs={
        'layer':'TEXT' if target is msp else 'DIMS', 'height':height,
        'style':'RASTER_ITALIC' if italic else 'RASTER_TEXT',
    })
    e.set_placement(start,finish,align=TextEntityAlignment.FIT)
    labels.append({'text':text,'source_box_px':list(box),'rotation':rotation})
    return e

def corridor(paths, width=18):
    mask = np.zeros_like(ink)
    for path in paths:
        pts = np.array(path,dtype=np.int32)
        cv2.polylines(mask,[pts],False,255,width)
    return mask

def dimension(dim, label, box, paths, expected=None, rotation=0, width=18):
    """Native definition plus a faithful vector display block from the scan."""
    dim.render()
    entity = dim.dimension
    block = doc.blocks[entity.dxf.geometry]
    block.delete_all_entities()
    text_region(label,box,block,rotation)
    mask = corridor(paths,width)
    pixels = cv2.bitwise_and(remaining,mask)
    # Keep arrow fills intact while adding centre lines for zero-area strokes.
    extract_lines(pixels,block,'DIMS',25)
    trace(pixels,block,'DIMS')
    remaining[mask>0] = 0
    # Leave nominal values dynamic when a CAD user edits the definition.
    dynamic={'180':'<>','6':'<>','34,6':'<>','15±0,5':'<>±0,5',
             '30±1,5':'<>±1,5','4×20°':'<>×20%%d','%%c27':'%%c<>',
             '%%c22':'%%c<>','%%c18':'%%c<>','M20×1,5-6h':'M<>×1,5-6h',
             '30%%d':'<>%%d','R0,5±0,3':'R<>±0,3'}
    entity.dxf.text=dynamic.get(label,label)
    entity.set_xdata('RASTER_RECONSTRUCTION',[(1000,'Visually checked label; raster-derived display'),(1000,label)])
    try:
        measurement = float(entity.get_measurement())
    except (TypeError,ValueError):
        measurement = None
    dimension_report.append({'label':label,'handle':entity.dxf.handle,
                             'measurement_mm_or_deg':measurement,'expected':expected})
    return entity

def linear_dim(p1,p2,base,angle,label,box,paths,expected=None,rotation=0,width=18):
    d=msp.add_linear_dim(base=base,p1=p1,p2=p2,angle=angle,text=label,
                         dimstyle='RASTER_MM',dxfattribs={'layer':'DIMS'})
    return dimension(d,label,box,paths,expected,rotation,width)

X = xy((under_px,0))[0]
Y = xy((0,483.25))[1]
XL, XE, XT = X-15.0, X+180.0, X+150.0
front = xy((2163.9,483.25))
RHEAD = 17.3
half_flat = math.sqrt(3.0)*RHEAD/2.0

# Dimensions are extracted before the overlapping body geometry is cleared.
linear_dim((X,Y),(XE,Y),xy((800,719)),0,'180',(1090,675,1200,716),
           [[(486,600),(486,732)],[(486,719),(1671,719)],[(1670,551),(1670,733)]],180)
linear_dim((XL,Y),(X,Y),xy((300,719)),0,'15±0,5',(195,665,365,712),
           [[(388,574),(388,733)],[(205,719),(486,719)]],15)
linear_dim((X,Y),(X+6,Y),xy((500,675)),0,'6',(565,625,623,672),
           [[(445,675),(612,675)],[(526,555),(526,688)]],6)
linear_dim((XT,Y),(XE,Y),xy((1500,679)),0,'30±1,5',(1480,625,1660,675),
           [[(1472,551),(1472,694)],[(1472,679),(1670,679)]],30)
linear_dim((XE-4,Y),(XE,Y),xy((1700,607)),0,'4×20°',(1730,558,1867,603),
           [[(1644,419),(1644,620)],[(1604,607),(1859,607)]],4)
linear_dim((XL,Y-13.5),(XL,Y+13.5),xy((270,500)),90,'%%c27',(220,420,264,553),
           [[(270,395),(270,572)],[(257,395),(393,395)],[(257,572),(393,572)]],27,90)
linear_dim((X,Y-11),(X,Y+11),xy((334,500)),90,'%%c22',(283,437,328,536),
           [[(334,411),(334,555)],[(320,411),(388,411)],[(320,555),(388,555)]],22,90)
linear_dim((X+80,Y-9),(X+80,Y+9),xy((1091,500)),90,'%%c18',(1040,302,1088,400),
           [[(1091,306),(1091,543)]],18,90)
linear_dim((XT,Y-10),(XT,Y+10),xy((1730,500)),90,'M20×1,5-6h',(1678,112,1727,407),
           [[(1730,128),(1730,549)],[(1644,418),(1744,418)],[(1644,549),(1744,549)]],20,90)
linear_dim((front[0],front[1]-RHEAD),(front[0],front[1]+RHEAD),xy((2440,500)),90,
           '34,6',(2383,448,2435,561),
           [[(2440,370),(2440,597)],[(2164,370),(2454,370)],[(2164,597),(2454,597)]],34.6,90)
linear_dim((front[0]-half_flat,front[1]),(front[0]+half_flat,front[1]),xy((2200,687)),0,
           '32*',(2115,638,2215,681),
           [[(2066,541),(2066,700)],[(2262,541),(2262,700)],[(2066,687),(2262,687)]],None)

angle_center = (XL,Y+13.5)
ang = msp.add_angular_dim_2l(base=xy((415,263)),
    line1=(angle_center,(angle_center[0]+10,angle_center[1]+10*math.sqrt(3))),
    line2=(angle_center,(angle_center[0],angle_center[1]+20)),
    text='30%%d',dimstyle='RASTER_MM',dxfattribs={'layer':'DIMS'})
dimension(ang,'30%%d',(390,208,468,259),
          [[(388,244),(388,395)],[(388,395),(461,269)],[(388,256),(410,259),(435,267),(462,279)]],30,-30,20)

# Transition radii are specified as a range, not a single exact nominal.
FILLET = 12.5
left_tangent_x = X+6+math.sqrt(2*FILLET*2-2*2)
right_tangent_x = XT-math.sqrt(2*FILLET*1-1)
left_center = (left_tangent_x,Y+9+FILLET)
right_center = (right_tangent_x,Y+9+FILLET)
for center,radius,tip,label,box,paths in [
    ((X+0.5,Y+11.5),0.5,(X+0.5,Y+11),'R0,5±0,3',(572,180,782,232),
     [[(486,411),(567,238),(779,238)]]),
    (left_center,FILLET,(X+6,Y+11),'R10...15',(635,288,810,332),
     [[(534,419),(631,336),(807,336)]]),
    (right_center,FILLET,(XT,Y+10),'R10...15',(1133,250,1305,294),
     [[(1453,421),(1327,298),(1137,298)]]),
]:
    d=msp.add_radius_dim(center=center,radius=radius,
        angle=math.degrees(math.atan2(tip[1]-center[1],tip[0]-center[0])),
        text=label,dimstyle='RASTER_MM',dxfattribs={'layer':'DIMS'})
    dimension(d,label,box,paths,radius)

# Native text. Deliberately missing source fields and the erased line before
# 'оси резьбы...' are not filled from the earlier vector PDF.
texts = [
 ('1. Допускается изготовление из Сталь 40Х ГОСТ 4543–71.',(1530,768,2650,819),0,True),
 ('2. Группа контроля III по ТУ 108–22–025–79.',(1530,822,2605,878),0,True),
 ('3. 28..35HRCэ (255..311HB, d10 3,8..3,45)',(1530,882,2470,938),0,True),
 ('4. Неуказанные предельные отклонения размеров:',(1530,940,2690,991),0,True),
 ('5.* Размеры для справок.',(1530,1050,2135,1106),0,True),
 ('оси резьбы не более 0,1мм на %%c30мм.',(1460,1163,2345,1220),0,True),
 ('7. Покрытие Кд9.Хр с последующим удалением',(1530,1220,2605,1276),0,True),
 ('водородной хрупкости.',(1460,1277,2045,1333),0,True),
 ('8. Технические требования по ГОСТ 1759–70.',(1530,1334,2580,1389),0,True),
 ('9. Резьбу контролировать до покрытия.',(1530,1390,2510,1446),0,True),
 ('Болт',(2050,1710,2230,1790),0,True),
 ('Литер.',(2410,1668,2495,1695),0,False),
 ('Масса',(2505,1668,2595,1695),0,False),
 ('Масшт.',(2615,1668,2731,1695),0,False),
 ('А',(2415,1715,2440,1763),0,False),
 ('0,55',(2510,1715,2605,1770),0,False),
 ('1:1',(2640,1715,2717,1770),0,False),
 ('Лист',(2410,1800,2496,1825),0,False),
 ('Листов 1',(2552,1800,2690,1825),0,False),
 ('Шестигр.',(1954,1858,2097,1905),0,True),
 ('32–5 ГОСТ8560–78',(2100,1840,2380,1880),0,True),
 ('45Х ГОСТ1051–88',(2100,1885,2375,1923),0,True),
 ('Копировал:',(1940,1930,2100,1961),0,True),
 ('Формат А3',(2450,1930,2640,1961),0,True),
 ('Изм.',(1524,1699,1564,1728),0,True),
 ('Лист',(1570,1699,1630,1728),0,True),
 ('№ докум.',(1640,1699,1776,1728),0,True),
 ('Подпись',(1790,1699,1880,1728),0,True),
 ('Дата',(1887,1699,1946,1728),0,True),
 ('Разраб.',(1524,1733,1630,1760),0,True),
 ('Провер.',(1524,1766,1630,1792),0,True),
 ('Рук. гр.',(1524,1799,1630,1825),0,True),
 ('Нач. бюро',(1524,1832,1630,1859),0,True),
 ('Нормок.',(1524,1865,1630,1892),0,True),
 ('Утв.',(1524,1898,1630,1925),0,True),
]
for args in texts:
    text_region(args[0],args[1],rotation=args[2],italic=args[3])

# Preserve small rotated margin lettering as vector outlines: exact glyphs
# are more useful here than an uncertain transcription or font substitution.

# Remove only body regions, retaining surface-finish symbols above surfaces.
remaining[367:600,385:489] = 0
remaining[408:558,489:1675] = 0
remaining[367:600,2048:2281] = 0
remaining[480:487,360:1852] = 0
remaining[480:487,2020:2308] = 0
remaining[340:630,2161:2168] = 0

def geometry_line(a,b):
    return line(a,b)

def arc_points(points):
    a,b,c = np.array([xy(p) for p in points])
    center=np.linalg.solve(2*np.vstack((b-a,c-a)),np.array([b@b-a@a,c@c-a@a]))
    radius=float(np.linalg.norm(a-center))
    angles=[math.degrees(math.atan2(*(p-center)[::-1]))%360 for p in (a,b,c)]
    start,middle,end=angles
    if (middle-start)%360 > (end-start)%360:
        start,end=end,start
    return msp.add_arc(center,radius,start,end,dxfattribs={'layer':'MAIN'})

# Bolt side view, nominal dimensions constrained by the visible labels.
geometry_line((XL,Y+13.5),(XL,Y-13.5))
geometry_line((X,Y-RHEAD),(X,Y+RHEAD))
hc = xy((402,483.25))[0]
for sign in (-1,1):
    geometry_line((hc,Y+sign*RHEAD),(X,Y+sign*RHEAD))
    geometry_line((XL,Y+sign*13.5),(hc,Y+sign*RHEAD))
    geometry_line((hc,Y+sign*8.65),(X,Y+sign*8.65))
    geometry_line((XL,Y+sign*11),(X,Y+sign*11))
    geometry_line((X+0.5,Y+sign*11),(X+6,Y+sign*11))
    geometry_line((left_tangent_x,Y+sign*9),(right_tangent_x,Y+sign*9))
    geometry_line((XT,Y+sign*10),(XE-4,Y+sign*10))
    chamfer_height=10-4*math.tan(math.radians(20))
    geometry_line((XE-4,Y+sign*10),(XE,Y+sign*chamfer_height))
    line((XT,Y+sign*chamfer_height),(XE,Y+sign*chamfer_height),'THIN')
    if sign==1:
        msp.add_arc((X+.5,Y+11.5),.5,180,270,dxfattribs={'layer':'MAIN'})
    else:
        msp.add_arc((X+.5,Y-11.5),.5,90,180,dxfattribs={'layer':'MAIN'})
    for center,start_point,end_point in [
        ((left_tangent_x,Y+sign*(9+FILLET)),(X+6,Y+sign*11),(left_tangent_x,Y+sign*9)),
        ((right_tangent_x,Y+sign*(9+FILLET)),(right_tangent_x,Y+sign*9),(XT,Y+sign*10)),
    ]:
        aa=math.degrees(math.atan2(start_point[1]-center[1],start_point[0]-center[0]))%360
        bb=math.degrees(math.atan2(end_point[1]-center[1],end_point[0]-center[0]))%360
        if (bb-aa)%360>180: aa,bb=bb,aa
        msp.add_arc(center,FILLET,aa,bb,dxfattribs={'layer':'MAIN'})
geometry_line((X+6,Y-11),(X+6,Y+11))
geometry_line((XT,Y-10),(XT,Y+10))
geometry_line((XE-4,Y-10),(XE-4,Y+10))
geometry_line((XE,Y-chamfer_height),(XE,Y+chamfer_height))
arc_points([(402,370),(393,398.5),(402,426.5)])
arc_points([(402,426.5),(393,483.25),(402,540)])
arc_points([(402,540),(393,568.5),(402,597)])

# End view: outer diameter is constrained by 34.6; the visibly inconsistent
# 32* callout is retained and explicitly reported instead of distorting it.
msp.add_circle(front,RHEAD,dxfattribs={'layer':'MAIN'})
msp.add_circle(front,88.5*SCALE,dxfattribs={'layer':'MAIN'})
hexagon=[(front[0]+RHEAD*math.cos(math.radians(a)),
          front[1]+RHEAD*math.sin(math.radians(a))) for a in (90,30,-30,-90,-150,150)]
for i in range(6): geometry_line(hexagon[i],hexagon[(i+1)%6])

for a,b in [((360,483.25),(1851,483.25)),((2020,483.25),(2307,483.25)),((2163.9,340),(2163.9,628))]:
    line(xy(a),xy(b),'AXIS').dxf.linetype='RASTER_CENTER'

# Recreate the portions of A and K leaders obscured by the geometry masks.
# Outside their body intersections, their original strokes are preserved.
for a,b in [((486,512),(671,666)),((2132,530),(1942,667))]:
    source_mask=corridor([[a,b]],12)
    restore=cv2.bitwise_and(ink,source_mask)
    remaining=cv2.bitwise_or(remaining,restore)

# Restore tips of surface-finish marks without duplicating the body surface.
for x1,y1,x2,y2 in [(847,381,905,423),(1480,375,1545,417)]:
    remaining[y1:y2,x1:x2]=ink[y1:y2,x1:x2]

# Long orthogonal frame/table strokes become LINE entities. All remaining
# short symbols and margin labels retain their original raster-derived form.
remaining=extract_lines(remaining,msp)
trace(remaining,msp,'THIN')

psp=doc.layouts.get('Layout1')
psp.page_setup(size=(420,297),margins=(0,0,0,0),units='mm')
psp.add_viewport(center=(210,148.5),size=(420,297),
                 view_center_point=(210,148.5),view_height=297,status=2)
doc.set_modelspace_vport(height=310,center=(210,148.5))
doc.saveas(OUTPUT)

check=ezdxf.readfile(OUTPUT)
audit=check.audit()
assert not audit.errors and not audit.fixes, (audit.errors,audit.fixes)
assert len(check.modelspace().query('DIMENSION'))==15
assert len(check.modelspace().query('IMAGE'))==0
for d in check.modelspace().query('DIMENSION'):
    assert len(check.blocks[d.dxf.geometry].query('TEXT'))==1
for item in dimension_report:
    if item['expected'] is not None:
        assert abs(item['measurement_mm_or_deg']-item['expected'])<1e-6,item

# An independent read/edit/regenerate/reopen check; the delivered file is
# untouched. This tests native dimension behaviour, not CAD GUI compatibility.
edited=ezdxf.readfile(OUTPUT)
test_dim=edited.entitydb[dimension_report[0]['handle']]
assert test_dim.dxf.text=='<>'
test_dim.dxf.defpoint3=test_dim.dxf.defpoint3+(1,0,0)
assert abs(float(test_dim.get_measurement())-181.0)<1e-8
test_dim.render()
edited_path=WORK/'dimension-edit-test.dxf'
edited.saveas(edited_path)
edited_check=ezdxf.readfile(edited_path)
edited_dimension=edited_check.entitydb[dimension_report[0]['handle']]
assert abs(float(edited_dimension.get_measurement())-181.0)<1e-8
dimension_edit_test={'original_mm':180,'edited_mm':181,'reopened':True,
                     'delivered_file_modified':False,'performed_with':'ezdxf, not a CAD application'}

backend=PyMuPdfBackend()
render_config=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
    color_policy=config.ColorPolicy.BLACK,lineweight_scaling=1.0)
Frontend(RenderContext(check),backend,config=render_config).draw_layout(check.modelspace(),finalize=True)
preview=backend.get_pixmap_bytes(layout.Page(420,297,layout.Units.mm),fmt='png',dpi=180)
(WORK/'preview.png').write_bytes(preview)

report={
 'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'source_type':'single raster image, no PDF text or vector paths',
 'image_size_px':[2779,1965],'source_pdf_page_mm':[float(page.mediabox.width)*25.4/72,float(page.mediabox.height)*25.4/72],
 'output':OUTPUT.name,'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
 'dxf_version':check.dxfversion,'units':'mm','layout':'A3, viewport 1:1',
 'scale_mm_per_px':SCALE,'calibration':'visible 180 mm dimension, measured under-head and end edges',
 'calibration_px':[under_px,end_px],
 'modelspace_entities':dict(Counter(e.dxftype() for e in check.modelspace())),
 'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),
 'dimensions':dimension_report,'texts':labels,
 'dimension_edit_test':dimension_edit_test,
 'limitations':[
  'Assisted one-off reconstruction; labels and regions were selected visually.',
  '32* source callout conflicts with about 29.96 mm across flats of the visible end-view hexagon; source label retained.',
  'Transitions labelled R10...15 reconstructed with radius 12.5 mm, within the given range.',
  'Small head arcs and unlabelled front inner circle reconstructed approximately from raster.',
  'Missing source text and blank title-block fields remain missing.',
  'Arial substitutes the original lettering. Small marginal lettering/surface symbols remain vector outlines.',
  'The line beginning with валов по h14 is preserved as vector outlines because the final tolerance expression is ambiguous.',
  'The d10 subscript in the hardness note is represented on the baseline in editable text.',
  'Native dimensions are non-associative. Their initial display blocks preserve source graphics.',
  'CAD-application opening was not tested: no installed CAD found, QCAD download permission declined.',
 ],
}
(WORK/'reconstruction-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['output','dxf_version','scale_mm_per_px','modelspace_entities','audit_errors','audit_fixes']},ensure_ascii=False,indent=2))
print('Preview:',WORK/'preview.png')

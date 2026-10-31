"""One-off reconstruction of the supplied vector PDF, not a general OCR model."""
import os
import sys
import json
import math
import tempfile
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
WORK = ROOT / '.tmp'
os.environ['XDG_CACHE_HOME'] = str(WORK / 'cache')
sys.path.insert(0, str(WORK / 'pdf-dxf-deps'))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.addons.drawing import Frontend, RenderContext, layout, config
from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend
from ezdxf.math import BoundingBox2d

source = json.loads((Path(tempfile.gettempdir()) / '44labs-bolt-analysis.json').read_text(encoding='utf-8'))
paths = source['paths']
assert len(paths) == 1343
# Calibrate the PDF plot using the explicitly dimensioned 180 mm shaft length.
SCALE = 180.0 / (715.7 - 208.16)
ORIGIN = np.array([2.96, 2.0])

def xy(point):
    return tuple((np.array(point, dtype=float) - ORIGIN) * SCALE)

doc = ezdxf.new('R2007', setup=True, units=4)
doc.header['$INSUNITS'] = 4
doc.header['$MEASUREMENT'] = 1
doc.header['$LUNITS'] = 2
doc.header['$LUPREC'] = 4
doc.header['$AUNITS'] = 0
doc.header['$AUPREC'] = 2
doc.header['$LWDISPLAY'] = False
doc.header['$INSBASE'] = (0, 0, 0)
doc.header['$LIMMIN'] = (0, 0)
doc.header['$LIMMAX'] = (420, 297)
doc.appids.new('PDF_RECONSTRUCTION')
for name, weight in [('MAIN',25), ('THIN',18), ('AXIS',18), ('DIMS',25), ('TEXT',18)]:
    doc.layers.new(name, dxfattribs={'color':7, 'lineweight':weight})
doc.styles.new('CAD_TEXT', dxfattribs={'font':'arial.ttf'})
doc.styles.new('CAD_ITALIC', dxfattribs={'font':'ariali.ttf'})
doc.dimstyles.new('BOLT_MM', dxfattribs={
    'dimtxt':5, 'dimasz':3, 'dimexo':0, 'dimexe':2, 'dimgap':1,
    'dimtad':1, 'dimdec':1, 'dimzin':8, 'dimdsep':44,
    'dimtxsty':'CAD_ITALIC', 'dimclrd':0, 'dimclre':0, 'dimclrt':0,
    'dimscale':1, 'dimlfac':1,
})
msp = doc.modelspace()
converted = Counter()
curve_fits = {}

def sample_path(p):
    pts = []
    current = None
    for seg in p['segments']:
        if seg[0] in ('m','l'):
            current = np.array(seg[1], dtype=float)
            pts.append(current)
        elif seg[0] == 'c':
            a,b,c = map(lambda v: np.array(v,dtype=float),seg[1:])
            start = current.copy()
            for t in np.linspace(0,1,33)[1:]:
                pts.append((1-t)**3*start+3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3*c)
            current = c
    return np.array(pts)

def circle_fit(p):
    pts = sample_path(p)
    mean = pts.mean(axis=0)
    q = pts - mean
    sol = np.linalg.lstsq(np.column_stack([2*q[:,0],2*q[:,1],np.ones(len(q))]),(q*q).sum(axis=1),rcond=None)[0]
    center = sol[:2] + mean
    radius = math.sqrt(sol[2] + sol[0]**2 + sol[1]**2)
    err = float(np.max(np.abs(np.linalg.norm(pts-center,axis=1)-radius)))
    return pts,center,radius,err

def add_path(target, p, layer_name=None):
    if layer_name is None:
        pid = p['id']
        if 852 <= pid <= 908 or 936 <= pid <= 946 or 1096 <= pid <= 1106:
            layer_name = 'AXIS'
        elif 909 <= pid <= 935 or pid in (947,948):
            layer_name = 'MAIN'
        elif 949 <= pid <= 1095 or 1250 <= pid <= 1333:
            layer_name = 'DIMS'
        else:
            layer_name = 'THIN'
    attrs = {'layer':layer_name,'lineweight':25 if p['width']>=.7 else 18}
    if p['op'] == 'b':
        pts = [xy(seg[1]) for seg in p['segments'] if seg[0] in ('m','l')]
        if pts and pts[0] == pts[-1]:
            pts.pop()
        if len(pts) >= 3:
            hatch = target.add_hatch(color=7,dxfattribs={'layer':layer_name})
            hatch.paths.add_polyline_path(pts,is_closed=True)
            converted['filled_paths'] += 1
        return
    # Recover original circular primitives from the PDF's cubic Bezier encoding.
    if p['id'] in (914,917,920,923,925,926,947,948,1035):
        pts,center,radius,error = circle_fit(p)
        if error*SCALE < .015:
            curve_fits[p['id']] = {'center':center.tolist(),'radius':radius,'error_mm':error*SCALE}
            if np.linalg.norm(pts[0]-pts[-1]) < .001:
                # Full-circle PDF endpoints provide the original radius exactly.
                first = np.array(p['segments'][0][1])
                end_points=np.array([p['segments'][0][1]]+[s[3] for s in p['segments'] if s[0]=='c'])
                center=(end_points.min(axis=0)+end_points.max(axis=0))/2
                radius=float(np.linalg.norm(first-center))
                target.add_circle(xy(center),radius*SCALE,dxfattribs=attrs)
                converted['circles'] += 1
            else:
                angles=np.unwrap(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))
                start,end=map(math.degrees,[angles[0],angles[-1]])
                if end<start:
                    start,end=end,start
                target.add_arc(xy(center),radius*SCALE,start%360,end%360,dxfattribs=attrs)
                converted['arcs'] += 1
            return
    vertices=[]
    start=None
    current=None
    def flush():
        nonlocal vertices
        if len(vertices)==2:
            target.add_line(vertices[0],vertices[1],dxfattribs=attrs)
        elif len(vertices)>2:
            closed=vertices[0]==vertices[-1]
            target.add_lwpolyline(vertices[:-1] if closed else vertices,close=closed,dxfattribs=attrs)
        vertices=[]
    for seg in p['segments']:
        if seg[0]=='m':
            flush()
            start=current=xy(seg[1]); vertices=[current]
        elif seg[0]=='l':
            current=xy(seg[1]); vertices.append(current)
        elif seg[0]=='c':
            flush()
            control=[current]+[xy(v) for v in seg[1:]]
            target.add_open_spline([(x,y,0) for x,y in control],degree=3,dxfattribs=attrs)
            current=control[-1]; vertices=[current]
        elif seg[0]=='h' and vertices:
            vertices.append(start)
    flush()

def text_from_paths(target, text, ids, rotation=0, italic=True, layer_name='TEXT'):
    """Use native TEXT, fit to the original SHX strokes, keep explicit content."""
    theta=math.radians(rotation)
    u=np.array([math.cos(theta),math.sin(theta)])
    v=np.array([-math.sin(theta),math.cos(theta)])
    pts=np.array([value for i in ids for seg in paths[i]['segments'] for value in seg[1:]])
    local_u=pts@u; local_v=pts@v
    low,high=float(local_v.min()),float(local_v.max())
    candidates=[round(float(z),2) for z in local_v if z<=low+(high-low)*.4]
    baseline=Counter(candidates).most_common(1)[0][0]
    height=max(high-baseline,1.5)*SCALE
    p1=u*local_u.min()+v*baseline
    p2=u*local_u.max()+v*baseline
    e=target.add_text(text,dxfattribs={
        'layer':layer_name, 'style':'CAD_ITALIC' if italic else 'CAD_TEXT',
        'height':height, 'rotation':rotation,
    })
    e.set_placement(xy(p1),xy(p2),align=TextEntityAlignment.FIT)
    e.set_xdata('PDF_RECONSTRUCTION',[(1000,'Text recovered from source PDF; font substituted with Arial.')])
    converted['native_text']+=1
    return e

# Size callouts: retain the original dimension-line geometry in anonymous
# DIMENSION blocks, with editable native TEXT replacing outlined SHX glyphs.
specs=[
    dict(name='M20x1.5-6h', ids=(949,966), glyphs=(955,966), kind='linear',
         p1=(704.42,606.8),p2=(704.42,663.2),base=(741.38,635),angle=90,
         text='M<>x1,5-6h',display='M20x1,5-6h',rotation=90),
    dict(name='chamfer4x20',ids=(967,979),glyphs=(974,979),kind='linear',
         p1=(704.42,663.2),p2=(715.7,659.06),base=(710,582.02),angle=0,
         text='<>x20%%d',display='4x20%%d',rotation=0),
    dict(name='thread_length30',ids=(980,992),glyphs=(985,992),kind='linear',
         p1=(631.1,610.88),p2=(715.7,610.88),base=(670,551.12),angle=0,
         text='<>',display='30%%p1,5',rotation=0,tolerance=1.5),
    dict(name='shaft_length180',ids=(993,1000),glyphs=(998,1000),kind='linear',
         p1=(208.16,586.22),p2=(715.7,619.76),base=(450,534.26),angle=0,
         text='<>',display='180',rotation=0),
    dict(name='head_length15',ids=(1001,1014),glyphs=(1007,1014),kind='linear',
         p1=(165.86,613.64),p2=(208.16,592.88),base=(180,534.26),angle=0,
         text='<>',display='15%%p0,5',rotation=0,tolerance=.5),
    dict(name='diameter22',ids=(1015,1023),glyphs=(1020,1023),kind='diameter',
         center=(225.08,635),mpoint=(225.08,666.02),text='<>',display='%%c22',rotation=90),
    dict(name='diameter27',ids=(1024,1032),glyphs=(1029,1032),kind='diameter',
         center=(165.86,634.97),mpoint=(165.86,673.04),text='<>',display='%%c27',rotation=90),
    dict(name='angle30',ids=(1033,1040),glyphs=(1038,1040),kind='angular',
         center=(165.86,673.04),p1=(198.08,728.78),p2=(165.86,737.42),base=(180,730),
         text='<>',display='30%%d',rotation=-15),
    dict(name='left_radius_range',ids=(1041,1051),glyphs=(1042,1050),kind='radius',
         curve=917,text='R10...15',display='R10...15',rotation=0),
    dict(name='shoulder_length6',ids=(1052,1059),glyphs=(1059,1059),kind='linear',
         p1=(208.16,586.22),p2=(225.08,603.98),base=(220,552.98),angle=0,
         text='<>',display='6',rotation=0),
    dict(name='radius0.5',ids=(1064,1076),glyphs=(1065,1075),kind='radius',
         center=(209.63,667.43),mpoint=(208.22,667.43),text='<>',display='R0,5%%p0,3',rotation=0,tolerance=.3),
    dict(name='head_width32_reference',ids=(1077,1086),glyphs=(1082,1086),kind='linear',
         p1=(885.08,610.58),p2=(969.56,610.58),base=(927,547.58),angle=0,
         text='32*',display='32*',rotation=0,
         warning='Original PDF label 32* conflicts with measured width about 29.96 mm. Geometry and label preserved.'),
    dict(name='head_height34.6',ids=(1087,1095),glyphs=(1092,1095),kind='linear',
         p1=(927.32,586.22),p2=(927.32,683.78),base=(1045.22,635),angle=90,
         text='<>',display='34,6',rotation=90),
    dict(name='diameter18',ids=(1288,1297),glyphs=(1294,1297),kind='diameter',
         center=(467.3,635),mpoint=(467.3,660.38),text='<>',display='%%c18',rotation=90),
    dict(name='right_radius_range',ids=(1298,1308),glyphs=(1300,1308),kind='radius',
         curve=914,text='R10...15',display='R10...15',rotation=0),
]
reserved={i for s in specs for i in range(s['ids'][0],s['ids'][1]+1)}
removed=set(reserved)

# Recover all horizontal title-block and technical-note strings, plus the
# vertical filing-strip labels. Small inclined surface-finish labels retain
# their original vector lettering to avoid ambiguous font/rotation changes.
text_report=[]
for ann in source['annotations']:
    aid=ann['id']
    if aid>51 or aid in (6,7):
        continue
    x0,y0,x1,y1=ann['bbox']
    matches=[]
    for p in paths:
        a,b,c,d=p['bbox']
        if p['id'] in removed or p['id'] in (819,820,1320,1321):
            continue
        if a>=x0-.25 and b>=y0-.25 and c<=x1+.25 and d<=y1+.25:
            matches.append(p['id'])
    if aid==40:
        matches=[819,820]
    if not matches:
        continue
    text=ann['text']
    if aid==35:
        text='оси резьбы не более 0,1мм на %%c30мм.'
    regular=aid in (3,16,17,18,19,20,21,24,25,44,45,46,47,48,49,50,51)
    text_from_paths(msp,text,matches,rotation=90 if 44<=aid<=51 else 0,italic=not regular)
    removed.update(matches)
    text_report.append({'annotation':aid,'text':text,'paths':matches})

for p in paths:
    if p['id'] not in removed:
        add_path(msp,p)

dimension_report=[]
for s in specs:
    override={'dimdec':1,'dimzin':8,'dimdsep':44,'dimtxt':5,'dimtxsty':'CAD_ITALIC'}
    if 'tolerance' in s:
        override.update(dimtol=1,dimtp=s['tolerance'],dimtm=s['tolerance'],dimtdec=1,dimtfac=1)
    attrs={'layer':'DIMS'}
    common=dict(text=s['text'],dimstyle='BOLT_MM',override=override,dxfattribs=attrs)
    if s['kind']=='linear':
        wrapper=msp.add_linear_dim(xy(s['base']),xy(s['p1']),xy(s['p2']),angle=s['angle'],**common)
    elif s['kind']=='diameter':
        wrapper=msp.add_diameter_dim(xy(s['center']),mpoint=xy(s['mpoint']),**common)
    elif s['kind']=='angular':
        wrapper=msp.add_angular_dim_3p(xy(s['base']),xy(s['center']),xy(s['p1']),xy(s['p2']),**common)
    elif s['kind']=='radius':
        if 'curve' in s:
            pts,center,radius,error=circle_fit(paths[s['curve']])
            center=tuple(center)
            mpoint=tuple(pts[0])
        else:
            center,mpoint=s['center'],s['mpoint']
        wrapper=msp.add_radius_dim(xy(center),mpoint=xy(mpoint),**common)
    wrapper.render()
    dim=wrapper.dimension
    block=doc.blocks[dim.dxf.geometry]
    block.delete_all_entities()
    glyphs=list(range(s['glyphs'][0],s['glyphs'][1]+1))
    for i in range(s['ids'][0],s['ids'][1]+1):
        if i not in glyphs:
            add_path(block,paths[i],'DIMS')
    text=text_from_paths(block,s['display'],glyphs,rotation=s['rotation'],layer_name='DIMS')
    p1,p2=text.dxf.insert,text.dxf.align_point
    dim.dxf.text_midpoint=(p1+p2)*.5
    dim.dxf.text_rotation=s['rotation']
    dim.dxf.dimtype=dim.dxf.dimtype|128
    measurement=float(dim.get_measurement())
    dim.dxf.actual_measurement=measurement
    data=[(1000,s['name']),(1000,'Reconstructed from vector PDF. Not associative with source entities.')]
    if 'warning' in s:
        data.append((1000,s['warning']))
    dim.set_xdata('PDF_RECONSTRUCTION',data)
    dimension_report.append({'name':s['name'],'kind':s['kind'],'label':s['display'],
                             'measurement':measurement,'warning':s.get('warning')})

paper=doc.layouts.get('Layout1')
paper.name='A3'
paper.page_setup(size=(420,297),margins=(0,0,0,0),units='mm',scale=(1,1))
paper.add_viewport(center=(210,148.5),size=(420,297),view_center_point=(210,148.5),view_height=297,status=2)
doc.set_modelspace_vport(height=310,center=(210,148.5))
doc.ezdxf_metadata()['SOURCE']='болт1.pdf; vector reconstruction; calibration: shaft length 180 mm'
doc.ezdxf_metadata()['LIMITATIONS']='PDF plot precision; non-associative dimensions; source label 32* differs from geometry (~29.96 mm); Arial font substitution.'

output=ROOT/'болт1.dxf'
doc.saveas(output)
check=ezdxf.readfile(output)
audit=check.audit()
assert not audit.errors, [str(e) for e in audit.errors]
assert not audit.fixes, [str(e) for e in audit.fixes]
assert check.dxfversion=='AC1021'
assert check.header['$INSUNITS']==4
assert len(check.modelspace().query('DIMENSION'))==15
assert len(check.modelspace().query('IMAGE'))==0
length=next(d['measurement'] for d in dimension_report if d['name']=='shaft_length180')
assert abs(length-180)<1e-8

report={
    'output':str(output),'version':check.dxfversion,'units':'mm','paper_mm':[420,297],
    'scale_mm_per_pdf_point':SCALE,
    'model_entities':dict(Counter(e.dxftype() for e in check.modelspace())),
    'dimensions':dimension_report,'native_text_groups':len(text_report),
    'converted':dict(converted),'curve_fits':curve_fits,
    'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),
}
(WORK/'bolt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

backend=PyMuPdfBackend()
configuration=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,
                                   color_policy=config.ColorPolicy.BLACK,
                                   lineweight_scaling=.65)
Frontend(RenderContext(check),backend,config=configuration).draw_layout(check.modelspace(),finalize=True)
png=backend.get_pixmap_bytes(layout.Page(420,297),dpi=160,
                            settings=layout.Settings(fit_page=False,scale=1),
                            render_box=BoundingBox2d([(0,0),(420,297)]))
(WORK/'bolt-preview.png').write_bytes(png)
print(json.dumps(report,ensure_ascii=False,indent=2))
print('PREVIEW',WORK/'bolt-preview.png')
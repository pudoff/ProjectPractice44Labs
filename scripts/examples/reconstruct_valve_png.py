"""Assisted PNG-to-DXF reconstruction of the supplied valve drawing.

Reads only root/клапан.png. Region selection and transcription were checked
visually; this is not a general OCR/ML pipeline. Fine strokes are converted
to centre-line vectors, with native dimensions and editable principal text.
"""
from pathlib import Path
from collections import Counter
import os, sys, json, math, hashlib

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(os.environ['TEMP'])/'44labs-valve-work'
WORK.mkdir(exist_ok=True)
os.environ['XDG_CACHE_HOME']=str(WORK/'cache')
DEPS=Path(os.environ['TEMP'])/'44labs-bolt-artifacts-bad6e25064e94550a32bab05af03086d/pdf-dxf-deps'
sys.path.append(str(DEPS))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.addons.drawing import Frontend, RenderContext, layout, config
from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

SOURCE=ROOT/'клапан.png'
# Legacy tracing demo; keep it separate from the semantic v2 reconstruction.
OUT=ROOT/'клапан_codex_v1_trace.dxf'
im=np.asarray(Image.open(SOURCE).convert('L'))
H,W=im.shape
F=W/1888.0
remaining=(im<150).astype(np.uint8)*255
assert (W,H)==(9925,7017)

def box_px(box):
    return tuple(round(v*F) for v in box)

def edge(box,axis):
    x1,y1,x2,y2=box_px(box)
    a=255-im[y1:y2,x1:x2].astype(float)
    weights=a.sum(axis=0 if axis=='x' else 1)
    weights=np.where(weights>=weights.max()*.4,weights,0)
    offset=x1 if axis=='x' else y1
    return float(np.dot(np.arange(len(weights))+offset,weights)/weights.sum())

top=edge((640,252,670,262),'y')
bottom=edge((640,965,670,978),'y')
back=edge((448,400,457,440),'x')
front=edge((242,580,252,605),'x')
SX=46/(back-front)
SY=160/(bottom-top)
X0=23.0
Y0=float(np.where((im<150).sum(axis=1)>8500)[0].max())

def xy(p):
    return ((float(p[0])-X0)*SX,(Y0-float(p[1]))*SY)

def uv(p):
    return xy((p[0]*F,p[1]*F))

doc=ezdxf.new('R2007',setup=True,units=4)
for k,v in {'$INSUNITS':4,'$MEASUREMENT':1,'$LUNITS':2,'$LUPREC':4,
            '$LWDISPLAY':False,'$LIMMIN':(0,0),'$LIMMAX':(420,297)}.items():
    doc.header[k]=v
for name,lw in [('MAIN',20),('THIN',13),('AXIS',13),('DIMS',13),('TEXT',13)]:
    doc.layers.new(name,dxfattribs={'color':7,'lineweight':lw})
doc.styles.new('VALVE_TEXT',dxfattribs={'font':'arial.ttf'})
doc.styles.new('VALVE_ITALIC',dxfattribs={'font':'ariali.ttf'})
doc.dimstyles.new('VALVE_MM',dxfattribs={
    'dimtxt':5,'dimasz':2.5,'dimexo':0,'dimexe':1.5,'dimtad':1,
    'dimgap':1,'dimdec':1,'dimzin':8,'dimdsep':44,
    'dimtxsty':'VALVE_ITALIC','dimscale':1,'dimlfac':1,
})
doc.appids.new('PNG_RECONSTRUCTION')
msp=doc.modelspace()
dimensions=[]
text_report=[]

def text_region(text,box,target=None,rotation=0,italic=True,height_factor=1):
    target=msp if target is None else target
    x1,y1,x2,y2=box_px(box)
    ink=(im[y1:y2,x1:x2]<150).astype(np.uint8)
    angular_text=text=='60%%d'
    if angular_text:
        # The dimension arc crosses the rectangle. Only erase isolated glyphs;
        # the connected arc touches the boundary and must be preserved.
        count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
        mask=np.zeros_like(ink)
        for i in range(1,count):
            x,y,w,h,area=stats[i]
            if x>0 and y>0 and x+w<ink.shape[1] and y+h<ink.shape[0] and area>3:
                mask[labels==i]=1
        ink=mask
        remaining[y1:y2,x1:x2][mask>0]=0
    else:
        remaining[y1:y2,x1:x2]=0
    yy,xx=np.nonzero(ink)
    if not len(xx): raise ValueError(('Empty text region',text,box))
    l,r=x1+int(xx.min()),x1+int(xx.max())+1
    t,b=y1+int(yy.min()),y1+int(yy.max())+1
    if angular_text:
        a,z=uv((325,342)),uv((350,383)); height=5
    elif rotation==90:
        a,z=xy((r,b)),xy((r,t)); height=(r-l)*SX
    elif rotation==0:
        a,z=xy((l,b)),xy((r,b)); height=(b-t)*SY*height_factor
    else:
        a=xy((l,b-10*F))
        length=math.hypot((r-l)*SX,(b-t)*SY)*.85
        z=(a[0]+length*math.cos(math.radians(rotation)),a[1]+length*math.sin(math.radians(rotation)))
        height=5
    e=target.add_text(text,dxfattribs={'layer':'TEXT' if target is msp else 'DIMS',
       'height':height,'style':'VALVE_ITALIC' if italic else 'VALVE_TEXT'})
    e.set_placement(a,z,align=TextEntityAlignment.FIT)
    text_report.append({'text':text,'image_region_1888':list(box)})
    return e

def vector_fill(mask,target,offset=(0,0),layer='DIMS'):
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return
    hierarchy=hierarchy[0]
    for i,c in enumerate(contours):
        if hierarchy[i][3]!=-1:continue
        q=cv2.approxPolyDP(c,.4,True).reshape(-1,2)+np.array(offset)
        if len(q)<2:continue
        if cv2.contourArea(c)<1:
            target.add_lwpolyline([xy(p) for p in q],close=len(q)>2,dxfattribs={'layer':layer,'lineweight':0})
            continue
        hatch=target.add_hatch(color=7,dxfattribs={'layer':layer})
        hatch.paths.add_polyline_path([xy(p) for p in q],is_closed=True,flags=1)
        child=hierarchy[i][2]
        while child!=-1:
            hole=cv2.approxPolyDP(contours[child],.4,True).reshape(-1,2)+np.array(offset)
            if len(hole)>2:
                hatch.paths.add_polyline_path([xy(p) for p in hole],is_closed=True,flags=0)
            child=hierarchy[child][0]

def capture_dimension_graphics(paths,target,width=4):
    points=np.array([p for path in paths for p in path])*F
    pad=math.ceil(width*F)+2
    x1,y1=np.floor(points.min(axis=0)).astype(int)-pad
    x2,y2=np.ceil(points.max(axis=0)).astype(int)+pad+1
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(W,x2),min(H,y2)
    mask=np.zeros((y2-y1,x2-x1),np.uint8)
    for path in paths:
        points=np.round(np.array(path)*F-[x1,y1]).astype(np.int32)
        cv2.polylines(mask,[points],False,255,max(2,round(width*F)))
    local=remaining[y1:y2,x1:x2]
    pixels=cv2.bitwise_and(local,mask)
    vector_fill(pixels,target,(x1,y1))
    local[mask>0]=0

def dimension(dim,label,box,paths,nominal,rotation=0,view_scale=1,dynamic=None,width=4):
    dim.render()
    e=dim.dimension
    block=doc.blocks[e.dxf.geometry]
    block.delete_all_entities()
    text_region(label,box,block,rotation)
    capture_dimension_graphics(paths,block,width)
    if dynamic is not None:e.dxf.text=dynamic
    e.set_xdata('PNG_RECONSTRUCTION',[(1000,'Raster-derived, visually checked annotation'),
                                    (1000,label),(1040,float(view_scale))])
    measured=float(e.get_measurement())/view_scale
    dimensions.append({'label':label,'handle':e.dxf.handle,'nominal':nominal,
        'raster_measurement_mm_or_deg':measured,'delta':measured-nominal,
        'view_scale':view_scale,'text_override':e.dxf.text})
    return e

def linear(label,box,p1,p2,base,angle,paths,nominal,rotation=0,view_scale=1):
    def refine(p):
        return (REFINED_X.get(p[0],p[0]),REFINED_Y.get(p[1],p[1]))
    p1,p2=refine(p1),refine(p2)
    d=msp.add_linear_dim(base=uv(base),p1=uv(p1),p2=uv(p2),angle=angle,
        text=label,dimstyle='VALVE_MM',override={'dimlfac':1/view_scale},
        dxfattribs={'layer':'DIMS'})
    # Keep the source nominal annotation; do not hide raster measurement error.
    return dimension(d,label,box,paths,nominal,rotation,view_scale)

print('Extracting semantic dimensions and text...',flush=True)
T,B=top/F,bottom/F
L,R=front/F,back/F
C=(T+B)/2
# Visible source anchors, refined where the long projection lines are clear.
Y135T=edge((470,307,550,318),'y')/F
Y135B=edge((470,910,550,922),'y')/F
Y150T=edge((470,273,550,285),'y')/F
Y150B=edge((470,943,550,956),'y')/F
REFINED_X={k:edge(box,'x')/F for k,box in {
    278.5:(276,603,282,609),301.2:(297,609,304,621),
    390.3:(388,610,393,630),390:(388,610,393,630),
    292:(289,775,295,805),410.3:(407,976,414,986),
    428.3:(425,1003,431,1016),383.1:(380,506,386,525),
    168.3:(165,1230,172,1250),302.6:(300,1230,306,1240),
}.items()}
REFINED_X[247]=L
REFINED_Y={k:edge(box,'y')/F for k,box in {
    521:(154,517,180,525),709:(154,706,180,713),
    543.5:(190,540,207,547),686:(190,682,207,690),
    552:(225,548,242,556),677:(225,673,242,681),
    566:(348,562,376,570),664:(348,659,376,667),
    588.8:(315,586,370,591),642.4:(315,641,370,645),
    275.8:(965,272,1030,279),309.6:(965,306,1030,313),
    679.6:(785,676,815,684),822.4:(785,818,815,826),
    1122.1:(177,1118,260,1126),1189.3:(177,1185,260,1193),
}.items()}
assert all(math.isfinite(v) for v in [*REFINED_X.values(),*REFINED_Y.values()])

linear('%%c160',(650,596,681,677),(R,T),(R,B),(683,600),90,
       [[(543,T),(692,T)],[(R,B),(693,B)],[(683,T),(683,B)]],160,90)
linear('%%c150±0,5',(609,565,642,722),(R,Y150T),(R,Y150B),(645,600),90,
       [[(R,Y150T),(655,Y150T)],[(R,Y150B),(655,Y150B)],[(645,Y150T),(645,Y150B)]],150,90)
linear('%%c135',(569,621,597,696),(410,Y135T),(410,Y135B),(600,600),90,
       [[(410,Y135T),(610,Y135T)],[(R,Y135B),(609,Y135B)],[(600,Y135T),(600,Y135B)]],135,90)
linear('%%c42',(116,574,144,639),(292,521),(292,709),(147,615),90,
       [[(138,521),(292,521)],[(138,709),(292,709)],[(147,521),(147,709)]],42,90)
linear('%%c32',(151,574,178,641),(247,543.5),(247,686),(181,615),90,
       [[(172,543.5),(247,543.5)],[(172,686),(247,686)],[(181,543.5),(181,686)]],32,90)
linear('%%c28',(188,581,216,641),(247,552),(247,677),(219,615),90,
       [[(209,552),(247,552)],[(209,677),(247,677)],[(219,552),(219,677)]],28,90)
linear('%%c22',(514,583,542,650),(390,566),(390,664),(546,615),90,
       [[(390,664),(555,664)],[(546,585),(546,664)]],22,90)
linear('M12×1,5-H7',(460,663,492,839),(390,588.8),(390,642.4),(497,615),90,
       [[(278,588.8),(505,588.8)],[(390,642.4),(506,642.4)],[(497,588.8),(497,843)]],12,90)

linear('25',(288,427,336,456),(278.5,615),(390.3,615),(300,458.8),0,
       [[(278.5,449),(278.5,594)],[(390.3,449),(390.3,487)],[(278.5,458.8),(390.3,458.8)]],25)
linear('20',(323,716,374,746),(301.2,615),(390.3,615),(330,749.4),0,
       [[(301.2,638),(301.2,758)],[(301.2,749.4),(390.3,749.4)]],20)
linear('14',(401,716,443,746),(390.3,615),(R,615),(420,749.4),0,
       [[(390.3,749.4),(R,749.4)]],14)
linear('10',(246,793,286,823),(L,615),(292,615),(260,826.8),0,
       [[(L,678),(L,836)],[(292,709),(292,836)],[(L,826.8),(292,826.8)]],10)
linear('46',(298,854,344,885),(L,615),(R,615),(320,888.8),0,
       [[(L,835),(L,898)],[(R,836),(R,898)],[(L,888.8),(R,888.8)]],46)
linear('4',(361,959,386,985),(410.3,960),(428.3,960),(419,989.8),0,
       [[(410.3,972),(410.3,1000)],[(360,989.8),(454,989.8)]],4)
linear('5,5',(479,992,530,1024),(428.3,960),(R,960),(440,1029.9),0,
       [[(428.3,972),(428.3,1040)],[(R,973),(R,1040)],[(400,1029.9),(532,1029.9)]],5.5)
linear('1,6×45%%d',(459,449,566,481),(383.1,588),(390.3,588),(386,487.2),0,
       [[(383.1,478),(383.1,641)],[(390.3,478),(390.3,664)],[(356,487.2),(568,487.2)]],1.6)

# Detail A is drawn at 5:1; DIMLFAC=0.2 returns its real dimension.
linear('1,5',(1009,203,1041,249),(903,275.8),(869,309.6),(1046,290),90,
       [[(903,275.8),(1056,275.8)],[(869,309.6),(1056,309.6)],[(1046,203),(1046,334)]],1.5,90,5)
linear('%%c32',(741,713,770,773),(839,679.6),(839,822.4),(773,750),90,
       [[(764,679.6),(839,679.6)],[(764,822.4),(839,822.4)],[(773,679.6),(773,822.4)]],32,90)
linear('15',(102,1129,130,1172),(168.3,1122.1),(168.3,1189.3),(133.5,1150),90,
       [[(123,1122.1),(168.3,1122.1)],[(123,1189.3),(168.3,1189.3)],[(133.5,1122.1),(133.5,1189.3)]],15,90)
linear('30',(205,1226,249,1255),(168.3,1150),(302.6,1150),(235,1257.6),0,
       [[(168.3,1189.3),(168.3,1268)],[(302.6,1189.3),(302.6,1268)],[(168.3,1257.6),(302.6,1257.6)]],30)

cc=uv((410.3,313))
ang=msp.add_angular_dim_2l(base=uv((322,342)),
    line1=(cc,(cc[0]-20,cc[1])),
    line2=(cc,(cc[0]-10,cc[1]-10*math.sqrt(3))),text='60%%d',dimstyle='VALVE_MM',dxfattribs={'layer':'DIMS'})
dimension(ang,'60%%d',(324,330,370,388),
    [[(311,313),(410.3,313)],[(320,313),(322,330),(330,351),(345,375),(367,393)]],60,-60,width=6)

# Readable notes and stamp fields become editable text. Radius callouts and
# fine tolerance/surface symbols remain original vector strokes to avoid
# inventing associations or altering superscripts.
texts=[
 ('1. Требования предъявляемые к отливке,',(1110,416,1754,453),True),
 ('по ТТМ 65–78.',(1064,456,1302,491),True),
 ('2. Неуказанные литейные радиусы 3...5мм.',(1110,494,1775,531),True),
 ('3. Неуказанные предельные отклонения',(1110,533,1737,570),True),
 ('размеров на механическую обработку:',(1064,573,1664,610),True),
 ('H14, h14, ±IT15/2.',(1064,611,1375,646),True),
 ('4. Покрытие: лак ПБС–2 ГОСТ901–78 IIА',(1110,648,1764,686),True),
 ('на необрабатываемых поверхностях.',(1064,688,1660,723),True),
 ('5. Маркировать шрифтом ПО–12',(1110,726,1646,761),True),
 ('ГОСТ 2930–62 в литье. Выпуклость букв 1,5мм.',(1064,764,1827,800),True),
 ('6. Допускается использовать отливку',(1110,802,1730,839),True),
 ('детали 0130–04–124.',(1064,841,1417,878),True),
 ('Клапан',(1400,1162,1524,1196),True),
 ('предохранительный',(1330,1199,1630,1234),True),
 ('Ал–9 ГОСТ2685–75',(1329,1253,1631,1293),True),
 ('Литер.',(1638,1134,1699,1152),False),
 ('Масса',(1707,1134,1769,1152),False),
 ('Масшт.',(1785,1134,1856,1152),False),
 ('А',(1640,1164,1661,1201),False),
 ('0,385',(1707,1173,1773,1202),False),
 ('1:1',(1795,1163,1846,1201),False),
 ('Лист',(1640,1223,1693,1240),False),
 ('Листов 1',(1734,1223,1829,1240),False),
 ('Изм.',(1035,1154,1063,1174),True),
 ('Лист',(1067,1154,1107,1174),True),
 ('№ докум.',(1117,1154,1208,1174),True),
 ('Подпись',(1215,1154,1276,1174),True),
 ('Дата',(1281,1154,1321,1174),True),
 ('Разраб.',(1035,1179,1102,1196),True),
 ('Провер.',(1035,1200,1105,1218),True),
 ('Рук. гр.',(1035,1222,1110,1240),True),
 ('Нач. бюро',(1035,1245,1114,1263),True),
 ('Нормок.',(1035,1267,1114,1285),True),
 ('Утв.',(1035,1290,1090,1307),True),
 ('Копировал:',(1324,1311,1423,1333),True),
 ('Формат А3',(1670,1311,1790,1333),True),
 ('Вариант',(757,534,888,571),True),
 ('А (5:1)',(832,84,946,140),True),
]
for i,(text,box,italic) in enumerate(texts):
    text_region(text,box,italic=italic,height_factor=.84 if i<12 or text=='А (5:1)' else 1)

# Isolate broad arrow/datum fills from thin drafting strokes. Skeleton edges
# preserve the surrounding fine strokes; filled markers remain vector hatches.
print('Preserving filled markers...',flush=True)
thick=cv2.morphologyEx(remaining,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
vector_fill(thick,msp,layer='THIN')
del thick

print('Tracing high-resolution stroke centre lines...',flush=True)
skel=skeletonize(remaining>0)
del remaining
coords=np.flatnonzero(skel)
S=skel.ravel()
size=S.size

# Exclude diagonal shortcuts when an orthogonal path already joins pixels.
def neighbours(v):
    y,x=divmod(v,W)
    for dy,dx in ((-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)):
        if not (0<=y+dy<H and 0<=x+dx<W):continue
        q=v+dy*W+dx
        if not S[q]:continue
        if dx and dy and (S[v+dx] or S[v+dy*W]):continue
        yield q

degree={int(v):sum(1 for _ in neighbours(int(v))) for v in coords}
visited=set()
stats=Counter()

def edge_key(a,b):return min(a,b)*size+max(a,b)

def entity_layer(points):
    mid=points[len(points)//2]/F
    u,v=mid
    if (234<u<466 and abs(v-C)<.8) or (814<u<934 and abs(v-751)<.8):
        return 'AXIS'
    geometry=((246<u<454 and 255<v<974) or (744<u<960 and 163<v<363)
              or (837<u<928 and 636<v<857) or (149<u<339 and 1091<v<1224))
    if not geometry:return 'THIN'
    diff=points[-1]-points[0]
    # Section hatching runs approximately at 45 degrees in the raster.
    if np.linalg.norm(diff)>15 and abs(abs(diff[0])-abs(diff[1]))<.035*np.linalg.norm(diff):
        return 'THIN'
    return 'MAIN'

def emit(path):
    if len(path)<3:return
    p=np.array([(v%W,v//W) for v in path],dtype=np.float64)
    length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    if length<2.5:return
    layer_name=entity_layer(p)
    attrs={'layer':layer_name}
    closed=path[0]==path[-1]
    chord=p[-1]-p[0]
    cl=float(np.linalg.norm(chord))
    if cl>2:
        dev=np.abs((p[:,0]-p[0,0])*chord[1]-(p[:,1]-p[0,1])*chord[0])/cl
        if dev.max()<=1.0:
            msp.add_line(xy(p[0]),xy(p[-1]),dxfattribs=attrs);stats['LINE']+=1;return
    # Recover circular geometry where a genuine arc is supported by the pixels.
    if len(p)>=12 and length>18:
        q=p-p.mean(axis=0)
        coeff=np.linalg.lstsq(np.column_stack((2*q[:,0],2*q[:,1],np.ones(len(q)))),(q*q).sum(axis=1),rcond=None)[0]
        radius2=coeff[2]+coeff[0]**2+coeff[1]**2
        if radius2>0:
            center=coeff[:2]+p.mean(axis=0);rad=math.sqrt(radius2)
            error=np.max(np.abs(np.linalg.norm(p-center,axis=1)-rad))
            angles=np.unwrap(np.arctan2(-(p[:,1]-center[1]),p[:,0]-center[0]))
            span=abs(float(angles[-1]-angles[0]))
            if error<=.75 and rad>3 and rad<10000 and span>.12:
                if closed and span>5.8:
                    msp.add_circle(xy(center),rad*(SX+SY)/2,dxfattribs=attrs);stats['CIRCLE']+=1;return
                if not closed and span<6.2:
                    a,b=np.degrees([angles[0],angles[-1]])%360
                    if angles[-1]<angles[0]:a,b=b,a
                    msp.add_arc(xy(center),rad*(SX+SY)/2,float(a),float(b),dxfattribs=attrs);stats['ARC']+=1;return
    simplified=cv2.approxPolyDP(p.astype(np.float32).reshape(-1,1,2),.55,closed).reshape(-1,2)
    if closed and np.linalg.norm(simplified[0]-simplified[-1])<.1:simplified=simplified[:-1]
    if len(simplified)>=2:
        msp.add_lwpolyline([xy(v) for v in simplified],close=closed,dxfattribs=attrs)
        stats['LWPOLYLINE']+=1

def follow(start,nxt):
    path=[start,nxt]
    visited.add(edge_key(start,nxt))
    previous,current=start,nxt
    while degree[current]==2:
        onward=[v for v in neighbours(current) if v!=previous]
        if not onward:break
        following=onward[0]
        key=edge_key(current,following)
        if key in visited:break
        visited.add(key);path.append(following)
        previous,current=current,following
        if current==start:break
    emit(path)

for start,d in degree.items():
    if d==2:continue
    for nxt in neighbours(start):
        if edge_key(start,nxt) not in visited:follow(start,nxt)
for start in degree:
    for nxt in neighbours(start):
        if edge_key(start,nxt) not in visited:follow(start,nxt)
print('Traced entities:',dict(stats),flush=True)
del visited,degree,coords,skel,S

psp=doc.layouts.get('Layout1')
psp.page_setup(size=(420,297),margins=(0,0,0,0),units='mm')
psp.add_viewport(center=(210,148.5),size=(420,297),view_center_point=(210,148.5),view_height=297,status=2)
doc.set_modelspace_vport(height=310,center=(210,148.5))
doc.saveas(OUT)
check=ezdxf.readfile(OUT)
audit=check.audit()
assert not audit.errors and not audit.fixes
assert len(check.modelspace().query('DIMENSION'))==21
assert not len(check.modelspace().query('IMAGE'))
assert len(check.modelspace().query('TEXT'))==len(texts)
assert abs(dimensions[-1]['raster_measurement_mm_or_deg']-60)<1e-7
for d in check.modelspace().query('DIMENSION'):
    assert len(check.blocks[d.dxf.geometry].query('TEXT'))==1
print('Rendering validation preview...',flush=True)
backend=PyMuPdfBackend()
cfg=config.Configuration(background_policy=config.BackgroundPolicy.WHITE,color_policy=config.ColorPolicy.BLACK,lineweight_scaling=1)
Frontend(RenderContext(check),backend,config=cfg).draw_layout(check.modelspace(),finalize=True)
preview=backend.get_pixmap_bytes(layout.Page(420,297,layout.Units.mm),fmt='png',dpi=210)
(WORK/'valve-preview.png').write_bytes(preview)
report={
 'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'output':OUT.name,'output_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
 'source_pixels':[W,H],'source_only':True,'ground_truth_dxf_read':False,
 'units':'mm','dxf_version':check.dxfversion,'sheet':'A3',
 'calibration':{'x_mm_per_pixel':SX,'y_mm_per_pixel':SY,'x_nominal_mm':46,'y_nominal_mm':160,
                'x_anchors_px':[front,back],'y_anchors_px':[top,bottom]},
 'dimensions':dimensions,'texts':text_report,
 'modelspace_entities':dict(Counter(e.dxftype() for e in check.modelspace())),
 'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),
 'limitations':[
  'Assisted reconstruction: annotations and regions were selected and transcribed visually.',
  'Geometry is raster-derived, with residual measurement errors; source nominal labels are preserved explicitly.',
  'Detail A remains drawn at 5:1; its dimension uses DIMLFAC=0.2.',
  'Radius leaders, fine tolerances, surface symbols and small margin text remain vector strokes.',
  'Dimensions are not associative. Nominal text overrides preserve source labels; regeneration may alter appearance.',
  'Section hatching and contours meeting at intersections are separate editable vector segments, not a parametric solid.',
  'Arial replaces the principal original lettering; blank source fields remain blank.',
  'Validated with ezdxf and rendering; not opened in a CAD application.',
 ],
}
(WORK/'valve-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('output','modelspace_entities','audit_errors','audit_fixes')},ensure_ascii=False,indent=2))

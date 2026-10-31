"""Package the generated architecture artwork as a one-slide PPTX and PDF.

The PNG is preserved unchanged. The PowerPoint slide contains one picture,
not independently editable text or shapes. Uses only stdlib/Pillow for PPTX.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import sys
import shutil
import zipfile
import xml.etree.ElementTree as ET
import json
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs'/'presentation'
DEST.mkdir(parents=True,exist_ok=True)
SOURCE=Path(sys.argv[1])
BASE='architecture-onepager'
if any((DEST/(BASE+suffix)).exists() for suffix in ('.png','.pptx','.pdf')):
    raise FileExistsError('Presentation outputs already exist; choose a new version instead.')
PNG=DEST/(BASE+'.png')
PPTX=DEST/(BASE+'.pptx')
PDF=DEST/(BASE+'.pdf')
shutil.copyfile(SOURCE,PNG)
with Image.open(PNG) as im:
    width,height=im.size
    im.verify()

S='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR='http://schemas.openxmlformats.org/package/2006/relationships'
DECL='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
CX,CY=12192000,6858000
ratio=min(CX/width,CY/height)
PICX,PICY=round(width*ratio),round(height*ratio)
X,Y=(CX-PICX)//2,(CY-PICY)//2
GROUP='''<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'''
COLOR_MAP='accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"'

def rels(items):
    return DECL+f'<Relationships xmlns="{PR}">'+''.join(
        f'<Relationship Id="{rid}" Type="{kind}" Target="{escape(target)}"/>'
        for rid,kind,target in items)+'</Relationships>'

def doc(root,body):
    return DECL+f'<p:{root} xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{S}">'+body+f'</p:{root}>'

description='Предлагаемая архитектура сервиса PDF/скан → DXF. Подготовка; OpenCV и PaddleOCR; DrawingGraph, масштаб и ограничения; локальная Qwen-VL как вспомогательный модуль; экспорт DXF AC1021 и проверка. Локальное развёртывание: FastAPI, Python worker, PostgreSQL, Qwen. Иллюстрированный слайд, текст является частью изображения.'
files={}
overrides={
 '/ppt/presentation.xml':'application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml',
 '/ppt/slides/slide1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slide+xml',
 '/ppt/slideMasters/slideMaster1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml',
 '/ppt/slideLayouts/slideLayout1.xml':'application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml',
 '/ppt/theme/theme1.xml':'application/vnd.openxmlformats-officedocument.theme+xml',
 '/docProps/core.xml':'application/vnd.openxmlformats-package.core-properties+xml',
 '/docProps/app.xml':'application/vnd.openxmlformats-officedocument.extended-properties+xml',
}
files['[Content_Types].xml']=DECL+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'+''.join(f'<Override PartName="{part}" ContentType="{ctype}"/>' for part,ctype in overrides.items())+'</Types>'
files['_rels/.rels']=rels([
 ('rId1',R+'/officeDocument','ppt/presentation.xml'),
 ('rId2','http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties','docProps/core.xml'),
 ('rId3',R+'/extended-properties','docProps/app.xml'),
])
files['docProps/core.xml']=DECL+f'''<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Архитектура сервиса PDF / скан → DXF</dc:title><dc:subject>Одностраничная презентация предлагаемой архитектуры</dc:subject><dc:creator>Проектный практикум / 44labs</dc:creator><dc:description>{escape(description)}</dc:description><cp:keywords>44labs, CAD, DXF, OpenCV, PaddleOCR, Qwen, architecture</cp:keywords></cp:coreProperties>'''
files['docProps/app.xml']=DECL+'''<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Architecture slide packager</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>1</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><ScaleCrop>false</ScaleCrop><HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Slide Titles</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs><TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>Архитектура сервиса</vt:lpstr></vt:vector></TitlesOfParts><Company>Совместный проект с 44labs</Company></Properties>'''
files['ppt/presentation.xml']=doc('presentation',f'<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz cx="{CX}" cy="{CY}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/>')
files['ppt/_rels/presentation.xml.rels']=rels([
 ('rId1',R+'/slideMaster','slideMasters/slideMaster1.xml'),
 ('rId2',R+'/slide','slides/slide1.xml'),
])
files['ppt/slideMasters/slideMaster1.xml']=doc('sldMaster','<p:cSld name="Architecture"><p:bg><p:bgPr><a:solidFill><a:srgbClr val="081626"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>'+GROUP+'</p:spTree></p:cSld>'+f'<p:clrMap {COLOR_MAP}/>'+'<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>')
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rels([
 ('rId1',R+'/slideLayout','../slideLayouts/slideLayout1.xml'),
 ('rId2',R+'/theme','../theme/theme1.xml'),
])
files['ppt/slideLayouts/slideLayout1.xml']=DECL+f'<p:sldLayout xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{S}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'+GROUP+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rels([('rId1',R+'/slideMaster','../slideMasters/slideMaster1.xml')])
picture=f'''<p:pic><p:nvPicPr><p:cNvPr id="2" name="Архитектура сервиса" descr="{escape(description)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{X}" y="{Y}"/><a:ext cx="{PICX}" cy="{PICY}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'''
files['ppt/slides/slide1.xml']=doc('sld','<p:cSld name="Архитектура сервиса"><p:spTree>'+GROUP+picture+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>')
files['ppt/slides/_rels/slide1.xml.rels']=rels([
 ('rId1',R+'/slideLayout','../slideLayouts/slideLayout1.xml'),
 ('rId2',R+'/image','../media/image1.png'),
])
colors={'dk1':'081626','lt1':'FFFFFF','dk2':'132C48','lt2':'E3F6FF','accent1':'73F1FF','accent2':'56D8E9','accent3':'60D6C1','accent4':'F5B957','accent5':'4283BA','accent6':'A6BCE5','hlink':'73F1FF','folHlink':'A6BCE5'}
color_scheme='<a:clrScheme name="Cyan blueprint">'+''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in colors.items())+'</a:clrScheme>'
font_scheme='<a:fontScheme name="Architecture"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
fill='<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
lines=''.join(f'<a:ln w="{w}" cap="flat" cmpd="sng" algn="ctr">{fill}<a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>' for w in (9525,25400,38100))
formats='<a:fmtScheme name="Architecture"><a:fillStyleLst>'+fill*3+'</a:fillStyleLst><a:lnStyleLst>'+lines+'</a:lnStyleLst><a:effectStyleLst>'+('<a:effectStyle><a:effectLst/></a:effectStyle>'*3)+'</a:effectStyleLst><a:bgFillStyleLst>'+fill*3+'</a:bgFillStyleLst></a:fmtScheme>'
files['ppt/theme/theme1.xml']=DECL+f'<a:theme xmlns:a="{A}" name="Architecture"><a:themeElements>'+color_scheme+font_scheme+formats+'</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'

for name,content in files.items():
    ET.fromstring(content)
with zipfile.ZipFile(PPTX,'w',zipfile.ZIP_DEFLATED) as z:
    for name,content in files.items():
        z.writestr(name,content.encode('utf-8'))
    z.write(PNG,'ppt/media/image1.png')
with zipfile.ZipFile(PPTX) as z:
    assert z.testzip() is None
    assert z.read('ppt/media/image1.png')==PNG.read_bytes()
    assert len(ET.fromstring(z.read('ppt/presentation.xml')).findall(f'{{{S}}}sldIdLst/{{{S}}}sldId'))==1
    # Resolve every internal package relationship to an existing ZIP member.
    import posixpath
    for name in z.namelist():
        if not name.endswith('.rels'): continue
        parent='' if name=='_rels/.rels' else posixpath.dirname(posixpath.dirname(name))
        for r in ET.fromstring(z.read(name)):
            resolved=posixpath.normpath(posixpath.join(parent,r.attrib['Target']))
            assert resolved in z.namelist(), (name,resolved)

# PDF is a document wrapper around the original image, not an image edit.
deps=Path.home()/'AppData/Local/Temp/44labs-bolt-artifacts-bad6e25064e94550a32bab05af03086d/pdf-dxf-deps'
sys.path.append(str(deps))
import pymupdf
pdf=pymupdf.open()
page=pdf.new_page(width=960,height=540)
page.draw_rect(page.rect,color=None,fill=(8/255,22/255,38/255))
page.insert_image(page.rect,filename=str(PNG),keep_proportion=True)
pdf.set_metadata({'title':'Архитектура сервиса PDF / скан → DXF','author':'Совместный проект с 44labs','subject':'Предложение для MVP'})
pdf.save(PDF,deflate=True)
pdf.close()
with pymupdf.open(PDF) as reopened:
    assert len(reopened)==1 and len(reopened[0].get_images())==1

metadata={'generator':'built-in image_gen','reference':'../../Example.png',
          'prompt':'architecture-slide.prompt.txt','image_px':[width,height],
          'slides':1,'aspect_ratio':'16:9','pptx_content':'single embedded raster image',
          'files':[p.name for p in (PNG,PPTX,PDF)],
          'validation':['PNG decoded','PPTX XML and relationships checked','one slide','embedded image unchanged','PDF reopened, one page'],
          'note':'PowerPoint application was not launched; package structure was validated.'}
(DEST/'architecture-onepager.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=False,indent=2))

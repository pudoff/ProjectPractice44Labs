"""Real-model single-dimension smoke. Synthetic control, not a quality benchmark."""
import argparse
import json
from pathlib import Path
import pymupdf
from app.infra.model_assets import configure_local_models
from app.services.recognition import recognize
from app.services.dxf import DxfExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_local_models()
    k = 72/25.4
    pdf = pymupdf.open()
    page = pdf.new_page(width=210*k, height=297*k)
    def line(a, b, width):
        page.draw_line([v*k for v in a], [v*k for v in b], width=width*k)
    # Standard A4 frame supplies the CV module's physical sheet calibration.
    page.draw_rect(pymupdf.Rect(20*k, 5*k, 205*k, 292*k), width=.6*k)
    for a, b in [((60, 110), (110, 110)), ((110, 110), (110, 130)), ((110, 130), (60, 130)), ((60, 130), (60, 110))]:
        line(a, b, .6)
    line((60, 95), (110, 95), .2)
    line((60, 93), (60, 110), .2)
    line((110, 93), (110, 110), .2)
    page.insert_text((81*k, 93*k), '50', fontsize=5*k)
    page.insert_text((180*k, 270*k), '1:1', fontsize=4*k)
    source = args.out/'linear-control.png'
    page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False).save(source)
    pdf.close()
    run = args.out/'run'
    run.mkdir(exist_ok=True)
    import hashlib
    graph = recognize(source, run, hashlib.sha256(source.read_bytes()).hexdigest())
    issues = DxfExporter().export(graph, run/'result.dxf')
    import ezdxf
    dxf_dimensions = list(ezdxf.readfile(run/'result.dxf').modelspace().query('DIMENSION')) if (run/'result.dxf').is_file() else []
    (run/'drawing.json').write_text(graph.model_dump_json(indent=2), encoding='utf-8')
    report = {'texts': len(graph.texts), 'dimensions': len(graph.dimensions), 'dxf_dimensions': len(dxf_dimensions),
              'dxf_measurements': [float(d.get_measurement()) for d in dxf_dimensions],
              'issues': [i.model_dump() for i in issues], 'cad_gui_checked': False,
              'scope': 'Synthetic control; actual PaddleOCR, drawcv and DXF exporter; no injected observations/points.'}
    (args.out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if not dxf_dimensions:
        raise SystemExit('No automatically linked dimension on the control image')


if __name__ == '__main__':
    main()

import argparse
import hashlib
import json
import math
from pathlib import Path
import pymupdf
import ezdxf
from app.infra.model_assets import configure_local_models
from app.services.recognition import recognize
from app.services.dxf import DxfExporter, DXF_SPECIAL_CHARACTERS
from app.tasks.job import _export


def control(out):
    k = 72/25.4
    pdf = pymupdf.open()
    page = pdf.new_page(width=210*k, height=297*k)
    def line(a, b, width):
        page.draw_line([v*k for v in a], [v*k for v in b], width=width*k)
    page.draw_rect(pymupdf.Rect(20*k, 5*k, 205*k, 292*k), width=.6*k)
    for a, b in [((60, 110), (110, 110)), ((110, 110), (110, 130)), ((110, 130), (60, 130)), ((60, 130), (60, 110))]:
        line(a, b, .6)
    line((60, 95), (110, 95), .2)
    line((60, 93), (60, 110), .2)
    line((110, 93), (110, 110), .2)
    page.insert_text((81*k, 93*k), '50', fontsize=5*k)
    page.insert_text((180*k, 270*k), '1:1', fontsize=4*k)
    source = out/'linear-control.png'
    page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False).save(source)
    pdf.close()
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Agreed single-page PDF or raster')
    parser.add_argument('--expected-text')
    parser.add_argument('--expected-mm', type=float)
    parser.add_argument('--tolerance-mm', type=float)
    args = parser.parse_args()
    if args.source and (not args.expected_text or args.expected_mm is None or args.tolerance_mm is None):
        parser.error('--source requires --expected-text, --expected-mm and --tolerance-mm')
    if args.source and (not math.isfinite(args.expected_mm) or args.expected_mm <= 0 or
                        not math.isfinite(args.tolerance_mm) or args.tolerance_mm < 0):
        parser.error('Expected measurement must be positive and tolerance nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    configure_local_models()
    source = args.source or control(args.out)
    if source.suffix.lower() == '.pdf':
        with pymupdf.open(source) as pdf:
            if len(pdf) != 1:
                parser.error('Extract the agreed PDF page into a single-page input first')
    run = args.out/'run'
    run.mkdir(exist_ok=True)
    if any(run.iterdir()):
        parser.error('Use a fresh output directory')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    graph = recognize(source, run, source_hash)
    state = _export(graph, run, source_hash)
    (run/'drawing.json').write_text(graph.model_dump_json(indent=2), encoding='utf-8')
    msp = ezdxf.readfile(run/'result.dxf').modelspace() if (run/'result.dxf').is_file() else None
    dimensions = list(msp.query('DIMENSION')) if msp is not None else []
    texts = [entity.dxf.text for entity in msp.query('TEXT')] if msp is not None else []
    expected = args.expected_mm if args.source else 50
    tolerance = args.tolerance_mm if args.source else .5
    checks = {
        'text_entity': (args.expected_text.translate(DXF_SPECIAL_CHARACTERS) in texts) if args.source else bool(texts),
        'dimension_measurement': any(abs(float(d.get_measurement()) - expected) <= tolerance for d in dimensions),
        'adapter_versions_parameters': all(graph.provenance.get(name, {}).get('module', {}).get('version') and
                                            graph.provenance.get(name, {}).get('params') for name in ('cv', 'ocr')),
    }
    unknown = graph.model_copy(deep=True)
    for dim in unknown.dimensions:
        dim.nominal_mm = None
        dim.recognized_text = '?'
    unknown_issues = DxfExporter().export(unknown, run/'unknown-value.dxf')
    checks['unknown_requires_review'] = any(i.code in ('unknown_dimension_value', 'unresolved_dimension_link') for i in unknown_issues)
    edited = graph.model_copy(deep=True)
    for primitive in edited.primitives:
        if primitive.kind == 'line':
            primitive.start = tuple(v * 1.1 for v in primitive.start)
            primitive.end = tuple(v * 1.1 for v in primitive.end)
        elif primitive.kind == 'polyline':
            primitive.vertices = [tuple(v * 1.1 for v in xy) for xy in primitive.vertices]
        else:
            primitive.center = tuple(v * 1.1 for v in primitive.center)
            primitive.radius *= 1.1
    edit_issues = DxfExporter().export(edited, run/'edited.dxf')
    edited_dimensions = list(ezdxf.readfile(run/'edited.dxf').modelspace().query('DIMENSION')) if (run/'edited.dxf').exists() else []
    baseline_dims = {d.id: d for d in graph.dimensions}
    from app.services.geometry import GeometryResolver
    resolver = GeometryResolver(edited)
    baseline_resolver = GeometryResolver(graph)
    def changed(dim):
        resolved = resolver.dimension(dim)
        original = baseline_resolver.dimension(baseline_dims[dim.id])
        angle = math.radians(dim.angle)
        def measurement(d):
            return abs((d.p2[0]-d.p1[0])*math.cos(angle) + (d.p2[1]-d.p1[1])*math.sin(angle))
        return (abs(measurement(resolved) - measurement(original)) > 1e-4 and
                any(abs(float(d.get_measurement()) - measurement(resolved)) < 1e-4 for d in edited_dimensions))
    checks['geometry_edit_recomputed_or_review'] = bool(edited.dimensions) and all(
        any(i.object_id == dim.id and i.code == 'dimension_recalculation_required' for i in edit_issues)
        or (resolver.dimension_has_anchors(dim) and changed(dim)) for dim in edited.dimensions)
    report = {
        'scope': 'specified sheet' if args.source else 'synthetic control; not sheet acceptance',
        'source': str(source.resolve()), 'source_sha256': source_hash,
        'expected_text': args.expected_text, 'expected_mm': expected, 'tolerance_mm': tolerance,
        'state': state, 'texts': texts, 'dxf_measurements': [float(d.get_measurement()) for d in dimensions],
        'checks': checks, 'provenance': graph.provenance, 'cad_gui_checked': False,
        'perturbation_issues': {'unknown': [i.model_dump() for i in unknown_issues],
                                'geometry_edit': [i.model_dump() for i in edit_issues]},
    }
    (args.out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'scope': report['scope'], 'checks': checks}, ensure_ascii=False, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit('Acceptance checks failed; see summary.json')


if __name__ == '__main__':
    main()

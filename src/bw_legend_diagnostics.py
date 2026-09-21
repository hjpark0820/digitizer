"""Read-only visualizations of the exact B&W legend models used by detection."""
from __future__ import annotations

import cv2
import numpy as np


GREEN = (60, 145, 0)
CYAN = (190, 135, 0)


def _text(image, text, xy, size=.48, colour=(40, 40, 40)):
    cv2.putText(image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)


def _paste(canvas, raster, box):
    """Nearest-neighbour display only; preserve aspect ratio and input pixels."""
    x, y, width, height = box
    h, w = raster.shape[:2]
    factor = min(width / w, height / h)
    resized = cv2.resize(raster, (max(1, round(w * factor)), max(1, round(h * factor))),
                         interpolation=cv2.INTER_NEAREST)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    rh, rw = resized.shape[:2]
    left, top = x + (width - rw) // 2, y + (height - rh) // 2
    canvas[top:top + rh, left:left + rw] = resized


def legend_diagnostic_steps(image, legend_area, templates, reports):
    """Render current models, never re-extract, classify or alter detector state.

    Coordinates are in the detector raster (the prepared native image in the
    web GUI). Green outlines show source swatch boxes; cyan shows marker extent.
    The soft template preserves measured ink/antialiasing, not an ideal symbol.
    """
    if not templates:
        return []
    boxes = [row['box'] for row in reports]
    if legend_area is None:
        legend_area = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes))
    x0, y0, x1, y1 = map(int, legend_area)
    # Padding keeps edge IDs visible even when the selected ROI is very tight.
    pad = 22
    source = cv2.copyMakeBorder(image[y0:y1, x0:x1], pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    overlay = image.copy()
    marks = source.copy()
    for row in reports:
        a, b, c, d = map(int, row['box'])
        colour = (20, 30, 210) if row.get('excluded_by_class_filter') or row.get('extraction_status')=='failed' else GREEN
        start = (a - x0 + pad, b - y0 + pad)
        stop = (c - x0 + pad - 1, d - y0 + pad - 1)
        cv2.rectangle(marks, start, stop, colour, 1)
        cv2.rectangle(overlay, (a, b), (c - 1, d - 1), colour, 1)
        _text(marks, row['swatch_id'], (start[0], max(12, start[1] - 4)), .35, colour)
        _text(overlay, row['swatch_id'], (a, max(12, b - 4)), .35, colour)
        if row.get('extraction_status')!='failed':
            ga, gb, gc, gd = map(int, row.get('glyph_box', row['box']))
            cv2.rectangle(marks, (ga - x0 + pad, gb - y0 + pad),
                          (gc - x0 + pad - 1, gd - y0 + pad - 1), CYAN, 1)
    source = cv2.addWeighted(marks, .8, source, .2, 0)
    # Thin, translucent annotations never replace the source used by detection.
    overlay = cv2.addWeighted(overlay, .8, image, .2, 0)
    width = 1120
    top_height = min(360, max(180, round(source.shape[0] * min(4., 1080 / source.shape[1]))))
    header = 116 + top_height
    row_height = 190
    by_template={t.swatch_id:t for t in templates}
    displayed=[r for r in reports if r['swatch_id'] in by_template or r.get('extraction_status')=='failed']
    panel = np.full((header + row_height * len(displayed) + 18, width, 3), 255, np.uint8)
    _text(panel, 'v46 B&W legend: extracted source pixels and actual matching templates', (20, 28), .63)
    _text(panel, 'Green: swatch box | cyan: marker extent | red: unresolved or class-filtered entry', (20, 53))
    modelled=any(t.model_completed for t in templates)
    _text(panel, ('Source preserved; fitted symbol models shown separately. Fill identity uses original grayscale.' if modelled else
                 'Display enlargement only; no ideal glyph replacement. Unknown shapes remain separate series.'), (20, 76))
    _paste(panel, source, (20, 88, 1080, top_height))
    for x, title in ((20, 'Source swatch (with line)'), (268, 'Actual soft-ink template'),
                     (516, 'Binary shape mask'), (764, 'Identity / shape evidence')):
        _text(panel, title, (x, header + 4), .46)
    for i, row in enumerate(displayed):
        template=by_template.get(row['swatch_id'])
        top = header + 18 + i * row_height
        a, b, c, d = map(int, row['box'])
        _paste(panel, image[b:d, a:c], (20, top, 224, 144))
        if template is None:
            _text(panel, 'NO USABLE TEMPLATE', (280, top+65), .60, (20,30,210))
            for j,line in enumerate([f'{row["swatch_id"]}: UNRESOLVED',
                'No points inferred for this entry',row.get('extraction_error_code','extraction_failed'),
                'Series ID and source box retained','Inspect swatch / expand legend ROI']):
                _text(panel,line,(764,top+15+j*23),.44,(20,30,210))
            cv2.line(panel,(20,top+162),(1100,top+162),(225,225,225),1)
            continue
        soft = np.uint8(np.rint(255 * (1 - np.clip(template.soft, 0, 1))))
        binary = np.where(template.mask, 0, 255).astype(np.uint8)
        _paste(panel, soft, (268, top, 224, 144))
        _paste(panel, binary, (516, top, 224, 144))
        evidence = row.get('shape_evidence') or {}
        lines = [f'{template.swatch_id}: {template.name}',
                 f'Source diameter: {template.diameter:.1f} px',
                 f"Outer shape: {evidence.get('best_shape', 'n/a')}",
                 f"Status: {evidence.get('status', row.get('classification', 'n/a'))}",
                 f"Fill style: {(row.get('fill_evidence') or {}).get('style', 'uncertain')}",
                 f"Pattern evidence: {evidence.get('patterned_internal_evidence', False)}",
                 'Geometry model; source fill retained' if template.model_completed else 'Line-crossing pixels retained']
        for j, line in enumerate(lines):
            _text(panel, line, (764, top + 15 + j * 21), .44)
        cv2.line(panel, (20, top + 162), (1100, top + 162), (225, 225, 225), 1)
    return [
        {'title': 'v46 legend source and actual templates', 'img_bgr': panel,
         'output_filename': 'v46_legend_diagnostic.png'},
        {'title': 'v46 legend swatch locations on source', 'img_bgr': overlay,
         'output_filename': 'v46_legend_overlay.png'},
    ]

"""Conservative, source-only repeated tick-label mask for line-only pilots.

No pixels are erased. This auxiliary mask disqualifies aligned compact neutral
glyph families as curve/occlusion evidence. It is not a general OCR system.
"""
from collections import defaultdict
import cv2
import numpy as np


def repeated_tick_text(image_bgr, plot_box, valid=None):
    x0, y0, x1, y1 = map(int, plot_box)
    crop = image_bgr[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    valid = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (h, w):
        raise ValueError('valid must be plot-local')
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    neutral = (np.ptp(crop.astype(float), axis=2) <= 30) & (gray < 215) & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8), 8)
    candidates = []
    for k in range(1, n):
        x, y, cw, ch, area = map(int, stats[k])
        if not (6 <= ch <= max(6, .08*h) and .2 <= cw/ch <= 3
                and 8 <= area <= .004*w*h and cw <= .08*w):
            continue
        binary = (labels[y:y+ch, x:x+cw] == k).astype(np.uint8)
        _, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0 if hierarchy is None else int((hierarchy[0, :, 3] >= 0).sum())
        candidates.append(dict(id=k, box=[x, y, cw, ch], bottom=y+ch,
                               height=ch, holes=holes))
    groups = []
    for c in sorted(candidates, key=lambda q:q['bottom']):
        group = next((g for g in groups if abs(c['bottom']-np.median([q['bottom'] for q in g]))
                      <= max(2, .15*np.median([q['height'] for q in g]))), None)
        if group is None:
            groups.append([c])
        else:
            group.append(c)
    selected, audits = set(), []
    for group in groups:
        median_h = float(np.median([q['height'] for q in group]))
        group = [q for q in group if .65*median_h <= q['height'] <= 1.5*median_h]
        if len(group) < 5 or sum(q['holes'] for q in group) < 2:
            continue
        left = min(q['box'][0] for q in group)
        right = max(q['box'][0]+q['box'][2] for q in group)
        if right-left < .5*w:
            continue
        selected.update(q['id'] for q in group)
        # Include tiny decimal dots adjacent to the accepted glyphs, not a row.
        bottom = float(np.median([q['bottom'] for q in group]))
        for k in range(1, n):
            x, y, cw, ch, area = map(int, stats[k])
            if (2 <= area <= median_h and ch <= .4*median_h and cw <= .5*median_h
                    and abs(y+ch-bottom) <= .3*median_h
                    and any(abs(x-(q['box'][0]+q['box'][2])) <= .8*median_h
                            or abs(x+cw-q['box'][0]) <= .8*median_h for q in group)):
                selected.add(k)
        audits.append(dict(baseline_source_y=bottom+y0, height=median_h,
                           glyph_count=len(group), hole_count=sum(q['holes'] for q in group),
                           source_x_span=[left+x0, right+x0]))
    core = np.isin(labels, list(selected)) if selected else np.zeros((h, w), bool)
    halo = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # Retain only neutral ink or its tiny immediate raster fringe.
    mask = halo & valid & (np.ptp(crop.astype(float), axis=2) <= 30)
    return mask, dict(method='repeated_compact_neutral_glyph_baselines',
                      families=audits, selected_components=len(selected), pixels=int(mask.sum()),
                      no_ocr=True, source_pixels_deleted=False,
                      limitation='Line-only pilot helper; sparse/unusual labels can remain undetected.')

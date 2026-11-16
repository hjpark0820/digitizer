"""Source-pixel legend templates from explicitly reviewed GUI entries.

The web payload uses inclusive boxes; detector rectangles are half-open. A
reviewed list replaces automatic entry discovery. No missing entry, shape label,
or reference marker image is invented by this adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_review(path, image, *, image_path=None, legend_box=None):
    """Validate a saved GUI review and return entries with half-open boxes."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('entries'), list):
        raise ValueError('Legend review must contain an entries list')
    entries = payload['entries']
    if not entries:
        raise ValueError('Add at least one legend entry before running detection')
    h, w = image.shape[:2]
    binding = payload.get('image') or {}
    if binding.get('width', w) != w or binding.get('height', h) != h:
        raise ValueError('Legend review belongs to a different image size')
    if image_path and binding.get('sha256'):
        if hashlib.sha256(Path(image_path).read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('Legend review belongs to a different source image')
    if legend_box is not None and payload.get('legend_box') is not None:
        if list(map(int, legend_box)) != payload['legend_box']:
            raise ValueError('Legend review belongs to a different legend selection')
    result, ids = [], set()
    for index, value in enumerate(entries):
        if not isinstance(value, dict):
            raise ValueError('Each reviewed legend entry must be an object')
        sid = str(value.get('id') or f'review-{index + 1}')
        if sid in ids:
            raise ValueError('Legend review entry IDs must be unique')
        ids.add(sid)
        box = value.get('box')
        if not isinstance(box, (tuple, list)) or len(box) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or
                not np.isfinite(v) or v != int(v) for v in box):
            raise ValueError('Reviewed swatch boxes require four finite integer pixels')
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 < w and 0 <= y0 < y1 < h):
            raise ValueError('Reviewed swatch box must be nonempty and inside the image')
        center = value.get('center', [(x0 + x1) / 2, (y0 + y1) / 2])
        if not isinstance(center, (tuple, list)) or len(center) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                for v in center):
            raise ValueError('Reviewed swatch center must contain two finite pixels')
        if not (x0 <= center[0] <= x1 and y0 <= center[1] <= y1):
            raise ValueError('Reviewed swatch center must be inside its crop')
        result.append(dict(value, id=sid, box=[x0, y0, x1 + 1, y1 + 1],
                           center=list(map(float, center)), input_source='user_reviewed_legend'))
    return result


def bw_swatches(entries):
    """Supply locations, never shape labels, to the established BW extractor."""
    return [(None, tuple(entry['box'])) for entry in entries]


def annotate_bw_reports(result, entries):
    """Keep GUI identities alongside detector IDs in exported diagnostics."""
    diagnostics = result.get('grid_diagnostics')
    if not isinstance(diagnostics, dict):
        return
    diagnostics['legend_input_source'] = 'user_reviewed_legend'
    diagnostics['reviewed_entries'] = entries
    for report in diagnostics.get('swatches', []):
        index = report.get('series_index')
        if isinstance(index, int) and 0 <= index < len(entries):
            report.update(review_id=entries[index]['id'],
                          input_source='user_reviewed_legend',
                          reviewed_box=entries[index]['box'])


def _color_candidate(image, reviewed):
    """Measure the observed marker near a reviewed center inside its crop."""
    import color_legend_hybrid_v46 as hybrid
    x0, y0, x1, y1 = reviewed['box']
    patch = image[y0:y1, x0:x1]
    brightness = patch.mean(axis=2)
    bright = brightness >= np.percentile(brightness, 85)
    paper = tuple(int(v) for v in np.median(patch[bright], axis=0))
    # The review locates the key. Existing pixel contrast and template geometry
    # still determine its colour, marker body, line nuisance and shape.
    contrast = np.linalg.norm(patch.astype(float) - np.asarray(paper), axis=2)
    support = contrast >= max(8., .20 * float(np.percentile(contrast, 99)))
    cx, cy = reviewed['center'][0] - x0, reviewed['center'][1] - y0
    candidates = []
    for group, box in hybrid._groups(support):
        try:
            model = hybrid._model(patch[group], paper)
            entry = hybrid._candidate(patch, group, box, paper, model)
        except ValueError:
            continue
        ex, ey = entry['center']
        distance = float(np.hypot(ex - cx, ey - cy))
        # A click on white paper between labels must not silently pick a remote
        # label component. The preview/crop can be corrected by the investigator.
        if distance > max(4., .75 * entry['diameter']):
            continue
        candidates.append((distance, entry))
    if not candidates:
        raise ValueError('No usable marker body near the reviewed center; adjust the crop or center')
    return patch, min(candidates, key=lambda item: item[0])[1], (x0, y0)


def reviewed_color_seeds(image, entries):
    """Initial palette from reviewed pixels only, before template preparation."""
    colors = []
    for index, reviewed in enumerate(entries):
        try:
            _, entry, _ = _color_candidate(image, reviewed)
            rgb = list(reversed(entry['model'].core_bgr))
        except ValueError:
            # Invalid entries are reported by extract_color_models. This seed
            # is bookkeeping only; no detection uses it as marker evidence.
            rgb = [0, 0, 0]
        colors.append(dict(name=f'color{index + 2:02d}', mean_rgb=rgb,
                           swatch_rgb=rgb, px_count=0))
    return colors


def extract_color_models(image, entries):
    """Build only the reviewed entries using the standard colour body model."""
    import color_legend_hybrid_v46 as hybrid
    models, reports, rejected = [], [], []
    for index, reviewed in enumerate(entries):
        try:
            patch, entry, origin = _color_candidate(image, reviewed)
            entry.update(native_index=index, discovery_source='user_reviewed_legend')
            model, report = hybrid._crop_template(patch, entry, origin, index + 1, [])
            report.update(review_id=reviewed['id'], reviewed_box=reviewed['box'],
                          input_source='user_reviewed_legend', extraction_status='usable')
            models.append(model)
            reports.append(report)
        except ValueError as error:
            record = dict(swatch_id=f'S{index + 1:02d}', native_index=index,
                          review_id=reviewed['id'], reviewed_box=reviewed['box'],
                          input_source='user_reviewed_legend', extraction_status='failed',
                          status='invalid_reviewed_marker', reason=str(error))
            reports.append(record)
            rejected.append(record)
    return models, reports, rejected

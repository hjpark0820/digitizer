"""Read and validate source-linked legend review without running a detector.

The browser API uses inclusive working-image boxes. Detector diagnostics use
half-open boxes; conversion happens only while reading those diagnostics.
Suggested crops are editable source-image proposals, not verified marker models.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path

import cv2
import numpy as np


SCHEMA = 'digitizer-legend-review-v1'
MAX_ENTRIES = 200


def _image(path):
    path = Path(path)
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('The saved working image could not be read.')
    height, width = image.shape[:2]
    return image, dict(width=width, height=height,
                       sha256=hashlib.sha256(raw).hexdigest(), sha256_kind='file_bytes')


def _numbers(values, count, label):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f'{label} must contain {count} numbers.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in values):
        raise ValueError(f'{label} must contain finite numbers.')
    return list(values)


def _box(box, image_info, label='Legend box'):
    box = _numbers(box, 4, label)
    if any(int(v) != v for v in box):
        raise ValueError(f'{label} must use whole pixel coordinates.')
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 <= x1 < image_info['width'] and
            0 <= y0 <= y1 < image_info['height']):
        raise ValueError(f'{label} must lie inside the current working image.')
    return [x0, y0, x1, y1]


def _within(inner, outer):
    return (outer[0] <= inner[0] <= inner[2] <= outer[2] and
            outer[1] <= inner[1] <= inner[3] <= outer[3])


def validate_review(payload, image_path, legend_box, *, allow_empty=False):
    """Validate a complete replacement entry list tied to one working image.

The encoded image hash prevents reviews from being reused after replacement or
resizing. An empty list is allowed for display/suggestion requests only; a rerun
must contain at least one entry, so deletion never silently invokes auto parsing.
    """
    if not isinstance(payload, dict):
        raise ValueError('Legend review must be a JSON object.')
    if payload.get('schema', SCHEMA) != SCHEMA:
        raise ValueError('This legend review format is not supported.')
    if payload.get('coordinate_convention', 'inclusive_xyxy') != 'inclusive_xyxy':
        raise ValueError('Legend review boxes must use inclusive image coordinates.')
    _, info = _image(image_path)
    if legend_box is None:
        raise ValueError('Select a legend region before reviewing its entries.')
    region = _box(legend_box, info)
    bound_image = payload.get('image') or {}
    if not isinstance(bound_image, dict):
        raise ValueError('Legend review image information is invalid.')
    supplied_hash = bound_image.get('sha256') or payload.get('image_sha256')
    if supplied_hash and supplied_hash != info['sha256']:
        raise ValueError('The legend review belongs to a different working image. Reload the review.')
    for key in ('width', 'height'):
        if key in bound_image and bound_image[key] != info[key]:
            raise ValueError('The working image dimensions changed. Reload the legend review.')
    if payload.get('legend_box') is not None and _box(payload['legend_box'], info) != region:
        raise ValueError('The selected legend region changed. Reload the legend review.')
    rows = payload.get('entries')
    if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
        raise ValueError(f'Legend review must contain a list of at most {MAX_ENTRIES} entries.')
    if not rows and not allow_empty:
        raise ValueError('Add at least one legend entry before running detection again.')
    entries, seen = [], set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Legend entry {index} is invalid.')
        entry_id = row.get('id')
        if not isinstance(entry_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', entry_id):
            raise ValueError(f'Legend entry {index} needs a valid identifier.')
        if entry_id in seen:
            raise ValueError('Legend entry identifiers must be unique.')
        seen.add(entry_id)
        box = _box(row.get('box'), info, f'Legend entry {index} box')
        if not _within(box, region):
            raise ValueError(f'Legend entry {index} must lie inside the selected legend region.')
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            raise ValueError(f'Legend entry {index} is too small; use a crop of at least 3 by 3 pixels.')
        center = _numbers(row.get('center', [(box[0]+box[2])/2, (box[1]+box[3])/2]),
                          2, f'Legend entry {index} center')
        if not (box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
            raise ValueError(f'Legend entry {index} center must lie inside its crop.')
        entry = dict(id=entry_id, box=box, center=center,
                     source='automatic' if row.get('source') == 'automatic' else 'manual',
                     status=str(row.get('status') or 'reviewed')[:100],
                     label=str(row.get('label') or entry_id)[:200])
        for key, maximum in (('reason', 2048), ('reason_code', 160)):
            if isinstance(row.get(key), str) and row[key]:
                entry[key] = row[key][:maximum]
        entries.append(entry)
    return dict(schema=SCHEMA, coordinate_convention='inclusive_xyxy', image=info,
                legend_box=region, entries=entries, warnings=[])


def _read_json(path, warnings):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError):
        warnings.append(f'The saved {path.name} diagnostic could not be read.')
        return None


def _diagnostic_rows(out_dir, mode, warnings):
    if mode == 'bw':
        diagnostic = _read_json(out_dir / 'v46_grid_diagnostics.json', warnings) or {}
        return diagnostic.get('swatches') or []
    hybrid = _read_json(out_dir / 'legend_hybrid_v46.json', warnings) or {}
    entries = hybrid.get('entries') or []
    reports = hybrid.get('hybrid_reports') or []
    if entries:
        def identity(row):
            return row.get('review_id') or row.get('swatch_id') or row.get('id')
        reports_by_id = {identity(row): row for row in reports
                         if isinstance(row, dict) and identity(row)}
        combined, seen = [], set()
        for row in entries:
            if not isinstance(row, dict):
                continue
            key = identity(row)
            # Keep the actual accepted crop, adding extraction status from its
            # report. A rejected reviewed entry has no accepted model/crop and
            # must be appended separately rather than disappearing from the UI.
            combined.append(dict(reports_by_id.get(key, {}), **row))
            if key:
                seen.add(key)
        for row in reports:
            if (isinstance(row, dict) and row.get('extraction_status') == 'failed'
                    and (not identity(row) or identity(row) not in seen)):
                combined.append(row)
                if identity(row):
                    seen.add(identity(row))
        return combined
    # A failed colour extraction may still have located individual source keys.
    if reports:
        return reports
    state = _read_json(out_dir / 'color_group_state_v46.json', warnings) or {}
    return [dict(row, id=name, box=row.get('legend_box'))
            for name, row in (state.get('series_models') or {}).items() if row.get('legend_box')]


def _entry_from_diagnostic(row, index, region, info, warnings):
    box = (row.get('box') or row.get('swatch_box') or row.get('marker_body_box')
           or row.get('reviewed_box'))
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        box = _numbers(box, 4, 'Saved swatch box')
        box = [int(box[0]), int(box[1]), int(box[2])-1, int(box[3])-1]
        clipped = [max(region[0], box[0]), max(region[1], box[1]),
                   min(region[2], box[2]), min(region[3], box[3])]
        if clipped[2]-clipped[0] < 2 or clipped[3]-clipped[1] < 2:
            return None
        _box(clipped, info)
    except ValueError:
        return None
    entry_id = str(row.get('review_id') or row.get('swatch_id') or row.get('id') or f'S{index:02}')
    entry_id = re.sub(r'[^A-Za-z0-9_-]', '_', entry_id)[:80] or f'S{index:02}'
    if clipped != box:
        warnings.append(f'{entry_id}: the displayed crop was limited to the selected legend region.')
    center = [(clipped[0]+clipped[2])/2, (clipped[1]+clipped[3])/2]
    for key in ('marker_center', 'source_center'):
        candidate = row.get(key)
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in candidate):
                if clipped[0] <= candidate[0] <= clipped[2] and clipped[1] <= candidate[1] <= clipped[3]:
                    center = list(candidate)
    status = row.get('extraction_status') or ('failed' if row.get('template_available') is False else 'extracted')
    entry = dict(id=entry_id, box=clipped, center=center, source='automatic', status=status,
                 label=str(row.get('label') or row.get('class_name') or row.get('shape_hint') or entry_id))
    if status == 'failed':
        reason = row.get('extraction_error') or row.get('reason') or row.get('error')
        code = row.get('extraction_error_code') or row.get('reason_code') or row.get('code')
        entry['reason'] = (str(reason)[:2048] if reason else
                           str(code).replace('_', ' ')[:2048] if code else
                           'A usable marker template could not be extracted from this crop.')
        if code:
            entry['reason_code'] = str(code)[:160]
    return entry


def load_review(out_dir, image_path, legend_box, mode):
    """Return the detected source crops, including failed or zero-entry jobs."""
    _, info = _image(image_path)
    warnings = []
    result = dict(schema=SCHEMA, coordinate_convention='inclusive_xyxy', image=info,
                  legend_box=None, entries=[], warnings=warnings)
    if legend_box is None:
        warnings.append('No legend region was selected for this run.')
        return result
    region = _box(legend_box, info)
    result['legend_box'] = region
    out_dir = Path(out_dir)
    rows = _diagnostic_rows(out_dir, mode, warnings)
    automatic, seen = [], set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        entry = _entry_from_diagnostic(row, index, region, info, warnings)
        if entry is None:
            continue
        if entry['id'] in seen:
            entry['id'] += f'_{index}'
        seen.add(entry['id'])
        automatic.append(entry)
    # Portable sessions retain this compact snapshot instead of the large
    # detector diagnostics. Import validates pixels before rebinding its hash.
    if not automatic:
        snapshot = _read_json(out_dir / 'legend_review_state.json', warnings)
        if snapshot is not None:
            try:
                automatic = validate_review(snapshot, image_path, region, allow_empty=True)['entries']
            except ValueError as error:
                warnings.append(str(error))
    supplied = _read_json(out_dir / 'legend_review_input.json', warnings)
    if supplied is not None:
        try:
            reviewed = validate_review(supplied, image_path, region, allow_empty=True)['entries']
            by_id = {entry['id']: entry for entry in automatic}
            for entry in reviewed:
                matching = by_id.get(entry['id'])
                entry['status'] = matching['status'] if matching else 'reviewed'
                for key in ('reason', 'reason_code'):
                    entry.pop(key, None)
                    if matching and key in matching:
                        entry[key] = matching[key]
            result['entries'] = reviewed
        except ValueError as error:
            warnings.append(str(error))
            result['entries'] = automatic
    else:
        result['entries'] = automatic
    if not result['entries']:
        warnings.append('No legend entries were saved. Click each missing marker center to add a source crop.')
    return result


def _clicked_source_box(image, region, x, y):
    """Measure the clicked source component, using strong ink for geometry.

    Pale antialias bridges must not join adjacent legend rows. Long strokes
    near the legend perimeter limit the crop, without changing source pixels.
    A component can surround a white click (an open marker); the click need
    not itself fall on an ink pixel.
    """
    a, b, c, d = region
    patch = image[b:d+1, a:c+1].astype(np.float32)
    flat = patch.reshape(-1, 3)
    brightness = flat.mean(axis=1)
    paper = np.median(flat[brightness >= np.percentile(brightness, 80)], axis=0)
    contrast = np.clip(np.linalg.norm(patch-paper, axis=2)/np.sqrt(3), 0, 255).astype(np.uint8)
    peak = float(contrast.max())
    if peak < 18:
        return None
    otsu, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    height, width = contrast.shape
    # Measure straight frame cores over most of the legend, not merely dark
    # columns occupied intermittently by stacked symbols or lettering.
    weak = np.uint8(contrast > max(12., otsu*.60))
    vertical = cv2.morphologyEx(weak, cv2.MORPH_OPEN,
                              np.ones((max(20, int(height*.65)), 1), np.uint8))
    horizontal = cv2.morphologyEx(weak, cv2.MORPH_OPEN,
                                np.ones((1, max(30, int(width*.65))), np.uint8))
    columns = np.flatnonzero(vertical.sum(axis=0) >= max(20, height*.60))
    rows = np.flatnonzero(horizontal.sum(axis=1) >= max(30, width*.60))
    def narrow_bands(positions):
        bands = np.split(positions, np.flatnonzero(np.diff(positions) > 1)+1)
        return [int(q) for band in bands
                if len(band) <= max(3., .08*min(width, height)) for q in band]
    # A large filled square may occupy most rows/columns too, but its broad
    # support band is a marker body rather than a thin legend border.
    columns, rows = narrow_bands(columns), narrow_bands(rows)
    limits = [0, 0, width-1, height-1]
    cx, cy = x-a, y-b
    for q in columns:
        if q < .20*width and q < cx-2:
            limits[0] = max(limits[0], int(q)+1)
        elif q > .80*width and q > cx+2:
            limits[2] = min(limits[2], int(q)-1)
    for q in rows:
        if q < .20*height and q < cy-2:
            limits[1] = max(limits[1], int(q)+1)
        elif q > .80*height and q > cy+2:
            limits[3] = min(limits[3], int(q)-1)
    left, top, right, bottom = limits
    if not (left <= cx <= right and top <= cy <= bottom):
        return None
    thresholds = [max(12., otsu), max(12., .55*peak), max(12., .70*peak),
                  max(12., .35*peak), max(12., .20*peak)]
    # Black legend text must not set the minimum contrast for a pale colour
    # symbol. Add a threshold measured near the click, away from distant text.
    radius = max(8, min(32, int(.16*min(width, height))))
    near = contrast[max(0, int(cy)-radius):min(height, int(cy)+radius+1),
                    max(0, int(cx)-radius):min(width, int(cx)+radius+1)]
    local_values = near[near > 12]
    if local_values.size:
        thresholds.append(max(12., .5*float(np.percentile(local_values, 90))))
    for threshold in dict.fromkeys(thresholds):
        ink = np.uint8(contrast > threshold)
        ink[:top] = 0; ink[bottom+1:] = 0
        ink[:, :left] = 0; ink[:, right+1:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        choices = []
        for component in range(1, count):
            sx, sy, sw, sh, area = map(int, stats[component])
            if min(sw, sh) < 3 or area < 5:
                continue
            # Include a modest click error, but do not select a distant label
            # or an adjacent legend entry merely because it has more ink.
            if not (sx-2 <= cx <= sx+sw+1 and sy-2 <= cy <= sy+sh+1):
                continue
            dx = abs((sx+(sw-1)/2)-cx)
            dy = abs((sy+(sh-1)/2)-cy)
            if dx > max(3., .65*sh, .15*sw) or dy > max(2., .50*sh):
                continue
            # A component covering most of a large legend is not a local
            # marker measurement.
            if ((sh > .75*height and height > 30) or
                    (sw > .85*width and width > 50)):
                continue
            score = dx/max(3, sh) + dy/max(3, sh)
            choices.append((score, sx, sy, sw, sh, component))
        if not choices:
            continue
        _, sx, sy, sw, sh, label_id = min(choices)
        # Width and height follow the actual ink extents independently. One
        # pixel preserves the antialiased fringe; it is not a fixed size prior.
        box = [max(left, sx-1), max(top, sy-1),
               min(right, sx+sw), min(bottom, sy+sh)]
        # Do not let the extra border pixel include a separate neighbouring
        # source component. The source image itself is never thresholded.
        own = labels == label_id
        for edge in ('top', 'bottom', 'left', 'right'):
            if edge == 'top' and box[1] < sy:
                if np.any(ink[box[1], sx:sx+sw] & ~own[box[1], sx:sx+sw]): box[1] = sy
            elif edge == 'bottom' and box[3] >= sy+sh:
                if np.any(ink[box[3], sx:sx+sw] & ~own[box[3], sx:sx+sw]): box[3] = sy+sh-1
            elif edge == 'left' and box[0] < sx:
                if np.any(ink[sy:sy+sh, box[0]] & ~own[sy:sy+sh, box[0]]): box[0] = sx
            elif edge == 'right' and box[2] >= sx+sw:
                if np.any(ink[sy:sy+sh, box[2]] & ~own[sy:sy+sh, box[2]]): box[2] = sx+sw-1
        box = [box[0]+a, box[1]+b, box[2]+a, box[3]+b]
        # An asymmetric triangle crop can have a different box midpoint from
        # the user-selected center. Preserve both rather than recentering it.
        box = [min(box[0], int(math.floor(x))), min(box[1], int(math.floor(y))),
               max(box[2], int(math.ceil(x))), max(box[3], int(math.ceil(y)))]
        return box
    return None


def propose_entry(image_path, legend_box, center, entries=None):
    """Propose an editable source crop from a user-selected marker center.

Source ink supplies independent width and height estimates. Existing entries
supply a fallback only when the clicked source symbol cannot be isolated.
No marker class, reference annotation, or recovered plot point is consulted.
    """
    image, info = _image(image_path)
    region = _box(legend_box, info)
    x, y = _numbers(center, 2, 'Marker center')
    if not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
        raise ValueError('Click inside the selected legend region.')
    existing = validate_review(dict(entries=entries or []), image_path, region,
                               allow_empty=True)['entries']
    measured_box = _clicked_source_box(image, region, x, y)
    if measured_box is not None:
        entry = dict(id='manual_'+uuid.uuid4().hex[:12], box=measured_box, center=[x, y],
                     source='manual', status='crop_proposed', label='Added marker',
                     proposal_note='Width and height measured from the clicked source pixels; check the preview before applying.')
        validate_review(dict(entries=[entry]), image_path, region)
        return entry
    widths = [e['box'][2]-e['box'][0]+1 for e in existing]
    heights = [e['box'][3]-e['box'][1]+1 for e in existing]
    prior_h = float(np.median(heights)) if heights else 19.0
    prior_w = float(np.median(widths)) if widths else 29.0
    crop_h, crop_w = prior_h, prior_w
    half_w, half_h = crop_w/2, crop_h/2
    for entry in existing:
        ex, ey = entry['center']
        if abs(ex-x) < max(prior_w, crop_w)*.8 and abs(ey-y) > 3:
            half_h = min(half_h, max(3, abs(ey-y)/2-1))
        if abs(ey-y) < max(prior_h, crop_h)*.7 and abs(ex-x) > 3:
            half_w = min(half_w, max(3, abs(ex-x)/2-1))
    box = [max(region[0], int(math.floor(x-half_w))),
           max(region[1], int(math.floor(y-half_h))),
           min(region[2], int(math.ceil(x+half_w))),
           min(region[3], int(math.ceil(y+half_h)))]
    entry = dict(id='manual_'+uuid.uuid4().hex[:12], box=box, center=[x, y],
                 source='manual', status='crop_proposed', label='Added marker')
    validate_review(dict(entries=[entry]), image_path, region)
    entry['proposal_note'] = 'The marker extent could not be measured reliably. Adjust the proposed crop before applying.'
    return entry

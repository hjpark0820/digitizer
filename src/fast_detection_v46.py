"""Optional GUI working-image cap; all detector/session pixels stay paired.

The source upload is never overwritten. The resized image is the native input
to the engines (including saved Step 5), not a second internal detector scale.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import cv2

MAX_SIDE = 1400
CONTEXT_ENV = 'CHARTOCODE_FAST_DETECTION_CONTEXT'
METADATA_FILE = 'fast_detection_v46.json'


def pixel_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _coordinate(value, scale):
    # OpenCV resizing maps pixel centres, not pixel corners.
    return (float(value) + .5) * scale - .5


def prepare_job(raster, in_path, out_dir, regions, enabled):
    """Return (native working input path, transformed region strings, report).

    Region coordinates are inclusive, matching the web API. Rounding happens
    once; the exact actual X/Y factors are saved, even for odd image dimensions.
    """
    out_dir = Path(out_dir)
    height, width = raster.shape[:2]
    factor = min(1., MAX_SIDE / max(width, height)) if enabled else 1.
    new_width, new_height = max(1, round(width*factor)), max(1, round(height*factor))
    sx, sy = new_width/width, new_height/height
    applied = (new_width, new_height) != (width, height)
    source_regions, working_regions, mapped = {}, {}, {}
    for name, value in regions.items():
        if not value.strip():
            mapped[name] = value
            continue
        box = [int(float(v)) for v in value.split(',')]
        target = [max(0, min(limit-1, round(_coordinate(v, scale))))
                  for v, scale, limit in zip(box, (sx, sy, sx, sy),
                                             (new_width, new_height, new_width, new_height))]
        if target[0] >= target[2] or target[1] >= target[3]:
            raise ValueError(f'{name} is too small at Fast detection resolution. Disable Fast detection or select a larger region.')
        source_regions[name], working_regions[name] = box, target
        mapped[name] = ','.join(map(str, target)) if applied else value
    working = cv2.resize(raster, (new_width, new_height), interpolation=cv2.INTER_AREA) if applied else raster
    report = dict(enabled=bool(enabled), applied=applied, max_side=MAX_SIDE,
                  source_size=[width, height], working_size=[new_width, new_height],
                  scale_x=sx, scale_y=sy, source_regions=source_regions,
                  working_regions=working_regions, source_sha256=pixel_hash(raster),
                  working_sha256=pixel_hash(working),
                  coordinate_space='working_image', axis_reading='source_image',
                  note='Results, editing and Step 5 use the paired working image. Small markers may lose detail; no full-resolution marker refinement is applied.')
    if applied:
        if not cv2.imwrite(str(out_dir/'source_input.png'), raster):
            raise OSError('Could not preserve the source image for axis reading')
        working_path = out_dir/'input.png'
        if not cv2.imwrite(str(working_path), working):
            raise OSError('Could not save the Fast detection working image')
    else:
        working_path = Path(in_path)
    (out_dir/METADATA_FILE).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return working_path, mapped, report


def calibration_source(image, plot_area):
    """Validated source raster/ROI for OCR; only set by this job's server env."""
    value = os.environ.get(CONTEXT_ENV)
    if not value:
        return None
    path = Path(value)
    info = json.loads(path.read_text(encoding='utf-8'))
    if not info.get('applied'):
        return None
    if list(image.shape[1::-1]) != info['working_size'] or pixel_hash(image) != info['working_sha256']:
        raise ValueError('Fast detection calibration belongs to a different working image')
    source = cv2.imread(str(path.parent/'source_input.png'))
    if source is None or pixel_hash(source) != info['source_sha256']:
        raise ValueError('Fast detection original image is unavailable or changed')
    scales = (info['scale_x'], info['scale_y'], info['scale_x'], info['scale_y'])
    source_box = []
    known_work = info['working_regions'].get('plot_area', [])
    known_source = info['source_regions'].get('plot_area', [])
    for index, (p, scale) in enumerate(zip(plot_area, scales)):
        # Preserve the original manual ROI rather than round-tripping its edges.
        source_box.append(known_source[index] if len(known_work) == 4 and p == known_work[index]
                          else round((p+.5)/scale-.5))
    return source, source_box, info


def working_calibration(result, info):
    """Retain numeric units/status, changing only the pixel anchor coordinates."""
    result = deepcopy(result)
    for axis, scale in (('x', info['scale_x']), ('y', info['scale_y'])):
        row = result['calibration'][axis]
        row['p0'], row['p1'] = (_coordinate(row[k], scale) for k in ('p0', 'p1'))
        if row.get('kind') == 'pixel':
            row['pixel_units'] = 'source_image_pixels'
            row['warning'] += ' Pixel distances refer to the source image, before Fast detection resizing.'
    result['diagnostics']['coordinate_space'] = 'source_image'
    result['diagnostics']['source_size'] = info['source_size']
    result['diagnostics']['working_size'] = info['working_size']
    result['diagnostics']['fast_detection'] = True
    return result

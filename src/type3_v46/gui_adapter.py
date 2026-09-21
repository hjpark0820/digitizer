"""Validated type-3 detection/correction handoff for the shared v46 GUI.

Measured centres, path-only knots and endpoints admitted by the user prior keep
separate provenance. None of these fields manufacture a marker glyph.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

DETECTION_NAME = 'type3_detection_v46.json'
VERSION = 'type3_detection_v46_v1'
BACKEND = 'v46_type3_path_v1'
STATE_NAME = 'color_correction_state.json'


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _inclusive(box):
    if box is None:
        return None
    values = np.asarray(box, float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= values[:2]):
        raise ValueError('Type3 detection boxes must be nonempty half-open xyxy boxes')
    return [float(values[0]), float(values[1]), float(values[2]-1), float(values[3]-1)]


def validate_detection(detection, image, edit_data, *, plot_area=None, legend_box=None):
    """Bind evidence to decoded pixels, source geometry and every data series."""
    if detection.get('version') != VERSION:
        raise ValueError('Unsupported type3 detection version; rerun v46 detection')
    if edit_data.get('series_mode') != 'line-only' or any(
            c.get('series_mode') != 'line-only' for c in edit_data.get('curves', [])):
        raise ValueError('Current GUI detection is not type3; do not reuse stale type3 evidence')
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if detection.get('image_sha256') != digest:
        raise ValueError('Type3 source image pixels changed; rerun detection')
    if detection.get('image_shape') is not None and detection['image_shape'] != list(image.shape):
        raise ValueError('Type3 source image shape changed; rerun detection')
    plot, legend = _inclusive(detection['plot_box']), _inclusive(detection.get('legend_box'))
    for box in (plot, legend):
        if box is not None and not (0 <= box[0] <= box[2] < image.shape[1] and 0 <= box[1] <= box[3] < image.shape[0]):
            raise ValueError('Type3 source box lies outside the image')
    if plot_area is not None and list(plot_area) != plot:
        raise ValueError('Type3 plot area changed; rerun detection')
    if legend_box is not None and list(legend_box) != legend:
        raise ValueError('Type3 legend area changed; rerun detection')
    names = [s['id'] for s in detection['series'] if s.get('role', 'data_series') == 'data_series']
    edited = [c.get('name') for c in edit_data.get('curves', [])]
    if (len(set(names)) != len(names) or len(set(edited)) != len(edited) or set(names) != set(edited)
            or any(not isinstance(s, str) or not s or Path(s).name != s or s in ('.', '..') for s in names)):
        raise ValueError('Type3 series identities differ from edit_data.json')
    return dict(version=VERSION, backend=BACKEND, image_sha256=digest,
                image_shape=list(image.shape), plot_area=plot, legend_box=legend,
                series_names=sorted(names))


def gui_point(point):
    """Editable pixel coordinate plus source identity/provenance, no enum glyph."""
    result = {key: deepcopy(point[key]) for key in (
        'id', 'candidate_id', 'kind', 'existence', 'original_L0', 'endpoint_evidence',
        'independent_measurement_evidence', 'strong_by_endpoint_prior', 'evidence_ref',
        'endpoint_side', 'step5_eligible', 'status') if key in point}
    result.update(x=float(point['x_px']), y=float(point['y_px']), marker_glyph_detected=False)
    return result


def export_correction(edit_data, summary):
    """Keep calibration/labels intact while updating active points and geometry."""
    updated = deepcopy(edit_data)
    by_id = {s['id']: s for s in summary['series']}
    for curve in updated['curves']:
        row = by_id[curve['name']]
        curve.update(points=[gui_point(p) for p in row['final_points']],
                     series_mode='line-only', marker_class=None, marker_glyph_detected=False,
                     correction_status=row['stop_reason'])
        path = (row['iterations'][-1].get('reconstructed_after', []) if row['iterations'] else [])
        if not path:
            path = sorted([[p['x'], p['y']] for p in curve['points']])
        curve['reconstruction'] = dict(model='linear', points=[dict(x=float(x), y=float(y)) for x, y in path],
            anchors=[dict(x=p['x'], y=p['y']) for p in curve['points']], status='unvalidated_type3_hypotheses')
    updated['correction'] = dict(backend=BACKEND, metric='chamfer', model='linear',
        reference_policy='estimated_path', inferred_weight=.25, existence='unvalidated_hypotheses',
        endpoint_suppressed_policy='global_endpoint_strong_prior_v1',
        requested_iterations=summary['requested_iterations'])
    return updated


def _overlay(image, curves):
    canvas = image.copy()
    radius = max(4, int(round(max(image.shape[:2])*.006)))
    for curve in curves:
        rgb = curve.get('rgb', [0, 0, 0])
        color = tuple(int(v) for v in rgb[::-1])
        for point in curve['points']:
            centre = (int(round(point['x'])), int(round(point['y'])))
            if point.get('strong_by_endpoint_prior'):
                x, y = centre
                vertices = np.array([[x,y-radius],[x+radius,y+radius],[x-radius,y+radius]],np.int32)
                cv2.polylines(canvas, [vertices], True, (255,255,255), 4, cv2.LINE_AA)
                cv2.polylines(canvas, [vertices], True, color, 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, centre, radius, (255,255,255), 4, cv2.LINE_AA)
                cv2.circle(canvas, centre, radius, color, 2, cv2.LINE_AA)
    return canvas


def correct_saved_detection(image_path, out_dir, *, plot_area=None, legend_box=None,
                            iterations=5, previous_path=None, engine_dir=None, use_edit_points=False):
    """Execute the typed private Step5 engine, preserving GUI resume identity."""
    from .step5_adapter import run_step5
    base = Path(out_dir)
    detection = json.loads((base/DETECTION_NAME).read_text(encoding='utf-8'))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Cannot read type3 source image')
    edit_path = base/'edit_data.json'
    edit_data = json.loads(edit_path.read_text(encoding='utf-8'))
    identity = validate_detection(detection, image, edit_data, plot_area=plot_area, legend_box=legend_box)
    previous = None
    if previous_path:
        previous = json.loads(Path(previous_path).read_text(encoding='utf-8'))
        if previous.get('backend') != BACKEND or previous.get('identity') != identity:
            raise ValueError('Previous correction is not the same type3 image/geometry/series')
        if set(previous.get('curves', {})) != set(identity['series_names']):
            raise ValueError('Previous type3 correction is missing a series')
    detection.update(identity=identity, source_image_path=str(Path(image_path).resolve()),
                     source_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest())
    target = base/'step5'
    index = 2
    while (target/'step5.json').exists() or (target/'native').exists():
        target = base/f'step5_type3_run_{index:03d}'
        index += 1
    extra = {'current_curves':{c['name']:[[p['x'],p['y']] for p in c['points']] for c in edit_data['curves']}} if use_edit_points else {}
    summary = run_step5(detection, target, iterations=iterations, previous_state=previous, engine_dir=engine_dir, **extra)
    updated = export_correction(edit_data, summary)
    _write(edit_path, updated)
    cv2.imwrite(str(base/'data_points_overlay.png'), _overlay(image, updated['curves']))
    state = dict(backend=BACKEND, identity=identity,
        curves={s['id']:[[p['x_px'],p['y_px']] for p in s['final_points']] for s in summary['series']},
        suppressed={s['id']:s['final_suppressed'] for s in summary['series']},
        path_state=summary['path_state'], path_options=dict(metric='chamfer',model='linear',
            reference_policy='estimated_path',inferred_weight=.25))
    _write(base/STATE_NAME, state)
    _write(base/'type3_correction_summary_v46.json', summary)
    print(f'[v46 type3 Step5] {sum(len(s["final_points"]) for s in summary["series"])} active centre hypotheses; '
          f'up to {iterations} iterations, strong outer-endpoint prior; no marker glyph inference.', flush=True)
    return summary

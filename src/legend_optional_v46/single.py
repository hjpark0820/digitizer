"""Native-pixel single-series detection without an invented legend.

The caller supplies the single-series condition. An observed plot sample sets
colour only; the recurrent-body miner and hybrid window verifier establish
marker evidence separately. Failed mining never establishes marker absence.
"""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import cv2
import numpy as np

import color_marker_evidence as base
import color_marker_evidence_v2 as evidence_v2
from color_marker_hybrid_v2 import analyze_evidence, select_candidates
from color_marker_runtime_v46 import production_config

from .palette import PaletteConfig, _spatial_support, estimate_plot_palette


VERSION = 'legend_optional_v46_observed_single_series'


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def spatial_valid(plot_box, exclusion_boxes=()):
    """Spatial permission, independent of foreground and palette source boxes."""
    x0, y0, x1, y1 = map(int, plot_box)
    valid = np.ones((y1-y0, x1-x0), bool)
    for box in exclusion_boxes:
        a, b = max(int(box[0]), x0)-x0, max(int(box[1]), y0)-y0
        c, d = min(int(box[2]), x1)-x0, min(int(box[3]), y1)-y0
        if a < c and b < d:
            valid[b:d, a:c] = False
    return valid


def boundary_axis_exclusions(image_bgr, plot_box):
    """Find observed long neutral strokes at plot boundaries for BW evidence.

    Internal grid lines remain available for the full-window verifier. Only
    long, thin boundary strokes are spatial exclusions; pixels are not painted
    over and an ordinary colour sample never enters the exclusions.
    """
    x0, y0, x1, y1 = plot_box
    crop = image_bgr[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    if min(height, width) < 20:
        return []
    paper = base._paper(crop)
    contrast = np.linalg.norm(crop.astype(np.float32)-paper, axis=2)
    neutral = ((contrast >= 45.) & (np.ptp(crop.astype(np.float32), axis=2) < 24.)).astype(np.uint8)
    max_thickness = max(4, min(12, int(round(.025*min(height, width)))))
    margin_x, margin_y = max(3, int(round(.04*width))), max(3, int(round(.04*height)))
    boxes = []
    for orientation, length in [('horizontal', width), ('vertical', height)]:
        kernel_length = max(12, int(np.ceil(.55*length)))
        kernel = np.ones((1, kernel_length) if orientation == 'horizontal' else (kernel_length, 1), np.uint8)
        opened = cv2.morphologyEx(neutral, cv2.MORPH_OPEN, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for index in range(1, count):
            a, b, w, h, _ = map(int, stats[index])
            if orientation == 'horizontal':
                eligible = w >= .55*width and h <= max_thickness and (b <= margin_y or b+h >= height-margin_y)
            else:
                eligible = h >= .55*height and w <= max_thickness and (a <= margin_x or a+w >= width-margin_x)
            if eligible:
                boxes.append([x0+max(0, a-1), y0+max(0, b-1),
                              x0+min(width, a+w+1), y0+min(height, b+h+1)])
    return boxes


def marker_evidence(image_bgr, plot_box, templates, *, valid=None, exclusion_boxes=()):
    """Adapt the frozen hybrid arrays without its swatch-as-legend exclusion."""
    x0, y0, x1, y1 = plot_box
    crop = image_bgr[y0:y1, x0:x1]
    allowed = spatial_valid(plot_box, exclusion_boxes) if valid is None else np.asarray(valid, bool).copy()
    if allowed.shape != crop.shape[:2]:
        raise ValueError('valid must have the native plot-local shape')
    value = evidence_v2.colour_evidence(crop, templates)
    for key in ('membership', 'colour_confidence', 'other'):
        value[key][:, ~allowed] = 0
    for key in ('all_ink', 'unknown'):
        value[key][~allowed] = 0
    value.update(crop_bgr=crop, templates=templates, template_errors=[], valid=allowed,
        excluded_boxes=[list(box) for box in exclusion_boxes], plot_box_source=list(plot_box),
        scale=1., scale_x=1., scale_y=1., source_center_offset=[x0, y0],
        native_shape=list(crop.shape[:2]), version=VERSION)
    return value


def _inside_valid(point, plot_box, valid):
    x0, y0, x1, y1 = plot_box
    x, y = float(point['x_px']), float(point['y_px'])
    if not (x0 <= x < x1 and y0 <= y < y1):
        return False
    px = min(valid.shape[1]-1, max(0, int(round(x-x0))))
    py = min(valid.shape[0]-1, max(0, int(round(y-y0))))
    return bool(valid[py, px])


def detect_markers(image_bgr, plot_box, templates, *, valid=None, exclusion_boxes=()):
    """Use the existing hybrid proposals, window scores and production cutoff."""
    evidence = marker_evidence(image_bgr, plot_box, templates,
        valid=valid, exclusion_boxes=exclusion_boxes)
    analysis = analyze_evidence(evidence)
    for point in analysis['candidates']:
        point.update(x_px=float(point['x']+plot_box[0]), y_px=float(point['y']+plot_box[1]),
                     diameter_source=float(point['diameter']), class_name='observed_marker',
                     kind='observed_marker', evidence_kind='observed_template_window')
    selection = select_candidates(analysis['candidates'], config=production_config(),
                                  joint=True, line_gate=True)
    for key in ('points', 'uncertain_points'):
        selection[key] = [point for point in selection[key]
                          if _inside_valid(point, plot_box, evidence['valid'])]
    selection.update(candidates=analysis['candidates'], timing=analysis['timing'],
                     proposal_diagnostics=analysis['proposal_diagnostics'])
    return selection, evidence


def _fields(evidence):
    membership = evidence['membership'][0]
    return dict(membership=membership, soft=membership,
        own=(membership >= .18) & evidence['valid'],
        confidence=evidence['colour_confidence'][0],
        colour_confidence=evidence['colour_confidence'][0],
        all_ink=evidence['all_ink'], other=evidence['other'][0],
        unknown=evidence['unknown'], paper=np.clip(1.-evidence['all_ink'], 0., 1.),
        valid=evidence['valid'])


def detect_single(image_bgr, plot_box, *, mode='color'):
    """Detect a caller-declared no-legend single series at original resolution.

    Input boxes use half-open source coordinates. Output points are original
    image pixel centres and all evidence fields are plot-local. The return
    value is JSON serializable, including observed template arrays. Curve
    tracing belongs to the runtime; ``paths`` remains empty in this detector.
    BW permits a spatially supported neutral palette and excludes measured
    boundary axes from palette sampling, mining and final candidate selection.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image_bgr must be uint8 HxWx3 BGR pixels')
    if mode not in ('color', 'bw'):
        raise ValueError("mode must be 'color' or 'bw'")
    plot = base._box(plot_box, image.shape)
    started = perf_counter()
    exclusions = boundary_axis_exclusions(image, plot) if mode == 'bw' else []
    valid = spatial_valid(plot, exclusions)
    palette = estimate_plot_palette(image, plot, valid=valid, allow_achromatic=mode == 'bw')
    result = dict(version=VERSION, status=palette['status'], mode=mode,
        plot_box=plot, plot_box_source=plot, legend_box=None, source_shape=list(image.shape[:2]),
        native_resolution=True, scale_x=1., scale_y=1., source_center_offset=plot[:2],
        coordinate_convention='half-open xyxy; x_px/y_px are source-image pixel centres',
        series=[], points=[], uncertain_points=[], suppressed_points=[], paths=[],
        templates=[], template_errors=[], palette=palette, valid=valid, fields={},
        exclusion_boxes=exclusions, selection_config=asdict(production_config()),
        evidence=dict(palette_sample_is_exclusion=False, marker_absence_established=False,
            single_series_condition='caller_supplied',
            axis_exclusion_policy='observed_long_neutral_boundary_strokes' if mode == 'bw' else 'none'),
        timing=dict(palette_seconds=perf_counter()-started), warnings=[])
    if palette.get('model') is None:
        return _plain(result)
    if palette.get('diagnostic_only'):
        # Neutral membership shares ink with axes and text. Explicit BW mode
        # and distributed coherent support permit mining, not marker absence.
        x0, y0, x1, y1 = plot
        soft, confidence = base._membership(image[y0:y1, x0:x1], palette['model'])
        spatial, _ = _spatial_support((soft >= .30) & (confidence >= .65) & valid, PaletteConfig())
        palette['spatial_support'] = spatial
        if not spatial['sufficient_for_single_series_palette']:
            result['status'] = 'insufficient_spatial_palette_support'
            palette.update(status=result['status'], diagnostic_model=palette['model'], model=None)
            result['warnings'].append('Neutral ink lacks coherent distributed support after boundary-axis exclusion.')
            return _plain(result)
        palette.update(status='estimated_achromatic', diagnostic_only=False,
            activation_basis='explicit_bw_single_series_with_spatial_support')
        palette['warnings'] = [warning for warning in palette['warnings']
                               if not warning.startswith('Achromatic diagnostic only:')]
        palette['warnings'].append('Neutral ink can include grid or text. Only verified recurrent bodies are reported as markers.')
    rgb = [int(round(value)) for value in np.asarray(palette['model']['bgr'])[::-1]]
    entry = dict(id='S01', label='Series 1', rgb=rgb, marker_state='unknown',
        template_source=None, colour_source='plot_observed',
        palette_source_box=palette['source_box'], source_kind='palette_sample_not_legend',
        single_series_condition='caller_supplied', class_name='observed_marker')
    result['series'] = [entry]
    spec = dict(id='S01', label='Series 1', color_rgb=rgb,
                swatch_box=palette['source_box'], legend_box=None)
    # Axis-only specs have no swatch/model and cannot become rival series.
    # The miner consumes their spatial boxes; the real plot seed stays valid.
    mining_specs = [spec]+[dict(id=f'axis_{i}', legend_box=box) for i, box in enumerate(exclusions)]
    tick = perf_counter()
    try:
        template = evidence_v2._mine_plot_template(image, plot, spec, mining_specs)
        template['provenance'].update(palette_source_kind='plot_observed_not_legend',
            palette_model_selection=palette.get('method'), actual_legend_present=False,
            boundary_axis_exclusion_boxes=exclusions)
        entry.update(template_source='plot_mined', template_provenance=template['provenance'],
                     diameter_px=float(template['diameter']))
    except ValueError as error:
        template = None
        entry['template_error'] = str(error)
        result['template_errors'].append(dict(series_id='S01', reason=str(error)))
    result['timing']['template_seconds'] = perf_counter()-tick
    if template is None:
        palette_entry = dict(id='S01', label='Series 1', rgb=rgb, model=palette['model'])
        evidence = marker_evidence(image, plot, [palette_entry], valid=valid, exclusion_boxes=exclusions)
        result['status'] = 'unresolved_markers'
        result['warnings'].append('No recurrent marker template was verified. Mining failure does not establish a marker-free curve.')
    else:
        tick = perf_counter()
        detection, evidence = detect_markers(image, plot, [template], valid=valid, exclusion_boxes=exclusions)
        result.update(points=detection['points'], uncertain_points=detection['uncertain_points'],
            suppressed_points=[dict(point, series_id=point.get('series_id') or 'S01',
                                    suppression_reason='uncertain_identity')
                               for point in detection['uncertain_points']],
            templates=[template], marker_diagnostics=detection)
        result['timing']['marker_seconds'] = perf_counter()-tick
        if result['points']:
            entry['marker_state'] = 'present'
            result['status'] = 'completed_markers'
        else:
            result['status'] = 'unresolved_markers'
            result['warnings'].append('An observed recurrent template was found, but no marker passed the full-window verifier.')
    result['fields'] = _fields(evidence)
    result['evidence'].update(native_shape=evidence['native_shape'],
        source_center_offset=evidence['source_center_offset'],
        excluded_boxes=exclusions, template_count=len(result['templates']))
    result['timing']['total_seconds'] = perf_counter()-started
    return _plain(result)

# Production snapshot of experiments/color_path_v45/shape_occlusion_explanation.py.
# Algorithms are local to src; no experimental runtime dependency.
"""Native-legend partial-shape explanations, not confirmed marker detections.

The image-backed proposal pool is independent of old blob-score maxima. Each
translation of the actual legend silhouette competes with a line-only
explanation. Other-colour ink can excuse missing target pixels but never
contributes positive target evidence. A path only limits where to search.

No shape names, plot locations, hand-drawn primitives, marker counts, data
points, or fitted-curve sampling grid enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Config:
    """Fixed, common native-pixel settings; scores are not probabilities."""

    path_corridor_diameters: float = 2.5
    missing_penalty: float = 1.0
    unexplained_target_penalty: float = 0.30
    edge_reward: float = 0.20
    minimum_exclusive_mass: float = 2.0
    minimum_explained_residual: float = 1.0
    minimum_visible_fraction: float = 0.008
    maximum_missing_fraction: float = 0.78
    maximum_invalid_fraction: float = 0.15
    supported_maximum_missing_fraction: float = 0.25
    supported_minimum_visible_fraction: float = 0.035
    supported_minimum_gain_fraction: float = 0.005
    nms_radius_diameters: float = 0.55
    local_max_radius_diameters: float = 0.10
    max_candidates: int = 40
    geometry_blur_sigma: float = 0.55


def _colour_component(source_rgb, box, palette, series_index):
    x0, y0, x1, y1 = map(int, box)
    source = np.asarray(source_rgb)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError('source_rgb must be HxWx3 RGB')
    if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
        raise ValueError('legend swatch box must be inside source image')
    crop = source[y0:y1, x0:x1, :3].astype(np.float32)
    directions = 255.0 - crop
    directions /= np.maximum(np.linalg.norm(directions, axis=2, keepdims=True), 1.0)
    references = 255.0 - np.asarray(palette, np.float32)
    if references.ndim != 2 or references.shape[1] != 3 or not 0 <= series_index < len(references):
        raise ValueError('palette must be Kx3 and series_index in range')
    references /= np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1.0)
    owner = np.argmax(directions @ references.T, axis=2)
    selected = ((np.ptp(crop, axis=2) >= 40) & (crop.min(2) <= 210)
                & (owner == series_index))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
    eligible = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 8]
    if not eligible:
        raise ValueError('No colour component available in legend swatch box')
    winner = max(eligible, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    return crop.astype(np.uint8), labels == winner


def extract_legend_shape(source_rgb, swatch_box, palette_rgb, series_index,
                         cfg=Config()):
    """Measure and retain the central native silhouette with attached ink.

    Flank columns estimate horizontal-connector thickness. The increase in
    occupied column height locates the marker footprint. No line-shaped pixels
    are erased inside that footprint: connector rows are downweighted because
    the original marker contour is unknowable there. The output is a native
    raster template, not an idealized triangle/circle or an inferred full shape.
    """
    crop, component = _colour_component(source_rgb, swatch_box, palette_rgb, series_index)
    ys, xs = np.nonzero(component)
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    height = bottom - top + 1
    columns = component.sum(0).astype(float)
    occupied = np.flatnonzero(columns)
    flank_count = max(1, int(np.ceil(0.2 * len(occupied))))
    flank_cols = np.r_[occupied[:flank_count], occupied[-flank_count:]]
    connector_thickness = float(np.median(columns[flank_cols]))
    marker_cols = np.flatnonzero(columns > connector_thickness + max(1.0, 0.10 * height))
    if len(marker_cols) < 2:
        raise ValueError('Legend component has no central height excess over connector')
    # Use the component-height scale to retain tapering tips that meet the line.
    marker_left, marker_right = int(marker_cols.min()), int(marker_cols.max())
    cx = int(round((marker_left + marker_right) / 2))
    cy = int(round((top + bottom) / 2))
    half_marker = max(int(np.ceil((height - 1) / 2)),
                      cx - marker_left + 1, marker_right - cx + 1)
    diameter = float(max(height, 2 * half_marker + 1))
    # Template includes a small background rim; native pixels are never rescaled.
    half = half_marker + 2
    size = 2 * half + 1
    template = np.zeros((size, size), np.float32)
    line_uncertain = np.zeros_like(template)
    template_rgb = np.full((size, size, 3), 255, np.uint8)
    crop_origin = np.array([cx - half, cy - half], int)
    rows_in_flanks = component[:, flank_cols].mean(1)
    line_rows = np.flatnonzero(rows_in_flanks >= 0.50)
    # Discard external connector by restricting the central footprint only.
    # Internal coloured pixels, including the whole connector band, are kept.
    for ty in range(size):
        sy = int(crop_origin[1] + ty)
        if not 0 <= sy < component.shape[0]:
            continue
        for tx in range(size):
            sx = int(crop_origin[0] + tx)
            if not 0 <= sx < component.shape[1]:
                continue
            template_rgb[ty, tx] = crop[sy, sx]
            if abs(sx - cx) <= half_marker:
                template[ty, tx] = component[sy, sx]
                if sy in line_rows:
                    line_uncertain[ty, tx] = 1.0
    if template.sum() < 5:
        raise ValueError('Native legend central footprint contains too little ink')
    expected = ndimage.gaussian_filter(template, cfg.geometry_blur_sigma,
                                       mode='constant').astype(np.float32)
    weight = expected * (1.0 - 0.75 * line_uncertain)
    boundary = template - cv2.erode(template, np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
    edge = ndimage.gaussian_filter(boundary, 0.60, mode='constant')
    edge *= 1.0 - 0.75 * line_uncertain
    x0, y0 = map(int, swatch_box[:2])
    info = {
        'diameter': diameter, 'height_native': int(height),
        'template_center_source': [x0 + cx, y0 + cy],
        'template_box_source': [x0 + int(crop_origin[0]), y0 + int(crop_origin[1]),
                                x0 + int(crop_origin[0]) + size,
                                y0 + int(crop_origin[1]) + size],
        'swatch_box_source': list(map(int, swatch_box)),
        'coloured_component_box_source': [x0+left, y0+top, x0+right+1, y0+bottom+1],
        'attached_line_rows': [y0 + int(y) for y in line_rows],
        'connector_thickness_native': connector_thickness,
        'native_template_ink_pixels': int(template.sum()),
        'weighted_template_area': float(weight.sum()),
        'geometry_blur_sigma': cfg.geometry_blur_sigma,
        'scale': 1.0, 'shape_name': 'not classified; actual native legend raster',
        'method': 'central column-height excess over flank connector; internal ink preserved',
        'note': 'Connector-band geometry is uncertain and downweighted, not erased or reconstructed. No plot-specific rescaling.',
    }
    fields = {'template_mask': template, 'template_weight': weight,
              'template_edge': edge.astype(np.float32), 'line_uncertain': line_uncertain,
              'component_mask': component, 'legend_rgb': crop,
              'template_rgb': template_rgb, 'template_expected': expected}
    return info, fields


def _correlate(field, kernel):
    return cv2.filter2D(np.asarray(field, np.float32), cv2.CV_32F,
                       np.asarray(kernel, np.float32), borderType=cv2.BORDER_CONSTANT)


def _patch(field, x, y, shape):
    """Integer-centred patch with explicit outside=zero, never wrap around."""
    out = np.zeros(shape, np.float32)
    h, w = shape
    x0, y0 = int(x) - w//2, int(y) - h//2
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(field.shape[1], x0+w), min(field.shape[0], y0+h)
    if xb > xa and yb > ya:
        out[ya-y0:yb-y0, xa-x0:xb-x0] = field[ya:yb, xa:xb]
    return out


def _validate_maps(soft, own, other_confidence, valid, path_distance, nuisance):
    soft = np.asarray(soft, np.float32)
    own = np.asarray(own, bool)
    other = np.asarray(other_confidence, np.float32)
    valid = np.asarray(valid, bool)
    distance = np.asarray(path_distance, np.float32)
    nuisance = np.asarray(nuisance, np.float32)
    if soft.ndim != 2 or any(v.shape != soft.shape for v in (own, other, valid, distance, nuisance)):
        raise ValueError('All plot fields must have the same HxW shape')
    for name, value in [('soft', soft), ('other_confidence', other), ('line_nuisance', nuisance)]:
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1+1e-6).any():
            raise ValueError(f'{name} must contain finite values in [0,1]')
    if np.isnan(distance).any() or (distance < 0).any():
        raise ValueError('path_distance must be nonnegative; infinity is allowed')
    return soft, own, other, valid, distance, nuisance


def analyze_shape_candidates(soft, own, other_confidence, valid, path_distance,
                             line_nuisance, shape_info, shape_fields,
                             foreground_markers=None, cfg=Config()):
    """Translate the actual template; compare H0=line and H1=line+marker.

    H0 target residual is unexplained exclusive target-colour mass in the
    template window after the supplied line nuisance. H1 explains the portion
    inside the translated template but adds a strong loss for expected pixels
    with no target colour and no actual other-colour occluder. Target-colour
    leftovers outside the shape also lower the ranking; other-colour leftovers
    never do. This is a heuristic weighted residual, not a generative likelihood.

    Returned ``evidence_supported`` is an image-comparison gate only. Existence
    stays unknown, and the parent evaluates conditional curve coverage separately.
    """
    soft, own, other, valid, distance, nuisance = _validate_maps(
        soft, own, other_confidence, valid, path_distance, line_nuisance)
    diameter = float(shape_info['diameter'])
    if not np.isfinite(diameter) or diameter < 2 or cfg.max_candidates < 1:
        raise ValueError('Valid native marker diameter and positive candidate cap required')
    weight = np.asarray(shape_fields['template_weight'], np.float32)
    expected = np.asarray(shape_fields.get('template_expected', shape_fields['template_mask']), np.float32)
    edge = np.asarray(shape_fields['template_edge'], np.float32)
    if weight.ndim != 2 or expected.shape != weight.shape or edge.shape != weight.shape or any(n % 2 == 0 for n in weight.shape):
        raise ValueError('Template kernels must have matching odd dimensions')
    area = float(weight.sum())
    if area < 1 or (weight < 0).any() or not np.isfinite(weight).all():
        raise ValueError('Template must have positive finite weight')
    # Exclusive target ownership prevents a second palette's weak soft response
    # from becoming positive shape evidence. Own pixels cannot be occluders.
    target = soft * own * valid
    occluder = np.clip(other, 0, 1) * (~own) * valid
    line = np.minimum(target, nuisance * own * valid)
    residual = np.maximum(target - line, 0)
    certainty = np.divide(weight, np.maximum(expected, 1e-8),
                          out=np.zeros_like(weight), where=expected > 1e-8)
    # Expected support is weighted by geometry uncertainty, including the
    # connector band, while positive evidence remains observed target pixels.
    visible = np.maximum(_correlate(target, weight), 0)
    explained = np.maximum(_correlate(residual, weight), 0)
    missing = np.maximum(_correlate((1.0-target)*(1.0-occluder)*valid, weight), 0)
    occluded = np.maximum(_correlate(occluder, weight), 0)
    invalid = np.clip(1.0 - _correlate(valid.astype(np.float32), weight)/area, 0, 1)
    # The marker window, not the whole plot, defines unexplained own-colour ink.
    # Background rim has modest weight to avoid rewarding an oversized crop.
    window = np.full(weight.shape, 0.25, np.float32)
    window = np.maximum(window, certainty)
    outside = np.maximum(window - weight, 0)
    excess = np.maximum(_correlate(residual, outside), 0)
    edge_evidence = np.maximum(_correlate(residual, edge), 0)
    edge_area = max(float(edge.sum()), 1e-6)
    gain = explained - cfg.missing_penalty*missing
    score = (gain - cfg.unexplained_target_penalty*excess)/area
    score += cfg.edge_reward*edge_evidence/edge_area
    corridor = valid & (distance <= cfg.path_corridor_diameters*diameter)
    eligible = (corridor & (visible >= cfg.minimum_exclusive_mass)
                & (visible/area >= cfg.minimum_visible_fraction)
                & (explained >= cfg.minimum_explained_residual)
                & (missing/area <= cfg.maximum_missing_fraction)
                & (invalid <= cfg.maximum_invalid_fraction))
    # Local maxima are over template-translation explanations, never over the
    # old blob-density field. Flat ridges are not converted into periodic points.
    radius = max(1, int(round(cfg.local_max_radius_diameters*diameter)))
    ranked_score = np.where(eligible, score, -np.inf)
    maxima = eligible & (ranked_score >= ndimage.maximum_filter(
        ranked_score, size=2*radius+1, mode='constant', cval=-np.inf)-1e-7)
    labels, count = ndimage.label(maxima, np.ones((3, 3), bool))
    pool = []
    for label, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        yy, xx = np.nonzero(labels[sl] == label)
        yy += sl[0].start
        xx += sl[1].start
        if max(np.ptp(xx), np.ptp(yy)) > 2*diameter:
            continue
        best = int(np.argmax(score[yy, xx]))
        pool.append((int(xx[best]), int(yy[best]), float(score[yy[best], xx[best]])))
    pool.sort(key=lambda row: (-row[2], row[1], row[0]))
    selected = []
    for x, y, rank in pool:
        if all(np.hypot(x-px, y-py) >= cfg.nms_radius_diameters*diameter
               for px, py, _ in selected):
            selected.append((x, y, rank))
            if len(selected) == cfg.max_candidates:
                break
    marker_footprints = np.zeros_like(target)
    yy, xx = np.indices(target.shape)
    for marker in foreground_markers or []:
        mx, my, mr = float(marker['x']), float(marker['y']), float(marker['radius'])
        if mr > 0:
            marker_footprints = np.maximum(marker_footprints,
                ((xx-mx)**2 + (yy-my)**2 <= mr**2).astype(np.float32))
    # Peer markers annotate only measured other-colour pixels, never synthesize
    # coverage within the whole disk or relax the candidate's evidence gates.
    peer_occlusion = _correlate(occluder*marker_footprints, weight)/area
    candidates = []
    h, w = weight.shape
    qy, qx = np.indices(weight.shape)
    for number, (x, y, rank) in enumerate(selected, 1):
        own_patch = _patch(residual, x, y, weight.shape)*weight
        quadrants = [float(own_patch[(qy < h//2) & (qx < w//2)].sum()),
                     float(own_patch[(qy < h//2) & (qx >= w//2)].sum()),
                     float(own_patch[(qy >= h//2) & (qx < w//2)].sum()),
                     float(own_patch[(qy >= h//2) & (qx >= w//2)].sum())]
        fragment_quadrants = sum(q >= 0.5 for q in quadrants)
        h0 = float(explained[y, x]+excess[y, x])
        h1 = float(excess[y, x]+cfg.missing_penalty*missing[y, x])
        metrics = {
            'h0_residual': h0, 'h1_residual': h1,
            'residual_gain': float(gain[y, x]),
            'residual_gain_fraction': float(gain[y, x]/area),
            'visible_support_fraction': float(visible[y, x]/area),
            'occluded_fraction': float(occluded[y, x]/area),
            'missing_unoccluded_fraction': float(missing[y, x]/area),
            'unexplained_target_fraction': float(excess[y, x]/area),
            'edge_support_fraction': float(edge_evidence[y, x]/edge_area),
            'exclusive_mass': float(visible[y, x]),
            'explained_residual_mass': float(explained[y, x]),
            'foreground_marker_occlusion_fraction': float(peer_occlusion[y, x]),
            'invalid_template_fraction': float(invalid[y, x]),
        }
        supported = (metrics['residual_gain_fraction'] >= cfg.supported_minimum_gain_fraction
                     and metrics['visible_support_fraction'] >= cfg.supported_minimum_visible_fraction
                     and metrics['missing_unoccluded_fraction'] <= cfg.supported_maximum_missing_fraction
                     and fragment_quadrants >= 2)
        candidates.append({
            'id': f'Q{number:03d}', 'x': float(x), 'y': float(y),
            'diameter': diameter, 'existence': 'unknown',
            'evidence_supported': bool(supported),
            'evidence_status': 'partial_shape_support' if supported else 'shape_ambiguous',
            'score': rank, 'metrics': metrics,
            'path_distance_px': float(distance[y, x]),
            'fragment_quadrants': int(fragment_quadrants),
            'quadrant_residual_mass': quadrants,
            'nuisance_note': 'H0 uses saved observed line nuisance; curved lines and short caps can remain. No candidate is a confirmed marker.',
        })
    fields = {'score': score, 'gain': gain/area, 'missing': missing/area,
              'visible': visible/area, 'occlusion': occluded/area,
              'excess': excess/area, 'eligibility': eligible, 'maxima': maxima,
              'exclusive_target': target, 'line': line, 'residual': residual,
              'other_occluder': occluder, 'peer_marker_occluder': occluder*marker_footprints}
    record = {'candidates': candidates, 'config': asdict(cfg),
              'counts': {'eligible_centres': int(eligible.sum()),
                         'translation_maxima_components': int(count),
                         'before_nms': len(pool), 'after_nms': len(candidates),
                         'evidence_supported': sum(p['evidence_supported'] for p in candidates)},
              'method': 'native silhouette translations; asymmetric H0-line vs H1-line-plus-marker residual',
              'confirmed_markers': 0, 'template_scale': 1.0,
              'notes': ['All hypotheses require exclusive target-colour residual; other colour is missing-pixel relief only.',
                        'Path is a corridor, not positive existence evidence; shape refinement never forces a sampling grid.',
                        'One observed straight-line nuisance model is a limited H0, not all possible curve or error-bar explanations.',
                        'No ideal shape reconstruction, automatic scaling, manual plot anchors, or data reference points.']}
    return record, fields

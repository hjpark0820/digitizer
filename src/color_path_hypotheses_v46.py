# Production snapshot of experiments/color_path_v45/path_marker_hypotheses.py.
# Algorithms are local to src; no experimental runtime dependency.
"""Image-backed *possible* marker locations, optionally ranked by a colour path.

This experiment does not accept detections or convert image positions to data.
The colour path is a fitted-curve hypothesis, not a measured-marker centreline.
Image proposals are therefore generated first, with an intentionally broad path
corridor used only for filtering and reranking. No peer-series sampling grid,
manual marker anchors, fixed marker count, or cross-gap path interpolation is
used. All returned locations retain ``existence='unknown'``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    """Shared settings fixed before running the real Panel C comparison."""

    line_length_diameters: float = 2.2
    line_orientations: int = 12
    support_radius_diameters: float = 0.52
    score_smoothing_diameters: float = 0.10
    prominence_radius_diameters: float = 1.4
    nms_radius_diameters: float = 0.62
    min_image_score: float = 0.010
    min_prominence: float = 0.006
    min_exclusive_support: float = 0.006
    min_target_support: float = 0.020
    min_compact_support: float = 0.010
    substantial_thickness_ratio: float = 0.105
    substantial_compact_support: float = 0.035
    substantial_exclusive_support: float = 0.025
    substantial_prominence: float = 0.010
    substantial_residual_aspect_ratio: float = 0.15
    path_corridor_diameters: float = 2.5
    path_sigma_diameters: float = 1.2
    weak_path_reliability: float = 0.55
    path_rank_floor: float = 0.30
    max_hypotheses: int = 40


def _disk(radius: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy <= max(radius, 1.0) ** 2).astype(np.float32)


def _mean(field: np.ndarray, radius: float) -> np.ndarray:
    kernel = _disk(radius)
    kernel /= kernel.sum()
    # Invalid regions stay zero and are not renormalised away. A clipped window
    # contains less evidence, rather than treating excluded legend ink as known.
    return cv2.filter2D(field, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def line_nuisance(signal: np.ndarray, diameter: float,
                  cfg: Config = Config()) -> np.ndarray:
    """Long straight runs in any orientation; a soft nuisance estimate only.

    Grayscale opening preserves image-valued long runs. The maximum across
    directions explains both arms of a crossing. Short caps or curved strokes
    can remain in the residual, so residual pixels are NOT accepted as markers.
    """
    radius = max(2, int(np.ceil(cfg.line_length_diameters * diameter / 2)))
    result = np.zeros_like(signal)
    for angle in np.linspace(0, np.pi, cfg.line_orientations, endpoint=False):
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
        dx, dy = int(round(radius * np.cos(angle))), int(round(radius * np.sin(angle)))
        cv2.line(kernel, (radius - dx, radius - dy),
                 (radius + dx, radius + dy), 1, 1)
        opened = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel,
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
        result = np.maximum(result, opened)
    return np.minimum(signal, result)


def path_proximity(shape, valid, path_local, path_filled, path_observed,
                   diameter, cfg=Config()):
    """Exact Euclidean distance to retained fractional path SAMPLES only.

    Rejected runs never become connecting segments. A retained endpoint can
    still lend broad contextual support nearby; that does not generate ink or
    certify a marker in the gap. No path means zero guidance, not a fake path.
    """
    path = np.asarray(path_local, dtype=np.float64)
    filled = np.asarray(path_filled, dtype=bool)
    observed = np.asarray(path_observed, dtype=bool)
    if path.ndim != 2 or path.shape[1] != 2 or filled.shape != (len(path),) or observed.shape != filled.shape:
        raise ValueError('path must be Nx2 with matching filled/observed vectors')
    distance = np.full(shape, np.inf, np.float32)
    reliability = np.zeros(shape, np.float32)
    state = np.zeros(shape, np.uint8)  # 0 none, 1 weak nearest sample, 2 observed.
    eligible = filled & np.isfinite(path).all(axis=1)
    if len(path):
        rounded = np.rint(np.nan_to_num(path, nan=-1, posinf=-1, neginf=-1)).astype(int)
        inside = ((path[:, 0] >= 0) & (path[:, 0] <= shape[1] - 1)
                  & (path[:, 1] >= 0) & (path[:, 1] <= shape[0] - 1))
        good = np.flatnonzero(eligible & inside)
        eligible[:] = False
        eligible[good] = valid[rounded[good, 1], rounded[good, 0]]
    points = path[eligible]
    if len(points):
        yy, xx = np.indices(shape)
        dist, nearest = cKDTree(points).query(np.column_stack((xx.ravel(), yy.ravel())))
        distance = dist.reshape(shape).astype(np.float32)
        obs = observed[eligible][nearest].reshape(shape)
        reliability = np.where(obs, 1.0, cfg.weak_path_reliability).astype(np.float32)
        state = np.where(obs, 2, 1).astype(np.uint8)
    weight = reliability * np.exp(-0.5 * (distance / (cfg.path_sigma_diameters * diameter)) ** 2)
    corridor = (distance <= cfg.path_corridor_diameters * diameter) & valid
    weight[~corridor] = 0
    state[~corridor] = 0
    return distance, weight.astype(np.float32), state


def _sample(field, x, y):
    return float(ndimage.map_coordinates(field.astype(np.float64), [[y], [x]],
                                         order=1, mode='nearest')[0])


def _residual_aspect(residual, own, x, y, radius):
    """Minor/major weighted covariance ratio of localized exclusive residual.

    A thick error-bar stem can inflate whole-mask thickness at a short cap.
    The cap's remaining horizontal residual is still one dimensional; this
    check keeps that hypothesis ambiguous without changing its centre/rank.
    """
    x0, x1 = max(0, int(np.floor(x - radius))), min(residual.shape[1], int(np.ceil(x + radius)) + 1)
    y0, y1 = max(0, int(np.floor(y - radius))), min(residual.shape[0], int(np.ceil(y + radius)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weight = residual[y0:y1, x0:x1] * own[y0:y1, x0:x1]
    weight = weight * ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2)
    mass = float(weight.sum())
    if mass <= 1e-8:
        return 0.0
    dx = xx - float((weight * xx).sum() / mass)
    dy = yy - float((weight * yy).sum() / mass)
    covariance = np.array([[(weight * dx * dx).sum(), (weight * dx * dy).sum()],
                           [(weight * dx * dy).sum(), (weight * dy * dy).sum()]]) / mass
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(max(0, eigenvalues[0]) / max(1e-8, eigenvalues[1]))


def _image_modes(fields, valid, diameter, cfg):
    """Localized image maxima, plateau deduplication, then spatial suppression.

    The image-only NMS is two dimensional, retaining separate vertical modes.
    Flat extended ridges are not subdivided into an artificial marker sequence.
    """
    score = fields['image_score']
    r = max(1, int(round(0.28 * diameter)))
    maxima = score >= ndimage.maximum_filter(score, size=2 * r + 1, mode='constant') - 1e-7
    eligible = (valid & maxima & (score >= cfg.min_image_score)
                & (fields['prominence'] >= cfg.min_prominence)
                & (fields['exclusive_support'] >= cfg.min_exclusive_support)
                & (fields['target_support'] >= cfg.min_target_support)
                & (fields['compact_support'] >= cfg.min_compact_support))
    labels, count = ndimage.label(eligible, np.ones((3, 3), bool))
    proposals = []
    flat_ridges = 0
    for label, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        local_y, local_x = np.nonzero(labels[sl] == label)
        yy, xx = local_y + sl[0].start, local_x + sl[1].start
        if max(np.ptp(xx), np.ptp(yy)) > 2 * diameter:
            flat_ridges += 1
            continue
        weights = score[yy, xx].astype(float)
        x, y = float(np.average(xx, weights=weights)), float(np.average(yy, weights=weights))
        # Keep the candidate on valid ROI pixels when a plateau borders a hole.
        if not valid[int(round(y)), int(round(x))]:
            best = np.argmax(weights)
            x, y = float(xx[best]), float(yy[best])
        proposals.append((x, y, _sample(score, x, y)))
    proposals.sort(key=lambda p: (-p[2], p[1], p[0]))
    retained = []
    for proposal in proposals:
        x, y, _ = proposal
        if all((x - px) ** 2 + (y - py) ** 2 >= (cfg.nms_radius_diameters * diameter) ** 2
               for px, py, _ in retained):
            retained.append(proposal)
    return retained, {'local_maximum_components': int(count),
                      'flat_ridges_ignored': int(flat_ridges),
                      'before_nms': len(proposals), 'after_nms': len(retained)}


def analyze_series(soft, own, valid, path_local, path_filled, path_observed,
                   diameter: float, cfg: Config = Config()):
    """Return JSON-ready hypotheses and interpretable, crop-local image fields.

    ``image_only`` saves the complete proposal pool, and ``path_guided`` filters
    and reranks that SAME pool before applying the output limit. Distant ink
    cannot consume the guided quota. The scores and region sizes are heuristics,
    not calibrated probabilities or statistical confidence intervals.
    """
    soft = np.asarray(soft, dtype=np.float32)
    own = np.asarray(own, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if soft.ndim != 2 or own.shape != soft.shape or valid.shape != soft.shape:
        raise ValueError('soft, own and valid must be same-shape 2D arrays')
    if not np.isfinite(soft).all() or (soft < 0).any() or (soft > 1).any():
        raise ValueError('soft values must be finite and in [0,1]')
    if not np.isfinite(diameter) or diameter < 2:
        raise ValueError('diameter must be finite and at least 2 pixels')
    if cfg.line_orientations < 1 or cfg.max_hypotheses < 1:
        raise ValueError('line_orientations and max_hypotheses must be positive')
    signal = soft * valid
    exclusive = signal * own
    line = line_nuisance(signal, diameter, cfg)
    residual = np.maximum(signal - line, 0)
    radius = cfg.support_radius_diameters * diameter
    target = _mean(signal, radius)
    compact = _mean(residual, radius)
    line_support = _mean(line, radius)
    own_support = _mean(exclusive, radius)
    own_fraction = np.clip(own_support / np.maximum(target, 1e-6), 0, 1)
    # Thickness is a nuisance diagnostic, not a hard marker existence test.
    # The exclusive mask avoids borrowing another colour's thick marker body.
    thickness = ndimage.distance_transform_edt(own & valid).astype(np.float32)
    thickness = ndimage.maximum_filter(thickness, footprint=_disk(radius).astype(bool), mode='constant')
    thickness_ratio = thickness / diameter
    thickness_factor = np.clip(thickness_ratio / 0.16, 0, 1)
    raw = compact * (0.35 + 0.65 * thickness_factor) * (0.40 + 0.60 * own_fraction)
    score = ndimage.gaussian_filter(raw, sigma=max(0.5, cfg.score_smoothing_diameters * diameter), mode='constant')
    score *= valid
    prominence = np.maximum(score - _mean(score, cfg.prominence_radius_diameters * diameter), 0)
    distance, path_weight, path_state = path_proximity(soft.shape, valid, path_local,
                                                     path_filled, path_observed, diameter, cfg)
    guided = score * (cfg.path_rank_floor + (1 - cfg.path_rank_floor) * path_weight)
    guided[path_state == 0] = 0
    fields = {'signal': signal, 'line': line, 'residual': residual,
              'target_support': target, 'line_support': line_support,
              'exclusive_support': own_support, 'compact_support': compact,
              'thickness_ratio': thickness_ratio, 'image_score': score,
              'prominence': prominence, 'path_distance': distance,
              'path_weight': path_weight, 'path_state': path_state,
              'guided_score': guided}
    proposals, counts = _image_modes(fields, valid, diameter, cfg)
    hypotheses = []
    valid_radius = ndimage.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] - np.sqrt(0.5)
    for i, (x, y, image_score) in enumerate(proposals, 1):
        factors = {name: _sample(fields[name], x, y)
                   for name in ['target_support', 'exclusive_support', 'compact_support',
                                'line_support', 'thickness_ratio', 'prominence', 'path_weight']}
        factors['residual_aspect_ratio'] = _residual_aspect(residual, own & valid, x, y, radius)
        ix, iy = int(round(x)), int(round(y))
        state_code = int(path_state[iy, ix])
        state_name = {0: 'none', 1: 'weak', 2: 'observed'}[state_code]
        dpath = float(distance[iy, ix])
        substantial = (factors['thickness_ratio'] >= cfg.substantial_thickness_ratio
                       and factors['compact_support'] >= cfg.substantial_compact_support
                       and factors['exclusive_support'] >= cfg.substantial_exclusive_support
                       and factors['prominence'] >= cfg.substantial_prominence
                       and factors['residual_aspect_ratio'] >= cfg.substantial_residual_aspect_ratio)
        reasons = ['localized target-colour residual; marker existence unverified']
        if not substantial:
            reasons.append('limited compact/exclusive/thickness evidence; a cap, curved line or occluded marker remains possible')
        if state_code == 0:
            reasons.append('outside retained-path corridor, or no retained path')
        elif state_code == 1:
            reasons.append('nearest retained path sample has weak colour evidence')
        else:
            reasons.append('near retained colour-supported path sample, not necessarily its measured centre')
        nominal_radius = diameter * (0.45 if substantial else 0.70)
        region_radius = min(nominal_radius, max(0.0, float(valid_radius[iy, ix]) - np.hypot(x - ix, y - iy)))
        hypotheses.append({'id': f'H{i:03d}', 'x': x, 'y': y,
                           'existence': 'unknown',
                           'kind': 'image_backed_possible' if substantial else 'line_or_occlusion_ambiguous',
                           'image_score': image_score,
                           'guided_score': _sample(guided, x, y),
                           'rank_factors': factors,
                           'path_distance_px': dpath if np.isfinite(dpath) else None,
                           'path_state': state_name,
                           'possible_center_region': {'cx': x, 'cy': y, 'rx': region_radius,
                                                      'ry': region_radius,
                                                      'nominal_radius_px': nominal_radius,
                                                      'roi_clipped': region_radius < nominal_radius,
                                                      'roi_rule': 'contained_in_valid_roi; excluded legend is not a possible centre',
                                                      'basis': 'heuristic_not_confidence_interval'},
                           'reasons': reasons})
    path_selected = sorted((h for h in hypotheses if h['path_state'] != 'none'),
                           key=lambda h: (-h['guided_score'], -h['image_score'], h['y'], h['x']))
    counts.update({'image_only_total': len(hypotheses), 'path_guided_total': len(path_selected),
                   'image_only_returned': len(hypotheses),
                   'path_guided_returned': min(len(path_selected), cfg.max_hypotheses),
                   'image_only_truncated': 0,
                   'image_only_display_limit': cfg.max_hypotheses,
                   'path_guided_truncated': max(0, len(path_selected) - cfg.max_hypotheses),
                   'exclusive_pixel_count': int(np.count_nonzero(own & valid)),
                   'retained_path_samples': int(np.count_nonzero(path_filled))})
    record = {'diameter': float(diameter), 'config': asdict(cfg), 'counts': counts,
              'image_only': hypotheses,
              # Separate dicts make caller source-coordinate conversion safe.
              'path_guided': deepcopy(path_selected[:cfg.max_hypotheses]),
              'warnings': ['All outputs are uncalibrated location hypotheses, not confirmed marker detections.',
                           'Curve/marker centres can differ; the path is only broad contextual evidence.',
                           'Fully hidden markers with no localized exclusive colour evidence cannot be proposed.',
                           'Line, error-bar cap and partially occluded marker ambiguity is intentionally retained.']}
    return record, fields

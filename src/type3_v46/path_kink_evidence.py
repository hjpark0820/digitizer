"""Multiscale bend evidence on an uncut, x-ordered path.

This module proposes geometric kinks, NOT measured data points. A continuous
hinge is compared with a straight line and a quadratic: the latter prevents
ordinary smooth curvature from becoming a dense set of putative measurements.
Inferred samples are retained and reported, not silently dropped or promoted to
observed ink. No source image, marker schedule, or numerical reference is read.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = "uncut_path_multiscale_kinks_v1"


def _fit_window(x, y, center, half_width, width, observed, spacing):
    indices = np.flatnonzero(np.abs(x - center) <= half_width + 1e-9)
    if len(indices) < 9:
        return None
    dx = x[indices] - center
    if (dx < 0).sum() < 4 or (dx > 0).sum() < 4:
        return None
    # Do not extrapolate at endpoints, or invent samples across missing x spans.
    if -dx.min() < .8 * half_width or dx.max() < .8 * half_width:
        return None
    if np.max(np.diff(x[indices])) > max(3 * spacing, 2 * width):
        return None
    t = dx / half_width
    yy = y[indices]
    designs = [np.column_stack([np.ones(len(t)), t]),
               np.column_stack([np.ones(len(t)), t, np.maximum(t, 0)]),
               np.column_stack([np.ones(len(t)), t, t * t])]
    coeff = [np.linalg.lstsq(a, yy, rcond=None)[0] for a in designs]
    rms = [float(np.sqrt(np.mean((yy - a @ b) ** 2))) for a, b in zip(designs, coeff)]
    straight, hinge, quadratic = rms
    left = float(coeff[1][1] / half_width)
    right = float((coeff[1][1] + coeff[1][2]) / half_width)
    turn = abs(math.atan(right) - math.atan(left))
    gain = straight - hinge
    energy_gain = (straight ** 2 - hinge ** 2) / max((.05 * width) ** 2, straight ** 2)
    quadratic_advantage = (quadratic - hinge) / max(.12 * width, quadratic)
    displacement = abs(right - left) * half_width / 4
    requirements = dict(gain=gain >= .15 * width, energy_gain=energy_gain >= .55,
                        bend_displacement=displacement >= .55 * width,
                        turn_angle=turn >= .08, better_than_smooth_quadratic=quadratic_advantage >= .16)
    requirements = {key: bool(value) for key, value in requirements.items()}
    supported = all(requirements.values())
    # Engineering score only: bounded evidence ratios, not calibrated probability.
    score = float(np.mean([min(1., gain / (.6 * width)), min(1., energy_gain),
                           min(1., displacement / (1.5 * width)),
                           min(1., turn / .35), min(1., max(0., quadratic_advantage))])) if supported else 0.
    left_obs = observed[indices[dx < 0]]
    right_obs = observed[indices[dx > 0]]
    return dict(half_width=float(half_width), x=float(center),
                fitted_y=float(coeff[1][0]), left_slope=left, right_slope=right,
                slope_change=right-left, turn_angle_degrees=math.degrees(turn),
                straight_rms=straight, hinge_rms=hinge, quadratic_rms=quadratic,
                rms_gain=gain, energy_gain=float(energy_gain),
                quadratic_advantage=float(quadratic_advantage),
                bend_displacement=float(displacement), supported=bool(supported),
                geometric_score=score, gates=requirements,
                observed_fraction=float(observed[indices].mean()),
                observed_fraction_left=float(left_obs.mean()),
                observed_fraction_right=float(right_obs.mean()),
                sample_count=int(len(indices)), source_index_span=[int(indices[0]), int(indices[-1])])


def _peaks(rows, x, radius):
    """Nonmaximum suppression per scale; plateaus select their center."""
    available = {i for i, row in enumerate(rows) if row and row['supported']}
    peaks = []
    while available:
        best_score = max(rows[i]['geometric_score'] for i in available)
        tied = sorted(i for i in available if abs(rows[i]['geometric_score'] - best_score) < 1e-10)
        # Distant equally strong corners must not average to a nonexistent corner.
        first = tied[0]
        cluster = [i for i in tied if x[i] - x[first] <= radius]
        best = cluster[len(cluster)//2]
        peaks.append(best)
        available = {i for i in available if abs(x[i]-x[best]) > radius}
    return peaks


def analyze_kinks(path_xy, line_width, observed=None):
    """Return geometric candidates and every per-x fit without changing the path.

    ``path_xy`` must be finite N-by-2, strictly increasing in native pixel x.
    ``observed`` is an optional boolean N-vector. False samples participate in
    fits equally but the output explicitly identifies their inferred support.
    Windows have full widths 16/24/32 times ``line_width``. Endpoint samples and
    large missing-x gaps lack two-sided evidence and are not forced to be kinks.
    All coordinates remain input plot-local. Index spans are inclusive.
    """
    xy = np.asarray(path_xy, dtype=float)
    width = float(line_width)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('path_xy must be a finite N-by-2 array')
    if not np.isfinite(width) or width <= 0:
        raise ValueError('line_width must be finite and positive')
    width = max(1., width)
    x, y = xy[:, 0], xy[:, 1]
    if len(x) > 1 and np.any(np.diff(x) <= 0):
        raise ValueError('path x must be strictly increasing; duplicate columns are ambiguous')
    if observed is None:
        obs = np.ones(len(x), dtype=bool)
        observation_provenance = 'unspecified; caller default is all observed'
    else:
        obs = np.asarray(observed)
        if obs.shape != (len(x),) or obs.dtype.kind != 'b':
            raise ValueError('observed must be a boolean N-vector')
        observation_provenance = 'caller-supplied observed/inferred flags'
    half_widths = np.asarray([8., 12., 16.]) * width
    config = dict(version=VERSION, effective_line_width=width,
                  full_window_widths=(2*half_widths).tolist(),
                  coordinate_system='input native plot-local pixels',
                  confidence_interpretation='geometric evidence score, not probability',
                  observation_provenance=observation_provenance,
                  strong_requires_scales=3, minimum_proposal_scales=2,
                  location_tolerance=3*width, final_minimum_separation=5*width,
                  fit_gates=dict(minimum_rms_gain=.15*width, minimum_energy_gain=.55,
                                 minimum_bend_displacement=.55*width,
                                 minimum_turn_radians=.08, minimum_quadratic_advantage=.16,
                                 minimum_strong_slope_consistency=.60),
                  input_path_modified=False,
                  limitations=['A bend is not proof of a measurement.',
                               'A smooth fitted curve can conceal actual measurement positions.',
                               'Raster or DP artifacts can mimic bends; source ink verification remains necessary.',
                               'Very short segments or endpoint measurements lack multiscale support.'])
    spacing = float(np.median(np.diff(x))) if len(x) > 1 else 1.
    scales = [[_fit_window(x, y, v, h, width, obs, spacing) for v in x] for h in half_widths]
    per_x = [dict(index=i, x=float(x[i]), y=float(y[i]), observed=bool(obs[i]),
                  scales=[scale[i] for scale in scales]) for i in range(len(x))]
    peaks = [_peaks(rows, x, max(2*width, .65*h)) for rows, h in zip(scales, half_widths)]
    proposals = []
    for locations in peaks:
        for index in locations:
            matched = []
            for other_scale, options in enumerate(peaks):
                near = [i for i in options if abs(x[i]-x[index]) <= 3*width]
                if near:
                    best = min(near, key=lambda i: (abs(x[i]-x[index]), -scales[other_scale][i]['geometric_score']))
                    matched.append((other_scale, best))
            if len(matched) < 2:
                continue
            rows = [scales[s][i] for s, i in matched]
            signs = np.sign([q['slope_change'] for q in rows])
            if not np.all(signs == signs[0]):
                continue
            # Smallest window localizes a sharp bend; larger windows confirm it.
            smallest_scale, chosen = min(matched)
            center = float(x[chosen])
            deltas = np.abs([q['slope_change'] for q in rows])
            slope_consistency = float(deltas.min()/max(1e-12, deltas.max()))
            spread = float(np.ptp([x[i] for _, i in matched]))
            score = float(np.mean([q['geometric_score'] for q in rows]) * len(rows)/3 * slope_consistency)
            strong = len(rows) == 3 and spread <= 3*width and slope_consistency >= .60
            obs_fraction = float(np.mean([q['observed_fraction'] for q in rows]))
            representative = scales[smallest_scale][chosen]
            proposals.append(dict(index=int(chosen), x=center, y=float(y[chosen]),
                fitted_y=representative['fitted_y'], strength='strong' if strong else 'tentative',
                geometric_score=score, scale_support=len(rows), location_spread=spread,
                left_slope=representative['left_slope'], right_slope=representative['right_slope'],
                slope_consistency=slope_consistency, observed_fraction=obs_fraction,
                center_observed=bool(obs[chosen]),
                evidence_kind='observed_geometry' if obs_fraction >= .8 and obs[chosen] else 'includes_inferred_geometry',
                marker_glyph_detected=False, measurement_confirmed=False,
                per_scale=rows))
    kept = []
    for candidate in sorted(proposals, key=lambda q: (q['strength']=='strong', q['scale_support'], q['geometric_score']), reverse=True):
        if all(abs(candidate['x']-q['x']) >= 5*width for q in kept):
            kept.append(candidate)
    kept.sort(key=lambda q:q['x'])
    for i, candidate in enumerate(kept, 1):
        candidate['id'] = f'K{i:03d}'
    return dict(candidates=kept, per_x=per_x, config=config)

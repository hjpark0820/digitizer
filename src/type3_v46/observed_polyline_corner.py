"""Refine a polyline corner from observed source pixels, not path samples.

The supplied path only limits the search corridor. All line/model fits use
actual same-series mask pixels. Scores and uncertainty bounds are engineering
diagnostics, not probabilities or statistical confidence intervals.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = 'observed_polyline_corner_v1'


def _source_columns(path, center, reach, own, soft, valid, width):
    h, w = own.shape
    radius = max(3, int(math.ceil(2*width)))
    left = max(0, int(math.ceil(max(path[0, 0], center-reach))))
    right = min(w-1, int(math.floor(min(path[-1, 0], center+reach))))
    points = []
    for col in range(left, right+1):
        expected = float(np.interp(col, path[:, 0], path[:, 1]))
        ya = max(0, int(math.floor(expected-radius)))
        yb = min(h, int(math.ceil(expected+radius))+1)
        yy = np.arange(ya, yb)
        if not len(yy):
            continue
        hit = own[yy, col] & valid[yy, col] & (soft[yy, col] >= .18)
        rows = yy[hit]
        if not len(rows):
            continue
        # A second distant branch must not be averaged into blank space.
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1)+1)
        run = min(runs, key=lambda r: abs(float(np.median(r))-expected))
        if len(run) > max(5, int(math.ceil(3*width))):
            continue
        mass = soft[run, col]
        points.append([float(col), float(np.average(run, weights=mass)), float(mass.max())])
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _robust_line(points, origin, width):
    if len(points) < 5:
        return None
    t = points[:, 0]-origin
    design = np.column_stack([np.ones(len(t)), t])
    base = np.clip(points[:, 2], .01, 1.)
    weight = base.copy()
    coef = np.zeros(2)
    for _ in range(5):
        coef = np.linalg.lstsq(design*np.sqrt(weight[:, None]), points[:, 1]*np.sqrt(weight), rcond=None)[0]
        residual = points[:, 1]-design@coef
        sigma = max(.20*width, 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weight = base*np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-12))
    residual = points[:, 1]-design@coef
    inliers = np.abs(residual) <= max(.65*width, 2.5*sigma)
    rms = float(np.sqrt(np.average(residual**2, weights=weight)))
    return dict(slope=float(coef[1]), y=float(coef[0]), reference_x=float(origin), rms=rms,
                inlier_fraction=float(inliers.mean()), source_columns=int(len(points)),
                x_span=float(np.ptp(points[:, 0])), robust_sigma=sigma)


def _intersection(left, right, origin):
    delta = left['slope']-right['slope']
    if abs(delta) < 1e-9:
        return None
    dx = (right['y']-left['y'])/delta
    return dict(x=float(origin+dx), y=float(left['y']+left['slope']*dx))


def _models(points, x, reach, width):
    t = (points[:, 0]-x)/reach
    yy = points[:, 1]
    weights = np.sqrt(np.clip(points[:, 2], .01, 1.))
    matrices = dict(straight=np.column_stack([np.ones(len(t)), t]),
                    hinge=np.column_stack([np.ones(len(t)), t, np.maximum(t, 0)]),
                    quadratic=np.column_stack([np.ones(len(t)), t, t*t]))
    scores = {}
    for name, design in matrices.items():
        coef = np.linalg.lstsq(design*weights[:, None], yy*weights, rcond=None)[0]
        scores[name+'_rms'] = float(np.sqrt(np.average((yy-design@coef)**2, weights=weights**2)))
    s, h, q = scores['straight_rms'], scores['hinge_rms'], scores['quadratic_rms']
    scores.update(energy_gain=float((s*s-h*h)/max((.05*width)**2, s*s)),
                  rms_gain=s-h, quadratic_advantage=(q-h)/max(.12*width, q))
    return scores


def _window(path, proposed_x, reach, own, soft, valid, width):
    points = _source_columns(path, proposed_x, reach, own, soft, valid, width)
    result = dict(reach=float(reach), source_points=points.tolist(), status='rejected', reason=None,
                  left=dict(points=[], fit=None), right=dict(points=[], fit=None),
                  intersection=None, models=None, gates={})
    guard = max(2., .75*width)
    current = float(proposed_x)
    for _ in range(3):
        lp = points[points[:, 0] <= current-guard]
        rp = points[points[:, 0] >= current+guard]
        lf, rf = _robust_line(lp, current, width), _robust_line(rp, current, width)
        result.update(left=dict(points=lp.tolist(), fit=lf), right=dict(points=rp.tolist(), fit=rf))
        if lf is None or rf is None:
            result['reason'] = 'insufficient_observed_columns_on_both_sides'
            return result
        angle = abs(math.degrees(math.atan(rf['slope'])-math.atan(lf['slope'])))
        result['turn_degrees'] = angle
        if angle < 8.:
            result['reason'] = 'near_parallel_or_insufficient_turn'
            return result
        cross = _intersection(lf, rf, current)
        result['intersection'] = cross
        if cross is None or not np.isfinite([cross['x'], cross['y']]).all():
            result['reason'] = 'undefined_intersection'
            return result
        if abs(cross['x']-proposed_x) > 4*width:
            result['reason'] = 'intersection_exceeds_bounded_x_shift'
            return result
        if abs(cross['x']-current) < .1*width:
            break
        current = cross['x']
    # The final fit records retain their reference_x; the intersection is truly
    # two-dimensional, not the average of two y intercepts at a fixed x.
    cross = result['intersection']
    if not path[0, 0] <= cross['x'] <= path[-1, 0]:
        result['reason'] = 'intersection_outside_input_path'
        return result
    py = float(np.interp(cross['x'], path[:, 0], path[:, 1]))
    xi, yi = int(round(cross['x'])), int(round(cross['y']))
    if not (0 <= yi < own.shape[0] and 0 <= xi < own.shape[1] and valid[yi, xi]):
        result['reason'] = 'intersection_outside_valid_plot'
        return result
    model_points = np.concatenate([lp, rp])
    models = _models(model_points, cross['x'], reach, width)
    fitted_origin = lf['reference_x']
    left_coverage = len(lp)/max(1., fitted_origin-max(path[0, 0], proposed_x-reach)-guard+1)
    right_coverage = len(rp)/max(1., min(path[-1, 0], proposed_x+reach)-fitted_origin-guard+1)
    result['left']['coverage'] = float(min(1., left_coverage))
    result['right']['coverage'] = float(min(1., right_coverage))
    angle = result['turn_degrees']
    gates = dict(clean_straight_arms=max(lf['rms'], rf['rms']) <= max(.6, .75*width),
                 mostly_inlier_arms=min(lf['inlier_fraction'], rf['inlier_fraction']) >= .8,
                 enough_observed_coverage=min(left_coverage, right_coverage) >= .40,
                 sufficient_arm_span=min(lf['x_span'], rf['x_span']) >= .45*reach,
                 small_center_gap=max(cross['x']-lp[:, 0].max(), rp[:, 0].min()-cross['x']) <= 2*width+guard,
                 path_corridor=abs(cross['y']-py) <= 3*width,
                 clear_straight_model_gain=models['energy_gain'] >= .60 and models['rms_gain'] >= .15*width,
                 hinge_beats_smooth_quadratic=models['quadratic_advantage'] >= .12,
                 low_hinge_residual=models['hinge_rms'] <= max(.7, .85*width))
    gates = {k: bool(v) for k, v in gates.items()}
    conditioning = abs(lf['slope']-rf['slope'])
    x_unc = max(.5*width, 2*math.hypot(lf['rms'], rf['rms'])/max(conditioning, 1e-9))
    y_unc = max(.5*width, max(abs(lf['slope']), abs(rf['slope']))*x_unc,
                lf['rms'], rf['rms'])
    result.update(models=models, gates=gates, shift_x=cross['x']-proposed_x,
                  path_y_at_intersection=py, geometry_uncertainty=dict(x=x_unc, y=y_unc),
                  conditioning_slope_difference=conditioning,
                  status='supported' if all(gates.values()) else 'rejected',
                  reason='observed_hinge_beats_straight_and_smooth' if all(gates.values()) else 'source_geometry_gate_failed')
    return result


def refine_observed_corner(path_xy, x, own, soft, valid, line_width):
    """Return source-only corner evidence; never a detected marker glyph.

    ``path_xy`` is finite x-ordered N-by-2 in plot-local native pixels. Masks
    are same-sized bool H-by-W; soft is a finite [0,1] source-color field. The
    output x/y stays in the same coordinate system, including rejected cases.
    Renderer-ready per_window left/right ``points`` are [x,y,weight]; their
    ``fit.y`` is at ``fit.reference_x``, not necessarily at the original x.
    """
    path = np.asarray(path_xy, dtype=float)
    om, vm, sm = np.asarray(own), np.asarray(valid), np.asarray(soft, dtype=float)
    width, initial_x = float(line_width), float(x)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all() or np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError('path_xy must be finite, strictly x-ordered, and have at least two points')
    if om.ndim != 2 or vm.shape != om.shape or sm.shape != om.shape or om.dtype.kind != 'b' or vm.dtype.kind != 'b':
        raise ValueError('own and valid must be same-sized boolean masks; soft must match')
    if not np.isfinite(sm).all() or np.any((sm < 0) | (sm > 1)):
        raise ValueError('soft must be finite and in [0,1]')
    if not np.isfinite([width, initial_x]).all() or width <= 0 or not path[0, 0] <= initial_x <= path[-1, 0]:
        raise ValueError('line_width must be positive; x must lie on the supplied path')
    width = max(1., width)
    windows = [_window(path, initial_x, r*width, om, sm, vm, width) for r in [8., 12., 16.]]
    initial_y = float(np.interp(initial_x, path[:, 0], path[:, 1]))
    result = dict(version=VERSION, status='rejected', reason='no_multiscale_observed_corner',
                  x=initial_x, y=initial_y, proposed_x=initial_x, proposed_y=initial_y,
                  geometric_score=0., score_is_probability=False, marker_glyph_detected=False,
                  source_points=windows[-1]['source_points'], per_window=windows,
                  geometry_uncertainty=None, supported_windows=0, location_spread=None,
                  config=dict(line_width=width, reaches=[8*width, 12*width, 16*width],
                              minimum_turn_degrees=8., maximum_x_shift=4*width,
                              maximum_path_y_deviation=3*width, minimum_supported_windows=2,
                              coordinate_system='native plot-local pixels',
                              path_used_as_positive_evidence=False),
                  limitations=['A visible corner is not proof that a measurement was taken there.',
                               'Endpoint and fully occluded corners require a separate evidence policy.',
                               'The path corridor can inherit an upstream wrong-series path.'])
    good = [w for w in windows if w['status'] == 'supported']
    if not good:
        return result
    xx = np.asarray([w['intersection']['x'] for w in good])
    yy = np.asarray([w['intersection']['y'] for w in good])
    result.update(x=float(np.median(xx)), y=float(np.median(yy)))
    directions = np.sign([w['right']['fit']['slope']-w['left']['fit']['slope'] for w in good])
    stable = np.ptp(xx) <= 2*width and np.ptp(yy) <= 2*width and np.all(directions == directions[0])
    ux = max(float(np.ptp(xx)), max(w['geometry_uncertainty']['x'] for w in good))
    uy = max(float(np.ptp(yy)), max(w['geometry_uncertainty']['y'] for w in good))
    result['geometry_uncertainty'] = dict(x=ux, y=uy,
        x_interval=[result['x']-ux, result['x']+ux], y_interval=[result['y']-uy, result['y']+uy],
        interpretation='Geometric fit/scale sensitivity bound; not a statistical confidence interval')
    result['supported_windows'] = len(good)
    result['location_spread'] = dict(x=float(np.ptp(xx)), y=float(np.ptp(yy)))
    result['geometric_score'] = float(len(good)/3 * max(0., 1-min(1., ux/(8*width))))
    if len(good) >= 2 and stable and max(ux, uy) <= 4*width:
        result.update(status='supported', reason='stable_observed_polyline_intersection')
    else:
        result.update(status='tentative', reason='single_scale_or_unstable_intersection')
    return result

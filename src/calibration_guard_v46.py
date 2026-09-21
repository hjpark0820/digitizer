"""Conservative export boundary for native colour v46 axis calibration.

OCR fits are proposals, not confirmed units. Reject unsafe automatic mappings
before any coordinate export. A rejected axis uses local pixels, independently
of the other axis. This module does not guess missing log exponents or repair
old files. Explicit complete manual ranges may intentionally decrease.
"""
import math
from copy import deepcopy

VERSION = 'axis_calibration_guard_v1'


def mapping_problem(mapping, p0, p1, kind, *, automatic, inliers=()):
    if mapping is None:
        return 'mapping_unavailable'
    if not all(math.isfinite(float(p)) for p in (p0, p1)) or p0 == p1:
        return 'invalid_pixel_span'
    try:
        values = [float(mapping(p0 + t * (p1-p0))) for t in (0, .5, 1)]
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return 'invalid_mapping_values'
    if not all(math.isfinite(v) for v in values):
        return 'nonfinite_mapping'
    if kind == 'log' and min(values) <= 0:
        return 'nonpositive_log_values'
    if values[0] == values[2]:
        return 'constant_mapping'
    increasing = values[0] < values[1] < values[2]
    decreasing = values[0] > values[1] > values[2]
    if not increasing and not decreasing:
        return 'nonmonotonic_mapping'
    if automatic and decreasing:
        return 'decreasing_automatic_axis_requires_confirmation'
    if automatic:
        # Two labels cannot independently corroborate an OCR fit.
        try:
            pairs = [(float(p), float(v)) for p, v in inliers
                     if p is not None and v is not None]
        except (TypeError, ValueError, OverflowError):
            return 'invalid_tick_evidence'
        pairs = [(p, v) for p, v in pairs if math.isfinite(p) and math.isfinite(v)]
        if len({p for p, _ in pairs}) < 3 or len({v for _, v in pairs}) < 3:
            return 'insufficient_independent_ticks'
    return None


def guard_calibration(env):
    """Called after manual overrides, before Excel/JSON export; no point edits."""
    pa = env.get('PLOT_AREA')
    if pa is None:
        return
    x0, y0, x1, y1 = map(float, pa)
    if not all(map(math.isfinite, (x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
        raise ValueError('Calibration requires a finite, positive plot rectangle')
    coords = dict(env.get('_coords') or {})
    native = env.get('_cal') or {}
    diagnostics = dict(version=VERSION, axes={})
    for axis, p0, p1 in (('x', x0, x1), ('y', y1, y0)):
        prefix = 'USER_' + axis.upper()
        lo, hi = env.get(prefix+'_MIN'), env.get(prefix+'_MAX')
        log = bool(env.get(prefix+'_LOG'))
        manual = lo is not None or hi is not None or log
        complete = lo is not None and hi is not None
        mapping = env.get('_'+axis+'2v')
        kind = coords.get(axis+'_kind', 'linear')
        inliers = native.get(axis+'_inliers') or []
        problem = None
        if manual:
            if log and kind != 'log':
                problem = 'manual_log_mapping_unavailable'
            # An incomplete override must not inherit an unsafe OCR endpoint.
            if not complete:
                problem = mapping_problem(native.get(axis+'2v'), p0, p1,
                    (native.get('coords') or {}).get(axis+'_kind', 'linear'),
                    automatic=True, inliers=inliers)
                if problem:
                    problem = 'incomplete_manual_range_with_unsafe_ocr'
            try:
                supplied = [float(v) for v in (lo, hi) if v is not None]
                if (not all(map(math.isfinite, supplied)) or
                        (log and any(v <= 0 for v in supplied)) or
                        (complete and float(lo) == float(hi))):
                    problem = 'invalid_manual_range'
            except (TypeError, ValueError, OverflowError):
                problem = 'invalid_manual_range'
        problem = problem or mapping_problem(mapping, p0, p1, kind,
                                             automatic=not manual, inliers=inliers)
        record = dict(status='accepted', source='manual' if manual else 'automatic',
                      kind=kind, warning='', reason=problem, inliers=inliers)
        if problem:
            # Distance from the left/bottom plot edge. Neither normalized data
            # nor inferred physical units; preserves the original orientation.
            direction = 1. if p1 > p0 else -1.
            mapping = lambda p, origin=p0, sign=direction: float(sign*(p-origin))
            kind = 'pixel'
            record.update(status='needs_calibration', kind=kind,
                warning=f'{axis.upper()} axis calibration not confirmed ({problem}); '
                        'showing local pixels. Enter both axis limits and the log setting.')
            print('[v46 calibration] '+record['warning'], flush=True)
        env['_'+axis+'2v'] = mapping
        coords.update({axis+'_min': float(mapping(p0)), axis+'_max': float(mapping(p1)),
                       axis+'_kind': kind})
        diagnostics['axes'][axis] = record
    env['_coords'] = coords
    env['_calibration_diagnostics_v46'] = diagnostics


def annotate_edit_data(ed, env):
    diagnostics = env.get('_calibration_diagnostics_v46')
    if diagnostics:
        ed['calibration_diagnostics'] = deepcopy(diagnostics)
        for axis, record in diagnostics['axes'].items():
            if (ed.get('calibration') or {}).get(axis):
                ed['calibration'][axis].update({k: record[k] for k in ('status', 'source', 'warning')})
    return ed

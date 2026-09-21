"""Image-backed colour marker hypotheses for v46 Step 5 review.

The active hybrid detector is not changed by this module. Native colour fields
are supplied by the caller, not re-estimated here. Directional retained paths,
local residual hypotheses and partial legend-shape explanations produce unknown
candidates. Only existing strong image evidence enters suppressed; ambiguous
candidates remain diagnostic-only. Curve support knots are never exported as points.
All internal maps are plot-local; returned positions/path samples are source
pixels and plot_box/legend_boxes use half-open xyxy coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from color_path_directional_v46 import DirectionalConfig, trace_directional
from color_path_hypotheses_v46 import Config as HypothesisConfig, analyze_series
from color_curve_explanation_v46 import Config as CurveConfig, analyze_explanation, evaluate_addition
from color_shape_occlusion_v46 import Config as ShapeConfig, analyze_shape_candidates, extract_legend_shape
from color_tentative_policy_v46 import strong_evidence, policy_info

VERSION = 'path_shape_tentative_v46'


@dataclass(frozen=True)
class Config:
    support_window: int = 41
    support_density: float = .25
    support_pad: int = 6
    path_observed_threshold: float = .18
    existing_marker_clearance_diameters: float = .60
    duplicate_candidate_clearance_diameters: float = .30
    minimum_curve_benefit: float = .01


def true_runs(values):
    edge = np.diff(np.r_[False, np.asarray(values, bool), False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def _fragment_profile(ok):
    """v45 fragment_profile defaults, with no drawing-type or slope-sign prior."""
    ok = np.asarray(ok, bool)
    rr = []
    if len(ok):
        edges = np.r_[0, np.flatnonzero(ok[1:] != ok[:-1]) + 1, len(ok)]
        rr = [(bool(ok[a]), int(b-a)) for a, b in zip(edges[:-1], edges[1:])]
    dash = np.array([length for value, length in rr if value], float)
    gap = np.array([length for value, length in rr if not value], float)
    duty = float(np.mean(ok)) if len(ok) else 0.
    if not gap.size:
        return dict(kind='solid', duty=duty, period=0., regularity=1., gap_budget=0.,
                    dash_med=float(dash.max() if dash.size else 0))
    periods = np.array([rr[k][1] + rr[k+1][1] for k in range(len(rr)-1) if rr[k][0]], float)
    regularity = (1.-float(np.median(np.abs(periods-np.median(periods))) * 1.4826 /
                          max(np.median(periods), 1.))) if periods.size >= 3 else 0.
    if regularity >= .55 and float(np.median(gap)) <= 5. and duty >= .45:
        kind = 'dashed'
        budget = max(float(np.percentile(gap, 90))*1.5, float(np.median(periods)))
    elif duty >= .90:
        kind = 'solid'
        budget = max(float(np.percentile(gap, 90))*2., 6.)
    else:
        kind = 'sparse'
        budget = float(np.percentile(gap, 75))
    return dict(kind=kind, duty=duty, period=float(np.median(periods)) if periods.size else 0.,
                regularity=regularity, gap_budget=float(np.clip(budget, 3., 40.)),
                dash_med=float(np.median(dash)) if dash.size else 0.)


def support_intervals(own, cfg=Config(), native_helpers=None):
    """Keep all density-supported x intervals, not only the longest span."""
    columns = np.asarray(own, bool).any(0)
    if not columns.any():
        return []
    # numpy.convolve('same') grows output when its kernel is wider than input.
    # Bounding the window only affects plots smaller than v45's 41px default.
    win = min(cfg.support_window, len(columns))
    helper = (native_helpers or {}).get('_edge_norm_density')
    if helper is not None:
        density = np.asarray(helper(columns, win), float)
    else:
        kernel = np.ones(win)
        density = np.convolve(columns.astype(float), kernel, mode='same') / np.maximum(
            np.convolve(np.ones_like(columns, dtype=float), kernel, mode='same'), 1.)
    intervals = []
    for a, b in true_runs(density >= cfg.support_density):
        a, b = max(0, a-cfg.support_pad), min(len(columns), b+cfg.support_pad+1)
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(b, intervals[-1][1])
        else:
            intervals.append([a, b])
    return intervals


def trace_retained_path(soft, own, valid, *, cfg=Config(), native_helpers=None):
    """Directional path with observed vs short-gap vs rejected samples explicit.

    Penalties are symmetric for rising/falling slopes, suitable for PK as well
    as concentration-efficacy charts. Distinct support intervals never connect
    through retained synthetic pixels. Final segmentation must split on filled.
    """
    soft = np.asarray(soft, np.float32) * np.asarray(valid, bool)
    own = np.asarray(own, bool) & np.asarray(valid, bool)
    intervals = support_intervals(own, cfg, native_helpers)
    if not intervals:
        return dict(path=np.empty((0, 2)), val=np.empty(0), filled=np.empty(0, bool),
                    observed=np.empty(0, bool), support_intervals=[], profile={}, status='no_own_colour_evidence')
    result = trace_directional(soft, intervals[0][0], intervals[-1][1]-1,
                               0, soft.shape[0]-1, DirectionalConfig())
    path, values = result['path'], result['val']
    pixels = np.rint(path).astype(int)
    scope = valid[pixels[:, 1], pixels[:, 0]]
    profiler = (native_helpers or {}).get('fragment_profile', _fragment_profile)
    profile = profiler(values >= cfg.path_observed_threshold)
    observed = np.zeros(len(path), bool)
    filled = np.zeros(len(path), bool)
    for a, b in intervals:
        member = (path[:, 0] >= a) & (path[:, 0] < b) & scope
        supported = member & (values >= cfg.path_observed_threshold)
        local = supported.copy()
        positions = np.flatnonzero(member)
        if not len(positions):
            continue
        left, right = positions[0], positions[-1]+1
        for i, j in true_runs(~supported[left:right]):
            i, j = i+left, j+left
            if (i > left and j < right and supported[i-1] and supported[j]
                    and j-i <= profile['gap_budget'] and member[i:j].all()):
                local[i:j] = True
        observed |= supported
        filled |= local
    return dict(path=path, val=values, observed=observed, filled=filled,
                support_intervals=intervals, profile=profile, status=result['status'],
                directional_config=result['config'], cost=result['cost'])


def _inside(x, y, box):
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _active_points(series, plot, legends):
    rows = []
    for i, point in enumerate(series.get('active_points', [])):
        if isinstance(point, dict):
            x, y = point.get('x_px', point.get('x')), point.get('y_px', point.get('y'))
            identifier = str(point.get('id', point.get('candidate_id', f'A{i+1:03d}')))
        else:
            x, y = point
            identifier = f'A{i+1:03d}'
        x, y = float(x), float(y)
        if np.isfinite([x, y]).all() and _inside(x, y, plot) and not any(_inside(x, y, box) for box in legends):
            rows.append(dict(id=identifier, x=x, y=y))
    return rows


def build_tentative_candidates(img_bgr, plot_box, series, *, legend_boxes=(), valid=None,
                               native_helpers=None, cfg=Config()):
    """Build source-pixel hypotheses with strong-only suppressed eligibility.

    Each series supplies id/rgb/diameter, plot-local soft/own maps and source
    active_points. Optional shape_info/shape_fields hold a completed composition
    template. Unsupported composition may supply the preserved observed raster;
    a swatch_box can also provide that fallback. Explicit marker_allowed=False
    permits curve-only legends to return paths without invented marker shapes.
    No input series, arrays or active_points are modified.
    """
    started = perf_counter()
    image = np.asarray(img_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('img_bgr must be HxWx3')
    plot = tuple(map(int, plot_box))
    if len(plot) != 4 or not (0 <= plot[0] < plot[2] <= image.shape[1] and 0 <= plot[1] < plot[3] <= image.shape[0]):
        raise ValueError('plot_box must be nonempty, half-open and inside image')
    legends = [tuple(map(int, box)) for box in legend_boxes]
    shape = (plot[3]-plot[1], plot[2]-plot[0])
    allowed = np.ones(shape, bool) if valid is None else np.asarray(valid, bool).copy()
    if allowed.shape != shape:
        raise ValueError('valid must be plot-local')
    for x0, y0, x1, y1 in legends:
        a, b = max(x0, plot[0])-plot[0], max(y0, plot[1])-plot[1]
        c, d = min(x1, plot[2])-plot[0], min(y1, plot[3])-plot[1]
        if a < c and b < d:
            allowed[b:d, a:c] = False
    identifiers = [str(s['id']) for s in series]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('series IDs must be unique; same RGB may still have distinct IDs')
    prepared = []
    for spec in series:
        soft, own = np.asarray(spec['soft'], np.float32), np.asarray(spec['own'], bool)
        if soft.shape != shape or own.shape != shape:
            raise ValueError('Each soft/own map must be plot-local')
        if not np.isfinite(soft).all() or (soft < 0).any() or (soft > 1).any():
            raise ValueError('soft must be finite in [0,1]')
        diameter = float(spec.get('diameter', 12.))
        if not np.isfinite(diameter) or diameter < 2:
            raise ValueError('diameter must be finite and at least 2 pixels')
        prepared.append(dict(spec=spec, id=str(spec['id']), rgb=np.asarray(spec['rgb'], float),
                             soft=soft*allowed, own=own & allowed, diameter=diameter,
                             anchors=_active_points(spec, plot, legends)))
    all_candidates, paths, diagnostics = [], [], []
    offset = np.array(plot[:2], float)
    for item in prepared:
        tick = perf_counter()
        spec, sid, diameter = item['spec'], item['id'], item['diameter']
        traced = trace_retained_path(item['soft'], item['own'], allowed, cfg=cfg, native_helpers=native_helpers)
        path_local = np.asarray(traced['path'], float).reshape(-1, 2)
        path_source = path_local + offset
        context = dict(traced, path=path_source, series_id=sid,
                       coordinate_convention='source-image pixel centres; rejected samples excluded by filled')
        paths.append(context)
        # max_support_knots=0 intentionally omits the path-derived K* proposals.
        explanation = analyze_explanation(path_source, traced['filled'], traced['observed'],
            item['anchors'], diameter, plot, CurveConfig(max_support_knots=0))
        row = dict(series_id=sid, active_anchor_count=len(item['anchors']),
                   path_samples=len(path_local), retained_path_samples=int(traced['filled'].sum()),
                   observed_path_samples=int(traced['observed'].sum()),
                   curve_explanation=explanation, raw_shape_candidates=0, raw_residual_candidates=0,
                   omitted_existing_anchor_neighbours=0, duplicate_candidates=0)
        if not spec.get('marker_allowed', True):
            row.update(status='line_only_no_marker_hypotheses', seconds=perf_counter()-tick)
            diagnostics.append(row)
            continue
        residual_record, fields = analyze_series(item['soft'], item['own'], allowed,
            path_local, traced['filled'], traced['observed'], diameter, HypothesisConfig())
        pool = []
        for p in residual_record['path_guided']:
            pool.append(dict(p, origin='path_residual', evidence_supported=p['kind']=='image_backed_possible',
                             score=float(p['guided_score']), evidence=dict(p['rank_factors'])))
        row['raw_residual_candidates'] = len(pool)
        shape_info, shape_fields = spec.get('shape_info'), spec.get('shape_fields')
        if shape_info is None or shape_fields is None:
            if spec.get('swatch_box') is not None:
                try:
                    shape_info, shape_fields = extract_legend_shape(image[..., ::-1], spec['swatch_box'],
                        [s['rgb'] for s in series], identifiers.index(sid))
                    row['template_source'] = 'observed_raster_fallback'
                except ValueError as exc:
                    row['shape_status'] = f'no_usable_shape: {exc}'
            else:
                row['shape_status'] = 'no_template_supplied'
        else:
            row['template_source'] = shape_info.get('method', 'caller_template')
        if shape_info is not None and shape_fields is not None:
            # Identical palette entries are different shape identities, not
            # different-colour occluders. Own pixels also never excuse absence.
            rivals = [r for r in prepared if r['id'] != sid and not np.allclose(r['rgb'], item['rgb'], atol=1.)]
            other = np.maximum.reduce([r['soft']*r['own'] for r in rivals]) if rivals else np.zeros(shape, np.float32)
            foreground = [dict(x=a['x']-offset[0], y=a['y']-offset[1], radius=.5*r['diameter'],
                               id=a['id'], series_id=r['id'], existence='unknown')
                          for r in rivals for a in r['anchors']]
            shape_record, _ = analyze_shape_candidates(item['soft'], item['own'], other, allowed,
                fields['path_distance'], fields['line'], shape_info, shape_fields,
                foreground_markers=foreground, cfg=ShapeConfig())
            row['shape_diagnostics'] = {k: v for k, v in shape_record.items() if k != 'candidates'}
            row['raw_shape_candidates'] = len(shape_record['candidates'])
            for p in shape_record['candidates']:
                pool.append(dict(p, origin='legend_shape_occlusion', evidence=dict(p['metrics'])))
        candidates = []
        for p in pool:
            x, y = float(p['x'])+offset[0], float(p['y'])+offset[1]
            if not _inside(x, y, plot) or any(_inside(x, y, box) for box in legends):
                continue
            local_x, local_y = int(round(x-offset[0])), int(round(y-offset[1]))
            if not allowed[local_y, local_x]:
                continue
            nearest = min((np.hypot(x-a['x'], y-a['y']) for a in item['anchors']), default=np.inf)
            if nearest <= cfg.existing_marker_clearance_diameters*diameter:
                row['omitted_existing_anchor_neighbours'] += 1
                continue
            effect = evaluate_addition(explanation, [x, y])
            benefit = effect['benefit']
            supported = bool(p.get('evidence_supported', False))
            status = ('image_supported_curve_helpful' if supported and benefit > cfg.minimum_curve_benefit else
                      'image_supported_no_curve_need' if supported else
                      'curve_helpful_but_image_ambiguous' if benefit > cfg.minimum_curve_benefit else
                      'image_backed_ambiguous')
            candidates.append(dict(series_id=sid, x=x, y=y, x_px=x, y_px=y,
                source=VERSION, origin=p['origin'], existence='unknown', state='suppressed',
                tentative=True, auto_promote=False, evidence_supported=supported, evidence_status=status,
                evidence=p['evidence'], score=float(p['score']), diameter=diameter,
                path_distance_px=p.get('path_distance_px'),
                curve_effect={k: v for k, v in effect.items() if k != 'after'},
                nearest_active_distance_px=float(nearest) if np.isfinite(nearest) else None,
                reasons=['Image-generated review candidate; not a path support knot or a confirmed datum.',
                         'Other-colour ink may explain occlusion but never supplies positive own-marker evidence.']))
        # Two proposal sources can represent the same location. Prefer explicit
        # shape support, then curve benefit; merging never creates a new centre.
        candidates.sort(key=lambda p: (not p['evidence_supported'],
            -p['curve_effect']['benefit'], p['origin'] != 'legend_shape_occlusion', -p['score'], p['y'], p['x']))
        retained = []
        for candidate in candidates:
            strong = strong_evidence(candidate)
            candidate.update(evidence_tier='strong' if strong else 'ambiguous',
                             step5_suppressed_eligible=strong,
                             state='suppressed' if strong else 'diagnostic_only')
            duplicate = next((q for q in retained if np.hypot(q['x']-candidate['x'], q['y']-candidate['y'])
                              <= cfg.duplicate_candidate_clearance_diameters*diameter), None)
            if duplicate is not None:
                row['duplicate_candidates'] += 1
                duplicate.setdefault('alternative_proposals', []).append(dict(
                    origin=candidate['origin'], x=candidate['x'], y=candidate['y'],
                    evidence_supported=candidate['evidence_supported']))
                continue
            candidate['candidate_id'] = f'{sid}-T{len(retained)+1:03d}'
            retained.append(candidate)
        all_candidates.extend(retained)
        row.update(status='completed', tentative_candidates=len(retained),
                   suppressed_candidates=sum(strong_evidence(p) for p in retained),
                   diagnostics_only_candidates=sum(not strong_evidence(p) for p in retained),
                   seconds=perf_counter()-tick)
        diagnostics.append(row)
    return dict(version=VERSION, candidates=all_candidates, paths=paths,
        diagnostics=diagnostics, config=asdict(cfg), suppressed_policy=policy_info(), confirmed_marker_count=0,
        active_points_changed=False, diagnostic_knots_exported=False, native_resolution=True,
        coordinate_convention='source-image pixel centres; half-open plot/legend boxes',
        seconds=perf_counter()-started)

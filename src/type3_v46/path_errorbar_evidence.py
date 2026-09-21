"""Local, observed error-bar structure evidence for a supplied path height.

This experiment-only scorer does NOT estimate measurement y, trace a path,
assign a series, erase pixels, or read files. ``path_y`` is a caller-supplied
plot-local pixel coordinate. Stem/cap extrema use inclusive pixel indices.

Unlike ``errorbar_stems._verify_cap``, a cap need not be isolated enough to
erase safely. A nearby curve and old ``strict=False`` flags are not penalties.
Actual short horizontal arms and a narrow vertical core are rechecked in the
supplied neutral mask. Three-pixel cap thickness is allowed at line_width=1.

"supported" means compatible local error-bar structure, NOT a confirmed
measurement. An I-shaped text glyph can be locally identical to an error bar;
the caller must independently check colored-curve support, bends, text/grid
context, and competing series. Scores are diagnostics, not probabilities.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'path_errorbar_structure_v1'


def _runs(indices):
    indices = np.asarray(indices, dtype=int)
    return [] if not len(indices) else np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)


def _nearest_run(row, x, offset, distance):
    runs = _runs(np.flatnonzero(row))
    if not runs:
        return None
    run = min(runs, key=lambda r: max(float(r[0]+offset)-x, x-float(r[-1]+offset), 0.0))
    if max(float(run[0]+offset)-x, x-float(run[-1]+offset), 0.0) > distance:
        return None
    return int(run[0]+offset), int(run[-1]+offset)


def _caps(mask, x, y0, y1, lw):
    height, width = mask.shape
    search = max(2, int(math.ceil(2*lw)))
    max_width = max(25, int(math.ceil(24*lw))+1)
    max_thickness = max(3, int(math.ceil(2.5*lw)))
    min_arm = max(2, int(math.ceil(1.5*lw)))
    center_guard = max(.5, .5*lw)
    rows = []
    wide_rows = 0
    # Full rows are not scanned: one extra pixel outside the maximum legal
    # reach reveals a continuing horizontal line without accepting a crop edge.
    xa = max(0, int(math.floor(x))-max_width-1)
    xb = min(width, int(math.ceil(x))+max_width+2)
    candidate_ys = sorted(set(range(max(0, y0-search), min(height, y0+search+1))) |
                          set(range(max(0, y1-search), min(height, y1+search+1))))
    for y in candidate_ys:
        run = _nearest_run(mask[y, xa:xb], x, xa, max(1, .75*lw))
        if run is None:
            continue
        a, b = run
        if b-a+1 > max_width:
            wide_rows += 1
            continue
        left, right = max(0.0, x-center_guard-a), max(0.0, b-x-center_guard)
        if max(left, right) < min_arm:
            continue
        rows.append(dict(y=y, x0=a, x1=b, left_arm=left, right_arm=right))
    groups = []
    for row in rows:
        if groups and row['y'] == groups[-1][-1]['y']+1 and (
                row['x0'] <= groups[-1][-1]['x1'] and row['x1'] >= groups[-1][-1]['x0']):
            groups[-1].append(row)
        else:
            groups.append([row])
    caps, rejected = [], []
    for group in groups:
        ya, yb = group[0]['y'], group[-1]['y']
        a, b = min(r['x0'] for r in group), max(r['x1'] for r in group)
        best = max(group, key=lambda r: min(r['left_arm'], r['right_arm']))
        both = best['left_arm'] >= min_arm and best['right_arm'] >= min_arm
        col_centers = []
        for col in range(a, b+1):
            if abs(col-x) <= max(1, lw):
                continue
            ys = np.flatnonzero(mask[ya:yb+1, col])+ya
            if len(ys):
                col_centers.append((col, float(np.median(ys))))
        slope = float(np.polyfit(*np.asarray(col_centers).T, 1)[0]) if len(col_centers) >= 3 else 0.0
        record = dict(y=float(np.median([r['y'] for r in group])), y0=ya, y1=yb,
                      x0=a, x1=b, width=b-a+1, thickness=yb-ya+1,
                      two_sided=bool(both), left_arm=float(best['left_arm']),
                      right_arm=float(best['right_arm']), horizontal_slope=slope,
                      strict_removal_required=False)
        if yb-ya+1 > max_thickness:
            record['reason'] = 'thick_horizontal_body_not_cap'
            rejected.append(record)
        elif abs(slope) > .30:
            record['reason'] = 'sloping_fragment_not_horizontal_cap'
            rejected.append(record)
        elif b-a+1 > max_width:
            record['reason'] = 'wide_horizontal_body_not_cap'
            rejected.append(record)
        else:
            caps.append(record)
    return caps, dict(rejected=rejected, too_wide_rows=wide_rows,
                     maximum_cap_width=max_width, maximum_cap_thickness=max_thickness,
                     minimum_cap_arm=min_arm, endpoint_search_radius=search)


def _closed_parallel_structure(mask, x, y0, y1, caps, lw):
    """A joined parallel upright is box/0-shaped evidence, not a single bar.

    Two neighboring unjoined error bars do not satisfy this check. This is a
    conservative shape veto, not a general OCR/text classifier.
    """
    if len(caps) < 2:
        return None
    top = min(caps, key=lambda c:c['y'])
    bottom = max(caps, key=lambda c:c['y'])
    a, b = max(top['x0'], bottom['x0']), min(top['x1'], bottom['x1'])
    ya, yb = top['y1']+1, bottom['y0']-1
    if yb-ya+1 < max(3, int(math.ceil(3*lw))):
        return None
    for col in range(a, b+1):
        if abs(col-x) <= max(3, 3*lw):
            continue
        radius = max(0, int(math.floor(.5*lw)))
        hit = mask[ya:yb+1, max(0,col-radius):min(mask.shape[1],col+radius+1)].any(axis=1)
        coverage = float(hit.mean())
        if coverage >= .70:
            return dict(x=int(col), y0=ya, y1=yb, coverage=coverage,
                        reason='parallel_uprights_joined_at_both_caps')
    return None


def score_errorbar_at_path(stem, neutral_mask, dark_field, path_y, line_width):
    """Return JSON-safe structural evidence at ``(stem['x'], path_y)``.

    ``neutral_mask`` is bool[H,W] in plot-local coordinates and must already
    respect the caller's valid/exclusion mask. ``dark_field`` is None or a
    finite float[H,W] in [0,1]; it records observed darkness only, never fills
    missing neutral pixels. Neither array is mutated. Existing stem cap hints
    and safe-removal flags are intentionally not used to decide acceptance.

    Output keys include status/reason/structural_score, vertical, caps,
    diagnostics, x/path_y, and version. One-sided cap evidence is tentative;
    short interruptions near a path are not automatically evidence of a bar.
    """
    mask = np.asarray(neutral_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('neutral_mask must be a two-dimensional boolean array')
    if dark_field is not None:
        dark = np.asarray(dark_field, dtype=float)
        if dark.shape != mask.shape or not np.isfinite(dark).all() or np.any((dark < 0) | (dark > 1)):
            raise ValueError('dark_field must match the mask and contain finite values in [0,1]')
    else:
        dark = None
    lw, x, py = float(line_width), float(stem['x']), float(path_y)
    if not np.isfinite([lw,x,py]).all() or lw <= 0:
        raise ValueError('line_width must be positive and x/path_y must be finite')
    y0, y1 = int(stem['y0']), int(stem['y1'])
    height, width = mask.shape
    result = dict(version=VERSION, status='rejected', reason=None,
                  structural_score=0.0, x=x, path_y=py, vertical={}, caps=[],
                  diagnostics=dict(strict_flags_ignored=True, pixels_modified=False,
                                   score_is_probability=False,
                                   limitation='Local I-shaped text is not distinguishable from an error bar without independent context.'))

    def finish(status, reason, score=0.0):
        result.update(status=status, reason=reason, structural_score=float(np.clip(score,0,1)))
        return result

    if not (0 <= x < width and 0 <= py < height and 0 <= y0 <= y1 < height):
        return finish('rejected','geometry_outside_mask')
    pad = max(2.0, 2*lw)
    if py < y0-pad or py > y1+pad:
        return finish('rejected','path_height_outside_observed_stem_span')
    minimum_height = max(5, int(math.ceil(4*lw)))
    if y1-y0+1 < minimum_height:
        return finish('rejected','vertical_span_too_short')
    caps, cap_diag = _caps(mask,x,y0,y1,lw)
    result['caps'] = caps
    result['diagnostics']['cap_search'] = cap_diag
    radius = max(3, int(math.ceil(3*lw)))
    xa, xb = max(0,int(math.floor(x))-radius), min(width,int(math.ceil(x))+radius+1)
    row_info = []
    all_supported_y = []
    broad_rows = 0
    excluded_cap_rows = set(y for c in caps for y in range(c['y0'],c['y1']+1))
    max_stem_width = max(3,int(math.ceil(2.5*lw)))
    for y in range(y0,y1+1):
        run = _nearest_run(mask[y,xa:xb],x,xa,max(1,.75*lw))
        if run is None:
            continue
        a,b = run
        all_supported_y.append(y)
        if y in excluded_cap_rows:
            continue
        if b-a+1 > max_stem_width:
            broad_rows += 1
            continue
        row_info.append((y,(a+b)/2,b-a+1))
    effective_height = sum(y not in excluded_cap_rows for y in range(y0,y1+1))
    coverage = len(row_info)/max(1,effective_height)
    near_path = min((abs(y-py) for y in all_supported_y),default=float('inf'))
    centers = np.asarray([r[1] for r in row_info])
    drift = float(np.percentile(centers,90)-np.percentile(centers,10)) if len(centers) else 0.0
    if len(row_info) >= 3:
        fit_slope = float(np.polyfit(np.asarray(row_info)[:,0],centers,1)[0])
    else:
        fit_slope = 0.0
    darkness = None
    if dark is not None and row_info:
        darkness = float(np.mean([dark[y,int(round(cx))] for y,cx,_ in row_info]))
    result['vertical'] = dict(y0=y0,y1=y1,minimum_height=minimum_height,
                              observed_narrow_rows=len(row_info),effective_rows=effective_height,
                              narrow_coverage=coverage,broad_rows=broad_rows,
                              median_width=float(np.median([r[2] for r in row_info])) if row_info else None,
                              maximum_width=max_stem_width,center_drift=drift,
                              center_slope=fit_slope,
                              nearest_observed_row_to_path=None if not np.isfinite(near_path) else float(near_path),
                              mean_observed_darkness=darkness)
    if len(row_info) < max(3,int(math.ceil(2*lw))) or coverage < .55:
        return finish('rejected','insufficient_narrow_vertical_support')
    if drift > max(2.0,1.5*lw) or abs(fit_slope) > .25:
        return finish('rejected','diagonal_not_vertical_structure')
    if near_path > max(2.5,2*lw):
        return finish('rejected','vertical_ink_does_not_reach_path_neighborhood')
    closed = _closed_parallel_structure(mask,x,y0,y1,caps,lw)
    result['diagnostics']['closed_structure'] = closed
    if closed:
        return finish('rejected','closed_or_digit_like_parallel_structure')
    if not caps:
        return finish('rejected','no_observed_short_horizontal_cap',.35*coverage)
    bilateral = any(c['two_sided'] for c in caps)
    cap_quality = 1.0 if bilateral else .5
    score = .45*min(coverage,1)+.35*cap_quality+.20*max(0,1-near_path/max(2.5,2*lw))
    if not bilateral:
        return finish('tentative','one_sided_cap_only',score)
    if coverage < .80:
        return finish('tentative','interrupted_vertical_support',score)
    return finish('supported','observed_vertical_and_bilateral_cap',score)

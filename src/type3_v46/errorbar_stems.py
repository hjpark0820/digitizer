"""Observed, scale-adaptive error-bar structure proposals (experiment only).

``propose_stems(own, valid, line_width)`` accepts native plot-local masks.
All x/y coordinates are pixel centres. Stem y0/y1 and cap x0/x1 are inclusive
extrema; cap width is x1-x0+1. No marker shape, observation schedule, antibody
identity, ground truth, or path is an input.

The column-run/edge-drift proposal algorithm is adapted from
src/run_A4_auto_v45.py::vertical_strokes/_column_spans/_span_drift. It is copied
here to avoid importing that CLI's side effects. Constants below are relative
to the supplied observed line width, not native-image-specific cap lengths.

stem_mask is PERMISSIVE proposal evidence, not a safe deletion mask. cap_mask
contains only actual pixels of isolated, two-sided terminal cap ARMS; the stem
intersection is preserved. A reported fragment is not automatically removable.
These structures propose an x coordinate, not a measurement y or a marker.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = 'observed_errorbar_stems_v2'
PROVENANCE = 'Adapted v45 vertical_strokes column-span edge-drift logic; experiment-local scale-adaptive cap verification'


def _runs(values):
    values = np.asarray(values, dtype=int)
    if not len(values):
        return []
    return np.split(values, np.flatnonzero(np.diff(values) > 1)+1)


def _column_spans(component, xoff, yoff):
    spans = {}
    for col in range(component.shape[1]):
        runs = _runs(np.flatnonzero(component[:, col]))
        if runs:
            run = max(runs, key=len)
            spans[col+xoff] = (int(run[0])+yoff, int(run[-1])+yoff)
    return spans


def _span_drift(a, b):
    # A cap can change one edge of a real vertical stroke. A diagonal carries
    # BOTH edges along, hence the smaller edge shift rather than centre drift.
    return min(abs(a[0]-b[0]), abs(a[1]-b[1]))


def _singleton_is_diagonal(column, spans, max_drift):
    """Allow a genuine 1px stem, not one column chopped from a steep diagonal."""
    own = spans[column]
    height = own[1]-own[0]+1
    for neighbor in (column-1, column+1):
        if neighbor not in spans:
            continue
        other = spans[neighbor]
        other_height = other[1]-other[0]+1
        if (other_height >= .70*height and _span_drift(own, other) > max_drift
                and _span_drift(own, other) >= .20*min(height, other_height)):
            return True
    return False


def _group_is_diagonal_end(group, spans, max_drift):
    """Rounded diagonal ends can hold one edge still across several columns.

    Require BOTH a strongly sheared group and a comparably tall continuing
    diagonal immediately outside it. One-pixel raster jitter, a true vertical
    core beside a short sloping connector, or a cap's width alone is not enough.
    """
    lo = min(spans[x][0] for x in group)
    hi = max(spans[x][1] for x in group)
    height = hi-lo+1
    common = max(0, min(spans[x][1] for x in group)-max(spans[x][0] for x in group)+1)
    if common/max(1, height) >= .80:
        return False
    for edge, neighbor in ((group[0], group[0]-1), (group[-1], group[-1]+1)):
        if neighbor in spans:
            other = spans[neighbor]
            if other[1]-other[0]+1 >= .65*height and _span_drift(spans[edge], other) > max_drift:
                return True
    return False


def _vertical_proposals(ink, parameters):
    opened = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((parameters['opening_height'], 1), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    proposals, skipped_diagonal_groups = [], 0
    for label in range(1, n):
        x, y, width, height, _ = map(int, stats[label])
        if height < parameters['minimum_height']:
            continue
        spans = _column_spans(labels[y:y+height, x:x+width] == label, x, y)
        columns = sorted(spans)
        if not columns:
            continue
        groups, current = [], [columns[0]]
        for col in columns[1:]:
            if col == current[-1]+1 and _span_drift(spans[col], spans[current[-1]]) <= parameters['maximum_edge_drift']:
                current.append(col)
            else:
                groups.append(current)
                current = [col]
        groups.append(current)
        for group in groups:
            if len(group) < parameters['minimum_columns']:
                skipped_diagonal_groups += 1
                continue
            if len(group) == 1 and _singleton_is_diagonal(group[0], spans, parameters['maximum_edge_drift']):
                skipped_diagonal_groups += 1
                continue
            if _group_is_diagonal_end(group, spans, parameters['maximum_edge_drift']):
                skipped_diagonal_groups += 1
                continue
            lo = min(spans[col][0] for col in group)
            hi = max(spans[col][1] for col in group)
            if hi-lo+1 < parameters['minimum_height']:
                continue
            x0, x1 = group[0], group[-1]
            support = np.zeros((hi-lo+1, x1-x0+1), bool)
            for col in group:
                a, b = spans[col]
                support[a-lo:b-lo+1, col-x0] = True
            row_centres = [float(np.flatnonzero(row).mean()+x0) for row in support if row.any()]
            proposals.append(dict(x=float(np.median(row_centres)), x0=x0, x1=x1,
                                  y0=lo, y1=hi, width=x1-x0+1, _support=support))
    proposals.sort(key=lambda p: (p['x'], p['y0'], p['y1']))
    return proposals, opened.astype(bool), skipped_diagonal_groups


def _row_cap_candidate(ink, stem, y, parameters):
    """Return only actually observed horizontal runs close to the stem end."""
    height, width = ink.shape
    x0, x1 = stem['x0'], stem['x1']
    gap = parameters['cap_center_gap']
    nearby = []
    for run in _runs(np.flatnonzero(ink[y])):
        a, b = int(run[0]), int(run[-1])
        if b >= x0-gap and a <= x1+gap:
            nearby.append((a, b))
    if not nearby:
        return None
    a, b = min(p[0] for p in nearby), max(p[1] for p in nearby)
    cap_width = b-a+1
    if cap_width > parameters['maximum_cap_width']:
        return None  # continuing horizontal curve, not a compact cap proposal
    left, right = max(0, x0-a), max(0, b-x1)
    arm = parameters['minimum_cap_arm']
    if max(left, right) < arm:
        return None
    if cap_width <= stem['width']+arm:
        return None
    central = bool(ink[y, x0:x1+1].any())
    both = left >= arm and right >= arm
    return dict(y=y, x0=a, x1=b, width=cap_width, left_arm=left,
                right_arm=right, two_sided=both, center_observed=central)


def _verify_cap(ink, valid, stem, row_group, end_kind, parameters):
    """Conservative cap removal: two arms, terminal placement and isolation."""
    height, width = ink.shape
    ya, yb = min(r['y'] for r in row_group), max(r['y'] for r in row_group)
    a, b = min(r['x0'] for r in row_group), max(r['x1'] for r in row_group)
    yy = float(np.median([r['y'] for r in row_group]))
    center_observed = any(r['center_observed'] for r in row_group)
    two_sided = any(r['two_sided'] for r in row_group)
    band_height = yb-ya+1
    guard = parameters['stem_intersection_guard']
    gx0 = max(a, int(math.floor(stem['x']))-guard)
    gx1 = min(b, int(math.ceil(stem['x']))+guard)
    arms = np.arange(a, b+1)
    arms = arms[(arms < gx0) | (arms > gx1)]
    neighbor_hits = np.zeros(len(arms), bool)
    probe = parameters['cap_isolation_probe']
    for y in (ya-probe, yb+probe):
        if 0 <= y < height and len(arms):
            neighbor_hits |= ink[y, arms]
    isolation_fraction = float(neighbor_hits.mean()) if len(arms) else 1.
    # Look beyond the horizontal run's ends for sloping continuations. A cap
    # ends; a tangent/connector may leave the row while still continuing nearby.
    continuations = []
    for edge, direction in ((a, -1), (b, 1)):
        count = 0
        for step in range(1, parameters['cap_continuation_probe']+1):
            xx = edge+direction*step
            if not 0 <= xx < width:
                break
            lo, hi = max(0, ya-probe), min(height, yb+probe+1)
            if ink[lo:hi, xx].any():
                count += 1
        continuations.append(count)
    clipped = (a == 0 or b == width-1 or ya == 0 or yb == height-1
               or not valid[ya:yb+1, a:b+1].all())
    strict = (two_sided and center_observed and not clipped
              and band_height <= parameters['maximum_cap_thickness']
              and isolation_fraction <= .15 and max(continuations, default=0) <= 1)
    reasons = []
    if not two_sided:
        reasons.append('one_sided_observed_fragment')
    if not center_observed:
        reasons.append('center_not_observed')
    if clipped:
        reasons.append('touches_image_or_invalid_boundary')
    if band_height > parameters['maximum_cap_thickness']:
        reasons.append('thick_or_sloping_horizontal_component')
    if isolation_fraction > .15:
        reasons.append('nearby_curve_or_other_ink_at_cap_arms')
    if max(continuations, default=0) > 1:
        reasons.append('ink_continues_beyond_horizontal_fragment')
    mask = np.zeros_like(ink)
    if strict:
        # Never erase the stem/curve crossing and never dilate beyond observed
        # cap pixels. Root's y estimator may use the bounds as an exclusion
        # window, but this mask alone does not delete arbitrary nearby curves.
        mask[ya:yb+1, a:b+1] = ink[ya:yb+1, a:b+1]
        mask[:, gx0:gx1+1] = False
    count = int(ink[ya:yb+1, a:b+1].sum())
    record = dict(y=yy, x0=a, x1=b, width=b-a+1,
                  y0=ya, y1=yb, thickness=band_height,
                  kind=end_kind+('_cap' if two_sided else '_cap_fragment'),
                  two_sided=two_sided, center_observed=center_observed,
                  strict=strict, source_support=count/max(1, band_height*(b-a+1)),
                  observed_pixels=count, left_arm=max(0, stem['x0']-a),
                  right_arm=max(0, b-stem['x1']),
                  isolation_fraction=isolation_fraction,
                  outward_continuation_columns=continuations,
                  removal_reasons=[] if strict else reasons,
                  cap_mask_pixels=int(mask.sum()))
    return record, mask


def _caps_for_stem(ink, valid, stem, parameters):
    caps, mask = [], np.zeros_like(ink)
    search = parameters['cap_end_search']
    for kind, endpoint in (('upper', stem['y0']), ('lower', stem['y1'])):
        candidates = []
        for y in range(max(0, endpoint-search), min(ink.shape[0], endpoint+search+1)):
            record = _row_cap_candidate(ink, stem, y, parameters)
            if record is not None:
                candidates.append(record)
        groups = []
        for candidate in candidates:
            if (groups and candidate['y'] <= groups[-1][-1]['y']+1
                    and min(candidate['x1'], groups[-1][-1]['x1']) >= max(candidate['x0'], groups[-1][-1]['x0'])):
                groups[-1].append(candidate)
            else:
                groups.append([candidate])
        for group in groups:
            center = float(np.median([r['y'] for r in group]))
            nearest_end = 'upper' if abs(center-stem['y0']) <= abs(center-stem['y1']) else 'lower'
            record, safe = _verify_cap(ink, valid, stem, group, nearest_end, parameters)
            if any(max(record['y0'], old['y0']) <= min(record['y1'], old['y1'])
                   and record['x0'] == old['x0'] and record['x1'] == old['x1'] for old in caps):
                continue
            caps.append(record)
            mask |= safe
    return sorted(caps, key=lambda p: p['y']), mask


def _shared_cap(a, b, line_width, require_row_overlap=False):
    """Return geometric agreement on the SAME observed cap, not nearest x."""
    if not (a['two_sided'] and b['two_sided'] and a['center_observed'] and b['center_observed']):
        return None
    if abs(a['y']-b['y']) > max(1., line_width):
        return None
    if require_row_overlap and max(a['y0'], b['y0']) > min(a['y1'], b['y1']):
        return None
    overlap = max(0, min(a['x1'], b['x1'])-max(a['x0'], b['x0'])+1)
    union = max(a['x1'], b['x1'])-min(a['x0'], b['x0'])+1
    ratio = overlap/max(1, union)
    if ratio < .65:
        return None
    ca, cb = .5*(a['x0']+a['x1']), .5*(b['x0']+b['x1'])
    if abs(ca-cb) > max(1., .75*line_width):
        return None
    return dict(cap_y=.5*(a['y']+b['y']), cap_center_x=.5*(ca+cb),
                horizontal_iou=float(ratio), source_cap_spans=[[a['x0'], a['x1']], [b['x0'], b['x1']]])


def _merge_evidence(a, b, line_width):
    if abs(a['x']-b['x']) > max(1.5, 1.5*line_width):
        return None
    overlap = max(0, min(a['y1'], b['y1'])-max(a['y0'], b['y0'])+1)
    fraction = overlap/max(1, min(a['y1']-a['y0']+1, b['y1']-b['y0']+1))
    if fraction < .35:
        return None
    shared = [value for ca in a['caps'] for cb in b['caps']
              if (value := _shared_cap(ca, cb, line_width)) is not None]
    if not shared:
        return None
    strongest = max(shared, key=lambda row: row['horizontal_iou'])
    return dict(**strongest, vertical_overlap_fraction=float(fraction))


def _merge_same_stem_fragments(stems, line_width):
    """Complete-link clustering: every member pair needs a shared cap witness.

    This is not measurement-time clustering. Nearby stems with separate caps
    remain separate. Observed pixel unions are retained; no gap is painted in.
    """
    groups = []
    for stem in stems:
        chosen = None
        for group in groups:
            if all(_merge_evidence(stem, member, line_width) is not None for member in group):
                chosen = group
                break
        if chosen is None:
            groups.append([stem])
        else:
            chosen.append(stem)
    merged, merge_log = [], []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        witnesses = [_merge_evidence(a, b, line_width) for i, a in enumerate(group) for b in group[i+1:]]
        x0, x1 = min(p['x0'] for p in group), max(p['x1'] for p in group)
        y0, y1 = min(p['y0'] for p in group), max(p['y1'] for p in group)
        support = np.zeros((y1-y0+1, x1-x0+1), bool)
        for member in group:
            support[member['y0']-y0:member['y1']-y0+1,
                    member['x0']-x0:member['x1']-x0+1] |= member['_support']
        # A symmetric shared cap locates the same physical stem more reliably
        # than averaging a thickened/diagonal side fragment into its x centre.
        cap_centres = [v['cap_center_x'] for v in witnesses]
        proposed_x = float(np.median(cap_centres))
        if not x0-line_width <= proposed_x <= x1+line_width:
            proposed_x = float(np.median([p['x'] for p in group]))
        caps = []
        for cap in sorted([c for p in group for c in p['caps']], key=lambda p: (p['y'], p['x0'])):
            index = next((i for i, old in enumerate(caps)
                          if _shared_cap(cap, old, line_width, require_row_overlap=True) is not None), None)
            if index is None:
                caps.append(dict(cap))
            elif (cap['strict'], cap['observed_pixels']) > (caps[index]['strict'], caps[index]['observed_pixels']):
                caps[index] = dict(cap)
        member_records = [{k: v for k, v in p.items() if k not in ('_support', 'caps')} for p in group]
        record = dict(id=group[0]['id'], x=proposed_x, x0=x0, x1=x1, y0=y0, y1=y1,
                      width=x1-x0+1, caps=caps, source_support=float(support.mean()),
                      observed_pixels=int(support.sum()), _support=support,
                      proposal_kind='shared_cap_vertical_fragments',
                      fragment_status='permissive_merged_structure_y_unresolved',
                      touches_boundary=any(p['touches_boundary'] for p in group),
                      merged_members=member_records, merge_evidence=witnesses,
                      interpretation='shared observed cap/overlapping vertical evidence gives one candidate x; measurement y unresolved')
        merged.append(record)
        merge_log.append(dict(member_ids=[p['id'] for p in group], member_x=[p['x'] for p in group],
                              merged_x=proposed_x, evidence=witnesses))
    merged.sort(key=lambda p: (p['x'], p['y0'], p['y1']))
    for index, stem in enumerate(merged, 1):
        stem['proposal_id_before_merge'] = stem['id']
        stem['id'] = f'ST{index:03d}'
    return merged, merge_log


def propose_stems(own, valid, line_width):
    """Propose observed vertical structures and measured terminal cap fragments.

    Return ``{stems, stem_mask, cap_mask, diagnostics}``. ``source_support`` is
    an observed-ink occupancy fraction, NOT the probability of a true data
    point. Both masks are bool and have the input shape. Multiple colors at
    the same x must be handled by the caller; no cross-series fusion happens.
    """
    own, valid = np.asarray(own), np.asarray(valid)
    if own.ndim != 2 or valid.ndim != 2 or own.shape != valid.shape:
        raise ValueError('own and valid must be same-shaped 2D masks')
    if own.dtype != bool or valid.dtype != bool:
        raise TypeError('own and valid must have boolean dtype')
    line_width = float(line_width)
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError('line_width must be finite and positive native pixels')
    lw = max(1., line_width)
    parameters = dict(
        # Odd kernels keep source-centred morphology (even OpenCV anchors can
        # shift a stem's extrema by a pixel).
        opening_height=max(5, 2*int(math.ceil(1.5*lw))+1),
        minimum_height=max(7, int(math.ceil(4.5*lw))),
        # Integer raster boundaries can move one pixel along a true 2px stem.
        maximum_edge_drift=max(1., .35*lw),
        # A surviving vertical core can be thinner than the color's horizontal
        # stroke. Single-column evidence stays permissive and uses the diagonal
        # neighbor guard instead of being deleted by a width requirement.
        minimum_columns=1,
        cap_end_search=max(2, int(math.ceil(1.5*lw))),
        cap_center_gap=max(1, int(math.ceil(.65*lw))),
        minimum_cap_arm=max(2, int(math.ceil(1.25*lw))),
        maximum_cap_width=max(12, int(math.ceil(32.*lw))),
        maximum_cap_thickness=max(2, int(math.ceil(2.*lw))),
        cap_isolation_probe=max(2, int(math.ceil(lw))),
        cap_continuation_probe=max(2, int(math.ceil(1.5*lw))),
        stem_intersection_guard=max(1, int(math.ceil(lw))))
    ink = own & valid
    stem_mask, cap_mask = np.zeros(own.shape, bool), np.zeros(own.shape, bool)
    if ink.size and ink.any():
        proposals, opened, skipped = _vertical_proposals(ink, parameters)
        short_parameters = dict(parameters,
                                opening_height=max(3, 2*int(math.ceil(.60*lw))+1),
                                minimum_height=max(5, int(math.ceil(2.*lw))))
        short_proposals, _, short_skipped = _vertical_proposals(ink, short_parameters)
    else:
        proposals, opened, skipped = [], np.zeros_like(ink), 0
        short_proposals, short_skipped = [], 0
        short_parameters = None
    for proposal in proposals:
        proposal['proposal_kind'] = 'vertical_core'
    short_rejected_no_cap = 0
    for proposal in short_proposals:
        # This pass only restores genuinely SHORT surviving vertical evidence;
        # it does not retry long primary failures with looser settings.
        if proposal['y1']-proposal['y0']+1 >= parameters['minimum_height']:
            continue
        caps, _ = _caps_for_stem(ink, valid, proposal, parameters)
        if not caps:
            short_rejected_no_cap += 1
            continue
        proposal['proposal_kind'] = 'short_vertical_fragment'
        proposals.append(proposal)
    proposals.sort(key=lambda p: (p['x'], p['y0'], p['y1']))
    stems = []
    for index, proposal in enumerate(proposals, 1):
        support = proposal.pop('_support')
        a, b, c, d = proposal['x0'], proposal['y0'], proposal['x1'], proposal['y1']
        local_support = support & ink[b:d+1, a:c+1]
        stem_mask[b:d+1, a:c+1] |= local_support
        caps, safe_cap_pixels = _caps_for_stem(ink, valid, proposal, parameters)
        cap_mask |= safe_cap_pixels
        stems.append(dict(id=f'ST{index:03d}', **proposal, caps=caps,
                          source_support=float(local_support.mean()),
                          observed_pixels=int(local_support.sum()),
                          _support=local_support,
                          fragment_status=('permissive_short_cap_supported_x_only'
                                           if proposal['proposal_kind'] == 'short_vertical_fragment' else
                                           'permissive_single_column_x_only' if proposal['width'] == 1 else
                                           'permissive_vertical_x_only'),
                          touches_boundary=(a == 0 or b == 0 or c == own.shape[1]-1 or d == own.shape[0]-1),
                          interpretation='vertical structure / candidate x only; measurement y unresolved'))
    stems, merge_log = _merge_same_stem_fragments(stems, lw)
    # These are exact subset guarantees, including on image/ROI boundaries.
    stem_mask &= ink
    cap_mask &= ink
    cap_mask &= ~stem_mask  # protect intersections with every proposed stem
    for stem in stems:
        stem.pop('_support', None)
        for cap in stem['caps']:
            cap['cap_mask_pixels'] = int(cap_mask[cap['y0']:cap['y1']+1, cap['x0']:cap['x1']+1].sum())
    return dict(stems=stems, stem_mask=stem_mask, cap_mask=cap_mask,
                diagnostics=dict(version=VERSION, provenance=PROVENANCE,
                    coordinate_system='native plot-local pixel centres; all extrema inclusive',
                    input_line_width=float(line_width), effective_line_width=lw,
                    parameters=parameters, observed_ink_pixels=int(ink.sum()),
                    short_pass_parameters=short_parameters,
                    opened_pixels=int(opened.sum()), stem_count=len(stems),
                    cap_count=sum(len(s['caps']) for s in stems),
                    strict_cap_count=sum(c['strict'] for s in stems for c in s['caps']),
                    skipped_diagonal_or_too_narrow_groups=skipped,
                    short_pass_diagonal_groups_rejected=short_skipped,
                    short_fragments_rejected_without_cap=short_rejected_no_cap,
                    shared_cap_x_clusters=merge_log,
                    stem_mask_policy='permissive x proposals; not a safe curve-deletion mask',
                    cap_mask_policy='strict observed terminal cap arms only; stem intersection preserved',
                    limitations=['Vertical structures alone do not identify measurement y or series identity.',
                                 'Short, occluded, nearly vertical curved strokes or tightly merged stems may remain ambiguous.',
                                 'A cap and a coincident short curve can be observationally indistinguishable.',
                                 'No ideal marker glyph or midpoint measurement is fabricated.']))

"""Export native v46 active/tentative/path evidence to colour Step 5.

The JSON schema is consumed by color_step5_payload_v46.load_payload(). Marker
hypotheses enter init_suppressed only. Segments describe full estimated colour paths,
not chords between active markers and not mathematical marker-support knots.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

from color_step5_payload_v46 import VERSION, pixel_sha256
from color_tentative_policy_v46 import filter_strong_candidates, policy_info, strong_evidence


_MARKER_CLASSES = {
    'circle': 'filled_circle', 'square': 'filled_square',
    'diamond': 'filled_rhombus', 'triangle_up': 'filled_triangle',
    'triangle_down': 'filled_inv_triangle',
}
# Exact indices in 5_correction_color.CLASS_NAMES; series order is unrelated.
_CLASS_INDICES = {'filled_circle': 0, 'filled_square': 2, 'filled_triangle': 6,
                  'filled_inv_triangle': 7, 'filled_rhombus': 9}


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _inside(x, y, inclusive):
    return inclusive is not None and inclusive[0] <= x <= inclusive[2] and inclusive[1] <= y <= inclusive[3]


def _inclusive(box, image_shape, nullable=False):
    if box is None and nullable:
        return None
    if len(box) != 4:
        raise ValueError('A half-open xyxy box requires four coordinates')
    values = np.asarray(box, float)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError('ROI bounds must be finite integers')
    x0, y0, x1, y1 = map(int, values)
    if not (0 <= x0 < x1 <= image_shape[1] and 0 <= y0 < y1 <= image_shape[0]):
        raise ValueError('Half-open ROI must lie inside image')
    return [x0, y0, x1-1, y1-1]


def _point(point, plot, legend, *, suppressed, marker_class, class_index):
    result = dict(point)
    # Active x/y are the actual GUI/export pixels. Preserve x_px/y_px metadata
    # without letting an unrounded edge centre disagree with edit_data.json.
    if suppressed:
        x = result.get('cx', result.get('x_px', result.get('x')))
        y = result.get('cy', result.get('y_px', result.get('y')))
    else:
        x = result.get('x', result.get('cx', result.get('x_px')))
        y = result.get('y', result.get('cy', result.get('y_px')))
    x, y = float(x), float(y)
    if not np.isfinite([x, y]).all():
        raise ValueError('Marker coordinates must be finite')
    if not _inside(x, y, plot) or _inside(x, y, legend):
        return None
    result.update(cx=x, cy=y, class_name='suppressed' if suppressed else marker_class,
                  class_idx=-1 if suppressed else class_index)
    if suppressed:
        result.update(existence='unknown', tentative=True, state='suppressed', auto_promote=False)
    return result


def cluster_active_columns(active_by_name, diameters):
    """Frozen v1 rule for replaying historical exports; not the live exporter."""
    diameter = float(np.median(diameters)) if diameters else 12.
    tolerance = float(np.clip(.25*diameter, 1., 6.))
    values = sorted((float(p['cx']), name, i) for name, points in active_by_name.items()
                    for i, p in enumerate(points))
    groups = []
    for item in values:
        if not groups or item[0]-groups[-1][0][0] > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    diagnostics = [dict(x=float(np.median([p[0] for p in group])),
                        active_count=len(group), series=sorted(set(p[1] for p in group)),
                        spread_px=float(np.ptp([p[0] for p in group]))) for group in groups]
    grid = [p['x'] for p in diagnostics] if len(groups) >= 2 else []
    return grid, dict(source='existing_active_marker_x_only', groups=diagnostics,
                      cluster_tolerance_px=tolerance, extrapolated_columns=0,
                      tentative_points_used=False, mathematical_knots_used=False,
                      status='observed_columns' if grid else 'fewer_than_two_active_x_groups_no_grid')


def cluster_marker_columns(active_by_name, diameters, suppressed_by_name=None):
    """Bounded 1-D clustering of active and strong suppressed source x values.

    Merge nearest adjacent groups only when their *full* combined span is at
    most .75 marker diameters. This is complete-link clustering, not a mode
    estimate or a transitive chain of close neighbours. One series gets one
    vote: its active median (weight 1), or suppressed median (weight .5).
    Suppressed remains unconfirmed and no marker coordinates are changed.

    Rare unresolved groups closer than .5 diameters compete for one grid
    boundary. The losing group is recorded, not merged through a wide chain
    and not removed from either marker pool. No uniform sampling is assumed.
    """
    valid_diameters = [float(d) for d in diameters
                       if d is not None and np.isfinite(float(d)) and float(d) > 0]
    diameter = max(1., float(np.median(valid_diameters)) if valid_diameters else 12.)
    max_span, min_separation = .75*diameter, .5*diameter
    values, excluded_suppressed = [], 0
    for state, pool in (('active', active_by_name), ('suppressed', suppressed_by_name or {})):
        for name, points in pool.items():
            indexed = list(enumerate(points))
            if state == 'suppressed':
                # Enforce the same policy even for callers bypassing the exporter.
                indexed = [(i, p) for i, p in indexed if strong_evidence(p)]
                excluded_suppressed += len(points)-len(indexed)
            for index, point in indexed:
                x = float(point['cx'])
                if not np.isfinite(x):
                    raise ValueError('Grid marker x coordinates must be finite')
                values.append(dict(x=x, series=name, state=state, index=index,
                                   candidate_id=point.get('candidate_id')))
    values.sort(key=lambda p: (p['x'], p['series'], p['state'], p['index']))
    groups = [[p] for p in values]
    while len(groups) > 1:
        spans = [right[-1]['x']-left[0]['x'] for left, right in zip(groups, groups[1:])]
        closest = int(np.argmin(spans))
        if spans[closest] > max_span:
            break
        groups[closest:closest+2] = [groups[closest]+groups[closest+1]]

    def describe(group):
        series = sorted(set(p['series'] for p in group))
        votes = []
        for name in series:
            current = [p for p in group if p['series'] == name]
            active_x = sorted(set(p['x'] for p in current if p['state'] == 'active'))
            chosen = active_x or sorted(set(p['x'] for p in current))
            votes.append(dict(series=name, x=float(np.median(chosen)),
                              weight=1. if active_x else .5,
                              source='active' if active_x else 'suppressed'))
        votes.sort(key=lambda v: (v['x'], v['series']))
        cumulative = np.cumsum([v['weight'] for v in votes])
        half = cumulative[-1]/2
        mid = int(np.searchsorted(cumulative, half))
        center = votes[mid]['x']
        if cumulative[mid] == half and mid+1 < len(votes):
            center = .5*(center+votes[mid+1]['x'])
        return dict(x=center, active_count=sum(p['state'] == 'active' for p in group),
                    suppressed_count=sum(p['state'] == 'suppressed' for p in group),
                    active_series_count=sum(v['source'] == 'active' for v in votes),
                    series=series, spread_px=group[-1]['x']-group[0]['x'],
                    series_votes=votes, members=group)

    proposals = [describe(g) for g in groups]
    ranked = sorted(proposals, key=lambda g: (-g['active_series_count'], -len(g['series']),
                                               g['spread_px'], g['x']))
    retained, conflicts = [], []
    for group in ranked:
        neighbours = [g for g in retained if abs(g['x']-group['x']) < min_separation]
        if neighbours:
            other = min(neighbours, key=lambda g: abs(g['x']-group['x']))
            conflicts.append(dict(group, reason='unresolved_close_column_not_a_marker_rejection',
                                  conflict_with_x=other['x']))
        else:
            retained.append(group)
    retained.sort(key=lambda g: g['x'])
    grid = [g['x'] for g in retained] if len(retained) >= 2 else []
    return grid, dict(source='existing_active_and_strong_suppressed_x',
        algorithm='bounded_complete_link_series_weighted_median_v2',
        groups=retained, unresolved_close_groups=conflicts,
        marker_diameter_px=diameter, maximum_cluster_span_px=max_span,
        minimum_column_separation_px=min_separation,
        active_vote_weight=1., suppressed_vote_weight=.5,
        input_active_count=sum(p['state'] == 'active' for p in values),
        input_suppressed_count=sum(p['state'] == 'suppressed' for p in values),
        excluded_non_strong_suppressed=excluded_suppressed,
        tentative_points_used=any(p['state'] == 'suppressed' for p in values),
        mathematical_knots_used=False, extrapolated_columns=0, uniform_spacing_enforced=False,
        marker_coordinates_changed=False,
        status='observed_columns' if grid else 'fewer_than_two_marker_x_groups_no_grid')


def _safe_segments(path_record, plot, legend, grid_xs, callback):
    """Split rejected samples, missing x columns and excluded ROI before v45 DP."""
    path = np.asarray(path_record.get('path', []), float).reshape(-1, 2)
    filled = np.asarray(path_record.get('filled', []), bool)
    observed = np.asarray(path_record.get('observed', filled), bool)
    if filled.shape != (len(path),) or observed.shape != filled.shape or not np.isfinite(path).all():
        raise ValueError('Path coordinates and retention arrays must agree and be finite')
    if len(path) > 1 and np.any(np.diff(path[:, 0]) <= 0):
        raise ValueError('Colour path x must be strictly increasing')
    scope = np.array([_inside(x, y, plot) and not _inside(x, y, legend) for x, y in path], bool)
    keep = filled & scope
    runs, current = [], []
    for i, point in enumerate(path):
        discontinuity = bool(current and point[0]-path[i-1, 0] > 1.5)
        if (not keep[i] or discontinuity) and current:
            runs.append(current)
            current = []
        if keep[i]:
            current.append(point.tolist())
    if current:
        runs.append(current)
    segments = []
    rejected_segments = 0
    for run in runs:
        if len(run) < 3:
            continue
        points = np.asarray(run, float)
        proposed = callback(points, np.ones(len(points), bool), grid_xs, dev=2., min_len=5.)
        for segment in proposed:
            values = np.asarray(segment, float)
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError('path_to_segments returned an invalid segment')
            # A DP shortcut around a legend corner must not cross the legend.
            samples = max(2, int(np.ceil(max(abs(values[2]-values[0]), abs(values[3]-values[1]))))*2+1)
            xs, ys = np.linspace(values[0], values[2], samples), np.linspace(values[1], values[3], samples)
            if not all(_inside(x, y, plot) and not _inside(x, y, legend) for x, y in zip(xs, ys)):
                rejected_segments += 1
                continue
            segments.append(values.tolist())
    return segments, dict(source='directional_colour_path_retained_runs_to_v45_path_to_segments',
        raw_path_samples=len(path), retained_path_samples=int(keep.sum()),
        observed_path_samples=int((observed & keep).sum()),
        short_gap_or_weak_samples=int((keep & ~observed).sum()), retained_runs=len(runs),
        rejected_outside_or_legend_segments=rejected_segments,
        deviation_px=2., minimum_segment_length_px=5., marker_chords_used=False,
        disconnected_path_runs_never_joined=True)


def _overview(destination, image, plot, curves):
    x0, y0, x1, y1 = plot
    crop = image[y0:y1+1, x0:x1+1]
    h, w = crop.shape[:2]
    header, footer, gutter = 94, 54+26*len(curves), 24
    canvas = np.full((header+h+footer, 2*w+3*gutter, 3), 255, np.uint8)
    for column in range(2):
        left = gutter+(w+gutter)*column
        faded = cv2.addWeighted(crop, .70, np.full_like(crop, 255), .30, 0)
        canvas[header:header+h, left:left+w] = faded
    def text(value, x, y, scale=.58, colour=(35,35,35)):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    text('v46 Step 5 input evidence (not correction output)', gutter, 26, .72)
    text('Active detections: green circles', gutter, 57)
    text('Strong-only suppressed: orange diamonds; source paths: series-colour segments', gutter, 80, .51)
    text('Existing active points only', gutter, header-8, .43)
    text('Suppressed review candidates + retained path segments', 2*gutter+w, header-8, .43)
    for row_index, row in enumerate(curves):
        colour = tuple(int(round(v)) for v in row['rgb'][::-1])
        for point in row['init_points']:
            pt = (round(point['cx']-x0+gutter), round(point['cy']-y0+header))
            cv2.circle(canvas, pt, 7, (255,255,255), 4, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (40,145,25), 2, cv2.LINE_AA)
        for a,b,c,d in row['segments_override']:
            p = (round(a-x0+2*gutter+w),round(b-y0+header))
            q = (round(c-x0+2*gutter+w),round(d-y0+header))
            cv2.line(canvas,p,q,(255,255,255),4,cv2.LINE_AA)
            cv2.line(canvas,p,q,colour,2,cv2.LINE_AA)
        for point in row['init_suppressed']:
            px, py = round(point['cx']-x0+2*gutter+w), round(point['cy']-y0+header)
            vertices = np.array([[px,py-7],[px+7,py],[px,py+7],[px-7,py]],np.int32)
            cv2.polylines(canvas,[vertices],True,(255,255,255),4,cv2.LINE_AA)
            cv2.polylines(canvas,[vertices],True,(0,110,245),2,cv2.LINE_AA)
        text(f"{row['name']}: {len(row['init_points'])} active | {len(row['init_suppressed'])} tentative | "
             f"{len(row['segments_override'])} path segments | {row['marker_class']}",
             gutter,header+h+26+row_index*26,.48)
    text('Tentative positions remain unknown. Gaps are not joined; no path-derived marker knots are exported.',
         gutter, canvas.shape[0]-12, .45)
    if not cv2.imwrite(str(destination/'color_step5_inputs_v46.png'), canvas):
        raise OSError('Cannot save v46 Step-5 overview PNG')


def export_step5_inputs(destination, image, plot, legend, entries, names, templates,
                       active_by_name, tentative, own_masks, path_to_segments):
    """Write a validated Step-5 payload, evidence archive and input overview.

    Input boxes are half-open. JSON plot_area/legend_box are inclusive as used
    by v45/GUI correction. Positions outside that integer-pixel domain are
    omitted rather than silently shifted to a different marker centre.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Step-5 reference image must be native uint8 BGR')
    inclusive_plot = _inclusive(plot, image.shape)
    inclusive_legend = _inclusive(legend, image.shape, nullable=True)
    if len(entries) != len(names) or len(set(names)) != len(names):
        raise ValueError('Entries and unique curve names must match')
    if any(not isinstance(name, str) or not name or Path(name).name != name for name in names):
        raise ValueError('Curve names must be safe directory names')
    archive, rows, diameters, cleaned_active = {}, [], [], {}
    paths = {str(p['series_id']): p for p in tentative.get('paths', [])}
    unknown_candidates = [p for p in tentative.get('candidates', []) if p.get('series_id') not in names]
    if unknown_candidates:
        raise ValueError('Tentative candidates contain an unknown series identity')
    counts = dict(active=0, tentative_suppressed=0, segments=0, excluded_active=0,
                  excluded_tentative=0, excluded_diagnostic_knots=0, excluded_non_strong=0)
    for index, (name, entry) in enumerate(zip(names, entries)):
        template = templates.get(name)
        composition = (template or {}).get('composition') or entry.get('composition') or {}
        model_name = composition.get('best_model_name', composition.get('best_model', {}).get('name'))
        marker_class = _MARKER_CLASSES.get(model_name, 'native_template' if template is not None else 'unsupported_no_template')
        engine_class_index = _CLASS_INDICES.get(marker_class, -1)
        active = []
        for p in active_by_name.get(name, []):
            q = _point(p, inclusive_plot, inclusive_legend, suppressed=False,
                       marker_class=marker_class, class_index=engine_class_index)
            if q is None:
                counts['excluded_active'] += 1
            else:
                active.append(q)
        cleaned_active[name] = active
        mask = np.asarray(own_masks[name], bool).copy()
        expected_shape = (inclusive_plot[3]-inclusive_plot[1]+1, inclusive_plot[2]-inclusive_plot[0]+1)
        if mask.shape != expected_shape:
            raise ValueError('Every own mask must use plot-local source pixels')
        if inclusive_legend is not None:
            a,b,c,d = inclusive_legend
            a,b,c,d = max(a,plot[0])-plot[0],max(b,plot[1])-plot[1],min(c+1,plot[2])-plot[0],min(d+1,plot[3])-plot[1]
            if a<c and b<d:
                mask[b:d,a:c] = False
        mask_key, template_key = f'mask_{index:03d}', None
        archive[mask_key] = mask
        center, diameter = None, None
        if template is not None:
            alpha = np.asarray(template['soft'], np.float32)
            if alpha.ndim != 2 or not alpha.size or not np.isfinite(alpha).all() or (alpha<0).any() or (alpha>1).any():
                raise ValueError('Native template alpha must be a finite [0,1] plane')
            center = list(map(float, template.get('center', [(alpha.shape[1]-1)/2,(alpha.shape[0]-1)/2])))
            if len(center)!=2 or not np.isfinite(center).all() or not (0<=center[0]<alpha.shape[1] and 0<=center[1]<alpha.shape[0]):
                raise ValueError('Template centre must lie inside native alpha raster')
            template_key = f'template_{index:03d}'
            archive[template_key] = alpha
            diameter = float(template['diameter'])
            diameters.append(diameter)
        suppressed = []
        image_candidates = []
        for p in tentative.get('candidates', []):
            if p.get('series_id') != name:
                continue
            if p.get('kind') == 'diagnostic_support_knot' or p.get('origin') == 'diagnostic_support_knot':
                counts['excluded_diagnostic_knots'] += 1
                continue
            image_candidates.append(p)
        strong_candidates, strong_audit = filter_strong_candidates(image_candidates)
        counts['excluded_non_strong'] += strong_audit['excluded_count']
        for p in strong_candidates:
            q = _point(p, inclusive_plot, inclusive_legend, suppressed=True,
                       marker_class=marker_class, class_index=engine_class_index)
            if q is None or any((q['cx']-a['cx'])**2+(q['cy']-a['cy'])**2 <= 1. for a in active):
                counts['excluded_tentative'] += 1
            else:
                suppressed.append(q)
        thickness = composition.get('line_params', {}).get('thickness')
        linewidth = float(thickness) if thickness is not None and np.isfinite(thickness) and thickness > 0 else None
        rows.append(dict(name=name,rgb=list(map(float,entry['rgb'])),class_idx=engine_class_index,series_index=index,
            marker_class=marker_class,marker_model_name=model_name,mask_key=mask_key,
            template_key=template_key,template_center=center,diameter=diameter,
            template_geometry='native_alpha' if template is not None else 'unavailable',
            template_provenance=(template or {}).get('provenance',{}),
            center_convention=(template or {}).get('center_convention',{}),
            symbol_scale=(template or {}).get('symbol_scale',1.),
            legend_diameter=(template or {}).get('legend_diameter',diameter),
            symbol_scale_status=(template or {}).get('symbol_scale_status','legacy_fixed_geometry'),
            render_linewidth=linewidth,render_linewidth_source='legend_composition_line_thickness' if linewidth else 'estimate_from_own_mask_at_correction',
            init_points=active,init_suppressed=suppressed,segments_override=[],
            suppressed_filter=strong_audit))
        # Bind native colour/window evidence for candidate-only correction.
        # No source swatch re-extraction or independently adjusted marker scale.
        from triangle_errorbar_v46 import triangle_prior
        rows[-1]['triangle_prior'] = triangle_prior(entry.get('report', {}))
        if template is not None:
            keys = ('id', 'rgb', 'soft', 'raw_soft', 'center', 'diameter', 'model', 'raw_bgr',
                    'weight', 'core', 'envelope', 'nuisance', 'central_connector', 'uncertain',
                    'face', 'hole_core', 'enclosed_paper', 'boundary_uncertain', 'observed_rim',
                    'completion_hidden_alpha', 'completion_visible_alpha',
                    'symbol_scale', 'legend_diameter', 'symbol_scale_status', 'symbol_scale_version')
            rows[-1]['verification_template'] = _plain({k: template[k] for k in keys if k in template})
    grid, grid_diagnostics = cluster_marker_columns(
        cleaned_active, diameters, {row['name']: row['init_suppressed'] for row in rows})
    for row in rows:
        name = row['name']
        from color_estimated_path_v46 import full_path_segments
        geometry = full_path_segments(paths.get(name,{}), inclusive_plot, inclusive_legend, grid, path_to_segments)
        segments, provenance = geometry['segments'], geometry['provenance']
        row.update(segments_override=segments,segment_provenance=provenance)
        counts['active'] += len(row['init_points'])
        counts['tentative_suppressed'] += len(row['init_suppressed'])
        counts['segments'] += len(segments)
    payload = dict(version=VERSION,image_sha256=pixel_sha256(image),image_shape=list(image.shape),
        image_hash_definition='SHA256 of contiguous native uint8 BGR image bytes, not encoded file bytes',
        plot_area=inclusive_plot,legend_box=inclusive_legend,grid_xs=grid,grid_diagnostics=grid_diagnostics,
        masks_npz='color_step5_evidence_v46.npz',curves=rows,counts=counts,
        reference_paths=_plain({name: paths[name] for name in names if name in paths}),
        path_correction=dict(version='v46_path_shape_v3',metric='chamfer',model='pchip',
                             reference_policy='estimated_path',inferred_weight=.25),
        suppressed_policy=policy_info(),
        coordinate_convention='source-image pixel centres; inclusive xyxy plot_area and legend_box',
        tentative_source=tentative.get('version'),diagnostic_knots_exported=False,
        notes=['init_suppressed contains only strong-image-evidence hypotheses; existence remains unknown.',
               'Weak hypotheses remain in color_tentative_v46.json, not in correction inputs.',
               'segments_override is explicit even when empty; no detector fallback or marker-chord substitution.',
               'Active and strong suppressed points contribute bounded x clusters; no periodic extrapolation.',
               'Grid grouping never moves marker centres or promotes suppressed hypotheses.',
               'Full existing estimated paths are included; ROI, legend exclusions and missing x columns still split pieces.',
               'Original observed/filled flags remain unchanged in reference_paths; inferred geometry is not image evidence.',
               'Colour Step-5 can subsequently propose ADD/REPLACE/DELETE/ACTIVATE; export itself promotes no marker.'])
    np.savez_compressed(destination/payload['masks_npz'], **archive)
    (destination/'step5_inputs.json').write_text(json.dumps(_plain(payload),indent=2,allow_nan=False),encoding='utf-8')
    _overview(destination,image,inclusive_plot,rows)
    return payload

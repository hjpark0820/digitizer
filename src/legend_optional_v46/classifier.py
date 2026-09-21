"""Fresh native-image evidence for marker, line-only, or uncertain routing.

A line legend establishes series colour only. Filled round markers are an
explicit working hypothesis; the scanner still requires observed compact ink.
Failure to find a recurrent radius never establishes marker absence on its own.
All public point/path coordinates are full-image native pixel centres. Density
arrays and radius-support audit coordinates remain plot-local and are labelled.
"""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from color_marker_evidence import _membership, colour_evidence
from color_path_directional_v46 import DirectionalConfig, trace_directional
from .disk import (Config, estimate_common_radius, measure_line_width,
                   normalized_disk_density, retain_observed_occluders, scan_path)


VERSION = 'legend_optional_disk_and_line_evidence_v46_v1'


def _inputs(image_bgr, plot_box, palette, valid):
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected a native uint8 BGR image')
    bounds = np.asarray(plot_box)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all()
            or not np.equal(bounds, np.floor(bounds)).all()):
        raise ValueError('Plot box must contain four integer half-open coordinates')
    x0, y0, x1, y1 = map(int, bounds)
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError('Plot box must be nonempty and inside the native image')
    rows = deepcopy(list(palette))
    ids = [s['id'] for s in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('At least one uniquely identified observed series is required')
    for s in rows:
        s.setdefault('label', s['id'])
        s.setdefault('role', 'data_series')
        # colour_evidence performs subtraction between these model vectors.
        s['model']['bgr'] = np.asarray(s['model']['bgr'], np.float32)
        s['model']['paper_bgr'] = np.asarray(s['model']['paper_bgr'], np.float32)
    crop = image[y0:y1, x0:x1]
    allowed = np.ones(crop.shape[:2], bool)
    if valid is not None:
        if np.shape(valid) != allowed.shape:
            raise ValueError('Validity mask must match the plot-local image shape')
        allowed &= np.asarray(valid, bool)
    # The box includes the axis boundary in native v46 ROIs. A source-pixel
    # inset is spatial exclusion, with no source-colour erasure or rescaling.
    allowed[[0, -1], :] = False
    allowed[:, [0, -1]] = False
    return crop, rows, allowed, [x0, y0, x1, y1]


def _trace(own, valid, width, padding):
    support = (own >= .35) & valid
    columns = np.flatnonzero(support.sum(axis=0) >= max(1., .5*width))
    if not len(columns):
        return None
    yy = np.flatnonzero(support.any(axis=1))
    return trace_directional(own, int(columns[0]), int(columns[-1]),
        max(0, int(yy[0])-padding), min(own.shape[0]-1, int(yy[-1])+padding),
        DirectionalConfig())


def _observed_legend_width(image, row):
    """Measure a narrow observed line key independently of plot JPEG speckles."""
    box = row.get('box') or row.get('swatch_box')
    if box is None:
        return None
    x0, y0, x1, y1 = map(int, box)
    if (not 0 <= x0 < x1 <= image.shape[1] or not 0 <= y0 < y1 <= image.shape[0]
            or x1-x0 < 3*(y1-y0)):
        return None
    soft, _ = _membership(image[y0:y1, x0:x1], row['model'])
    counts = (soft >= .35).sum(axis=0)
    positive = counts[counts > 0]
    if len(positive) < .75*(x1-x0):
        return None
    return float(np.median(positive))


def _compact_core_count(distance, width):
    """Count even isolated thick bodies, which preclude marker-free certainty."""
    peak = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    thick = (distance >= peak-1e-5) & (distance >= max(2.8, 1.35*width))
    count, _ = cv2.connectedComponents(thick.astype(np.uint8), connectivity=8)
    return int(count-1)


def _unexplained_dense_windows(own, valid, xy, dense, radius, width):
    """Test dense windows against independently long observed straight strokes.

    A curve, stem and nearby caps can jointly fill a small disk. Segment votes
    must extend beyond the disk diameter, so a disk by itself cannot supply the
    explaining line. This affects classification evidence only; original ink
    and the exact production disk scanner remain unchanged.
    """
    unexplained = []
    extent = math.ceil(4*radius)
    for j in np.flatnonzero(dense):
        cx, cy = np.rint(xy[j]).astype(int)
        x0, x1 = max(0, cx-extent), min(own.shape[1], cx+extent+1)
        y0, y1 = max(0, cy-extent), min(own.shape[0], cy+extent+1)
        patch = own[y0:y1, x0:x1]
        available = valid[y0:y1, x0:x1]
        binary = ((patch >= .35) & available).astype(np.uint8)
        minimum_length = math.ceil(3*radius)
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=minimum_length,
            minLineLength=minimum_length, maxLineGap=1)
        explained = np.zeros_like(binary)
        if lines is not None:
            for ax, ay, bx, by in np.asarray(lines).reshape(-1, 4):
                cv2.line(explained, (int(ax), int(ay)), (int(bx), int(by)),
                         1, max(1, math.ceil(width)))
        yy, xx = np.indices(patch.shape)
        scope = ((xx-(xy[j, 0]-x0))**2+(yy-(xy[j, 1]-y0))**2 <= radius**2) & available
        residual = float((patch*(1-explained))[scope].mean()) if scope.any() else 1.
        if residual >= Config().own_compact_residual:
            unexplained.append(int(j))
    return unexplained


def _compact_cavity_count(own, valid, xy, width):
    """Visible compact enclosed gaps guard against mistaking hollow glyphs for lines."""
    binary = ((own >= .35) & valid).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not len(xy):
        return 0
    count = 0
    for contour, node in zip(contours, np.asarray(hierarchy).reshape(-1, 4)):
        if node[3] < 0 or cv2.contourArea(contour) < max(4., width):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 3 or max(w, h) > max(24., 8*width) or max(w, h) > 3*min(w, h):
            continue
        center = np.array([x+(w-1)/2, y+(h-1)/2])
        j = int(np.argmin(abs(xy[:, 0]-center[0])))
        if np.linalg.norm(xy[j]-center) <= max(4., 2*width):
            # Small closed gaps can also lie between an error-bar cap and its
            # curve. Require enclosing ink unexplained by long source strokes.
            unresolved = _unexplained_dense_windows(own, valid, center[None, :],
                np.ones(1, bool), max(3., 1.5*width), width)
            count += bool(unresolved)
    return count


def _line_evidence(own, other, valid, width, distance, trace, legend_width=None):
    """Positive evidence for a supported ordinary stroke, with sparse guards.

    Thinness uses inscribed ink width on the observed curve. A second disk-area
    check catches compact shoulders/rims that a thin interior ridge can miss;
    broad vertical error bars alone do not fill a disk. Unexplained compact ink
    makes the decision uncertain, even if most of the curve is a thin line.
    """
    shape = own.shape
    # A colour's global 25th-percentile ridge can measure JPEG fragments instead
    # of its actual stroke. A directly observed plain legend line provides an
    # independent width for classifying plot bodies; disk sizing stays unchanged.
    classification_width = max(width, legend_width or 0.)
    audit = dict(compact_core_count=_compact_core_count(distance, classification_width),
                 native_plot_shape=list(shape), measured_line_width=float(width),
                 legend_line_width=legend_width,
                 classification_stroke_width=float(classification_width))
    if trace is None:
        audit.update(passes=False, failures=['no_observed_curve_columns'],
                     observed_column_count=0, path_column_count=0)
        return audit
    xy = np.asarray(trace['path'])
    width = classification_width
    ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, shape[1]-1)
    iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, shape[0]-1)
    observed = (np.asarray(trace['val']) >= .35) & valid[iy, ix]
    n = int(observed.sum())
    fraction = float(observed.mean()) if len(observed) else 0.
    supported_distance = distance[iy[observed], ix[observed]]
    ridge_middle = float(np.percentile(supported_distance, 50)) if n else 0.
    ridge_high = float(np.percentile(supported_distance, 95)) if n else 0.
    baseline_radius = max(ridge_middle, (width+.5)/2)
    thick_fraction = float(np.mean(supported_distance > max(2.4, 1.55*baseline_radius))) if n else 1.
    probe_radius = max(3., 1.5*width)
    density = normalized_disk_density(own, valid, probe_radius)[iy, ix]
    dense = observed & (density >= .68)
    dense_count = int(dense.sum())
    unexplained_dense = _unexplained_dense_windows(own, valid, xy, dense, probe_radius, width)
    cavity_count = _compact_cavity_count(own, valid, xy, width)
    ambiguity = (other[iy, ix] >= .2) & observed
    ambiguity_fraction = float(ambiguity.sum()/max(n, 1))
    span = float(xy[-1, 0]-xy[0, 0]+1) if len(xy) else 0.
    conditions = dict(
        sufficient_native_resolution=min(shape) >= 64 and shape[1] >= 96,
        many_observed_columns=n >= max(64, math.ceil(.25*shape[1])),
        substantial_curve_span=span >= max(64, .30*shape[1]),
        sustained_colour_support=fraction >= .75,
        ordinary_stroke_width=width <= max(5., .025*min(shape)),
        no_isolated_thick_bodies=audit['compact_core_count'] == 0,
        no_compact_hollow_bodies=cavity_count == 0,
        stable_thin_curve=thick_fraction <= .025 and ridge_high <= max(2.4, 1.55*baseline_radius),
        no_unexplained_compact_windows=len(unexplained_dense) == 0,
        unambiguous_curve_colour=ambiguity_fraction <= .08)
    audit.update(observed_column_count=n, path_column_count=len(xy),
        observed_fraction=fraction, horizontal_span_px=span,
        ridge_median_px=ridge_middle, ridge_p95_px=ridge_high,
        thick_curve_fraction=thick_fraction, compact_probe_radius_px=probe_radius,
        dense_compact_column_count=dense_count,
        unexplained_compact_column_count=len(unexplained_dense),
        compact_cavity_count=cavity_count,
        other_colour_fraction=ambiguity_fraction,
        conditions=conditions, passes=all(conditions.values()),
        failures=[name for name, passed in conditions.items() if not passed])
    return audit


def classify_series(image_bgr, plot_box, palette, valid=None):
    """Classify source evidence and scan disks without choosing GUI behavior.

    ``palette`` contains id/rgb/model and optional label/role/legend provenance.
    ``valid`` is plot-local. Every colour remains in fields, including colours
    whose path/body evidence fails. Callers may pass the returned series,
    membership masks and valid mask to type3_v46.pipeline.detect only when the
    explicit ``chart_kind`` is ``line-only``.
    """
    crop, rows, allowed, plot = _inputs(image_bgr, plot_box, palette, valid)
    fields = colour_evidence(crop, rows)
    fields['valid'] = allowed
    fields['coordinate_system'] = 'plot-local native pixels'
    # Preserve original continuous fields for diagnostics/occlusion. Restricted
    # copies are used only in source-domain computations.
    own = fields['membership']*allowed
    total = fields['all_ink']*allowed
    other = fields['other']*allowed
    widths_and_distances = [measure_line_width(a, allowed) for a in own]
    radius = None
    try:
        radius, size = estimate_common_radius(own, allowed)
        size['status'] = 'recurrent_radius_supported'
    except ValueError as error:
        size = dict(status='unresolved', reason=str(error),
                    failure_implies_no_markers=False)
    size['coordinate_system'] = 'plot-local native pixels'
    config = Config(use_long_line_guard=True)
    origin = np.asarray(plot[:2], float)
    enriched, paths, line_audits = [], [], []
    for index, row in enumerate(rows):
        sid = row['id']
        width, distance = widths_and_distances[index]
        trace = _trace(own[index], allowed, width, math.ceil(radius or 2*width))
        line_audit = _line_evidence(own[index], other[index], allowed, width, distance, trace,
                                    _observed_legend_width(np.asarray(image_bgr), row))
        line_audit['series_id'] = sid
        line_audits.append(line_audit)
        base = {key: deepcopy(value) for key, value in row.items() if key != 'model'}
        base.update(line_width=width, candidates=[], profile={}, path_xy=np.empty((0, 2)))
        if trace is None:
            base['status'] = 'no_colour_support'
            enriched.append(base)
            continue
        raw_xy = np.asarray(trace['path'])
        ix, iy = np.rint(raw_xy).astype(int).T
        observed = (np.asarray(trace['val']) >= .35) & allowed[iy, ix]
        full_xy = raw_xy+origin
        paths.append(dict(series_id=sid, raw_path_source=full_xy,
            raw_observed=observed, path_color_values=trace['val'],
            cost=trace['cost'], coordinate_system='full-image native pixels'))
        if radius is not None:
            found = scan_path(raw_xy, own[index], total, other[index], allowed,
                              radius, width, config)
            found['path_xy'] += origin
            found['profile']['x'] += origin[0]
            found['profile']['y'] += origin[1]
            for candidate in found['candidates']:
                candidate['x'] += origin[0]
                candidate['y'] += origin[1]
                candidate['path_x'] += origin[0]
                candidate['path_y'] += origin[1]
                candidate.update(id=sid+'_'+candidate['id'], series_id=sid,
                    x_px=candidate['x'], y_px=candidate['y'],
                    kind='disk_marker_hypothesis', marker_glyph_detected=candidate['status'] == 'active')
            base.update(found, diameter=2*radius)
        else:
            base.update(path_xy=full_xy, profile=dict(x=full_xy[:, 0].copy(),
                                                     y=full_xy[:, 1].copy()))
        base.update(path_cost=trace['cost'], status='source_evidence_scanned')
        enriched.append(base)
    if radius is not None:
        enriched = retain_observed_occluders(enriched, config)
    candidates = [p for s in enriched for p in s['candidates']]
    points = [p for p in candidates if p['status'] == 'active']
    suppressed = [p for p in candidates if p['status'] == 'suppressed']
    for s in enriched:
        s['counts'] = dict(Counter(p['status'] for p in s['candidates']))
    equivalent = np.asarray(fields['equivalent_colours'], bool)
    rgb = np.asarray(fields['palette_rgb'], float)
    distances = np.linalg.norm(rgb[:, None]-rgb[None, :], axis=-1)
    sigma = np.asarray([row['model']['residual_sigma'] for row in rows], float)
    indistinguishable = equivalent & (distances <= np.maximum(12., 2*np.maximum(sigma[:, None], sigma[None, :])))
    ambiguous_palette = bool((indistinguishable & ~np.eye(len(rows), dtype=bool)).any())
    if radius is not None and len(points) >= 3 and not ambiguous_palette:
        chart_kind, reason = 'markers', 'recurrent_common_radius_and_path_supported_compact_bodies'
    elif (not ambiguous_palette and line_audits
          and all(a['passes'] for a in line_audits)):
        chart_kind, reason = 'line-only', 'sustained_thin_source_curves_without_compact_body_evidence'
    else:
        chart_kind, reason = 'uncertain', 'insufficient_or_conflicting_source_evidence'
    if chart_kind == 'line-only':
        for row, line_audit in zip(enriched, line_audits):
            row['line_width'] = line_audit['classification_stroke_width']
    audit = dict(decision=chart_kind, reason=reason, size_estimation=size,
        line_only_evidence=line_audits, ambiguous_palette=ambiguous_palette,
        palette_ambiguity_method='occlusion_equivalence_and_rgb_distance_within_two_colour_residual_sigmas',
        radius_proposal_overruled_by_positive_stroke_evidence=chart_kind == 'line-only' and radius is not None,
        marker_hypothesis='filled round bodies with one recurrent common radius',
        resolution_limitation='Glyphs no wider than the resolved connecting stroke cannot be distinguished reliably',
        radius_failure_alone_selects_line_only=False,
        active_body_count=len(points), suppressed_occlusion_count=len(suppressed),
        native_resolution=True, uses_stored_detections=False,
        thresholds_apply_identically_to_all_series=True)
    return dict(version=VERSION, chart_kind=chart_kind, audit=audit,
        radius=radius, size_estimation=size, series=enriched, paths=paths,
        points=points, suppressed=suppressed, candidates=candidates, fields=fields,
        plot_box=plot, coordinate_system='full-image native pixel centres',
        native_resolution=True)

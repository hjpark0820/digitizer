"""Observed legend evidence with explicit identity and paper-hole boundaries.

Version 1 is deliberately imported without mutation.  Pixels and supplied
structural boxes are the only inference inputs; labels remain output metadata.
"""
from __future__ import annotations

import math
from copy import deepcopy

import cv2
import numpy as np

import color_marker_evidence as base

VERSION = "image_only_palette_and_legend_evidence_v2"


def _two_dimensional(mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 6:
        return False
    eigenvalues = np.linalg.eigvalsh(np.cov(np.column_stack((xx, yy)).T))
    return bool(eigenvalues[0] >= .65 and
                eigenvalues[0] / max(1., eigenvalues[-1]) >= .12)


def _observed_body(support):
    """Locate compact ink or a two-dimensional excursion from a connector.

    Coordinates along/across the observed connector replace horizontal-row
    assumptions.  Only the crop boundary is inferred: returned template ink
    always comes from the original pixels, including ink in the connector band.
    """
    a, b, c, d = base._tight(support)
    yy, xx = np.nonzero(support)
    points = np.column_stack((xx, yy)).astype(np.float32)
    if max(c-a, d-b) <= 1.9*min(c-a, d-b) and _two_dimensional(support):
        return (a, b, c, d), None
    covariance = np.cov(points.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    primary = eigenvectors[:, -1]
    angle = math.atan2(float(primary[1]), float(primary[0]))
    candidates = []
    # PCA is exact for symmetric rotated keys; nearby measured orientations
    # allow an asymmetric marker to perturb the overall covariance slightly.
    for theta in angle + np.deg2rad(np.array([-8., -4., 0., 4., 8.])):
        axis = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-axis[1], axis[0]])
        u, v = points@axis, points@normal
        lo, hi = np.percentile(u, [2, 98])
        tails = (u <= lo+.18*(hi-lo)) | (u >= hi-.18*(hi-lo))
        if tails.sum() < 6:
            continue
        mid_v = float(np.median(v[tails]))
        half_stroke = max(.65, float(np.percentile(np.abs(v[tails]-mid_v), 90)))
        outside = np.abs(v-mid_v) > half_stroke+.8
        if outside.sum() < 6:
            continue
        outside_u, outside_v = u[outside], v[outside]
        across = float(np.ptp(outside_v)+1)
        if across < max(4., 3.*half_stroke):
            continue
        # Both sides must contain actual marker ink; a cap on one side alone
        # does not establish a complete marker body.
        if min(np.sum(outside_v > mid_v), np.sum(outside_v < mid_v)) < 2:
            continue
        mid_u = float(.5*(outside_u.min()+outside_u.max()))
        half_body = max(.5*across, .5*float(np.ptp(outside_u)+1))+1.
        if 2*half_body > .80*(hi-lo+1):
            continue
        inside = (np.abs(u-mid_u) <= half_body) & (np.abs(v-mid_v) <= .5*across+1.5)
        selected = points[inside].astype(int)
        mask = np.zeros_like(support)
        mask[selected[:, 1], selected[:, 0]] = True
        if not _two_dimensional(mask):
            continue
        box = base._tight(mask)
        quality = float(outside.sum()) / (1.+half_stroke)
        candidates.append((quality, box, float(theta)))
    if not candidates:
        raise ValueError("Legend key is line-only or lacks a complete two-dimensional observed body")
    _, box, theta = max(candidates, key=lambda item: item[0])
    return box, theta


def _legend_support(image, spec):
    box = base._box(spec.get('swatch_box', spec.get('swatch')), image.shape)
    sx0, sy0, sx1, sy1 = box
    model = base._swatch_model(image, box, spec.get('color_rgb'))
    margin = max(2, min(6, int(round(.12*(sy1-sy0)))))
    x0, y0, x1, y1 = base._box([sx0-margin, sy0-margin, sx1+margin, sy1+margin], image.shape)
    soft, _ = base._membership(image[y0:y1, x0:x1], model)
    # Weak unrelated connector ink cannot set the extent of a strongly
    # coloured glyph.  Weak edges are retained by the final source extraction.
    support = soft >= .32
    seed = np.zeros_like(support)
    seed[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = True
    return model, (x0, y0, x1, y1), support, seed


def _repeated_collinear_strokes(support, seed):
    """Recognize a dashed stroke from several equally thin source components.

    A single short rectangle is ambiguous. Three or more separate components
    must establish the same line, thickness and dash extent, with no distinct
    two-dimensional bulge. All measurements rotate with the observed line.
    """
    n, labels, _, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    components = []
    for k in range(1, n):
        selected = labels == k
        if int((selected & seed).sum()) < 3:
            continue
        yy, xx = np.nonzero(selected)
        components.append(np.column_stack((xx, yy)).astype(np.float32))
    if len(components) < 3:
        return False
    points = np.concatenate(components)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    if eigenvalues[0]/max(1., eigenvalues[-1]) > .08:
        return False
    axis = eigenvectors[:, -1]
    normal = np.array([-axis[1], axis[0]])
    along, across, centers, intervals = [], [], [], []
    for component in components:
        u, v = component@axis, component@normal
        along.append(float(np.ptp(u)+1))
        across.append(float(np.ptp(v)+1))
        centers.append(float(np.median(v)))
        intervals.append((float(u.min()), float(u.max())))
    thickness = float(np.median(across))
    extent = float(np.median(along))
    if max(across) > 1.6*min(across) or max(along) > 1.6*min(along):
        return False
    if np.ptp(centers) > max(1.2, .40*thickness) or extent < 1.35*thickness:
        return False
    intervals.sort()
    if any(right >= next_left for (_, right), (next_left, _) in zip(intervals, intervals[1:])):
        return False
    total_extent = intervals[-1][1]-intervals[0][0]+1
    return total_extent >= max(3.5*extent, 5.*thickness)


def _recover_observed_template(image, spec, context=None):
    model, (x0, y0, x1, y1), support, seed = context or _legend_support(image, spec)
    n, labels, _, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    eligible = [(int(((labels == k) & seed).sum()), k) for k in range(1, n)]
    if not eligible or max(eligible)[0] < 6:
        raise ValueError("No complete observed graphical component intersects the swatch")
    support = labels == max(eligible)[1]
    body, angle = _observed_body(support)
    a, b, c, d = body
    source_box = [x0+a, y0+b, x0+c, y0+d]
    result = base.build_series_template(image, spec, _body_source_box=source_box,
                                        _source_model=model)
    result['connector_direction'] = 'observed_orientation' if angle is not None else None
    result['provenance'].update(extraction='complete_observed_component',
                                connector_angle_radians=angle)
    if angle is not None:
        yy, xx = np.mgrid[:result['soft'].shape[0], :result['soft'].shape[1]]
        cx, cy = result['center']
        distance = -(xx-cx)*math.sin(angle)+(yy-cy)*math.cos(angle)
        result['central_connector'] = np.abs(distance) <= 1.5
    return result


def _template_regions(template):
    """Separate enclosed source paper from the outer antialiasing envelope."""
    soft = template['soft']
    support = soft >= .18
    face = np.zeros(support.shape, np.uint8)
    yy, xx = np.nonzero(support)
    if len(xx) >= 3:
        cv2.fillConvexPoly(face, cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)), 1)
    face = face.astype(bool)
    envelope = cv2.dilate(face.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    inner_face = cv2.erode(face.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary = envelope & ~inner_face
    contrast = np.linalg.norm(template['raw_bgr'].astype(np.float32)-
                              template['model']['paper_bgr'], axis=-1)
    observed_paper = (contrast <= 30.) & (soft < .12)
    # Four-connected paper cannot leak diagonally across a visible thin rim.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~support).astype(np.uint8), connectivity=4)
    enclosed = np.zeros_like(support)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if x == 0 or y == 0 or x+w == support.shape[1] or y+h == support.shape[0] or area < 2:
            continue
        island = labels == k
        if float(observed_paper[island].mean()) >= .70:
            enclosed |= island & observed_paper
    # Core holes omit the uncertain inner antialiased rim.  Small enclosed
    # islands may consequently provide no negative fill evidence.
    hole_core = cv2.erode(enclosed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rim = support & (cv2.dilate(enclosed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    central = np.asarray(template['central_connector'], bool)
    uncertain = central & inner_face & ~rim if enclosed.any() else np.zeros_like(support)
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    weight = soft*np.where(distance >= 1.5, 1., .65)
    weight[uncertain] *= .25
    template.update(face=face, envelope=envelope, hole_core=hole_core,
                    enclosed_paper=enclosed, boundary_uncertain=boundary,
                    observed_rim=rim, uncertain=uncertain, weight=weight.astype(np.float32),
                    hollow_fraction=float(enclosed.sum()/max(1, face.sum())),
                    evidence_version=VERSION)
    return template


def build_series_template(image_bgr, spec, *, _body_source_box=None, _source_model=None):
    """Extract original ink, recovering rejected X/+ bodies without shape labels."""
    image = np.asarray(image_bgr, np.uint8)
    if _body_source_box is not None:
        return _template_regions(base.build_series_template(image, spec,
            _body_source_box=_body_source_box, _source_model=_source_model))
    context = _legend_support(image, spec)
    if _repeated_collinear_strokes(context[2], context[3]):
        raise ValueError("Legend key is line-only; repeated collinear equal-width dash components")
    try:
        result = base.build_series_template(image, spec)
        support = result['soft'] >= .32
        a, b, c, d = base._tight(support)
        if not _two_dimensional(support) or min(c-a, d-b) < .45*max(c-a, d-b):
            result = _recover_observed_template(image, spec, context)
    except ValueError:
        result = _recover_observed_template(image, spec, context)
    return _template_regions(result)


def _equivalent_models(first, second):
    a = first['paper_bgr']-first['bgr']
    b = second['paper_bgr']-second['bgr']
    cosine = float(a@b/max(1., np.linalg.norm(a)*np.linalg.norm(b)))
    return cosine >= math.cos(math.radians(9)) and np.linalg.norm(first['bgr']-second['bgr']) < 85.


def _mine_plot_template(image, plot_box, spec, exclusion_specs):
    own_box = base._box(spec.get('swatch_box', spec.get('swatch')), image.shape)
    own = base._swatch_model(image, own_box, spec.get('color_rgb'))
    for rival in exclusion_specs:
        if rival is spec:
            continue
        try:
            box = base._box(rival.get('swatch_box', rival.get('swatch')), image.shape)
            model = base._swatch_model(image, box, rival.get('color_rgb'))
        except ValueError:
            continue
        if _equivalent_models(own, model):
            raise ValueError("Shared-palette legend has insufficient identity evidence; plot mining is disabled")
    # The frozen miner uses ids when excluding palette entries. Give the miner
    # local positional tokens so caller ids/labels cannot change its decisions.
    positional = [dict(item, id=f'palette_{i}') for i, item in enumerate(exclusion_specs)]
    own_index = next(i for i, item in enumerate(exclusion_specs) if item is spec)
    result = base._mine_plot_template(image, plot_box, positional[own_index], positional)
    result['id'] = str(spec.get('id', spec.get('label', 'series')))
    result['label'] = str(spec.get('label', spec.get('id', 'series')))
    result['provenance']['identity_basis'] = 'unique_observed_palette_with_recurrent_body'
    return _template_regions(result)


def colour_evidence(crop_bgr, templates):
    """Preserve v1 colour semantics, including same-palette occluder exclusion."""
    result = base.colour_evidence(crop_bgr, templates)
    from color_blend_uncertainty_v46 import blend_uncertainty, VERSION
    result['blend_uncertainty'] = blend_uncertainty(crop_bgr, templates,
        result['membership'], result['colour_confidence'], result['equivalent_colours'], result['paper_bgr'])
    result['blend_uncertainty_version'] = VERSION
    return result


def _rescale_template(template, scale_x, scale_y):
    shape = template['soft'].shape
    size = (max(3, round(shape[1]*scale_x)) | 1, max(3, round(shape[0]*scale_y)) | 1)
    matrix = np.array([[scale_x, 0., (size[0]-1)/2.-scale_x*(shape[1]-1)/2.],
                       [0., scale_y, (size[1]-1)/2.-scale_y*(shape[0]-1)/2.]], np.float32)
    for key in ('soft', 'weight', 'raw_soft'):
        template[key] = cv2.warpAffine(template[key], matrix, size, flags=cv2.INTER_LINEAR)
    for key in ('core', 'envelope', 'nuisance', 'central_connector', 'uncertain',
                'face', 'hole_core', 'enclosed_paper', 'boundary_uncertain', 'observed_rim'):
        template[key] = cv2.warpAffine(template[key].astype(np.uint8), matrix, size,
                                       flags=cv2.INTER_NEAREST) > 0
    template['raw_bgr'] = cv2.warpAffine(template['raw_bgr'], matrix, size,
        flags=cv2.INTER_LINEAR, borderValue=tuple(float(v) for v in template['model']['paper_bgr']))
    # Resampling an uncertain boundary must never promote it to a hole core.
    template['hole_core'] &= template['face'] & ~template['boundary_uncertain']
    template['hole_core'] &= template['soft'] < .12
    template['source_diameter'] = template['diameter']
    template['diameter'] *= .5*(scale_x+scale_y)
    template['center'] = [(size[0]-1)/2., (size[1]-1)/2.]


def prepare_evidence(image_bgr, plot_box, series_specs, max_side=1400, *, template_overrides=None,
                     ignore_furniture=True, guide_policy='legacy_band',scale_policy='fixed_1x',
                     palette_recovery='off', frozen_colour_evidence=None):
    """Return the v1 evidence contract plus explicit observed body region masks.

    Resize coordinates retain OpenCV's half-pixel source-centre offset. Failed
    keys are reported in template_errors; an entirely unusable legend raises.
    v46 may supply already prepared legend templates by series ID. Explicit
    None means a line-only key: never re-extract or mine it into a marker.
    ``guide_policy='model'`` opts into frozen periodic-dot explanations (the
    production runtime selects it for supported filled-triangle legends).
    ``scale_policy='shared_symbol'`` calibrates once after colour/ignore fields
    are built and before proposal extraction. The low-level default stays fixed
    so saved working templates can be reloaded without applying scale twice.
    ``palette_recovery`` is opt-in for fresh runtime detection. Correction
    restores ``frozen_colour_evidence`` rather than relearning with scaled
    templates; legacy callers and saved sessions keep the default off policy.
    """
    image = np.asarray(image_bgr, np.uint8)
    if scale_policy not in ('fixed_1x','shared_symbol'):
        raise ValueError('Unknown colour symbol scale policy')
    if guide_policy not in {'model','legacy_band','none'}:
        raise ValueError('guide_policy must be model, legacy_band or none')
    specs = list(series_specs)
    box = base._box(plot_box, image.shape)
    x0, y0, x1, y1 = box
    native = image[y0:y1, x0:x1].copy()
    templates, errors = [], []
    for spec in specs:
        sid = str(spec.get('id', 'series'))
        if template_overrides is not None and sid in template_overrides:
            template = deepcopy(template_overrides[sid])
            if template is None:
                errors.append(dict(id=sid, reason='Prepared line-only legend; marker extraction disabled'))
                continue
            if str(template['id']) != sid:
                raise ValueError('Prepared legend template identity differs from series ID')
            templates.append(template)
            continue
        try:
            template = build_series_template(image, spec)
        except ValueError as exc:
            reason = str(exc)
            if 'line-only' not in reason.lower():
                errors.append(dict(id=str(spec.get('id', 'series')), reason=reason))
                continue
            try:
                template = _mine_plot_template(image, box, spec, specs)
            except ValueError as mining_error:
                errors.append(dict(id=str(spec.get('id', 'series')), reason=reason+'; '+str(mining_error)))
                continue
        templates.append(template)
    if not templates:
        raise ValueError(f"No usable legend templates: {errors}")
    factor = min(1., float(max_side)/max(native.shape[:2])) if max_side else 1.
    w, h = max(1, round(native.shape[1]*factor)), max(1, round(native.shape[0]*factor))
    scale_x, scale_y = w/native.shape[1], h/native.shape[0]
    crop = cv2.resize(native, (w, h), interpolation=cv2.INTER_AREA) if factor < 1 else native
    if factor < 1:
        for template in templates:
            _rescale_template(template, scale_x, scale_y)
    if frozen_colour_evidence is not None:
        if factor != 1. or palette_recovery != 'off':
            raise ValueError('Frozen colour fields cannot be resized or recalibrated')
        from color_recovery_evidence_v46 import restore
        result = restore(frozen_colour_evidence, [str(t['id']) for t in templates], box)
    else:
        result = colour_evidence(crop, templates)
    valid = np.ones((h, w), bool)
    excluded = []
    for spec in specs:
        value = spec.get('legend_box') or spec.get('swatch_box', spec.get('swatch'))
        if value is None:
            continue
        a, b, c, d = map(float, value)
        left, top = max(0, math.floor((a-x0)*scale_x)), max(0, math.floor((b-y0)*scale_y))
        right, bottom = min(w, math.ceil((c-x0)*scale_x)), min(h, math.ceil((d-y0)*scale_y))
        if right > left and bottom > top:
            valid[top:bottom, left:right] = False
            excluded.append([left, top, right, bottom])
    if frozen_colour_evidence is None and palette_recovery != 'off':
        from color_recovery_evidence_v46 import apply_recovery
        apply_recovery(crop, templates, result, valid, policy=palette_recovery)
    for key in ('membership', 'colour_confidence', 'other'):
        result[key][:, ~valid] = 0
    result['blend_uncertainty'][:, ~valid] = 0
    result['all_ink'][~valid] = 0
    result['unknown'][~valid] = 0
    # Ignore observed furniture, NOT marker-centre locations. Keep source and
    # raw colour evidence intact for audit and display. Detect at native scale
    # then area-resample the mask with the same geometry as the image.
    from marker_ignore_v46 import build_ignore_mask
    native_valid = np.ones(native.shape[:2], bool)
    for spec in specs:
        region = spec.get('legend_box') or spec.get('swatch_box', spec.get('swatch'))
        if region is not None:
            a, b, c, d = map(int, region)
            native_valid[max(0,b-y0):max(0,min(native.shape[0],d-y0)),
                         max(0,a-x0):max(0,min(native.shape[1],c-x0))] = False
    diameter = min(t['diameter']/(.5*(scale_x+scale_y)) for t in templates)
    ignored, ignore_report = build_ignore_mask(native, valid=native_valid,
        marker_diameter=diameter, include_guides=guide_policy=='legacy_band') if ignore_furniture else (
        np.zeros(native.shape[:2], bool), dict(enabled=False, ignored_pixels=0))
    if ignore_furniture and guide_policy=='model':
        from marker_guide_model_v46 import fit_guides, palette_fields
        guide, error, guide_report = fit_guides(native,ignore_report['structures'],
            valid=native_valid,marker_diameter=diameter)
        if factor<1:
            guide=cv2.resize(guide,(w,h),interpolation=cv2.INTER_AREA)
            error=cv2.resize(error,(w,h),interpolation=cv2.INTER_AREA)
        if guide_report['fitted_count']:
            middle,low,high=palette_fields(guide,error,templates)
            for value in (middle,low,high):value[:,~valid]=0
            result.update(guide_membership=middle,guide_lower=low,guide_upper=high)
        result.update(guide_darkness=guide,guide_error=error,guide_report=guide_report)
    ignored = ignored.astype(np.float32)
    if factor < 1:
        ignored = cv2.resize(ignored, (w, h), interpolation=cv2.INTER_AREA)
    result['ignore_mask'] = ignored
    result['ignore_report'] = ignore_report
    result['guide_policy'] = guide_policy if ignore_furniture else 'disabled'
    result.update(crop_bgr=crop, templates=templates, template_errors=errors,
        valid=valid, excluded_boxes=excluded, plot_box_source=box, scale=float(factor),
        scale_x=float(scale_x), scale_y=float(scale_y),
        source_center_offset=[x0+.5/scale_x-.5, y0+.5/scale_y-.5],
        native_shape=list(native.shape[:2]), version=VERSION)
    if scale_policy=='shared_symbol':
        from color_shared_scale_v46 import apply_shared_scale
        apply_shared_scale(result)
    return result

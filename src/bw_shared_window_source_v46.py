"""One authoritative legend source for native BW window and raster evidence.

This adapter never discovers another crop/connector or estimates another size.
Completed marker alpha is kept complete. A geometry-only filled envelope is
never substituted for the measured appearance of a patterned/gray marker.
"""
from collections.abc import Mapping
import hashlib
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

VERSION = 'shared-native-window-source-v2-photometry'


def _fingerprint(value):
    a = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def audit_source_models(templates, models, competition):
    """Check model transport, not correctness of inferred semantic labels.

    Required alpha and mask must be those of the actual native window. Raster
    brightness may use different documented units; it must be the exact shared
    appearance, never an independent crop/connector reconstruction.
    """
    rows = []
    ready = bool(competition.ready)
    for t in templates:
        source = models[t.key]
        rec, prepared, meta = source['record'], source['prepared'], source['meta']
        h,w = t.mask.shape
        checks = dict(required_alpha_equal=np.array_equal(prepared['required_alpha'],np.asarray(t.raw_soft,np.float32)),
            required_mask_equal=np.array_equal(prepared['required_mask'],np.asarray(t.mask,bool)),
            canvas_equal=tuple(prepared['marker'].shape)==tuple(t.mask.shape),
            local_center_equal=np.array_equal(rec['crop_center'],[(w-1)/2.,(h-1)/2.]),
            diameter_equal=float(rec['diameter'])==float(t.diameter),
            no_independent_connector_extraction=meta.get('connector_reextracted') is False)
        if ready:
            checks['raster_appearance_equal']=np.array_equal(competition.prepared[t.key]['marker'],prepared['marker'])
            checks['raster_weight_equal']=np.array_equal(competition.prepared[t.key]['weight'],prepared['weight'])
            checks['raster_record_equal']=competition.templates[t.key][0]==rec
            checks['raster_uses_native_registration']=competition.report.get('source_model_policy')=='shared_native_window'
        rows.append(dict(swatch_id=t.key, class_name=t.name, marker_kind=t.marker_kind,
            absolute_legend_center=list(map(float,t.marker_center)),swatch_box=list(t.swatch_box),
            canvas_shape=[h,w],native_diameter=float(t.diameter),checks=checks,
            consistent=all(checks.values()),window_alpha_sha256=_fingerprint(t.raw_soft),
            raster_required_alpha_sha256=_fingerprint(prepared['required_alpha']),
            raster_appearance_sha256=_fingerprint(prepared['marker']),
            measured_gray_sha256=_fingerprint(prepared['source_gray']),
            appearance_source=meta['appearance_source'],source_model_completed=meta['source_model_completed'],
            gray_units='0=black,255=white',raster_units='absolute_paper_referenced_darkness',
            window_units='native_required_ink_alpha'))
    mismatch=sum(not r['consistent'] for r in rows)
    return dict(version='legend-source-contract-audit-v1',enabled=True,
        status=('mismatch' if mismatch else 'verified_shared' if ready else 'source_built_raster_unavailable'),
        comparison_complete=ready,source_reextracted=False,swatch_count=len(rows),
        mismatch_count=mismatch,sources=rows,
        scope='pixel_geometry_transport_only_not_semantic_shape_accuracy',
        raster_status=competition.report.get('status'))


def native_window_variant(record, prepared, scale):
    """Use the full window's resize dimensions and integer placement exactly.

    Native verification resizes the complete canvas to round(width*scale),
    then floor-centers that array in an odd window. An even resized array is
    consequently centered half a pixel above/left of its integer anchor.
    Affine resampling around a nominal center is NOT the same registration.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Native window variant scale must be finite and positive')
    h, w = prepared['marker'].shape
    size = (max(3, round(w*scale)), max(3, round(h*scale)))
    radius = int(np.ceil(max(size)/2.))+2
    side = 2*radius+1
    left, top = (side-size[0])//2, (side-size[1])//2
    def place(field):
        # Match _uncertain_shape's direct INTER_LINEAR resize, not weighted
        # conditional resampling, which would move uncertain source edges.
        resized = cv2.resize(np.asarray(field, np.float32), size,
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((side, side), np.float32)
        canvas[top:top+size[1], left:left+size[0]] = resized
        return canvas
    return place(prepared['marker']), place(prepared['weight']), place(prepared['cover']), radius


def _array(value, shape, name, *, unit=True):
    a = np.asarray(value, np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite and aligned with the native marker')
    if unit and (np.any(a < 0) or np.any(a > 1)):
        raise ValueError(f'{name} must be in [0, 1]')
    return a.copy()


def _transport(value, source, target, *, fill=0.):
    """Translate original observations into the already fitted marker frame."""
    oh, ow = np.asarray(value).shape
    h, w = target.mask.shape
    delta = np.asarray(target.marker_center, float)-np.asarray(source.marker_center, float)
    transform = np.float32([[1, 0, (w-ow)/2-delta[0]],
                           [0, 1, (h-oh)/2-delta[1]]])
    return cv2.warpAffine(np.asarray(value, np.float32), transform, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=float(fill))


def _completed_photometry(template, alpha, original, report, geometry_detail):
    """Convert the *documented* alpha units to absolute source darkness.

    The native grid's core_gray is a conservative normalization reference,
    not necessarily the physical ink colour. A composed marker is rendered
    coverage (including any fitted ink_level); the ring fallback instead fits
    normalized observed ink and must retain that distinct unit conversion.
    """
    policy = report.get('template_policy')
    contrast = max(24., float(template.ink.paper_gray)-float(template.ink.core_gray))
    if (policy == 'pure_ring_with_measured_line_nuisance' or
            geometry_detail.get('mode') == 'line_band_plus_inferred_open_ring'):
        return contrast/255., dict(photometry_source='normalized_ring_fit_inverse_grid_units',
            alpha_units='normalized_observed_ink_with_fitted_gain',
            ink_level_already_in_alpha=True)
    composition = report.get('composition') or {}
    if policy == 'pure_model_marker':
        if composition.get('status') != 'supported_simple_shape_model':
            raise ValueError(f'{template.key}: completed composition lacks supported photometry')
        fg = np.asarray(composition.get('foreground_rgb'), float)
        bg = np.asarray(composition.get('background_rgb'), float)
        if (fg.shape != (3,) or bg.shape != (3,) or not np.isfinite(fg).all() or
                not np.isfinite(bg).all() or np.any(fg < -1.e-5) or
                np.any(bg > 255.00001) or np.any(bg-fg <= 0)):
            raise ValueError(f'{template.key}: invalid completed composition foreground/background')
        # render_model() has already applied params.ink_level to marker_alpha.
        # Multiplying that parameter here again would darken gray markers twice.
        factor = float(np.dot(bg-fg, [.299, .587, .114])/255.)
        return factor, dict(photometry_source='supported_composition_foreground_background',
            alpha_units='rendered_coverage_with_fitted_ink_level',
            ink_level_already_in_alpha=True, foreground_rgb=fg.tolist(), background_rgb=bg.tolist())

    # Completed geometric silhouettes without a composition report retain the
    # measured intensity on their own stroke. A black connector must not set
    # a gray marker's amplitude: exclude the already known connector support
    # and use a robust ratio, not a swatch-wide maximum or a grid core value.
    gray = getattr(template, 'source_gray', None)
    if gray is None:
        raise ValueError(f'{template.key}: completed marker lacks physical source photometry')
    gray = np.asarray(gray, np.float32)
    if (gray.shape != alpha.shape or not np.isfinite(gray).all() or
            np.any(gray < 0) or np.any(gray > 255)):
        raise ValueError(f'{template.key}: invalid completed marker source_gray')
    nuisance = _array(template.line_nuisance, alpha.shape, f'{template.key}.line_nuisance')
    if original is not None:
        nuisance = np.maximum(nuisance, _transport(original.line_nuisance, original, template))
    weight = _array(template.valid_weight, alpha.shape, f'{template.key}.valid_weight')
    support = ((alpha >= .6*float(alpha.max())) & np.asarray(template.mask, bool) &
               (nuisance < .01) & (weight >= .2))
    if int(support.sum()) < 3:
        raise ValueError(f'{template.key}: insufficient independent marker photometry support')
    darkness = np.clip((float(template.ink.paper_gray)-gray)/255., 0, 1)
    factor = float(np.median(darkness[support]/alpha[support]))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f'{template.key}: no positive completed marker source contrast')
    return factor, dict(photometry_source='robust_measured_marker_stroke_excluding_connector',
        alpha_units='completed_model_fitted_to_measured_gray',
        photometry_support_pixels=int(support.sum()), connector_support_excluded=True)


def build_source_models(templates, geometry=None, legend_reports=None):
    """Return ``key -> {record, fields, prepared, meta}`` for Competition.

    Every array is in the template's native canvas; ``crop_center`` is its
    local array center, NOT the absolute legend marker_center. Native diameter
    and externally approved scales retain their common reference unchanged.
    ``marker`` is absolute grayscale darkness, while ``required_alpha`` retains
    the normalized required-ink model used by the original full window.
    """
    out = {}
    descriptors = getattr(geometry, 'descriptors', {}) if geometry is not None else {}
    originals = getattr(geometry, 'originals', {}) if geometry is not None else {}
    filled_keys = getattr(geometry, 'filled_keys', set()) if geometry is not None else set()
    details = getattr(geometry, 'details', {}) if geometry is not None else {}
    report_by_key = {r['swatch_id']: r for r in (legend_reports or [])}
    for t in templates:
        key = t.key
        if not key or key in out:
            raise ValueError('Shared source templates require distinct nonempty identities')
        if not getattr(t.ink, 'achromatic', False):
            raise ValueError(f'{key}: shared native source supports achromatic templates only')
        mask = np.asarray(t.mask, bool)
        if mask.ndim != 2 or min(mask.shape) < 3 or not mask.any():
            raise ValueError(f'{key}: empty or invalid native marker mask')
        shape = mask.shape; h, w = shape
        diameter = float(t.diameter)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(f'{key}: invalid native marker diameter')
        alpha = _array(t.raw_soft, shape, f'{key}.raw_soft')
        paper = float(t.ink.paper_gray); core = float(t.ink.core_gray)
        if not np.isfinite(paper+core) or not (0 <= core < paper <= 255):
            raise ValueError(f'{key}: invalid native source photometry')
        contrast = max(24., paper-core)
        style = descriptors.get(key, {}).get('style')
        observed_fill = key in filled_keys and style in ('patterned', 'gray', 'partial')
        original = originals.get(key)
        if observed_fill and original is None:
            raise ValueError(f'{key}: geometry-only silhouette has no original appearance')
        appearance_source = original if observed_fill else t
        # Native source_gray is an observation, not a rendered silhouette.
        # Complete model alpha is used only when that model already belongs
        # to the full-window template and is not a filled search envelope.
        complete = bool(getattr(t, 'model_completed', False)) and not observed_fill
        photometry = dict(photometry_source='measured_source_gray_or_normalized_observation',
                          alpha_units='observed_ink')
        absolute_factor = contrast/255.
        if complete:
            absolute_factor, photometry = _completed_photometry(t, alpha, original,
                report_by_key.get(key, {}), details.get(key, {}))
            marker = np.clip(alpha*absolute_factor, 0, 1)
            weight = _array(t.valid_weight, shape, f'{key}.valid_weight')
            nuisance = _array(t.line_nuisance, shape, f'{key}.line_nuisance')
            appearance_kind = 'authoritative_completed_marker_alpha'
        else:
            gray = getattr(appearance_source, 'source_gray', None)
            if gray is not None:
                gray = np.asarray(gray, np.float32)
                if gray.shape != appearance_source.mask.shape or not np.isfinite(gray).all() or np.any((gray < 0)|(gray > 255)):
                    raise ValueError(f'{key}: invalid observed source_gray')
                source_paper = float(appearance_source.ink.paper_gray)
                observed = np.clip((source_paper-gray)/255., 0, 1)
            else:
                source_contrast = max(24., float(appearance_source.ink.paper_gray)-float(appearance_source.ink.core_gray))
                observed = _array(appearance_source.raw_soft, appearance_source.mask.shape,
                                  f'{key}.original.raw_soft')*source_contrast/255.
            if observed_fill:
                marker = _transport(observed, original, t)
                weight = _transport(original.valid_weight, original, t, fill=1.)
                nuisance = _transport(original.line_nuisance, original, t)
                appearance_kind = 'original_observed_fill_transported_to_native_geometry'
            else:
                marker = _array(observed, shape, f'{key}.observed')
                weight = _array(t.valid_weight, shape, f'{key}.valid_weight')
                nuisance = _array(t.line_nuisance, shape, f'{key}.line_nuisance')
                appearance_kind = 'native_observed_gray'
            # Existing ambiguity remains ambiguity; there is no row detector,
            # connector subtraction, or replacement by invented white pixels.
            weight = np.minimum(weight, np.where(nuisance > .01, .06, 1.))
        marker = np.asarray(marker, np.float32)
        if not np.isfinite(marker).all() or float(np.sum(marker**2)) <= 1.e-8:
            raise ValueError(f'{key}: shared appearance has no finite marker energy')
        weight = np.asarray(weight, np.float32)
        source_open = bool(t.marker_kind == 'open' or str(t.name).startswith('open_'))
        if complete and source_open:
            # Opaque coverage is drawing order, never additional positive ink.
            # The enclosed region belongs to the supported native open model.
            holes = binary_fill_holes(mask)&~mask
            cover = np.maximum(alpha, holes.astype(np.float32))
        else:
            peak = max(float(marker.max()), .05)
            ink_core = marker > .28*peak
            holes = binary_fill_holes(ink_core)&~ink_core
            holes &= (marker < .45*peak)&(nuisance < .01)
            if not source_open:
                holes[:] = False
            cover = np.maximum(np.clip(marker/peak, 0, 1), holes.astype(np.float32))
        yy, xx = np.nonzero(mask)
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
        center = [(w-1)/2., (h-1)/2.]
        rec = dict(version=VERSION, status='shared_native_window_template',
            bbox=bbox, crop=[0, 0, w, h], center=center, crop_center=center,
            diameter=diameter, paper=paper, class_name=t.name,
            connector_detected=bool(np.any(nuisance > .01)),
            source_raster_exact=not complete, model_completed=complete)
        measured_gray = getattr(appearance_source, 'source_gray', None)
        if measured_gray is not None:
            measured_gray = (_transport(measured_gray, original, t, fill=paper) if observed_fill
                             else np.asarray(measured_gray, np.float32).copy())
        rendered_gray = (paper-255*marker).astype(np.float32)
        prepared = dict(marker=marker.copy(), weight=weight.copy(), cover=cover.astype(np.float32),
            required_alpha=alpha.copy(), required_mask=mask.copy(),
            required_weight=_array(t.required_weight if t.required_weight is not None else mask,
                                   shape, f'{key}.required_weight'),
            source_uncertainty=nuisance.copy(), holes=holes.copy(),
            source_gray=rendered_gray.copy() if measured_gray is None else measured_gray,
            rendered_marker_gray=rendered_gray, source_observed=marker.copy())
        meta = dict(version=VERSION, source_open=source_open,
            source_hole_pixels=int(holes.sum()), uncertain_body_pixels=int(np.count_nonzero(weight < .2)),
            appearance_source=appearance_kind, authoritative_template=key,
            native_diameter=diameter, geometry_scale_reference='native_window_template',
            connector_reextracted=False, connector_subtracted_again=False,
            source_pixels_modified=False, source_model_completed=complete,
            source_gray_is_measured=measured_gray is not None,
            filled_silhouette_used_as_appearance=False,
            contrast_to_absolute_ink=absolute_factor, coordinate_frame='native_template_canvas',
            **photometry)
        fields = dict(target=marker.copy(), target_weight=weight.copy(),
                      source_gray=prepared['source_gray'].copy())
        out[key] = dict(record=rec, fields=fields, prepared=prepared, meta=meta)
    if not out:
        raise ValueError('At least one shared native window source is required')
    return out


def validate_source_models(source_models, template_keys, approved_scales):
    """Validate the public injection boundary; fail instead of partial rivals."""
    keys = set(template_keys)
    if not isinstance(source_models, Mapping) or set(source_models) != keys:
        raise ValueError('Shared source models must contain exactly the native template identities')
    if not isinstance(approved_scales, Mapping) or set(approved_scales) != keys:
        raise ValueError('Approved shared scales must contain exactly the native template identities')
    for key in keys:
        model = source_models[key]
        if not isinstance(model, Mapping) or not all(k in model for k in ('record', 'fields', 'prepared', 'meta')):
            raise ValueError(f'{key}: malformed shared source model')
        rec, prepared = model['record'], model['prepared']
        if not np.isfinite(float(rec['diameter'])) or float(rec['diameter']) <= 0:
            raise ValueError(f'{key}: invalid shared source diameter')
        marker = np.asarray(prepared['marker'])
        if marker.ndim != 2 or min(marker.shape) < 3:
            raise ValueError(f'{key}: invalid shared source canvas')
        for field in ('marker', 'weight', 'cover'):
            _array(prepared[field], marker.shape, f'{key}.{field}')
        if float(np.sum(marker**2)) <= 1.e-8:
            raise ValueError(f'{key}: shared source has no marker energy')
        center = np.asarray(rec['crop_center'], float)
        if center.shape != (2,) or not np.isfinite(center).all() or np.any(center < 0) or center[0] >= marker.shape[1] or center[1] >= marker.shape[0]:
            raise ValueError(f'{key}: shared source center is outside its canvas')
        scale = approved_scales[key]
        if scale is not None and (not np.isfinite(float(scale)) or float(scale) <= 0):
            raise ValueError(f'{key}: approved native scale must be finite and positive or unresolved None')

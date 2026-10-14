"""Resolve an ambiguous hollow legend key against native line/body rivals.

Binary hole area is unreliable when a thick rim and connector leave only a
few paper pixels. An optional native-raster fit supplies identity evidence,
not replacement observations. Proposal geometry is completed separately.
"""
from copy import deepcopy
import math
import time

import cv2
import numpy as np

from legend_composition_v46 import Config, render_model
from legend_open_composition_v46 import LineEvidenceError

VERSION = 'line-crossed-hollow-native-competition-v1'
NAMES = dict(open_circle='open_circle', open_ellipse='open_circle',
             open_square='open_square', open_rectangle='open_square',
             open_triangle_up='open_triangle', open_triangle_down='open_inv_triangle')
FAMILIES = dict(open_circle='circle', open_ellipse='circle', open_square='square',
                open_rectangle='square', open_triangle_up='triangle',
                open_triangle_down='inv_triangle')


def infer_line_crossed_hollow(image, template, report):
    """Revise automatic identity only after a supported original-pixel fit.

The existing fitter compares hollow, filled and line-only explanations using
its normal quality/margin thresholds. Source paper must independently remain
inside the winning outline, away from the independently fitted connector.
No arrays in the observed template or caller's image are replaced.
"""
    from bw_fill_identity_v46 import observed_hole, describe_fill
    shape = report.get('shape_evidence', {})
    if (report.get('classification') == 'supplied' or not template.ink.achromatic
            or template.model_completed
            or template.name.startswith('open_')
            or report.get('line_analysis', {}).get('status') != 'separated'
            or shape.get('strong_hollow_evidence')
            or shape.get('patterned_internal_evidence')
            or int(observed_hole(template).sum()) < 2):
        return template, report

    from legend_layered_composition_v46 import fit_layered
    a, b, c, d = map(int, template.swatch_box)
    crop = image[b:d, a:c]
    cfg = Config(enable_open_dashed=True)
    started = time.perf_counter()
    try:
        fit, fields = fit_layered(crop[..., ::-1], cfg=cfg, small_hollow=True)
    except LineEvidenceError as error:
        report['line_crossed_hollow_attempt'] = dict(version=VERSION,
            status='unresolved_connector', error_code=error.code,
            fit_seconds=time.perf_counter()-started)
        return template, report
    attempt = dict(version=VERSION, status=fit['status'],
                   best_model_name=fit['best_model_name'],
                   winner_other_family_margin=fit['winner_other_family_margin'],
                   fit_seconds=time.perf_counter()-started,
                   source_pixels_modified=False)
    report['line_crossed_hollow_attempt'] = attempt
    name = fit['best_model_name']
    if fit['status'] != 'supported_simple_shape_model' or name not in NAMES:
        return template, report

    # A fitted cavity is only a sampling region. Its original source pixels,
    # including unexplained interior dark strokes, decide whether it is paper.
    body = render_model(name.removeprefix('open_'), fit['best_model']['params'],
                        fit['line_params'], crop.shape[:2], cfg)['marker_alpha']
    paper = ((body > .90) & (fields['marker_alpha'] < .20)
             & (fields['line_alpha'] < .20))
    ga, gb, gc, gd = map(int, report['glyph_box'])
    native = paper[gb-b:gd-b, ga-a:gc-a]
    region = np.zeros(template.mask.shape, bool)
    top = (region.shape[0] - native.shape[0]) // 2
    left = (region.shape[1] - native.shape[1]) // 2
    region[top:top+native.shape[0], left:left+native.shape[1]] = native
    from bw_fill_identity_v46 import tone_ink
    values = tone_ink(template)[region]
    attempt.update(independent_paper_pixels=int(values.size),
                   independent_paper_mean_ink=float(values.mean()) if values.size else None)
    if values.size < 2 or float(values.mean()) > .25 or float(np.mean(values > .5)) > .15:
        attempt['status'] = 'unexplained_interior_ink'
        return template, report

    previous = template.name
    template.name = NAMES[name]
    template.shape_hint = template.name
    template.marker_kind = 'open'
    template.interior_fill = float(values.mean())
    template.line_crossed_hollow_region = region
    template.line_crossed_hollow_model = fit
    # The old shallow binary fill sample may have promoted the connector to
    # required body ink. A resolved hollow cannot retain that solid-only boost.
    affected = template.line_nuisance & template.mask
    template.required_weight[affected] = np.minimum(template.required_weight[affected], .20)
    report['raw_shape_evidence'] = deepcopy(shape)
    shape = dict(shape, method='native_line_and_hollow_model_competition',
                 status='classified', best_shape=FAMILIES[name],
                 shape_hint=template.name, strong_hollow_evidence=True,
                 hollow_evidence_source=VERSION,
                 identity_basis='native_composition_residual_and_rival_margin',
                 shape_confidence_policy='raw_binary_scores_are_not_composition_confidence')
    report.update(class_name=template.name, shape_hint=template.name,
                  shape_evidence=shape, line_crossed_hollow=fit,
                  original_class=previous, restored_filled_interior_pixels=0,
                  strong_required_pixels=int((template.required_weight >= .85).sum()),
                  uncertain_line_pixels=int(affected.sum()),
                  semantic_fill='open', proposal_geometry='observed_source_raster')
    report['fill_evidence'] = describe_fill(template, shape)
    attempt['status'] = 'supported_hollow_identity'
    return template, report


def complete_hollow_geometry(original, report, *, local_completion=False):
    """Build a separate pure outline from the already validated legend fit."""
    import partial_swatch_detector as D
    template = deepcopy(original)
    fit = template.line_crossed_hollow_model
    params = fit['best_model']['params']
    line = fit['line_params']
    diameter = max(params['width'], params['height'])
    radius = int(math.ceil(diameter/2 + 3*line['blur_sigma'])) + 2
    size = 2*radius + 1
    pure = render_model(fit['best_model_name'], dict(params, cx=radius, cy=radius),
                        line, (size, size), Config())['marker_alpha']
    mask = pure >= .25
    a, b, _, _ = template.swatch_box
    center = (a+params['cx'], b+params['cy'])
    oldh, oldw = original.mask.shape
    transform = np.float32([[1, 0, radius-(oldw-1)/2-(center[0]-original.marker_center[0])],
                            [0, 1, radius-(oldh-1)/2-(center[1]-original.marker_center[1])]])
    template.observed_source_soft = cv2.warpAffine(original.raw_soft, transform, (size, size))
    template.source_gray = cv2.warpAffine(original.source_gray, transform, (size, size),
                                          borderValue=original.ink.paper_gray)
    template.raw_soft = pure.copy(); template.soft = pure.copy()
    template.raw_mask = mask.copy(); template.mask = mask
    template.line_nuisance = np.zeros_like(mask)
    template.valid_weight = np.ones_like(pure)
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    template.required_weight = np.where(mask, np.where(depth <= 1.05,
        .35+.40*pure, .85+.15*pure), 0).astype(np.float32)
    template.edge = cv2.Canny(mask.astype(np.uint8)*255, 40, 100) > 0
    template.orientation = D._edge_orientation(pure)
    template.marker_center = center; template.diameter = float(diameter)
    template.model_completed = True
    template.small_hollow_model = dict(name=fit['best_model_name'], params=params,
        blur_sigma=line['blur_sigma'], legend_family_margin=fit['winner_other_family_margin'])
    # The native source sampling region belongs to the original coordinates;
    # GeometryFirst has already retained its measured descriptor separately.
    del template.line_crossed_hollow_region
    report = deepcopy(report)
    report.update(template_policy='validated_hollow_geometry_then_original_fill',
        model_completed=True, fitted_diameter=template.diameter, source_center=list(center),
        small_hollow_model=template.small_hollow_model, semantic_fill='open',
        proposal_geometry='fitted_hollow_outline',
        expected_ink_source='fitted marker outline; source gray retained separately')
    if local_completion:
        from bw_legend_local_completion_v46 import postprocess_geometry
        template,report=postprocess_geometry(original,template,report,path='complete_hollow_geometry',fitted_line=True)
    return template, report

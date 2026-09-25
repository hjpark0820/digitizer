"""A verified hollow model may change size without changing pen width.

This is a calibration hypothesis, never an unconditional template replacement.
Only a shared, independently rechecked scale can enable it for detection.
"""
import cv2
import numpy as np


POLICY = 'shared_hollow_geometry_fixed_stroke_v1'


def eligible(template):
    c = template.get('composition') or {}
    p = (c.get('best_model') or {}).get('params') or {}
    return bool(c.get('template_policy') == 'supported_composition_standalone_shape'
                and c.get('status') == 'supported_simple_shape_model'
                and c.get('best_model_name', '').startswith('open_')
                and all(np.isfinite(p.get(k, np.nan)) and p[k] > 0
                        for k in ('width', 'height', 'stroke_width')))


def render_scaled(original, scaled, scale):
    """Render once on the working pixel grid; retain immutable source facts.

    ``scaled`` already has all observed fields transformed by scaled_template.
    Only model-derived shape/negative-space fields are rebuilt here. Stroke
    width and edge spread remain source-pixel quantities, not marker-size units.
    """
    from color_marker_evidence_v2 import _template_regions
    from legend_composition_v46 import render_model
    if not eligible(original):
        raise ValueError('Fixed-stroke scaling requires a supported hollow composition')
    c = original['composition']
    p = dict(c['best_model']['params'])
    p.update(cx=scaled['center'][0], cy=scaled['center'][1],
             width=p['width']*scale, height=p['height']*scale)
    lp = dict(c['line_params'], x0=-2., x1=-1.)
    alpha = render_model(c['best_model_name'], p, lp, scaled['soft'].shape)['marker_alpha']
    support = alpha >= .18
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    scaled.update(soft=alpha, core=alpha >= .60,
        weight=(alpha*np.where(distance >= 1.5, 1., .65)).astype(np.float32),
        nuisance=(scaled['raw_soft'] >= .15) & ~support,
        central_connector=np.zeros_like(support), uncertain=np.zeros_like(support))
    scaled = _template_regions(scaled)
    # At native 6–9 px sizes, eroding again can erase the entire negative space.
    # The rendered rim and its halo already exclude edge-contaminated pixels.
    hole = scaled['face'] & (alpha < .12) & ~scaled['boundary_uncertain']
    scaled.update(hole_core=hole, uncertain=np.zeros_like(support),
        hollow_fraction=float(hole.sum()/max(int(scaled['face'].sum()), 1)),
        completion_hidden_alpha=np.zeros_like(alpha), completion_visible_alpha=alpha.copy(),
        symbol_stroke_policy=POLICY, working_stroke_width=float(p['stroke_width']),
        working_shape_params=p)
    scaled['provenance'] = dict(scaled.get('provenance', {}),
        working_shape_policy=POLICY, hidden_pixels_are_observed=False)
    return scaled

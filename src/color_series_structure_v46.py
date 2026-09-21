"""Image-evidence series routing. A traced path is not proof of a connector.

All coordinates are source pixels. Furniture is explained in separate fields;
the input image and supplied marker identities are never changed. Thresholds
are dimensionless heuristics, not calibrated classification probabilities.
"""
from copy import deepcopy

import cv2
import numpy as np

from color_marker_evidence import _estimate_model
from color_marker_evidence_v2 import _template_regions, prepare_evidence
from color_marker_window_v2 import verify_window
from color_blend_uncertainty_v46 import window_kwargs
from color_tentative_v46 import trace_retained_path
from triangle_errorbar_v46 import compare_cap

VERSION = 'colour_series_structure_v1'


def xy(points):
    return np.asarray([(p.get('cx', p.get('x')), p.get('cy', p.get('y')))
                       if isinstance(p, dict) else p for p in points], float).reshape(-1, 2)


def sample(field, points):
    p = np.asarray(points, np.float32).reshape(-1, 2)
    return cv2.remap(np.asarray(field, np.float32), p[:, 0][None], p[:, 1][None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel()


def prepare(image, payload):
    """Reuse frozen native alpha and current colour/window/guide primitives.

    New exports preserve full template fields. Old bound payloads can recover
    region masks from their native alpha; no legend re-detection or size search.
    """
    p = payload['plot_area']; plot = [p[0], p[1], p[2]+1, p[3]+1]
    lg = payload.get('legend_box')
    legend = None if lg is None else [lg[0], lg[1], lg[2]+1, lg[3]+1]
    templates, specs, priors = {}, [], {}
    for row in payload['curves']:
        sid = row['name']
        alpha = row.get('marker_alpha')
        if alpha is None:
            continue
        a = np.asarray(alpha, np.float32)
        saved = row.get('verification_template')
        if saved and 'model' in saved and 'raw_bgr' in saved:
            t = deepcopy(saved)
            for key, value in list(t.items()):
                if isinstance(value, list) and key not in ('id', 'rgb', 'center', 'source_center'):
                    arr = np.asarray(value)
                    if arr.ndim >= 2:
                        t[key] = arr
            t['model'] = dict(t['model'], bgr=np.asarray(t['model']['bgr'], np.float32),
                             paper_bgr=np.asarray(t['model']['paper_bgr'], np.float32))
            if not np.array_equal(np.asarray(t['soft'], np.float32), a):
                raise ValueError('Step-5 verification template differs from bound marker alpha')
            if (t.get('id') != sid or not np.allclose(t['center'], row['template_center'])
                    or not np.isclose(t['diameter'], row['diameter'])
                    or not np.allclose(t['rgb'], row['rgb'])):
                raise ValueError('Step-5 verification template identity/geometry mismatch')
        else:
            model = _estimate_model(image[p[1]:p[3]+1, p[0]:p[2]+1], rgb=row['rgb'])
            raw = np.clip(model['paper_bgr']-a[..., None]*(model['paper_bgr']-model['bgr']), 0, 255).astype(np.uint8)
            t = _template_regions(dict(id=sid, rgb=row['rgb'], soft=a.copy(), raw_soft=a.copy(),
                center=row['template_center'], diameter=row['diameter'], raw_bgr=raw, model=model,
                central_connector=np.zeros(a.shape, bool), core=a >= .6, nuisance=np.zeros(a.shape, bool),
                provenance=dict(source='bound_alpha_legacy_handoff', raw_rgb_is_alpha_reconstruction=True)))
        t['id'] = sid
        templates[sid] = t
        specs.append(dict(id=sid, legend_box=legend))
        prior = row.get('triangle_prior')
        if prior is None:
            # Legacy payloads retain the class but not the measured legend prior.
            cls = row.get('marker_class', '')
            applicable = cls in ('filled_triangle', 'filled_inv_triangle')
            prior = dict(applicable=applicable, strength='bound_class',
                         orientation='down' if 'inv_' in cls else 'up')
        priors[sid] = prior
    if not templates:
        return None
    evidence = prepare_evidence(image, plot, specs, max_side=0,
                                template_overrides=templates, guide_policy='model',scale_policy='fixed_1x')
    evidence['priors'] = priors
    return evidence


def clean_fields(evidence, index):
    own = evidence['membership'][index]
    guide = evidence.get('guide_upper')
    # Upper guide explanation gives conservative independent curve support.
    residual = np.maximum(own-(guide[index] if guide is not None else 0), 0)
    valid = evidence['valid'].astype(float)*(1-evidence['ignore_mask'])
    clean = residual*valid
    # A different observed palette colour is not affirmative target-curve ink.
    clean *= 1-np.clip(evidence['other'][index], 0, 1)
    return clean, valid


def classify(points, field, valid, diameter, offset=(0, 0)):
    """Differentiate repeated chords, off-chord continuous ink, and no line.

    Missing ink under other colours/axes does not establish absence. Sparse or
    near-collinear evidence may abstain. Guide dots never provide path support.
    """
    pts = xy(points)-np.asarray(offset)
    groups = []
    for point in sorted(pts, key=lambda q: q[0]):
        if groups and point[0]-groups[-1][0][0] <= .4*diameter:
            groups[-1].append(point)
        else:
            groups.append([point])
    anchors = np.asarray([np.median(g, axis=0) for g in groups]).reshape(-1, 2)
    d = float(diameter)
    ink = (field >= .22).astype(np.uint8)
    distance = cv2.distanceTransform(1-ink, cv2.DIST_L2, 5)
    pairs = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        delta = b-a; length = float(np.linalg.norm(delta))
        if length < 2*d or delta[0] < 1.4*d:
            continue
        unit = delta/length; normal = np.array([-unit[1], unit[0]])
        q = a+np.linspace(.65*d, length-.65*d, max(12, int(length)))[:, None]*unit
        visible = sample(valid, q) > .8
        if visible.mean() < .8:
            continue
        hit = sample(distance, q) <= max(1., .09*d)
        side = .5*((sample(distance, q+normal*max(3., .35*d)) <= 1).mean()+
                   (sample(distance, q-normal*max(3., .35*d)) <= 1).mean())
        pairs.append(dict(a=(a+offset).tolist(), b=(b+offset).tolist(),
                          coverage=float(hit[visible].mean()), contrast=float(hit.mean()-side)))
    traced = trace_retained_path(field, ink > 0, valid > .8)
    path = np.asarray(traced['path'], float).reshape(-1, 2)
    observed = np.asarray(traced['observed'], bool)
    # Trim external tails, preserve only internally supported short gaps.
    if observed.any():
        lo, hi = np.flatnonzero(observed)[[0, -1]]
        for key in ('path', 'val', 'observed', 'filled'):
            traced[key] = np.asarray(traced[key])[lo:hi+1]
        path, observed = traced['path'], traced['observed']
    traced['path'] = path+np.asarray(offset)
    supported = []
    deviations = []
    for pair in pairs:
        a, b = np.array(pair['a'])-offset, np.array(pair['b'])-offset
        use = (path[:, 0] >= a[0]+.65*d) & (path[:, 0] <= b[0]-.65*d)
        n = max(1, int(b[0]-a[0]-1.3*d))
        target = a[1]+(path[:, 0]-a[0])*(b[1]-a[1])/(b[0]-a[0])
        # No nearby marker x column may supply the inter-marker line evidence.
        good = use & observed
        support = min(1., float(good.sum())/n)
        deviation = float(np.median(abs(path[good, 1]-target[good]))) if good.any() else None
        pair.update(path_support=support, path_deviation_px=deviation)
        supported.append(support)
        if support >= .65 and deviation is not None:
            deviations.append(deviation)
    n = len(pairs)
    connected = sum(p['coverage'] >= .8 and p['contrast'] >= .12 for p in pairs)
    fraction = connected/max(n, 1)
    path_support = float(np.mean(supported)) if supported else 0.
    bending = sum(v > max(1.25, .13*d) for v in deviations)
    label, reason = 'uncertain', 'insufficient_or_ambiguous_non_guide_evidence'
    if n >= 3 and path_support >= .55 and bending >= 2 and fraction < .95:
        label, reason = 'fitted_curve', 'continuous_own_colour_ink_repeatedly_departs_from_marker_chords'
    elif n >= 3 and fraction >= .8:
        label, reason = 'connected', 'repeated_direct_chord_ink'
    elif n >= 3 and fraction <= .25 and path_support < .25:
        label, reason = 'markers_only', 'no_repeated_inter_marker_ink_in_visible_gaps'
    return dict(version=VERSION, mode=label, reason=reason, eligible_pairs=n,
                connected_fraction=fraction, inter_marker_path_support=path_support,
                curved_gap_count=bending, pairs=pairs, path_record=traced,
                confidence_kind='heuristic_not_probability', source_pixels_changed=False)


class Verifier:
    """Current colour window + triangle/T guard, cached at fixed native centres."""
    def __init__(self, evidence):
        self.evidence = evidence
        self.indices = {t['id']: i for i, t in enumerate(evidence['templates'])}
        self.cache = {}

    def __call__(self, sid, point):
        x, y = xy([point])[0]
        structural = point.get('source') == 'colocated_original_marker'
        key = sid, float(x), float(y), point.get('candidate_id') if structural else None
        if key in self.cache:
            return self.cache[key]
        e = self.evidence; i = self.indices[sid]; template = e['templates'][i]
        offset = np.asarray(e['source_center_offset'])
        window = verify_window(e['membership'][i], e['other'][i], template, x-offset[0], y-offset[1],
            debug=True, ignore_mask=np.maximum(e['ignore_mask'], ~e['valid']),
            **window_kwargs(e,i),
            guide_model=None if 'guide_membership' not in e else
                (e['guide_membership'][i], e['guide_lower'][i], e['guide_upper'][i]))
        guard = compare_cap(window, template, e['priors'][sid])
        score = float(guard['score'])
        accepted = (score >= .35 and not window['line_only_reject'] and
                    window['observed_fraction'] >= .35 and window['missing'] <= .35)
        result = {k: window[k] for k in ('score', 'missing', 'marker_support', 'observed_fraction',
                    'line_only_reject', 'ignored_marker_fraction', 'guide_explained_marker_fraction')}
        result.update(score=score, accepted=bool(accepted), triangle_guard=guard,
                      thresholds=dict(score=.35, observed_fraction=.35, missing=.35))
        if structural:
            result = partial_occlusion_decision(point, result)
        self.cache[key] = result
        return result


def partial_occlusion_decision(point, full_test):
    """Use independently visible fragments, never the donor colour as ink.

    Evidence is generated afresh from immutable L0 donors in the series router.
    A fully hidden/line-only alternative remains suppressed even if a permissive
    full-window score passes. The partial gate does not require a complete body.
    """
    ev = point.get('image_evidence', {})
    checks = ev.get('checks', {})
    required = ('nonempty_template','own_core_pixels','off_path_pixels',
                'multiple_sectors','donor_occlusion','explained_template')
    fragment = (point.get('activation_eligible') is True and
                all(checks.get(k) is True for k in required))
    coverage = float(ev.get('visible_fragment_coverage', 0.))
    partial = bool(fragment and coverage >= .65)
    accepted = bool(fragment and (full_test['accepted'] or partial))
    return dict(full_test, accepted=accepted,
        score=max(float(full_test['score']), coverage) if accepted else float(full_test['score']),
        full_window_accepted=bool(full_test['accepted']),
        partial_fragment_accepted=partial, visible_fragment_coverage=coverage,
        review_status='partial_fragment' if accepted else point.get('review_status','insufficient_image_evidence'),
        validation_basis='independent_visible_fragment_plus_L0_occluder' if accepted else 'retained_suppressed',
        other_colour_counts_as_positive=False)

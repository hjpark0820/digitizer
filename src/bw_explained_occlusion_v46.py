"""Opt-in failed-grid recovery behind one already verified foreground marker.

The donor roster is frozen before this pass.  Unlike proximity suppression,
the null hypothesis asks whether the foreground marker already explains the
pixels.  A second marker must supply independently visible source boundaries
and improve that same scene.  Hollow foregrounds are tested both as opaque
paper-filled bodies and as transparent strokes; the chosen rendering order
also determines which target pixels can be considered hidden.
"""
from dataclasses import asdict
import math
from time import perf_counter

import cv2
import numpy as np

import bw_partial_visibility_v46 as pv
from bw_partial_source_profiles_v46 import profile_evidence
from bw_suppressed_v46 import encode_marker_mask


VERSION = 'bw-explained-marker-occlusion-v1'
POLICY = 'frozen_active_ownership_saved_grid_same_domain_v2'


def _loss(observed, model, domain, norm, config):
    value = float(np.sum(domain * (observed - model) ** 2))
    for axis in (0, 1):
        w = np.minimum(np.take(domain, range(domain.shape[axis] - 1), axis=axis),
                       np.take(domain, range(1, domain.shape[axis]), axis=axis))
        value += config.boundary_weight * float(np.sum(w *
            (np.diff(observed, axis=axis) - np.diff(model, axis=axis)) ** 2))
    return value / norm


def _layers(marker, front, lines):
    return {
        'transparent_foreground_strokes': (
            pv._union(marker, front, lines), front),
        'opaque_foreground_paper_body': (
            np.clip(front + (1 - pv._face(front)) * pv._union(marker, lines), 0, 1),
            pv._face(front).astype(np.float32)),
    }


def verify_scene(observed, target, front, rivals=None, *, available=None,
                 source_reliability=None, config=None,
                 donor_certificate_valid=False, holdout=None, active_context=(),
                 target_class=None, circle_source=None):
    """Measure one fixed pair.  Caller must establish donor provenance.

    ``donor_certificate_valid`` may only be true after a current source-bound
    certificate check. Native donors instead need held-out source support.
    This pure diagnostic helper never changes active points.
    """
    cfg = config or pv.Config()
    obs = pv._field(observed, 'observed')
    own = pv._field(target, 'target', obs.shape)
    donor = pv._field(front, 'front', obs.shape)
    valid = np.ones_like(obs) if available is None else pv._field(available, 'available', obs.shape)
    alternative = {str(k): pv._field(v, 'rival', obs.shape) for k, v in (rivals or {}).items()}
    context = [(str(row['id']), pv._field(row['marker'], 'active source marker', obs.shape))
               for row in active_context]
    if len({key for key, _ in context}) != len(context):
        raise ValueError('Frozen active source owners must have unique IDs')
    owners = pv._union(*[marker for _, marker in context]) if context else np.zeros_like(obs)
    circle_required = target_class in ('filled_circle', 'open_circle')
    if circle_required:
        from bw_circle_outline_identity_v46 import outline_identity
        # Production supplies geometry from the complete frozen source raster,
        # translated to this ROI. Never fit a smaller circle to target ink.
        if circle_source is None:
            circle_source = outline_identity(own, nuisance=None if source_reliability is None
                                             else 1-np.asarray(source_reliability, np.float32))
    # The same slightly enlarged body envelope is used across the small
    # saved-grid alignment family, never selected from matching target ink.
    target_face = pv._face(own)
    withheld = target_face if holdout is None else np.asarray(holdout, bool)
    if withheld.shape != obs.shape:
        raise ValueError('Pair holdout must match the observed source domain')
    if donor_certificate_valid:
        donor_audit = dict(accepted=True, method='fresh_source_bound_partial_certificate',
                           amplitude=1., center_and_scale_frozen=True)
    else:
        _, _, donor_audit = pv._neighbour_fit(obs, withheld, donor,
            pv._face(donor).astype(np.float32), valid, cfg)
        # Keep the actual calibrated source tone; the held-out fit is a
        # qualification test, not permission to refit the donor to the target.
        donor_audit['method'] = 'native_source_support_outside_target_envelope'
        donor_audit['rendered_amplitude'] = 1.
    all_faces = np.maximum.reduce([withheld, pv._face(donor),
                                   *[pv._face(m) for m in alternative.values()]])
    from bw_exterior_line_family_v46 import external_strokes
    line_holdout = np.maximum.reduce([all_faces, *[pv._face(marker) for _, marker in context]])
    lines, line_records = external_strokes(obs * valid, line_holdout)
    lines = np.clip(lines, 0, 1)
    line_cover = np.minimum(lines, obs)
    domain = cv2.dilate(all_faces.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) * valid
    norm = max(float(np.sum(domain * obs * obs)), 1.e-6)
    # Active source owners are present in H0, H1 and every rival at identical
    # centers, scales and tones. They cannot be omitted to make a duplicate
    # look useful. Only the separately verified donor can erase paper: the
    # other owners contribute ink, never an invented opaque body mask.
    background = pv._union(lines, owners)
    null_models = _layers(np.zeros_like(own), donor, background)
    null_losses = {k: _loss(obs, m[0], domain, norm, cfg) for k, m in null_models.items()}
    # Both H0 and every identity receive exactly the same allowed layer list.
    null_loss = min(null_losses.values())
    models = _layers(own, donor, background)
    model_losses = {k: _loss(obs, m[0], domain, norm, cfg) for k, m in models.items()}
    best = min(model_losses, key=model_losses.get)
    rival_rows = {}
    for name, marker in alternative.items():
        losses = {k: _loss(obs, m[0], domain, norm, cfg)
                  for k, m in _layers(marker, donor, background).items()}
        order = min(losses, key=losses.get)
        rival_rows[name] = dict(loss=losses[order], order=order)
    rival_loss = min([r['loss'] for r in rival_rows.values()] or [null_loss])
    layer_rows = {}
    for name, (_, foreground_cover) in models.items():
        # Paper erasure can only come from a named opaque foreground body.
        # Predicted lines never erase white missing target ink.
        visible_foreground_cover = (np.minimum(foreground_cover, obs)
            if name == 'transparent_foreground_strokes' else foreground_cover)
        cover = np.maximum.reduce([visible_foreground_cover, line_cover, np.minimum(owners, obs)])
        profile = profile_evidence(obs, own, available=valid, occlusion=cover,
            config=cfg, source_reliability=source_reliability)
        circle = dict(required=circle_required, accepted=True)
        if circle_required:
            outline = None
            if circle_source.get('circle_supported') and circle_source.get('geometry'):
                nuisance = cover if source_reliability is None else np.maximum(cover, 1-np.asarray(source_reliability, np.float32))
                outline = outline_identity(obs, nuisance=nuisance, available=valid,
                    geometry=circle_source['geometry'], source_ink_level=circle_source['source_ink_level'])
            circle = dict(required=True, accepted=bool(outline and outline['circle_supported']),
                          source=circle_source, observed=outline)
        independent = own * own * (1 - foreground_cover) * (1 - background) * valid
        unique_fraction = float(independent.sum() / max(float(np.sum(own * own * valid)), 1.e-8))
        explained_ink = pv._union(donor, background)
        remaining = np.clip((obs - explained_ink) / np.maximum(1 - explained_ink, .02), 0, 1)
        unique_recall = float(np.sum(independent * np.clip(remaining / np.maximum(own, .05), 0, 1)) /
                              max(float(independent.sum()), 1.e-8))
        stage_b = dict(scene_loss=model_losses[name], null_loss=null_loss,
            marker_improvement=null_loss-model_losses[name]-cfg.marker_cost,
            identity_margin=rival_loss-model_losses[name], unique_fraction=unique_fraction,
            unique_recall=unique_recall, neighbors_independently_supported=bool(donor_audit['accepted']),
            target_order=name, fixed_denominator=norm, shared_domain_pixels=int(np.count_nonzero(domain)),
            donor_evidence=donor_audit, exterior_lines=line_records,
            ownership_context=dict(policy='all_frozen_active_source_ink_in_both_hypotheses',
                active_ids=[key for key, _ in context],
                source_sha256={key: pv._array_hash(marker) for key, marker in context},
                context_can_erase_paper=False, centers_scales_and_tones_frozen=True),
            target_model_losses=model_losses, null_model_losses=null_losses, rival_models=rival_rows,
            source_pixels_unchanged=True, same_domain_for_null_and_all_identities=True)
        stage_b['circle_outline_identity'] = circle
        passed = bool(profile['stage_a_passed'] and circle['accepted'] and pv._numeric_acceptance(profile, stage_b, cfg))
        layer_rows[name] = dict(accepted=passed, stage_a=profile, stage_b=stage_b)
    # Do not cherry-pick a worse opaque explanation solely to hide failures.
    selected = layer_rows[best]
    a, b = selected['stage_a'], selected['stage_b']
    reason = ('explained_occlusion_source_pair_supported' if selected['accepted'] else
              'foreground_not_independently_verified' if not donor_audit['accepted'] else
              a['reason'] if not a['stage_a_passed'] else
              'circle_outline_not_independently_supported' if not b['circle_outline_identity']['accepted'] else
              'foreground_alone_explains_candidate' if b['marker_improvement'] < cfg.minimum_improvement or
                  b['unique_fraction'] < cfg.minimum_unique_fraction else
              'target_identity_unresolved' if b['identity_margin'] < cfg.minimum_identity_margin else
              'scene_residual_or_unique_support_insufficient')
    return dict(version=VERSION, accepted=selected['accepted'], reason=reason,
        stage_a=a, stage_b=b, layers=layer_rows, selected_layer=best,
        score=float(np.clip(1-b['scene_loss'], 0, 1)), config=asdict(cfg),
        score_kind='one_minus_same_region_scene_loss_not_probability',
        policy=POLICY, reference_points_read=False, source_pixels_modified=False)


def _binding(point):
    value = pv._point_binding(point)
    value['donor_certificate_sha256'] = pv._json_hash(point.get('partial_visibility_certificate'))
    value['marker_aspect'] = float(point.get('marker_aspect', 1.))
    value['class_name'] = point.get('class_name')
    value['source'] = point.get('source', 'bw-grid-v46')
    value['original_detection'] = bool(point.get('original_detection', True))
    return value


def _qualify_donor(image, point, active):
    cert = point.get('partial_visibility_certificate')
    if cert:
        if cert.get('version') == VERSION:
            return False, False
        valid = bool(pv.validate_certificate(image, point, active).get('valid'))
        return valid, valid
    native = bool(point.get('original_detection', True) and
        point.get('source', 'bw-grid-v46') == 'bw-grid-v46' and point.get('marker_mask'))
    return native, False


def _inputs(image, patch_box, model_pack, point, front_point, plot, exclusions):
    a, b, c, d = patch_box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(model_pack['source_paper'])
    obs = np.clip((paper-gray[b:d, a:c])/255., 0, 1)
    shape = obs.shape
    origin = a, b
    target = pv._stamp(np.asarray(model_pack['target'], np.float32), point['cx'], point['cy'], origin, shape)
    front = pv._stamp(np.asarray(model_pack['front'], np.float32), front_point['cx'], front_point['cy'], origin, shape,
        tuple(model_pack['front_offset']))
    rivals = {k: pv._stamp(np.asarray(v, np.float32), point['cx'], point['cy'], origin, shape)
              for k, v in model_pack['rivals'].items()}
    reliability = 1-pv._stamp(1-np.asarray(model_pack['reliability'], np.float32),
                             point['cx'], point['cy'], origin, shape)
    yy, xx = np.indices(shape)
    valid = (xx+a >= plot[0]) & (xx+a < plot[2]) & (yy+b >= plot[1]) & (yy+b < plot[3])
    for ea, eb, ec, ed in exclusions:
        valid &= ~((xx+a >= ea) & (xx+a < ec) & (yy+b >= eb) & (yy+b < ed))
    return obs, target, front, rivals, reliability, valid


def _context_bindings(active, point, front_point, model_pack, patch_box):
    """Select owners by frozen source support inside the fixed crop, not fit.

    New EO hypotheses never bootstrap one another's evidence. Existing
    native/independently recovered active markers remain competing owners,
    even when their class differs from the proposed target's class.
    """
    a, b, c, d = patch_box
    rasters = dict(model_pack['rivals'])
    rasters[model_pack['target_key']] = model_pack['target']
    shape, origin = (d-b, c-a), (a, b)
    # Relevance is intersection with the identical comparison domain, not
    # merely proximity or presence elsewhere in the wider line-fit crop.
    target_faces = [pv._face(pv._stamp(np.asarray(marker, np.float32),
        point['cx'], point['cy'], origin, shape)) for marker in rasters.values()]
    target_faces.append(pv._face(pv._stamp(np.asarray(model_pack['front'], np.float32),
        front_point['cx'], front_point['cy'], origin, shape)))
    domain = cv2.dilate(np.maximum.reduce(target_faces).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.where(domain)
    if not len(xs):
        return []
    left, right, top, bottom = xs.min()+a, xs.max()+a, ys.min()+b, ys.max()+b
    rows = []
    skip = {str(point.get('point_id', '')), str(front_point.get('point_id', ''))}
    for q in active:
        if str(q.get('point_id', '')) in skip or q.get('source') == VERSION:
            continue
        key = str(q.get('swatch_id', q.get('template', '')))
        geometry = model_pack['source_geometries'].get(key)
        # Broad source-only rejection before stamping. Missing local models
        # cause abstention, rather than silently omitting a competing owner.
        radius = max(float(q.get('effective_diameter', q.get('source_diameter', 0.))), 1.)
        if geometry:
            ratio = float(q.get('marker_scale', 1.))/geometry['scale']
            aspect = float(q.get('marker_aspect', 1.))
            if not math.isfinite(ratio*aspect) or ratio <= 0 or aspect <= 0:
                raise ValueError('Invalid frozen active source geometry')
            radius = max(radius, max(geometry['shape'])/2.*ratio*max(aspect, 1.)+2.)
        if q['cx']+radius < left or q['cx']-radius > right or q['cy']+radius < top or q['cy']-radius > bottom:
            continue
        if geometry is None or key not in rasters:
            raise ValueError('Local active marker lacks a frozen source model')
        mismatch = (abs(float(q.get('marker_scale', 1.))-geometry['scale']) > 1.e-6 or
            abs(float(q.get('effective_diameter', geometry['diameter']))-geometry['diameter']) > 1.e-6 or
            abs(aspect-1.) > 1.e-6)
        raster = np.asarray(rasters[key], np.float32)
        if mismatch:
            # A differently scaled owner may reach the ROI even if the
            # template at the proposal scale would miss it. Use conservative
            # transformed source support bounds solely to test relevance;
            # never silently render that owner at the wrong approved scale.
            yy, xx = np.where(raster > .03)
            if not len(xx):
                raise ValueError('Empty local active source model')
            rx = max(ratio*aspect, float(q.get('effective_diameter', geometry['diameter']))/geometry['diameter'])
            ry = max(ratio, float(q.get('effective_diameter', geometry['diameter']))/geometry['diameter'])
            xa = max(0, int(math.floor(q['cx']-a+(xx.min()-(raster.shape[1]-1)/2.)*rx-2)))
            xc = min(shape[1], int(math.ceil(q['cx']-a+(xx.max()-(raster.shape[1]-1)/2.)*rx+3)))
            yb = max(0, int(math.floor(q['cy']-b+(yy.min()-(raster.shape[0]-1)/2.)*ry-2)))
            yd = min(shape[0], int(math.ceil(q['cy']-b+(yy.max()-(raster.shape[0]-1)/2.)*ry+3)))
            if xa >= xc or yb >= yd or not np.any(domain[yb:yd, xa:xc]):
                continue
            raise ValueError('Local active marker geometry differs from its frozen source model: '+
                             str(q.get('point_id', ''))+' / '+key)
        marker = pv._stamp(raster, q['cx'], q['cy'], origin, shape)
        if not np.any((marker > .03) & domain):
            continue
        rows.append(_binding(q))
    return sorted(rows, key=lambda row: (row['point_id'], row['swatch_id'], row['cx'], row['cy']))


def _context_inputs(model_pack, patch_box):
    a, b, c, d = patch_box
    rasters = dict(model_pack['rivals'])
    rasters[model_pack['target_key']] = model_pack['target']
    return [dict(id=row['point_id'], marker=pv._stamp(
        np.asarray(rasters[row['swatch_id']], np.float32), row['cx'], row['cy'], (a, b), (d-b, c-a)))
        for row in model_pack['active_context']]


def _circle_source_in_patch(model_pack, point, patch_box):
    source = model_pack.get('source_circle_outline')
    if source is None:
        return None
    result = dict(source)
    if source.get('geometry'):
        geometry = dict(source['geometry'])
        shape = np.shape(model_pack['target'])
        dx, dy = point['cx']-patch_box[0]-(shape[1]-1)/2., point['cy']-patch_box[1]-(shape[0]-1)/2.
        geometry['center'] = [source['geometry']['center'][0]+dx, source['geometry']['center'][1]+dy]
        for key in ('extent', 'raw_extent'):
            if key in geometry:
                geometry[key] = [value+(dx if i%2 == 0 else dy) for i, value in enumerate(geometry[key])]
        result['geometry'] = geometry
    return result


def validate_certificate(image, point, active):
    """Replay the bound source pair and revalidate its original donor chain."""
    failed = dict(valid=False, explained_occlusion_validated=False, score=0.,
                  identity_margin=0., neighbor_ids=[], reason='invalid_explained_occlusion_certificate')
    try:
        cert = point['partial_visibility_certificate']
        if cert['version'] != VERSION or cert['policy'] != POLICY:
            return dict(failed, reason='explained_occlusion_policy_changed')
        binding = cert['binding']
        if binding['target'] != pv._point_binding(point):
            return dict(failed, reason='target_geometry_changed')
        if abs(float(point.get('marker_aspect', 1.))-1.) > 1.e-6:
            return dict(failed, reason='target_geometry_changed')
        if point.get('class_name') != cert['models']['target_class']:
            return dict(failed, reason='target_semantic_class_changed')
        if list(image.shape) != binding['image_shape']:
            return dict(failed, reason='source_shape_changed')
        if pv._array_hash(image) != binding['image_sha256']:
            return dict(failed, reason='source_image_changed')
        a, b, c, d = binding['patch_box']
        if not (0 <= a < c <= image.shape[1] and 0 <= b < d <= image.shape[0]):
            return dict(failed, reason='invalid_source_patch')
        if pv._array_hash(image[b:d, a:c]) != binding['patch_sha256']:
            return dict(failed, reason='source_patch_changed')
        if pv._json_hash(cert['models']) != binding['model_sha256']:
            return dict(failed, reason='source_models_changed')
        donor = next((q for q in active if str(q.get('point_id')) == binding['front']['point_id']), None)
        if donor is None or _binding(donor) != binding['front']:
            return dict(failed, reason='foreground_geometry_or_certificate_changed')
        eligible, certified = _qualify_donor(image, donor, active)
        if not eligible:
            return dict(failed, reason='foreground_certificate_no_longer_valid')
        context = _context_bindings(active, point, donor, cert['models'], binding['patch_box'])
        if context != cert['models']['active_context']:
            return dict(failed, reason='active_source_ownership_context_changed')
        obs, own, front, rivals, reliability, valid = _inputs(image, binding['patch_box'],
            cert['models'], point, donor, binding['plot'], binding['exclusions'])
        replay = verify_scene(obs, own, front, rivals, available=valid,
            source_reliability=reliability, config=pv.Config(**cert['config']),
            donor_certificate_valid=certified,
            active_context=_context_inputs(cert['models'], binding['patch_box']),
            target_class=cert['models']['target_class'],
            circle_source=_circle_source_in_patch(cert['models'], point, binding['patch_box']))
        if not replay['accepted']:
            return dict(failed, reason='source_pair_replay_rejected', replay_reason=replay['reason'])
        return dict(valid=True, explained_occlusion_validated=True, score=replay['score'],
            reason='bound_explained_occlusion_scene_valid', identity_margin=replay['stage_b']['identity_margin'],
            scene_improvement=replay['stage_b']['marker_improvement'],
            minimum_scene_improvement=cert['config']['minimum_improvement'],
            neighbor_ids=[str(donor['point_id'])], explained_neighbor_ids=[str(donor['point_id'])],
            ownership_context_ids=replay['stage_b']['ownership_context']['active_ids'],
            required_ink_visibility=replay['stage_a']['required_ink_visibility'])
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return failed


def recover(image, plot, exclusions, templates, scales, kept, records, series_indices,
            log_fn=print, *, config=None):
    """Retry saved grid candidates against a frozen, independently valid donor."""
    cfg = config or pv.Config()
    started = perf_counter()
    report = dict(version=VERSION, policy=POLICY, status='completed', added=0, accepted=[],
        rejected=[], candidates=[], suppressed=[], donor_ids=[], seconds=0., tested_candidates=0,
        budget_exhausted=False, reference_points_read=False,
        scope='verified_foreground_plus_frozen_active_ownership_per_saved_grid_candidate')
    if np.ndim(image) != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError('Finite native BGR image required')
    chroma = image.max(axis=2).astype(float)-image.min(axis=2).astype(float)
    if np.count_nonzero(chroma > 24) > .01*chroma.size:
        report['status'] = 'color_route_unchanged'
        return [], report
    by_key = {t.key: t for t in templates}
    models = {}
    for template in templates:
        scale = scales.get(template.key)
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
            continue
        raster = pv._source_template_raster(template, float(scale))
        if raster is not None and template.diameter*float(scale) <= cfg.maximum_diameter:
            models[template.key] = dict(marker=raster[0], mask=raster[1], reliability=raster[2],
                diameter=float(template.diameter*scale), scale=float(scale))
            if template.name in ('filled_circle', 'open_circle'):
                from bw_circle_outline_identity_v46 import outline_identity
                models[template.key]['circle_outline'] = outline_identity(raster[0], nuisance=1-raster[2])
    roster = []
    for q in kept:
        key = q.get('swatch_id', q.get('template'))
        if key not in models:
            continue
        model = models[key]
        if (abs(float(q.get('marker_scale', 1.))-model['scale']) > 1.e-6 or
            abs(float(q.get('effective_diameter', model['diameter']))-model['diameter']) > 1.e-6 or
            abs(float(q.get('marker_aspect', 1.))-1.) > 1.e-6):
            continue
        eligible, certified = _qualify_donor(image, q, kept)
        if eligible:
            roster.append((q, certified))
    report['donor_ids'] = [str(q.get('point_id', '')) for q, _ in roster]
    additions = []
    used = set()
    for row in records:
        if report['tested_candidates'] >= cfg.maximum_candidates or perf_counter()-started > cfg.maximum_seconds:
            report.update(status='candidate_or_time_budget', budget_exhausted=True)
            break
        key = row.get('template', row.get('swatch_id'))
        if key not in models or row.get('selected'):
            continue
        # A saved grid proposal supplies existence evidence, never a label.
        if float(row.get('score', 0)) < cfg.proposal_minimum:
            continue
        x = float(row.get('aligned_x', row.get('x', 0)))
        y = float(row.get('aligned_y', row.get('y', 0)))
        model = models[key]
        diameter = model['diameter']
        tag = key, round(x, 2), round(y, 2)
        if tag in used or not pv._inside(x, y, plot, exclusions):
            continue
        used.add(tag)
        if any(q.get('swatch_id', q.get('template')) == key and
               math.hypot(x-q['cx'], y-q['cy']) < max(1., .10*diameter) for q in [*kept, *additions]):
            continue
        nearby = [(q, certified) for q, certified in roster
                  if math.hypot(x-q['cx'], y-q['cy']) < .90*(diameter+models[q['swatch_id']]['diameter'])/2.]
        nearby.sort(key=lambda value: (math.hypot(x-value[0]['cx'], y-value[0]['cy']), str(value[0]['point_id'])))
        for front_point, certified in nearby[:cfg.maximum_neighbours]:
            if report['tested_candidates'] >= cfg.maximum_candidates:
                break
            front_model = models[front_point['swatch_id']]
            radius = int(math.ceil(1.55*max(diameter, front_model['diameter'])))
            box = [max(0, int(min(x, front_point['cx'])-radius)),
                   max(0, int(min(y, front_point['cy'])-radius)),
                   min(image.shape[1], int(max(x, front_point['cx'])+radius+1)),
                   min(image.shape[0], int(max(y, front_point['cy'])+radius+1))]
            if max(box[2]-box[0], box[3]-box[1]) > cfg.maximum_crop_side:
                continue
            pack = dict(target=model['marker'].tolist(), reliability=model['reliability'].tolist(),
                target_key=key, target_class=by_key[key].name,
                source_circle_outline=model.get('circle_outline'),
                front=front_model['marker'].tolist(),
                # cx/cy is the already aligned center. Saved mask offsets
                # anchor a different exported canvas, not this centered raster.
                front_offset=[0., 0.],
                source_paper=float(np.percentile(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 95)),
                source_geometries={k: dict(scale=m['scale'], diameter=m['diameter'], shape=list(m['marker'].shape))
                                   for k, m in models.items()},
                rivals={k: m['marker'].tolist() for k, m in models.items() if k != key})
            point = dict(cx=x, cy=y)
            try:
                pack['active_context'] = _context_bindings(kept, point, front_point, pack, box)
            except ValueError as error:
                audit = dict(candidate_index=row.get('candidate_index'), swatch_id=key, x=x, y=y,
                    front_id=str(front_point['point_id']), accepted=False,
                    reason='active_source_ownership_unavailable', detail=str(error))
                report['tested_candidates'] += 1
                report['candidates'].append(audit)
                report['rejected'].append(audit)
                continue
            obs, own, front, rivals, reliability, valid = _inputs(image, box, pack, point, front_point, plot, exclusions)
            if not np.any(pv._face(own) & pv._face(front)):
                continue
            scene = verify_scene(obs, own, front, rivals, available=valid, source_reliability=reliability,
                config=cfg, donor_certificate_valid=certified,
                active_context=_context_inputs(pack, box), target_class=pack['target_class'],
                circle_source=_circle_source_in_patch(pack, point, box))
            report['tested_candidates'] += 1
            audit = dict(candidate_index=row.get('candidate_index'), swatch_id=key, x=x, y=y,
                front_id=str(front_point['point_id']), donor_was_certified=certified, **scene)
            report['candidates'].append(audit)
            if not scene['accepted']:
                report['rejected'].append(audit)
                continue
            template = by_key[key]
            from bw_pipeline_v46 import CLASSES
            point_id = 'EO%03d' % (len(additions)+len(report['suppressed'])+1)
            while any(q.get('point_id') == point_id for q in [*kept, *additions, *report['suppressed']]):
                point_id += '_p'
            point.update(class_name=template.name, shape_hint=template.shape_hint or template.name,
                template=key, swatch_id=key, class_idx=series_indices[key],
                shape_idx=CLASSES.index(template.name) if template.name in CLASSES else CLASSES.index('unknown_marker'),
                confidence=scene['stage_a']['visible_fit_quality'], confidence_kind='independent_visible_profile_fit_not_probability',
                source=VERSION, point_id=point_id, original_detection=True, marker_mask=encode_marker_mask(model['mask']),
                marker_offset_x=0., marker_offset_y=0., marker_scale=model['scale'], marker_aspect=1.,
                source_diameter=float(template.diameter), effective_diameter=diameter,
                required_recall=scene['stage_a']['direct_rim_recall'], visible_coverage=scene['stage_a']['visible_coverage'],
                required_ink_visibility=scene['stage_a']['required_ink_visibility'], scene_score=scene['score'])
            cert = dict(scene, models=pack, binding=dict(target=pv._point_binding(point), front=_binding(front_point),
                image_shape=list(image.shape), image_sha256=pv._array_hash(image), patch_box=box,
                patch_sha256=pv._array_hash(image[box[1]:box[3], box[0]:box[2]]),
                model_sha256=pv._json_hash(pack), plot=list(plot), exclusions=[list(v) for v in exclusions]))
            point['partial_visibility_certificate'] = cert
            if point['required_ink_visibility'] <= .50:
                point.update(candidate_id=point_id, original_detection=False, status='suppressed_hypothesis',
                    state='suppressed', priority=0., admission_reason='explained_pair_heavily_occluded')
                report['suppressed'].append(point)
            else:
                additions.append(point)
                report['accepted'].append(dict(audit, point_id=point_id))
            break
    report.update(added=len(additions), seconds=perf_counter()-started)
    log_fn('[v46 explained occlusion] %s; added=%s; suppressed=%s; tested=%s' %
           (report['status'], len(additions), len(report['suppressed']), report['tested_candidates']))
    return additions, report

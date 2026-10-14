"""Opt-in grouped recovery with fixed detected-marker context and source profiles.

The existing grouped recovery is the normal dependency phase, unchanged. Its
output and the supplied active points are frozen before unresolved proposals
are evaluated. This module does not detect/refit neighbours, introduce a new
line detector, call the partial recovery search, or issue a partial certificate.
All scene alternatives use one source raster bank, observation, and loss.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

import bw_grouped_raster_recovery_v46 as grouped
import bw_layered_composite_v46 as layered
import bw_observed_raster_v46 as raster
from bw_partial_source_profiles_v46 import profile_evidence
from bw_raster_identity_v46 import features, joint_decide
from bw_suppressed_v46 import encode_marker_mask, decode_marker_mask


VERSION = 'bw-grouped-fixed-active-scene-v2'
CLASSES = ['filled_circle', 'open_circle', 'filled_square', 'open_square',
           'open_triangle', 'open_inv_triangle', 'filled_triangle', 'filled_inv_triangle',
           'open_rhombus', 'filled_rhombus', 'x_marker', 'plus_marker', 'unknown_marker']


@dataclass(frozen=True)
class Config:
    maximum_seconds: float = 20.
    maximum_bodies: int = 24
    maximum_centres: int = 120
    centres_per_body: int = 5
    maximum_neighbours: int = 3
    proposal_minimum: float = .60
    maximum_models_per_batch: int = 96
    minimum_visible_quality: float = .72
    minimum_visible_coverage: float = .25
    minimum_visible_profiles: int = 8
    minimum_supported_fraction: float = .65
    maximum_missing_paper: float = .18


def _hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def _union(a, b):
    return 1.-(1.-a)*(1.-b)


def _move(array, shift, shape):
    h, w = array.shape
    ty = (shape[0]-h)/2.+shift[1]
    tx = (shape[1]-w)/2.+shift[0]
    return cv2.warpAffine(array.astype(np.float32), np.float32([[1, 0, tx], [0, 1, ty]]),
                          (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.)


def _inside(x, y, plot, exclusions):
    return (plot[0] <= x < plot[2] and plot[1] <= y < plot[3]
            and not any(a <= x < c and b <= y < d for a, b, c, d in exclusions))


def _binding(point, marker, cover, geometry):
    return dict(point_id=str(point.get('point_id', '')), swatch_id=point.get('swatch_id'),
                center=[float(point['cx']), float(point['cy'])],
                source=point.get('source'), original_detection=bool(point.get('original_detection', True)),
                saved_marker_offset=[float(point.get('marker_offset_x', 0.)),
                                     float(point.get('marker_offset_y', 0.))],
                marker_sha256=_hash(marker), cover_sha256=_hash(cover), **geometry)


def _donor_model(point, competition, templates, native_sources=None):
    """Render saved geometry/photometry only. Never fit a donor to target pixels."""
    key = point.get('swatch_id', point.get('template'))
    if key not in templates or not point.get('original_detection', True):
        return None, 'not_original_active'
    if point.get('partial_visibility_certificate') or point.get('partial_visibility_hypothesis'):
        return None, 'dependent_rescue_not_a_donor'
    source = point.get('source', 'bw-grid-v46')
    if source == grouped.VERSION:
        saved = point.get('layered_composite', {})
        parameters = saved.get('parameters', {})
        scale = point.get('observed_raster_scale')
        if key not in competition.templates or scale is None or not parameters:
            return None, 'grouped_render_binding_unavailable'
        marker, weight, cover, _ = competition._variant(key, float(scale))
        gain = float(parameters.get('gain', 1.))
        marker = np.clip(marker*gain, 0, 1)
        cover = np.maximum(cover, marker)
        geometry = dict(scale=float(scale), gain=gain, model_source='saved_grouped_raster',
                        effective_diameter=float(point['effective_diameter']),
                        source_model_policy=competition.report.get('source_model_policy', 'observed_raster'))
    elif source == 'bw-grid-v46' and point.get('marker_mask'):
        t = templates[key]
        scale = float(point.get('marker_scale', 1.))
        if (abs(float(point.get('marker_aspect', 1.))-1.) > 1.e-6 or
                abs(float(point.get('effective_diameter', t.diameter*scale))-t.diameter*scale) > 1.e-4):
            return None, 'native_geometry_not_reproducible'
        from bw_shared_window_source_v46 import native_window_variant
        if native_sources is not None and key in native_sources:
            model = native_sources[key]
            marker, weight, cover, _ = native_window_variant(model['record'], model['prepared'], scale)
            photometry = model.get('meta', {})
            expected_mask = np.asarray(model['prepared']['required_mask'], bool)
        else:
            gray = getattr(t, 'source_gray', None)
            if gray is None:
                return None, 'native_gray_unavailable'
            gray = np.asarray(gray, np.float32)
            if gray.shape != t.mask.shape:
                return None, 'native_gray_canvas_mismatch'
            if getattr(t, 'model_completed', False):
                return None, 'completed_native_shared_source_binding_required'
            ink = np.clip((float(t.ink.paper_gray)-gray)/255., 0, 1)
            weight = np.asarray(t.valid_weight, np.float32)
            peak = max(float(ink.max()), .05)
            core = ink > .28*peak
            holes = binary_fill_holes(core) & ~core
            if not (t.marker_kind == 'open' or t.name.startswith('open_')):
                holes[:] = False
            coverage = np.maximum(np.clip(ink/peak, 0, 1), holes.astype(np.float32))
            marker, weight, cover, _ = native_window_variant({},
                dict(marker=ink, weight=weight, cover=coverage), scale)
            photometry = dict(appearance_source='native_observed_gray', source_gray_is_measured=True)
            expected_mask = np.asarray(t.mask, bool)
        # The saved mask lives at the grid-seed canvas; cx/cy already contain
        # the aligned center. Recenter that mask by MINUS the saved offset to
        # verify the binding, never add that offset to the rendered gray donor.
        saved = decode_marker_mask(point['marker_mask']).astype(np.float32)
        canvas = (max(saved.shape[0], marker.shape[0]), max(saved.shape[1], marker.shape[1]))
        canvas = tuple(s+(s%2 == 0) for s in canvas)
        centered_saved = _move(saved, (-float(point.get('marker_offset_x', 0.)),
                                       -float(point.get('marker_offset_y', 0.))), canvas) > .5
        resized = cv2.resize(expected_mask.astype(np.uint8),
            (max(3, round(expected_mask.shape[1]*scale)), max(3, round(expected_mask.shape[0]*scale))),
            interpolation=cv2.INTER_NEAREST)
        native_expected = np.zeros(marker.shape, np.float32)
        top, left = (marker.shape[0]-resized.shape[0])//2, (marker.shape[1]-resized.shape[1])//2
        native_expected[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
        expected = _move(native_expected, (0., 0.), canvas) > .5
        near_saved = cv2.dilate(centered_saved.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        near_expected = cv2.dilate(expected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        support = min(float(np.sum(expected & near_saved)/max(expected.sum(), 1)),
                      float(np.sum(centered_saved & near_expected)/max(centered_saved.sum(), 1)))
        if support < .90:
            return None, 'native_saved_mask_registration_mismatch'
        geometry = dict(scale=scale, gain=1., model_source='saved_native_source_gray',
                        effective_diameter=float(t.diameter*scale), photometry=photometry,
                        saved_mask_registration_support=support,
                        saved_mask_sha256=_hash(saved),
                        saved_mask_anchor_policy='grid_seed_canvas_minus_aligned_offset',
                        gray_anchor_policy='already_aligned_active_center_no_offset_added')
    else:
        return None, 'unsupported_active_provenance'
    if not np.isfinite(marker).all() or not np.any(marker > .05):
        return None, 'invalid_donor_raster'
    return dict(point=point, marker=marker, cover=np.clip(cover, 0, 1),
                binding=_binding(point, marker, cover, geometry)), None


def _profile_reliability(competition, key, scale):
    """Observation validity is not the raster loss's spatial weighting.

    Grouped's spatial falloff is deliberately zero on distant white pixels.
    Passing it to profile_evidence made the required paper bands unknowable.
    Transport only genuine source uncertainty, with the same raster alignment.
    """
    prepared = competition.prepared[key]
    marker = prepared['marker']
    uncertain = prepared.get('source_uncertainty', prepared.get('uncertain'))
    uncertainty = np.zeros_like(marker, np.float32) if uncertain is None else np.asarray(uncertain, np.float32)
    reliability = np.where(uncertainty > .01, 0., 1.).astype(np.float32)
    rec = competition.templates[key][0]
    if competition.report.get('source_model_policy') == 'shared_native_window':
        from bw_shared_window_source_v46 import native_window_variant
        value, _, _, _ = native_window_variant(rec,
            dict(marker=reliability, weight=np.ones_like(marker), cover=np.ones_like(marker)), scale)
    else:
        value, sampled, _ = raster.raster_variant(reliability, np.ones_like(marker),
            rec['crop_center'], scale, competition._phase())
        value = np.where(sampled > .999, value, 0.)
    target_shape = competition.banks[key]['variants'][scale][0].shape
    return _move(value, (0., 0.), target_shape)


def _line_validity(patch, lines, body, center):
    """The grouped .45 one-sided exterior-continuation rule, unchanged."""
    yy, xx = np.indices(patch.shape)
    results = {}
    for family in lines:
        for line in family:
            ident = id(line)
            if ident in results:
                continue
            ys, xs = np.nonzero(line > .01)
            if not len(xs):
                results[ident] = False
                continue
            horizontal = np.ptp(xs) >= np.ptp(ys)
            axis = (xx-center[0]) if horizontal else (yy-center[1])
            scores = []
            for sign in (-1, 1):
                weight = (~body)*(axis*sign > 0)
                mass = float(np.sum(line*weight))
                scores.append(float(np.sum(np.minimum(patch, line)*weight)/max(mass, 1.e-6)))
            results[ident] = max(scores) >= .45
    return [all(results[id(line)] for line in family) for family in lines]


def _losses(models, patch_feature, weight, energy, boundary_weight):
    fm = features(np.asarray(models, np.float32), weight, boundary_weight)
    return np.maximum(np.sum((fm-patch_feature[None])**2, axis=1), 0.)/energy


def _scene_variants(marker, cover, lines, orders, neighbor, neighbor_cover, has_neighbors):
    # The line bank and line/front-back choices remain the grouped bank's.
    own = layered.compose(marker, cover, lines, orders)
    if not has_neighbors:
        return [('no_neighbor', own)]
    return [('ink_union', _union(own, neighbor)),
            ('neighbor_in_front', np.clip(neighbor+(1.-neighbor_cover)*own, 0, 1)),
            ('target_in_front', layered.compose(np.clip(marker+(1.-cover)*neighbor, 0, 1),
                                               np.maximum(cover, neighbor_cover), lines, orders))]


def _profile_summary(value):
    return {k: v for k, v in value.items() if k != 'profiles'}


def score_family(competition, centers, donors, cfg=Config(), *, profile_cache=None,
                 deadline=math.inf, minimum_visible=None, plot=None, exclusions=()):
    """Fixed-ROI M+N+L/N+L/rival comparison; arrays never use GT or donor refits.

    The original grouped marker banks are reconstructed in small batches to
    add a fixed detected-marker scene. Every identity retains all the same
    line families, gains, scales and drawing-order choices. Boundary profiles
    use that very raster, selected gain, and fixed context; not a second swatch.
    """
    competition._bank()
    centers = [tuple(map(float, c)) for c in centers]
    anchor = centers[0]
    pad = int(math.ceil(max(math.hypot(x-anchor[0], y-anchor[1]) for x, y in centers)))+1
    size = 2*competition.radius+1+2*pad
    shape = (size, size)
    patch = cv2.getRectSubPix(competition.ink, (size, size), anchor)
    yy, xx = np.indices(shape)
    source_x = xx+anchor[0]-(size-1)/2.
    source_y = yy+anchor[1]-(size-1)/2.
    available = ((source_x >= 0) & (source_x < competition.ink.shape[1]) &
                 (source_y >= 0) & (source_y < competition.ink.shape[0]))
    if plot is not None:
        available &= ((source_x >= plot[0]) & (source_x < plot[2]) &
                      (source_y >= plot[1]) & (source_y < plot[3]))
    for a, b, c, d in exclusions:
        available &= ~((source_x >= a) & (source_x < c) & (source_y >= b) & (source_y < d))
    shifts = [(x-anchor[0], y-anchor[1]) for x, y in centers]
    weight = np.maximum.reduce([_move(competition.weight, shift, shape) for shift in shifts])
    neighbor = np.zeros(shape, np.float32)
    neighbor_cover = np.zeros(shape, np.float32)
    bindings = []
    for donor in donors:
        p = donor['point']; offset = (p['cx']-anchor[0], p['cy']-anchor[1])
        nm = _move(donor['marker'], offset, shape)
        nc = _move(donor['cover'], offset, shape)
        neighbor = _union(neighbor, nm)
        neighbor_cover = np.maximum(neighbor_cover, nc)
        bindings.append(donor['binding'])
    # The same added neighbour support belongs to every candidate and null.
    weight = np.maximum(weight, (np.maximum(neighbor, neighbor_cover) > .03).astype(np.float32))
    weight *= available
    pf = features(patch, weight, competition.cfg.boundary_weight)[0]
    energy = max(float(np.sum(pf*pf)), 1.e-6)
    common = dict(anchor=list(anchor), roi_size=size, observation_sha256=_hash(patch),
                  weight_sha256=_hash(weight), fixed_denominator=energy,
                  donor_bindings=bindings, donor_geometry_and_tone_frozen=True,
                  lines_policy='unchanged_grouped_line_bank_and_0.45_continuation',
                  source_policy=competition.report.get('source_model_policy', 'observed_raster'),
                  available_sha256=_hash(available),
                  profile_source='same_selected_grouped_marker_raster',
                  evidence_certificate=False)
    rows = []; complete = True
    for center, shift in zip(centers, shifts):
        if perf_counter() > deadline:
            complete = False; break
        # Share shifted arrays by object identity to retain grouped's cache.
        shifted = {}
        families = []
        for family in competition.family_lines:
            row = []
            for line in family:
                if id(line) not in shifted:
                    shifted[id(line)] = _move(line, shift, shape)
                row.append(shifted[id(line)])
            families.append(row)
        body = _move(competition.context_body.astype(np.float32), shift, shape) > .5
        phase = competition._phase()
        line_center = ((size-1)/2.+shift[0]+phase[0], (size-1)/2.+shift[1]+phase[1])
        valid = _line_validity(patch*available, families, body, line_center)
        null_best = math.inf; null_parameters = None
        for family_id, lines in enumerate(families):
            if not valid[family_id]:
                continue
            line_ink = np.zeros(shape, np.float32)
            for line in lines:
                line_ink = _union(line_ink, line)
            alternatives = [_union(neighbor, line_ink)]
            if donors:
                alternatives.append(np.clip(neighbor+(1-neighbor_cover)*line_ink, 0, 1))
            values = _losses(alternatives, pf, weight, energy, competition.cfg.boundary_weight)
            value = float(values.min()+len(lines)*competition.cfg.line_cost)
            if value < null_best:
                null_best = value; null_parameters = dict(family_id=family_id, line_count=len(lines))
        scores = {}
        for key, bank in competition.banks.items():
            best = None; batch = []; metadata = []
            def consume():
                nonlocal best
                if not batch:
                    return
                losses = _losses(batch, pf, weight, energy, competition.cfg.boundary_weight)
                for loss, item in zip(losses, metadata):
                    value = float(loss+item['parameters']['line_count']*competition.cfg.line_cost)
                    if best is None or value < best['loss']:
                        best = dict(item, loss=value)
                batch.clear(); metadata.clear()
            variants = {scale: (_move(value[0], shift, shape), _move(value[1], shift, shape),
                               _move(_profile_reliability(competition, key, scale), shift, shape))
                        for scale, value in bank['variants'].items()}
            for parameters, family_id in zip(bank['parameters'], bank['line_ids']):
                if not valid[family_id]:
                    continue
                marker, cover, reliability = variants[parameters['raster_scale']]
                marker = np.clip(marker*parameters['gain'], 0, 1)
                cover = np.maximum(cover, marker)
                lines = families[family_id]
                orders = [p['order'] for p in parameters['lines']]
                for order, model in _scene_variants(marker, cover, lines, orders, neighbor,
                                                     neighbor_cover, bool(donors)):
                    batch.append(model)
                    metadata.append(dict(parameters=parameters, family_id=family_id, neighbor_order=order))
                if len(batch) >= cfg.maximum_models_per_batch:
                    consume()
                    if perf_counter() > deadline:
                        complete = False; break
            consume()
            if not complete:
                break
            if best is None:
                continue
            parameters = best['parameters']
            marker, cover, reliability = variants[parameters['raster_scale']]
            marker = np.clip(marker*parameters['gain'], 0, 1)
            lines = families[best['family_id']]
            line_ink = np.zeros(shape, np.float32)
            for line in lines:
                line_ink = _union(line_ink, line)
            # Real white pixels cannot be declared hidden by a predicted line.
            line_occlusion = np.minimum(line_ink, patch)
            nc = neighbor_cover if best['neighbor_order'] == 'neighbor_in_front' else neighbor
            if best['neighbor_order'] == 'target_in_front':
                nc = np.zeros_like(nc)
            occlusion = np.maximum(nc, line_occlusion)
            profile = profile_evidence(patch, marker, available=available.astype(np.float32),
                                       occlusion=occlusion, config=cfg,
                                       profile_cache=profile_cache, source_reliability=np.clip(reliability, 0, 1))
            score_weight = _move(bank['variants'][parameters['raster_scale']][2], shift, shape)
            independent = score_weight*marker*marker*available*(1.-neighbor_cover)*(1.-line_ink)**2
            denominator = max(float(np.sum(score_weight*marker*marker*available)), 1.e-8)
            fraction = float(independent.sum()/denominator)
            nuisance = _union(neighbor, line_ink)
            remaining = np.clip((patch-nuisance)/np.maximum(1.-nuisance, .02), 0, 1)
            visible = float(np.sum(independent*np.clip(remaining/np.maximum(marker, .05), 0, 1)) /
                            max(float(independent.sum()), 1.e-8))
            improvement = null_best-best['loss']
            visible_floor = max(competition.cfg.minimum_visible, minimum_visible or 0.)
            supported = bool(best['loss'] <= competition.cfg.maximum_loss and
                             improvement >= competition.cfg.minimum_improvement and
                             fraction >= competition.cfg.minimum_independent_fraction and
                             visible >= visible_floor and profile['stage_a_passed'])
            scores[key] = dict(loss=best['loss'], line_only_loss=null_best, line_improvement=improvement,
                independent_fraction=fraction, independent_visible=visible,
                parameters=parameters, neighbor_order=best['neighbor_order'],
                profile=_profile_summary(profile), passed=supported,
                local_null_loss=null_best, null_parameters=null_parameters, line_continuation_observed=True)
        if not complete:
            break  # Never decide using an incompletely scored rival set.
        rows.append(dict(center=list(center), scores=scores))
    if not complete or len(rows) != len(centers):
        return dict(status='budget_incomplete_family', rows=rows, common=common,
                    winner=None, reason='all_rivals_and_centers_not_completed')
    # A shifted line-only explanation must challenge every candidate center,
    # not merely the marker model evaluated at that same shift.
    all_nulls = [(score['local_null_loss'], row['center'], score['null_parameters'])
                 for row in rows for score in row['scores'].values()]
    common_null, null_center, null_parameters = min(all_nulls, key=lambda item: item[0])
    common.update(family_null_loss=common_null, family_null_center=null_center,
                  family_null_parameters=null_parameters)
    best = {}; supported = {}
    for row in rows:
        for key, score in row['scores'].items():
            score['line_only_loss'] = common_null
            score['line_improvement'] = common_null-score['loss']
            score['passed'] &= score['line_improvement'] >= competition.cfg.minimum_improvement
            item = dict(score, center=row['center'])
            if key not in best or score['loss'] < best[key]['loss']:
                best[key] = item
            if score['passed'] and (key not in supported or score['loss'] < supported[key]['loss']):
                supported[key] = item
    winner, reason, margin = joint_decide(best, supported, competition.cfg)
    chosen_key = winner or (min(best, key=lambda key: best[key]['loss']) if best else None)
    chosen = supported[winner] if winner else best.get(chosen_key)
    return dict(status='completed', winner=winner, reason=reason, margin=margin,
                selected_key=chosen_key, selected=chosen, rows=rows, common=common)


def _proposals(records, grid_evidence, competition, plot, exclusions, cfg):
    out = []
    # Existing aligned grid records already include small-grid provenance.
    # A nonstandard evidence cache is deliberately not interpreted as GT or
    # as a second source of independent votes.
    for row in records:
        key = row.get('template', row.get('swatch_id'))
        if key not in competition.banks or row.get('selected'):
            continue
        if float(row.get('score', 0.)) < cfg.proposal_minimum:
            continue
        x = float(row.get('aligned_x', row.get('x', math.nan)))
        y = float(row.get('aligned_y', row.get('y', math.nan)))
        if not np.isfinite([x, y]).all() or not _inside(x, y, plot, exclusions):
            continue
        scale = competition.scales.get(key)
        if scale is None:
            continue
        out.append(dict(x=x, y=y, loss=max(0., 1.-float(row.get('score', 0.))),
                        diameter=float(competition.templates[key][0]['diameter']*scale),
                        series_id=key, candidate_index=row.get('candidate_index')))
    return out


def _recover_impl(competition, plot, exclusions, native_templates, kept, records, series_indices,
            cfg=Config(), log_fn=print, minimum_visible=None, source_tone_guard=None,
            window_guard=None, *, grid_evidence=None, donor_source_models=None, _completed=None):
    started = perf_counter()
    report = dict(version=VERSION, status='unsupported', config=asdict(cfg), added=0,
                  accepted=[], rejected=[], suppressed=[], replaced_point_ids=[], replacements=[],
                  scene_added_ids=[], scene_added=0, seconds=0., reference_points_read=False,
                  source_pixels_modified=False, default_enabled=False,
                  donor_policy='frozen_original_native_and_normal_grouped_active',
                  certificates_issued=False, new_stage_replaces_existing_active=False)
    if not competition.ready:
        report.update(reason='raster_competitor_unavailable',
                      competitor_status=competition.report.get('status'), seconds=perf_counter()-started)
        return [], report
    normal, normal_report = grouped.recover(competition, plot, exclusions, native_templates, kept,
        records, series_indices, log_fn=log_fn, minimum_visible=minimum_visible,
        source_tone_guard=source_tone_guard, window_guard=window_guard)
    report.update(normal_grouped=normal_report, replaced_point_ids=normal_report.get('replaced_point_ids', []),
                  replacements=normal_report.get('replacements', []), accepted=list(normal_report.get('accepted', [])),
                  normal_added=len(normal), normal_seconds=normal_report.get('seconds', 0.))
    # The outer guard keeps the completed normal phase if any subsequent
    # experimental context/profile operation fails. Never erase normal points
    # because an optional rescue arm raised after doing useful work.
    if _completed is not None:
        _completed.update(points=normal, report=dict(report, accepted=list(report['accepted'])), started=started)
    removed = set(report['replaced_point_ids'])
    frozen = [p for p in kept if p.get('point_id') not in removed]+list(normal)
    by_key = {t.key: t for t in native_templates}
    donors = []; donor_rejections = []
    for point in frozen:
        donor, reason = _donor_model(point, competition, by_key, donor_source_models)
        if donor is not None:
            donors.append(donor)
        else:
            donor_rejections.append(dict(point_id=point.get('point_id'), reason=reason))
    phase_started = perf_counter(); deadline = phase_started+cfg.maximum_seconds
    competition._bank()
    seeds = _proposals(records, grid_evidence, competition, plot, exclusions, cfg)
    groups = grouped.group_proposals(seeds, grouped.Config(maximum_bodies=cfg.maximum_bodies,
                                                         centres_per_body=cfg.centres_per_body))
    scene = dict(status='completed', bodies=len(groups), proposals=len(seeds), tested_centres=0,
                 families=[], donor_ids=[d['binding']['point_id'] for d in donors],
                 donor_rejections=donor_rejections, added=0, suppressed=0,
                 budget_exhausted=False, profile_search_reused_saved_grid=True,
                 proposal_source='saved_aligned_native_records_only',
                 grid_evidence_available=grid_evidence is not None,
                 grid_evidence_cache_consumed=False,
                 proposal_body_cap_reached=len(groups) >= cfg.maximum_bodies,
                 no_neighbor_fitting=True, no_new_line_detection=True,
                 no_partial_recover_call=True, same_swatch_donors_permitted=True)
    new = []; pending = []; profile_cache = {}; used_ids = {str(p.get('point_id')) for p in frozen}
    for group in groups:
        if perf_counter() >= deadline or scene['tested_centres'] >= cfg.maximum_centres:
            scene.update(status='candidate_or_time_budget', budget_exhausted=True); break
        base = group['centres'][0]; centers = list(group['centres'])
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (base[0]+dx, base[1]+dy)
            if q not in centers and _inside(*q, plot, exclusions):
                centers.append(q)
        centers = centers[:min(cfg.centres_per_body, cfg.maximum_centres-scene['tested_centres'])]
        # Close to an existing center is an already represented identity, not
        # permission to mask that identity with itself and re-add it.
        matching_seeds = [s for s in seeds if math.hypot(s['x']-base[0], s['y']-base[1]) < .35*group['diameter']+1.e-6]
        seed_keys = {s['series_id'] for s in matching_seeds}
        if seed_keys and all(any(p.get('swatch_id') == key and
                math.hypot(p['cx']-base[0], p['cy']-base[1]) <= max(1., .15*group['diameter'])
                for p in frozen) for key in seed_keys):
            report['rejected'].append(dict(x=base[0], y=base[1], reason='existing_center_already_active')); continue
        nearby = [d for d in donors if any(math.hypot(d['point']['cx']-x, d['point']['cy']-y) <
                  .65*(group['diameter']+d['binding']['effective_diameter']) for x, y in centers)]
        if len(nearby) > cfg.maximum_neighbours:
            report['rejected'].append(dict(x=base[0], y=base[1], reason='frozen_neighbor_budget',
                                           donor_count=len(nearby))); continue
        # Cheap source-defined fragment precheck; final acceptance never uses
        # its withheld unknown ink as a proved occluder.
        plausible = False; prechecks = []
        for key in seed_keys:
            scale = competition.scales[key]
            marker, cover, _ = competition.banks[key]['variants'][scale]
            reliability = _profile_reliability(competition, key, scale)
            for center in centers:
                patch = cv2.getRectSubPix(competition.ink, (len(marker), len(marker)), tuple(map(float, center)))
                prof = profile_evidence(patch, marker, config=cfg, provisional=True,
                                        source_reliability=reliability, profile_cache=profile_cache)
                prechecks.append(dict(swatch_id=key, center=list(center),
                                      **_profile_summary(prof)))
                if prof['stage_a_passed']:
                    plausible = True; break
            if plausible: break
        if not plausible:
            report['rejected'].append(dict(x=base[0], y=base[1], reason='source_fragment_precheck_failed',
                                           prechecks=prechecks)); continue
        result = score_family(competition, centers, nearby, cfg, profile_cache=profile_cache,
                              deadline=deadline, minimum_visible=minimum_visible, plot=plot, exclusions=exclusions)
        scene['tested_centres'] += len(result['rows'])
        scene['families'].append(result)
        if result['status'] != 'completed':
            scene.update(status='candidate_or_time_budget', budget_exhausted=True); break
        chosen = result.get('selected'); key = result.get('selected_key')
        if not chosen or key not in by_key or competition.scales.get(key) is None:
            continue
        x, y = chosen['center']; parameters = chosen['parameters']; scale = parameters['raster_scale']
        diameter = float(competition.templates[key][0]['diameter']*scale)
        if any(p.get('swatch_id') == key and math.hypot(p['cx']-x, p['cy']-y) <= max(1., .15*diameter)
               for p in [*frozen, *new, *pending]):
            report['rejected'].append(dict(x=x, y=y, series_id=key, reason='existing_center_already_active')); continue
        profile = chosen['profile']
        body_supported = bool(profile['stage_a_passed'] and chosen['loss'] <= competition.cfg.maximum_loss
                              and chosen['line_improvement'] >= competition.cfg.minimum_improvement
                              and chosen['independent_fraction'] >= competition.cfg.minimum_independent_fraction
                              and chosen['independent_visible'] >= max(competition.cfg.minimum_visible, minimum_visible or 0.))
        if not body_supported:
            report['rejected'].append(dict(x=x, y=y, series_id=key, reason=result['reason'],
                                           loss=chosen['loss'], profile=profile)); continue
        heavy = profile['required_ink_visibility'] <= .50
        active = bool(result['winner'] == key and not heavy)
        # Keep existing guards on an exactly localized new candidate; never
        # silently use an experimental certificate to bypass those guards.
        guard = None
        if source_tone_guard is not None:
            guard = source_tone_guard(key, x, y, diameter, [d['point'] for d in nearby])
            if guard.get('decision') == 'conflict':
                active = False
        if competition.report.get('source_model_policy') == 'shared_native_window':
            # Its physical-window contract has not yet acquired donor-aware
            # measurements. Do not bypass that contract for a new active.
            active = False
            guard = dict(decision='deferred', reason='shared_window_requires_scene_aware_guard')
        marker, weight, cover, _ = competition._variant(key, scale)
        mask = (marker > max(.03, .35*float(marker.max()))) & (weight > .2)
        point_id = 'GS%03d' % (len(new)+len(pending)+1)
        while point_id in used_ids:
            point_id += '_r'
        used_ids.add(point_id)
        t = by_key[key]
        point = dict(point_id=point_id, class_name=t.name, shape_hint=t.shape_hint or t.name,
            swatch_id=key, template=key, class_idx=series_indices[key],
            shape_idx=CLASSES.index(t.name) if t.name in CLASSES else CLASSES.index('unknown_marker'),
            cx=x, cy=y, confidence=profile['visible_fit_quality'],
            confidence_kind='visible_source_profile_fit_not_probability', source=VERSION,
            original_detection=bool(active), marker_mask=encode_marker_mask(mask),
            marker_offset_x=0., marker_offset_y=0., marker_scale=diameter/float(t.diameter), marker_aspect=1.,
            source_diameter=float(t.diameter), effective_diameter=diameter,
            observed_raster_scale=scale, observed_raster_diameter=float(competition.templates[key][0]['diameter']),
            required_recall=profile['direct_rim_recall'], visible_coverage=profile['visible_coverage'],
            required_ink_visibility=profile['required_ink_visibility'], scene_score=1.-chosen['loss'],
            grouped_scene_evidence=dict(version=VERSION, comparison=chosen, margin=result['margin'],
                common=result['common'], guard=guard, ordinary_step5_validation_required=True,
                certificate=False, donor_snapshot_frozen=True))
        audit = dict(series_id=key, x=x, y=y, point_id=point_id, diameter=diameter,
                     loss=chosen['loss'], independent_visible=chosen['independent_visible'],
                     line_improvement=chosen['line_improvement'], parameters=parameters,
                     profile=profile, donor_ids=[d['binding']['point_id'] for d in nearby], guard=guard)
        if active:
            new.append(point); report['accepted'].append(audit)
        else:
            point.update(candidate_id=point_id, state='suppressed', status='suppressed_hypothesis',
                         priority=0., admission_reason=('grouped_scene_heavily_occluded' if heavy else
                                                       'grouped_scene_identity_or_guard_unresolved'))
            pending.append(point)
    scene.update(seconds=perf_counter()-phase_started, added=len(new), suppressed=len(pending))
    report.update(status=scene['status'], scene_recovery=scene, scene_added=len(new),
                  scene_added_ids=[p['point_id'] for p in new], suppressed=pending,
                  added=len(normal)+len(new), seconds=perf_counter()-started,
                  native_points_preserved=normal_report.get('native_points_preserved', True))
    log_fn('[v46 grouped scene] normal=%d; scene=%d; pending=%d; %s' %
           (len(normal), len(new), len(pending), scene['status']))
    return [*normal, *new], report


def recover(competition, plot, exclusions, native_templates, kept, records, series_indices,
            cfg=Config(), log_fn=print, minimum_visible=None, source_tone_guard=None,
            window_guard=None, *, grid_evidence=None, donor_source_models=None):
    """Preserve the completed normal grouped phase on optional scene errors."""
    completed = {}
    try:
        return _recover_impl(competition, plot, exclusions, native_templates, kept, records,
            series_indices, cfg, log_fn, minimum_visible, source_tone_guard, window_guard,
            grid_evidence=grid_evidence, donor_source_models=donor_source_models, _completed=completed)
    except Exception as error:
        if not completed:
            raise
        report = completed['report']
        report.update(status='scene_error_normal_preserved', error_type=type(error).__name__,
            error=str(error), added=len(completed['points']), scene_added=0,
            scene_added_ids=[], suppressed=[], seconds=perf_counter()-completed['started'],
            native_points_preserved=report['normal_grouped'].get('native_points_preserved', True),
            scene_recovery=dict(status='error', error_type=type(error).__name__, error=str(error),
                                completed_normal_preserved=True))
        log_fn('[v46 grouped scene] optional scene error; completed normal points preserved: '+str(error))
        return completed['points'], report

"""Opt-in, local full-source hollow-marker evidence under measured occlusion.

The strict detector supplies a frozen neighbour roster, not target locations.
Good-looking target pixels never define visibility: only independently checked
neighbour models and externally measured lines can hide a source profile.  All
scene alternatives use the same pixels and denominator.  This module neither
changes the native verifier nor enables the global joint-window detector.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from bw_suppressed_v46 import encode_marker_mask, decode_marker_mask


LEGACY_VERSION = 'bw-partial-visibility-source-scene-v1'
FAMILY_VERSION = 'bw-partial-visibility-source-scene-v2'
CONNECTOR_VERSION = 'bw-partial-visibility-source-scene-v3'
VERSION = 'bw-partial-visibility-source-scene-v4'
SOURCE_VERSION = 'bw-partial-visibility-source-scene-v5'
LINE_VISIBILITY_POLICY = 'independent_line_prediction_intersect_observed_ink-v1'


@dataclass(frozen=True)
class Config:
    maximum_candidates: int = 120
    maximum_profile_centres: int = 6000
    maximum_proposal_regions: int = 120
    maximum_seconds: float = 35.
    maximum_neighbours: int = 3
    maximum_diameter: float = 80.
    maximum_crop_side: int = 257
    peaks_per_region: int = 5
    maximum_shift: int = 8
    maximum_family_centres: int = 4
    shift_fraction: float = .22
    proposal_minimum: float = .60
    minimum_visible_quality: float = .72
    minimum_visible_coverage: float = .25
    minimum_visible_profiles: int = 8
    minimum_supported_fraction: float = .65
    maximum_missing_paper: float = .18
    minimum_neighbor_fraction: float = .18
    minimum_neighbor_recall: float = .78
    maximum_neighbor_missing: float = .24
    minimum_unique_fraction: float = .12
    minimum_unique_recall: float = .78
    maximum_scene_loss: float = .40
    minimum_improvement: float = .025
    minimum_identity_margin: float = .015
    marker_cost: float = .015
    boundary_weight: float = .20

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite partial-visibility configuration required: '+name)
        for name in ('maximum_candidates', 'maximum_profile_centres', 'maximum_proposal_regions', 'maximum_neighbours', 'maximum_crop_side',
                     'peaks_per_region', 'maximum_shift', 'maximum_family_centres', 'minimum_visible_profiles'):
            if not isinstance(getattr(self, name), int):
                raise ValueError('Integer partial-visibility limit required: '+name)


def _field(value, name, shape=None):
    a = np.asarray(value)
    if a.ndim != 2 or min(a.shape, default=0) < 3 or not np.isfinite(a).all():
        raise ValueError(name+' must be a finite two-dimensional source array')
    if shape is not None and a.shape != shape:
        raise ValueError(name+' must have the same shape as the source')
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError(name+' must lie in [0, 1]')
    return a.astype(np.float32, copy=True)


def _face(marker):
    return binary_fill_holes(marker > max(.12, .25*float(marker.max())))


def _union(*arrays):
    result = np.zeros_like(arrays[0])
    for a in arrays:
        result = 1-(1-result)*(1-a)
    return result


def provisional_profile_evidence(observed, marker, *, available=None, config=None,
                                 profile_backend='legacy', profile_cache=None, work=None,
                                 source_reliability=None):
    """Stage A, before occluders: fixed profiles, unknown ink is not a match.

    Paper on BOTH flanks identifies a measurable profile independently of
    whether the required rim is present. A missing rim is therefore counted
    against quality, not removed from the visible set. Ink on a flank is an
    unresolved possible occlusion, never positive evidence of a square.
    """
    cfg=config or Config()
    if profile_backend=='source_raster':
        from bw_partial_source_profiles_v46 import profile_evidence as source_profiles
        return source_profiles(observed,marker,available=available,config=cfg,
            provisional=True,profile_cache=profile_cache,work=work,
            source_reliability=source_reliability)
    cached_key=None
    if profile_backend=='vectorized' and profile_cache is not None:
        cached_key=('provisional',_array_hash(np.asarray(observed)),_array_hash(np.asarray(marker)),
            _array_hash(np.asarray(available)) if available is not None else None,_json_hash(asdict(cfg)))
        if cached_key in profile_cache:
            if work is not None:work['provisional_cache_hits']=work.get('provisional_cache_hits',0)+1
            return dict(profile_cache[cached_key])
    raw=profile_evidence(observed,marker,available=available,config=cfg,
        profile_backend=profile_backend,profile_cache=profile_cache,work=work)
    measurable=[];unknown=0;missing=0;supported=0;sides=set()
    for p in raw['profiles']:
        if not p['source_supported'] or not p['visible']:continue
        outside,rim,inside=p['observed']
        if outside<=.30 and inside<=.30:
            measurable.append(p)
            if rim<.30:missing+=1
            if p['quality']>=cfg.minimum_visible_quality and p['supported']:
                supported+=1;sides.add(p['side'])
        else:unknown+=1
    quality=float(np.mean([p['quality'] for p in measurable])) if measurable else 0.
    diverse=bool(any(s%2==0 for s in sides) and any(s%2==1 for s in sides))
    passed=bool(len(measurable)>=4 and supported>=4 and diverse
                and quality>=cfg.minimum_visible_quality)
    result=dict(version=VERSION,stage_a_passed=passed,visible_fit_quality=quality,
        visible_coverage=len(measurable)/36.,visible_profiles=len(measurable),
        supported_profiles=supported,visible_sides=len(sides),spatial_diversity=diverse,
        unknown_ink_profiles=unknown,missing_paper_profiles=missing,total_profiles=36,
        missing_paper_fraction=missing/max(len(measurable),1),
        reason='partial_boundary_proposal' if passed else 'insufficient_provisional_boundary',
        evaluated_before_occluder_fit=True,unknown_counts_as_positive=False,
        selection_basis='fixed_flank_paper_not_best_fitting_rim_pixels')
    if cached_key is not None:
        if len(profile_cache)>=512:profile_cache.clear()
        profile_cache[cached_key]=dict(result)
    return result


def _profile_samples(a, x, y, work=None):
    if work is not None:work['profile_remap_calls']=work.get('profile_remap_calls',0)+1
    return cv2.remap(a, np.asarray(x, np.float32)[None], np.asarray(y, np.float32)[None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]


def _profile_plan(source, geometry, cache=None, work=None):
    """Exact old sample coordinates, batched; no changed interpolation/gates."""
    key=('source_profile_plan',_array_hash(source))
    if cache is not None and key in cache:
        if work is not None:work['source_plan_hits']=work.get('source_plan_hits',0)+1
        return cache[key]
    cx,cy,rx,ry,ix,iy=geometry
    xs=[];ys=[]
    for side in range(4):
        nx,ny=((0,1),(-1,0),(0,-1),(1,0))[side]
        outer,inner=((ry,iy),(rx,ix),(ry,iy),(rx,ix))[side]
        thickness=outer-inner
        if thickness<1.:return None
        extent=(ix,iy,ix,iy)[side]
        bands=(np.linspace(-2.,-.75,5),
            np.linspace(min(.6,.2*thickness),thickness-min(.6,.2*thickness),7),
            np.linspace(thickness+.75,thickness+min(2.5,.60*inner),5))
        for offset in np.linspace(-.65,.65,9):
            tangent=offset*extent
            bx,by=((cx+tangent,cy-ry),(cx+rx,cy+tangent),
                (cx+tangent,cy+ry),(cx-rx,cy+tangent))[side]
            xs.append(np.concatenate([bx+nx*d for d in bands]))
            ys.append(np.concatenate([by+ny*d for d in bands]))
    # Width one preserves the scalar interpolation accumulation used by the
    # old 5/7-sample calls (wider SIMD rows can differ in the last float bit).
    mx=np.asarray(xs,np.float32).reshape(-1,1);my=np.asarray(ys,np.float32).reshape(-1,1)
    sampled=cv2.remap(source,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(36,17)
    expected=np.column_stack([np.median(sampled[:,a:b],axis=1) for a,b in ((0,5),(5,12),(12,17))])
    plan=dict(mx=mx,my=my,expected=expected)
    if cache is not None:
        # Per-recover cache is bounded; it is never shared between images.
        if len(cache)>=512:cache.clear()
        cache[key]=plan
    if work is not None:
        work['source_plan_builds']=work.get('source_plan_builds',0)+1
        work['profile_remap_calls']=work.get('profile_remap_calls',0)+1
    return plan


def profile_evidence(observed, marker, *, available=None, occlusion=None, config=None,
                     profile_backend='legacy', profile_cache=None, work=None,
                     source_reliability=None):
    """Measure all fixed source profiles, with visibility independent of fit.

    ``occlusion`` must come from independently verified scene components, not
    a threshold of dark pixels inside the candidate. The pure helper reports
    evidence; only ``verify_scene`` can authorize an automatic addition.
    """
    cfg = config or Config()
    if profile_backend=='source_raster':
        from bw_partial_source_profiles_v46 import profile_evidence as source_profiles
        return source_profiles(observed,marker,available=available,occlusion=occlusion,config=cfg,
            profile_cache=profile_cache,work=work,source_reliability=source_reliability)
    if profile_backend not in ('legacy','vectorized'):
        raise ValueError('Unknown partial-visibility profile backend')
    if work is not None:work['profile_calls']=work.get('profile_calls',0)+1
    obs = _field(observed, 'observed')
    source = _field(marker, 'marker', obs.shape)
    valid = np.ones_like(obs) if available is None else _field(available, 'available', obs.shape)
    cover = np.zeros_like(obs) if occlusion is None else _field(occlusion, 'occlusion', obs.shape)
    mask = source > max(.12, .25*float(source.max()))
    face = _face(source)
    hole = face & ~mask
    result = dict(version=VERSION, stage_a_passed=False, reason='incomplete_source_hollow',
                  visible_fit_quality=0., visible_coverage=0., visible_profiles=0,
                  total_profiles=36, visible_sides=0, supported_profiles=0,
                  supported_fraction=0., missing_paper_fraction=1., ink_covered_fraction=0.,
                  direct_rim_recall=0., profiles=[], source_pixels_unchanged=True,
                  visibility_policy='fixed_profiles_minus_independent_scene_coverage')
    yy, xx = np.nonzero(face); hy, hx = np.nonzero(hole)
    if len(hx) < 9 or len(xx) < 20:
        return result
    cx, cy = (xx.min()+xx.max())/2., (yy.min()+yy.max())/2.
    rx, ry = (xx.max()-xx.min()+1)/2., (yy.max()-yy.min()+1)/2.
    ix, iy = (hx.max()-hx.min()+1)/2., (hy.max()-hy.min()+1)/2.
    sampled=None;plan=None
    if profile_backend=='vectorized':
        plan=_profile_plan(source,(cx,cy,rx,ry,ix,iy),profile_cache,work)
        if plan is None:return result
        sampled=[cv2.remap(a,plan['mx'],plan['my'],cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT).reshape(36,17) for a in (obs,valid,cover)]
        sampled_values=np.column_stack([np.median(sampled[0][:,a:b],axis=1)
            for a,b in ((0,5),(5,12),(12,17))])
        sampled_visible=np.all(sampled[1]>.99,axis=1)&np.all(sampled[2]<.20,axis=1)
        if work is not None:work['profile_remap_calls']=work.get('profile_remap_calls',0)+3
    profiles = []
    for side in range(4):
        nx, ny = ((0,1),(-1,0),(0,-1),(1,0))[side]
        outer, inner = ((ry,iy),(rx,ix),(ry,iy),(rx,ix))[side]
        thickness = outer-inner
        if thickness < 1.:
            return result
        extent = (ix,iy,ix,iy)[side]
        bands = (np.linspace(-2.,-.75,5),
                 np.linspace(min(.6,.2*thickness), thickness-min(.6,.2*thickness),7),
                 np.linspace(thickness+.75, thickness+min(2.5,.60*inner),5))
        for offset_index,offset in enumerate(np.linspace(-.65,.65,9)):
            tangent = offset*extent
            bx, by = ((cx+tangent,cy-ry),(cx+rx,cy+tangent),
                      (cx+tangent,cy+ry),(cx-rx,cy+tangent))[side]
            values, expected = [], []
            visible = True
            for band_index,distances in enumerate(bands):
                px, py = bx+nx*distances, by+ny*distances
                if sampled is None:
                    values.append(float(np.median(_profile_samples(obs,px,py,work))))
                    expected.append(float(np.median(_profile_samples(source,px,py,work))))
                    visible &= bool(np.all(_profile_samples(valid,px,py,work)>.99)
                                    and np.all(_profile_samples(cover,px,py,work)<.20))
                else:
                    row=side*9+offset_index;a,b=((0,5),(5,12),(12,17))[band_index]
                    values.append(float(sampled_values[row,band_index]))
                    expected.append(float(plan['expected'][row,band_index]))
                    visible &= bool(sampled_visible[row])
            contrast = min(values[1]-values[0],values[1]-values[2])
            source_contrast = min(expected[1]-expected[0],expected[1]-expected[2])
            # A damaged source profile is not positive target evidence. Keep
            # it in the coverage denominator; do not select a better subset.
            source_supported = source_contrast >= .20
            quality = float(np.clip(contrast/max(source_contrast,.25),0,1))
            supported = bool(visible and source_supported and contrast>=.20 and values[1]>=.35)
            profiles.append(dict(side=side, offset=float(offset), visible=bool(visible),
                source_supported=source_supported, observed_contrast=float(contrast),
                source_contrast=float(source_contrast), quality=quality,
                supported=supported, observed=values, expected=expected))
    visible = [p for p in profiles if p['visible'] and p['source_supported']]
    n = len(visible)
    quality = float(np.mean([p['quality'] for p in visible])) if n else 0.
    supported = sum(p['supported'] for p in visible)
    side_counts = [sum(p['supported'] and p['side']==s for p in profiles) for s in range(4)]
    sides = [s for s,nside in enumerate(side_counts) if nside>=2]
    diverse = bool(any(s%2==0 for s in sides) and any(s%2==1 for s in sides))
    rim_mass = source*mask*valid
    mass = max(float(rim_mass.sum()),1.e-8)
    recall = float(np.sum(rim_mass*np.clip(obs/np.maximum(source,.05),0,1))/mass)
    independent = rim_mass*(1-cover)
    # Paper replacing required ink remains a contradiction. Dark unexplained
    # pixels cannot excuse this error; independently modelled cover can.
    missing = float(np.sum(independent*np.clip((source-obs)/np.maximum(source,.05),0,1))/mass)
    coverage = n/36.
    fraction = supported/max(n,1)
    passed = bool(n>=cfg.minimum_visible_profiles and coverage>=cfg.minimum_visible_coverage
                  and quality>=cfg.minimum_visible_quality and fraction>=cfg.minimum_supported_fraction
                  and diverse and missing<=cfg.maximum_missing_paper)
    result.update(stage_a_passed=passed, reason='visible_source_profiles_supported' if passed else
        'missing_required_rim_on_paper' if missing>cfg.maximum_missing_paper else
        'insufficient_independent_profile_evidence', visible_fit_quality=quality,
        visible_coverage=coverage, visible_profiles=n, supported_profiles=supported,
        supported_fraction=float(fraction), visible_sides=len(sides), side_support=side_counts,
        spatial_diversity=diverse, missing_paper_fraction=missing,
        ink_covered_fraction=float(np.sum(rim_mass*cover)/mass), direct_rim_recall=recall,
        profiles=profiles)
    return result


def _neighbour_fit(observed, target_face, marker, cover, available, cfg):
    """Fit amplitude only on held-out pixels; center/scale stay frozen."""
    outside = available*(~cv2.dilate(target_face.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool))
    mass = marker*marker
    fraction = float(np.sum(mass*outside)/max(float(mass.sum()),1.e-8))
    weights = outside*(marker>.15)
    denominator = float(np.sum(weights*marker*marker))
    amplitude = float(np.clip(np.sum(weights*observed*marker)/max(denominator,1.e-8),.8,1.2))
    fitted = np.clip(marker*amplitude,0,1)
    visible_mass = fitted*weights
    recall = float(np.sum(visible_mass*np.clip(observed/np.maximum(fitted,.05),0,1))
                   /max(float(visible_mass.sum()),1.e-8))
    missing = float(np.sum(visible_mass*np.maximum(fitted-observed,0))
                    /max(float(visible_mass.sum()),1.e-8))
    accepted = bool(denominator>=4 and fraction>=cfg.minimum_neighbor_fraction
                    and recall>=cfg.minimum_neighbor_recall and missing<=cfg.maximum_neighbor_missing)
    return fitted, np.maximum(cover,fitted), dict(accepted=accepted,
        outside_target_fraction=fraction, outside_target_recall=recall,
        outside_target_missing=missing, amplitude=amplitude,
        fitted_on='outside_full_target_envelope', center_and_scale_frozen=True)


def _scene_models(marker, face, neighbor, neighbor_cover, lines):
    """Finite declared drawing orders; no invented/erased observation pixels."""
    return {
        'through_ink_union': _union(marker,neighbor,lines),
        'target_in_front': np.clip(marker+(1-face)*_union(neighbor,lines),0,1),
        'neighbor_in_front': np.clip(neighbor+(1-neighbor_cover)*_union(marker,lines),0,1),
        'lines_in_front_target_over_neighbor': _union(np.clip(marker+(1-face)*neighbor,0,1),lines),
    }


def _numeric_acceptance(stage_a, stage_b, cfg):
    """The saved-certificate validator uses the same numeric acceptance gates."""
    try:
        a = stage_a
        b = stage_b
        return bool(a['visible_profiles']>=cfg.minimum_visible_profiles
            and a['visible_coverage']>=cfg.minimum_visible_coverage
            and a['visible_fit_quality']>=cfg.minimum_visible_quality
            and a['supported_fraction']>=cfg.minimum_supported_fraction
            and a['spatial_diversity'] and a['missing_paper_fraction']<=cfg.maximum_missing_paper
            and b['scene_loss']<=cfg.maximum_scene_loss
            and b['marker_improvement']>=cfg.minimum_improvement
            and b['identity_margin']>=cfg.minimum_identity_margin
            and b['unique_fraction']>=cfg.minimum_unique_fraction
            and b['unique_recall']>=cfg.minimum_unique_recall
            and b['neighbors_independently_supported'])
    except (KeyError, TypeError, ValueError):
        return False


def prepare_scene_context(observed, targets, neighbors=(), rival_families=(), *, available=None, config=None):
    """Freeze one nuisance model and comparison domain for a local family.

    ``targets`` is a bounded list of aligned rasters on the same source
    canvas; ``rival_families`` contains one identity-to-raster mapping per
    center. All envelopes are held out during line measurement. Neighbor
    geometry is unchanged and its tone is fitted outside every target.
    """
    cfg=config or Config();obs=_field(observed,'observed')
    valid=np.ones_like(obs) if available is None else _field(available,'available',obs.shape)
    own=[_field(t,'family target',obs.shape) for t in targets]
    if not own or len(own)>cfg.maximum_family_centres:
        raise ValueError('Scene target family must obey maximum_family_centres')
    families=list(rival_families)
    if families and len(families)!=len(own):
        raise ValueError('One rival family is required per target center')
    if not families:families=[{} for _ in own]
    alternatives={};alternative_identities={}
    for index,family in enumerate(families):
        for key,value in family.items():
            raster=_field(value,'family rival',obs.shape)
            name=str(key) if len(own)==1 else str(key)+'@'+_array_hash(raster)[:12]
            alternatives[name]=raster
            alternative_identities[name]=str(key)
    target_face=np.maximum.reduce([_face(t) for t in own])
    prepared = []; audits = []
    for i,n in enumerate(neighbors):
        marker = _field(n['marker'],'neighbor marker',obs.shape)
        cover = _field(n.get('cover',_face(marker)),'neighbor cover',obs.shape)
        fitted,cover,audit = _neighbour_fit(obs,target_face,marker,cover,valid,cfg)
        supported_face=cv2.dilate(_face(marker).astype(np.uint8),np.ones((3,3),np.uint8))>0
        cover_supported=not bool(np.any((cover>.20)&~supported_face))
        audit.update(id=str(n.get('id',i)), independently_verified=bool(n.get('independently_verified',False)))
        audit.update(cover_supported_by_neighbor_model=cover_supported)
        audit['accepted'] &= audit['independently_verified'] and cover_supported
        audits.append(audit)
        if audit['accepted']: prepared.append((fitted,cover))
    neighbor = _union(*[n[0] for n in prepared]) if prepared else np.zeros_like(obs)
    neighbor_cover = np.maximum.reduce([n[1] for n in prepared]) if prepared else np.zeros_like(obs)
    all_faces = np.maximum.reduce([target_face.astype(np.float32),neighbor_cover,
                                 *[_face(m).astype(np.float32) for m in alternatives.values()]])
    # Source-derived lines are held out from every body, including rivals.
    from bw_exterior_line_family_v46 import external_strokes, POLICY as line_policy
    lines, line_records = external_strokes(obs*valid,all_faces>.10)
    lines = np.clip(lines,0,1)
    domain = cv2.dilate((all_faces>.10).astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)*valid
    norm = max(float(np.sum(domain*obs*obs)),1.e-6)
    # A predicted line is a scene hypothesis, not proof that source evidence
    # is hidden. Actual paper cannot occlude a boundary. This intersection
    # can only REMOVE claimed line coverage, never invent a dark occluder.
    # Keep the full fixed line model for every scene loss and unique-target
    # test below: missing or falsely extrapolated line ink remains penalized.
    line_occlusion=np.minimum(lines,obs)
    arrays=dict(neighbor=neighbor,neighbor_cover=neighbor_cover,lines=lines,
                line_occlusion=line_occlusion,domain=domain)
    manifest=dict(version='frozen-local-family-v1',source_sha256=_array_hash(obs),
        exterior_line_policy=line_policy,
        line_visibility_policy=LINE_VISIBILITY_POLICY,
        available_sha256=_array_hash(valid),config_sha256=_json_hash(asdict(cfg)),
        target_sha256=sorted(_array_hash(t) for t in own),
        rival_sha256={k:_array_hash(v) for k,v in sorted(alternatives.items())},
        rival_identities=alternative_identities,
        array_sha256={k:_array_hash(v) for k,v in arrays.items()},
        neighbor_evidence=audits,exterior_lines=line_records,
        fixed_denominator=norm,shared_domain_pixels=int(np.count_nonzero(domain)))
    for a in [*arrays.values(),*alternatives.values()]:a.setflags(write=False)
    return dict(arrays,alternatives=alternatives,alternative_identities=alternative_identities,
                audits=audits,line_records=line_records,
                norm=norm,manifest=manifest,signature=_json_hash(manifest))


def verify_scene(observed, target, neighbors=(), rivals=None, *, available=None, config=None,
                 prepared_context=None,profile_backend='legacy',profile_cache=None,work=None,
                 source_reliability=None):
    """Compare source identities using fixed family-wide context evidence.

    Without ``prepared_context`` this remains the singleton scoring helper.
    With a context, line measurement, neighbors, rivals and denominator are
    identical for every eligible center; only the target's profiles change.
    """
    cfg = config or Config()
    obs = _field(observed,'observed'); own = _field(target,'target',obs.shape)
    valid = np.ones_like(obs) if available is None else _field(available,'available',obs.shape)
    target_face = _face(own)
    # Stage A always precedes any context fitting for the singleton route;
    # recovery has already run this stage before constructing a family.
    provisional=provisional_profile_evidence(obs,own,available=valid,config=cfg,
        profile_backend=profile_backend,profile_cache=profile_cache,work=work,
        source_reliability=source_reliability)
    context=prepared_context if prepared_context is not None else prepare_scene_context(
        obs,[own],neighbors,[rivals or {}],available=valid,config=cfg)
    manifest=context['manifest']
    if (_json_hash(manifest)!=context['signature'] or
        manifest['source_sha256']!=_array_hash(obs) or manifest['available_sha256']!=_array_hash(valid) or
        manifest['config_sha256']!=_json_hash(asdict(cfg)) or _array_hash(own) not in manifest['target_sha256'] or
        context['norm']!=manifest['fixed_denominator'] or
        context['audits']!=manifest['neighbor_evidence'] or context['line_records']!=manifest['exterior_lines'] or
        context['alternative_identities']!=manifest['rival_identities'] or
        set(context['alternatives'])!=set(manifest['rival_sha256']) or
        any(manifest['array_sha256'][k]!=_array_hash(context[k]) for k in manifest['array_sha256']) or
        any(manifest['rival_sha256'][k]!=_array_hash(v) for k,v in context['alternatives'].items())):
        raise ValueError('Prepared scene context does not match the source, family or configuration')
    neighbor=context['neighbor'];neighbor_cover=context['neighbor_cover'];lines=context['lines']
    domain=context['domain'];norm=context['norm'];audits=context['audits'];line_records=context['line_records']
    alternatives=context['alternatives']
    visibility = np.maximum(neighbor_cover,context['line_occlusion'])
    stage_a = profile_evidence(obs,own,available=valid,occlusion=visibility,config=cfg,
        profile_backend=profile_backend,profile_cache=profile_cache,work=work,
        source_reliability=source_reliability)
    # One observed-data denominator for H0, every center and every rival.
    def loss(model):
        pixel = float(np.sum(domain*(obs-model)**2))
        edge = 0.
        for axis in (0,1):
            w = np.minimum(np.take(domain,range(domain.shape[axis]-1),axis=axis),
                           np.take(domain,range(1,domain.shape[axis]),axis=axis))
            edge += float(np.sum(w*(np.diff(obs,axis=axis)-np.diff(model,axis=axis))**2))
        return (pixel+cfg.boundary_weight*edge)/norm
    null_models = {'through':_union(neighbor,lines),
                   'neighbor_in_front':np.clip(neighbor+(1-neighbor_cover)*lines,0,1)}
    null_order = min(null_models,key=lambda k:loss(null_models[k]))
    null_loss = loss(null_models[null_order])
    models = _scene_models(own,target_face.astype(np.float32),neighbor,neighbor_cover,lines)
    order = min(models,key=lambda k:loss(models[k]))
    scene_loss = loss(models[order])
    rival_rows = {}
    for key,marker in alternatives.items():
        variants = _scene_models(marker,_face(marker).astype(np.float32),neighbor,neighbor_cover,lines)
        best = min(variants,key=lambda k:loss(variants[k]))
        identity=context['alternative_identities'][key]
        candidate=dict(loss=float(loss(variants[best])),order=best,raster_sha256=_array_hash(marker))
        if identity not in rival_rows or candidate['loss']<rival_rows[identity]['loss']:
            rival_rows[identity]=candidate
    rival_loss = min([p['loss'] for p in rival_rows.values()] or [null_loss])
    independent = own*own*(1-neighbor_cover)*(1-lines)*valid
    fraction = float(independent.sum()/max(float(np.sum(own*own*valid)),1.e-8))
    remaining = np.clip((obs-_union(neighbor,lines))/np.maximum(1-_union(neighbor,lines),.02),0,1)
    unique_recall = float(np.sum(independent*np.clip(remaining/np.maximum(own,.05),0,1))
                          /max(float(independent.sum()),1.e-8))
    stage_b = dict(scene_loss=float(scene_loss),null_loss=float(null_loss),
        marker_improvement=float(null_loss-scene_loss-cfg.marker_cost),
        identity_margin=float(rival_loss-scene_loss), unique_fraction=fraction,
        unique_recall=unique_recall, neighbors_independently_supported=all(n['accepted'] for n in audits),
        neighbor_evidence=audits, neighbor_ids=[n['id'] for n in audits],
        target_order=order, null_order=null_order, rival_models=rival_rows,
        target_model_losses={k:float(loss(m)) for k,m in models.items()},
        null_model_losses={k:float(loss(m)) for k,m in null_models.items()},
        fixed_denominator=norm, shared_domain_pixels=int(np.count_nonzero(domain)),
        scene_context_id=context['signature'],family_centres=len(manifest['target_sha256']),
        line_visibility_policy=LINE_VISIBILITY_POLICY,
        source_pixels_unchanged=True, exterior_lines=line_records,
        nuisance_fit_region='outside_union_of_all_family_candidate_and_neighbor_bodies')
    accepted = (provisional['stage_a_passed'] and _numeric_acceptance(stage_a,stage_b,cfg)
                and (profile_backend!='source_raster' or stage_a['stage_a_passed']))
    reason = ('source_scene_supported' if accepted else
        'neighbor_not_independently_supported' if not stage_b['neighbors_independently_supported'] else
        'partial_profile_evidence_insufficient' if not provisional['stage_a_passed'] or not stage_a['stage_a_passed'] else
        'target_not_independently_needed' if fraction<cfg.minimum_unique_fraction or
            stage_b['marker_improvement']<cfg.minimum_improvement else
        'target_identity_unresolved' if stage_b['identity_margin']<cfg.minimum_identity_margin else
        'scene_residual_or_unique_support_insufficient')
    # Fit quality and visibility remain separate measurements. This scene
    # score is normalized fit, not a probability or a coverage-weighted fit.
    confidence = float(np.clip(1.-scene_loss,0,1))
    return dict(version=SOURCE_VERSION if profile_backend=='source_raster' else VERSION,
        accepted=accepted,pending=not accepted,reason=reason,
        score=confidence,identity_margin=stage_b['identity_margin'],neighbor_ids=stage_b['neighbor_ids'],
        score_kind='one_minus_common_region_scene_loss_not_probability',
        stage_a_provisional=provisional,stage_a=stage_a,stage_b=stage_b,config=asdict(cfg),reference_points_read=False,
        scene_context=manifest,source_pixels_modified=False)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def _point_binding(point):
    return dict(point_id=str(point.get('point_id','')),swatch_id=str(point.get('swatch_id',point.get('template',''))),
        cx=float(point['cx']),cy=float(point['cy']),marker_scale=float(point.get('marker_scale',1.)),
        effective_diameter=float(point.get('effective_diameter',0.)),
        marker_offset_x=float(point.get('marker_offset_x',0.)),marker_offset_y=float(point.get('marker_offset_y',0.)),
        marker_mask_sha256=_json_hash(point.get('marker_mask')))


def _local_context(point,active,family_centres=None,*,strict_only=False):
    diameter=float(point.get('effective_diameter',0.))
    centres=family_centres or [[point['cx'],point['cy']]]
    rows=[]
    for q in active:
        if q.get('point_id')==point.get('point_id'):continue
        if strict_only and (q.get('source','bw-grid-v46')!='bw-grid-v46' or
                            not q.get('original_detection',True) or q.get('partial_visibility_certificate')):
            continue
        if any(math.hypot(float(q['cx'])-x,float(q['cy'])-y) < .90*(
                diameter+float(q.get('effective_diameter',diameter))) for x,y in centres):
            rows.append(_point_binding(q))
    return sorted(rows,key=lambda p:(p['point_id'],p['swatch_id'],p['cx'],p['cy']))


def validate_certificate(image, point, active):
    """Reject stale source/geometry/context; do not trust a stored passed flag."""
    failed = dict(valid=False,score=0.,reason='missing_partial_visibility_certificate',identity_margin=0.,neighbor_ids=[])
    cert = point.get('partial_visibility_certificate')
    if not isinstance(cert,dict): return failed
    if cert.get('version')=='bw-explained-marker-occlusion-v1':
        from bw_explained_occlusion_v46 import validate_certificate as validate_explained
        return validate_explained(image,point,active)
    try:
        if cert.get('version') not in (SOURCE_VERSION,VERSION,CONNECTOR_VERSION,FAMILY_VERSION,LEGACY_VERSION):
            return dict(failed,reason='certificate_version_changed')
        if cert.get('version') in (SOURCE_VERSION,VERSION,CONNECTOR_VERSION,FAMILY_VERSION):
            context=cert['scene_context']
            if (context.get('version')!='frozen-local-family-v1' or
                _json_hash(context)!=cert['stage_b']['scene_context_id'] or
                context['fixed_denominator']!=cert['stage_b']['fixed_denominator'] or
                context['shared_domain_pixels']!=cert['stage_b']['shared_domain_pixels'] or
                context['config_sha256']!=_json_hash(cert['config'])):
                return dict(failed,reason='scene_context_binding_changed')
            if cert.get('version') in (SOURCE_VERSION,VERSION,CONNECTOR_VERSION):
                from bw_exterior_line_family_v46 import POLICY as line_policy, LEGACY_POLICY as legacy_line_policy
                expected_line_policy=legacy_line_policy if cert.get('version')==CONNECTOR_VERSION else line_policy
                if context.get('exterior_line_policy')!=expected_line_policy:
                    return dict(failed,reason='exterior_line_policy_changed')
            if cert.get('version') in (SOURCE_VERSION,VERSION) and (
                context.get('line_visibility_policy')!=LINE_VISIBILITY_POLICY or
                cert['stage_b'].get('line_visibility_policy')!=LINE_VISIBILITY_POLICY or
                'line_occlusion' not in context['array_sha256']):
                return dict(failed,reason='line_visibility_policy_changed')
        cfg = Config(**cert['config'])
        provisional=cert['stage_a_provisional']
        if cert.get('version')==SOURCE_VERSION:
            from bw_partial_source_profiles_v46 import SOURCE_PLAN_VERSION
            for measurement in (provisional,cert['stage_a']):
                if (measurement.get('source_plan_version')!=SOURCE_PLAN_VERSION or
                        not measurement.get('stage_a_passed') or
                        not 0<=float(measurement.get('required_ink_visibility',-1))<=1):
                    return dict(failed,reason='source_profile_gates_failed')
        source_provisional=(cert.get('version')==SOURCE_VERSION and
            provisional.get('provisional_fragment_gate') and provisional.get('evidence_only') and
            provisional.get('supported_source_fraction',0)>=cfg.minimum_visible_coverage)
        if not (provisional['evaluated_before_occluder_fit'] and
                provisional['visible_profiles']>=4 and provisional['supported_profiles']>=4 and
                provisional['spatial_diversity'] and (source_provisional or
                provisional['visible_fit_quality']>=cfg.minimum_visible_quality)):
            return dict(failed,reason='provisional_profile_gates_failed')
        if not _numeric_acceptance(cert['stage_a'],cert['stage_b'],cfg):
            return dict(failed,reason='certificate_numeric_gates_failed')
        if not all(np.isfinite(float(v)) for v in (cert['score'],cert['identity_margin'])):
            return dict(failed,reason='nonfinite_certificate')
        binding = cert['binding']
        if binding['target']!=_point_binding(point): return dict(failed,reason='target_geometry_changed')
        if list(np.shape(image))!=binding['image_shape']: return dict(failed,reason='source_shape_changed')
        a,b,c,d = binding['patch_box']
        if not (0<=a<c<=image.shape[1] and 0<=b<d<=image.shape[0]):
            return dict(failed,reason='invalid_bound_patch')
        if _array_hash(image[b:d,a:c])!=binding['patch_sha256']:
            return dict(failed,reason='source_patch_changed')
        roster = {str(p.get('point_id','')):p for p in active}
        for old in binding['neighbors']:
            current = roster.get(old['point_id'])
            if current is None or _point_binding(current)!=old:
                return dict(failed,reason='neighbor_context_changed')
        family_centres=binding.get('family_centres') if cert.get('version') in (SOURCE_VERSION,VERSION,CONNECTOR_VERSION,FAMILY_VERSION) else None
        if cert.get('version') in (SOURCE_VERSION,VERSION,CONNECTOR_VERSION,FAMILY_VERSION) and cert['stage_b']['family_centres']>1:
            if (not isinstance(family_centres,list) or len(family_centres)!=cert['stage_b']['family_centres'] or
                [point['cx'],point['cy']] not in family_centres or
                not np.isfinite(np.asarray(family_centres,float)).all()):
                return dict(failed,reason='family_geometry_binding_changed')
        strict_only=cert.get('version')==SOURCE_VERSION
        if strict_only and binding.get('active_context_policy')!='strict_native_roster_v1':
            return dict(failed,reason='source_roster_policy_changed')
        if binding['strict_active_context']!=_local_context(point,active,family_centres,strict_only=strict_only):
            return dict(failed,reason='local_active_context_changed')
        return dict(valid=True,score=float(np.clip(1.-cert['stage_b']['scene_loss'],0,1)),reason='bound_source_scene_certificate_valid',
                    identity_margin=float(cert['stage_b']['identity_margin']),
                    scene_improvement=float(cert['stage_b']['marker_improvement']),
                    minimum_scene_improvement=cfg.minimum_improvement,
                    neighbor_ids=[p['point_id'] for p in binding['neighbors']])
    except (KeyError,TypeError,ValueError,OverflowError,IndexError):
        return dict(failed,reason='malformed_partial_visibility_certificate')


def _template_raster(template, scale):
    mask = np.asarray(template.mask,bool)
    soft = np.asarray(getattr(template,'soft',mask),np.float32)
    if soft.shape!=mask.shape or not np.isfinite(soft).all(): soft=mask.astype(np.float32)
    soft = np.clip(soft,0,1)*mask
    resized = cv2.resize(soft,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    binary = cv2.resize(mask.astype(np.uint8),None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)>0
    side = max(resized.shape)|1
    canvas=np.zeros((side,side),np.float32); full=np.zeros((side,side),bool)
    y=(side-resized.shape[0])//2;x=(side-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    full[y:y+binary.shape[0],x:x+binary.shape[1]]=binary
    return canvas,full


def _source_template_raster(template, scale):
    """Native legend gray and connector uncertainty, never a named-shape glyph.

    Do not multiply by the binary mask or stretch a pale marker to black.
    The uncertainty map affects source profile eligibility, not target pixels.
    Missing required ink on reliable source boundaries remains penalized.
    """
    gray=getattr(template,'source_gray',None)
    if gray is None:
        return None
    gray=np.asarray(gray,np.float32)
    if gray.shape!=np.shape(template.mask) or not np.isfinite(gray).all():
        return None
    paper=float(template.ink.paper_gray)
    soft=np.clip((paper-gray)/255.,0,1)
    reliability=np.ones_like(soft)
    nuisance=np.asarray(template.line_nuisance,bool)
    if nuisance.shape==soft.shape:
        reliability[nuisance]=0.
    resized=cv2.resize(soft,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    reliable=cv2.resize(reliability,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
    side=max(resized.shape)|1
    canvas=np.zeros((side,side),np.float32);weight=np.ones_like(canvas)
    y=(side-resized.shape[0])//2;x=(side-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    weight[y:y+resized.shape[0],x:x+resized.shape[1]]=reliable
    mask=canvas>max(.04,.25*float(canvas.max()))
    return canvas,mask,weight


def _stamp(marker, x, y, origin, shape, offset=(0.,0.)):
    cx=(marker.shape[1]-1)/2.+offset[0];cy=(marker.shape[0]-1)/2.+offset[1]
    return cv2.warpAffine(np.asarray(marker,np.float32),
        np.float32([[1,0,x-origin[0]-cx],[0,1,y-origin[1]-cy]]),
        (shape[1],shape[0]),flags=cv2.INTER_LINEAR)


def _inside(x,y,plot,exclusions):
    return (plot[0]<=x<plot[2] and plot[1]<=y<plot[3]
        and not any(a<=x<c and b<=y<d for a,b,c,d in exclusions))


def _refine_profiles(ink,model,proposal,plot,exclusions,cfg,deadline,remaining,*,
                     strategy='legacy',profile_backend='legacy',profile_cache=None,
                     centre_cache=None,work=None):
    """Bounded translation ranked only by provisional, pre-occlusion evidence.

    ``centre_cache`` is call-scoped to one recover() invocation: source image,
    scale, ROI, exclusions and configuration are frozen for that cache.
    """
    # Validate the real source strictly, before any interpolation tolerance.
    # getRectSubPix's float32 arithmetic can overshoot a valid unit field by
    # a few ulps at fractional coordinates. Only its local output may be
    # bounded; accepting out-of-range source pixels would hide invalid data.
    ink=np.asarray(ink)
    if ink.ndim!=2 or min(ink.shape,default=0)<3 or not np.isfinite(ink).all():
        raise ValueError('ink must be a finite two-dimensional source array')
    if np.any(ink<0) or np.any(ink>1):
        raise ValueError('ink must lie in [0, 1]')
    diameter=model['diameter'];shift=min(cfg.maximum_shift,max(2,int(math.ceil(cfg.shift_fraction*diameter))))
    radius=int(math.ceil(diameter));size=2*radius+1
    offsets=sorted(set(range(-shift,shift+1,2))|{0})
    target=_stamp(model['marker'],radius,radius,(0,0),(size,size))
    reliability=(1-_stamp(1-model['reliability'],radius,radius,(0,0),(size,size))
                 if 'reliability' in model else None)
    checked=[];count=0
    def evaluate(x,y):
        nonlocal count
        if count>=remaining or perf_counter()>deadline or not _inside(x,y,plot,exclusions):return
        token=(model['key'],float(x),float(y))
        if centre_cache is not None and token in centre_cache:
            profile=centre_cache[token]
            if work is not None:work['centre_cache_hits']=work.get('centre_cache_hits',0)+1
        else:
            obs=cv2.getRectSubPix(ink,(size,size),(float(x),float(y)))
            # Keep _field/profile_evidence strict. The bounded tolerance is
            # confined to this trusted interpolation of already valid ink.
            tolerance=128*np.finfo(np.float32).eps
            if (not np.isfinite(obs).all() or np.any(obs < -tolerance)
                    or np.any(obs > 1+tolerance)):
                raise ValueError('Interpolated profile observation exceeds roundoff tolerance')
            if np.any(obs<0) or np.any(obs>1):
                obs=np.clip(obs,0.,1.)
            yy,xx=np.indices(obs.shape);px=xx+x-radius;py=yy+y-radius
            valid=(px>=plot[0])&(px<plot[2])&(py>=plot[1])&(py<plot[3])
            for a,b,c,d in exclusions:valid &= ~((px>=a)&(px<c)&(py>=b)&(py<d))
            profile=provisional_profile_evidence(obs,target,available=valid,config=cfg,
                profile_backend=profile_backend,profile_cache=profile_cache,work=work,
                source_reliability=reliability)
            count+=1
            if centre_cache is not None:centre_cache[token]=profile
        # Coverage is a separate tie-breaker; quality is not multiplied by it.
        rank=(profile['stage_a_passed'],profile['visible_fit_quality'],
              profile['supported_profiles'],-profile['missing_paper_profiles'],
              -math.hypot(x-proposal['x'],y-proposal['y']))
        checked.append((rank,dict(proposal,x=float(x),y=float(y),stage_a_provisional=profile)))
    if strategy=='evidence_reuse' and proposal.get('proposal_source')=='upstream_partial_grid' and not proposal.get('native_passed'):
        # Captured failure rows are evidence, not permission for a new dense
        # search. First ask the unchanged paired-boundary test at that center.
        evaluate(proposal['x'],proposal['y'])
        if not checked or not checked[0][1]['stage_a_provisional']['stage_a_passed']:
            if work is not None:work['weak_grid_single_checks']=work.get('weak_grid_single_checks',0)+1
            return [],count,checked[0][1]['stage_a_provisional'] if checked else None
        if work is not None:work['weak_grid_boundary_promotions']=work.get('weak_grid_boundary_promotions',0)+1
    local_offsets=(-1,0,1) if strategy=='evidence_reuse' else offsets
    seeds=(proposal.get('grid_seed_centres') or [[proposal['x'],proposal['y']]]) if strategy=='evidence_reuse' else [[proposal['x'],proposal['y']]]
    locally_seen=set()
    for sx,sy in seeds[:cfg.maximum_family_centres]:
        for dy in local_offsets:
            for dx in local_offsets:
                xy=(sx+dx,sy+dy)
                if xy not in locally_seen:evaluate(*xy);locally_seen.add(xy)
            if count>=remaining or perf_counter()>deadline:break
        if count>=remaining or perf_counter()>deadline:break
    # A grid location is only a proposal. If its immediate neighborhood lacks
    # the old fixed-boundary evidence, retain the bounded legacy alignment.
    if strategy=='evidence_reuse' and not any(row[1]['stage_a_provisional']['stage_a_passed'] for row in checked):
        if work is not None:work['alignment_fallbacks']=work.get('alignment_fallbacks',0)+1
        visited={(row[1]['x'],row[1]['y']) for row in checked}
        for dy in offsets:
            for dx in offsets:
                xy=(proposal['x']+dx,proposal['y']+dy)
                if xy not in visited:evaluate(*xy)
            if count>=remaining or perf_counter()>deadline:break
    coarse=sorted(checked,key=lambda q:q[0],reverse=True)[:2]
    existing={(q[1]['x'],q[1]['y']) for q in checked}
    for _,p in coarse:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                xy=(p['x']+dx,p['y']+dy)
                if xy not in existing and abs(xy[0]-proposal['x'])<=shift and abs(xy[1]-proposal['y'])<=shift:
                    evaluate(*xy);existing.add(xy)
    chosen=[]
    for _,p in sorted(checked,key=lambda q:q[0],reverse=True):
        if not p['stage_a_provisional']['stage_a_passed']:continue
        if any(math.hypot(p['x']-q['x'],p['y']-q['y'])<2 for q in chosen):continue
        chosen.append(p)
        if len(chosen)>=cfg.maximum_family_centres:break
    return chosen,count,checked[0][1]['stage_a_provisional'] if checked else None


def _proposals(ink, template, plot, exclusions, roster, records, cfg, deadline,work=None):
    """Bounded local NCC; centres are observations, never predicted curve y."""
    key=template['key'];marker=template['marker'];d=template['diameter']
    proposals=[];regions=[]
    for row in records:
        if perf_counter()>deadline or len(regions)>=cfg.maximum_proposal_regions:break
        if row.get('template',row.get('swatch_id'))!=key or row.get('selected'): continue
        x=float(row.get('aligned_x',row.get('x',0.)));y=float(row.get('aligned_y',row.get('y',0.)))
        if _inside(x,y,plot,exclusions):
            proposals.append(dict(x=x,y=y,proposal_score=float(row.get('score',0.)),
                                  proposal_source='saved_grid_candidate',candidate_index=row.get('candidate_index')))
            regions.append((x,y,.35*d,'grid_neighborhood'))
    for point in roster:
        if perf_counter()>deadline or len(regions)>=cfg.maximum_proposal_regions:break
        regions.append((float(point['cx']),float(point['cy']),1.15*d,'strict_neighbor_region'))
    halfx=(marker.shape[1]-1)/2.;halfy=(marker.shape[0]-1)/2.
    for x,y,radius,kind in regions:
        if perf_counter()>deadline:break
        a=max(int(plot[0]),int(math.floor(x-radius-halfx)))
        b=max(int(plot[1]),int(math.floor(y-radius-halfy)))
        c=min(int(plot[2]),int(math.ceil(x+radius+halfx+1)))
        e=min(int(plot[3]),int(math.ceil(y+radius+halfy+1)))
        crop=ink[b:e,a:c]
        if crop.shape[0]<marker.shape[0] or crop.shape[1]<marker.shape[1]:continue
        response=cv2.matchTemplate(crop,marker,cv2.TM_CCORR_NORMED)
        if work is not None:work['ncc_regions']=work.get('ncc_regions',0)+1
        suppress=max(3,int(round(.17*d)))
        for _ in range(cfg.peaks_per_region):
            _,score,_,loc=cv2.minMaxLoc(response)
            if not math.isfinite(score) or score<cfg.proposal_minimum:break
            px,py=loc;cx=a+px+halfx;cy=b+py+halfy
            response[max(0,py-suppress):py+suppress+1,max(0,px-suppress):px+suppress+1]=-1
            if _inside(cx,cy,plot,exclusions):
                proposals.append(dict(x=float(cx),y=float(cy),proposal_score=float(score),proposal_source=kind))
    # Preserve grid candidates and the strongest image peaks without spending
    # the scene budget repeatedly on almost identical locations.
    selected=[]
    for p in sorted(proposals,key=lambda q:(q['proposal_source']!='saved_grid_candidate',-q['proposal_score'])):
        if any(math.hypot(p['x']-q['x'],p['y']-q['y'])<2.5 for q in selected):continue
        selected.append(p)
        if len(selected)>=cfg.maximum_candidates:break
    return selected


def _reused_proposals(ink,model,plot,exclusions,roster,records,grid_evidence,cfg,deadline,work):
    """Balance reused small-grid bodies with observed local-neighbor peaks.

    Sector values order source checks, never bypass profile/scene gates. Raw
    seeds require actual clustered support; they are not accepted identities.
    """
    key=model['key'];diameter=model['diameter'];seeds=[]
    for row in records:
        if row.get('template',row.get('swatch_id'))!=key or row.get('selected'):continue
        x=float(row.get('aligned_x',row.get('x',float('nan'))))
        y=float(row.get('aligned_y',row.get('y',float('nan'))))
        if math.isfinite(x+y) and _inside(x,y,plot,exclusions):
            seeds.append(dict(x=x,y=y,proposal_score=float(row.get('score') or 0.),
                proposal_source='saved_aligned_grid',candidate_index=row.get('candidate_index'),
                boundary_priority=1.,seed_stage='final_window_record'))
    work['saved_aligned_grid_seeds']=len(seeds)
    for row in grid_evidence or []:
        if row.get('template',row.get('swatch_id'))!=key:continue
        sectors=row.get('sectors') or []
        support=int((row.get('seed_support') or {}).get('supporting_cells',0))
        # A matched sector or a clustered grid hypothesis can propose a body;
        # the source-bound full test remains the only admission decision.
        coverage=max([float(s.get('required_coverage',0.)) for s in sectors] or [0.])
        visible_quality=sum(float(s.get('required_coverage',0.))*float(s.get('visible_fraction',0.))
            *(1-float(s.get('mismatch_fraction',1.)))*(1-float(s.get('occluded_fraction',0.)))
            for s in sectors)/max(len(sectors),1)
        if coverage<=0 and support<=0:
            work['unsupported_grid_seeds']=work.get('unsupported_grid_seeds',0)+1;continue
        x=float(row.get('x',float('nan')));y=float(row.get('y',float('nan')))
        if not math.isfinite(x+y) or not _inside(x,y,plot,exclusions):continue
        seeds.append(dict(x=x,y=y,proposal_score=float(row.get('score') or 0.),
            proposal_source='upstream_partial_grid',boundary_priority=visible_quality,
            native_passed=bool(row.get('native_passed',False)),
            seed_stage=row.get('stage'),upstream_rejection=row.get('rejection'),
            supporting_cells=support,grid_sector_evidence=sectors))
        work['upstream_partial_grid_seeds']=work.get('upstream_partial_grid_seeds',0)+1
    grouped=[];radius=max(2.5,.25*diameter)
    grid_budget=max(1,cfg.maximum_candidates//2)
    for p in sorted(seeds,key=lambda q:(q['proposal_source']!='saved_aligned_grid',
            -q['boundary_priority'],-q['proposal_score'],q['x'],q['y'])):
        group=next((q for q in grouped if math.hypot(q['x']-p['x'],q['y']-p['y'])<radius),None)
        if group is not None:
            work['spatial_seed_duplicates']=work.get('spatial_seed_duplicates',0)+1
            members=group['grid_seed_centres']
            if len(members)<cfg.maximum_family_centres and all(math.hypot(x-p['x'],y-p['y'])>=2 for x,y in members):
                members.append([p['x'],p['y']])
            continue
        grouped.append(dict(p,grid_seed_centres=[[p['x'],p['y']]]))
        if len(grouped)>=grid_budget:break
    # Reserve an independent half-budget for measured local-neighbor peaks.
    # An unresolved grid body must not postpone this work until all raw seeds
    # have consumed the center budget. No neighbor's center is a target label.
    fallback_roster=[]
    for q in roster:
        if any(math.hypot(p['cx']-q['cx'],p['cy']-q['cy'])<.35*diameter for p in fallback_roster):continue
        fallback_roster.append(q)
    work['fallback_neighbor_regions']=len(fallback_roster)
    fallback=_proposals(ink,model,plot,exclusions,fallback_roster,[],cfg,deadline,work)
    fallback=fallback[:max(0,cfg.maximum_candidates-len(grouped))]
    interleaved=[]
    for i in range(max(len(grouped),len(fallback))):
        if i<len(grouped):interleaved.append(grouped[i])
        if i<len(fallback):interleaved.append(fallback[i])
    work['interleaved_neighbor_seeds']=len(fallback)
    work['body_searches']=work.get('body_searches',0)+len(interleaved)
    return interleaved


def recover(image, plot, exclusions, templates, scales, kept, records, series_indices,
            log_fn=print, *, config=None,grid_evidence=None,strategy='legacy',profile_backend=None):
    """Return conservative additions; all-shape source profiles are opt-in."""
    cfg=config or Config();started=perf_counter()
    if strategy not in ('legacy','evidence_reuse','source_raster'):
        raise ValueError('Unknown partial visibility search strategy')
    all_shapes=strategy=='source_raster'
    search_strategy='evidence_reuse' if all_shapes else strategy
    backend=profile_backend or ('source_raster' if all_shapes else 'vectorized' if strategy=='evidence_reuse' else 'legacy')
    if backend not in ('legacy','vectorized','source_raster'):raise ValueError('Unknown profile backend')
    if all_shapes != (backend=='source_raster'):
        raise ValueError('Source-raster profiles require source_raster strategy')
    version=SOURCE_VERSION if all_shapes else VERSION
    work={};profile_cache={} if backend!='legacy' else None
    centre_cache={} if search_strategy=='evidence_reuse' else None
    report=dict(version=version,status='not_applicable',added=0,config=asdict(cfg),
        accepted=[],pending=[],rejected=[],candidates=[],suppressed=[],families=[],
        seconds=0.,tested_candidates=0,profile_centres=0,budget_exhausted=False,
        strategy=strategy,profile_backend=backend,work=work,
        reference_points_read=False,source_pixels_modified=False,
        scope='native_bw_all_source_shapes_local_scene' if all_shapes else 'native_bw_open_square_local_source_scene',
        global_joint_window_required=False)
    if np.ndim(image)!=3 or image.shape[2]!=3 or not np.isfinite(image).all():
        raise ValueError('Finite native BGR source image required')
    chroma=image.max(axis=2).astype(float)-image.min(axis=2).astype(float)
    if np.count_nonzero(chroma>24)>.01*chroma.size:
        report['status']='color_route_unchanged';return [],report
    by_key={t.key:t for t in templates}
    eligible=list(templates) if all_shapes else [t for t in templates if t.name=='open_square']
    if not eligible:report['status']='no_source_templates' if all_shapes else 'no_open_square';return [],report
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink=np.clip((float(np.percentile(gray,95))-gray)/255.,0,1)
    models={}
    for t in templates:
        scale=scales.get(t.key)
        if scale is None or not math.isfinite(float(scale)) or float(scale)<=0:
            report['status']='unresolved_shared_scale';return [],report
        if t.diameter*float(scale)>cfg.maximum_diameter:
            report['status']='source_diameter_budget';return [],report
        source_model=_source_template_raster(t,float(scale)) if all_shapes else None
        if all_shapes and source_model is None:
            report['status']='missing_native_legend_raster';return [],report
        marker,mask=source_model[:2] if all_shapes else _template_raster(t,float(scale))
        models[t.key]=dict(key=t.key,marker=marker,mask=mask,scale=float(scale),diameter=float(t.diameter*scale))
        if all_shapes:models[t.key]['reliability']=source_model[2]
    # Never bootstrap from this rescue or another additive recovery arm.
    roster=[p for p in kept if p.get('original_detection',True)
            and p.get('source','bw-grid-v46')=='bw-grid-v46'
            and p.get('swatch_id',p.get('template')) in models
            and p.get('marker_mask') and not p.get('partial_visibility_certificate')]
    report['strict_neighbor_ids']=[str(p.get('point_id','')) for p in roster]
    additions=[];tested=0;profile_centres=0;budget=False;seen_centres=set()
    for template in eligible:
        model=models[template.key]
        if not all_shapes and (not _face(model['marker']).any() or (_face(model['marker']) & ~model['mask']).sum()<9):
            report['rejected'].append(dict(swatch_id=template.key,reason='source_not_complete_hollow'));continue
        proposals=(_reused_proposals(ink,model,plot,exclusions,roster,records,grid_evidence,cfg,
            started+cfg.maximum_seconds,work) if search_strategy=='evidence_reuse' else
            _proposals(ink,model,plot,exclusions,roster,records,cfg,started+cfg.maximum_seconds,work))
        if perf_counter()-started>cfg.maximum_seconds:
            budget=True;break
        def refined_proposals():
            nonlocal budget,profile_centres
            # Interleave one seed's localization and scene checks. Dense
            # profiling must not consume the entire budget before ANY scene.
            def is_resolved(x,y):
                return any(q.get('swatch_id',q.get('template'))==template.key and
                    math.hypot(x-q['cx'],y-q['cy'])<.65*model['diameter'] for q in [*kept,*additions])
            def localize(seed,mode):
                nonlocal budget,profile_centres
                if perf_counter()-started>cfg.maximum_seconds or profile_centres>=cfg.maximum_profile_centres:
                    budget=True;return []
                variants,centres_used,diagnostic=_refine_profiles(ink,model,seed,plot,exclusions,cfg,
                    started+cfg.maximum_seconds,cfg.maximum_profile_centres-profile_centres,
                    strategy=mode,profile_backend=backend,profile_cache=profile_cache,
                    centre_cache=centre_cache,work=report['work'])
                profile_centres+=centres_used
                if not variants:
                    report['rejected'].append(dict(seed,swatch_id=template.key,reason='provisional_profiles_failed',
                                                   stage_a_provisional=diagnostic))
                return variants
            for seed in proposals:
                if search_strategy=='evidence_reuse' and is_resolved(seed['x'],seed['y']):
                    work['resolved_body_searches_skipped']=work.get('resolved_body_searches_skipped',0)+1;continue
                count_before=len(additions)
                variants=localize(seed,search_strategy)
                if budget:return
                if variants:yield variants
                if search_strategy=='evidence_reuse' and variants and len(additions)==count_before:
                    # A provisional local pass is not a complete scene pass.
                    # Keep the old bounded alignment family if context fails.
                    work['scene_alignment_fallbacks']=work.get('scene_alignment_fallbacks',0)+1
                    expanded=localize(seed,'legacy')
                    if budget:return
                    if expanded:yield expanded
            if search_strategy=='evidence_reuse' and not work.get('interleaved_neighbor_seeds'):
                # Weak/failed grid bodies must not permanently mask the
                # strict-neighbor fallback. Only a recovered own body can.
                deferred=[q for q in roster if not is_resolved(q['cx'],q['cy']) and
                    any(p.get('proposal_source') in ('saved_aligned_grid','upstream_partial_grid') and
                        math.hypot(p['x']-q['cx'],p['y']-q['cy'])<.65*model['diameter'] for p in proposals)]
                work['deferred_neighbor_regions']=work.get('deferred_neighbor_regions',0)+len(deferred)
                extra=_proposals(ink,model,plot,exclusions,deferred,[],cfg,started+cfg.maximum_seconds,work)
                for seed in extra:
                    if is_resolved(seed['x'],seed['y']):continue
                    work['deferred_neighbor_seeds']=work.get('deferred_neighbor_seeds',0)+1
                    expanded=localize(seed,'legacy')
                    if budget:return
                    if expanded:yield expanded
        for proposed_family in refined_proposals():
            if tested>=cfg.maximum_candidates or perf_counter()-started>cfg.maximum_seconds:
                budget=True;break
            diameter=model['diameter'];family=[]
            for p in proposed_family:
                x,y=p['x'],p['y'];location=(template.key,round(x,2),round(y,2))
                if search_strategy=='evidence_reuse':
                    location+=(tuple(sorted((round(q['x'],2),round(q['y'],2)) for q in proposed_family)),)
                if location in seen_centres:continue
                seen_centres.add(location)
                if any(q.get('swatch_id',q.get('template'))==template.key and
                       math.hypot(x-q['cx'],y-q['cy'])<.88*(diameter+float(q.get('effective_diameter',diameter)))/2.
                       for q in [*kept,*additions]):
                    report['rejected'].append(dict(p,swatch_id=template.key,reason='same_series_existing_marker'));continue
                family.append(p)
            if not family:continue
            # A center-independent crop and donor roster; all family bodies
            # are withheld together before measuring ANY exterior line.
            x=(min(p['x'] for p in family)+max(p['x'] for p in family))/2.
            y=(min(p['y'] for p in family)+max(p['y'] for p in family))/2.
            neighbors=[q for q in roster if q.get('swatch_id',q.get('template'))!=template.key and
                any(math.hypot(p['x']-q['cx'],p['y']-q['cy'])<
                    .90*(diameter+float(q.get('effective_diameter',diameter))) for p in family)]
            neighbors=sorted(neighbors,key=lambda q:(str(q.get('point_id','')),q['cx'],q['cy']))
            if len(neighbors)>cfg.maximum_neighbours:
                report['pending'].extend(dict(p,swatch_id=template.key,reason='neighbor_budget') for p in family);continue
            radius=int(math.ceil(max([*[math.hypot(x-p['x'],y-p['y'])+1.55*diameter for p in family],*[
                math.hypot(x-q['cx'],y-q['cy'])+.8*float(q.get('effective_diameter',diameter)) for q in neighbors]])))
            if 2*radius+1>cfg.maximum_crop_side:
                report['pending'].extend(dict(p,swatch_id=template.key,reason='crop_budget') for p in family);continue
            a=max(0,int(math.floor(x-radius)));b=max(0,int(math.floor(y-radius)))
            c=min(image.shape[1],int(math.ceil(x+radius+1)));e=min(image.shape[0],int(math.ceil(y+radius+1)))
            observed=ink[b:e,a:c];origin=(a,b);shape=observed.shape
            yy,xx=np.indices(shape);valid=((xx+a>=plot[0])&(xx+a<plot[2])&(yy+b>=plot[1])&(yy+b<plot[3]))
            for ea,eb,ec,ed in exclusions:valid &= ~((xx+a>=ea)&(xx+a<ec)&(yy+b>=eb)&(yy+b<ed))
            targets=[_stamp(model['marker'],p['x'],p['y'],origin,shape) for p in family]
            rival_families=[{k:_stamp(m['marker'],p['x'],p['y'],origin,shape)
                            for k,m in models.items() if k!=template.key} for p in family]
            neighbor_models=[]
            for q in neighbors:
                mask=decode_marker_mask(q['marker_mask']).astype(np.float32)
                marker=_stamp(mask,q['cx'],q['cy'],origin,shape,
                              (float(q.get('marker_offset_x',0.)),float(q.get('marker_offset_y',0.))))
                neighbor_models.append(dict(marker=marker,cover=_face(marker),
                    id=str(q.get('point_id','')),independently_verified=True))
            context=prepare_scene_context(observed,targets,neighbor_models,rival_families,available=valid,config=cfg)
            family_audit=dict(scene_context_id=context['signature'],centres=[[p['x'],p['y']] for p in family],
                fixed_denominator=context['norm'],shared_domain_pixels=context['manifest']['shared_domain_pixels'],
                source_patch_box=[a,b,c,e],neighbor_ids=[str(q.get('point_id','')) for q in neighbors],
                lines_fitted_once=True,tested_centres=0,selected_center=None)
            report['families'].append(family_audit)
            accepted=[];pending_scenes=[]
            for p,target in zip(family,targets):
                if tested>=cfg.maximum_candidates or perf_counter()-started>cfg.maximum_seconds:
                    budget=True;break
                scene=verify_scene(observed,target,available=valid,config=cfg,prepared_context=context,
                    profile_backend=backend,profile_cache=profile_cache,work=report['work'],
                    source_reliability=(1-_stamp(1-model['reliability'],p['x'],p['y'],origin,shape)
                                        if all_shapes else None))
                tested+=1;family_audit['tested_centres']+=1
                diagnostic=dict(p,swatch_id=template.key,family_selected=False,**scene)
                report['candidates'].append(diagnostic)
                if not scene['accepted']:
                    report['pending' if scene['stage_a']['stage_a_passed'] else 'rejected'].append(diagnostic)
                    if all_shapes and scene['stage_a']['stage_a_passed']:
                        pending_scenes.append((p,scene,diagnostic))
                else:accepted.append((p,scene,diagnostic))
            if not accepted:
                if not pending_scenes:
                    if budget:break
                    continue
                accepted=pending_scenes
            # An invalid high-scoring center can never displace a valid one.
            p,scene,diagnostic=min(accepted,key=lambda r:(r[1]['stage_b']['scene_loss'],
                -r[1]['stage_a']['visible_fit_quality'],-r[1]['stage_a']['visible_coverage'],r[0]['x'],r[0]['y']))
            x,y=p['x'],p['y'];diagnostic['family_selected']=True
            family_audit['selected_center']=[x,y]
            point_id='PV%03d'%(len(additions)+len(report['suppressed'])+1)
            while any(q.get('point_id')==point_id for q in [*kept,*additions,*report['suppressed']]):point_id+='_p'
            from bw_pipeline_v46 import CLASSES
            point=dict(class_name=template.name,shape_hint=template.shape_hint or template.name,
                template=template.key,swatch_id=template.key,class_idx=series_indices[template.key],
                shape_idx=CLASSES.index(template.name) if template.name in CLASSES else CLASSES.index('unknown_marker'),
                cx=float(x),cy=float(y),confidence=scene['stage_a']['visible_fit_quality'],
                confidence_kind='independent_visible_profile_fit_not_probability',
                visible_coverage=scene['stage_a']['visible_coverage'],scene_score=scene['score'],
                required_recall=scene['stage_a']['direct_rim_recall'],source=version,
                point_id=point_id,original_detection=True,marker_mask=encode_marker_mask(model['mask']),
                marker_offset_x=0.,marker_offset_y=0.,marker_scale=model['scale'],marker_aspect=1.,
                source_diameter=float(template.diameter),effective_diameter=diameter,
                ink_evidence=dict(version=version,direct_required_recall=scene['stage_a']['direct_rim_recall'],
                    visible_fit_quality=scene['stage_a']['visible_fit_quality'],
                    visible_coverage=scene['stage_a']['visible_coverage'],
                    confidence_kind='independent_visible_profile_fit_not_probability'))
            certificate=dict(scene,binding=dict(target=_point_binding(point),neighbors=[_point_binding(q) for q in neighbors],
                family_centres=family_audit['centres'],
                strict_active_context=_local_context(point,[*kept,*additions],family_audit['centres'],strict_only=all_shapes),
                **({'active_context_policy':'strict_native_roster_v1'} if all_shapes else {}),
                image_shape=list(image.shape),patch_box=[a,b,c,e],patch_sha256=_array_hash(image[b:e,a:c])))
            if scene['accepted']:
                point['partial_visibility_certificate']=certificate
            else:
                # Unresolved scenes are hypotheses, never accepted certificates.
                point['partial_visibility_hypothesis']=certificate
            heavy=all_shapes and scene['stage_a']['required_ink_visibility']<=.50
            if all_shapes and (heavy or not scene['accepted']):
                point.update(candidate_id=point_id,original_detection=False,status='suppressed_hypothesis',
                    state='suppressed',priority=0.,
                    admission_reason='partial_scene_heavily_occluded' if scene['accepted'] else 'partial_scene_identity_unresolved')
                point['required_ink_visibility']=scene['stage_a']['required_ink_visibility']
                if not any(q.get('swatch_id')==template.key and
                    math.hypot(q['cx']-x,q['cy']-y)<.65*diameter for q in report['suppressed']):
                    report['suppressed'].append(point)
            else:
                additions.append(point);report['accepted'].append(dict(diagnostic,point_id=point_id))
            if budget:break
        if budget:break
    report.update(status='candidate_or_time_budget' if budget else 'completed',added=len(additions),
                  tested_candidates=tested,profile_centres=profile_centres,
                  seconds=perf_counter()-started,budget_exhausted=budget)
    log_fn('[v46 partial visibility] '+report['status']+'; added='+str(len(additions))+
           '; tested='+str(tested))
    return additions,report

"""Compare same-series marker subsets on one fixed image/nuisance canvas.

Pruning only: no colour reassignment, new centres, per-candidate scale, or
expected measurement count. Small existing NMS stays as a cheap first pass.
All subset hypotheses share the same pixels, denominator and fitted lines.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
import math
from time import perf_counter

import cv2
import numpy as np

from color_marker_window_v2 import _connector, _crop, _enclosed_holes, WindowConfig

VERSION = 'same_series_joint_body_v1'


@dataclass(frozen=True)
class JointBodyConfig:
    neighbourhood_diameter: float = 1.05
    maximum_candidates: int = 10
    marker_cost: float = .08
    minimum_improvement: float = .025
    minimum_own_support: float = .10
    maximum_single_loss: float = .80
    boundary_weight: float = .10


def stamp(source, center, x, y, shape):
    return cv2.warpAffine(np.asarray(source, np.float32),
        np.float32([[1, 0, x-center[0]], [0, 1, y-center[1]]]),
        (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


def clusters(points, fraction):
    """Connected proximity groups, strictly within a single series identity."""
    remaining = set(range(len(points)))
    while remaining:
        group = {min(remaining)}
        stack = list(group)
        remaining -= group
        while stack:
            i = stack.pop(); a = points[i]
            found = [j for j in sorted(remaining) if points[j]['series_id'] == a['series_id'] and
                     math.hypot(points[j]['x']-a['x'], points[j]['y']-a['y']) <=
                     fraction*min(a['diameter'], points[j]['diameter'])]
            group.update(found); remaining.difference_update(found); stack.extend(found)
        yield [points[i] for i in sorted(group)]


def compare_group(evidence, index, points, *, config=None, debug=False):
    cfg = config or JointBodyConfig()
    t = evidence['templates'][index]
    report = dict(candidate_ids=[p['candidate_id'] for p in points], series_id=t['id'],
                  applied=False, reason='insufficient_candidates')
    if len(points) < 2:
        return report
    if len(points) > cfg.maximum_candidates:
        return dict(report, reason='bounded_search_preserve_large_cluster')
    a = np.asarray(t['soft'], np.float32).copy()
    center = t.get('center', [(a.shape[1]-1)/2, (a.shape[0]-1)/2])
    hole = np.asarray(t.get('hole_core', _enclosed_holes(a)), bool)
    uncertain = np.asarray(t.get('uncertain', np.zeros(a.shape, bool)), bool)
    rim = np.asarray(t.get('observed_rim', np.zeros(a.shape, bool)), bool)
    if hole.any():
        a[uncertain & ~rim] = 0
    face = np.asarray(t.get('face', (a > .15) | hole), np.float32)
    body_y, body_x = np.nonzero(face)
    if not len(body_x) or a.sum() < 3:
        return dict(report, reason='insufficient_template_body')
    d = float(t['diameter'])
    xy = np.asarray([[p['x'], p['y']] for p in points], float)
    middle = (xy.min(axis=0)+xy.max(axis=0))/2
    body_radius = max(d/2, float(np.hypot(body_x-center[0], body_y-center[1]).max()))
    inner = float(np.linalg.norm(xy-middle, axis=1).max())+body_radius+2
    outer = inner+max(8, 1.2*d)
    r = math.ceil(outer+2)
    shape = (2*r+1, 2*r+1)
    origin = middle-r
    own = np.asarray(evidence['membership'][index], np.float32)
    valid_full = np.asarray(evidence.get('valid', np.ones(own.shape)), np.float32)
    valid_full = valid_full*(1-np.asarray(evidence.get('ignore_mask', np.zeros(own.shape)), np.float32))
    sample = lambda field: np.clip(_crop(field, *middle, r), 0, 1)
    valid = sample(valid_full)
    observed = sample(own*valid_full)
    other = sample(evidence['other'][index]*valid_full)
    if 'paper' in evidence:
        paper = sample(evidence['paper'])*valid
    elif 'crop_bgr' in evidence:
        paper = sample(np.clip((np.min(evidence['crop_bgr'],axis=2).astype(float)-220)/30,0,1))*valid
    else:
        paper = (1-observed)*(1-other)*valid
    blend = sample(evidence['blend_uncertainty'][index]) if 'blend_uncertainty' in evidence else observed*0
    guide = sample(evidence['guide_upper'][index]*valid_full) if 'guide_upper' in evidence else observed*0
    maps = [stamp(a, center, *(p-origin), shape) for p in xy]
    faces = [stamp(face, center, *(p-origin), shape) for p in xy]
    all_faces = np.maximum.reduce(faces)
    # Same full-body domain for every subset, including inner paper of holes.
    domain = cv2.dilate((all_faces>.1).astype(np.uint8), np.ones((5,5),np.uint8)).astype(np.float32)*valid
    yy, xx = np.mgrid[-r:r+1,-r:r+1]
    radius = np.hypot(xx, yy)
    external = (radius>=inner) & (radius<=outer) & (valid>.95)
    line, fits, confidence = _connector(np.maximum(observed-guide,0), other,
        valid*(all_faces<.1), external, inner, outer, d, WindowConfig())
    nuisance = np.maximum(line, guide)
    mass = max(float(a.sum()), 1.)  # Never changes with marker count.
    missing_weight = np.maximum((1-other)*(1-.65*blend), 1.75*paper)
    grad = lambda x: cv2.morphologyEx(x,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    obs_boundary = grad(observed)
    tests = []
    model_cache = {}
    for n in range(1,len(points)+1):
        for ids in combinations(range(len(points)),n):
            marker = np.maximum.reduce([maps[i] for i in ids])
            models = [('through', np.maximum(nuisance,marker))]
            if hole.any():
                union_face = np.maximum.reduce([faces[i] for i in ids])
                models.append(('opaque_hollow_face',np.maximum(nuisance*(1-union_face),marker)))
            choices = []
            for composition, model in models:
                deficit = np.maximum(model-observed,0)*missing_weight
                extra = np.maximum(observed-model,0)
                pixel_loss = float(((deficit+extra)*domain).sum())/mass
                edge_loss = float((np.abs(grad(model)-obs_boundary)*domain*(1-other)).sum())/mass
                image_loss = pixel_loss+cfg.boundary_weight*edge_loss
                choices.append((image_loss,composition,model,pixel_loss,edge_loss))
            image_loss, composition, model, pixel_loss, edge_loss = min(choices,key=lambda z:z[0])
            row = dict(indices=list(ids),candidate_ids=[points[i]['candidate_id'] for i in ids],
                       count=n,image_loss=image_loss,pixel_loss=pixel_loss,boundary_loss=edge_loss,
                       objective=image_loss+cfg.marker_cost*n,composition=composition)
            tests.append(row)
            if debug: model_cache[tuple(ids)] = model
    winner = min(tests,key=lambda q:(q['objective'],q['count'],q['candidate_ids']))
    full = tests[-1]
    single = min((q for q in tests if q['count']==1),key=lambda q:q['objective'])
    gain = full['objective']-winner['objective']
    # An almost invisible group or a bad one-body model must not collapse merely
    # because we prefer fewer markers. Keep all original hypotheses on abstention.
    support = [float(np.minimum(np.maximum(observed-nuisance,0),m).sum())/mass for m in maps]
    selected_support = min(support[i] for i in winner['indices'])
    reliable = (selected_support >= cfg.minimum_own_support and
                winner['image_loss']/winner['count'] <= cfg.maximum_single_loss)
    applied = winner['count']<len(points) and gain>=cfg.minimum_improvement and reliable
    kept = winner['candidate_ids'] if applied else report['candidate_ids']
    report.update(applied=applied, reason='joint_body_simplification' if applied else
        'insufficient_visible_body_preserve' if not reliable else 'no_clear_simplification',
        kept_ids=kept, removed_ids=[p['candidate_id'] for p in points if p['candidate_id'] not in kept],
        best=winner, best_single=single, full=full, improvement=gain, own_support=support,
        crop_origin=origin.tolist(), diameter=d, marker_mass=mass, shared_domain_pixels=float(domain.sum()),
        line_confidence=confidence, line_count=len(fits), models_evaluated=len(tests),
        same_pixels_and_denominator=True, source_pixels_changed=False,
        top_models=sorted(tests,key=lambda q:q['objective'])[:10])
    if debug:
        report['maps'] = dict(observed=observed,other=other,paper=paper,domain=domain,nuisance=nuisance,
                            single=model_cache[tuple(single['indices'])],
                            best=model_cache[tuple(winner['indices'])],full=model_cache[tuple(full['indices'])])
    return report


def consolidate(evidence, selection, *, config=None):
    """Return a fresh selection; preserve all candidates and rejection reasons."""
    cfg=config or JointBodyConfig(); start=perf_counter()
    absent=[key for key in ('templates','membership','other') if key not in evidence]
    if absent:
        return dict(selection,joint_body_selection=dict(version=VERSION,configuration=asdict(cfg),
            groups=[],before_count=len(selection['points']),after_count=len(selection['points']),
            removed_count=0,seconds=perf_counter()-start,status='unavailable_preserved',missing_fields=absent))
    indices={str(t['id']):i for i,t in enumerate(evidence['templates'])}
    reports=[]; removed=set()
    for group in clusters(selection['points'],cfg.neighbourhood_diameter):
        if len(group)<2: continue
        report=compare_group(evidence,indices[str(group[0]['series_id'])],group,config=cfg)
        reports.append(report); removed.update(report.get('removed_ids',[]))
    result=dict(selection,points=[p for p in selection['points'] if p['candidate_id'] not in removed],
                rejected=list(selection['rejected']))
    for report in reports:
        for cid in report.get('removed_ids',[]):
            result['rejected'].append(dict(candidate_id=cid,reason='same_series_joint_body_redundant',
                                          representatives=report['kept_ids']))
    result['joint_body_selection']=dict(version=VERSION,configuration=asdict(cfg),groups=reports,
        before_count=len(selection['points']),after_count=len(result['points']),removed_count=len(removed),
        seconds=perf_counter()-start,scope='same-series only; pruning among verified centers',
        scale_refitted=False,ambiguous_large_clusters_preserved=True)
    return result

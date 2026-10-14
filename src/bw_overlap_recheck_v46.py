"""Bounded Step-5 replay of retained typed overlaps against confirmed owners.

No new centers, scales, shapes, or detection search are introduced. The native
gray legend is frozen; all local active ink is present in both hypotheses.
The existing explained-occlusion verifier supplies the unchanged profile,
missing-paper, donor holdout, null-improvement, and rival-identity gates.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

import bw_partial_visibility_v46 as pv
from bw_explained_occlusion_v46 import verify_scene

VERSION = 'bw-step5-retained-overlap-v1'


def _source(model):
    pack = model.get('overlap_source') or {}
    if pack.get('version') != 'saved-legend-gray-overlap-v1':
        return None
    ink = np.asarray(pack.get('ink'), np.float32)
    reliability = np.asarray(pack.get('reliability'), np.float32)
    if (ink.ndim != 2 or min(ink.shape) < 3 or max(ink.shape) > 257 or
            reliability.shape != ink.shape or not np.isfinite(ink).all() or
            not np.isfinite(reliability).all() or np.min(ink) < 0 or np.max(ink) > 1 or
            np.min(reliability) < 0 or np.max(reliability) > 1 or
            pack.get('class_name') != model.get('class_name')):
        return None
    return ink, reliability


def _geometry_supported(point, model):
    return (abs(float(point.get('marker_scale', 1.))-float(model.get('marker_scale', 1.))) < 1.e-6 and
            abs(float(point.get('marker_aspect', 1.))-1.) < 1.e-6 and
            point.get('class_name') == model.get('class_name') and
            abs(float(point.get('source_diameter', 0.))-float(model.get('source_diameter', 0.))) < 1.e-6)


def _bounded_scene_box(image_shape, requested, footprints, maximum_side):
    """Trim surplus context, never a target/rival/donor source footprint.

    Eight pixels retain the native transition samples and interpolation halo.
    No image resizing or change to the visibility and identity gates occurs.
    A scene whose full measured rasters cannot fit remains ineligible.
    """
    height, width = image_shape[:2]
    halo = 8
    required = [max(0, math.floor(min(x-(shape[1]-1)/2. for x,y,shape in footprints))-halo),
                max(0, math.floor(min(y-(shape[0]-1)/2. for x,y,shape in footprints))-halo),
                min(width, math.ceil(max(x+(shape[1]-1)/2. for x,y,shape in footprints))+halo+1),
                min(height, math.ceil(max(y+(shape[0]-1)/2. for x,y,shape in footprints))+halo+1)]
    if max(required[2]-required[0], required[3]-required[1]) > maximum_side:
        return None
    box = [min(requested[0],required[0]),min(requested[1],required[1]),
           max(requested[2],required[2]),max(requested[3],required[3])]
    for start, end in ((0, 2), (1, 3)):
        if box[end]-box[start] > maximum_side:
            lo = max(box[start], required[end]-maximum_side)
            hi = min(required[start], box[end]-maximum_side)
            origin = int(round((box[start]+box[end]-maximum_side)/2.))
            box[start] = min(max(origin, lo), hi)
            box[end] = box[start]+maximum_side
    return box


def evaluate(image, point, active, models, plot, legend=None):
    """Return freshly measured local authority, never trust a saved pass flag.

    Contextual rescues cannot act as foreground donors: no recursive chain of
    inferred markers. An ordinary Step-5 confirmed point can act as a donor,
    but must still pass the independent source test outside the target body.
    """
    failed = dict(version=VERSION, valid=False, score=0., neighbor_ids=[],
                  identity_margin=0., reason='overlap_source_unavailable', reviews=[])
    sid = point.get('swatch_id')
    target_model = models.get(sid, {})
    target_source = _source(target_model)
    if target_source is None or not _geometry_supported(point, target_model):
        return failed
    x,y = float(point['cx']),float(point['cy'])
    diameter = float(target_model.get('effective_diameter', max(target_source[0].shape)))
    cfg = pv.Config()
    if diameter > cfg.maximum_diameter:
        return dict(failed, reason='overlap_local_geometry_budget')
    sources = {key:_source(model) for key,model in models.items()}
    # All identities must have source pixels. Otherwise a winning shape would
    # only mean an unavailable rival was silently omitted.
    if any(source is None for source in sources.values()):
        return dict(failed, reason='overlap_rival_source_unavailable')
    nearby = [q for q in active if q.get('point_id') != point.get('point_id') and
        math.hypot(q['cx']-x,q['cy']-y) < .90*(diameter+
            float(models.get(q['swatch_id'],{}).get('effective_diameter',diameter)))/2.]
    nearby.sort(key=lambda q:(math.hypot(q['cx']-x,q['cy']-y),str(q.get('point_id'))))
    donors = [q for q in nearby if q.get('swatch_id') in sources and
        not any(key in q for key in ('pending_identity','partial_visibility_certificate',
            'partial_visibility_hypothesis','overlap_hypothesis','overlap_recheck','seed_bootstrap')) and
        _geometry_supported(q,models[q['swatch_id']])]
    if not donors:
        return dict(failed, reason='no_independent_confirmed_foreground')
    reviews=[]
    for donor in donors[:cfg.maximum_neighbours]:
        donor_source = sources[donor['swatch_id']][0]
        radius = int(math.ceil(1.55*max(diameter,max(donor_source.shape))))
        left=max(0,int(min(x,donor['cx'])-radius))
        top=max(0,int(min(y,donor['cy'])-radius))
        right=min(image.shape[1],int(max(x,donor['cx'])+radius+1))
        bottom=min(image.shape[0],int(max(y,donor['cy'])+radius+1))
        requested=[left,top,right,bottom]
        footprints=[(x,y,source[0].shape) for source in sources.values()]
        footprints.append((donor['cx'],donor['cy'],donor_source.shape))
        box=_bounded_scene_box(image.shape,requested,footprints,cfg.maximum_crop_side)
        if box is None:
            reviews.append(dict(donor_id=donor['point_id'],accepted=False,
                reason='overlap_source_footprints_exceed_window_budget',
                requested_patch_box=requested,maximum_crop_side=cfg.maximum_crop_side))
            continue
        left,top,right,bottom=box
        shape=(bottom-top,right-left);origin=(left,top)
        observed=(255-cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2GRAY).astype(np.float32))/255.
        own=pv._stamp(target_source[0],x,y,origin,shape)
        front=pv._stamp(donor_source,donor['cx'],donor['cy'],origin,shape)
        if not np.any(pv._face(own)&pv._face(front)):
            continue
        reliability=1-pv._stamp(1-target_source[1],x,y,origin,shape)
        rivals={key:pv._stamp(source[0],x,y,origin,shape)
                for key,source in sources.items() if key!=sid}
        yy,xx=np.indices(shape);xx+=left;yy+=top
        valid=(xx>=plot[0])&(xx<plot[2])&(yy>=plot[1])&(yy<plot[3])
        if legend:
            valid &= ~((xx>=legend[0])&(xx<legend[2])&(yy>=legend[1])&(yy<legend[3]))
        context=[];unavailable=False
        for q in active:
            if q.get('point_id') in (point.get('point_id'),donor.get('point_id')):
                continue
            model=models.get(q.get('swatch_id'),{})
            size=float(model.get('effective_diameter',diameter))
            if not (left-size<=q['cx']<=right+size and top-size<=q['cy']<=bottom+size):
                continue
            if q.get('swatch_id') not in sources or not _geometry_supported(q,model):
                unavailable=True;break
            context.append(dict(id=str(q['point_id']),marker=pv._stamp(
                sources[q['swatch_id']][0],q['cx'],q['cy'],origin,shape)))
        if unavailable:
            reviews.append(dict(donor_id=donor['point_id'],accepted=False,
                                reason='active_source_geometry_unavailable'))
            continue
        scene=verify_scene(observed,own,front,rivals,available=valid,
            source_reliability=reliability,config=cfg,active_context=context,
            target_class=point['class_name'])
        reviews.append(dict(donor_id=donor['point_id'],patch_box=box,
            requested_patch_box=requested,context_padding_trimmed=box!=requested,
            source_footprints_preserved=True,**scene))
        if scene['accepted']:
            return dict(version=VERSION,valid=True,explained_occlusion_validated=True,
                reason='retained_hypothesis_source_pair_verified',score=scene['score'],
                identity_margin=scene['stage_b']['identity_margin'],
                scene_improvement=scene['stage_b']['marker_improvement'],
                minimum_scene_improvement=cfg.minimum_improvement,
                neighbor_ids=[str(donor['point_id'])],explained_neighbor_ids=[str(donor['point_id'])],
                target_center=[x,y],donor_center=[float(donor['cx']),float(donor['cy'])],
                target_class=point['class_name'],source_pixels_unchanged=True,
                reference_points_read=False,reviews=reviews)
    return dict(failed,reason=reviews[-1]['reason'] if reviews else 'no_intersecting_foreground',reviews=reviews)

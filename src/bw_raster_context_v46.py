"""Context checks for an observed marker plus centered-line hypothesis.

Local marker similarity alone can explain a letter fragment. Lines used to
explain the body must continue on at least one side outside it. Extra nearby
ink is measured symmetrically, without calling it missing marker ink.
"""
import cv2
import numpy as np
from bw_centered_composite_v46 import stroke
from bw_layered_composite_v46 import compose


def external_strokes(patch, body, *, centered_only=False):
    """Explain extra ink only with thin strokes measured OUTSIDE the body.

    This mask can reduce a nuisance cost, never increase marker recall.
    Width and strength are fitted on the exterior so a filled glyph cannot
    explain away its own interior as an error bar.
    """
    h,w=patch.shape;yy,xx=np.indices(patch.shape,dtype=float)
    outer=cv2.dilate(body.astype(np.uint8),np.ones((3,3),np.uint8))==0
    ys,xs=np.nonzero(body);diameter=max(np.ptp(xs)+1,np.ptp(ys)+1) if len(xs) else 3
    # Suppress the held-out body even during proposal generation.
    binary=np.uint8((patch>.25)&outer)*255
    lines=cv2.HoughLinesP(binary,1,np.pi/180,threshold=max(4,round(.4*diameter)),
        minLineLength=max(4.,.5*diameter),maxLineGap=1)
    cover=np.zeros_like(patch);records=[]
    if lines is None:return cover,records
    for a,b,c,d in lines.reshape(-1,4):
        vx,vy=float(c-a),float(d-b);length=np.hypot(vx,vy)
        if length<1:continue
        vx/=length;vy/=length;normal=(-vy,vx)
        perp=(xx-a)*normal[0]+(yy-b)*normal[1]
        if centered_only:
            center_distance=abs(((w-1)/2-a)*normal[0]+((h-1)/2-b)*normal[1])
            if center_distance>1.5:continue
            if records and not (abs(normal[0])>.94 or abs(records[0]['normal'][0])>.94):continue
        core=outer&(abs(perp)<.8)
        side=outer&(abs(perp)>2.5)&(abs(perp)<3.5)
        if core.sum()<4 or side.sum()<3:continue
        strength=float(np.median(patch[core]));contrast=strength-float(np.median(patch[side]))
        if strength<.25 or contrast<.15:continue
        offset=a*normal[0]+b*normal[1]
        if any(abs(np.dot(normal,q['normal']))>.97 and abs(abs(offset)-abs(q['offset']))<1.5 for q in records):continue
        profile=[];positions=np.arange(-2.5,2.51,.5)
        for delta in positions:
            band=outer&(abs(perp-delta)<.4)
            profile.append(float(np.median(patch[band])) if band.sum()>=3 else 0.)
        predicted=np.interp(perp,positions,profile,left=0,right=0)
        cover=np.maximum(cover,predicted)
        records.append(dict(normal=list(normal),offset=float(offset),strength=strength))
        if len(records)==(2 if centered_only else 4):break
    return cover,records


def evidence(ink, center, marker, cover, parameters, reliability=None):
    r=len(marker)//2
    patch=cv2.getRectSubPix(ink,(len(marker),len(marker)),tuple(map(float,center)))
    body=np.maximum(marker,cover)>.08
    yy0,xx0=np.nonzero(body)
    if len(xx0)>=3:
        envelope=np.zeros_like(body,np.uint8)
        cv2.fillConvexPoly(envelope,cv2.convexHull(np.column_stack((xx0,yy0)).astype(np.int32)),1)
        body=envelope>0  # Nuisance exclusion only, never positive marker ink.
    yy,xx=np.indices(marker.shape)
    lines=[p['amplitude']*stroke(len(marker),(r,r),p['angle'],p['width'])
           for p in parameters['lines']]
    scaled=np.clip(marker*parameters['gain'],0,1)
    model=compose(scaled,np.maximum(cover,scaled),lines,
                  [p['order'] for p in parameters['lines']])
    extensions=[]
    # The body band is excluded: marker ink cannot validate its own nuisance.
    outside=cv2.dilate(body.astype(np.uint8),np.ones((3,3),np.uint8))==0
    for p,line in zip(parameters['lines'],lines):
        angle=np.deg2rad(p['angle']);along=(xx-r)*np.cos(angle)+(yy-r)*np.sin(angle)
        ray_scores=[]
        for sign in (-1,1):
            w=line*outside*(sign*along>0)
            if w.sum()<.1:continue
            support=np.clip(patch/np.maximum(line,.05),0,1)
            ray_scores.append(float((w*support).sum()/w.sum()))
        extensions.append(max(ray_scores,default=0.))
    # Require precision as well as recall, on the identical observed canvas.
    # Small antialiasing/model errors are not additional objects.
    nuisance,measured=external_strokes(patch,body,centered_only=True)
    extra=np.maximum(patch-np.maximum(model,nuisance)-.15,0.)
    weight=np.ones_like(patch) if reliability is None else np.asarray(reliability,np.float32)
    extra_fraction=float(np.sum(weight*extra**2)/max(np.sum(weight*patch**2),1.e-6))
    return dict(version='observed-raster-context-v1',line_extensions=extensions,
                minimum_line_extension=min(extensions,default=1.),
                external_strokes=measured,
                extra_ink_fraction=extra_fraction,
                passed=bool(min(extensions,default=1.)>=.45 and extra_fraction<=.20))


def native_identity_supported(record, expected_keys=None):
    """Preserve a full-window winner, not merely a high ink recall.

An explicit same-observation rival comparison and line-null comparison must
both support this identity. A weak grid vote or high confidence is not enough.
"""
    g=record.get('geometry_first',{})
    rivals=g.get('model_losses',{})
    # A specialized hollow-only comparison is not an identity verdict
    # against a triangle/X/filled rival that was never evaluated.
    if expected_keys is not None and not set(expected_keys).issubset(rivals):
        return False
    own=rivals.get(record.get('template'))
    if record.get('decision')!='verified' or g.get('decision')!='compatible' or own is None:
        return False
    if g.get('native_context',{}).get('passed') is False:
        return False
    other=[v for k,v in rivals.items() if k!=record['template']]
    # Scene fallback uses an asymmetric 0.4-weighted surplus-ink loss;
    # its improvement is not on the original verifier's MSE scale.
    minimum_improvement=.012 if g.get('version')=='small-hollow-scene-fallback-v1' else .06
    return bool(other and min(other)>own and own<=.15 and
                g.get('marker_improvement',0.)>=minimum_improvement and
                g.get('missing_rim_fraction',1.)<=.25)


def native_body_supported(records, key, center, diameter):
    """Reuse a verified larger-window body, not its old identity ranking.

    This only supplies the physical-body/context check. The caller must still
    require the complete observed-raster competition to select the same key.
    A rejected or merely high-recall candidate cannot bypass context checks.
    An overlap-abstained hollow can provide body evidence only when the full
    window still prefers that same hollow, explains its rim, and beats the
    line-only alternative. Final identity still requires ALL raster rivals.
    """
    x,y=center
    for p in records:
        g=p.get('geometry_first',{})
        if p.get('template')!=key:continue
        verified=(p.get('decision')=='verified' and g.get('decision')=='compatible'
                  and g.get('geometry_decision')=='verified')
        losses=g.get('model_losses',{});own=losses.get(key)
        supported_overlap=(p.get('decision')=='ambiguous' and g.get('decision')=='abstain'
            and g.get('reason')=='small_hollow_overlap_or_model_residual'
            and own is not None and own<=.15 and own==min(losses.values())
            and g.get('missing_rim_fraction',1.)<=.25 and g.get('marker_improvement',0.)>=.06)
        if not (verified or supported_overlap):continue
        if np.hypot(p.get('aligned_x',p['x'])-x,p.get('aligned_y',p['y'])-y)>max(1.,.15*diameter):
            continue
        if p.get('ink_evidence',{}).get('direct_required_recall',0.)<.80:continue
        return dict(candidate_index=p.get('candidate_index'),center=[p.get('aligned_x',p['x']),p.get('aligned_y',p['y'])],
                    source='verified_native_full_window_body' if verified else 'independently_supported_hollow_overlap',
                    identity_supplied_by='complete_raster_competition')
    return None

"""Automatic shared scale from source-supported circular inner/outer boundaries.

Gray correlation proposes sites only. Both circle and square boundaries compete
over the SAME scale/translation grid; only independently circular sites vote.
Final anchor windows still need the unmodified ink/hole verification thresholds.
No saved detections, hand-picked coordinates, or per-marker scales are used.
"""
import cv2
import numpy as np
from bw_circle_boundary_v46 import VERSION, geometry, boundary_profiles, profile_identity
from bw_hollow_boundary_v46 import source_darkness
from bw_shared_scale_v46 import gray_match_map
from partial_swatch_detector import ink_membership

SCALES=tuple(round(.5+.025*i,3) for i in range(33))


def calibrate_circle_scale(image,template,plot,ignore_regions=(),log_fn=lambda *a:None,window_check=None):
    from occlusion_aware_window_verifier import verify_marker_window, _uncertain_shape
    x0,y0,x1,y1=plot;t=template
    crop=image[y0:y1,x0:x1];dark=source_darkness(crop)
    gradients=(cv2.Sobel(dark,cv2.CV_32F,1,0,ksize=3,scale=1/8),
               cv2.Sobel(dark,cv2.CV_32F,0,1,ksize=3,scale=1/8))
    allowed=np.ones(dark.shape,np.float32)
    for a,b,c,d in ignore_regions:
        l,r=max(0,a-x0),min(x1,c)-x0;u,v=max(0,b-y0),min(y1,d)-y0
        if l<r and u<v:allowed[u:v,l:r]=0
    observed=cv2.GaussianBlur(ink_membership(crop,t.ink).astype(np.float32),(3,3),.55)
    best=np.full(dark.shape,-1.,np.float32);geometries=[]
    for scale in SCALES:
        np.maximum(best,gray_match_map(observed,t,scale,allowed>0),out=best)
        mask=_uncertain_shape(t,scale,1.)[0];dims=geometry(mask)
        if dims:
            center,outer,inner=dims
            dims=((center[0]-mask.shape[1]//2,center[1]-mask.shape[0]//2),outer,inner)
        geometries.append(dims)
    maxima=cv2.dilate(best,np.ones((5,5),np.uint8))
    yy,xx=np.nonzero((best>=.35)&(best>=maxima-1e-6))
    seeds=[]
    for x,y in sorted(zip(xx,yy),key=lambda xy:-float(best[xy[1],xy[0]])):
        if any(np.hypot(x-a,y-b)<max(3.,.15*t.diameter) for a,b in seeds):continue
        seeds.append((int(x),int(y)))
        if len(seeds)>=80:break
    candidates=[]
    for x,y in seeds:
        evidence=[];profiles=[];square_profiles=[];diamond_profiles=[];centers=[]
        for dims in geometries:
            if dims is None:
                profiles.append(0.);square_profiles.append(0.);diamond_profiles.append(0.);centers.append((x,y));continue
            offset,outer,inner=dims;local=[]
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    cx,cy=x+dx+offset[0],y+dy+offset[1]
                    e=boundary_profiles(dark,(cx,cy),outer,inner,available=allowed,gradients=gradients)
                    e['proposal_center']=[x+dx+x0,y+dy+y0]
                    local.append(e)
            e=max(local,key=lambda e:e['circle_score'])
            profiles.append(e['circle_score']);square_profiles.append(max(p['square_score'] for p in local))
            diamond_profiles.append(max(p['diamond_score'] for p in local))
            centers.append(e['proposal_center']);evidence.extend(local)
        identity=profile_identity(evidence,getattr(t,'circle_rival_shapes',None))
        q=dict(x=x+x0,y=y+y0,quality=float(best[y,x]),profile=profiles,
               raw_circle_profile=list(profiles),
               square_profile=square_profiles,diamond_profile=diamond_profiles,profile_scales=list(SCALES),
               independent_anchor_evidence=identity,accepted=False)
        if identity['accepted']:
            # A shape score alone cannot authorize a scale. Re-use final source
            # window verification and keep the original core/hole requirements.
            trials=[];valid_profile=np.full(len(SCALES),-1.)
            for j in np.argsort(profiles)[::-1]:
                if profiles[j]<max(.35,max(profiles)-.06):continue
                cx,cy=centers[j]
                v=verify_marker_window(image,t,cx,cy,search_scales=(SCALES[j],),lock_aspect=True)
                if window_check is not None:window_check(t,v)
                offset,outer,inner=geometries[j]
                aligned=boundary_profiles(dark,(v.aligned_x-x0+offset[0],v.aligned_y-y0+offset[1]),
                                          outer,inner,available=allowed,gradients=gradients)
                circular=(aligned['circle_score']>=.35 and aligned['supported_fraction']>=.55
                    and sum(s>=.4 for s in aligned['sector_support'])>=6
                    and aligned['circle_score']-identity['rival_score']>=.055)
                trials.append(dict(scale=SCALES[j],x=v.aligned_x,y=v.aligned_y,
                    decision=v.decision,strict_core_recall=float(v.strict_core_recall),
                    boundary_score=aligned['circle_score'],aligned_arc_supported=bool(circular)))
                if v.decision=='verified' and circular:valid_profile[j]=aligned['circle_score']
            q['window_trials']=trials
            if np.any(valid_profile>=0):
                j=int(np.argmax(valid_profile));q.update(accepted=True,profile=valid_profile.tolist(),
                    best_scale=SCALES[j],anchor_boundary_score=float(valid_profile[j]))
            else:q['independent_anchor_evidence']['reason']='circle_identity_supported_but_no_verified_scale'
        candidates.append(q)
    anchors=[]
    for q in sorted(candidates,key=lambda q:-q.get('anchor_boundary_score',-1)):
        if not q['accepted']:continue
        if any(abs(q['x']-a['x'])<.6*t.diameter for a in anchors):continue
        anchors.append(q)
        if len(anchors)>=5:break
    if anchors:
        curves=np.array([a['profile'] for a in anchors])
        common=np.all(curves>=0,axis=0)
        # Do not average incompatible anchors into a size none of them verifies.
        if not common.any():
            votes=(curves>=0).sum(axis=0)
            mean=np.maximum(curves,0).sum(axis=0)/np.maximum(votes,1)
            j=max(range(len(SCALES)),key=lambda j:(int(votes[j]),float(mean[j])))
            anchors=[a for a in anchors if a['profile'][j]>=0]
            curves=np.array([a['profile'] for a in anchors]);common=np.all(curves>=0,axis=0)
            status=('largest_verified_consensus' if len(anchors)>=2 else 'weak_consensus_single_verified_anchor')
        else:status='multi_site_consensus' if len(anchors)>=3 else 'weak_consensus'
        loss=(curves.max(axis=1,keepdims=True)-curves).mean(axis=0);loss[~common]=2.
        index=int(np.argmin(loss));scale=SCALES[index]
        near=[s for s,v,ok in zip(SCALES,loss,common) if ok and v<=loss[index]+.015]
    else:
        loss=np.zeros(len(SCALES));scale=1.;near=[];status='no_reliable_circular_anchor_fallback_1x'
    report=dict(version=VERSION,scale=scale,status=status,shape=t.shape_hint,anchors=anchors,candidates=candidates,
        scales=list(SCALES),mean_regret=loss.tolist(),near_optimal_scales=near,
        criterion='paired circular boundary regret on verified scales; cross-scale square rival',
        identity_policy='source circular arcs before common scale; gray NCC is proposal only',
        manual_coordinates_used=False,saved_detections_used=False,per_marker_scales=False,
        proposal_threshold=.35,max_sites=80)
    log_fn(f'[v46 circle scale] {t.key}: scale={scale:.3f}, anchors={len(anchors)}, status={status}')
    return report

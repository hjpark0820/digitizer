"""Opt-in, source-raster competition on common small cells.

No symbol-name precedence: only differences between observed legend bodies
vote for identity. Shared ink abstains. Thin crossing strokes cannot provide
positive identity evidence. This pilot handles compact filled bodies only.
"""
from __future__ import annotations
from collections import Counter
import math
import cv2
import numpy as np

VERSION = 'pooled-body-raster-identity-v2'


def pooled_pair_evidence(ma, mb, positive, negative, valid, ink, gx, gy, quadrant, a, b):
    """Each physical difference pixel votes once, even in a tiny grid fragment.

    Small cells are diagnostic partitions, not independent repetitions. Require
    enough total evidence spread over the body, not four pixels in EVERY cell.
    Nuisance/shared/antialiased pixels abstain rather than pretending to be ink.
    """
    common=(ma>.65)&(mb>.65)&valid
    recall=float(ink[common].mean()) if common.any() else 0.
    ua=(ma>.65)&(mb<.35)&valid;ub=(mb>.65)&(ma<.35)&valid
    difference=ua|ub
    value=(ua.astype(float)-ub.astype(float))*(positive.astype(float)-1.5*negative.astype(float))
    cells=[]
    for tx,ty in sorted(set(zip(gx[difference].tolist(),gy[difference].tolist()))):
        sel=difference&(gx==tx)&(gy==ty);n=int(sel.sum())
        score=float(value[sel].sum()/n)
        cells.append(dict(cell=[int(tx),int(ty)],pixels=n,score=score,
                          winner=a if score>=.25 else b if score<=-.25 else None))
    margin=float(value[difference].mean()) if difference.any() else 0.
    winner=a if margin>=.18 else b if margin<=-.18 else None
    signed=value if winner==a else -value
    # 1.5 is one strongly contradictory white pixel, or >1 positive pixels.
    # At least two separated quadrants are still mandatory.
    sectors=sum(float(signed[difference&(quadrant==q)].sum())>=1.5 for q in range(4)) if winner else 0
    informative=int((difference&(positive|negative)).sum())
    decisive=bool(winner and recall>=.72 and difference.sum()>=6 and informative>=3 and sectors>=2)
    return dict(a=a,b=b,margin=margin,common_recall=recall,winner=winner if decisive else None,
                cells=cells,supporting_quadrants=int(sectors),difference_pixels=int(difference.sum()),
                informative_pixels=informative,aggregation='unique physical pixels across all cells')


def body_boundary_evidence(observed, expected, kernel_size):
    """Symmetric outline comparison ONLY on an independently isolated body.

    Apply the same compact-body opening to both masks so its rounding of corners
    cannot systematically reward circles. No raw surrounding ink is penalized.
    """
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    model=cv2.morphologyEx((expected>.5).astype(np.uint8),cv2.MORPH_OPEN,kernel).astype(bool)
    observed=np.asarray(observed,bool)
    def edge(mask):
        return mask&~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),
                              borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    oe,me=edge(observed),edge(model)
    if min(oe.sum(),me.sum())<6:return dict(status='insufficient_outline',score=None)
    od=cv2.distanceTransform((~oe).astype(np.uint8),cv2.DIST_L2,5)
    md=cv2.distanceTransform((~me).astype(np.uint8),cv2.DIST_L2,5)
    forward=float(od[me].mean());reverse=float(md[oe].mean())
    distance=.5*(forward+reverse)
    diameter=max(np.ptp(np.nonzero(observed)[0])+1,np.ptp(np.nonzero(observed)[1])+1)
    # One fixed pixel-aware tolerance, shared by all competing shapes.
    score=float(np.exp(-distance/max(.75,.075*diameter)))
    return dict(status='isolated_body',score=score,model_to_body_px=forward,
                body_to_model_px=reverse,symmetric_distance_px=distance,
                iou=float((model&observed).sum()/max(1,(model|observed).sum())),
                surrounding_ink_penalized=False)


def body_model(template):
    if template.marker_kind != 'filled':
        return None
    soft=np.asarray(template.raw_soft,np.float32)
    binary=(soft>=.5).astype(np.uint8)
    distance=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    depth=max(1.5,.24*float(distance.max()))
    n,labels,stats,_=cv2.connectedComponentsWithStats((distance>=depth).astype(np.uint8))
    if n<2:return None
    k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    ys,xs=np.where(labels==k);margin=max(1,int(round(depth)))
    a,b=max(0,xs.min()-margin),max(0,ys.min()-margin)
    c,d=min(soft.shape[1],xs.max()+margin+1),min(soft.shape[0],ys.max()+margin+1)
    crop=soft[b:d,a:c].copy()
    if min(crop.shape)<7 or not .70<=crop.shape[1]/crop.shape[0]<=1.43:return None
    if np.mean(crop>.5)<.50:return None
    offset=((a+c-1)/2-(soft.shape[1]-1)/2,(b+d-1)/2-(soft.shape[0]-1)/2)
    return crop,offset


def body_pixels(template):
    model=body_model(template)
    return None if model is None else model[0]


class GridIdentityCompetition:
    def __init__(self,image,templates,plot_area,ignore_regions=(),observed_scale_range=None,
                 silhouette_models=False,body_opening_fraction=.16,occlusion_mask=None,
                 boundary_priority=False,colour_observation=None):
        from bw_colour_visibility_v46 import validate_mask
        self.other = validate_mask(occlusion_mask, image.shape[:2])
        self.boundary_priority=bool(boundary_priority)
        if observed_scale_range is None:
            observed_scale_range=(.50,1.30) if self.boundary_priority else (.70,1.25)
        self.plot=tuple(plot_area)
        models={t.key:b for t in templates if (b:=body_model(t)) is not None}
        if silhouette_models:
            # These have already been decomposed into pure geometry by the
            # geometry-first controller. Re-extracting only their deep core
            # truncates triangle tips and can exclude triangles altogether.
            for t in templates:
                if not getattr(t,'model_completed',False) or t.marker_kind!='filled':continue
                soft=np.asarray(t.raw_soft,np.float32);yy,xx=np.nonzero(soft>.5)
                if len(xx)<6:continue
                a,b,c,d=xx.min(),yy.min(),xx.max()+1,yy.max()+1
                crop=soft[b:d,a:c]
                if min(crop.shape)<7:continue
                offset=((a+c-1)/2-(soft.shape[1]-1)/2,(b+d-1)/2-(soft.shape[0]-1)/2)
                models[t.key]=(crop,offset)
        self.bodies={k:b[0] for k,b in models.items()}
        self.body_offsets={k:b[1] for k,b in models.items()}
        sizes=[max(b.shape) for b in self.bodies.values()]
        self.step=max(3,int(round(.5*np.median(sizes)))) if sizes else 3
        self.diameter=float(np.median(sizes)) if sizes else 6.
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.source_ink=np.clip((255.-gray.astype(np.float32))/191.,0.,1.)
        ink=(gray<180).astype(np.uint8)
        if colour_observation is not None:
            colour_observation.validate(image)
            self.source_ink*=colour_observation.own
            ink &= (colour_observation.own>=.5).astype(np.uint8)
        valid=np.zeros_like(ink);x0,y0,x1,y1=self.plot;valid[y0:y1,x0:x1]=1
        for a,b,c,d in ignore_regions:valid[b:d,a:c]=0
        if self.other is not None:
            valid[self.other >= .5] = 0
        if colour_observation is not None:
            # Low target membership is not itself proof of white paper.
            valid &= ((colour_observation.own>=.5)|(colour_observation.paper>=.5)).astype(np.uint8)
        ink*=valid
        inside=cv2.distanceTransform(ink,cv2.DIST_L2,5)
        outside=cv2.distanceTransform(1-ink,cv2.DIST_L2,5)
        # Remove only long straight strokes from POSITIVE identity evidence.
        # Source image/marker pixels used by the detector are never changed.
        length=max(9,int(math.ceil(1.6*self.diameter)))|1
        nuisance=np.zeros_like(ink)
        kernels=[np.ones((1,length),np.uint8),np.ones((length,1),np.uint8),
                 np.eye(length,dtype=np.uint8),np.fliplr(np.eye(length,dtype=np.uint8))]
        for kernel in kernels:nuisance|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
        self.positive=(inside>=1.4)&~nuisance.astype(bool)&valid.astype(bool)
        self.negative=(outside>1.)&valid.astype(bool)
        if colour_observation is not None:self.negative &= colour_observation.paper>=.5
        self.ink=ink.astype(bool);self.valid=valid.astype(bool)
        self.nuisance=nuisance.astype(bool)
        # A shared, measured body extent prevents an oversized square from
        # losing to an inscribed circle merely because its corners fall outside
        # the real marker. This calibration never supplies detection centers:
        # it only aligns the two competing raster explanations to the SAME
        # compact component. Noncompact/overlapped bodies cause abstention.
        kernel_size=max(3,int(round(body_opening_fraction*self.diameter)))|1
        self.body_kernel_size=kernel_size
        compact=cv2.morphologyEx(ink,cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size)))
        _,self.body_labels,stats,centers=cv2.connectedComponentsWithStats(compact)
        self.observed_bodies={}
        for label,(a,b,w,h,area) in enumerate(stats[1:],1):
            scale=max(w,h)/self.diameter
            if observed_scale_range[0]<=scale<=observed_scale_range[1] and .70<=w/h<=1.43 and .48<=area/(w*h)<=1:
                self.observed_bodies[label]=dict(center=[a+(w-1)/2,b+(h-1)/2],
                    scale=float(scale),box=[int(a),int(b),int(a+w),int(b+h)])
        self.cache={};self.bank_cache={};self.counts=Counter();self.candidate_rows=[]

    def bank(self,scale,offset=(0.,0.)):
        key=(round(float(scale),5),tuple(offset))
        if key in self.bank_cache:return self.bank_cache[key]
        # All identities explain the SAME measured body extent. Applying one
        # multiplier to unequal legend-crop sizes made the smaller triangle
        # permanently inscribed in the larger circle/square hypotheses.
        # This comparison-only normalization does not change the detector's
        # shared symbol scale or any proposed/verified marker coordinates.
        resized={}
        for k,b in self.bodies.items():
            yy,xx=np.nonzero(b>.5)
            crop=b[yy.min():yy.max()+1,xx.min():xx.max()+1]
            factor=self.diameter*scale/max(crop.shape)
            resized[k]=cv2.resize(crop,(max(3,round(crop.shape[1]*factor)),
                max(3,round(crop.shape[0]*factor))),interpolation=cv2.INTER_LINEAR)
        size=(max((max(b.shape) for b in resized.values()),default=3)+6)|1
        masks={}
        for k,b in resized.items():
            a=np.zeros((size,size),np.float32);h,w=b.shape
            if self.boundary_priority:
                # All models share the measured body centre, including its
                # half-pixel phase. Integer insertion shifts even-sized bodies.
                transform=np.float32([[1,0,(size-w)/2+offset[0]],
                                      [0,1,(size-h)/2+offset[1]]])
                a=cv2.warpAffine(b,transform,(size,size),flags=cv2.INTER_LINEAR)
            else:
                a[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=b
            masks[k]=a
        self.bank_cache[key]=(size,masks)
        return size,masks

    def evaluate(self,x,y,scale=1.,detail=False):
        proposed=[float(x),float(y)]
        ix,iy=int(round(x)),int(round(y))
        label=int(self.body_labels[iy,ix]) if 0<=iy<self.body_labels.shape[0] and 0<=ix<self.body_labels.shape[1] else 0
        body=self.observed_bodies.get(label)
        if body is None or np.linalg.norm(np.asarray(body['center'])-proposed)>.30*self.diameter:
            return dict(winner=None,pairs=[],cell_votes={},reason='no_isolated_compact_body')
        x,y=body['center'];scale=body['scale']
        cx,cy=int(round(x)),int(round(y));key=(cx,cy,round(float(scale),5))
        if key in self.cache and not detail:return self.cache[key]
        size,masks=self.bank(scale,(x-cx,y-cy));half=size//2;left,top=cx-half,cy-half
        if len(masks)<2 or left<0 or top<0 or left+size>self.ink.shape[1] or top+size>self.ink.shape[0]:
            return dict(winner=None,pairs=[],cell_votes={})
        cut=np.s_[top:top+size,left:left+size]
        if self.other is not None and any(np.any((v>.5)&(self.other[cut]>=.5)) for v in masks.values()):
            # A clipped body cannot calibrate or discriminate complete shapes.
            # Preserve rival hypotheses instead of treating the cut as paper.
            return dict(winner=None,pairs=[],cell_votes={},reason='other_colour_occludes_identity_body')
        pos,neg,valid=self.positive[cut],self.negative[cut],self.valid[cut]
        yy,xx=np.indices((size,size));gx=(xx+left-self.plot[0])//self.step;gy=(yy+top-self.plot[1])//self.step
        quadrant=(xx>=half).astype(int)+2*(yy>=half).astype(int)
        ids=list(masks);wins={k:[] for k in ids};pairrows=[];votes={}
        observed=self.body_labels[cut]==label
        boundary={k:body_boundary_evidence(observed,v,self.body_kernel_size) for k,v in masks.items()}
        normal={}
        if self.boundary_priority:
            from bw_boundary_evidence_v46 import transition_evidence,compare_transitions
            normal={k:transition_evidence(self.source_ink[cut],v,valid=valid,
                        nuisance=self.nuisance[cut]) for k,v in masks.items()}
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                size_ratio=max(self.bodies[a].shape)/max(self.bodies[b].shape)
                if not .8<=size_ratio<=1.25:
                    pairrows.append(dict(a=a,b=b,winner=None,cells=[],
                        reason='incomparable_legend_body_sizes'))
                    continue
                pair=pooled_pair_evidence(masks[a],masks[b],pos,neg,valid,self.ink[cut],gx,gy,quadrant,a,b)
                if self.boundary_priority:
                    outline=compare_transitions(normal[a],normal[b],a,b)
                    pair['pixel_winner']=pair['winner']
                    pair['winner']=outline['winner']
                    pair['normal_comparison']=outline
                if pair['winner']:
                    wins[pair['winner']].append(b if pair['winner']==a else a)
                for cell in pair['cells']:
                    if cell['winner']:votes.setdefault(tuple(cell['cell']),[]).append(cell['winner'])
                pairrows.append(pair)
        champions=[k for k in ids if len(wins[k])==len(ids)-1]
        defeated={r for rivals in wins.values() for r in rivals}
        # Do not require one shape to defeat EVERY legend entry. Incomparable
        # shapes remain alternatives; a decisive pairwise loser cannot silently
        # win again through a near-tie in the generic required-ink rank.
        contenders=[k for k in ids if k not in defeated]
        result=dict(version=VERSION,winner=champions[0] if len(champions)==1 else None,pairs=pairrows,
                    contenders=contenders,defeated=sorted(defeated),body_boundary=boundary,
                    boundary_priority=self.boundary_priority,normal_boundary=normal,
                    body_id=int(label),
                    cell_votes={f'{x},{y}':list(set(v)) for (x,y),v in votes.items()},
                    center=[float(x),float(y)],scale=float(scale),step=self.step,
                    common_geometry='observed compact body; same scale and center for all identities')
        self.cache[key]=result
        if detail:
            result={**result,'box':[left,top,left+size,top+size],
                    'masks':{k:v.tolist() for k,v in masks.items()},'positive':pos.tolist(),
                    'negative':neg.tolist(),'nuisance':self.nuisance[cut].tolist(),
                    'observed_body':observed.tolist()}
        return result

    def filter_hypotheses(self,template,hypotheses,**kwargs):
        if template.key not in self.bodies or len(self.bodies)<2:return hypotheses
        stride=kwargs['grid_step'];kept=[]
        for h in hypotheses:
            result=self.for_template(template,h.x,h.y)
            # A raw vote only consults the SAME common physical cell for all
            # legends. Ambiguous/shared-body evidence passes unchanged.
            tx=int((h.cell_left+.5*stride-self.plot[0])//self.step)
            ty=int((h.cell_top+.5*stride-self.plot[1])//self.step)
            winners=result['cell_votes'].get(f'{tx},{ty}',[])
            if len(winners)==1 and winners[0]!=template.key and result['winner']==winners[0]:
                self.counts['local_votes_removed']+=1
            else:kept.append(h)
        self.counts['local_votes_input']+=len(hypotheses)
        return kept

    def candidate_decision(self,template,x,y):
        result=self.for_template(template,x,y)
        accepted=result['winner'] in (None,template.key)
        self.candidate_rows.append(dict(template=template.key,x=float(x),y=float(y),
            accepted=accepted,**result))
        self.counts['candidate_kept' if accepted else 'candidate_vetoed']+=1
        return accepted

    def for_template(self,template,x,y,total_scale=None):
        scale=template.proposal_scale if total_scale is None else total_scale
        if template.key not in self.bodies:return dict(winner=None,pairs=[],cell_votes={})
        dx,dy=self.body_offsets[template.key]
        return self.evaluate(x+scale*dx,y+scale*dy,scale)

    def report(self):
        return dict(version=VERSION,enabled=True,
                    boundary_priority=self.boundary_priority,
                    eligible_bodies={k:list(v.shape) for k,v in self.bodies.items()},
                    body_offsets=self.body_offsets,
                    measured_body_count=len(self.observed_bodies),
                    common_grid_step=self.step,counts=dict(self.counts),candidate_decisions=self.candidate_rows,
                    note='Shared pixels abstain; positive identity ignores long straight strokes. Opt-in compact-filled pilot.')

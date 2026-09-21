"""Separate BW marker geometry from fill identity; never edit source pixels.

Patterned markers propose centres with a filled silhouette on a separate,
small-hole-closed search raster. Their hatch phase is never a matching target.
Final fill identity is measured on the original grayscale image. Supported
open-circle legends use the existing line + marker inverse-rendering fitter.
"""
from copy import deepcopy
from collections import Counter
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.optimize import least_squares

import partial_swatch_detector as D
from bw_fill_identity_v46 import describe_fill
from bw_shared_scale_v46 import calibrate_swatch_scales, SEARCH_SCALES
from legend_composition_v46 import Config, render_model
from bw_hollow_boundary_v46 import (eligible as paired_hollow, square_interior_evidence,
                                   VERSION as HOLLOW_BOUNDARY_VERSION)

VERSION = 'bw-geometry-then-fill-v4-layered-legend'
FAMILIES = dict(square='square', circle='circle', inv_triangle='triangle_down',
                triangle='triangle_up', rhombus='diamond')
CLASSES = dict(square='filled_square', circle='filled_circle', triangle_down='filled_inv_triangle',
               triangle_up='filled_triangle', diamond='filled_rhombus')


def separate_open_ring(original):
    """Infer a ring across the independently observed horizontal line band.

    The connector may be irregularly dashed in a scanned image. Its phase
    need not match a periodic renderer: line rows are fitted as a nuisance,
    while ring geometry must explain the remaining observed arc pixels.
    """
    t=deepcopy(original);target=t.raw_soft.astype(np.float32)
    h,w=target.shape;cx=(w-1)/2;cy=(h-1)/2
    line_rows=t.line_nuisance.any(axis=1)
    weight=np.ones(target.shape,np.float32);weight[line_rows]=0
    # Restrict to the marker neighbourhood, excluding cropped flank ends.
    yy,xx=np.indices(target.shape)
    weight[np.abs(xx-cx)>.6*t.diameter]=0
    lp=dict(x0=-10.,x1=-9.,cx=0.,cy=-10.,thickness=0.,blur_sigma=.5)
    def params(v):return dict(cx=float(v[0]),cy=float(v[1]),width=float(v[2]),height=float(v[3]),stroke_width=float(v[4]))
    def raster(v):return render_model('open_circle',params(v),lp,(h,w))['marker_alpha']
    lo=[cx-2,cy-2,.65*t.diameter,.65*t.diameter,.6,.3]
    hi=[cx+2,cy+2,1.1*t.diameter,1.1*t.diameter,3.5,1.2]
    fit=least_squares(lambda v:((raster(v)*v[5]-target)*weight).ravel(),
        [cx,cy,.87*t.diameter,.87*t.diameter,1.5,.8],bounds=(lo,hi),diff_step=.003,max_nfev=150)
    p=params(fit.x);rmse=float(np.sqrt(np.sum(fit.fun**2)/max(1,weight.sum())))
    if rmse>.20:return None
    centered=fit.x.copy();centered[:2]=cx,cy
    pure=raster(centered);mask=pure>=.22
    t.raw_soft=np.clip(pure*fit.x[5],0,1);t.soft=t.raw_soft.copy()
    t.mask=mask;t.raw_mask=mask.copy();t.line_nuisance=np.zeros_like(mask)
    t.required_weight=np.where(mask,.5+.5*pure,0).astype(np.float32)
    t.valid_weight=np.ones_like(pure);t.edge=cv2.Canny(mask.astype(np.uint8)*255,40,100)>0
    t.orientation=D._edge_orientation(t.soft)
    # Geometry and appearance must share the fitted centre and marker-only
    # diameter. Retaining the union diameter makes scale/fill wrong even when
    # the matching raster is already a pure ring.
    shift=np.float32([[1,0,cx-fit.x[0]],[0,1,cy-fit.x[1]]])
    t.observed_source_soft=cv2.warpAffine(original.raw_soft,shift,(w,h))
    if original.source_gray is not None:
        t.source_gray=cv2.warpAffine(original.source_gray,shift,(w,h),borderValue=t.ink.paper_gray)
    t.marker_center=(original.marker_center[0]+fit.x[0]-cx,original.marker_center[1]+fit.x[1]-cy)
    t.diameter=float(max(p['width'],p['height']))
    t.marker_kind='open';t.interior_fill=0.
    t.model_completed=True
    return t,dict(mode='line_band_plus_inferred_open_ring',status='supported',params=p,
        contrast=float(fit.x[5]),fit_rmse=rmse,line_rows=np.flatnonzero(line_rows).tolist(),
        line_phase_not_matched=True,hidden_arc_is_inferred=True,fitted_diameter=t.diameter,
        original_union_diameter=original.diameter,source_center=list(t.marker_center))


def ring_fill_descriptor(original,ring,shape_evidence=None):
    """Measure observed gray inside the recovered ring, excluding its line.

    The reconstructed white hole is not evidence of an open marker. Only the
    source gray is sampled. Transport the independently observed line band
    into the fitted model's coordinate system before computing fill.
    """
    measurement=deepcopy(ring)
    h,w=ring.mask.shape;oh,ow=original.mask.shape
    dx=ring.marker_center[0]-original.marker_center[0]
    dy=ring.marker_center[1]-original.marker_center[1]
    transform=np.float32([[1,0,(w-ow)/2-dx],[0,1,(h-oh)/2-dy]])
    nuisance=cv2.warpAffine(original.line_nuisance.astype(np.float32),transform,(w,h))>.05
    measurement.line_nuisance=nuisance
    if measurement.source_gray is None:
        # Legacy templates may lack unclipped gray. Use the recoverable
        # observed raster, never the newly synthesized matching ring.
        measurement.raw_soft=cv2.warpAffine(original.raw_soft,transform,(w,h))
    descriptor=describe_fill(measurement,shape_evidence)
    descriptor.update(remeasured_after_ring_fit=True,synthetic_fill_used=False,
                      excluded_line_pixels=int(nuisance.sum()))
    return descriptor


def silhouette_template(original, report):
    """Fit only the outer envelope; internal hatch pixels do not enter loss."""
    t = deepcopy(original)
    evidence = report.get('shape_evidence', {})
    family = FAMILIES.get(evidence.get('best_shape'))
    if family is None or evidence.get('shape_confidence', 0) < .75:
        return None
    ys, xs = np.nonzero(t.mask & ~t.line_nuisance)
    if len(xs) < 6:
        return None
    envelope = np.zeros(t.mask.shape, np.uint8)
    cv2.fillConvexPoly(envelope, cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32)), 1)
    target = cv2.GaussianBlur(envelope.astype(np.float32), (3,3), .5)
    h,w = target.shape; cx=(w-1)/2; cy=(h-1)/2
    width=float(np.ptp(xs)+1); height=float(np.ptp(ys)+1)
    line=dict(x0=-10.,x1=-9.,cx=0.,cy=-10.,thickness=0.,blur_sigma=.45)
    cfg=Config(supersample=3)
    def pars(v):return dict(cx=float(v[0]),cy=float(v[1]),width=float(v[2]),height=float(v[3]))
    fit=least_squares(lambda v:(render_model(family,pars(v),line,(h,w),cfg)['marker_alpha']-target).ravel(),
        [cx,cy,width,height],bounds=([cx-3,cy-3,.6*width,.6*height],[cx+3,cy+3,1.3*width,1.3*height]),
        diff_step=.003,max_nfev=80)
    p=pars(fit.x)
    pure=render_model(family,dict(p,cx=cx,cy=cy),line,(h,w),cfg)['marker_alpha']
    mask=pure>=.25
    t.observed_source_soft=t.raw_soft.copy()
    t.raw_soft=pure.copy();t.soft=pure.copy();t.raw_mask=mask.copy();t.mask=mask
    t.line_nuisance=np.zeros_like(mask);t.valid_weight=np.ones_like(pure)
    depth=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    t.required_weight=np.where(mask,np.where(depth<=1.05,.35+.4*pure,.85+.15*pure),0).astype(np.float32)
    t.edge=cv2.Canny(mask.astype(np.uint8)*255,40,100)>0;t.orientation=D._edge_orientation(pure)
    t.name=CLASSES[family];t.shape_hint=t.name;t.marker_kind='filled';t.interior_fill=1.
    # Keep the source size convention. One common scale still applies to all
    # markers of this swatch, and appearance is sampled in source coordinates.
    t.model_completed=True
    return t,dict(family=family,params=p,fit_rmse=float(np.sqrt(np.mean(fit.fun**2))),
                  source='legend outer envelope, not hatch phase',silhouette_is_inferred=True)


def fill_statistics(values):
    values=np.asarray(values,np.float32)
    if not values.size:return dict(mean=0.,std=0.,light=1.,dark=0.)
    return dict(mean=float(values.mean()),std=float(values.std()),
                light=float(np.mean(values<.35)),dark=float(np.mean(values>.75)))


def compare_fill(expected, observed, count, fraction):
    """Order-independent grayscale statistics, not hatch-pixel correlation."""
    style=expected.get('style','uncertain')
    mean=float(expected.get('mean_ink',observed['mean']))
    sd=float(expected.get('std_ink',observed['std']))
    loss=abs(mean-observed['mean'])+.25*abs(sd-observed['std'])
    result=dict(style=style,decision='abstain',reason='insufficient_visible_interior',
                loss=float(loss),observed=observed,expected_mean=mean,
                visible_pixels=int(count),visible_fraction=float(fraction),phase_matching=False)
    if count<4 or fraction<.30:return result
    if style=='uncertain':
        if not expected.get('opaque_body_supported'):
            result['reason']='legend_fill_unresolved'
            return result
        # Keep the uncertain tone label: independent opaque-body evidence can
        # support geometry without pretending that the tone is known solid.
        conflict=observed['mean']<.74 or observed['light']>.20 or loss>.22
    elif style=='solid':
        conflict=observed['mean']<.74 or observed['light']>.20
    elif style=='open':
        conflict=observed['mean']>.30
    elif style in ('patterned','gray','partial'):
        conflict=(observed['mean']>.86 or observed['mean']<.15 or abs(mean-observed['mean'])>.30)
    else:return result
    result.update(decision='conflict' if conflict else 'compatible',
                  reason=('fill_distribution_mismatch' if conflict else
                          'independent_opaque_body_supported' if style=='uncertain' else
                          'phase_independent_fill_supported'))
    return result


def compact_iou(local,env,diameter):
    kernel=max(3,int(round(.30*diameter))|1)
    compact=cv2.morphologyEx(local,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
    n,components,_,_=cv2.connectedComponentsWithStats(compact)
    choices=[(int(((components==i)&env.astype(bool)).sum()),i) for i in range(1,n)]
    if not choices:return None
    body=components==max(choices)[1]
    if body.sum()>2.5*max(1,env.sum()):return None
    return float((body&env.astype(bool)).sum()/max(1,(body|env.astype(bool)).sum()))


def ring_hole_evidence(observed,mask,available,diameter):
    """Only long thin crossing strokes can explain ink in a hollow centre.

    Short repeated hatch strokes confined to the marker cannot be excused as
    a connecting line. Pixel values remain the original paper-relative gray.
    """
    hole=binary_fill_holes(mask)&~mask
    hole=cv2.erode(hole.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    ink=(observed>.22).astype(np.uint8);length=max(7,int(round(1.1*diameter)))|1
    strokes=np.zeros_like(ink)
    for angle in np.linspace(0,np.pi,12,endpoint=False):
        kernel=np.zeros((length,length),np.uint8);r=length//2
        dx,dy=int(round(r*np.cos(angle))),int(round(r*np.sin(angle)))
        cv2.line(kernel,(r-dx,r-dy),(r+dx,r+dy),1,1)
        strokes|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    thin=(strokes>0)&(cv2.distanceTransform(ink,cv2.DIST_L2,5)<=max(1.4,.17*diameter))
    visible=hole&available&~thin
    return dict(mean=float(observed[visible].mean()) if visible.any() else None,
                count=int(visible.sum()),fraction=float(visible.sum()/max(1,hole.sum())))


def ring_radial_evidence(observed,available,center,diameter):
    """Require curved ring ridges, not the white gap between parallel lines.

    Each radial ray looks for ink with lighter pixels on BOTH radial sides.
    A crossing line may hide one side of a diameter; its opposite arc can
    still support that direction. Two unsupported diameter directions are
    allowed for crossing error bars. This is a conservative source test, not
    additional synthetic ink, and cannot promote a rejected candidate.
    """
    cx,cy=center;angles=np.linspace(0,2*np.pi,24,endpoint=False)
    soft=cv2.GaussianBlur(observed.astype(np.float32),(3,3),.65)
    def sample(array,radius):
        return cv2.remap(array.astype(np.float32),np.float32(cx+radius*np.cos(angles))[None,:],
                         np.float32(cy+radius*np.sin(angles))[None,:],cv2.INTER_LINEAR)[0]
    spacing=max(2.,.20*diameter);tolerance=max(1.,.13*diameter)
    scores=[]
    for radius in np.linspace(diameter/2-tolerance,diameter/2+.8*tolerance,8):
        peak=sample(soft,radius)
        ridge=np.minimum(peak-sample(soft,radius-spacing),peak-sample(soft,radius+spacing))
        valid=np.minimum.reduce([sample(available,radius+s) for s in (-spacing,0,spacing)])>.99
        scores.append(np.where(valid,ridge,-1.))
    ridge=np.max(scores,axis=0);pairs=np.maximum(ridge[:12],ridge[12:])
    supported=int((pairs>.20).sum())
    return dict(supported=supported>=10,supported_diameters=supported,total_diameters=12,
                required_diameters=10,ridge_threshold=.20,radial_ridges=ridge.tolist(),
                method='source_ink_ridge_between_lighter_radial_neighbours')


class GeometryFirst:
    def __init__(self,image,templates,reports,plot,legend,exclusions):
        self.source=image
        self.originals={t.key:deepcopy(t) for t in templates}
        self.descriptors={t.key:describe_fill(t,r.get('shape_evidence')) for t,r in zip(templates,reports)}
        self.plot=plot;self.exclusions=exclusions;self.counts=Counter();self.details={}
        self.templates=[];self.reports=[];self.open_keys=set();self.filled_keys=set()
        # No transformation if the legend does not contain an independently
        # identified patterned fill or open circle/square. Other BW profiles stay put.
        eligible=any(d['style']=='patterned' for d in self.descriptors.values()) or any(t.name in ('open_circle','open_square') for t in templates)
        if not eligible:
            self.templates=templates;self.reports=reports;self.enabled=False;return
        for original,report in zip(templates,reports):
            t=deepcopy(original);r=deepcopy(report);style=self.descriptors[t.key]['style']
            if t.name in ('open_circle','open_square') and legend is not None:
                from bw_composed_legend_v46 import compose_templates
                new,rr,_=compose_templates(image,legend,[t],[r])
                if new[0].model_completed and new[0].name==original.name:
                    t,r=new[0],rr[0];self.open_keys.add(t.key)
                    self.details[t.key]=dict(mode='line_plus_pure_'+t.name,status='supported')
                else:
                    fallback=separate_open_ring(t) if t.name=='open_circle' else None
                    if fallback is not None:
                        t,detail=fallback;self.open_keys.add(t.key);self.details[t.key]=detail
                        r.update(composition=rr[0].get('composition'),template_policy='pure_ring_with_measured_line_nuisance',
                                 model_completed=True,ring_decomposition=detail)
                    else:
                        self.details[t.key]=dict(mode='observed_fallback',status=rr[0].get('composition',{}).get('status'))
                if t.key in self.open_keys:
                    self.descriptors[t.key]=ring_fill_descriptor(original,t,r.get('shape_evidence'))
                    analysis=report.get('line_analysis',{})
                    self.details[t.key]['compact_line_recovery']=(t.name=='open_circle' and analysis.get('status')=='separated'
                        and analysis.get('aspect',float('inf'))<=1.7)
                    r.update(fitted_diameter=t.diameter,source_center=list(t.marker_center))
            elif not t.model_completed and (style in ('solid','patterned','gray','partial') or
                                            self.descriptors[t.key].get('opaque_body_supported')):
                fitted=silhouette_template(t,r)
                if fitted is not None:
                    t,fit=fitted;self.filled_keys.add(t.key)
                    r.update(class_name=t.name,shape_hint=t.name,template_policy='filled_geometry_then_original_fill',
                        original_class=original.name,geometry_model=fit)
                    self.details[t.key]=dict(mode='filled_geometry_then_fill_distribution',style=style,**fit)
            r['geometry_first']=self.details.get(t.key,dict(mode='observed_fallback'))
            r['fill_evidence']=self.descriptors[t.key]
            self.templates.append(t);self.reports.append(r)
        self.enabled=bool(self.open_keys or self.filled_keys)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.darkness=np.clip((255-gray)/255.,0,1)
        # Close only 1-2 pixel internal gaps in an auxiliary search raster.
        # The filled search image is never used for the final tone evidence.
        soft=np.clip((255-gray)/191.,0,1)
        closed=cv2.morphologyEx(soft,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        self.geometry_image=np.repeat(np.uint8(np.rint(255-191*closed))[...,None],3,axis=2)
        self.valid=np.zeros(self.darkness.shape,np.uint8)
        a,b,c,d=self.plot;self.valid[b:d,a:c]=1
        for a,b,c,d in self.exclusions:self.valid[b:d,a:c]=0
        self.compact_sources={False:(gray<190).astype(np.uint8),True:(closed>.34).astype(np.uint8)}
        self.by_key={t.key:t for t in self.templates}
        # Compete against symbols that actually occur in this legend. Absent
        # polygons are not alternate series labels; generic low-signal/window
        # guards still reject paper, lines and incomplete outlines.
        shapes={t.name.replace('filled_','').replace('open_','').replace('rhombus','diamond') for t in self.templates}
        rivals=tuple(k for k in ('square','diamond','triangle','inv_triangle') if k in shapes)
        for t in self.templates:t.circle_rival_shapes=rivals

    def image_for(self,t):
        return self.geometry_image if t.key in self.filled_keys else self.source

    def calibrate(self,log_fn):
        reports={}
        # Separate size from identity: duplicate outer shapes must not veto one
        # another before the independent fill stage has even run.
        for t in self.templates:
            from bw_circle_boundary_v46 import eligible as paired_circle
            if paired_circle(t):
                from bw_circle_scale_v46 import calibrate_circle_scale
                reports[t.key]=calibrate_circle_scale(self.source,t,self.plot,self.exclusions,log_fn,
                    window_check=lambda template,result:self.assess(template,result,record=False))
                continue
            r=calibrate_swatch_scales(self.image_for(t),[t],self.plot,self.exclusions,log_fn,
                anchor_filter=self.anchor_evidence,
                **({'search_scales':tuple(round(.5+.025*i,3) for i in range(33))}
                   if paired_hollow(t) else {}))[t.key]
            r['identity_policy']='geometry_only; fill identity evaluated after center/size'
            if paired_hollow(t):
                r['identity_policy']='paired square boundaries and source hole; one shared scale'
            reports[t.key]=r
        return reports

    def anchor_evidence(self,t,q):
        from occlusion_aware_window_verifier import _crop_padded
        scale=q.get('profile_scales',SEARCH_SCALES)[int(np.argmax(q['profile']))]
        side=max(3,round(t.mask.shape[0]*scale))
        mask=cv2.resize(t.mask.astype(np.uint8),(side,side),interpolation=cv2.INTER_NEAREST)
        if paired_hollow(t):
            # Keep real external flanks for independently supported crossing dashes.
            padding=max(4,int(np.ceil(t.diameter*scale)))
            mask=np.pad(mask,padding).astype(bool);side=mask.shape[0]
            observed=_crop_padded(self.darkness,(q['x'],q['y']),side)
            valid=_crop_padded(self.valid,(q['x'],q['y']),side).astype(bool)
            evidence=square_interior_evidence(observed,mask,valid)
            evidence['accepted']=evidence['decision']=='compatible'
            return evidence
        region=cv2.distanceTransform(binary_fill_holes(mask).astype(np.uint8),cv2.DIST_L2,5)>=max(1.5,.18*t.diameter*scale)
        observed=_crop_padded(self.darkness,(q['x'],q['y']),side)
        stats=fill_statistics(observed[region])
        evidence=compare_fill(self.descriptors[t.key],stats,int(region.sum()),1.)
        # Scale anchors must be cleaner than final overlapping candidates.
        # A bright open centre, for example, is not a filled body's anchor.
        evidence['accepted']=evidence['decision']=='compatible'
        if t.key in self.filled_keys:
            local=_crop_padded(self.compact_sources[True],(q['x'],q['y']),side)
            iou=compact_iou(local,binary_fill_holes(mask).astype(np.uint8),t.diameter*scale)
            evidence['isolated_outer_iou']=iou
            # This is diagnostic, not a hard identity gate: a partial outline
            # may still supply size evidence after the fill test.
        return evidence

    def assess(self,t,v,record=True):
        """Final, source-only tone plus compact-shape evidence at aligned centre."""
        key=t.key;mask=v.template_mask
        env=binary_fill_holes(mask).astype(np.uint8)
        diameter=t.diameter*v.scale
        depth=cv2.distanceTransform(env,cv2.DIST_L2,5)
        region=depth>=max(1.5,.18*diameter)
        size=mask.shape[0]
        from occlusion_aware_window_verifier import _crop_padded
        observed=_crop_padded(self.darkness,(v.x,v.y),size,fill=0.)
        available=_crop_padded(self.valid,(v.x,v.y),size).astype(bool)
        binary=(observed>=.35).astype(np.uint8)
        # Independently extending thin lines are nuisance, never a hatch phase.
        length=max(7,int(round(1.25*diameter)))|1
        strokes=np.zeros_like(binary)
        for kernel in (np.ones((1,length),np.uint8),np.ones((length,1),np.uint8)):
            strokes|=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        thin=strokes.astype(bool)&(cv2.distanceTransform(binary,cv2.DIST_L2,5)<=1.4)
        visible=region&available&~thin
        result=compare_fill(self.descriptors[key],fill_statistics(observed[visible]),int(visible.sum()),
                            float(visible.sum()/max(1,region.sum())))
        result.update(geometry_required_recall=float(v.required_recall),
                      geometry_decision=v.decision,source_pixels_unchanged=True)
        if key in self.open_keys:
            # The existing ring guard measures the actual hole, excluding
            # independently supported crossing strokes. A filled-envelope
            # erosion includes ring pixels and wrongly darkens that hole.
            hollow=v.compute_diagnostics.get('open_interior',{})
            if paired_hollow(t):
                # One authoritative source measurement, also used by the window
                # guard. Do not veto a dash twice with incompatible length rules.
                if hollow.get('version')!=HOLLOW_BOUNDARY_VERSION:
                    hollow=square_interior_evidence(observed,mask,available)
                result.update(decision=hollow['decision'],reason=hollow['reason'],
                              loss=hollow.get('own_ink_fraction'),paired_hollow=hollow)
                long_hole=None
            else:
                long_hole=ring_hole_evidence(observed,mask,available,diameter)
                result['long_stroke_hole']=long_hole
            if not paired_hollow(t) and hollow.get('status')=='hollow_interior_supported':
                result.update(decision='compatible',reason='original_hollow_interior_supported',
                              loss=float(hollow['own_ink_fraction']))
            elif not paired_hollow(t) and hollow.get('status')=='unobserved':
                result.update(decision='abstain',reason='unobserved_hollow_interior')
            if long_hole is not None and long_hole['count']>=4 and long_hole['mean']>.30:
                result.update(decision='conflict',reason='ink_inside_hole_not_explained_by_crossing_line')
            # A white interior alone is not a ring. Require ink distributed
            # around its perimeter, including sectors away from T/cross arms.
            yy,xx=np.indices(mask.shape);mx=size//2+v.aligned_x-v.x;my=size//2+v.aligned_y-v.y
            sector=np.floor((np.arctan2(yy-my,xx-mx)+np.pi)*12/(2*np.pi)).astype(int)%12
            arc=[]
            for k in range(12):
                pixels=mask&available&(sector==k)
                arc.append(float(observed[pixels].mean()) if pixels.any() else 0.)
            required=max(.10,.22*(255-t.ink.core_gray)/255.)
            good=sum(a>=required for a in arc)
            result.update(ring_sector_ink=arc,ring_supported_sectors=good,ring_sector_threshold=required)
            if good<11:
                result.update(decision='conflict',reason='incomplete_ring_perimeter')
            if self.details.get(key,{}).get('compact_line_recovery'):
                radial=ring_radial_evidence(observed,available,(mx,my),diameter)
                result['radial_ring']=radial
                if not radial['supported'] and result['decision']=='compatible':
                    result.update(decision='abstain',reason='ring_arcs_not_independently_supported')
            from bw_circle_boundary_v46 import eligible as circular, final_circle_evidence
            if circular(t):
                shape=final_circle_evidence(observed,mask,available,rival_shapes=t.circle_rival_shapes)
                result['circle_shape']=shape
                # New shape evidence may demote but never repair an existing
                # missing-ink/interior rejection or invent an occluded marker.
                if shape['decision']=='conflict':result.update(decision='conflict',reason=shape['reason'])
                elif shape['decision']=='abstain' and result['decision']=='compatible':
                    result.update(decision='abstain',reason=shape['reason'])
        # Compare compact outer bodies to penalize fitting an inscribed circle
        # or triangle inside a square. Thin connectors are removed only from
        # this auxiliary geometry comparison, never from the source raster.
        local=_crop_padded(self.compact_sources[key in self.filled_keys],(v.x,v.y),size)
        outer_iou=compact_iou(local,env,diameter) if key in self.filled_keys else None
        result['compact_outer_iou']=outer_iou
        if result['decision']=='conflict':v.decision='rejected'
        elif result['decision']=='abstain' and v.decision=='verified':v.decision='ambiguous'
        # Do not let a perfectly matched tiny T/cap establish a patterned body.
        if key in self.filled_keys and outer_iou is not None and outer_iou<.40:
            v.decision='rejected';result.update(decision='conflict',reason='incompatible_compact_outer_shape')
        if record:self.counts[f'{key}:{result["decision"]}']+=1
        v.compute_diagnostics['geometry_first']=result
        return result

    def appearance(self,t,v):
        """Step 5 keeps observed hatch ink; filled search raster is not data ink."""
        if self.descriptors[t.key]['style'] not in ('patterned','gray','partial'):
            return v.template_mask
        original=self.originals[t.key]
        mask=original.mask
        n=v.template_mask.shape[0];h,w=mask.shape
        transform=np.float32([[v.scale,0,n//2+(v.aligned_x-v.x)-v.scale*(w-1)/2],
                              [0,v.scale,n//2+(v.aligned_y-v.y)-v.scale*(h-1)/2]])
        return cv2.warpAffine(mask.astype(np.uint8),transform,(n,n),flags=cv2.INTER_NEAREST)>0

    def report(self):
        return dict(version=VERSION,enabled=self.enabled,series=self.details,counts=dict(self.counts),
            search_only_hole_closing_kernel=[3,3],fill_matching='phase-independent original grayscale distribution',
            original_pixels_modified=False,models_are_observations=False)

"""Measurement-center hypotheses from stems and independent off-stem curves.

No marker glyph, stem midpoint, stored path, measurement schedule, or numerical
reference is used. Pixel-coordinate fits below are engineering uncertainty
diagnostics, not statistical confidence intervals.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class CenterConfig:
    reach_widths: float = 24.
    fit_span_widths: float = 12.
    minimum_columns_widths: float = 2.
    maximum_gap_widths: float = 8.
    agreement_widths: float = 2.
    maximum_uncertainty_widths: float = 4.


def _fragments(curve, soft, stem, side, width, cfg):
    h,w=curve.shape;x=float(stem['x']);reach=max(16,int(np.ceil(cfg.reach_widths*width)))
    pad=max(6,3*width)
    a=max(0,int(np.floor(x-reach))) if side=='left' else max(0,int(np.floor(x))+1)
    b=min(w,int(np.ceil(x))) if side=='left' else min(w,int(np.ceil(x+reach))+1)
    c=max(0,int(np.floor(stem['y0']-pad)));d=min(h,int(np.ceil(stem['y1']+pad))+1)
    if a>=b or c>=d:return []
    n,labels,stats,_=cv2.connectedComponentsWithStats(curve[c:d,a:b].astype(np.uint8),8)
    out=[];minimum=max(4,int(np.ceil(cfg.minimum_columns_widths*width)))
    for label in range(1,n):
        yy,xx=np.nonzero(labels==label);xx=xx+a;yy=yy+c
        columns=np.unique(xx)
        if len(columns)<minimum:continue
        endpoint=float(columns[-1] if side=='left' else columns[0]);gap=abs(x-endpoint)
        if gap>max(5,cfg.maximum_gap_widths*width):continue
        length=max(12,cfg.fit_span_widths*width)
        columns=columns[columns>=endpoint-length] if side=='left' else columns[columns<=endpoint+length]
        if len(columns)<minimum:continue
        # A connected component can still contain two distant same-color curves.
        # Do not average their column runs into an ink-free fictitious center.
        split_columns=[]
        for col in columns:
            iy=np.sort(yy[xx==col])
            if np.any(np.diff(iy)>max(2.,1.5*width)):split_columns.append(col)
        if len(split_columns)>=max(2,.2*len(columns)):continue
        columns=columns[~np.isin(columns,split_columns)]
        if len(columns)<minimum:continue
        ys=[];weights=[]
        for col in columns:
            iy=yy[xx==col];mass=np.maximum(.01,soft[iy,col])
            # Center the observed stroke without selecting its brightest edge.
            ys.append(float(np.average(iy,weights=mass)));weights.append(float(min(3.,mass.sum())))
        xs=columns.astype(float);ys=np.asarray(ys);weights=np.asarray(weights)
        slope,intercept=np.polyfit(xs-x,ys,1,w=np.sqrt(weights))
        residual=ys-(slope*(xs-x)+intercept);rms=float(np.sqrt(np.average(residual**2,weights=weights)))
        if rms>max(1.5,1.1*width):continue
        # Two biased straight fits can agree on a curved trace. Compare fits at
        # different spans, and a quadratic diagnostic, without promoting it to y.
        near=np.argsort(abs(xs-x))[:max(minimum,int(np.ceil(len(xs)*.55)))]
        short_intercept=float(np.polyfit(xs[near]-x,ys[near],1,w=np.sqrt(weights[near]))[1])
        quadratic_intercept=float(np.polyfit(xs-x,ys,2,w=np.sqrt(weights))[2]) if len(xs)>=minimum+2 else float(intercept)
        instability=max(abs(short_intercept-intercept),abs(quadratic_intercept-intercept))
        # Geometry-only uncertainty; long extrapolation or steep stems is explicit.
        spread=max(1.,float(np.std(xs)))
        uncertainty=max(.6,rms)*np.sqrt(1+(gap/spread)**2)+abs(float(slope))*max(.5,float(stem.get('width',width))/2)+instability
        if not stem['y0']-2*width<=intercept<=stem['y1']+2*width:continue
        out.append(dict(side=side,y=float(intercept),slope=float(slope),rms=rms,gap=gap,
            uncertainty=float(uncertainty),intercept_instability=float(instability),short_span_intercept=short_intercept,
            quadratic_intercept_diagnostic=quadratic_intercept,source_columns=int(len(xs)),x_span=float(np.ptp(xs)),
            source_box=[int(xs.min()),int(np.floor(ys.min())),int(xs.max())+1,int(np.ceil(ys.max()))+1],
            fit_points=np.column_stack([xs,ys]).tolist(),endpoint=[endpoint,float(ys[-1] if side=='left' else ys[0])],
            source_kind='same-series off-stem off-cap pixels'))
    return sorted(out,key=lambda q:(q['gap'],q['rms'],-q['source_columns']))


def _outside_cap(fit,cap,side,width):
    columns=np.asarray(fit['fit_points'])[:,0]
    beyond=columns<float(cap['x0'])-width if side=='left' else columns>float(cap['x1'])+width
    return int(beyond.sum())>=max(3,int(np.ceil(width)))


def _horizontal_confounds(own,stems,width):
    """Observed short horizontal arms are ambiguity cues, never automatic erasers.

    A cap of a fragmented/neighboring stem can be absent from that stem's cap
    list. Detect its raster extent independently. A short flat data trace can
    look identical, so this check lowers certainty rather than deleting ink.
    """
    known=[float(c['width']) for s in stems for c in s.get('caps',[]) if c.get('two_sided',False)]
    maximum=max(8*width,2*np.median(known)) if known else 16*width
    length=max(5,int(np.ceil(4*width)))
    opened=cv2.morphologyEx(own.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(opened,8)
    arms=[]
    for i in range(1,n):
        x,y,w,h,_=stats[i]
        if not length<=w<=maximum or h>max(3,2*width):continue
        arms.append(dict(x0=float(x),x1=float(x+w-1),y=float(y+(h-1)/2),width=float(w),kind='short_horizontal_ambiguity_cue'))
    return arms


def estimate_centers(own,valid,stem_result,line_width,soft=None,cfg=CenterConfig()):
    own=np.asarray(own,bool);valid=np.asarray(valid,bool);width=max(1.,float(line_width))
    if own.ndim!=2 or valid.shape!=own.shape:raise ValueError('Masks must be same-sized 2D arrays')
    if soft is None:soft=own.astype(float)
    soft=np.asarray(soft,float)
    if soft.shape!=own.shape or not np.isfinite(soft).all():raise ValueError('Invalid soft field')
    stems=stem_result['stems']
    sm=np.asarray(stem_result['stem_mask'],bool);cm=np.asarray(stem_result['cap_mask'],bool)
    if sm.shape!=own.shape or cm.shape!=own.shape:raise ValueError('Structure masks differ in size')
    # A one-pixel boundary margin removes anti-aliased structure edges as well.
    excluded=cv2.dilate((sm|cm).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    curve=own & valid & ~excluded
    horizontal_confounds=_horizontal_confounds(own&valid,stems,width)
    candidates=[];points=[];uncertain=[]
    for index,stem in enumerate(stems):
        left=_fragments(curve,soft,stem,'left',width,cfg)
        right=_fragments(curve,soft,stem,'right',width,cfg)
        base=dict(id=stem.get('id',f'E{index+1:02d}'),x=float(stem['x']),y=None,
            y_interval=[float(stem['y0']),float(stem['y1'])],stem=stem,
            left_fits=left,right_fits=right,kind='errorbar_center',marker_glyph_detected=False,
            y_source='independent off-stem curve fits; never bar midpoint')
        hypotheses=[]
        for l in left:
            for r in right:
                tolerance=max(cfg.agreement_widths*width,l['uncertainty']+r['uncertainty'])
                disagreement=abs(l['y']-r['y'])
                if disagreement>tolerance:continue
                masses=np.array([1/max(.6,l['uncertainty'])**2,1/max(.6,r['uncertainty'])**2])
                y=float(np.average([l['y'],r['y']],weights=masses))
                nearby=[cap for cap in stem.get('caps',[]) if abs(y-cap['y'])<=2*width]
                cap_confounded=any(not(_outside_cap(l,cap,'left',width) and _outside_cap(r,cap,'right',width)) for cap in nearby)
                neighboring=[cap for cap in horizontal_confounds if cap['x0']-width<=stem['x']<=cap['x1']+width and abs(y-cap['y'])<=width]
                neighboring_confounded=any(not(_outside_cap(l,cap,'left',width) and _outside_cap(r,cap,'right',width)) for cap in neighboring)
                quality=1/(1+(l['gap']+r['gap'])/(4*width)+disagreement/(2*width)+(l['rms']+r['rms'])/width)
                hypotheses.append(dict(y=y,left=l,right=r,disagreement=disagreement,tolerance=tolerance,
                    fit_uncertainty=max(l['uncertainty'],r['uncertainty'],disagreement/2),
                    cap_confounded=cap_confounded,neighboring_cap_confounded=neighboring_confounded,
                    horizontal_ambiguity_cues=neighboring,rank_score=float(quality)))
        hypotheses.sort(key=lambda p:p['rank_score'],reverse=True)
        if hypotheses:
            selected=hypotheses[0];base.update(y=selected['y'],selected=selected,hypotheses=hypotheses)
            competing=any(abs(p['y']-selected['y'])>3*width and p['rank_score']>=.7*selected['rank_score'] for p in hypotheses[1:])
            if selected['cap_confounded']:status,reason='uncertain','insufficient_curve_support_beyond_cap'
            elif selected['neighboring_cap_confounded']:status,reason='uncertain','short_horizontal_arm_may_be_neighboring_cap'
            elif competing:status,reason='uncertain','multiple_independent_curve_intersections'
            elif max(selected['left']['intercept_instability'],selected['right']['intercept_instability'])>width:status,reason='uncertain','curvature_or_fit_span_instability'
            elif selected['fit_uncertainty']>cfg.maximum_uncertainty_widths*width:status,reason='uncertain','large_extrapolation_uncertainty'
            elif not any(cap.get('strict',False) for cap in stem.get('caps',[])):status,reason='uncertain','unconfirmed_vertical_structure_no_strict_caps'
            else:status,reason='accepted','left_right_curve_intersections_agree'
            uncertainty=max(width,selected['fit_uncertainty']);base['y_interval']=[selected['y']-uncertainty,selected['y']+uncertainty]
        elif left or right:
            choices=left+right;chosen=min(choices,key=lambda f:(f['gap'],f['rms']))
            base.update(y=chosen['y'],selected_side=chosen)
            status='uncertain';reason='left_right_disagree' if left and right else 'one_sided_curve_evidence'
        else:
            status,reason='unresolved','no_off_stem_curve_evidence'
        base.update(status=status,reason=reason)
        # Form all competing interpretations before spatial validation: an
        # invalid first branch must not hide another valid branch, nor may an
        # alternate bypass the same valid-plot check used for the first branch.
        rows=[base]
        if hypotheses and competing:
            levels=[base['y']]
            for hypothesis in hypotheses[1:]:
                if hypothesis['rank_score']<.7*hypotheses[0]['rank_score']:continue
                if all(abs(hypothesis['y']-y)>3*width for y in levels):
                    alternate=dict(base,id=f"{base['id']}_alt{len(levels)}",y=hypothesis['y'],selected=hypothesis,
                                   status='uncertain',reason='alternate_curve_intersection')
                    spread=max(width,hypothesis['fit_uncertainty'])
                    alternate['y_interval']=[hypothesis['y']-spread,hypothesis['y']+spread]
                    rows.append(alternate);levels.append(hypothesis['y'])
        for row in rows:
            # Do not clamp a guessed center to the stem midpoint or plot border.
            if row['y'] is not None:
                ix=int(round(row['x']));iy=int(round(row['y']))
                if not (0<=iy<own.shape[0] and 0<=ix<own.shape[1] and valid[iy,ix]):
                    row.update(status='unresolved',reason='predicted_center_outside_valid_plot',y=None)
            candidates.append(row)
            if row['status']=='accepted':points.append(row)
            elif row['y'] is not None:uncertain.append(row)
    return dict(candidates=candidates,points=points,uncertain_points=uncertain,curve_mask=curve,
        horizontal_ambiguity_cues=horizontal_confounds,config=asdict(cfg),interpretation='Structure-based measurement centers, not visually present marker glyphs')

"""Opt-in filled/open primitive + solid/dashed connector inverse rendering.

Only legend RGB enters fitting. The returned marker_alpha never contains line
pixels. The original observed_alpha is separately retained for audit/evidence.
"""
from dataclasses import asdict
import numpy as np
from scipy import optimize
from legend_composition_v46 import _calibrate, render_model, FAMILIES, _family


class LineEvidenceError(ValueError):
    """Expected lack of independent connector observations for an optional fit."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code=code


def bounded_dash_gaps(profile,flank):
    """Count distinct light gaps bracketed by independent connector ink.

    Two real one-pixel gaps are evidence even when four fully white samples
    do not exist. Padding, the hollow symbol centre and a single break cannot
    establish a periodic connector. Antialias shoulders may surround a gap.
    """
    x=np.asarray(profile);flank=np.asarray(flank,bool)
    low=np.flatnonzero((x<.20)&flank)
    groups=np.split(low,np.flatnonzero(np.diff(low)>1)+1) if len(low) else []
    count=0
    for g in groups:
        a,b=int(g[0]),int(g[-1]);left=max(0,a-3);right=min(len(x),b+4)
        # Both bright-to-ink transitions must belong to the observed flank.
        before=np.flatnonzero((x[left:a]>.60)&flank[left:a])+left
        after=np.flatnonzero((x[b+1:right]>.60)&flank[b+1:right])+b+1
        if len(before) and len(after) and flank[before[-1]:after[0]+1].all():count+=1
    return count


def line_candidates(alpha,cfg,maximum_dashed=1):
    if alpha.ndim!=2 or min(alpha.shape)<2 or not np.isfinite(alpha).all():
        raise ValueError('Line fitting requires a finite 2-D observation of at least 2x2 pixels')
    h,w=alpha.shape;xs=np.arange(w)
    support=alpha>.25
    spans=np.array([np.ptp(np.flatnonzero(col))+1 if col.any() else 0 for col in support.T])
    tall=spans>max(5.,.30*h)
    columns=np.flatnonzero(tall)
    # Flanks are outside the measured marker bulge, not a fixed crop quarter.
    # Fixed crop quarters miss most short-dash evidence and alias the period.
    flank=(xs<columns[0]-1)|(xs>columns[-1]+1) if len(columns) else (xs<.25*w)|(xs>.75*w)
    if not flank.any():
        raise LineEvidenceError('no_independent_line_flanks',
            'Marker/frame support spans the crop: no independent columns remain to fit a connector')
    profile=alpha[:,flank].mean(axis=1);peak=int(np.argmax(profile))
    use=np.abs(np.arange(h)-peak)<=max(3,h*.15)
    cy=float(np.sum(profile*use*np.arange(h))/max(np.sum(profile*use),1e-9))
    xp=alpha[max(0,peak-1):min(h,peak+2)].max(axis=0)
    occupied=xp>.20
    cols=occupied&flank
    thickness=float(np.clip(np.mean(alpha[:,cols].sum(axis=0)) if cols.any() else 2.,.5,.25*h))
    nz=np.flatnonzero(alpha.max(axis=0)>.15)
    base=dict(x0=float(nz[0]-.5) if len(nz) else 0.,x1=float(nz[-1]+.5) if len(nz) else w-1.,
        cy=cy,cx=(w-1)/2,thickness=thickness,slope=0.,blur_sigma=.4,style='solid')
    guesses=[base]
    # Find a repeated on/off pattern on independent flanks. Marker pixels in
    # the middle do not choose phase or period. Continuous fits refine seeds.
    scored=[]
    dash_evidence=flank.copy()
    if maximum_dashed>1 and len(nz):
        # Padding outside a solid connector is not an observed periodic gap.
        # Otherwise a gap hidden wholly under a filled marker can spuriously
        # win as a "dashed" line with the same visible pixels as a solid one.
        dash_evidence &= (xs>nz[0]+2)&(xs<nz[-1]-2)
    repeated_gaps=bounded_dash_gaps(xp,flank)
    if ((xp[dash_evidence]>.5).sum()>=4 and
        ((xp[dash_evidence]<.2).sum()>=4 or repeated_gaps>=2)):
        for period in np.arange(5.,max(6.,.8*w),1.):
            for duty in (.35,.50,.65,.80):
                for phase in np.arange(0,period,1.5):
                    distance=np.abs((xs-phase+period/2)%period-period/2)
                    pred=np.clip(.5+period*duty/2-distance,0,1)*((xs>=base['x0'])&(xs<=base['x1']))
                    loss=float(((pred-xp)[flank]**2).mean())
                    scored.append((loss,period,period*duty,phase))
        scored.sort()
        seeds=[]
        for row in scored:
            if all(abs(row[1]-s[1])>3 for s in seeds):seeds.append(row)
            if len(seeds)==2:break
        for _,period,length,phase in seeds:guesses.append(dict(base,style='dashed',dash_period=period,dash_length=length,dash_phase=phase))
    output=[]
    for lp in guesses:
        dashed=lp['style']=='dashed'
        initial=[cy,thickness,0.,.4];low=[max(0,cy-3),.3,-.1,.05];high=[min(h-1,cy+3),max(1.,2*thickness),.1,1.3]
        if dashed:
            initial +=[lp['dash_period'],lp['dash_length'],lp['dash_phase']]
            low +=[max(3.,lp['dash_period']-3),max(.5,lp['dash_length']-3),lp['dash_phase']-4]
            high +=[lp['dash_period']+3,lp['dash_length']+3,lp['dash_phase']+4]
        def unpack(v):
            out=dict(lp,cy=float(v[0]),thickness=float(v[1]),slope=float(v[2]),blur_sigma=float(v[3]))
            if dashed:out.update(dash_period=float(v[4]),dash_length=float(min(v[5],.95*v[4])),dash_phase=float(v[6]))
            return out
        def residual(v):return (render_model('line_only',{},unpack(v),alpha.shape,cfg)['line_alpha'][:,flank]-alpha[:,flank]).ravel()
        fitted=optimize.least_squares(residual,initial,bounds=(low,high),diff_step=.003,max_nfev=cfg.maximum_evaluations)
        fit=unpack(fitted.x);fit['flank_mse']=float(np.mean(residual(fitted.x)**2));output.append(fit)
    output.sort(key=lambda p:p['flank_mse']+(cfg.complexity_cost_per_parameter*3 if p['style']=='dashed' else 0))
    # Keep solid and the best periodic hypothesis for joint model comparison.
    dashed=[p for p in output if p['style']=='dashed'][:maximum_dashed]
    solid=next(p for p in output if p['style']=='solid')
    return [solid]+dashed,flank


def fit_extended(rgb,nominal,cfg):
    alpha,fg,bg,colour_error=_calibrate(rgb,nominal)
    h,w=alpha.shape;lines,flank=line_candidates(alpha,cfg);models=[]
    # The same evaluation region is used for every model and line style.
    tall=np.array([np.ptp(np.flatnonzero(col))+1 if col.any() else 0 for col in (alpha>.25).T])>max(5.,.3*h)
    ids=np.flatnonzero(tall);pad=max(4.,.35*h)
    roi=[max(0,int(ids[0]-pad)),0,min(w,int(ids[-1]+pad+1)),h] if len(ids) else [0,0,w,h]
    sl=np.s_[roi[1]:roi[3],roi[0]:roi[2]]
    for lp in lines:
        line=render_model('line_only',{},lp,alpha.shape,cfg)['line_alpha']
        excess=np.maximum(alpha-line,0)
        # Central marker region excludes disconnected outer dashes during init.
        init_region=(excess>.2);init_region[:,:int(.20*w)]=False;init_region[:,int(.80*w):]=False
        ys,xs=np.nonzero(init_region)
        if len(xs):cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2;width=max(4.,xs.max()-xs.min()+1.);height=max(4.,ys.max()-ys.min()+1.)
        else:cx,cy=(w-1)/2,lp['cy'];width=height=max(3.,h/3)
        def assess(name,pars,n):
            fields=render_model(name,pars,lp,alpha.shape,cfg);diff=(fields['composite_alpha']-alpha)[sl]
            mse=float((diff**2).mean());ink=max(float(excess[sl].sum()),float(fields['visible_marker_evidence'][sl].sum()),1.)
            cost=cfg.complexity_cost_per_parameter*(n+(3 if lp['style']=='dashed' else 0))
            return dict(name=name,params=pars,line_params=lp,loss=mse+cost,data_loss=mse,
                mean_absolute_error=float(abs(diff).mean()),marker_relative_error=float(abs(diff).sum()/ink),
                complexity_penalty=cost,free_marker_parameters=n)
        models.append(assess('line_only',{},0))
        if excess.sum()<cfg.minimum_ink_mass:continue
        for name in (*FAMILIES,'open_circle'):
            equal=name in ('circle','square','open_circle');hollow=name=='open_circle'
            initial=[cx,cy,min(width,height) if hollow else (width+height)/2] if equal else [cx,cy,width,height]
            low=[max(0.,cx-width*.45),max(0.,cy-height*.4),max(2.,min(width,height)*.45)]
            high=[min(w-1.,cx+width*.45),min(h-1.,cy+height*.4),min(w*.8,max(width,height)*1.6)]
            if equal:high[2]=min(h*.95,high[2])
            else:low+=[max(2.,height*.45)];high+=[min(h*.95,height*1.6)]
            if hollow:initial+=[max(1.,.12*height)];low+=[.5];high+=[max(1.,.32*height)]
            def unpack(v):
                p=dict(cx=float(v[0]),cy=float(v[1]),width=float(v[2]),height=float(v[2] if equal else v[3]))
                if hollow:p['stroke_width']=float(v[3])
                return p
            def residual(v):return (render_model(name,unpack(v),lp,alpha.shape,cfg)['composite_alpha']-alpha)[sl].ravel()
            start=np.clip(initial,np.array(low)+1e-5,np.array(high)-1e-5)
            fitted=optimize.least_squares(residual,start,bounds=(low,high),diff_step=.003,max_nfev=cfg.maximum_evaluations)
            models.append(assess(name,unpack(fitted.x),len(initial)))
    models.sort(key=lambda p:p['loss']);best=models[0];lp=best['line_params']
    alternate=next((m for m in models if _family(m['name'])!=_family(best['name'])),None)
    linebest=min((m for m in models if m['name']=='line_only'),key=lambda p:p['loss'])
    improvement=linebest['data_loss']-best['data_loss'];margin=None if alternate is None else alternate['loss']-best['loss']
    colour_mae=float(colour_error.mean())
    status=('blank' if alpha.sum()<cfg.minimum_ink_mass else
        'line_only_or_no_resolved_marker' if best['name']=='line_only' or improvement<cfg.minimum_marker_improvement else
        'unknown_poor_fit' if best['mean_absolute_error']>cfg.maximum_marker_mae or best['marker_relative_error']>cfg.maximum_marker_relative_error or colour_mae>cfg.maximum_colour_residual else
        'ambiguous_shape_family' if margin is None or margin<cfg.minimum_family_margin else 'supported_simple_shape_model')
    fields=render_model(best['name'],best['params'],lp,alpha.shape,cfg)
    fields.update(observed_alpha=alpha,residual=alpha-fields['composite_alpha'],colour_residual=colour_error,
        flank_mask=np.broadcast_to(flank[None,:],alpha.shape).astype(np.uint8))
    centroid=None
    if best['name']!='line_only':
        p=best['params'];centroid=[p['cx'],p['cy']]
        if best['name'] in ('triangle_up','triangle_down'):centroid[1]+=p['height']/6*(1 if best['name']=='triangle_up' else -1)
    return dict(status=status,best_model=best,best_model_name=best['name'],models=models,line_params=lp,
        foreground_rgb=(fg*255).tolist(),background_rgb=(bg*255).tolist(),fit_roi=roi,config=asdict(cfg),
        winner_runner_margin=models[1]['loss']-best['loss'],winner_other_family_margin=margin,
        line_only_data_loss=linebest['data_loss'],marker_improvement=improvement,colour_residual_mae=colour_mae,
        inferred_centroid=centroid,hidden_fraction=float(fields['hidden_marker_by_line'].sum()/max(fields['marker_alpha'].sum(),1e-9)),
        template_policy='pure_model_marker_if_supported',version='open_dashed_composition_v1',
        caveats=['Model-completed pixels are not observations. Open-circle and periodic dash extension is opt-in.',
                 'Unsupported fits must remain explicit observed fallbacks.']),fields

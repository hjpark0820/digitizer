"""Native-pixel comparison for resolved tiny hollow legend models.

No erosion, fake interior samples or image upscaling: fit a complete raster
and compare both ink and paper with filled/rival/line-only explanations.
Crossing strokes are predicted ONLY from independent external source flanks.
"""
from functools import lru_cache
import cv2
import numpy as np
from legend_composition_v46 import render_model,Config

VERSION='small-hollow-source-composition-v1'


def eligible(t):
    return bool(getattr(t,'small_hollow_model',None)) and t.ink.achromatic


def nuisance(observed,center,diameter,available):
    """At most two independently extending thin strokes; not marker fill."""
    cx,cy=center;yy,xx=np.indices(observed.shape);radius=diameter/2
    distances=np.linspace(radius+.75,radius+3.25,7)
    choices=[]
    for angle in np.linspace(0,np.pi,36,endpoint=False):
        ux,uy=np.cos(angle),np.sin(angle);nx,ny=-uy,ux
        for offset in (-.75,0.,.75):
            def sample(perp):
                along=np.r_[-distances,distances]
                xs=np.float32(cx+ux*along+nx*(offset+perp))[None,:]
                ys=np.float32(cy+uy*along+ny*(offset+perp))[None,:]
                v=cv2.remap(observed,xs,ys,cv2.INTER_LINEAR)[0]
                valid=cv2.remap(available.astype(np.float32),xs,ys,cv2.INTER_LINEAR)[0]>.99
                return v,valid
            values,valid=sample(0)
            if not valid.all():continue
            strength=min(float(np.median(values[:7])),float(np.median(values[7:])))
            if strength<.20:continue
            side=np.mean([sample(-1.5)[0].mean(),sample(1.5)[0].mean()])
            if strength-side<.10:continue
            choices.append((strength-side,strength,angle,offset))
    background=np.zeros_like(observed);selected=[]
    for score,strength,angle,offset in sorted(choices,reverse=True):
        if any(abs(np.sin(angle-a))<.65 for a in selected):continue
        cross=-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)-offset
        background=np.maximum(background,strength*np.exp(-.5*(cross/.65)**2))
        selected.append(angle)
        if len(selected)==2:break
    return background.astype(np.float32),len(selected)


@lru_cache(maxsize=1024)
def _bank(name,width,height,stroke,blur,scale,size):
    c=size//2;rows=[]
    for dy in (-.5,0.,.5):
        for dx in (-.5,0.,.5):
            p=dict(cx=c+dx,cy=c+dy,width=width*scale,height=height*scale,
                   stroke_width=stroke*scale,interior_mode='transparent')
            lp=dict(cy=c,cx=c,x0=0,x1=0,thickness=0.,blur_sigma=blur*scale)
            fields=render_model(name,p,lp,(size,size),Config())
            rows.append(fields['marker_alpha'])
    return np.stack(rows)


def evidence(observed,template,center,scale=1.,available=None,occlusion=None):
    """Symmetric whole-body comparison on a common observed region."""
    observed=np.asarray(observed,np.float32)
    if observed.ndim!=2 or not np.isfinite(observed).all():
        raise ValueError('Finite two-dimensional source ink required')
    valid=np.ones(observed.shape,bool) if available is None else np.asarray(available,bool).copy()
    if valid.shape!=observed.shape:raise ValueError('Source/availability shapes must agree')
    if occlusion is not None:
        if np.shape(occlusion)!=observed.shape:raise ValueError('Source/occlusion shapes must agree')
        valid &= np.asarray(occlusion)<.5
    models=getattr(template,'small_hollow_rivals',None) or {template.key:template.small_hollow_model}
    scales=getattr(template,'small_hollow_scales',{})
    diameter=max(max(m['params']['width'],m['params']['height'])*scales.get(k,scale)
                 for k,m in models.items())
    # Sample source pixels onto an integer-centred crop. Interpolation is for
    # alignment only, not an increase in independent sample count.
    size=2*int(np.ceil(diameter+5))+1;c=size//2
    yy,xx=np.indices((size,size),dtype=np.float32)
    xs=xx+float(center[0])-c;ys=yy+float(center[1])-c
    source=cv2.remap(observed,xs,ys,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    seen=cv2.remap(valid.astype(np.float32),xs,ys,cv2.INTER_LINEAR)>.99
    base,lines=nuisance(source,(c,c),diameter,seen)
    roi=(abs(xx-c)<=diameter/2+1)&(abs(yy-c)<=diameter/2+1)
    out=dict(version=VERSION,decision='abstain',reason='insufficient_small_body_visibility',
             loss=1.,source_pixels_unchanged=True,crossing_strokes=lines)
    if np.mean(seen[roi])<.90:return out
    roi &= seen
    losses={};filled_losses={};missing_rims={};physical={}
    for key,m in models.items():
        p=m['params']
        model_scale=getattr(template,'small_hollow_scales',{}).get(key,scale)
        args=(p['width'],p['height'],p['stroke_width'],m['blur_sigma'],model_scale,size)
        marker=_bank(m['name'],*args)
        marker_levels=np.clip(marker[:,None]*np.array([.75,1.,1.25])[None,:,None,None],0,1)
        pred=np.maximum(marker_levels,base)
        errors=np.mean((pred[:,:,roi]-source[roi])**2,axis=2)
        best=np.unravel_index(errors.argmin(),errors.shape)
        losses[key]=float(errors[best])
        expected=marker_levels[best]
        rim=(expected>.25)&roi
        missing_rims[key]=float(np.maximum(expected[rim]-source[rim],0).sum()/max(expected[rim].sum(),1.e-6))
        residual=pred[best]-source
        physical[key]=dict(
            missing_model_ink_mse=float(np.mean(np.maximum(residual[roi],0)**2)),
            extra_observed_ink_mse=float(np.mean(np.maximum(-residual[roi],0)**2)),
            paper_on_rim=float(np.sum(expected[rim]*np.clip((.18-source[rim])/.18,0,1))/max(expected[rim].sum(),1.e-6)))
        filled=_bank(m['name'].removeprefix('open_'),*args)
        filled_pred=np.maximum(np.clip(filled[:,None]*np.array([.75,1.,1.25])[None,:,None,None],0,1),base)
        filled_losses[key]=float(np.mean((filled_pred[:,:,roi]-source[roi])**2,axis=2).min())
    loss=losses[template.key];line_loss=float(np.mean((base[roi]-source[roi])**2))
    rival=min([v for k,v in losses.items() if k!=template.key] or [1.])
    improvement=line_loss-loss;fill_margin=filled_losses[template.key]-loss
    out.update(loss=loss,model_losses=losses,line_only_loss=line_loss,marker_improvement=improvement,
               filled_loss=filled_losses[template.key],hollow_margin=fill_margin,rival_margin=rival-loss,
               missing_rim_fraction=missing_rims[template.key],physical_residual=physical[template.key])
    if missing_rims[template.key]>.30:
        out.update(decision='conflict',reason='small_missing_required_rim')
    elif (loss>.065 and improvement>=.012 and missing_rims[template.key]<=.15
          and physical[template.key]['paper_on_rim']<=.08 and fill_margin>=.008):
        # Surplus ink cannot be declared harmless same-colour occlusion, but
        # is not equivalent to physically missing white rim pixels either.
        # Preserve a hypothesis for a joint/raster explanation, not an active.
        out.update(decision='abstain',reason='small_hollow_overlap_or_model_residual',
                   requires_joint_explanation=True)
    elif loss>.065 or improvement<.012:
        out.update(decision='conflict',reason='small_outline_not_explained')
    elif fill_margin<.008:
        out.update(decision='abstain',reason='small_hollow_not_resolved_against_filled')
    elif rival-loss<.002:
        out.update(decision='abstain',reason='small_shape_identity_unresolved')
    else:out.update(decision='compatible',reason='small_hollow_source_composition_supported')
    return out


def guard(result,template,occlusion=None):
    from partial_swatch_detector import ink_membership
    center=(result.template_mask.shape[1]//2+result.aligned_x-result.x,
            result.template_mask.shape[0]//2+result.aligned_y-result.y)
    rec=evidence(ink_membership(result.plot_window,template.ink),template,center,result.scale,occlusion=occlusion)
    result.compute_diagnostics['small_hollow']=rec
    result.compute_diagnostics['pre_small_hollow_decision']=result.decision
    if rec['decision']=='conflict':result.decision='rejected'
    elif rec['decision']=='abstain' and result.decision=='verified':result.decision='ambiguous'
    return result

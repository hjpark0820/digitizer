"""Bounded source-scene retry for all-unanchored tiny hollow BW legends.

Not the default verifier: activated only after zero reliable anchors for every
legend identity and at least three jointly verified sites for a retry scale.
No reference labels, file names, or synthesized target ink are used.
"""
import cv2
import numpy as np
from bw_small_hollow_v46 import _bank, eligible

VERSION="small-hollow-scene-fallback-v1"


def needs_scene_retry(templates,reports):
    return bool(len(templates)>=2 and set(reports)=={t.key for t in templates}
                and all(eligible(t) for t in templates)
                and all(r.get('status')=='no_reliable_anchor_fallback_1x' for r in reports.values()))


def evidence(observed,template,center,scale=1.,available=None,occlusion=None,*,envelope=None):
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
    diameter=max(diameter,envelope or 0.)
    # Sample source pixels onto an integer-centred crop. Interpolation is for
    # alignment only, not an increase in independent sample count.
    size=2*int(np.ceil(diameter+5))+1;c=size//2
    yy,xx=np.indices((size,size),dtype=np.float32)
    xs=xx+float(center[0])-c;ys=yy+float(center[1])-c
    source=cv2.remap(observed,xs,ys,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    seen=cv2.remap(valid.astype(np.float32),xs,ys,cv2.INTER_LINEAR)>.99
    roi=(abs(xx-c)<=diameter/2+1)&(abs(yy-c)<=diameter/2+1)
    from bw_raster_context_v46 import external_strokes
    # Error bars often extend on ONE side only. Fit their actual outside
    # pixels; a line is an explanation for surplus ink, not required ink in
    # an opaque marker's white interior.
    measured,strokes=external_strokes(np.where(seen,source,0.),roi)
    base=np.clip(measured/max(float(source.max()),.25),0,1)
    lines=len(strokes)
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
        missing=np.maximum(marker_levels-source,0.)
        extra=np.maximum(source-marker_levels,0.)*(1-base)
        errors=np.mean((1.5*missing[:,:,roi]**2+.4*extra[:,:,roi]**2),axis=2)
        best=np.unravel_index(errors.argmin(),errors.shape)
        losses[key]=float(errors[best])
        expected=marker_levels[best]
        rim=(expected>.25)&roi
        missing_rims[key]=float(np.maximum(expected[rim]-source[rim],0).sum()/max(expected[rim].sum(),1.e-6))
        physical[key]=dict(
            missing_model_ink_mse=float(np.mean(missing[best][roi]**2)),
            extra_observed_ink_mse=float(np.mean(extra[best][roi]**2)),
            paper_on_rim=float(np.sum(expected[rim]*np.clip((.18-source[rim])/.18,0,1))/max(expected[rim].sum(),1.e-6)))
        filled=_bank(m['name'].removeprefix('open_'),*args)
        filled_pred=np.clip(filled[:,None]*np.array([.75,1.,1.25])[None,:,None,None],0,1)
        filled_losses[key]=float(np.mean((1.5*np.maximum(filled_pred[:,:,roi]-source[roi],0)**2+
             .4*(np.maximum(source[roi]-filled_pred[:,:,roi],0)*(1-base[roi]))**2),axis=2).min())
    loss=losses[template.key];line_loss=float(np.mean(.4*(source[roi]*(1-base[roi]))**2))
    rival=min([v for k,v in losses.items() if k!=template.key] or [1.])
    improvement=line_loss-loss;fill_margin=filled_losses[template.key]-loss
    out.update(loss=loss,model_losses=losses,line_only_loss=line_loss,marker_improvement=improvement,
               filled_loss=filled_losses[template.key],hollow_margin=fill_margin,rival_margin=rival-loss,
               missing_rim_fraction=missing_rims[template.key],physical_residual=physical[template.key],
               external_strokes=strokes,line_policy='outside_observed_extra_only')
    out['model_evidence']={k:dict(loss=v,missing=missing_rims[k],fill_margin=filled_losses[k]-v,
                                improvement=line_loss-v) for k,v in losses.items()}
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


def calibrate_joint(source,templates,valid,reports):
    """Common sites/envelope for identity AND scale; no reference coordinates.

    Existing scale proposals supply at most 40 locations. Each legend model
    may select its scale, but must beat the other identities at the same site.
    Fewer than three independent supporting sites leave native calibration.
    """
    eligible_templates=[t for t in templates if eligible(t)]
    if len(eligible_templates)<2 or len(eligible_templates)!=len(templates):return reports
    scales=tuple(round(.7+.05*i,3) for i in range(13))
    envelope=max(t.diameter for t in templates)*max(scales)
    sites=[]
    for p in sorted([p for r in reports.values() for p in r.get('candidates',[])],key=lambda p:-p.get('quality',0.)):
        if all(np.hypot(p['x']-q['x'],p['y']-q['y'])>max(3.,.4*envelope) for q in sites):sites.append(p)
        if len(sites)==40:break
    anchors={t.key:[] for t in templates};first=templates[0]
    for p in sites:
        rows=[(s,evidence(source,first,(p['x'],p['y']),s,valid,envelope=envelope)) for s in scales]
        models={k:min([dict(r['model_evidence'][k],scale=s) for s,r in rows if 'model_evidence' in r],
                      key=lambda r:r['loss'],default=None) for k in anchors}
        if any(v is None for v in models.values()):continue
        keys=sorted(models,key=lambda k:models[k]['loss']);key=keys[0];m=models[key]
        margin=models[keys[1]]['loss']-m['loss']
        if m['loss']<=.065 and m['missing']<=.25 and m['fill_margin']>=.008 and m['improvement']>=.012 and margin>=.004:
            anchors[key].append(dict(x=p['x'],y=p['y'],margin=margin,**m))
    for t in templates:
        aa=anchors[t.key]
        reports[t.key]['joint_scene_anchors']=aa
        if len(aa)>=3:
            previous=reports[t.key]['scale'];scale=float(np.median([p['scale'] for p in aa]))
            reports[t.key].update(scale=scale,status='joint_scene_verified',previous_scale=previous,
                reason='independent_common_site_hollow_identity_and_scale',joint_scene_envelope=envelope)
    return reports

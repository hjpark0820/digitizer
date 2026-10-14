"""Bounded, evidence-preserving refinement of an already proposed marker.

Neither a visible fragment nor a prior location is promoted by itself. Every
replacement is reverified at its new, locked center with the original window
rules; exported masks and direct-ink evidence therefore describe that center.
"""
from dataclasses import replace
import cv2
import numpy as np

VERSION = 'same-evidence-final-alignment-v1'


def locked_window(image, template, center, *, scale, aspect=1., occlusion_mask=None,
                  uncertainty_mask=None, colour_observation=None):
    """Re-evaluate one supplied geometry without silently moving its center."""
    import occlusion_aware_window_verifier as W
    x,y=map(float,center)
    prepared=list(W._prepare_uncertain_marker_window(image,template,x,y,
        search_scales=(scale,),lock_aspect=True,colour_observation=colour_observation))
    # A bounded alignment repair must not introduce another scale/aspect fit.
    prepared[1]=0
    prepared[-1]=((scale,aspect,*W._uncertain_shape(template,scale,aspect)),)
    prepared=tuple(prepared)
    _,_,size,_,membership,_,nearby,shapes=prepared
    other=None if occlusion_mask is None else W._crop_padded(occlusion_mask,(x,y),size)
    uncertainty=None if uncertainty_mask is None else W._crop_padded(uncertainty_mask,(x,y),size)
    shortlist=W._uncertain_cpu_shortlist(membership,nearby,shapes,size,0,
        occlusion=other,uncertainty=uncertainty)
    if not shortlist:return None
    stats=dict(backend='cpu_locked_final_alignment',used_cuda=False,center_locked=True)
    result=W._finalize_uncertain_marker_window(template,x,y,True,prepared,shortlist,stats,other)
    if other is not None:
        result.occluded_fraction=float(other[result.template_mask].mean()) if result.template_mask.any() else 0.
        result.visible_fraction=1.-result.occluded_fraction
        result.required_missing &= other<.5
        result.compute_diagnostics.update(explicit_colour_occlusion=True,independently_visible_ink_required=True)
    if getattr(template,'model_completed',False) and template.marker_kind=='open':
        from open_marker_guard_v46 import apply_guard
        result=apply_guard(result,template,other)
    return result


def visibility(image, template, center, *, scale, aspect, occlusion_mask,
               colour_observation=None):
    """Exactly the mask/denominator used by the pipeline's final colour gate."""
    import partial_swatch_detector as D
    from occlusion_aware_window_verifier import _uncertain_shape
    from bw_colour_visibility_v46 import measure
    x,y=map(float,center)
    mask,*_=_uncertain_shape(template,scale,aspect)
    proxy=replace(template,mask=mask,valid_weight=np.ones(mask.shape,np.float32),
        edge=cv2.Canny(mask.astype(np.uint8)*255,40,100)>0,diameter=template.diameter*scale)
    other=D._extract_aligned_patch(occlusion_mask,x,y,mask.shape)
    if colour_observation is None:
        patch=np.stack([D._extract_aligned_patch(image[:,:,c],x,y,mask.shape,fill=255) for c in range(3)],axis=-1)
        own=D.ink_membership(patch,template.ink);paper=None
    else:
        own=D._extract_aligned_patch(colour_observation.membership(template.ink),x,y,mask.shape)
        paper=D._extract_aligned_patch(colour_observation.paper,x,y,mask.shape)
    return measure(proxy,own,other,paper=paper)


def preserve_visible_seed(image,template,seed,result,*,occlusion_mask,
                          uncertainty_mask=None,colour_observation=None,assess=None,
                          verification_image=None):
    """Retry the existing seed only when alignment erased its colour evidence."""
    if result.decision not in ('verified','ambiguous') or not template.ink.achromatic:
        return result
    aligned=(float(result.aligned_x),float(result.aligned_y));seed=tuple(map(float,seed))
    if np.hypot(aligned[0]-seed[0],aligned[1]-seed[1])<.5:return result
    opts=dict(scale=result.scale,aspect=result.aspect_ratio,occlusion_mask=occlusion_mask,
              colour_observation=colour_observation)
    current=visibility(image,template,aligned,**opts)
    if current['candidate_supported'] and current['identity_observable']:return result
    prior=visibility(image,template,seed,**opts)
    if not (prior['candidate_supported'] and prior['identity_observable']):return result
    trial=locked_window(image if verification_image is None else verification_image,
        template,seed,uncertainty_mask=uncertainty_mask,**opts)
    if trial is not None and assess is not None:assess(template,trial)
    report=dict(version=VERSION,reason='alignment_lost_observable_source_colour',
        original_center=list(aligned),seed_center=list(seed),original_visibility=current,
        seed_visibility=prior,adopted=False,reference_points_read=False)
    if trial is not None and trial.decision=='verified':
        # Recheck the returned geometry, not a different pre-search mask.
        final=visibility(image,template,(trial.aligned_x,trial.aligned_y),
            scale=trial.scale,aspect=trial.aspect_ratio,occlusion_mask=occlusion_mask,
            colour_observation=colour_observation)
        if final['candidate_supported'] and final['identity_observable']:
            report.update(adopted=True,final_visibility=final)
            trial.compute_diagnostics['final_alignment']=report
            return trial
    result.compute_diagnostics['final_alignment']=report
    return result


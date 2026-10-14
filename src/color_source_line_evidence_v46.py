"""Source-colour body check against an exterior-supported straight stroke.

This is a conservative negative guard for recovered legend rasters, not a
marker proposal generator. The line is measured outside the body; unknown
colour cannot become positive evidence or silently behave as white paper.
"""
from functools import lru_cache
import cv2
import numpy as np
import partial_swatch_detector as D

VERSION='source-colour-independent-body-v1'


@lru_cache(maxsize=4)
def _line_bank(radius,diameter_half):
    size=2*radius+1;yy,xx=np.indices((size,size),dtype=np.float32)
    xx-=radius;yy-=radius
    lines=[];parameters=[]
    for angle in range(0,180,5):
        rad=np.deg2rad(angle);perp=-xx*np.sin(rad)+yy*np.cos(rad)
        for width in np.arange(1.,min(9.,diameter_half+1.),.5):
            for offset in (-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.):
                lines.append(np.clip(.5+(width/2-np.abs(perp-offset))/.8,0,1))
                parameters.append(dict(angle_degrees=angle,width_px=float(width),offset_px=offset))
    return np.asarray(lines,np.float32),parameters


def body_line_evidence(image,template,x,y,scale=1.,occlusion_mask=None):
    """Return a line-only veto only when source evidence independently proves it."""
    result=dict(version=VERSION,applicable=False,supported=True,reason='not_source_colour_recovery')
    if not getattr(template,'source_colour_body',None):return result
    diameter=float(template.diameter)*float(scale)
    if not 4.<=diameter<=32.:
        return dict(result,reason='unsupported_body_size')
    radius=int(np.ceil(diameter));size=2*radius+1
    own=D.ink_membership(image,template.ink)
    patch=D._extract_aligned_patch(own,x,y,(size,size))
    other=np.zeros((size,size),np.float32) if occlusion_mask is None else D._extract_aligned_patch(occlusion_mask,x,y,(size,size))
    valid=np.clip(1-other,0,1)
    yy,xx=np.indices((size,size),dtype=np.float32);xx-=radius;yy-=radius
    radial=np.hypot(xx,yy)
    body=radial<=.55*diameter
    context=(radial>=.65*diameter)&(radial<=diameter)
    context_weight=context*valid
    body_weight=body*valid
    visible=float(body_weight.sum()/max(body.sum(),1))
    context_mass=float((context_weight*patch).sum())
    result.update(applicable=True,diameter=diameter,visible_fraction=visible,context_ink_mass=context_mass)
    if visible<.75 or context_mass<3.:
        return dict(result,reason='insufficient_unoccluded_context')
    lines,parameters=_line_bank(radius,round(diameter/2,1))
    den=np.sum(lines*lines*context_weight,axis=(1,2))
    amp=np.clip(np.sum(lines*patch*context_weight,axis=(1,2))/np.maximum(den,1.e-6),0.,1.)
    predicted=lines*amp[:,None,None]
    context_energy=max(float(np.sum(context_weight*patch*patch)),1.e-6)
    losses=np.sum(context_weight*(patch-predicted)**2,axis=(1,2))/context_energy
    best=int(np.argmin(losses));line=predicted[best]
    # Refine the stroke's transverse gray profile from exterior pixels only.
    # The source may have a broad antialiased/JPEG edge, not our bank's ideal
    # 0.8px transition. Those ordinary edge pixels must not masquerade as a
    # new marker body. No pixel inside the body contributes to this profile.
    best_context=float(losses[best]);line_fit=dict(parameters[best],amplitude=float(amp[best]))
    for angle in np.arange(parameters[best]['angle_degrees']-5,parameters[best]['angle_degrees']+5.1,1.):
        rad=np.deg2rad(angle);perp=-xx*np.sin(rad)+yy*np.cos(rad)
        phase=(perp+radius)*2
        bins=np.rint(phase).astype(int)
        count=4*radius+1;ok=(bins>=0)&(bins<count)
        w=context_weight*ok
        totals=np.bincount(bins[ok].ravel(),weights=w[ok].ravel(),minlength=count)
        sums=np.bincount(bins[ok].ravel(),weights=(w*patch)[ok].ravel(),minlength=count)
        profile=np.divide(sums,totals,out=np.zeros_like(sums),where=totals>=2.)
        # Only the one thin stroke selected by the coarse context fit is a
        # nuisance model; remote parallel curves cannot fill the body.
        coords=np.arange(count)/2-radius
        within=np.abs(coords-parameters[best]['offset_px'])<=parameters[best]['width_px']/2+1.25
        profile*=within
        observed_line=np.interp(phase,np.arange(count),profile,left=0.,right=0.).astype(np.float32)
        loss=float(np.sum(context_weight*(patch-observed_line)**2)/context_energy)
        if loss<best_context:
            best_context=loss;line=observed_line
            line_fit=dict(parameters[best],angle_degrees=float(angle),profile='exterior_observed_grayscale',
                profile_step_px=.5,profile_ink=profile.tolist())
    body_energy=max(float(np.sum(body_weight*patch*patch)),1.e-6)
    body_loss=float(np.sum(body_weight*(patch-line)**2)/body_energy)
    # Positive residual is observed own-colour body beyond the independently
    # measured stroke. Foreign ink supplies neither credit nor white penalty.
    independent=float(np.sum(body_weight*np.maximum(patch-line,0))/max(float(np.sum(body_weight*patch)),1.e-6))
    supported=not (best_context<=.20 and body_loss<=.12 and independent<=.12)
    result.update(supported=bool(supported),reason='independent_body_or_uncertain_line' if supported else 'source_patch_explained_by_exterior_line',
        context_line_loss=best_context,body_line_loss=body_loss,independent_body_fraction=independent,
        line_fit=line_fit)
    return result

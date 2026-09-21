"""Negative evidence for filled colour glyphs: paper != unclassified ink.

Raw hue separation can excuse foreign coloured ink even if palette matching
failed. It never supplies target ink, changes a template, or shrinks the
positive-support denominator. All thresholds are explicit image heuristics.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from color_paper_contradiction_v46 import filled_eligibility

VERSION = 'raw_paper_window_contradiction_v1'


@dataclass(frozen=True)
class Config:
    paper_low: float = 225.
    paper_high: float = 245.
    paper_weight: float = 2.
    unresolved_ink_weight: float = .15
    hue_low_degrees: float = 20.
    hue_high_degrees: float = 45.
    saturation_low: float = .12
    saturation_high: float = .30
    noise_area_fraction: float = .005


def evaluate(raw_bgr, template, marker, observed, other, available, *, prior=None,
             native_pixel_area=1., config=Config(),target_confidence=None,legacy_missing_weight=None):
    """Return local negative-evidence maps, or abstain for unsupported glyphs."""
    eligible, reason = filled_eligibility(template, prior)
    report = dict(version=VERSION, applied=False, reason=reason,
        configuration=asdict(config), other_ink_is_positive_evidence=False,
        positive_denominator_changed=False, source_pixels_changed=False)
    if not eligible:
        return {}, report
    a=np.asarray(marker,np.float32); raw=np.asarray(raw_bgr,np.float32)
    if raw.shape != (*a.shape,3) or not np.isfinite(raw).all() or np.any((raw<0)|(raw>255)):
        raise ValueError('Raw window pixels must be finite BGR 0..255 matching the marker canvas')
    if (not np.isfinite(native_pixel_area) or native_pixel_area<=0 or
            config.paper_high<=config.paper_low or config.paper_weight<0 or
            not 0<=config.unresolved_ink_weight<=1 or
            config.hue_high_degrees<=config.hue_low_degrees or
            config.saturation_high<=config.saturation_low):
        raise ValueError('Invalid physical window configuration')
    valid=np.clip(np.broadcast_to(np.asarray(available,np.float32),a.shape),0,1)
    other=np.clip(np.asarray(other,np.float32),0,1)
    paper=np.clip((raw.min(axis=2)-config.paper_low)/(config.paper_high-config.paper_low),0,1)
    # HSV hue is insensitive to darkening, unlike matching a single RGB ink
    # against white. Achromatic/very dark pixels do not establish foreign hue.
    hsv=cv2.cvtColor(raw/255.,cv2.COLOR_BGR2HSV)
    model=template.get('model',{}).get('bgr')
    if model is None and 'rgb' in template:
        model=np.asarray(template['rgb'])[::-1]
    foreign=np.zeros_like(a)
    if model is not None:
        target=cv2.cvtColor(np.asarray(model,np.float32).reshape(1,1,3)/255.,cv2.COLOR_BGR2HSV)[0,0]
        if target[1]>=config.saturation_high:
            delta=np.abs(hsv[...,0]-target[0]);delta=np.minimum(delta,360-delta)
            foreign=(np.clip((delta-config.hue_low_degrees)/(config.hue_high_degrees-config.hue_low_degrees),0,1)*
                np.clip((hsv[...,1]-config.saturation_low)/(config.saturation_high-config.saturation_low),0,1)*
                np.clip((raw.max(axis=2)-12)/20.,0,1))
    # Paper itself overrides contradictory occlusion flags. Foreign ink only
    # waives absence; observed/positive target membership is never replaced.
    occluder=np.maximum(other,foreign)*(1-paper)
    # Only uncertain colour identity earns the softer deficit. A confidently
    # modelled but faint target-colour stroke retains the original shape test.
    confidence=np.zeros_like(a) if target_confidence is None else np.clip(np.asarray(target_confidence,np.float32),0,1)
    if confidence.shape!=a.shape or not np.isfinite(confidence).all():
        raise ValueError('Target-colour confidence must match the local glyph canvas')
    ordinary=1. if legacy_missing_weight is None else np.clip(np.asarray(legacy_missing_weight,np.float32),0,1)
    unresolved_weight=config.unresolved_ink_weight+(1-config.unresolved_ink_weight)*confidence
    weight=paper+(1-paper)*np.minimum(ordinary,unresolved_weight)
    mass=max(float(a.sum()),1.)
    noise=max(float(native_pixel_area),config.noise_area_fraction*float(template['diameter'])**2)
    contradiction=a*paper*valid
    amount=float(contradiction.sum())
    cost=max(0.,amount-noise)/mass
    report.update(applied=True,reason='filled_raw_paper_vs_coloured_ink',
        expected_mass=mass,noise_budget=noise,paper_mass=amount,paper_cost=cost,
        paper_penalty=config.paper_weight*cost,
        foreign_hue_fraction=float((a*foreign*valid).sum())/mass,
        target_support_unchanged=float((np.minimum(observed,a)*valid).sum())/mass)
    return dict(paper=paper,foreign_hue=foreign,other=occluder,
        missing_weight=weight,paper_contradiction=contradiction,target_confidence=confidence),report

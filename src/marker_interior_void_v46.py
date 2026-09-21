"""A non-compensable ceiling for missing ink inside a reliably filled glyph.

The stable core excludes every point near the template boundary, approximating
the intersection of the glyph under a small translation uncertainty. Connected
missing core ink is a contradiction, even when unrelated strokes match elsewhere.
With raw paper evidence, only actual white source pixels form this hard ceiling;
target-colour uncertainty is not a physical hole. The old membership-only API
remains available without raw pixels. Ignored/out-of-image pixels are untestable.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

VERSION = 'stable_filled_core_void_v2_blend_weighted'
RAW_PAPER_VERSION = 'stable_filled_core_void_v3_raw_paper'


@dataclass(frozen=True)
class Config:
    boundary_uncertainty_pixels: float = 1.25
    boundary_uncertainty_fraction: float = .06
    minimum_expected_ink: float = .80
    minimum_filled_fraction: float = .85
    ink_allowance: float = .12
    component_deficit: float = .50
    minimum_core_pixels: int = 4
    # Ignore a single native-pixel defect; scale this area at higher resolution.
    noise_area_fraction: float = .005
    penalty_strength: float = 3.


def inspect_interior(marker, observed, other, available, diameter, *,
                     hole=None, uncertain=None, config=Config(), debug=False, missing_weight=None,
                     paper_probability=None, native_pixel_area=1.):
    marker=np.clip(np.asarray(marker,np.float32),0,1)
    observed=np.clip(np.asarray(observed,np.float32),0,1)
    other=np.clip(np.asarray(other,np.float32),0,1)
    available=np.broadcast_to(np.asarray(available,np.float32),marker.shape)
    if observed.shape!=marker.shape or other.shape!=marker.shape:
        raise ValueError('Interior evidence maps must have the same shape')
    if not np.isfinite(native_pixel_area) or native_pixel_area<=0:
        raise ValueError('Native pixel area must be positive and finite')
    if paper_probability is not None:
        paper_probability=np.asarray(paper_probability,np.float32)
        if (paper_probability.shape!=marker.shape or not np.isfinite(paper_probability).all() or
                np.any((paper_probability<0)|(paper_probability>1))):
            raise ValueError('Raw paper probability must be a same-sized finite 0..1 array')
    support=(marker>=.5).astype(np.uint8)
    depth=cv2.distanceTransform(support,cv2.DIST_L2,5)-.5
    margin=max(config.boundary_uncertainty_pixels,config.boundary_uncertainty_fraction*diameter)
    zero=np.zeros_like(marker)
    result=dict(version=RAW_PAPER_VERSION if paper_probability is not None else VERSION,
        applied=False,score_ceiling=1.,reason='insufficient_filled_core',
        boundary_margin_pixels=float(margin),core_pixels=0,visible_core_mass=0.,
        largest_missing_component_mass=0.,noise_area_budget=max(native_pixel_area,config.noise_area_fraction*diameter**2),
        source_pixels_changed=False,other_ink_is_positive_evidence=False,
        depends_on_connector_or_t=False)
    result['negative_evidence_basis']='raw_white_paper' if paper_probability is not None else 'target_colour_deficit_legacy'
    if debug:result['maps']=dict(core=zero.copy(),missing=zero.copy(),components=zero.copy(),depth=depth)
    # Never reinterpret a legitimately open/partially filled legend as filled.
    if hole is not None and np.any(hole):
        result['reason']='open_template';return result
    yy,xx=np.nonzero(support)
    if len(xx)<3:return result
    hull=np.zeros_like(support)
    cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)
    fill=float(support.sum()/max(1,hull.sum()))
    result['filled_fraction']=fill
    if fill<config.minimum_filled_fraction:
        result['reason']='not_a_reliably_filled_template';return result
    core=(depth>margin)&(marker>=config.minimum_expected_ink)
    if uncertain is not None:core &= ~np.asarray(uncertain,bool)
    visible=np.clip(available,0,1)*(1-other if paper_probability is None else 1.)
    core_mass=float((marker*core*visible).sum())
    result.update(core_pixels=int(core.sum()),visible_core_mass=core_mass)
    if core.sum()<config.minimum_core_pixels or core_mass<2.:
        return result
    required=np.maximum(marker-config.ink_allowance,0)
    deficit=np.divide(np.maximum(required-observed,0),required,
        out=np.zeros_like(marker),where=required>0)
    weight = 1. if missing_weight is None else np.clip(np.asarray(missing_weight, np.float32), .35, 1.)
    negative=(deficit*visible*core*weight if paper_probability is None else
              paper_probability*visible*core)
    # Threshold only groups connected, substantial contradictions. The actual
    # evidence/penalty remains continuous; there is no single-pixel veto.
    n,labels=cv2.connectedComponents((negative>=config.component_deficit).astype(np.uint8),connectivity=8)
    component_map=np.zeros_like(marker);masses=[]
    for label in range(1,n):
        region=labels==label
        mass=float((negative*marker*region).sum())
        masses.append(mass);component_map[region]=negative[region]
    largest=max(masses,default=0.)
    budget=result['noise_area_budget']
    severity=max(0.,largest-budget)/budget
    ceiling=math.exp(-config.penalty_strength*severity)
    result.update(applied=True,score_ceiling=float(ceiling),component_count=len(masses),
        component_masses=sorted(masses,reverse=True),largest_missing_component_mass=largest,
        core_missing_fraction=float((negative*marker).sum()/max(core_mass,1e-6)),
        reason='unexplained_connected_interior_void' if ceiling<.999 else 'no_significant_interior_void')
    if debug:result['maps']=dict(core=(core*visible).astype(np.float32),missing=negative,
        components=component_map,depth=depth)
    return result

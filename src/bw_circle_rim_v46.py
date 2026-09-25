"""Direct paper contradiction on an aligned circular stroke, not its white hole.

Neighbour ink cannot erase this loss. Confidence and expected antialias coverage
discount raster fringes; explicit other-colour occlusion removes constraints but
never supplies positive ink. No source pixels or marker size are changed.
"""
import numpy as np

VERSION='circle-direct-rim-v2'


def unmeasurable_core_supported(rim):
    """Absent core samples are not missing ink. Require direct, distributed rim.

    A contrast/model mismatch may lower recall despite visible gray strokes;
    actual paper or an absent arc still prevents this route. Hole and round
    shape guards run separately and retain their vetoes.
    """
    return bool(rim.get('core_pixels')==0 and rim.get('visible_fraction',0.)>=.85
                and rim.get('direct_rim_recall',0.)>=.72
                and rim.get('paper_loss',1.)<=.05
                and rim.get('missing_arc_fraction',1.)<=1/12.)


def rim_evidence(observed,mask,weight,expected,core,occlusion=None):
    observed=np.asarray(observed,np.float32)
    visible=np.ones_like(observed) if occlusion is None else 1.-np.clip(occlusion,0,1)
    mass=np.asarray(weight,np.float32)*np.asarray(expected,np.float32)*mask
    total=max(float(mass.sum()),1.e-6)
    use=mass*visible
    seen=float(use.sum())/total
    direct=np.minimum(observed/np.maximum(expected,1.e-6),1.)
    direct_recall=float((direct*use).sum()/max(float(use.sum()),1.e-6))
    # Gradual paper likelihood: never call pale antialiased ink fully absent.
    paper=np.clip((.18-observed)/.18,0.,1.)
    loss=float((paper*use).sum()/max(float(use.sum()),1.e-6))
    yy,xx=np.indices(mask.shape)
    sectors=np.floor((np.arctan2(yy-(mask.shape[0]-1)/2,xx-(mask.shape[1]-1)/2)+np.pi)*6/np.pi).astype(int)%12
    gaps=[]
    for i in range(12):
        select=mask&(sectors==i);den=float(mass[select].sum())
        seen_mass=float(use[select].sum())
        gaps.append(float((paper*use)[select].sum()/max(seen_mass,1.e-6))
                    if den>.15 and seen_mass>=.5*den else None)
    missing=[g is not None and g>.5 for g in gaps]
    longest=run=0
    for m in missing*2:
        run=run+1 if m else 0;longest=max(longest,min(run,12))
    contiguous=longest/12.
    measured=bool(np.any(core&(visible>=.5)))
    return dict(version=VERSION,paper_loss=loss,missing_arc_fraction=contiguous,
        penalty=loss+.15*contiguous,direct_rim_recall=direct_recall,
        visible_fraction=seen,sector_paper_loss=gaps,core_pixels=int((core&(visible>=.5)).sum()),
        measured_core_recall=(float(direct[core&(visible>=.5)].mean()) if measured else None),
        core_status='measured' if measured else 'unmeasurable_direct_rim_fallback',
        observed_ink_added=False,neighbour_credit_can_cancel_paper=False)

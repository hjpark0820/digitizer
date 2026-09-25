"""Bounded marker-existence evidence, not a calibrated class probability.

Compare one marker+lines model with a line-only null on the SAME pixels.
Spatial neighbours and the number of candidates never enter this score.
Selection/deletion policy remains outside this module.
"""
from copy import deepcopy
import math

VERSION='marker_vs_line_confidence_v1'


def assess(probes, *, occluded_fraction=0., minimum_improvement=.04,
           maximum_loss=.55, minimum_independent_fraction=.20, minimum_visible=.55):
    """Use one coherent subpixel probe, never maxima from different probes.

    C = independent_visible * fit_quality * sigmoid(marker_gain / margin).
    fit_quality is one within the existing acceptable loss and decays beyond
    it. The existing comparison margin sets the transition width, not labels
    from a particular chart. Unobservable/invalid measurements have score=None.
    """
    if not all(math.isfinite(float(v)) and v>0 for v in
               (minimum_improvement,maximum_loss,minimum_independent_fraction,minimum_visible)):
        raise ValueError('Marker confidence scales must be positive and finite')
    base=dict(version=VERSION,score=None,status='unmeasured',calibrated_probability=False,
        candidate_density_used=False,minimum_improvement=minimum_improvement,
        maximum_loss=maximum_loss,minimum_independent_fraction=minimum_independent_fraction,
        minimum_visible=minimum_visible)
    if not math.isfinite(float(occluded_fraction)) or not 0<=occluded_fraction<=1:
        return dict(base,status='invalid_observation')
    if occluded_fraction>=.35:return dict(base,status='occluded_unobservable')
    rows=[]
    for i,s in enumerate(probes):
        try:
            loss=float(s['geometry_loss']);gain=float(s['line_improvement'])
            visible=float(s['independent_visible']);fraction=float(s['independent_fraction'])
        except (KeyError,TypeError,ValueError,OverflowError):continue
        if not all(math.isfinite(v) for v in (loss,gain,visible,fraction)):continue
        if loss<0 or not (0<=visible<=1 and 0<=fraction<=1):continue
        if fraction<minimum_independent_fraction:continue
        advantage=1./(1.+math.exp(-max(-60.,min(60.,gain/minimum_improvement))))
        fit_quality=1./(1.+max(0.,loss-maximum_loss)/maximum_loss)
        score=visible*fit_quality*advantage
        rows.append(dict(probe_index=i,score=score,geometry_loss=loss,
            line_improvement=gain,independent_visible=visible,independent_fraction=fraction,
            fit_quality=fit_quality,line_advantage_factor=advantage))
    if not rows:return dict(base,status='insufficient_independent_observation')
    selected=max(rows,key=lambda r:r['score'])
    gains=[r['line_improvement'] for r in rows]
    # A missing probe cannot establish a robust line-only contradiction.
    contradiction=len(rows)==len(probes) and max(gains)<-.5*minimum_improvement
    status=('line_explained' if contradiction else 'marker_supported'
            if selected['line_improvement']>=minimum_improvement and selected['geometry_loss']<=maximum_loss
            and selected['independent_visible']>=minimum_visible
            else 'ambiguous')
    return dict(base,status=status,score=selected['score'],selected=selected,
        minimum_marker_gain=min(gains),maximum_marker_gain=max(gains),
        observable_probe_count=len(rows),probe_count=len(probes),probes=rows)


def update_point(point, assessment):
    """Attach a measured confidence without repeatedly discounting old scores.

    Legacy confidence is retained as provenance, not as a factor in C. Unknown
    observations keep their numeric prior for old consumers, explicitly marked
    unmeasured; new callers should read marker_confidence.score/status together.
    Manual coordinates/scores are never overwritten by this adapter.
    """
    out=deepcopy(point)
    if assessment is None or point.get('manual_edit'):return out
    if 'detector_confidence' not in out:
        out['detector_confidence']=out.get('prior_detector_confidence',out.get('confidence'))
        out['detector_confidence_kind']=out.get('confidence_kind','legacy_detector_score')
    out['marker_confidence']=deepcopy(assessment)
    score=assessment.get('score')
    if score is None:
        out['confidence_kind']='unmeasured_preserved_prior'
    else:
        if not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Marker confidence must be finite and in [0,1]')
        out.update(confidence=float(score),confidence_kind=VERSION)
    return out

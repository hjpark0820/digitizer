"""Source-only automatic duplicate-square legend identity resolver.

No case IDs, label files, target lists, plot points, or uniqueness constraint are
used. Duplicate automatic filled_square sources trigger continuous grayscale
comparison. A robust circle can change only when a separately robust square
peer corroborates genuinely different, tone-normalized geometry. The peer is a
safety check, not a substitute for positive evidence in the proposed circle.
Repeated true shapes, including different tones, are valid unchanged outcomes.

Public helper: resolve_legend_sources(image_bgr, templates, reports)
    -> (deep-copied templates, deep-copied reports, JSON-friendly plan).
Call after native legend extraction and before geometry completion. This pure
resolver never patches global functions and preserves original source arrays.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from bw_legend_gray_identity_v46 import classify as gray_classify, METHOD_VERSION as GRAY_VERSION
from bw_fill_identity_v46 import describe_fill


VERSION = "joint-gray-duplicate-square-v1-frozen-2026-10-03"
RULES = dict(
    trigger_class="filled_square", minimum_duplicate_count=2,
    diameter_factors=[.90, 1., 1.10], minimum_agreeing_runs=2,
    opposing_accepted_cue_veto=True,
    minimum_peer_corner_ratio_gap=.20,
    minimum_peer_corner_interval_gap=.10,
    minimum_peer_annular_margin_gap=.04,
    maximum_peer_score_bonus=.10,
    existing_circle_overlap_gate=.75,
    weak_evidence_policy="retain original class and raster; diagnostic abstention",
    peer_role="corroboration/safety only; never rescues weak own evidence or forces uniqueness",
)
ARRAY_FIELDS = ("raw_soft", "raw_mask", "line_nuisance", "valid_weight", "soft", "mask",
                "edge", "orientation", "required_weight", "source_gray", "observed_source_soft")


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if is_dataclass(value):return _json(asdict(value))
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value


def _hash(array):
    if array is None:return None
    a=np.ascontiguousarray(array);h=hashlib.sha256()
    h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def _snapshot(template):
    return dict(arrays={key:_hash(getattr(template,key,None)) for key in ARRAY_FIELDS},
        marker_center=_json(template.marker_center),diameter=float(template.diameter),
        swatch_box=_json(template.swatch_box),marker_kind=template.marker_kind,
        interior_fill=float(template.interior_fill),model_completed=bool(template.model_completed),
        proposal_scale=float(template.proposal_scale),ink=_json(template.ink),
        matching_profile=template.matching_profile,swatch_id=template.swatch_id)


def _eligible(template,report):
    return (template.name==RULES['trigger_class'] and bool(template.ink.achromatic)
        and not template.model_completed and report.get('classification')!='supplied')


def _source_patch(image,template,report):
    """Observed RGB crop plus native connector rows, never a completed raster."""
    cx,cy=map(float,template.marker_center);diameter=float(template.diameter)
    radius=int(math.ceil(1.25*max(RULES['diameter_factors'])*diameter+1.))
    x0,y0=int(math.floor(cx))-radius,int(math.floor(cy))-radius
    x1,y1=int(math.floor(cx))+radius+1,int(math.floor(cy))+radius+1
    ih,iw=image.shape[:2]
    xa,ya,xb,yb=max(0,x0),max(0,y0),min(iw,x1),min(ih,y1)
    patch=np.full((y1-y0,x1-x0,3),255,np.uint8)
    valid=np.zeros(patch.shape[:2],bool)
    if xa<xb and ya<yb:
        patch[ya-y0:yb-y0,xa-x0:xb-x0]=image[ya:yb,xa:xb,::-1]
        valid[ya-y0:yb-y0,xa-x0:xb-x0]=True
    nuisance=np.asarray(template.line_nuisance,bool)
    native_rows=np.flatnonzero(nuisance.any(axis=1)).astype(float)
    glyph=report.get('glyph_box')
    aligned=not native_rows.size
    row_y=np.array([],float)
    origin=None
    if glyph is not None and len(glyph)==4 and all(np.isfinite(glyph)):
        gx,gy,gr,gb=map(float,glyph)
        gh,gw=gb-gy,gr-gx
        integer_box=all(abs(v-round(v))<1e-6 for v in glyph)
        center_matches=abs(cx-(gx+gr-1)/2)<1e-6 and abs(cy-(gy+gb-1)/2)<1e-6
        if (integer_box and center_matches and 0<gh<=nuisance.shape[0] and 0<gw<=nuisance.shape[1]):
            # _model uses floor padding. An even glyph on an odd template
            # canvas is NOT centered at (canvas_size-1)/2: the offset is .5px.
            pad_top=(nuisance.shape[0]-int(round(gh)))//2
            row_y=native_rows+gy-pad_top-y0
            origin=dict(glyph_box=list(glyph),template_pad_top=pad_top,
                        template_row_zero_global_y=gy-pad_top)
            aligned=True
    context=dict(valid_mask=valid,paper_gray=float(template.ink.paper_gray),connector_rows=row_y.tolist())
    meta=dict(crop_box=[x0,y0,x1,y1],clipped_image_box=[xa,ya,xb,yb],
        center_local=[cx-x0,cy-y0],diameter=diameter,native_connector_rows_local=row_y.tolist(),
        nuisance_provenance="original template.line_nuisance translated by exact glyph_box plus floor padding; no GT",
        native_connector_alignment_valid=bool(aligned),native_connector_alignment=origin,
        image_rgb_crop_hash=_hash(patch),valid_fraction=float(valid.mean()))
    return patch,(cx-x0,cy-y0),context,meta


def _finite_values(runs,key):
    return [float(r[key]) for r in runs if r.get(key) is not None and np.isfinite(r[key])]


def _evaluate_source(image,template,report):
    patch,center,context,meta=_source_patch(image,template,report)
    runs=[]
    for factor in RULES['diameter_factors']:
        result=gray_classify(patch,center,float(template.diameter)*factor,context)
        annular,corner=result['annular'],result['corner']
        fits=annular.get('features',{}).get('model_fits',{})
        margin=(float(fits['square']['relative_rmse']-fits['circle']['relative_rmse'])
                if 'circle' in fits and 'square' in fits else None)
        ratio=corner.get('features',{}).get('corner_to_axis_ratio')
        agreement=annular['decision'] if annular['decision']==corner['decision'] else 'unknown'
        runs.append(dict(diameter_factor=factor,diameter=float(template.diameter)*factor,
            annular_decision=annular['decision'],corner_decision=corner['decision'],
            agreeing_decision=agreement,annular_margin_circle_positive=margin,
            corner_to_axis_ratio=ratio,full_measurements=result))
    counts=Counter(r['agreeing_decision'] for r in runs)
    accepted_cues={r[key] for r in runs for key in ('annular_decision','corner_decision')
                   if r[key] in ('circle','square')}
    independent='unknown';reason='insufficient_own_cue_agreement_across_size_search'
    for family in ('circle','square'):
        opposite='square' if family=='circle' else 'circle'
        if counts[family]>=RULES['minimum_agreeing_runs'] and opposite not in accepted_cues:
            independent=family;reason='robust_own_annular_and_corner_agreement';break
    if accepted_cues=={'circle','square'}:reason='opposing_accepted_cues_veto'
    if not meta['native_connector_alignment_valid']:
        independent='unknown';reason='native_connector_alignment_unavailable'
    ratios=_finite_values(runs,'corner_to_axis_ratio')
    margins=_finite_values(runs,'annular_margin_circle_positive')
    median_ratio=None if not ratios else float(np.median(ratios))
    median_margin=None if not margins else float(np.median(margins))
    # This continuous value is descriptive. Positive means circle-like; the
    # hard own-source agreement above remains mandatory regardless of peers.
    signed_score=(0. if median_ratio is None or median_margin is None else float(.5*(
        np.clip((.61-median_ratio)/.11,-1.,1.)+np.clip(median_margin/.075,-1.,1.))))
    return dict(swatch_id=template.key,original_class=template.name,eligible=True,
        independent_gray_decision=independent,independent_reason=reason,
        own_agreeing_runs=dict(counts),own_signed_score_circle_positive=signed_score,
        median_corner_to_axis_ratio=median_ratio,
        corner_ratio_interval=None if not ratios else [min(ratios),max(ratios)],
        median_annular_margin_circle_positive=median_margin,
        patch=meta,runs=runs,peer_candidates=[],joint_final_decision='keep_original',
        joint_resolution='abstained',applied_class=template.name,identity_changed=False)


def _peer_evidence(own,peer):
    ratio_gap=interval_gap=radial_gap=None
    if own['median_corner_to_axis_ratio'] is not None and peer['median_corner_to_axis_ratio'] is not None:
        ratio_gap=peer['median_corner_to_axis_ratio']-own['median_corner_to_axis_ratio']
        interval_gap=peer['corner_ratio_interval'][0]-own['corner_ratio_interval'][1]
    if own['median_annular_margin_circle_positive'] is not None and peer['median_annular_margin_circle_positive'] is not None:
        radial_gap=own['median_annular_margin_circle_positive']-peer['median_annular_margin_circle_positive']
    distinct=(ratio_gap is not None and interval_gap is not None and radial_gap is not None
        and ratio_gap>=RULES['minimum_peer_corner_ratio_gap']
        and interval_gap>=RULES['minimum_peer_corner_interval_gap']
        and radial_gap>=RULES['minimum_peer_annular_margin_gap'])
    square=peer['independent_gray_decision']=='square'
    strength=float(max(0.,-peer['own_signed_score_circle_positive'])) if square and distinct else 0.
    return dict(swatch_id=peer['swatch_id'],independent_gray_decision=peer['independent_gray_decision'],
        peer_corner_to_axis_ratio=peer['median_corner_to_axis_ratio'],
        peer_annular_margin_circle_positive=peer['median_annular_margin_circle_positive'],
        corner_ratio_gap_peer_minus_own=ratio_gap,corner_interval_gap_peer_min_minus_own_max=interval_gap,
        annular_margin_gap_own_minus_peer=radial_gap,
        different_original_grayscale_geometry=bool(distinct),robust_square_peer=square,
        eligible_corroborating_peer=bool(square and distinct),soft_support=strength,
        reason='distinct_robust_square_peer' if square and distinct else
            'peer_not_robust_square' if not square else 'insufficient_geometry_separation')


def _apply_circle(template,report,decision):
    before=_snapshot(template)
    original_shape=deepcopy(report.get('shape_evidence',{}));scores=original_shape.get('shape_scores',{})
    circle_score=float(scores['circle'])
    template.name=template.shape_hint='filled_circle'
    working=deepcopy(original_shape)
    for field in ('polygon_consensus','corroborated_shape','outline_identity','shape_override_reason','shape_abstention_reason'):
        working.pop(field,None)
    rivals=[key for key in scores if key!='circle'];runner=max(rivals,key=scores.get) if rivals else None
    marker=dict(version=VERSION,automatic=True,manual_oracle=False,
        source_only=True,plot_reference_points_read=False,from_class='filled_square',to_class='filled_circle',
        peer_ids=[p['swatch_id'] for p in decision['peer_candidates'] if p['eligible_corroborating_peer']],
        peer_role=RULES['peer_role'],source_pixels_changed=False,source_geometry_changed=False)
    working.update(method='joint_grayscale_duplicate_square_source_resolution',status='classified',
        best_shape='circle',shape_hint='filled_circle',shape_confidence=circle_score,
        runner_up=runner,shape_margin=None if runner is None else circle_score-float(scores[runner]),
        identity_source='robust_own_grayscale_with_distinct_square_peer',automatic=True,
        manual_oracle=False,confidence_provenance='unchanged original binary circle overlap; not grayscale confidence or probability')
    report.update(class_name='filled_circle',shape_hint='filled_circle',
        classification='automatic_joint_gray',
        joint_identity_original_shape_evidence=original_shape,
        joint_identity_original_classification=report.get('classification'),
        shape_evidence=working,joint_identity_resolution=marker)
    report['joint_identity_original_fill_evidence']=deepcopy(report.get('fill_evidence'))
    report['fill_evidence']=describe_fill(template,working)
    template.joint_identity_resolution=deepcopy(marker)
    after=_snapshot(template)
    if before!=after:raise AssertionError('Gray identity intervention changed original source raster or geometry')
    decision.update(preserved_before=before,preserved_after=after,
        original_shape_evidence=original_shape,working_shape_evidence=working,
        all_original_arrays_and_geometry_unchanged=True)


def resolve_legend_sources(image,templates,reports):
    """Evaluate source pixels only; return copied models and an auditable plan."""
    started=perf_counter()
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected original uint8 BGR image')
    copied_templates=deepcopy(templates);copied_reports=deepcopy(reports)
    by_id={r['swatch_id']:r for r in copied_reports}
    eligible=[t for t in copied_templates if _eligible(t,by_id[t.key])]
    trigger=len(eligible)>=RULES['minimum_duplicate_count']
    sources=[_evaluate_source(image,t,by_id[t.key]) for t in eligible]
    source_by_id={s['swatch_id']:s for s in sources}
    for own in sources:
        own['peer_candidates']=[_peer_evidence(own,peer) for peer in sources if peer is not own]
        peers=[p for p in own['peer_candidates'] if p['eligible_corroborating_peer']]
        own['peer_soft_support']=max((p['soft_support'] for p in peers),default=0.)
        own['joint_signed_score_circle_positive']=own['own_signed_score_circle_positive']+RULES['maximum_peer_score_bonus']*own['peer_soft_support']
        score=by_id[own['swatch_id']].get('shape_evidence',{}).get('shape_scores',{}).get('circle')
        own['existing_binary_circle_overlap']=score
        fit_gate=score is not None and np.isfinite(score) and score>=RULES['existing_circle_overlap_gate']
        if not trigger:
            reason='no_duplicate_automatic_filled_square_group';resolution='not_triggered'
        elif own['independent_gray_decision']=='square':
            reason='own_gray_supports_original_square';resolution='confirmed_square'
        elif own['independent_gray_decision']!='circle':
            reason='weak_own_evidence_peer_cannot_override';resolution='abstained'
        elif not peers:
            reason='independent_circle_but_no_distinct_robust_square_peer';resolution='abstained'
        elif not fit_gate:
            reason='existing_circle_overlap_cannot_pass_unchanged_geometry_gate';resolution='abstained'
        else:
            reason='robust_own_circle_corroborated_by_distinct_square_peer';resolution='corrected'
            own.update(joint_final_decision='circle',applied_class='filled_circle',identity_changed=True)
        own.update(joint_reason=reason,joint_resolution=resolution)
    untouched=[]
    for original,template in zip(templates,copied_templates):
        decision=source_by_id.get(template.key)
        if decision and decision['identity_changed']:
            _apply_circle(template,by_id[template.key],decision)
        else:
            untouched.append(dict(swatch_id=template.key,source_arrays_and_geometry_unchanged=_snapshot(original)==_snapshot(template),
                class_unchanged=original.name==template.name))
    plan=dict(version=VERSION,gray_dependency_version=GRAY_VERSION,source_only=True,manual_oracle=False,
        case_id_used=False,target_allowlist_used=False,plot_reference_points_read=False,
        rules=deepcopy(RULES),trigger=dict(automatic_filled_square_count=len(eligible),triggered=trigger,
            eligible_ids=[t.key for t in eligible]),sources=sources,
        changed_ids=[s['swatch_id'] for s in sources if s['identity_changed']],
        untouched_sources=untouched,elapsed_seconds=perf_counter()-started,
        interpretation='Joint relation is conservative safety/corroboration of independently positive gray evidence, not forced assignment or demonstrated added classifier accuracy.')
    return copied_templates,copied_reports,_json(plan)

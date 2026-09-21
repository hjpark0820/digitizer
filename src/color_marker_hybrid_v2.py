"""Image-only hybrid v2: explicit reconstruction and joint colour identities.

The original v1 modules/model/results remain unchanged. Candidate proposals
reuse the v1 path and fragment machinery. No reference coordinates enter any
function in this module. References only calibrate thresholds in experiments.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

from color_marker_hybrid import extract_candidates as propose_candidates,HybridConfig
from color_marker_evidence_v2 import prepare_evidence
from color_marker_window_v2 import verify_window
from color_blend_uncertainty_v46 import window_kwargs
from color_paper_contradiction_v46 import DEFAULT_WEIGHT as PAPER_CENTER_WEIGHT


@dataclass(frozen=True)
class SelectionConfig:
    threshold: float = .50
    same_series_spacing: float = .36
    competing_center_distance: float = .55
    minimum_independent_fraction: float = .22
    minimum_independent_area_fraction: float = .025
    identity_ambiguity_margin: float = .025


def _plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:_plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [_plain(x) for x in v]
    return v


def _ink_footprint(own,template,x,y):
    """Observed target-colour evidence at required positions, not a shape guess."""
    a=np.asarray(template['soft'],np.float32)
    h,w=a.shape;cx,cy=template.get('center',[(w-1)/2,(h-1)/2])
    left=int(math.floor(x-cx));top=int(math.floor(y-cy))
    moved=cv2.warpAffine(a,np.float32([[1,0,x-cx-left],[0,1,y-cy-top]]),
                         (w+2,h+2),flags=cv2.INTER_LINEAR)
    yy,xx=np.mgrid[:h+2,:w+2];xx=xx+left;yy=yy+top
    inside=(xx>=0)&(yy>=0)&(xx<own.shape[1])&(yy<own.shape[0])&(moved>.15)
    xx=xx[inside];yy=yy[inside]
    values=moved[inside]*own[yy,xx]
    return {(int(px),int(py)):float(v) for px,py,v in zip(xx,yy,values) if v>.05}


def _novel_fraction(a,b):
    mass=sum(a.values())
    # Already explained pixels do not become a second marker merely because
    # the two templates have different antialias strength at that location.
    unique=sum(v for k,v in a.items() if k not in b)
    return unique/max(mass,1.e-8),unique


def _mutually_independent(p,q,cfg):
    d=min(p['diameter'],q['diameter'])
    for a,b in [(p,q),(q,p)]:
        fraction,mass=a.get('independent_evidence',{}).get(str(b['candidate_id']),[0.,0.])
        if fraction<cfg.minimum_independent_fraction or mass<max(1.5,cfg.minimum_independent_area_fraction*d*d):return False
    return True


def analyze_evidence(evidence,*,window_config=None,proposal_config=None):
    """Generate candidates, verify all unique centres, cache identity evidence."""
    started=perf_counter()
    raw=propose_candidates(evidence,config=proposal_config or HybridConfig(
        retain_alternative_centres=True,paper_contradiction_weight=PAPER_CENTER_WEIGHT))
    candidates=[];lookup={};window_seconds=0.;footprints={}
    index_by_id={str(t['id']):i for i,t in enumerate(evidence['templates'])}
    for p in raw['candidates']:
        key=(p['series_id'],round(p['x'],4),round(p['y'],4))
        if key in lookup:
            old=candidates[lookup[key]]
            old['sources']=sorted(set(old['sources'])|set(p['sources']))
            continue
        si=index_by_id[p['series_id']];t=evidence['templates'][si]
        begin=perf_counter()
        window=verify_window(evidence['membership'][si],evidence['other'][si],t,
                             p['x'],p['y'],config=window_config,debug=False,
                             ignore_mask=evidence.get('ignore_mask'),
                             **window_kwargs(evidence, si),
                             guide_model=None if 'guide_membership' not in evidence else
                                 (evidence['guide_membership'][si],evidence['guide_lower'][si],evidence['guide_upper'][si]))
        window_seconds+=perf_counter()-begin
        c=dict(p,candidate_id=len(candidates),window=_plain(window),score=float(window['score']))
        if 'symbol_scale' in t:
            c.update(symbol_scale=t['symbol_scale'],legend_diameter=t['legend_diameter'],
                     window_template_diameter=float(t['diameter']),scale_policy='shared_symbol')
        c['rival_series']=[str(q['id']) for j,q in enumerate(evidence['templates'])
                           if j!=si and evidence['equivalent_colours'][si,j]]
        c['independent_evidence']={}
        lookup[key]=len(candidates);candidates.append(c)
        if c['rival_series']:
            guide=0 if 'guide_upper' not in evidence else evidence['guide_upper'][si]
            footprints[c['candidate_id']]=_ink_footprint(np.maximum(evidence['membership'][si]-guide,0)*(1-evidence.get('ignore_mask',0)),t,c['x'],c['y'])
    # Store counterfactual extra-ink support once so threshold sweeps do not
    # rerun image analysis, and cannot accidentally consult reference pixels.
    for i,c in enumerate(candidates):
        if not c['rival_series']:continue
        for q in candidates[i+1:]:
            if q['series_id'] not in c['rival_series']:continue
            if math.hypot(c['x']-q['x'],c['y']-q['y'])>.75*min(c['diameter'],q['diameter']):continue
            a,b=footprints[c['candidate_id']],footprints[q['candidate_id']]
            c['independent_evidence'][str(q['candidate_id'])]=list(_novel_fraction(a,b))
            q['independent_evidence'][str(c['candidate_id'])]=list(_novel_fraction(b,a))
    return {'candidates':candidates,'proposal_diagnostics':raw['diagnostics'],
            'timing':{'proposals_seconds':raw['seconds'],'window_seconds':window_seconds,
                      'analysis_seconds':perf_counter()-started},
            'version':'color_marker_hybrid_v2'}


def select_candidates(candidates,*,config=None,threshold=None,joint=True,line_gate=True):
    """Final score + confident-line veto, then centre and identity competition.

    Different palette colours never veto each other. Same-colour overlapping
    markers may both survive if the second has independently observed ink.
    A near tie is returned explicitly as an uncertain identity, not fabricated
    multiple labels. No reference coordinate or expected count is inspected.
    """
    cfg=config or SelectionConfig()
    cutoff=cfg.threshold if threshold is None else float(threshold)
    rejected=[];pool=[]
    for p in candidates:
        if line_gate and p['window'].get('line_only_reject',False):
            rejected.append({'candidate_id':p['candidate_id'],'reason':'connector_only_explanation'});continue
        if float(p['score'])<cutoff:
            rejected.append({'candidate_id':p['candidate_id'],'reason':'window_score'});continue
        pool.append(dict(p))
    pool.sort(key=lambda p:(-p['score'],p['candidate_id']))
    nms=[]
    for p in pool:
        if any(p['series_id']==q['series_id'] and math.hypot(p['x']-q['x'],p['y']-q['y'])<
               cfg.same_series_spacing*min(p['diameter'],q['diameter']) for q in nms):
            rejected.append({'candidate_id':p['candidate_id'],'reason':'same_series_duplicate'})
        else:nms.append(p)
    if not joint:return {'points':nms,'uncertain_points':[],'rejected':rejected}
    accepted=[];uncertain=[]
    for p in nms:
        conflicts=[]
        for q in accepted:
            if q['series_id'] not in p.get('rival_series',[]):continue
            d=min(p['diameter'],q['diameter'])
            if math.hypot(p['x']-q['x'],p['y']-q['y'])>cfg.competing_center_distance*d:continue
            independent=_mutually_independent(p,q,cfg)
            if not independent:conflicts.append(q)
        if not conflicts:
            # Do not emit another label for a centre already marked ambiguous.
            ambiguous=next((q for q in uncertain if p['series_id'] in q['competing_palette'] and
                math.hypot(p['x']-q['x'],p['y']-q['y'])<cfg.competing_center_distance*min(p['diameter'],q['diameter'])),None)
            if ambiguous is None:accepted.append(p)
            else:
                if _mutually_independent(p,ambiguous,cfg):
                    accepted.append(p)
                else:
                    ambiguous['alternative_series']=sorted(set(ambiguous['alternative_series'])|{p['series_id']})
                    ambiguous['competing_palette']=sorted(set(ambiguous['competing_palette'])|set(p.get('rival_series',[])))
                    rejected.append({'candidate_id':p['candidate_id'],'reason':'already_ambiguous_identity'})
            continue
        q=max(conflicts,key=lambda z:z['score'])
        if q['score']-p['score']<cfg.identity_ambiguity_margin:
            accepted.remove(q)
            alternatives=sorted(set([q['series_id'],p['series_id']]))
            unknown=dict(q,series_id=None,series_label='Uncertain same-colour identity',
                         alternative_series=alternatives,identity_status='ambiguous',
                         competing_palette=sorted(set(alternatives)|set(q.get('rival_series',[]))|set(p.get('rival_series',[]))))
            uncertain.append(unknown)
            rejected.append({'candidate_id':q['candidate_id'],'reason':'ambiguous_identity'})
            rejected.append({'candidate_id':p['candidate_id'],'reason':'ambiguous_identity'})
        else:
            rejected.append({'candidate_id':p['candidate_id'],'reason':'ink_already_explained_by_other_shape',
                             'winner':q['candidate_id']})
    return {'points':accepted,'uncertain_points':uncertain,'rejected':rejected}


def detect_markers(image_bgr,plot_box,series_specs,*,config=None,threshold=None,max_side=1400,
                   joint=True,line_gate=True,window_config=None):
    """Public image-only v2 API. All output coordinates include source offsets."""
    start=perf_counter()
    if isinstance(config,(str,Path)):
        config=SelectionConfig(**json.loads(Path(config).read_text(encoding='utf-8'))['selection_config'])
    try:evidence=prepare_evidence(image_bgr,plot_box,series_specs,max_side=max_side)
    except ValueError as exc:
        if not str(exc).startswith('No usable legend templates:'):raise
        return {'points':[],'uncertain_points':[],'template_errors':[str(exc)],'seconds':perf_counter()-start}
    result=analyze_evidence(evidence,window_config=window_config)
    sx,sy=evidence['scale_x'],evidence['scale_y'];ox,oy=evidence['source_center_offset']
    for p in result['candidates']:
        p.update(x_px=p['x']/sx+ox,y_px=p['y']/sy+oy,diameter_source=p['diameter']/(.5*(sx+sy)))
    selected=select_candidates(result['candidates'],config=config,threshold=threshold,joint=joint,line_gate=line_gate)
    return {**selected,'template_errors':evidence.get('template_errors',[]),'seconds':perf_counter()-start,
            'diagnostics':result['proposal_diagnostics'],'version':result['version']}

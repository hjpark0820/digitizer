"""Observed error-bar candidates along an entire supplied raw-path corridor.

This proposal stage does not require a bend, known measurement x coordinates,
marker detections, or a sampling schedule. It changes no thresholds in the
existing cap/fragment or single-cap verifiers. It never fills/erases source
pixels. A supported bar is structural evidence, not an accepted data point;
the caller must verify same-series source pixels and resolve series ambiguity.

All arrays/coordinates are native plot-local. Extrema in stem/cap records are
inclusive. Endpoint flags describe source observations and proximity only;
the supplied all-color mask cannot prove a particular series' identity.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from .errorbar_stems import propose_stems
from .path_errorbar_evidence import _runs, score_errorbar_at_path
from .path_guided_errorbar_fragments import score_fragments_near_kink

VERSION = 'whole_path_corridor_bars_v1'


def _cap_seeds(path,neutral,valid,lw,cw):
    """Seed actual short horizontal runs, measuring width before ROI clipping."""
    min_width=max(5,int(math.ceil(4*lw))+1)
    max_width=max(25,int(math.ceil(24*lw))+1)
    yreach=max(6,int(math.ceil(4*cw)))
    records=[];counts=Counter()
    # Global image rows are read once. Full observed row runs prevent a
    # corridor boundary from cutting a grid line into a fictitious short cap.
    for y in range(neutral.shape[0]):
        for run in _runs(np.flatnonzero(neutral[y])):
            a,b=int(run[0]),int(run[-1]);length=b-a+1
            if length<min_width:continue
            if length>max_width:counts['wide_horizontal_runs']+=1;continue
            x=.5*(a+b)
            if not path[0,0]<=x<=path[-1,0]:continue
            py=float(np.interp(x,path[:,0],path[:,1]))
            if abs(y-py)>yreach:continue
            if not valid[y,a:b+1].all():counts['cap_run_crosses_invalid_scope']+=1;continue
            records.append(dict(x=x,y=y,x0=a,x1=b,width=length))
    # Exact repeated cap-center seeds only; no near-x clustering or scheduling.
    return sorted({r['x'] for r in records}),records,dict(counts)


def _cap_box(cap):
    return tuple(int(cap[k]) for k in ('x0','y0','x1','y1'))


def _canonical_stem_id(stem,caps):
    body='_'.join(str(int(stem[k])) for k in ('x0','y0','x1','y1'))
    cap_part='__'.join('_'.join(map(str,box)) for box in sorted({_cap_box(c) for c in caps}))
    return 'observed_stem_'+body+'__caps_'+cap_part


def _scope_ok(stem,caps,neutral,valid,x,y):
    h,w=valid.shape;ix,iy=int(round(x)),int(round(y))
    if not (0<=ix<w and 0<=iy<h and valid[iy,ix]):return False,'candidate_outside_valid_plot'
    for c in caps:
        a,top,b,bottom=_cap_box(c)
        if not (0<=a<=b<w and 0<=top<=bottom<h) or not valid[top:bottom+1,a:b+1].all():
            return False,'observed_cap_crosses_invalid_scope'
        # Detect a long source line truncated by valid-mask exclusions.
        for col in (a-1,b+1):
            if 0<=col<w and np.any(neutral[top:bottom+1,col]&~valid[top:bottom+1,col]):
                return False,'cap_end_is_invalid_scope_cut'
    return True,None


def _endpoint_metadata(path,x,neutral,colored,valid,lw,cw):
    proximity=max(2.,cw)
    rows=[]
    for label,index in [('left',0),('right',-1)]:
        px,py=map(float,path[index]);ix,iy=int(round(px)),int(round(py))
        inside=0<=ix<valid.shape[1] and 0<=iy<valid.shape[0] and bool(valid[iy,ix])
        observed_neutral=bool(inside and neutral[iy,ix])
        observed_chromatic=bool(inside and colored[iy,ix])
        near=abs(x-px)<=proximity
        rows.append(dict(side=label,path_endpoint=[px,py],distance_x=abs(x-px),near=near,
                         valid=bool(inside),observed_neutral=observed_neutral,
                         observed_chromatic=observed_chromatic,
                         observed_source_ink=observed_neutral or observed_chromatic,
                         eligible_for_parent_one_sided_check=bool(near and observed_chromatic)))
    eligible=[r['side'] for r in rows if r['eligible_for_parent_one_sided_check']]
    return dict(endpoints=rows,eligible_sides=eligible,near_path_endpoint=any(r['near'] for r in rows),
                proximity_pixels=proximity,requires_parent_same_series_endpoint_evidence=True,
                is_automatic_acceptance=False,
                interpretation='Only an observed chromatic path endpoint can enter the caller\'s one-sided source check; no endpoint creates a bar.')


def _same_structure(a,b,lw):
    """Shared observed cap + overlapping core, never x proximity alone."""
    if a['stem']['id']==b['stem']['id']:return True
    sa,sb=a['stem'],b['stem']
    if abs(sa['x']-sb['x'])>max(2.,2*lw):return False
    if min(sa['y1'],sb['y1'])<max(sa['y0'],sb['y0']):return False
    # Preserve two full bars with different extents even if their x is equal.
    contained=((sa['y0']>=sb['y0'] and sa['y1']<=sb['y1']) or
               (sb['y0']>=sa['y0'] and sb['y1']<=sa['y1']))
    if not contained:return False
    for ca in a['caps']:
        for cb in b['caps']:
            if min(ca['y1'],cb['y1'])<max(ca['y0'],cb['y0']):continue
            intersection=max(0,min(ca['x1'],cb['x1'])-max(ca['x0'],cb['x0'])+1)
            union=max(ca['x1'],cb['x1'])-min(ca['x0'],cb['x0'])+1
            if intersection/union>=.65:return True
    return False


def scan_path_bars(path_xy,neutral,colored,valid,line_width=1.,curve_width=4.):
    """Return ``proposals`` plus a rejection/provenance ``audit``.

    Proposals contain x/y, stem, full bar evidence, bar_status/status/reason,
    caps, vertical, source_methods and endpoint_eligibility. Single-cap stems
    and directly paired-cap structures use the existing unchanged scorers.
    The all-color ``colored`` mask must contain observed chromatic ink only.
    """
    path=np.asarray(path_xy,float);n=np.asarray(neutral);c=np.asarray(colored);v=np.asarray(valid)
    if n.ndim!=2 or any(a.dtype!=np.bool_ or a.shape!=n.shape for a in (n,c,v)):
        raise ValueError('neutral, colored and valid must be matching bool[H,W] arrays')
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):
        raise ValueError('path_xy must be finite and strictly increasing in x')
    lw,cw=float(line_width),float(curve_width)
    if not np.isfinite([lw,cw]).all() or min(lw,cw)<=0:raise ValueError('Positive finite line and curve widths required')
    ink=n&v;chromatic=c&v
    seeds,seed_records,seed_counts=_cap_seeds(path,n,v,lw,cw)
    candidates=[];records=[];rejects=Counter();pair_evaluations=0

    def add(evidence,stem,method,seed=None,ambiguous_ids=()):
        nonlocal candidates
        bar=deepcopy(evidence);st=deepcopy(stem)
        # Old ST### indices depend on the proposal roster. Use source geometry
        # instead so independently called series share stable source identity.
        if method=='single_cap_vertical':
            st['detector_id']=st['id'];st['id']=_canonical_stem_id(st,bar.get('caps',[]))
        x=float(bar.get('x',st['x']));y=float(bar.get('path_y',np.interp(x,path[:,0],path[:,1])))
        ok,scope_reason=_scope_ok(st,bar.get('caps',[]),n,v,x,y)
        if not ok:bar.update(status='rejected',reason=scope_reason)
        if ambiguous_ids and bar['status']!='rejected':
            bar.update(status='tentative',reason='multiple_compatible_cap_pairs')
        record=dict(method=method,seed_x=seed,physical_id=st['id'],status=bar['status'],reason=bar['reason'],
                    x=x,y=y,bar=bar,ambiguous_physical_ids=list(ambiguous_ids))
        records.append(record)
        if bar['status']=='rejected':rejects[bar['reason']]+=1;return
        candidates.append(dict(x=x,y=y,stem=st,bar=bar,bar_status=bar['status'],status=bar['status'],
                               reason=bar['reason'],caps=bar.get('caps',[]),vertical=bar.get('vertical',{}),
                               source_methods=[method],source_seed_x=[] if seed is None else [seed],
                               ambiguous_physical_ids=list(ambiguous_ids),
                               endpoint_eligibility=_endpoint_metadata(path,x,n,c,v,lw,cw)))

    for sx in seeds:
        scored=score_fragments_near_kink(sx,path,ink,chromatic,line_width=lw,curve_width=cw)
        pair_evaluations+=1
        viable=[p for p in scored.get('candidates',[]) if p['status']!='rejected']
        ambiguous=[p['stem']['id'] for p in viable] if len(viable)>1 else []
        if scored.get('candidates'):
            for pair in scored['candidates']:add(pair,pair['stem'],'paired_caps_corridor',sx,ambiguous)
        else:
            rejects[scored['reason']]+=1
            records.append(dict(method='paired_caps_corridor',seed_x=sx,status='rejected',reason=scored['reason'],bar=scored))
    stems=propose_stems(n,v,lw)
    considered=0
    for stem in stems['stems']:
        x=float(stem['x'])
        if not path[0,0]<=x<=path[-1,0]:continue
        py=float(np.interp(x,path[:,0],path[:,1]));reach=max(6,4*cw)
        if py<stem['y0']-reach or py>stem['y1']+reach:continue
        considered+=1
        evidence=score_errorbar_at_path(stem,ink,None,py,lw)
        add(evidence,stem,'single_cap_vertical')

    # Complete-link groups prevent a third fragment transitively merging two
    # neighboring structures. Keep all evidence-source aliases in the output.
    groups=[]
    for item in sorted(candidates,key=lambda p:(p['stem']['x'],p['stem']['y0'],p['stem']['id'])):
        group=next((g for g in groups if all(_same_structure(item,q,lw) for q in g)),None)
        if group is None:groups.append([item])
        else:group.append(item)
    output=[];dedup=[];rank={'supported':2,'tentative':1}
    for group in groups:
        best=deepcopy(max(group,key=lambda p:(rank[p['bar_status']],len(p['caps']),p['bar']['structural_score'])))
        ids=sorted({p['stem']['id'] for p in group})
        methods=sorted({m for p in group for m in p['source_methods']})
        seed_x=sorted({x for p in group for x in p['source_seed_x']})
        ambiguity=sorted({s for p in group for s in p['ambiguous_physical_ids']})
        best.update(source_methods=methods,source_seed_x=seed_x,source_structure_aliases=ids,
                    ambiguous_physical_ids=ambiguity)
        # A different seed window must not silently remove a physical ambiguity.
        if ambiguity:
            best.update(bar_status='tentative',status='tentative',reason='multiple_compatible_cap_pairs')
            best['bar'].update(status='tentative',reason='multiple_compatible_cap_pairs')
        output.append(best)
        if len(group)>1:dedup.append(dict(physical_id=best['stem']['id'],input_count=len(group),source_ids=ids,methods=methods))
    output.sort(key=lambda p:(p['x'],p['y'],p['stem']['id']))
    return dict(version=VERSION,proposals=output,audit=dict(
        path_samples=len(path),path_samples_removed=0,cap_seed_runs=seed_records,
        cap_seed_count=len(seeds),pair_evaluations=pair_evaluations,
        source_stem_count=len(stems['stems']),corridor_stems_considered=considered,
        stem_proposer_diagnostics=stems['diagnostics'],seed_rejections=seed_counts,
        rejection_counts=dict(rejects),evaluations=records,deduplicated_structures=dedup,
        proposal_count=len(output),proposal_status_counts=dict(Counter(p['bar_status'] for p in output)),
        pixels_filled=0,pixels_erased=0,thresholds_changed=False,known_sampling_coordinates_used=False,
        limitation='I-shaped text can be locally identical to a bar. Parent must verify same-series pixels, valid text/grid context and series ambiguity.'))

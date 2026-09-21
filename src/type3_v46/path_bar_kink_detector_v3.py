"""Source-intersection corners plus corridor-wide error-bar proposals.

Experiment-only. Paths remain uncut hints; neither a neighboring series' x
nor an inferred path sample is positive evidence for a measurement center.
"""
from copy import deepcopy
import numpy as np
from .path_bar_kink_detector import curve_support,decide,abstain_shared_bars
from .path_bar_kink_detector_v2 import canonicalize_physical_bars


def _endpoint_check(path,x,curve,own,soft,valid,width):
    """Allow one-sided fits only at a source-supported outer path endpoint.

    An internal occlusion is not an endpoint. The candidate must be within
    three stroke widths of a raw-domain boundary and have essentially no
    observed own-color continuation on the outward side of the bar.
    """
    side='left' if x-path[0,0]<path[-1,0]-x else 'right'
    distance=float(min(x-path[0,0],path[-1,0]-x))
    inward='right' if side=='left' else 'left';outward=side
    inner=curve[inward];outer=curve[outward]
    fit=inner['fit'];guard=max(2.,width)
    near=distance<=max(4.,3*width)
    good=fit is not None and inner['source_columns']>=6 and fit['rms']<=max(1.,.5*width)
    # Check source pixels outside the raw domain too, not just truncated fit
    # windows. A falsely shortened path must not manufacture an endpoint.
    hits=[];sgn=-1 if side=='left' else 1
    for d in range(int(np.ceil(guard)),int(np.ceil(6*width))+1):
        xx=int(round(x+sgn*d))
        if not 0<=xx<own.shape[1]:continue
        yy=float(fit['y']+fit['slope']*(xx-x)) if fit else float(np.interp(xx,path[:,0],path[:,1]))
        rows=np.arange(max(0,int(round(yy))-int(np.ceil(width))),min(own.shape[0],int(round(yy))+int(np.ceil(width))+1))
        hits.append(bool(len(rows) and (own[rows,xx]&valid[rows,xx]&(soft[rows,xx]>=.18)).any()))
    empty=bool(len(hits)>=4 and np.mean(hits)<=.10)
    return dict(eligible=bool(near and good and empty and outer['source_columns']<=2),side=side,
                raw_endpoint_distance=distance,inward_fit=fit,outward_observed_fraction=float(np.mean(hits)) if hits else None,
                interpretation='Conservative outer-endpoint test, not arbitrary one-sided occlusion')


def _bar_candidate(q,path,obs,own,soft,valid,width,grid_dominated):
    x=float(q['x']);raw_y=float(np.interp(x,path[:,0],path[:,1]))
    curve=curve_support(path,x,own,soft,valid,width)
    endpoint=_endpoint_check(path,x,curve,own,soft,valid,width)
    status,reason=decide(q['bar'],None,curve,grid_dominated)
    y=raw_y
    if curve['bilateral']:
        y=float(curve['intersection_y']) # At a measured bar x, average consistent arm heights, not an x-y corner intersection.
    elif q['bar']['status']=='supported' and endpoint['eligible'] and not grid_dominated:
        y=float(endpoint['inward_fit']['y']);status,reason='accepted','observed_errorbar_at_supported_outer_endpoint'
    stem=q['stem']
    if not stem['y0']-2<=y<=stem['y1']+2 or abs(y-raw_y)>max(4.,2*width):
        status,reason='tentative','source_height_inconsistent_with_errorbar'
    if q['bar'].get('reason')=='multiple_compatible_cap_pairs':
        status,reason='tentative','multiple_compatible_physical_errorbars'
    ix,iy=int(round(x)),int(round(y))
    if not (0<=ix<valid.shape[1] and 0<=iy<valid.shape[0] and valid[iy,ix]):
        status,reason='rejected','corrected_center_outside_valid_plot'
    return dict(x=x,y=y,raw_center=[x,raw_y],stem=deepcopy(stem),bar=deepcopy(q['bar']),kink=None,
        origin='whole_path_errorbar_search',curve=curve,endpoint=endpoint,status=status,reason=reason,
        score=float(4*(q['bar']['status']=='supported')+curve['bilateral']),
        path_observed_at_center=bool(obs[np.argmin(abs(path[:,0]-x))]),
        interpretation='Measurement-center hypothesis from observed bar and source curve; not a marker glyph')


def detect_series(path_xy,observed,own,soft,valid,neutral,dark,line_width,baseline,
                  *,grid_mask,colored_ink_mask,polyline_mode=False):
    from .observed_polyline_corner import refine_observed_corner
    from .path_corridor_bar_proposals import scan_path_bars
    path=np.asarray(path_xy,float);obs=np.asarray(observed,bool);width=float(line_width)
    result=deepcopy(baseline);candidates=deepcopy(baseline['candidates'])
    for p in candidates:p['previous_candidate_id']=p['id'];p['previous_status']=p['status']
    dominated=result['grid_fraction']>=.65;corner_audit=[]
    for kink in baseline['kinks']['candidates']:
        corner=refine_observed_corner(path,kink['x'],own,soft,valid,width)
        corner_audit.append(dict(kink_id=kink['id'],evidence=corner))
        if corner['status']=='rejected':continue
        x=float(corner['x']);y=float(corner['y']);ix=int(round(x));iy=int(round(y))
        if not (0<=ix<own.shape[1] and 0<=iy<own.shape[0] and valid[iy,ix]):continue
        curve=curve_support(path,x,own,soft,valid,width)
        supported=corner['status']=='supported' and polyline_mode and not dominated
        status='accepted' if supported else 'tentative'
        reason='verified_source_polyline_corner' if supported else 'source_corner_uncertain_or_not_polyline'
        if dominated:reason='path_dominated_by_long_horizontal_ink'
        if supported:
            for old in candidates:
                if old['stem'] is None and old.get('kink') and old['kink']['id']==kink['id'] and old['status']!='rejected':
                    old.update(status='rejected',reason='replaced_by_source_intersection',merged_into_x=x)
        candidates.append(dict(x=x,y=y,raw_center=[float(kink['x']),float(kink['y'])],stem=None,
            bar=dict(status='rejected',reason='bar_not_required_for_verified_polyline_corner'),kink=deepcopy(kink),
            corner=corner,origin='source_line_intersection',curve=curve,status=status,reason=reason,
            score=6. if supported else 2.,path_observed_at_center=bool(obs[np.argmin(abs(path[:,0]-x))]),
            interpretation='Measurement-center hypothesis; independent source corner, no errorbar witness required'))
    scan=scan_path_bars(path,neutral,colored_ink_mask,valid,line_width=1.,curve_width=width)
    for q in scan['proposals']:
        candidates.append(_bar_candidate(q,path,obs,own,soft,valid,width,dominated))
    # Keep before-dedup decisions auditable. Never merge only on x across colors.
    rank={'accepted':2,'tentative':1,'rejected':0};selected=[]
    for p in sorted(candidates,key=lambda p:(rank[p['status']],p.get('corner',{}).get('status')=='supported',p['score']),reverse=True):
        if p['status']=='rejected':continue
        nearby=next((q for q in selected if abs(q['x']-p['x'])<=max(3.,2*width) and abs(q['y']-p['y'])<=2*width),None)
        if nearby:
            p['pre_duplicate_status']=p['status'];p.update(status='rejected',reason='duplicate_local_measurement_hypothesis',merged_into_x=nearby['x'])
            nearby.setdefault('merged_evidence',[]).append(dict(origin=p['origin'],x=p['x'],y=p['y'],reason=p.get('previous_status',p['pre_duplicate_status'])))
        else:selected.append(p)
    candidates.sort(key=lambda p:(p['x'],p['y']))
    for i,p in enumerate(candidates,1):p['id']=f'C{i:03d}'
    result.update(candidates=candidates,points=[p for p in candidates if p['status']=='accepted'],
        tentative_points=[p for p in candidates if p['status']=='tentative'],source_corner_audit=corner_audit,
        whole_path_bar_audit=scan,raw_path_samples=len(path),path_samples_removed=0,polyline_mode=bool(polyline_mode),
        policy='Source-intersection corner verification plus whole-path observed errorbar search; no path trimming')
    return result


def reconcile_series(results):
    output,physical=canonicalize_physical_bars(results)
    output,ambiguity=abstain_shared_bars(output)
    return output,dict(shared_bar_groups=physical,ambiguities=ambiguity)

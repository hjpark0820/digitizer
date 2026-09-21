"""Add a kink-guided partial-bar search without discarding uncut paths.

The first pass still uses the frozen vertical-stem proposals. The second pass
can inspect caps and short vertical fragments directly near a geometric kink,
so a colored curve interrupting the neutral stem is not an upstream veto.
"""
from copy import deepcopy
import numpy as np
from .path_bar_kink_detector import detect_series as first_pass, curve_support, decide
from .path_errorbar_evidence import score_errorbar_at_path


def detect_series(path_xy, observed, own, soft, valid, stems, neutral, dark,
                  line_width, grid_mask=None, colored_ink_mask=None):
    from .path_guided_errorbar_fragments import score_fragments_near_kink
    result=first_pass(path_xy,observed,own,soft,valid,stems,neutral,dark,line_width,grid_mask)
    path=np.asarray(path_xy,float);obs=np.asarray(observed,bool);width=float(line_width)
    if colored_ink_mask is None:
        raise ValueError('Observed chromatic pixels are required, not an inferred path mask')
    candidates=deepcopy(result['candidates'])
    # Retain all rejected frozen-stem reasons in this version.
    bar_audit=[]
    for stem in stems:
        if path[0,0]<=stem['x']<=path[-1,0]:
            py=float(np.interp(stem['x'],path[:,0],path[:,1]))
            bar_audit.append(dict(stem_id=stem['id'],evidence=score_errorbar_at_path(stem,neutral,dark,py,1.)))
    fragments=[]
    for kink in result['kinks']['candidates']:
        evidence=score_fragments_near_kink(kink['x'],path,neutral,colored_ink_mask,
                                          line_width=1.,curve_width=width)
        fragments.append(dict(kink_id=kink['id'],evidence=evidence))
        if evidence['status']=='rejected':continue
        x=float(evidence['x']);y=float(evidence['path_y']);ix=int(round(x));iy=int(round(y))
        if not (0<=ix<own.shape[1] and 0<=iy<own.shape[0] and valid[iy,ix]):continue
        curve=curve_support(path,x,own,soft,valid,width)
        status,reason=decide(evidence,kink,curve,result['grid_fraction']>=.65)
        if evidence['reason']=='multiple_compatible_cap_pairs':
            status,reason='tentative','multiple_compatible_physical_errorbars'
        # The cap-refined x and its originating bare kink describe the same
        # hypothesis even when refinement exceeds the generic local NMS radius.
        if status!='rejected':
            for old in candidates:
                if old['origin']=='path_bend_x' and old['kink'] and old['kink']['id']==kink['id']:
                    old['pre_duplicate_status']=old['status']
                    old.update(status='rejected',reason='superseded_by_cap_refined_same_kink',merged_into_x=x)
        candidates.append(dict(x=x,y=y,stem=deepcopy(evidence['stem']),bar=evidence,
            kink=deepcopy(kink),origin='kink_guided_observed_caps',curve=curve,status=status,reason=reason,
            path_observed_at_center=bool(obs[np.argmin(abs(path[:,0]-x))]),
            score=float(3*(evidence['status']=='supported')+(evidence['status']=='tentative')+
                        2*(kink['strength']=='strong')+curve['bilateral']),
            interpretation='Inferred measurement center, not an observed marker glyph'))
    rank={'accepted':2,'tentative':1,'rejected':0};selected=[]
    for p in sorted(candidates,key=lambda p:(rank[p['status']],p['score'],p['stem'] is not None),reverse=True):
        if p['status']=='rejected':continue
        q=next((q for q in selected if abs(q['x']-p['x'])<=max(3.,1.5*width) and abs(q['y']-p['y'])<=2*width),None)
        if q:
            p['pre_duplicate_status']=p['status'];p.update(status='rejected',reason='duplicate_local_hypothesis',merged_into_x=q['x'])
        else:selected.append(p)
    candidates.sort(key=lambda p:(p['x'],p['y']))
    for i,p in enumerate(candidates,1):p['id']=f'C{i:03d}'
    result.update(candidates=candidates,points=[p for p in candidates if p['status']=='accepted'],
        tentative_points=[p for p in candidates if p['status']=='tentative'],
        frozen_stem_audit=bar_audit,kink_guided_fragment_audit=fragments,
        policy='Uncut path plus frozen stems AND direct kink-guided observed cap/fragment evidence; no path trimming')
    return result


def canonicalize_physical_bars(results):
    """Merge old/new identifiers only with a shared observed cap witness.

    Complete-link groups avoid transitive merging of neighboring bars. Same
    concentration/x alone is never a merge criterion. Preserve all provenance.
    """
    output=deepcopy(results);members=[]
    for sid,d in output.items():
        for p in d['candidates']:
            if p['stem'] is not None and p['status']!='rejected':members.append((sid,p))

    def same(a,b):
        sa,sb=a[1]['stem'],b[1]['stem']
        if abs(sa['x']-sb['x'])>2.:return False
        if min(sa['y1'],sb['y1'])<max(sa['y0'],sb['y0']):return False
        for ca in a[1]['bar'].get('caps',[]):
            for cb in b[1]['bar'].get('caps',[]):
                if min(ca['y1'],cb['y1'])<max(ca['y0'],cb['y0']):continue
                intersection=max(0,min(ca['x1'],cb['x1'])-max(ca['x0'],cb['x0'])+1)
                union=max(ca['x1'],cb['x1'])-min(ca['x0'],cb['x0'])+1
                if intersection/union>=.65:return True
        return False

    groups=[]
    for member in sorted(members,key=lambda m:(m[1]['stem']['x'],m[1]['stem']['y0'],m[0])):
        group=next((g for g in groups if all(same(member,n) for n in g)),None)
        if group is None:groups.append([member])
        else:group.append(member)
    audit=[];rank={'accepted':2,'tentative':1,'rejected':0}
    for i,group in enumerate(groups,1):
        identity=f'physical_bar_{i:03d}'
        audit.append(dict(physical_id=identity,source_ids=sorted({p['stem']['id'] for _,p in group}),
                          series_ids=sorted({sid for sid,_ in group})))
        seen=set()
        for sid,p in sorted(group,key=lambda sp:(rank[sp[1]['status']],sp[1]['score']),reverse=True):
            p['stem']['detector_id']=p['stem']['id'];p['stem']['id']=identity
            if sid in seen:
                p['pre_duplicate_status']=p['status'];p.update(status='rejected',reason='duplicate_same_physical_bar')
            else:seen.add(sid)
    for d in output.values():
        d['points']=[p for p in d['candidates'] if p['status']=='accepted']
        d['tentative_points']=[p for p in d['candidates'] if p['status']=='tentative']
    return output,audit

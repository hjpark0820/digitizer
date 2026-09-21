"""Joint measurement-center hypotheses from uncut paths, bends and error bars.

The source has no marker glyphs. A raw interpolated path supplies candidate y,
not positive pixel evidence. Structural caps are distinct from safe-to-erase
caps. Nothing is erased or promoted just because a missing path was bridged.
"""
from copy import deepcopy
import math
import numpy as np


def curve_support(path, x, own, soft, valid, width):
    """Independent same-series source columns on each side of candidate x."""
    guard=max(2,int(math.ceil(.75*width)));reach=max(18,int(math.ceil(7*width)))
    radius=max(2,int(math.ceil(width)));output={}
    for name,sign in [('left',-1),('right',1)]:
        points=[];coverage=[];owned=[]
        for offset in range(guard,reach+1):
            xx=int(round(x+sign*offset))
            if not path[0,0]<=xx<=path[-1,0] or not 0<=xx<own.shape[1]:continue
            yy=float(np.interp(xx,path[:,0],path[:,1]));mid=int(round(yy))
            rows=np.arange(max(0,mid-radius),min(own.shape[0],mid+radius+1))
            if not len(rows):continue
            mass=soft[rows,xx]*valid[rows,xx];eligible=own[rows,xx]&valid[rows,xx]&(mass>=.18)
            coverage.append(bool((mass>=.18).any()));owned.append(bool(eligible.any()))
            if eligible.any():
                pts=rows[eligible];weights=mass[eligible]
                points.append([xx,float(np.average(pts,weights=weights)),float(weights.max())])
        evidence=dict(points=points,available_columns=len(coverage),soft_coverage=float(np.mean(coverage)) if coverage else 0.,
                      own_coverage=float(np.mean(owned)) if owned else 0.,source_columns=len(points),fit=None)
        if len(points)>=4:
            a=np.asarray(points);slope,intercept=np.polyfit(a[:,0]-x,a[:,1],1,w=np.sqrt(a[:,2]))
            rms=float(np.sqrt(np.average((a[:,1]-slope*(a[:,0]-x)-intercept)**2,weights=a[:,2])))
            evidence['fit']=dict(slope=float(slope),y=float(intercept),rms=rms)
        output[name]=evidence
    left=output['left']['fit'];right=output['right']['fit'];path_y=float(np.interp(x,path[:,0],path[:,1]))
    good=lambda e:e['fit'] is not None and e['fit']['rms']<=max(1.5,.75*width) and e['soft_coverage']>=.35
    bilateral=good(output['left']) and good(output['right'])
    if bilateral:
        bilateral=abs(left['y']-right['y'])<=max(3.,1.5*width) and max(abs(left['y']-path_y),abs(right['y']-path_y))<=max(4.,2*width)
    output.update(bilateral=bool(bilateral),one_side=bool(good(output['left']) or good(output['right'])),
        bend_degrees=abs(math.degrees(math.atan(right['slope'])-math.atan(left['slope']))) if left and right else 0.,
        intersection_y=.5*(left['y']+right['y']) if left and right else None,
        interpretation='Independent observed own-color source pixels, not raw-path samples')
    return output


def decide(bar,kink,curve,grid_dominated=False):
    strong_kink=bool(kink and kink['strength']=='strong')
    source_bend=curve['bilateral'] and curve['bend_degrees']>=6.
    if grid_dominated:
        return 'tentative','path_dominated_by_long_horizontal_ink'
    if bar['status']=='supported' and curve['bilateral']:
        return 'accepted','errorbar_and_bend' if strong_kink and source_bend else 'errorbar_and_bilateral_curve'
    if bar['status']=='tentative' and strong_kink and source_bend:
        return 'accepted','partial_errorbar_and_observed_bend'
    if bar['status']!='rejected' and curve['one_side']:
        return 'tentative','errorbar_with_incomplete_curve_support'
    if strong_kink:
        return 'tentative','bend_without_verified_errorbar'
    return 'rejected','insufficient_independent_structure_or_bend'


def detect_series(path_xy,observed,own,soft,valid,stems,neutral,dark,line_width,grid_mask=None):
    from .path_kink_evidence import analyze_kinks
    from .path_errorbar_evidence import score_errorbar_at_path
    path=np.asarray(path_xy,float);obs=np.asarray(observed,bool);width=float(line_width)
    if path.ndim!=2 or path.shape[1]!=2 or len(obs)!=len(path) or len(path)<2:raise ValueError('Invalid path/observed arrays')
    if not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):raise ValueError('Path must be finite and increasing in x')
    if not all(np.asarray(a).shape==own.shape for a in [soft,valid,neutral,dark]):raise ValueError('Evidence shapes differ')
    kink_result=analyze_kinks(path,width,observed=obs);kinks=kink_result['candidates']
    pix=np.rint(path).astype(int);in_bounds=(pix[:,0]>=0)&(pix[:,0]<own.shape[1])&(pix[:,1]>=0)&(pix[:,1]<own.shape[0])
    grid_fraction=float(np.mean(grid_mask[pix[in_bounds,1],pix[in_bounds,0]])) if grid_mask is not None and in_bounds.any() else 0.
    proposals=[];matched=set();radius=max(4.,2*width)
    for stem in stems:
        x=float(stem['x'])
        if not path[0,0]<=x<=path[-1,0]:continue
        y=float(np.interp(x,path[:,0],path[:,1]))
        bar=score_errorbar_at_path(stem,neutral,dark,y,1.)
        near=[(i,k) for i,k in enumerate(kinks) if abs(k['x']-x)<=radius]
        best=min(near,key=lambda ik:abs(ik[1]['x']-x)) if near else None
        # A rejected structural fragment must not drag a separate kink away
        # from its own estimated x location.
        if bar['status']=='rejected':continue
        if best:matched.add(best[0])
        proposals.append(dict(x=x,y=y,stem=deepcopy(stem),bar=bar,kink=deepcopy(best[1]) if best else None,origin='observed_errorbar_x'))
    for i,k in enumerate(kinks):
        if i not in matched:
            proposals.append(dict(x=float(k['x']),y=float(np.interp(k['x'],path[:,0],path[:,1])),stem=None,
                bar=dict(status='rejected',reason='no_matched_structural_errorbar',structural_score=0.),kink=deepcopy(k),origin='path_bend_x'))
    for p in proposals:
        x=p['x'];y=p['y'];ix=int(round(x));iy=int(round(y))
        if not (0<=ix<own.shape[1] and 0<=iy<own.shape[0] and valid[iy,ix]):
            p.update(status='rejected',reason='outside_valid_plot',score=0.,curve={});continue
        curve=curve_support(path,x,own,soft,valid,width)
        status,reason=decide(p['bar'],p['kink'],curve,grid_fraction>=.65)
        p.update(curve=curve,status=status,reason=reason,path_observed_at_center=bool(obs[np.argmin(abs(path[:,0]-x))]),
            score=float(3*(p['bar']['status']=='supported')+1*(p['bar']['status']=='tentative')+
                        2*bool(p['kink'] and p['kink']['strength']=='strong')+curve['bilateral']),
            interpretation='Inferred measurement center, not an observed marker glyph')
    # Merge duplicate local hypotheses, not experimental time columns. A
    # physical stem takes priority over a nearby bare kink at equal status.
    selected=[]
    rank={'accepted':2,'tentative':1,'rejected':0}
    for p in sorted(proposals,key=lambda p:(rank[p['status']],p['score'],p['stem'] is not None),reverse=True):
        neighbor=next((q for q in selected if abs(q['x']-p['x'])<=max(3.,1.5*width) and abs(q['y']-p['y'])<=2*width),None)
        if neighbor is not None:
            p['pre_duplicate_status']=p['status'];p.update(status='rejected',reason='duplicate_local_hypothesis',merged_into_x=neighbor['x'])
        else:selected.append(p)
    proposals.sort(key=lambda p:(p['x'],p['y']))
    for i,p in enumerate(proposals,1):p['id']=f'C{i:03d}'
    return dict(candidates=proposals,points=[p for p in proposals if p['status']=='accepted'],
        tentative_points=[p for p in proposals if p['status']=='tentative'],kinks=kink_result,
        grid_fraction=grid_fraction,raw_path_samples=len(path),path_samples_removed=0,
        policy='Use raw path geometry, structural error bars and independent source-color curve support; no path trimming')


def abstain_shared_bars(results):
    output=deepcopy(results);groups={}
    for sid,result in output.items():
        for p in result['candidates']:
            if p['stem'] is not None and p['status']=='accepted':groups.setdefault(p['stem']['id'],[]).append((sid,p))
    audit=[]
    for stem_id,members in groups.items():
        if len({sid for sid,p in members})>1:
            audit.append(dict(stem_id=stem_id,series_ids=sorted({sid for sid,p in members})))
            for sid,p in members:p.update(status='tentative',reason='shared_neutral_errorbar_series_ambiguity')
    for result in output.values():
        result['points']=[p for p in result['candidates'] if p['status']=='accepted']
        result['tentative_points']=[p for p in result['candidates'] if p['status']=='tentative']
    return output,audit

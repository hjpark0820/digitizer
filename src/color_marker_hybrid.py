"""Generic colour-marker candidates and occlusion-aware window verification.

Inputs are pixels, plot/legend geometry, and image-extracted templates. No
reference point, sampling schedule, filename or antibody identity enters the
detector. A separately trained, plain-JSON forest may score the dimensionless
window features. The rule scorer remains available without sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import math
import json
from pathlib import Path
import cv2
import numpy as np

from color_path_proposals import propose_path_markers

FEATURE_NAMES = [
    'required_support','visible_missing','core_missing','same_color_extra',
    'hole_extra','occluded_fraction','positive_visible_support','edge_alignment',
    'body_density','context_density','line_explained_ratio','bulge_ratio',
    'grid_vote_fraction','grid_fragment_score','path_density','residual_density',
    'thickness_over_diameter','path_support','stem_fraction','unknown_fraction',
    'color_confidence','template_fill','template_aspect','line_width_over_diameter',
    'from_path','from_grid','source_agreement','rule_score',
]


@dataclass(frozen=True)
class HybridConfig:
    grid_cell_fraction: float = .50
    grid_stride_fraction: float = .50
    candidate_spacing_fraction: float = .24
    final_spacing_fraction: float = .36
    maximum_candidates_per_series: int = 650
    fragment_support_floor: float = .52
    minimum_grid_peak: float = .23
    maximum_refinement_fraction: float = .18
    # v46/v2 opts in explicitly; the historical v1/forest proposal contract
    # remains unchanged for callers that construct the default config.
    retain_alternative_centres: bool = False
    alternative_radius_fraction: float = .70
    maximum_alternative_centres: int = 2
    # Opt-in keeps v1 feature/forest contracts and historical callers intact.
    paper_contradiction_weight: float = 0.


def _filter(a, k, normalize=False):
    k=np.asarray(k,np.float32)
    if normalize:k=k/max(float(k.sum()),1.e-6)
    return cv2.filter2D(np.asarray(a,np.float32),-1,k,borderType=cv2.BORDER_CONSTANT)


def _embed(a, size):
    """Place a template about the same continuous centre as its source patch."""
    a=np.asarray(a,np.float32)
    dy=(size-a.shape[0])/2;dx=(size-a.shape[1])/2
    return cv2.warpAffine(a,np.float32([[1,0,dx],[0,1,dy]]),(size,size),
                          flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def _geometry(template):
    diameter=max(3.,float(template['diameter']))
    a=np.asarray(template['soft'],np.float32)
    # The evidence module supplies the marker body separately from its
    # horizontal connector. Do not cut a stripe through the marker here.
    size=max(a.shape[0],a.shape[1],int(math.ceil(2.6*diameter)))|1
    expected=np.clip(_embed(a,size),0,1)
    required=np.clip(_embed(template.get('weight',a),size),0,1)
    uncertain=_embed(template.get('uncertain',np.zeros_like(a)),size)>.1
    core=_embed(template.get('core',a>.65),size)>.5
    envelope=_embed(template.get('envelope',a>.1),size)>.2
    envelope=cv2.dilate(envelope.astype(np.uint8),np.ones((3,3),np.uint8))>0
    if not core.any():core=expected>.35
    core=core & ~uncertain
    yy,xx=np.mgrid[:size,:size];c=(size-1)/2
    disk=(xx-c)**2+(yy-c)**2<=(.78*diameter)**2
    context=(xx-c)**2+(yy-c)**2<=(1.25*diameter)**2
    hole=envelope & (expected<.10) & ~uncertain
    return dict(diameter=diameter,size=size,expected=expected,required=required,core=core.astype(np.float32),
                envelope=envelope,body=disk.astype(np.float32),context=context.astype(np.float32),
                extra=(disk & ~envelope).astype(np.float32),hole=hole.astype(np.float32))


def _fragment_maps(own, geometry, config, other=None, ignored=None):
    """Overlapping template-cell responses aligned to common marker centres.

    Each small fragment is slid over every plot position by correlation. Its
    template offset converts that fragment match into a centre hypothesis.
    Coincident hypotheses vote. These are dense overlapping windows, not the
    older exclusive plot partition or independent observations.
    """
    expected=geometry.get('required',geometry['expected']);d=geometry['diameter'];size=geometry['size']
    yy,xx=np.nonzero(expected>.15)
    if not len(xx):return np.zeros_like(own),np.zeros_like(own),0
    cell=max(3,round(d*config.grid_cell_fraction));step=max(1,round(cell*config.grid_stride_fraction))
    total=np.zeros_like(own);votes=np.zeros_like(own);count=0
    for y0 in range(int(yy.min()),int(yy.max())+1,step):
        for x0 in range(int(xx.min()),int(xx.max())+1,step):
            part=np.zeros((size,size),np.float32)
            part[y0:y0+cell,x0:x0+cell]=expected[y0:y0+cell,x0:x0+cell]
            if part.sum()<max(1.8,.035*expected.sum()):continue
            score=np.clip(_filter(own,part,True),0,1)
            if other is not None or ignored is not None:
                covered_field = np.zeros_like(own) if other is None else other
                if ignored is not None:
                    covered_field = 1-(1-covered_field)*(1-ignored)
                covered=np.clip(_filter(covered_field,part,True),0,1)
                # Visible own-colour evidence is still required. Occlusion
                # normalizes only partial fragments, never creates a vote.
                score=np.clip(score/np.maximum(1-covered,.45),0,1)*np.minimum(1,score/.25)
            total+=score;votes+=(score>=config.fragment_support_floor);count+=1
    if not count:return total,votes,0
    return total/count,votes/count,count


def _window_maps(own, other, unknown, confidence, template, path_maps, config, ignored=None):
    g=_geometry(template);d=g['diameter'];expected=g['expected'];required=g['required'];mass=max(float(expected.sum()),1.)
    # A one-pixel AA band is a soft allowance, not arbitrary per-point scaling.
    nearby=cv2.dilate(own,np.ones((3,3),np.uint8))
    supported=.65*own+.35*nearby
    direct=_filter(own,required,True)
    occlusion=np.clip(_filter(other,required,True),0,1)
    observable = 1 if ignored is None else 1-ignored
    missing=_filter((1-supported)*(1-other)*observable,required,True)
    core_missing=_filter((1-supported)*(1-other)*observable,g['core'],True)
    visible_positive=_filter(own*(1-other),required,True)
    extra=_filter(own,g['extra'])/mass
    holes=_filter(own,g['hole'])/mass
    body_density=_filter(own,g['body'],True)
    context_density=_filter(own,g['context'],True)
    # Explain connectors with actual target-colour support on narrow lines
    # through the candidate, keeping this evidence separate from marker ink.
    line_width=float(path_maps.get('line_width',np.ones((1,1))).flat[0])
    line_best=np.zeros_like(own);size=g['size'];c=size//2
    for angle in np.linspace(0,np.pi,12,endpoint=False):
        k=np.zeros((size,size),np.uint8);r=1.2*d
        dx=r*math.cos(float(angle));dy=r*math.sin(float(angle))
        cv2.line(k,(round(c-dx),round(c-dy)),(round(c+dx),round(c+dy)),1,
                 max(1,round(min(line_width,.2*d))))
        line_best=np.maximum(line_best,_filter(own,k))
    context_mass=_filter(own,g['context'])
    line_ratio=np.clip(line_best/np.maximum(context_mass,1.),0,1)
    bulge=np.clip((body_density-context_density)/np.maximum(body_density,.04),-1,1)
    gx=cv2.Sobel(own,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(own,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)+.1
    tx=cv2.Sobel(expected,cv2.CV_32F,1,0,ksize=3)
    ty=cv2.Sobel(expected,cv2.CV_32F,0,1,ksize=3)
    tm=np.sqrt(tx*tx+ty*ty);tw=(tm>.25).astype(np.float32);norm=max(float(tw.sum()),1.)
    edge=(_filter(gx/mag,tw*tx/(tm+.01))+_filter(gy/mag,tw*ty/(tm+.01)))/norm
    fragment,votes,nparts=_fragment_maps(own,g,config,other,ignored)
    rule=np.clip(.53*direct+.20*np.maximum(edge,0)+.18*np.maximum(bulge,0)+.12*votes
                 -.25*missing-.07*np.minimum(extra,2)-.08*holes-.12*line_ratio,0,1)
    maps={'required_support':direct,'visible_missing':missing,'core_missing':core_missing,
          'same_color_extra':np.clip(extra,0,5),'hole_extra':np.clip(holes,0,3),
          'occluded_fraction':occlusion,'positive_visible_support':visible_positive,
          'edge_alignment':edge,'body_density':body_density,'context_density':context_density,
          'line_explained_ratio':line_ratio,'bulge_ratio':bulge,
          'grid_vote_fraction':votes,'grid_fragment_score':fragment,
          'path_density':path_maps['density'],'residual_density':path_maps['residual_density'],
          'thickness_over_diameter':np.clip(path_maps['thickness']/d,0,3),
          'path_support':_filter(path_maps['path'],g['body'],True),
          'stem_fraction':_filter(path_maps['stem'],g['body'],True),
          'unknown_fraction':_filter(unknown,expected,True),
          'color_confidence':_filter(confidence,expected,True),'rule_score':rule}
    constants={'template_fill':float(expected.sum())/max(float(g['envelope'].sum()),1),
               'template_aspect':float(template['soft'].shape[1])/max(template['soft'].shape[0],1),
               'line_width_over_diameter':line_width/d}
    return maps,constants,dict(fragment_count=nparts,cell_size=max(3,round(d*config.grid_cell_fraction)),
                              line_width=line_width,diameter=d)


def _peaks(field, valid, distance, minimum, cap):
    width=max(3,int(round(2*distance+1)))|1
    maxima=cv2.dilate(field,np.ones((width,width),np.uint8))
    mask=((field>=maxima-1.e-6)&(field>=minimum)&valid).astype(np.uint8)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(mask,8)
    points=[]
    for i in range(1,n):
        x0,y0,w,h,_=stats[i];local=field[y0:y0+h,x0:x0+w]
        eligible=labels[y0:y0+h,x0:x0+w]==i
        ys,xs=np.nonzero(eligible)
        if not len(xs):continue
        # Mean of a tied response plateau avoids top-left scan-order bias.
        x=float(x0+xs.mean());y=float(y0+ys.mean())
        points.append({'x':x,'y':y,'score':float(local[eligible].max()),'source':'grid'})
    points.sort(key=lambda z:z['score'],reverse=True)
    return points[:cap],len(points)


def _source_flags(p):
    return ('path' in p['sources'],'grid' in p['sources'])


def _alternative_centres(field, valid, x, y, diameter, occupied, config):
    """Keep separate, image-supported lobes; never move a centre to a prior.

    The cheap local rule is only a proposal score. A cap can win that rule
    while a nearby marker loses. Both must reach the same final verifier.
    Search two-dimensional local maxima, not a hand-coded vertical shift.
    """
    if not config.retain_alternative_centres:
        return []
    radius = max(2, round(config.alternative_radius_fraction*diameter))
    x0, x1 = max(0, round(x)-radius), min(field.shape[1], round(x)+radius+1)
    y0, y1 = max(0, round(y)-radius), min(field.shape[0], round(y)+radius+1)
    local = field[y0:y1, x0:x1]
    if not local.size:
        return []
    floor = max(.16, .40*float(local.max()))
    peaks, _ = _peaks(local, valid[y0:y1, x0:x1], max(1., .12*diameter), floor, 12)
    result = []
    for p in peaks:
        px, py = p['x']+x0, p['y']+y0
        if (px-x)**2+(py-y)**2 > radius**2:
            continue
        if any((px-qx)**2+(py-qy)**2 < max(2., .20*diameter)**2
               for qx, qy in [*occupied, *[(q[0],q[1]) for q in result]]):
            continue
        result.append((px, py, 'alternative_local_peak'))
        if len(result) >= config.maximum_alternative_centres:
            break
    return result


def extract_candidates(evidence: dict, config: HybridConfig | None=None, *, keep_debug=False):
    """Return image-derived candidates + reusable features, before final scoring."""
    config=config or HybridConfig();start=perf_counter()
    candidates=[];diagnostics=[];debug=[]
    valid=np.asarray(evidence['valid'],bool)
    ignored = np.asarray(evidence.get('ignore_mask', np.zeros(valid.shape)), np.float32)
    observable = 1-ignored
    for si,template in enumerate(evidence['templates']):
        ts=perf_counter();d=max(3.,float(template['diameter']))
        own=np.asarray(evidence['membership'][si],np.float32)*valid*observable
        guide_upper=evidence.get('guide_upper')
        nuisance=np.zeros_like(own) if guide_upper is None else np.asarray(guide_upper[si],np.float32)*valid*observable
        # A repeated dot is not an independent marker fragment. Unlike a band
        # mask, subtraction preserves ink between dots and excess ink at dots.
        own=np.maximum(own-nuisance,0)
        proposal_ignored=1-(1-ignored)*(1-nuisance)
        path,path_maps=propose_path_markers(own,d)
        other=np.asarray(evidence['other'][si],np.float32)*observable
        conf=np.asarray(evidence['colour_confidence'][si],np.float32)*observable
        maps,constants,diag=_window_maps(own,other,np.asarray(evidence['unknown'],np.float32)*observable,conf,
                                       template,path_maps,config,proposal_ignored)
        center_field = maps['rule_score']
        paper_cost = np.zeros_like(center_field)
        paper_report = None
        if config.paper_contradiction_weight:
            from color_paper_contradiction_v46 import cost_map
            paper_cost, paper_report = cost_map(evidence, si, weight=config.paper_contradiction_weight)
            center_field = center_field-config.paper_contradiction_weight*paper_cost
            diag['paper_contradiction'] = paper_report
        grid_field=(.5*maps['grid_fragment_score']+.25*maps['grid_vote_fraction']+
                    .25*np.maximum(maps['edge_alignment'],0))*(.65+.35*maps['required_support'])
        grid,ngrid=_peaks(grid_field,valid,config.candidate_spacing_fraction*d,config.minimum_grid_peak,config.maximum_candidates_per_series)
        merged=[]
        for p in [*path,*grid]:
            x,y=float(p['x']),float(p['y']);ix=int(round(x));iy=int(round(y))
            if not (0<=ix<own.shape[1] and 0<=iy<own.shape[0] and valid[iy,ix]):continue
            match=next((q for q in merged if (q['x']-x)**2+(q['y']-y)**2<=(.16*d)**2),None)
            if match is not None:
                match['sources'].add(p['source']);continue
            merged.append({'x':x,'y':y,'sources':{p['source']}})
        # A repeated whole-body fit can place the center in occluding foreign
        # ink, where target-colour peaks cannot propose it. It still undergoes
        # the ordinary center/window/triangle verification below.
        body=evidence.get('symbol_scale_calibration',{}).get('series',{}).get(str(template['id']),{}).get('split_body',{})
        if body.get('applied'):
            for p in body['centers']:
                merged.append(dict(x=p['x'],y=p['y'],sources={'occluded_whole_body'}))
        # Refine with complete-window image evidence, never with a truth point.
        radius=max(1,round(config.maximum_refinement_fraction*d))
        for p in merged:
            ix,iy=round(p['x']),round(p['y'])
            x0=max(0,ix-radius);x1=min(own.shape[1],ix+radius+1)
            y0=max(0,iy-radius);y1=min(own.shape[0],iy+radius+1)
            objective=center_field[y0:y1,x0:x1].copy()
            yy,xx=np.mgrid[y0:y1,x0:x1]
            objective-=.008*((xx-p['x'])**2+(yy-p['y'])**2)/(d*d)
            objective[~valid[y0:y1,x0:x1]]=-np.inf
            ry,rx=np.unravel_index(int(np.argmax(objective)),objective.shape)
            x=x0+int(rx);y=y0+int(ry)
            flags=_source_flags(p)
            positions=[(float(x),float(y),'window_refined')]
            # A rule-based refinement may shift a partly covered body toward
            # adjacent ink. Retain the original image proposal as a competing
            # hypothesis; the shared verifier/NMS resolves it later.
            if (x-p['x'])**2+(y-p['y'])**2>.5**2:
                positions.append((p['x'],p['y'],'original_proposal'))
            positions.extend(_alternative_centres(center_field, valid, p['x'], p['y'], d,
                [(cx,cy) for cx,cy,_ in positions], config))
            for cx,cy,position_kind in positions:
                ix,iy=round(cx),round(cy)
                values={k:float(v[iy,ix]) for k,v in maps.items()}
                values.update(constants,from_path=float(flags[0]),from_grid=float(flags[1]),
                              source_agreement=float(all(flags)))
                if values['required_support']<.12:continue
                record={'x':float(cx),'y':float(cy),'series_id':str(template['id']),
                        'series_label':template.get('label',str(template['id'])),
                        'diameter':d,'rgb':np.asarray(template['rgb']).tolist(),
                        'sources':sorted(p['sources']),'features':[values[n] for n in FEATURE_NAMES],
                        'rule_score':values['rule_score'],'proposal_x':p['x'],'proposal_y':p['y'],
                        'position_kind':position_kind}
                if paper_report is not None:
                    record['center_scoring'] = dict(version=paper_report['version'],
                        applied=paper_report['applied'], reason=paper_report['reason'],
                        base_score=values['rule_score'], paper_cost=float(paper_cost[iy,ix]),
                        weight=config.paper_contradiction_weight,
                        penalty=float(config.paper_contradiction_weight*paper_cost[iy,ix]),
                        score=float(center_field[iy,ix]))
                candidates.append(record)
        diag.update(series_id=str(template['id']),path_proposals=len(path),grid_peaks_before_cap=ngrid,
                    grid_proposals=len(grid),merged_proposals=len(merged),seconds=perf_counter()-ts)
        if 'symbol_scale' in template:
            diag.update(symbol_scale=template['symbol_scale'],legend_diameter=template['legend_diameter'],
                        scale_policy='shared_symbol')
        diagnostics.append(diag)
        if keep_debug:
            debug.append({'membership':own,'grid_score':grid_field,'path_density':path_maps['density'],
                          'rule_score':maps['rule_score'],'missing':maps['visible_missing'],
                          'same_color_extra':maps['same_color_extra'],
                          'center_score':center_field,'paper_contradiction_cost':paper_cost})
    return {'candidates':candidates,'diagnostics':diagnostics,'debug':debug,
            'seconds':perf_counter()-start,'feature_names':FEATURE_NAMES,'config':asdict(config)}


def forest_scores(features,model):
    """Evaluate a self-contained JSON forest without executable pickle files."""
    x=np.asarray(features,np.float32)
    if x.size==0:return np.zeros(0,np.float32)
    if model.get('feature_names')!=FEATURE_NAMES:raise ValueError('Incompatible feature schema')
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES):raise ValueError('Invalid feature matrix')
    answer=np.zeros(len(x),np.float64)
    for t in model['trees']:
        left=np.asarray(t['children_left'],int);right=np.asarray(t['children_right'],int)
        feature=np.asarray(t['feature'],int);threshold=np.asarray(t['threshold'],float)
        probability=np.asarray(t['positive_fraction'],float);nodes=np.zeros(len(x),int)
        active=np.ones(len(x),bool)
        while np.any(active):
            ii=np.flatnonzero(active);nn=nodes[ii];split=left[nn]>=0
            active[ii[~split]]=False;ii=ii[split];nn=nn[split]
            if len(ii):nodes[ii]=np.where(x[ii,feature[nn]]<=threshold[nn],left[nn],right[nn])
        answer+=probability[nodes]
    return answer/max(len(model['trees']),1)


def select_candidates(candidates, *, model=None, threshold=None, branch='hybrid',
                      config: HybridConfig | None=None):
    config=config or HybridConfig()
    pool=[dict(c) for c in candidates if branch=='hybrid' or branch in c['sources']]
    if threshold is None:threshold=float(model.get('threshold',.5)) if model else .36
    scores=forest_scores([c['features'] for c in pool],model) if model else np.array([c['rule_score'] for c in pool])
    ranked=[]
    for c,s in zip(pool,scores):
        c['score']=float(s)
        if s>=threshold:ranked.append(c)
    ranked.sort(key=lambda c:c['score'],reverse=True);accepted=[]
    for c in ranked:
        duplicate=False
        for q in accepted:
            if c['series_id']!=q['series_id']:continue
            source=all(k in c and k in q for k in ['x_px','y_px','diameter_source'])
            xkey,ykey,dkey=('x_px','y_px','diameter_source') if source else ('x','y','diameter')
            distance=config.final_spacing_fraction*min(c[dkey],q[dkey])
            if (c[xkey]-q[xkey])**2+(c[ykey]-q[ykey])**2<distance**2:
                duplicate=True;break
        if not duplicate:accepted.append(c)
    return accepted


def detect_markers(image_bgr,plot_box,series_specs,*,model=None,branch='hybrid',max_side=1400,threshold=None):
    """Public image-only API; metadata names label outputs but are not features."""
    from color_marker_evidence import prepare_evidence
    start=perf_counter()
    if isinstance(model,(str,Path)):model=json.loads(Path(model).read_text(encoding='utf-8'))
    try:
        evidence=prepare_evidence(image_bgr,plot_box,series_specs,max_side=max_side)
    except ValueError as exc:
        if not str(exc).startswith('No usable legend templates:'):raise
        return {'points':[],'diagnostics':[],'template_errors':[str(exc)],
                'feature_names':FEATURE_NAMES,'seconds':perf_counter()-start,'model_used':model is not None}
    result=extract_candidates(evidence)
    sx=evidence.get('scale_x',evidence.get('scale',1.));sy=evidence.get('scale_y',evidence.get('scale',1.))
    offset=evidence.get('source_center_offset',plot_box[:2])
    for p in result['candidates']:
        p['x_px']=p['x']/sx+offset[0];p['y_px']=p['y']/sy+offset[1]
        p['diameter_source']=p['diameter']/(.5*(sx+sy))
    points=select_candidates(result['candidates'],model=model,branch=branch,threshold=threshold)
    return {'points':points,'diagnostics':result['diagnostics'],'feature_names':FEATURE_NAMES,
            'template_errors':evidence.get('template_errors',[]),
            'seconds':perf_counter()-start,'model_used':model is not None}

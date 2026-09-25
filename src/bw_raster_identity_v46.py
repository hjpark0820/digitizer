"""Common-observation competition of measured legend rasters and centered lines.

The same source patch, center, weight and denominator are used for every
identity, including the line-only null. No canonical glyph or reference point
is consulted. Geometry/grid evidence remains a proposal, not a final veto.
"""
from dataclasses import dataclass, asdict
import time
import cv2
import numpy as np
import bw_observed_raster_v46 as R
import bw_layered_composite_v46 as L
from bw_centered_composite_v46 import stroke

VERSION = 'common-observed-raster-identity-v1'


@dataclass(frozen=True)
class Config:
    maximum_loss: float = .55
    minimum_improvement: float = .04
    identity_margin: float = .025
    minimum_independent_fraction: float = .20
    minimum_visible: float = .70
    line_cost: float = .015
    boundary_weight: float = .35
    maximum_diameter: float = 28.
    maximum_centres: int = 2000
    angles: tuple = tuple(range(0, 180, 30))
    widths: tuple = (1., 2.)
    amplitudes: tuple = (.25, .5, .85)
    gains: tuple = (.85, 1., 1.15)


def features(images, weight, boundary_weight=.35):
    a = np.atleast_3d(images) if images.ndim < 2 else images
    if a.ndim == 2: a = a[None]
    return np.concatenate([(a*np.sqrt(weight)).reshape(len(a), -1),
        (np.diff(a, axis=1)*np.sqrt(boundary_weight*np.minimum(weight[1:], weight[:-1]))).reshape(len(a), -1),
        (np.diff(a, axis=2)*np.sqrt(boundary_weight*np.minimum(weight[:, 1:], weight[:, :-1]))).reshape(len(a), -1)], axis=1).astype(np.float32)


def decide(scores, cfg=Config()):
    """An unobservable/near-tied rival cannot be silently discarded."""
    ranked = sorted(scores, key=lambda k: scores[k]['loss'])
    if not ranked: return None, 'no_models', 0.
    first = ranked[0]; a = scores[first]
    margin = scores[ranked[1]]['loss']-a['loss'] if len(ranked)>1 else 1.
    if not a['passed']: return None, 'marker_not_independently_supported', float(margin)
    if margin < cfg.identity_margin: return None, 'ambiguous_raster_identity', float(margin)
    return first, 'observed_raster_winner', float(margin)


class Competition:
    def __init__(self, image, plot, exclusions, templates, cfg=Config(), *, legend_reports=None):
        self.cfg=cfg; self.ready=False; self.cache={}; self.banks={}
        self.report=dict(version=VERSION, status='not_applicable', config=asdict(cfg),
                         reference_points_read=False, source_pixels_modified=False)
        chroma=image.max(axis=2).astype(float)-image.min(axis=2)
        if np.count_nonzero(chroma>24)>.01*chroma.size:
            self.report['status']='colour_source_not_supported'; return
        if image.shape[0]*image.shape[1]>1_000_000 or len(templates)>10:
            self.report['status']='resource_limit_image_or_swatch_count';return
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.ink=np.clip((np.percentile(gray,95)-gray)/255,0,1)
        self.templates={}; self.prepared={}; self.source_meta={}
        legend_by_id={r['swatch_id']:r for r in (legend_reports or [])}
        for t in templates:
            a,b,c,d=map(int,t.swatch_box)
            rec,f=R.extract(gray[b:d,a:c],connected_line=legend_by_id.get(t.key,{}).get('connected_line'))
            if f is None or rec['diameter']>cfg.maximum_diameter:
                self.report['status']='unresolved_or_oversized_rival'; return
            self.templates[t.key]=(rec,f)
            self.prepared[t.key],self.source_meta[t.key]=L.prepare_source(rec,f)
        from bw_observed_scale_v46 import apply
        calibration,_=apply(image,plot,(0,0,0,0),self.templates,exclusions=exclusions)
        self.scales={k:v['scale'] for k,v in calibration['series'].items()}
        self.report['calibration']=calibration['series']
        # Unmeasured rivals remain competitors over bounded scale hypotheses;
        # they are neither omitted nor assigned an invented 1x calibration.
        self.scale_hypotheses={k:((v,) if v is not None else R.Config().scales) for k,v in self.scales.items()}
        self.radius=max(L.variants(rec,self.prepared[k],s,(0.,0.))[3]
                        for k,(rec,f) in self.templates.items() for s in self.scale_hypotheses[k])
        self.report.update(status='ready',scales=self.scales,scale_hypotheses=self.scale_hypotheses,
                           common_radius=self.radius,source_meta=self.source_meta)
        self.ready=True

    def _bank(self):
        if self.banks: return
        cfg=self.cfg; size=2*self.radius+1; center=(self.radius,self.radius)
        all_variants={}; weight=np.ones((size,size),np.float32)
        def pad(a,fill=0.):
            r=(size-len(a))//2
            return np.pad(a,r,constant_values=fill)
        for key,(rec,f) in self.templates.items():
            options=getattr(self,'scale_hypotheses',{}).get(key,(self.scales[key],))
            for scale in options:
                m,w,c,r=L.variants(rec,self.prepared[key],scale,(0.,0.))
                # Unknown conditional ink, including a zero estimate, is not
                # observed paper. All rivals receive the SAME uncertainty.
                uncertain=((w>1.e-5)&(w<.2) if getattr(self,'observation',None) is None
                           else (w<.2)&(m>.02))
                # Colour subclasses already carry separate own/other/paper
                # visibility. Preserve that validated policy in this BW repair.
                weight=np.minimum(weight,np.where(pad(uncertain),.2,1.).astype(np.float32))
                all_variants[(key,scale)]=(pad(m),pad(c),pad(w))
        # The computational canvas includes padding for interpolation/lines.
        # It is not all marker evidence: remote neighbouring markers and axes
        # must not veto every family. Use the UNION of observed body coverage,
        # with a one-pixel exterior band, shared by every symbol AND null.
        support=np.maximum.reduce([np.maximum(m,c) for m,c,w in all_variants.values()])>.03
        common_region=cv2.dilate(support.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        weight*=common_region
        singles=[(amp*stroke(size,center,a,w),dict(angle=a,width=w,amplitude=amp))
                 for a in cfg.angles for w in cfg.widths for amp in cfg.amplitudes]
        families=[([],[])]+[([a],[p]) for a,p in singles]
        families += [([a,b],[pa,pb]) for a,pa in singles if pa['angle']==90
                     for b,pb in singles if pb['angle']!=90]
        nulls=[]
        for lines,pars in families:
            full=np.zeros((size,size),np.float32)
            for line in lines:full=1-(1-full)*(1-line)
            nulls.append(full)
        self.weight=weight; self.nulls=np.asarray(nulls)
        self.family_lines=[lines for lines,pars in families]
        self.context_body=common_region
        nf=features(self.nulls,weight,cfg.boundary_weight)
        self.null_bank=(nf,np.sum(nf*nf,axis=1),np.array([len(p) for _,p in families])*cfg.line_cost)
        for (key,scale),(m,c,w) in all_variants.items():
            models=[];pars=[];line_ids=[]
            for j,(lines,lp) in enumerate(families):
                orders=[tuple('front' for _ in lines)]
                if self.source_meta[key]['source_open'] and lines:
                    orders += [('behind',)] if len(lines)==1 else [('front','behind'),('behind','front'),('behind','behind')]
                for order in orders:
                    for gain in cfg.gains:
                        body=np.clip(m*gain,0,1)
                        models.append(L.compose(body,np.maximum(c,body),lines,order))
                        pars.append(dict(gain=gain,raster_scale=scale,line_count=len(lines),lines=[dict(p,order=o) for p,o in zip(lp,order)]))
                        line_ids.append(j)
            ff=features(np.asarray(models),weight,cfg.boundary_weight)
            b=dict(features=ff,norms=np.sum(ff*ff,axis=1),
                cost=np.array([p['line_count']*cfg.line_cost for p in pars]),
                parameters=pars,line_ids=line_ids,marker=m,cover=c,marker_weight=w,
                variants={scale:(m,c,w)})
            if key not in self.banks:self.banks[key]=b
            else:
                old=self.banks[key]
                for name in ('features','norms','cost'):old[name]=np.concatenate((old[name],b[name]))
                for name in ('parameters','line_ids'):old[name].extend(b[name])
                old['variants'].update(b['variants'])

    def score(self, centers):
        if not self.ready:return []
        self._bank(); cfg=self.cfg; r=self.radius; size=2*r+1
        # Subpixel source sampling, identical for all candidate identities.
        patches=np.asarray([cv2.getRectSubPix(self.ink,(size,size),(float(x),float(y))) for x,y in centers])
        f=features(patches,self.weight,cfg.boundary_weight)
        energy=np.maximum(np.sum(f*f,axis=1),1.e-6)
        nf,nn,nc=self.null_bank
        nl=np.maximum(energy[:,None]-2*f@nf.T+nn,0)/energy[:,None]+nc
        # A nuisance line must have exterior support. Otherwise arbitrary
        # interior strokes could turn one symbol into an unsupported rival.
        # Independent full-window winners remain protected in filter_records;
        # recovery can also reuse a verified body at the same center.
        yy,xx=np.indices((size,size));line_ok={}
        for lines in self.family_lines:
            for line in lines:
                ident=id(line)
                if ident in line_ok:continue
                # Two opposite half-planes; at least one continuation permits
                # an endpoint. The body itself cannot authorize a line.
                ys,xs=np.nonzero(line>.01)
                horizontal=np.ptp(xs)>=np.ptp(ys)
                axis=(xx-r) if horizontal else (yy-r)
                ray=[]
                for sign in (-1,1):
                    w=(~self.context_body)*(axis*sign>0)
                    mass=float(np.sum(line*w))
                    ray.append(np.sum(np.minimum(patches,line)*w,axis=(1,2))/max(mass,1.e-6))
                line_ok[ident]=np.maximum(*ray)>=.45
        valid=np.stack([np.logical_and.reduce([line_ok[id(line)] for line in lines])
                        if lines else np.ones(len(centers),bool) for lines in self.family_lines],axis=1)
        nl=np.where(valid,nl,np.inf)
        null_best=nl.min(axis=1); null_id=nl.argmin(axis=1)
        outputs=[{} for _ in centers]
        for key,b in self.banks.items():
            losses=np.maximum(energy[:,None]-2*f@b['features'].T+b['norms'],0)/energy[:,None]+b['cost']
            losses=np.where(valid[:,b['line_ids']],losses,np.inf)
            indices=losses.argmin(axis=1)
            for i,j in enumerate(indices):
                full=self.nulls[b['line_ids'][j]]; gain=b['parameters'][j]['gain']
                m,cover,marker_weight=b['variants'][b['parameters'][j]['raster_scale']]
                mass=marker_weight*m*m; independent=mass*(1-full)**2
                fraction=float(independent.sum()/max(mass.sum(),1.e-8))
                remain=np.clip((patches[i]-full)/np.maximum(1-full,.02),0,1)
                visible=float(np.sum(independent*np.clip(remain/np.maximum(gain*m,1.e-6),0,1))/max(independent.sum(),1.e-8))
                loss=float(losses[i,j]); improvement=float(null_best[i]-loss)
                outputs[i][key]=dict(loss=loss, line_only_loss=float(null_best[i]),line_improvement=improvement,
                    line_continuation_observed=bool(valid[i,b['line_ids'][j]]),
                    independent_fraction=fraction,independent_visible=visible,parameters=b['parameters'][j],
                    passed=bool(loss<=cfg.maximum_loss and improvement>=cfg.minimum_improvement and
                                fraction>=cfg.minimum_independent_fraction and visible>=cfg.minimum_visible))
        result=[]
        for center,scores in zip(centers,outputs):
            winner,reason,margin=decide(scores,cfg)
            result.append(dict(version=VERSION,center=list(center),common_radius=r,winner=winner,
                               reason=reason,margin=margin,scores=scores))
        return result

    def filter_records(self, records):
        if not self.ready:return records
        centers=list(dict.fromkeys((float(p['aligned_x']),float(p['aligned_y'])) for p in records))
        if len(centers)>self.cfg.maximum_centres:
            self.report['status']='final_candidate_budget_legacy_fallback'
            kept=[]
            for p in records:
                if p.get('grid_identity',{}).get('winner') not in (None,p['template']):
                    p['exclusion_reason']='grid_identity_conflict'
                else:kept.append(p)
            return kept
        start=time.perf_counter()
        values=[]
        for a in range(0,len(centers),64):values.extend(self.score(centers[a:a+64]))
        lookup=dict(zip(centers,values)); kept=[]
        for p in records:
            q=lookup[(float(p['aligned_x']),float(p['aligned_y']))]
            p['legacy_grid_identity']=p.pop('grid_identity',{})
            p['raster_identity']=q
            score=q['scores'][p['template']]
            from bw_raster_context_v46 import native_identity_supported
            if native_identity_supported(p, self.templates):
                # A differently weighted, smaller window cannot silently
                # overturn independent full-window model/line-null evidence.
                p['raster_native_preserved']='independent_full_window_identity'
                kept.append(p)
            elif q['winner']==p['template']:
                p['selection_rank']=1.-score['loss']; kept.append(p)
            elif score['passed']:
                # Retain physically plausible alternatives in the typed pool.
                p['exclusion_reason']='identity_ambiguous'
                p['raster_deferred_reason']=q['reason'] if q['winner'] is None else 'another_observed_raster_wins'
            else:
                # A line-only/unsupported-body fit is not a strong identity
                # alternative for Step 5. Recovery may independently revisit
                # the original source at another center, never this rejection.
                p['exclusion_reason']='raster_marker_not_supported'
                p['raster_deferred_reason']='insufficient_independent_marker_evidence'
        self.report.update(status='completed',centres=len(centers),verified_input=len(records),
                           retained=len(kept),seconds=time.perf_counter()-start,comparisons=values)
        return kept

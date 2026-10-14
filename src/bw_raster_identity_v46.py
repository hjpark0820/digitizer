"""Common-observation competition of measured legend rasters and centered lines.

The same source patch, center, weight and denominator are used for every
identity, including the line-only null. No canonical glyph or reference point
is consulted. Geometry/grid evidence remains a proposal, not a final veto.
"""
from dataclasses import dataclass, asdict
from copy import copy
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
    maximum_body_hypotheses: int = 12
    angles: tuple = tuple(range(0, 180, 30))
    widths: tuple = (1., 2.)
    amplitudes: tuple = (.25, .5, .85)
    gains: tuple = (.85, 1., 1.15)
    visible_first_detection: bool = False


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


def joint_decide(best, supported, cfg=Config()):
    """decide() over one body's shared set of localisations.

    best[k]: identity k's lowest loss anywhere in the set; supported[k]: its
    lowest loss where it also passes the unchanged support tests. The winner is
    the identity with the best SUPPORTED localisation, and it must beat every
    rival's best localisation (supported or not) by the unchanged identity
    margin, so an unsupported but better-fitting rival still blocks it.
    """
    if not best:return None,'no_models',0.
    if not supported:
        first=min(best,key=lambda k:best[k]['loss'])
        rivals=[best[k]['loss'] for k in best if k!=first]
        return None,'marker_not_independently_supported',float(min(rivals)-best[first]['loss']) if rivals else 1.
    first=min(supported,key=lambda k:supported[k]['loss'])
    rivals=[best[k]['loss'] for k in best if k!=first]
    margin=float(min(rivals)-supported[first]['loss']) if rivals else 1.
    if margin<cfg.identity_margin:
        return None,'ambiguous_raster_identity',margin
    return first,'observed_raster_winner',margin


def verified_body_extent(record):
    """Transfer physical native pixels, never a scale with another reference.

    The polygon is the hull of the mask actually verified by the full window.
    It only proposes geometry; every identity and the line-only null must still
    be tested on a common observation. Missing/weak geometry grants nothing.
    """
    g=record.get('geometry_first',{})
    if (record.get('decision')!='verified' or g.get('decision')!='compatible'
        or g.get('geometry_decision')!='verified'
        or g.get('native_context',{}).get('passed') is False
        or record.get('ink_evidence',{}).get('direct_required_recall',0.)<.80
        or record.get('boundary_recall',0.)<.80
        or record.get('missing_fraction',1.)>.25):
        return None
    polygon=np.asarray(record.get('footprint_polygon',[]),np.float64)
    if polygon.ndim!=2 or polygon.shape[1:]!=(2,) or len(polygon)<3 or not np.isfinite(polygon).all():
        return None
    extent=np.ptp(polygon,axis=0)+1.
    if min(extent)<3. or max(extent)/min(extent)>2.:
        return None
    center=np.array([record['aligned_x'],record['aligned_y']],float)
    if not np.all((center>=polygon.min(axis=0)-.5)&(center<=polygon.max(axis=0)+.5)):
        return None
    return dict(version='verified_native_body_extent_v1',candidate_index=record.get('candidate_index'),
        source='full_window_verified_rendered_mask_hull',center=center.tolist(),
        native_extent=extent.tolist(),source_scale_not_transferred=True,
        identity_supplied_by='complete_same_domain_raster_and_line_null_competition')


def entry_territory_source(gray, box, legend, rec, f):
    """Identity raster of a legend entry whose box raster carries foreign ink.

    ``rec, f`` = R.extract on the swatch box. The box is the caller's entry
    box (often widened by the 1-2 px border-recovery pad), not the extent of
    this entry's ink. When the box extraction observed no connector band it
    seeded the body from every box column, so connector stubs ending before
    the box's 20% flank, a legend-frame or reference-line column and label
    text became marker body. That is measured as a body reaching beyond the
    legend glyph box (the glyph the window stage verifies with) by more than
    one straddling boundary pixel; the one pixel is the quantisation between
    the two extent rules (legend: bulge columns +-1 or antialiased support;
    raster: columns above 25% of the strongest column), not a structure.

    Such an entry is re-extracted with the legend's measured extent of the
    entry's own connected ink as territory and the legend glyph box as the
    body-seed domain (R.extract(territory, body_bounds)): the connector test
    then reads where an attached connector really ends, and only glyph ink
    seeds the body. The re-extraction crop is the box extended to hold the
    2-px observed exterior around that ink (R.extract's own raster support,
    body +-2 px), which the box clips where the glyph touches it. Any other
    extraction is
    returned unchanged (byte-identical raster). A box extraction that already
    observed a connector band has its flanks excluded; a body beyond the
    glyph there is glyph ink the legend did not include, and is kept.
    """
    glyph = legend.get('glyph_box')
    ink = (legend.get('line_analysis') or {}).get('entry_ink_box')
    if (f is None or glyph is None or ink is None or legend.get('connected_line') is None
            or rec.get('connector_detected')):
        return rec, f
    a, b, c, d = box
    gx0, gy0 = int(np.floor(glyph[0])) - a, int(np.floor(glyph[1])) - b
    gx1, gy1 = int(np.ceil(glyph[2])) - a, int(np.ceil(glyph[3])) - b
    x0, y0, x1, y1 = rec['bbox']
    if gx0 - 1 <= x0 and gy0 - 1 <= y0 and x1 <= gx1 + 1 and y1 <= gy1 + 1:
        return rec, f
    # The re-extraction crop also holds the raster's own 2-px observed
    # exterior around the entry ink, which a box ending on the glyph clips.
    H, W = gray.shape
    A, B = max(0, min(a, int(ink[0]) - 2)), max(0, min(b, int(ink[1]) - 2))
    C, D = min(W, max(c, int(ink[2]) + 2)), min(H, max(d, int(ink[3]) + 2))
    tx0, ty0 = max(0, int(ink[0]) - A), max(0, int(ink[1]) - B)
    tx1, ty1 = min(C - A, int(ink[2]) - A), min(D - B, int(ink[3]) - B)
    if tx1 - tx0 < 4 or ty1 - ty0 < 3:
        return rec, f
    bounds = (max(0, gx0 + a - A), max(0, gy0 + b - B), min(C - A, gx1 + a - A), min(D - B, gy1 + b - B))
    own, fields = R.extract(gray[B:D, A:C], connected_line=legend.get('connected_line'),
                            territory=(tx0, ty0, tx1, ty1), body_bounds=bounds)
    if fields is None:
        return rec, f
    own['source_crop'] = [A, B, C, D]
    own['box_extraction'] = dict(box=[a, b, c, d], bbox=list(rec['bbox']), diameter=rec['diameter'],
                                 connector_detected=rec['connector_detected'])
    return own, fields


def screen_tone_calibration(image, plot, exclusions, templates, legend_by_id, cfg):
    """Scale of legend rasters measured on the tone map of a detected bitonal screen.

    bw_observed_scale_v46.boundary_evidence multiplies adjacent-pixel gray
    differences pair by pair. On a bitonal screen every instance of a marker
    is a different set of inked pixels (bw_halftone_screen_v46), so that
    product measures the screen phase: no plot marker verifies the legend
    raster and the series is left unmeasured. The tone map (ink density over
    one period of the detected screen, the pipeline's own descreen) is the
    continuous-tone image legend and plot share. The unchanged apply() is run
    on it, with legend rasters extracted from the same tone map by the same
    extractor. Only the SCALE is returned: a zoom factor of the legend raster.
    The competition keeps scoring source pixels with its source rasters.
    No screen detected, or a tone raster that is unresolved or oversized:
    nothing is measured and every scale stays as the source pixels left it.
    """
    from bw_halftone_screen_v46 import detect_screen,descreen
    from bw_observed_scale_v46 import apply
    screen=detect_screen(image)
    report=dict(version='screen-tone-scale-calibration-v1',status='no_bitonal_screen',
                screen_detected=bool(screen.get('detected')),period=screen.get('period'),
                scales={},applied=[])
    if not screen.get('detected'):return report
    tone=descreen(image,screen)
    gray=cv2.cvtColor(tone,cv2.COLOR_BGR2GRAY).astype(np.float32)
    rasters={}
    for t in templates:
        a,b,c,d=map(int,t.swatch_box)
        legend=legend_by_id.get(t.key,{})
        rec,f=R.extract(gray[b:d,a:c],connected_line=legend.get('connected_line'))
        if f is None or rec['diameter']>cfg.maximum_diameter:
            report['status']='tone_raster_unresolved_or_oversized';return report
        rec,f=entry_territory_source(gray,(a,b,c,d),legend,rec,f)
        if rec['diameter']>cfg.maximum_diameter:
            report['status']='tone_raster_unresolved_or_oversized';return report
        rasters[t.key]=(rec,f)
    calibration,_=apply(tone,plot,(0,0,0,0),rasters,exclusions=exclusions)
    report.update(status='calibrated',
        scales={k:v['scale'] for k,v in calibration['series'].items()},
        reasons={k:v['reason'] for k,v in calibration['series'].items()},
        anchor_profiles={k:len(v['anchor_profiles']) for k,v in calibration['series'].items()},
        tone_raster_diameters={k:rec['diameter'] for k,(rec,f) in rasters.items()})
    return report


class Competition:
    def __init__(self, image, plot, exclusions, templates, cfg=Config(), *, legend_reports=None,
                 source_models=None, approved_scales=None):
        self.cfg=cfg; self.ready=False; self.cache={}; self.banks={}
        self.visible_exclusions=tuple(tuple(map(float, box)) for box in (exclusions or []))
        self.visible_plot_bounds=tuple(map(float, plot))
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
        if source_models is not None:
            from bw_shared_window_source_v46 import validate_source_models,VERSION as SOURCE_VERSION
            validate_source_models(source_models,[t.key for t in templates],approved_scales)
            for t in templates:
                source=source_models[t.key];rec=source['record'];f=source['fields']
                if rec['diameter']>cfg.maximum_diameter:
                    self.report['status']='unresolved_or_oversized_rival'; return
                self.templates[t.key]=(rec,f)
                self.prepared[t.key]=source['prepared']
                self.source_meta[t.key]=source['meta']
            self.scales={k:None if approved_scales[k] is None else float(approved_scales[k]) for k in self.templates}
            self.report.update(source_model_version=SOURCE_VERSION,source_model_policy='shared_native_window',
                source_models_shared_with_window=True,
                source_reextracted=False,independent_scale_calibration=False,
                calibration={k:dict(scale=v,status='unresolved' if v is None else 'shared_native_approved',
                                   scale_reference='native_window_template') for k,v in self.scales.items()})
        else:
            if approved_scales is not None:
                raise ValueError('approved_scales requires shared source_models')
            legend_by_id={r['swatch_id']:r for r in (legend_reports or [])}
            for t in templates:
                a,b,c,d=map(int,t.swatch_box)
                legend=legend_by_id.get(t.key,{})
                rec,f=R.extract(gray[b:d,a:c],connected_line=legend.get('connected_line'))
                if f is None or rec['diameter']>cfg.maximum_diameter:
                    self.report['status']='unresolved_or_oversized_rival'; return
                # The roster gate above stays on the box rasters: the repair
                # never enables a competition that the snapshot disables.
                rec,f=entry_territory_source(gray,(a,b,c,d),legend,rec,f)
                if rec['diameter']>cfg.maximum_diameter:
                    self.report['status']='unresolved_or_oversized_rival'; return
                self.templates[t.key]=(rec,f)
                self.prepared[t.key],self.source_meta[t.key]=L.prepare_source(rec,f,
                    restore_filled_interior=bool(legend.get('restored_filled_interior_pixels',0)))
            from bw_observed_scale_v46 import apply
            calibration,_=apply(image,plot,(0,0,0,0),self.templates,exclusions=exclusions)
            self.scales={k:v['scale'] for k,v in calibration['series'].items()}
            self.report['calibration']=calibration['series']
            if any(v is None for v in self.scales.values()):
                # A series the source pixels leave unmeasured is measured once
                # more on the tone map of a detected bitonal screen (same
                # apply(), same thresholds). A series measured on the source
                # pixels keeps that scale; a series neither image verifies
                # stays unmeasured (bounded hypotheses below, no invented 1x).
                tone=screen_tone_calibration(image,plot,exclusions,templates,legend_by_id,cfg)
                for k,v in tone['scales'].items():
                    if self.scales[k] is None and v is not None:
                        self.scales[k]=float(v);tone['applied'].append(k)
                self.report['screen_tone_calibration']=tone
        # Unmeasured rivals remain competitors over bounded scale hypotheses;
        # they are neither omitted nor assigned an invented 1x calibration.
        self.scale_hypotheses={k:((v,) if v is not None else R.Config().scales) for k,v in self.scales.items()}
        self.radius=max(self._variant(k,s)[3]
                        for k,(rec,f) in self.templates.items() for s in self.scale_hypotheses[k])
        self.report.update(status='ready',scales=self.scales,scale_hypotheses=self.scale_hypotheses,
                           common_radius=self.radius,source_meta=self.source_meta)
        self.ready=True

    def _variant(self, key, scale):
        if getattr(self,'report',{}).get('source_model_policy') == 'shared_native_window':
            from bw_shared_window_source_v46 import native_window_variant
            return native_window_variant(self.templates[key][0],self.prepared[key],scale)
        return L.variants(self.templates[key][0],self.prepared[key],scale,self._phase())

    def _phase(self):
        """Sub-pixel offset of every model (marker AND nuisance lines) from the
        integer observation centre. The source patch itself is never resampled.
        (0,0) everywhere except the joint body decision's phase copies."""
        return tuple(map(float,getattr(self,'phase',(0.,0.))))

    def _phase_competitor(self, phase):
        """Same rivals, scales, weights and lines, rendered half a pixel apart."""
        phase=tuple(map(float,phase))
        if phase==self._phase():return self
        banks=self.__dict__.setdefault('phase_competitors',{})
        if phase not in banks:
            b=copy(self);b.banks={};b.cache={};b.phase=phase;b.phase_competitors={}
            banks[phase]=b
        return banks[phase]

    def _bank(self):
        if self.banks: return
        cfg=self.cfg; size=2*self.radius+1; ph=self._phase()
        center=(self.radius+ph[0],self.radius+ph[1])
        all_variants={}; weight=np.ones((size,size),np.float32)
        def pad(a,fill=0.):
            r=(size-len(a))//2
            return np.pad(a,r,constant_values=fill)
        for key,(rec,f) in self.templates.items():
            options=getattr(self,'scale_hypotheses',{}).get(key,(self.scales[key],))
            for scale in options:
                m,w,c,r=self._variant(key,scale)
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
        yy,xx=np.indices((size,size));line_ok={};ph=self._phase()
        for lines in self.family_lines:
            for line in lines:
                ident=id(line)
                if ident in line_ok:continue
                # Two opposite half-planes; at least one continuation permits
                # an endpoint. The body itself cannot authorize a line.
                ys,xs=np.nonzero(line>.01)
                horizontal=np.ptp(xs)>=np.ptp(ys)
                axis=(xx-r-ph[0]) if horizontal else (yy-r-ph[1])
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

    def filter_records(self, records, *, preserve_supported_alternatives=False):
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
        for a in range(0,len(centers),64):values.extend(self.score_cached(centers[a:a+64]))
        lookup=dict(zip(centers,values))
        # A verified physical body can correct a different calibration's
        # wrong-size hypothesis, but cannot transfer its identity verdict.
        # All rivals retain their original scales AND receive each identical
        # native body-size hypothesis before one shared-domain comparison.
        physical=self._physical_body_comparisons(records,lookup)
        # The per-record verdicts above are final unless they are inconsistent
        # for one physical body claimed by several series; only then is that
        # body re-decided once, jointly (see _joint_body_decisions).
        joint=self._joint_body_decisions(records,lookup,physical)
        kept=[]
        for p in records:
            center=(float(p['aligned_x']),float(p['aligned_y']))
            q=physical.get(center,lookup[center])
            if center in physical:
                p['raster_calibrated_identity']=lookup[center]
            p['legacy_grid_identity']=p.pop('grid_identity',{})
            own=q['scores'][p['template']]
            decision=joint.get(id(p))
            if decision is not None:
                p['raster_joint_body']=decision['summary']
                if decision['action'] is not None:
                    # Only the records the joint decision changes carry it
                    # as their identity; their own verdict stays recorded.
                    p['raster_own_centre_identity']=q
                    q=decision['comparison']
            p['raster_identity']=q
            score=q['scores'][p['template']]
            from bw_raster_context_v46 import native_identity_supported
            if native_identity_supported(p, self.templates):
                # A differently weighted, smaller window cannot silently
                # overturn independent full-window model/line-null evidence.
                p['raster_native_preserved']='independent_full_window_identity'
                if preserve_supported_alternatives:p['pre_contour_raster_retained']=True
                kept.append(p)
            elif q['winner']==p['template']:
                if preserve_supported_alternatives:p['pre_contour_raster_retained']=True
                # Rank stays the record's own-centre measurement, comparable
                # with every other record's rank (all measured at own centres).
                p['selection_rank']=1.-own['loss']; kept.append(p)
            elif score['passed']:
                # Retain physically plausible alternatives in the typed pool.
                p['exclusion_reason']='identity_ambiguous'
                p['raster_deferred_reason']=q['reason'] if q['winner'] is None else 'another_observed_raster_wins'
                if preserve_supported_alternatives:
                    # A supported body with unresolved identity is input for
                    # further identity evidence, NOT an active detection yet.
                    p['pre_contour_raster_retained']=False
                    p['selection_rank']=1.-own['loss']
                    kept.append(p)
            else:
                # A line-only/unsupported-body fit is not a strong identity
                # alternative for Step 5. Recovery may independently revisit
                # the original source at another center, never this rejection.
                p['exclusion_reason']='raster_marker_not_supported'
                p['raster_deferred_reason']='insufficient_independent_marker_evidence'
        if self.cfg.visible_first_detection and getattr(self,'observation',None) is None:
            from bw_visible_marker_v46 import rescue_records
            kept=rescue_records(self,records,kept)
        self.report.update(status='completed',centres=len(centers),verified_input=len(records),
                           retained=len(kept),seconds=time.perf_counter()-start,
                           comparisons=[physical.get(center,lookup[center]) for center in centers],
                           physical_body_centres=len(physical),
                           joint_bodies=list({id(d['summary']):d['summary'] for d in joint.values()}.values()))
        if preserve_supported_alternatives:
            self.report.update(supported_alternatives_preserved=True,
                legacy_retained=sum(bool(p.get('pre_contour_raster_retained')) for p in kept),
                deferred_for_contour=sum(not p.get('pre_contour_raster_retained',True) for p in kept),
                retained_count_scope='temporary_identity_input_not_active_points')
        return kept

    def _joint_body_decisions(self, records, calibrated, physical):
        """Re-decide a physical body only where the per-record verdicts are inconsistent.

        Verified fits of different series that claim one body (the cross-series
        geometry of bw_marker_selection.same_body_reason) are each judged at
        their own integer centre. Those verdicts (calibrated or physical-body,
        exactly as used below) stay final unless they are inconsistent:

          * conflicting_own_centre_winners: two or more claimant series each win
            at their own centre (or keep a native full-window identity). The
            body would then be labelled by select_verified comparing 1-loss
            values measured at DIFFERENT centres, which is not identity evidence.
          * unresolved_despite_decisive_evidence: no claimant is kept, yet the
            old path's own comparison at some claimed centre decisively named
            a claimant series: the verdict there names ANOTHER claimant, or the
            physical-body comparison found a winner and discarded it only by
            its exact-extent test. The output (no label) contradicts that
            evidence.

        A single decisive own-centre winner is never re-decided, and neither is
        an honest abstention: a tie at the claimed centres with no decisive
        verdict anywhere, or a body no series supports there. No new label is
        created where the old path abstained for lack of evidence.

        For an inconsistent body every identity is scored on ONE shared set of
        localisations: each claimed integer centre and its half-pixel
        neighbours (the claims are rounded to integers, so the body centre lies
        within half a pixel of a claim). Half-pixel offsets shift the MODELS
        (marker and nuisance lines); the source patch is never resampled. Every
        series keeps exactly its own-centre scale hypotheses (no claimant
        footprint and no other stage's scale enters). joint_decide applies the
        unchanged support tests and identity margin. The result is applied only
        if the winner is one of the series the old verdicts already named
        decisively for this body (the conflicting winners, or the supported
        series named by the unacted evidence), the same decision over the claimed integer centres
        alone does not name a different decisive winner (the half-pixel
        refinement may resolve a tie there, never overturn a verdict), and the
        winner is the best-fitting identity at the records that will carry its
        label (their own centres +/- half a pixel). A tie or any other outcome
        keeps the old verdicts.
        """
        if getattr(self,'observation',None) is not None:
            return {}  # Chromatic subclasses retain their separately tested policy.
        from bw_marker_selection import same_body_reason
        from bw_raster_context_v46 import native_identity_supported
        n=len(records);parent=list(range(n))
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i in range(n):
            for j in range(i+1,n):
                a,b=records[i],records[j]
                if a['template']==b['template']:continue
                # Footprints of fits further apart than their two diameters
                # cannot overlap; skip the polygon test for them.
                reach=float(a['effective_diameter'])+float(b['effective_diameter'])
                if abs(a['aligned_x']-b['aligned_x'])>reach or abs(a['aligned_y']-b['aligned_y'])>reach:
                    continue
                if same_body_reason(a,b):
                    parent[root(i)]=root(j)
        groups={}
        for i in range(n):groups.setdefault(root(i),[]).append(records[i])
        out={};pending=[];counts={}
        for members in groups.values():
            if len({p['template'] for p in members})<2:continue
            xs=[float(p['aligned_x']) for p in members];ys=[float(p['aligned_y']) for p in members]
            diameter=min(float(p['effective_diameter']) for p in members)
            span=max(np.hypot(a-b,c-d) for a,c in zip(xs,ys) for b,d in zip(xs,ys))
            old={};winners=set();supported=set();named=set()
            series={p['template'] for p in members}
            for p in members:
                c=(float(p['aligned_x']),float(p['aligned_y']))
                q=physical.get(c,calibrated[c]);old[id(p)]=q
                if native_identity_supported(p,self.templates) or q['winner']==p['template']:
                    winners.add(p['template'])
                if q['scores'][p['template']]['passed']:
                    supported.add(p['template'])
                # Decisive evidence the old path produced here for a claimant
                # series but did not act on: the verdict at this claim names
                # ANOTHER claimant series, or the physical-body comparison
                # found a winner and discarded it only by its exact-extent test.
                if q['winner'] is not None and q['winner']!=p['template'] and q['winner'] in series:
                    named.add(q['winner'])
                ph=q.get('physical_body_hypothesis') or {}
                if q.get('reason')=='physical_body_extent_mismatch' and ph.get('unconstrained_winner') in series:
                    named.add(ph['unconstrained_winner'])
            if span>.5*diameter:
                case='claims_spread_beyond_one_radius'   # not one body
            elif len(winners)>=2:
                case='conflicting_own_centre_winners'
            elif not winners and named&supported:
                case='unresolved_despite_decisive_evidence'
            elif winners:
                case='single_own_centre_winner_kept'
            elif supported:
                case='tie_abstention_kept'
            else:
                case='no_supported_claim_kept'
            counts[case]=counts.get(case,0)+1
            summary=dict(version='joint_body_on_inconsistency_v1',case=case,
                claimants=[dict(template=p['template'],x=float(p['aligned_x']),y=float(p['aligned_y']),
                                candidate_index=p.get('candidate_index'),own_winner=old[id(p)]['winner'],
                                own_margin=float(old[id(p)]['margin']),
                                own_supported=bool(old[id(p)]['scores'][p['template']]['passed']))
                            for p in members],
                own_centre_winners=sorted(winners),supported_claims=sorted(supported),
                decisively_named_unkept=sorted(named),
                applied=False,winner=None)
            for p in members:out[id(p)]=dict(summary=summary,action=None,comparison=None)
            if case=='conflicting_own_centre_winners':
                pending.append((members,old,summary,winners))
            elif case=='unresolved_despite_decisive_evidence':
                pending.append((members,old,summary,named&supported))
        self.report['joint_body_cases']=counts
        if not pending:
            return out
        native=self.report.get('source_model_policy')=='shared_native_window'
        steps=(0.,) if native else (-.5,0.,.5)
        # Shared localisation set per body: every claimed centre and its
        # half-pixel neighbours, identical for every identity.
        sites=[sorted({(float(round(p['aligned_x']))+dx,float(round(p['aligned_y']))+dy)
                       for p in members for dx in steps for dy in steps}) for members,*_ in pending]
        state=[({},{},{},{}) for _ in pending]  # best, where, supported, supported_where
        site_losses=[{} for _ in pending]       # (x, y) -> {identity: loss}
        phases=sorted({(x-np.floor(x),y-np.floor(y)) for s in sites for x,y in s})
        # Phase-major: one phase bank is resident at a time (memory bound).
        for phase in phases:
            competitor=self._phase_competitor(phase)
            for site,(best,where,sup,sup_where),losses in zip(sites,state,site_losses):
                centres=[(float(np.floor(x)),float(np.floor(y))) for x,y in site
                         if (x-np.floor(x),y-np.floor(y))==phase]
                if not centres:continue
                for centre,result in zip(centres,competitor.score_cached(centres)):
                    at=[centre[0]+phase[0],centre[1]+phase[1]]
                    losses[tuple(at)]={key:sc['loss'] for key,sc in result['scores'].items()}
                    for key,sc in result['scores'].items():
                        if key not in best or sc['loss']<best[key]['loss']:
                            best[key]=sc;where[key]=at
                        if sc['passed'] and (key not in sup or sc['loss']<sup[key]['loss']):
                            sup[key]=sc;sup_where[key]=at
            if competitor is not self:
                self.phase_competitors.pop(phase,None)
        applied=0
        for (members,old,summary,eligible),site,(best,where,sup,sup_where),losses in zip(pending,sites,state,site_losses):
            winner,reason,margin=joint_decide(best,sup,self.cfg)
            # The same decision restricted to the claimed integer centres (the
            # observation's native sites; the calibrated comparisons already
            # made there). The half-pixel refinement may resolve a tie left
            # there, but it may not overturn a decisive verdict there: that
            # body's identity depends on sub-pixel localisation, so the joint
            # result is not robust and the old verdicts stay.
            native_best,native_sup={},{}
            for p in members:
                for key,sc in calibrated[(float(p['aligned_x']),float(p['aligned_y']))]['scores'].items():
                    if key not in native_best or sc['loss']<native_best[key]['loss']:native_best[key]=sc
                    if sc['passed'] and (key not in native_sup or sc['loss']<native_sup[key]['loss']):native_sup[key]=sc
            native_winner,native_reason,native_margin=joint_decide(native_best,native_sup,self.cfg)
            summary.update(claimed_centre_winner=native_winner,claimed_centre_reason=native_reason,
                           claimed_centre_margin=native_margin)
            scores={k:sup.get(k,v) for k,v in best.items()}
            comparison=dict(version=VERSION,center=[float(np.mean([p['aligned_x'] for p in members])),
                                                    float(np.mean([p['aligned_y'] for p in members]))],
                common_radius=self.radius,winner=winner,reason=reason,margin=margin,scores=scores,
                joint_body=summary)
            summary.update(joint_winner=winner,joint_reason=reason,joint_margin=margin,
                localisations=len(site),best_centres={k:sup_where.get(k,v) for k,v in where.items()},
                best_fit_losses={k:v['loss'] for k,v in best.items()},
                supported_losses={k:v['loss'] for k,v in sup.items()},
                scale_hypotheses_unchanged=True,claimant_footprint_sizes_used=False,
                source_resampled=False,thresholds_unchanged=True)
            if winner is None or winner not in eligible:
                summary['outcome']='own_centre_verdicts_kept_'+('joint_tie' if winner is None else 'joint_winner_not_plausible')
                continue
            if native_winner is not None and native_winner!=winner:
                summary['outcome']='own_centre_verdicts_kept_claimed_centres_decide_otherwise'
                continue
            # The label lands on the winner's OWN records (the conflicting
            # winners it keeps, or the typed alternatives it promotes), at their
            # own centres. The winner must be the best-fitting identity there
            # (their sites: own centre +/- half a pixel); a winner whose fit is
            # best only at another series' claim would label a record whose
            # own position the evidence does not support.
            conflict=summary['case']=='conflicting_own_centre_winners'
            labelled=[p for p in members if p['template']==winner and
                      ((old[id(p)]['winner']==winner or native_identity_supported(p,self.templates)) if conflict
                       else old[id(p)]['scores'][winner]['passed'])]
            local={}
            for p in labelled:
                cx,cy=float(round(p['aligned_x'])),float(round(p['aligned_y']))
                for at,row in losses.items():
                    if abs(at[0]-cx)<=.5 and abs(at[1]-cy)<=.5:
                        for key,loss in row.items():local[key]=min(local.get(key,np.inf),loss)
            local_best=min(local,key=local.get) if local else None
            summary.update(own_site_best=local_best,own_site_losses={k:float(v) for k,v in local.items()})
            if local_best!=winner:
                summary['outcome']='own_centre_verdicts_kept_winner_not_best_at_its_own_records'
                continue
            changed=False
            for p in members:
                q=old[id(p)];own=q['scores'][p['template']]
                if summary['case']=='conflicting_own_centre_winners':
                    # Demote only the contradicted own-centre winners; the
                    # joint winner's records and every non-winner keep theirs.
                    if (p['template']!=winner and q['winner']==p['template']
                            and not native_identity_supported(p,self.templates)):
                        out[id(p)].update(action='demote',comparison=comparison);changed=True
                elif p['template']==winner and own['passed']:
                    # The winner's own supported typed alternatives become
                    # active; nothing unsupported at its own centre is promoted.
                    out[id(p)].update(action='promote',comparison=comparison);changed=True
            summary.update(applied=changed,winner=winner,
                           outcome='joint_winner_applied' if changed else 'joint_winner_equals_old_roster')
            applied+=changed
        self.report['joint_body_applied']=applied
        return out

    def _physical_body_comparisons(self, records, calibrated):
        """Isolated, bounded banks: no mutation of calibrated/global caches."""
        if getattr(self,'observation',None) is not None:
            return {}  # Chromatic subclasses retain their separately tested policy.
        native={}
        for record in records:
            center=(float(record['aligned_x']),float(record['aligned_y']))
            if calibrated[center].get('winner') is not None:
                continue  # Never perturb an already supported calibrated winner.
            body=verified_body_extent(record)
            if body is None or max(body['native_extent'])>self.cfg.maximum_diameter:
                continue
            native.setdefault(tuple(body['center']),[]).append(body)
        families={}
        for center, bodies in native.items():
            sizes=tuple(sorted({round(max(body['native_extent']),6) for body in bodies}))
            families.setdefault(sizes,[]).append(center)
        result={}
        for sizes, centers in sorted(families.items())[:self.cfg.maximum_body_hypotheses]:
            hypotheses={}
            source_extents={}
            valid=True
            for key,(rec,fields) in self.templates.items():
                box=np.asarray(rec.get('bbox',[]),float)
                if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:]-box[:2])<=0:
                    valid=False;break
                # These extents are measured in the observed-source extractor's
                # own native raster, not the geometry model's scale reference.
                source_extent=box[2:]-box[:2]
                source_extents[key]=float(max(source_extent))
                original=getattr(self,'scale_hypotheses',{}).get(key,(self.scales[key],))
                hypotheses[key]=tuple(sorted(set(float(s) for s in original if s is not None)
                    | {size/float(max(source_extent)) for size in sizes}))
            if not valid:
                continue
            bank=copy(self)
            bank.banks={};bank.cache={};bank.scale_hypotheses=hypotheses
            bank.radius=max(bank._variant(key,scale)[3]
                for key,(rec,fields) in self.templates.items() for scale in hypotheses[key])
            for start in range(0,len(centers),64):
                chunk=centers[start:start+64]
                for center, comparison in zip(chunk,bank.score(chunk)):
                    unconstrained_winner=comparison['winner']
                    selected_extent=None;extent_matches=None
                    if unconstrained_winner is not None:
                        selected_scale=comparison['scores'][unconstrained_winner]['parameters']['raster_scale']
                        selected_extent=source_extents[unconstrained_winner]*selected_scale
                        extent_matches=bool(any(abs(selected_extent-size)<=1.e-6 for size in sizes))
                        if not extent_matches:
                            # Enlarging the shared observation may make an old
                            # scale win. That is not evidence for the admitted
                            # physical body. Abstain without dropping that
                            # rival or promoting a worse, size-matched model.
                            comparison['winner']=None
                            comparison['reason']='physical_body_extent_mismatch'
                    comparison['physical_body_hypothesis']=dict(
                        version='same_physical_body_all_rivals_v1',native_bodies=native[center],
                        native_major_extents=list(sizes),scale_hypotheses=hypotheses,
                        calibrated_scales_retained=True,all_rivals_compared=True,
                        same_domain_for_marker_and_line_null=True,thresholds_unchanged=True,
                        unconstrained_winner=unconstrained_winner,
                        selected_model_native_major_extent=selected_extent,
                        selected_model_matches_admitted_extent=extent_matches)
                    result[center]=comparison
        self.report['physical_body_families']=len(families)
        self.report['physical_body_family_budget_exhausted']=len(families)>self.cfg.maximum_body_hypotheses
        return result

    def score_cached(self, centers):
        """Call-local memoization for final alignment and the unchanged filter."""
        if not hasattr(self,'cache'):self.cache={}
        keys=[tuple(map(float,c)) for c in centers]
        missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
        for start in range(0,len(missing),64):
            chunk=missing[start:start+64]
            self.cache.update(zip(chunk,self.score(chunk)))
        return [self.cache[k] for k in keys]

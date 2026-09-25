"""Common-scene identity competition after colour-group proposals.

Marker existence is tested against the SAME line-only observation first.
Two plausible tied identities preserve an unassigned body instead of being
invented active detections. The GUI/CLI grouped chromatic path enables this
adapter by default; native grayscale BW, unique-colour and neutral groups
retain their existing paths. No chart names or reference data occur here.
"""
from dataclasses import asdict, replace
import time
import cv2
import numpy as np
import bw_observed_raster_v46 as R
import bw_layered_composite_v46 as L
from bw_raster_identity_v46 import Competition as Base, Config, features, decide
from bw_suppressed_v46 import encode_marker_mask
from color_marker_evidence import _swatch_model
from color_palette_identity_v46 import paper_mixture_evidence

VERSION='colour-group-common-scene-identity-v1'


class Competition(Base):
    def __init__(self,image,group,scales,cfg=None):
        self.cfg=cfg or replace(Config(),minimum_visible=.55,maximum_centres=600)
        self.ready=False;self.banks={};self.cache={};self.observation=group.get('colour_observation')
        self.report=dict(version=VERSION,status='not_applicable',config=asdict(self.cfg),
            reference_points_read=False,source_pixels_modified=False)
        if self.observation is None or len(group['templates'])<2:return
        if image.shape[0]*image.shape[1]>1_000_000 or len(group['templates'])>10:
            self.report['status']='resource_limit_image_or_swatch_count';return
        self.observation.validate(image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.ink=np.clip((255-gray)/255,0,1)
        self.proposal_ink=self.ink*self.observation.own
        self.templates={};self.prepared={};self.source_meta={};self.colour_fits={};self.scales={}
        for t in group['templates']:
            a,b,c,d=map(int,t.swatch_box)
            rec,f=R.extract(gray[b:d,a:c])
            if f is None or rec['diameter']>self.cfg.maximum_diameter:
                self.report['status']='unresolved_source_body';return
            native_scale=float(scales.get(t.key,1.))
            # The BW scale is relative to its canonical swatch diameter,
            # not the separately extracted raster's measured diameter.
            scale=native_scale*float(getattr(t,'diameter',rec['diameter']))/rec['diameter']
            if not np.isfinite(scale) or not .4<=scale<=2.:
                self.report['status']='invalid_scale';return
            self.templates[t.key]=(rec,f);self.scales[t.key]=scale
            self.prepared[t.key],self.source_meta[t.key]=L.prepare_source(rec,f)
            self.colour_fits[t.key]=paper_mixture_evidence(image,_swatch_model(image,t.swatch_box))['fit']
        self.radius=max(L.variants(rec,self.prepared[k],self.scales[k],(0.,0.))[3] for k,(rec,f) in self.templates.items())
        self.report.update(status='ready',scales=self.scales,common_radius=self.radius,source_meta=self.source_meta)
        self.ready=True

    def calibrate(self,image,plot,legend):
        """Reuse verified source-boundary scale profiles, not shape-only sizes.

        Restrict scale anchors with measured colour-role support afterwards;
        unrelated same-gray bodies cannot supply a series' calibration.
        No point labels or expected point counts are supplied.
        """
        from bw_observed_scale_v46 import apply,choose_scale
        templates={k:(rec,dict(f,target=self.prepared[k]['marker'],target_weight=self.prepared[k]['weight']))
            for k,(rec,f) in self.templates.items()}
        calibration,_=apply(image,plot,legend,templates)
        for key,v in calibration['series'].items():
            rec=self.templates[key][0];prepared=self.prepared[key]
            for site in v['anchor_profiles']:
                m,w,cover,r=L.variants(rec,prepared,site['best_scale'],(0.,0.))
                get=lambda p:cv2.getRectSubPix(p.astype(np.float32),(len(m),len(m)),(float(site['x']),float(site['y'])))
                mass=w*m*m;support=get(self.observation.own)*get(self.colour_fits[key])
                agreement=float(np.sum(mass*support)/max(mass.sum(),1.e-8))
                site['colour_role_support']=agreement
                site['weight']*=agreement**2 if agreement>=.55 else 0.
            selected,reason=choose_scale(v['trials'],v['anchor_profiles'])
            v.update(scale=selected,reason=reason,previous_raster_scale=self.scales[key])
            if selected is not None:self.scales[key]=selected
            else:v['fallback']='native_physical_size_preserved'
        self.radius=max(L.variants(rec,self.prepared[k],self.scales[k],(0.,0.))[3] for k,(rec,f) in self.templates.items())
        self.banks={}
        self.report.update(calibration=calibration,scales=self.scales,common_radius=self.radius)

    def score(self,centers):
        if not self.ready or not centers:return []
        self._bank();cfg=self.cfg;r=self.radius;size=2*r+1
        get=lambda plane:np.asarray([cv2.getRectSubPix(plane.astype(np.float32),(size,size),(float(x),float(y))) for x,y in centers])
        patches=get(self.ink);own=get(self.observation.own);valid=1-get(self.observation.other)
        # Known foreign ink may hide a body, but cannot provide its positive
        # evidence. Actual paper remains observable and penalizes model ink.
        confidence=np.concatenate([valid.reshape(len(centers),-1),
            np.minimum(valid[:,1:],valid[:,:-1]).reshape(len(centers),-1),
            np.minimum(valid[:,:,1:],valid[:,:,:-1]).reshape(len(centers),-1)],axis=1)
        f=features(patches,self.weight,cfg.boundary_weight)
        energy=np.maximum(np.sum(f*f*confidence,axis=1),1.e-6)
        def losses(bank,cost):
            return np.maximum(energy[:,None]-2*(f*confidence)@bank.T+confidence@(bank*bank).T,0)/energy[:,None]+cost
        nf,nn,nc=self.null_bank
        nl=losses(nf,nc);null_best=nl.min(axis=1)
        outputs=[{} for _ in centers]
        for key,b in self.banks.items():
            ls=losses(b['features'],b['cost']);indices=ls.argmin(axis=1)
            colour=get(self.colour_fits[key])
            for i,j in enumerate(indices):
                full=self.nulls[b['line_ids'][j]];m=b['marker'];gain=b['parameters'][j]['gain']
                mass=b['marker_weight']*m*m;independent=mass*(1-full)**2
                fraction=float(independent.sum()/max(mass.sum(),1.e-8))
                remain=np.clip((patches[i]-full)/np.maximum(1-full,.02),0,1)
                visible=float(np.sum(independent*own[i]*np.clip(remain/np.maximum(gain*m,1.e-6),0,1))/max(independent.sum(),1.e-8))
                raw=float(ls[i,j]);improvement=float(null_best[i]-raw)
                colour_weight=independent*own[i]*np.clip(patches[i]/.15,0,1)
                colour_loss=float(np.sum(colour_weight*(1-colour[i]))/max(colour_weight.sum(),1.e-8))
                total=raw+.20*colour_loss
                passed=bool(raw<=cfg.maximum_loss and improvement>=cfg.minimum_improvement and
                            fraction>=cfg.minimum_independent_fraction and visible>=cfg.minimum_visible)
                outputs[i][key]=dict(loss=total,geometry_loss=raw,colour_loss=colour_loss,
                    line_only_loss=float(null_best[i]),line_improvement=improvement,independent_fraction=fraction,
                    independent_visible=visible,parameters=b['parameters'][j],passed=passed)
        result=[]
        for center,scores in zip(centers,outputs):
            winner,reason,margin=decide(scores,cfg)
            present=any(s['passed'] for s in scores.values())
            result.append(dict(version=VERSION,center=list(center),common_radius=r,winner=winner,
                reason=reason,margin=margin,marker_present=present,scores=scores))
        return result


def refine(image,group,result,plot,legend):
    """Return a replacement group result; failure never partially edits it."""
    start=time.perf_counter();diag=result['diagnostics']
    scales={s['swatch_id']:s.get('applied_scale',1.) for s in diag['swatches']}
    comp=Competition(image,group,scales)
    if not comp.ready:
        return result,comp.report
    comp.calibrate(image,plot,legend)
    permitted=np.zeros(image.shape[:2],bool);a,b,c,d=plot;permitted[b:d,a:c]=True
    a,b,c,d=legend;permitted[b:d,a:c]=False
    proposals=[]
    for key,(rec,f) in comp.templates.items():
        p=comp.prepared[key]
        fields=dict(target=p['marker'],target_weight=p['weight'])
        loss,px,py=R.evaluate_scale(comp.proposal_ink,rec,fields,comp.scales[key],R.Config())
        proposals += [dict(x=q['x'],y=q['y'],loss=q['loss'],diameter=rec['diameter']*comp.scales[key])
            for q in R.local_candidates(loss,px,py,rec['diameter']*comp.scales[key],permitted,.85,cap=100)]
    for p in diag['candidates']:
        x,y=float(p['aligned_x']),float(p['aligned_y'])
        if 0<=round(y)<len(permitted) and 0<=round(x)<permitted.shape[1] and permitted[round(y),round(x)]:
            proposals.append(dict(x=x,y=y,loss=max(0.,1.-p['score']),diameter=10.))
    from bw_grouped_raster_recovery_v46 import group_proposals,Config as GroupConfig
    bodies=group_proposals(proposals,GroupConfig(centres_per_body=3,maximum_bodies=150))
    centers=list(dict.fromkeys(c for b in bodies for c in b['centres']))
    scored={}
    for offset in range(0,len(centers),32):
        if time.perf_counter()-start>60:
            # An incomplete reclassification may erase untouched native points.
            return result,dict(comp.report,status='budget_legacy_preserved',seconds=time.perf_counter()-start)
        batch=centers[offset:offset+32]
        scored.update(zip(batch,comp.score(batch)))
    choices=[];unassigned=[]
    for bi,body in enumerate(bodies):
        vals=[scored[c] for c in body['centres']]
        resolved=[q for q in vals if q['winner'] is not None]
        if resolved:
            q=min(resolved,key=lambda q:q['scores'][q['winner']]['loss']);choices.append((bi,q))
        else:
            present=[q for q in vals if q['marker_present']]
            if present:
                q=min(present,key=lambda q:min(s['loss'] for s in q['scores'].values()))
                unassigned.append(dict(body_id=bi,comparison=q,reason=q['reason'],marker_present=True))
    bykey={t.key:t for t in group['templates']};kept=[];accepted=[]
    native_meta={p['swatch_id']:p for p in result['suppressed']+result['kept']}
    for bi,q in sorted(choices,key=lambda v:v[1]['scores'][v[1]['winner']]['loss']):
        key=q['winner'];s=q['scores'][key];t=bykey[key];rec=comp.templates[key][0]
        diameter=rec['diameter']*comp.scales[key];x,y=q['center']
        if any(np.hypot(p['cx']-x,p['cy']-y)<.65*min(diameter,p['effective_diameter']) for p in kept):continue
        m,w,c,r=L.variants(rec,comp.prepared[key],comp.scales[key],(0.,0.))
        mask=(m>max(.03,.35*m.max()))&(w>.2)
        if not mask.any():continue
        meta=native_meta.get(key,{})
        kept.append(dict(class_name=t.name,shape_hint=t.shape_hint or t.name,template=key,swatch_id=key,
            class_idx=meta.get('class_idx'),shape_idx=meta.get('shape_idx'),
            cx=x,cy=y,confidence=s['independent_visible'],point_id=f'CR{len(kept)+1:03}',source=VERSION,
            original_detection=True,marker_mask=encode_marker_mask(mask),marker_offset_x=0.,marker_offset_y=0.,
            source_diameter=float(t.diameter),observed_body_diameter=rec['diameter'],
            effective_diameter=diameter,marker_scale=diameter/float(t.diameter),marker_aspect=1.,
            relative_identity=q))
        accepted.append(dict(body_id=bi,comparison=q))
    # A tied body remains in the diagnostic output, never masquerades as a
    # typed Step-5 candidate. Legacy typed hypotheses are retained separately.
    unassigned=[q for q in unassigned if not any(np.hypot(p['cx']-q['comparison']['center'][0],p['cy']-q['comparison']['center'][1])<.65*p['effective_diameter'] for p in kept)]
    report=dict(comp.report,status='completed',seconds=time.perf_counter()-start,proposals=len(proposals),
        bodies=len(bodies),centres=len(centers),before=len(result['kept']),after=len(kept),
        accepted=accepted,unassigned_markers=unassigned,comparisons=list(scored.values()),
        native_points_reclassified=True,positive_evidence_from_foreign_ink=False)
    from bw_step5_v46 import key as point_key
    active_keys={point_key(p) for p in kept}
    report['superseded_suppressed_ids']=[p.get('point_id') for p in result['suppressed'] if point_key(p) in active_keys]
    out=dict(result);out['kept']=kept;out['diagnostics']=dict(diag,group_relative_identity=report)
    out['suppressed']=[p for p in result['suppressed'] if point_key(p) not in active_keys]
    out['unassigned_markers']=unassigned
    return out,report

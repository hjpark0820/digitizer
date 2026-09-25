"""Observed grayscale scale calibration using verified, observed boundaries.

No canonical shapes, panel names, reference positions or fixed-count anchors.
The baseline comparator is unchanged. Source intensity steps, not gradients of
the reliability map, check the observed boundary before any scale is selected.
"""
from dataclasses import asdict, dataclass
import time
import cv2
import numpy as np
import bw_observed_raster_v46 as R
from functools import lru_cache

VERSION = 'bw-observed-verified-scale-v1'


@dataclass(frozen=True)
class Config:
    minimum_edge_recall: float = .55
    minimum_sector_recall: float = .35
    minimum_sectors: int = 3
    boundary_loss_weight: float = .35
    max_candidates_per_scale: int = 250
    profile_center_radius: int = 1


def boundary_evidence(ink, rec, fields, p, gain=1., cfg=Config()):
    """Signed adjacent gray differences, with observable ink/paper on both sides.

    The measured template is not binarized. Reliability is attached to each
    adjacent-pixel pair, so a low-confidence connector creates no fake edge.
    All supported sectors remain in the denominator; extra/reversed edges
    contribute to the symmetric squared residual.
    """
    phase = (p['x']-p['ix'], p['y']-p['iy'])
    t, w, radius = R.raster_variant(fields['target'], fields['target_weight'],
                                    rec['crop_center'], p['scale'], phase)
    size = len(t)
    pad = cv2.copyMakeBorder(ink, radius, radius, radius, radius,
                             cv2.BORDER_CONSTANT, value=0)
    z = pad[p['iy']:p['iy']+size, p['ix']:p['ix']+size] / gain
    arrays = []
    for axis in (0, 1):
        dt, dz = np.diff(t, axis=axis), np.diff(z, axis=axis)
        if axis == 0:
            pair_weight = np.minimum(w[1:], w[:-1])
        else:
            pair_weight = np.minimum(w[:, 1:], w[:, :-1])
        yy, xx = np.indices(dt.shape, dtype=float)
        yy += .5 if axis == 0 else 0
        xx += .5 if axis == 1 else 0
        sector = (xx >= radius+phase[0]).astype(int) + 2*(yy >= radius+phase[1])
        arrays.append((dt.ravel(), dz.ravel(), pair_weight.ravel(), sector.ravel()))
    dt, dz, weight, sector = [np.concatenate([v[i] for v in arrays]) for i in range(4)]
    energy_by_pixel = weight*dt*dt
    energy = float(energy_by_pixel.sum())
    if energy < 1.e-7:
        return dict(passed=False, reason='no_observable_boundary', loss=None,
                    recall=0., sector_recall=[], supported_sectors=0)
    supported = weight*np.clip(dt*dz, 0, dt*dt)
    recall = float(supported.sum()/energy)
    residual = float(np.sum(weight*(dt-dz)**2)/energy)
    sector_energy = [float(energy_by_pixel[sector == i].sum()) for i in range(4)]
    sector_recall = [float(supported[sector == i].sum()/max(e, 1.e-9))
                     for i, e in enumerate(sector_energy)]
    usable = [e >= .04*energy for e in sector_energy]
    count = sum(u and r >= cfg.minimum_sector_recall for u, r in zip(usable, sector_recall))
    passed = recall >= cfg.minimum_edge_recall and count >= cfg.minimum_sectors
    return dict(passed=bool(passed), reason='distributed_observed_edges' if passed else 'weak_or_one_sided_boundary',
                loss=residual, recall=recall, sector_recall=sector_recall,
                sector_energy=sector_energy, supported_sectors=int(count))


def choose_scale(trials, profiles):
    """Compare the SAME physical sites at all scales, with bounded regret.

    Site weights use only the best verified source evidence, never counts or
    reference labels. Missing a real verified site is penalized; nonexistent
    extra sites are not invented to reach a quota. A single site is permitted
    but reported as provisional, never disguised as multi-marker consensus.
    """
    for t in trials:
        t['anchor_count'] = len(t['anchors'])
        t['scale_objective'] = None
    usable = [p for p in profiles if p['weight'] > 1.e-8]
    if not usable:
        return None, 'insufficient_verified_scale_anchors'
    denom = sum(p['weight'] for p in usable)
    for j, t in enumerate(trials):
        regrets = []
        for p in usable:
            q = p['by_scale'][j]
            regret = min(1., max(0., q['joint_loss']-p['best_loss'])) if q['reason'] == 'verified_scale_anchor' else 1.
            regrets.append(p['weight']*regret)
        t['scale_objective'] = float(sum(regrets)/denom)
    winner = min(trials, key=lambda t: t['scale_objective'])
    return winner['scale'], 'single_anchor_provisional' if len(usable) == 1 else 'verified_same_site_scale_profiles'


def apply(image, plot, legend, templates, raster_cfg=R.Config(), cfg=Config(), *, exclusions=None):
    start = time.perf_counter()
    if not templates or any(f is None for rec, f in templates.values()):
        raise ValueError('All source swatches must participate in identity competition')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(np.percentile(gray, 95))
    ink = np.clip((paper-gray)/255., 0, 1)
    permitted = np.zeros(gray.shape, bool)
    x0, y0, x1, y1 = plot
    permitted[y0:y1, x0:x1] = True
    for a, b, c, d in (exclusions if exclusions is not None else [legend]):
        permitted[max(0,b):max(0,d), max(0,a):max(0,c)] = False
    maps = {}; variants = {}; series = {}
    for key, (rec, f) in templates.items():
        variants[key] = []
        for scale in raster_cfg.scales:
            loss, px, py = R.evaluate_scale(ink, rec, f, scale, raster_cfg)
            variants[key].append((loss, px, py))
        # A competing swatch is free to choose its own scale too. Do not hide
        # a better rival behind an already-wrong selected scale.
        maps[key] = np.minimum.reduce([v[0] for v in variants[key]])
    def measure(key, scale, p):
        rec, f = templates[key]
        p = dict(p, series_id=key, scale=scale, diameter=rec['diameter']*scale)
        v = verify(ink, rec, f, p)
        edge = boundary_evidence(ink, rec, f, p, v['gain'], cfg)
        p.update(line_guard=v, boundary=edge,
                 joint_loss=float(p['loss']+cfg.boundary_loss_weight*edge['loss']) if edge['loss'] is not None else None)
        if p['loss'] > raster_cfg.maximum_loss:
            reason = 'gray_loss_exceeded'
        elif not v['passed']:
            reason = v['reason']
        elif not edge['passed']:
            reason = edge['reason']
        else:
            reason = 'verified_scale_anchor'
        p['reason'] = reason
        return p

    # First verify bodies at every scale, THEN compare identities. A rival
    # with a low gray residual but no observable boundary cannot monopolize
    # the identity margin.
    checked_by_key = {}; valid_bodies = []
    for key, (rec, f) in templates.items():
        trials = []
        for scale, (loss, px, py) in zip(raster_cfg.scales, variants[key]):
            candidates = R.local_candidates(loss, px, py, rec['diameter']*scale,
                           permitted, raster_cfg.maximum_loss, cap=cfg.max_candidates_per_scale)
            checked = [measure(key, scale, p) for p in candidates]
            valid_bodies.extend(p for p in checked if p['reason'] == 'verified_scale_anchor')
            trials.append(dict(scale=scale, anchors=[], candidates=checked,
                               checked_count=len(checked), candidate_cap=cfg.max_candidates_per_scale,
                               cap_reached=len(checked) == cfg.max_candidates_per_scale))
        checked_by_key[key] = trials

    def identify(p):
        if p['reason'] != 'verified_scale_anchor':
            return p
        rivals = [q for q in valid_bodies if q['series_id'] != p['series_id']
                  and np.hypot(q['x']-p['x'], q['y']-p['y']) <= .25*min(q['diameter'], p['diameter'])]
        rival = min(rivals, key=lambda q: q['joint_loss']) if rivals else None
        margin = rival['joint_loss']-p['joint_loss'] if rival else None
        reason = 'ambiguous_verified_swatch' if margin is not None and margin < raster_cfg.minimum_identity_margin else p['reason']
        return dict(p, rival=rival['series_id'] if rival else None,
                    rival_loss=rival['joint_loss'] if rival else None, margin=margin, reason=reason)

    proposals = []; rejected = []
    for key, trials in checked_by_key.items():
        rec, f = templates[key]
        pool = []
        for t in trials:
            t['candidates'] = [identify(p) for p in t['candidates']]
            t['anchors'] = [p for p in t['candidates'] if p['reason'] == 'verified_scale_anchor']
            pool.extend(t['anchors'])
        seeds = []
        for p in sorted(pool, key=lambda p: p['joint_loss']):
            if all(np.hypot(p['x']-q['x'], p['y']-q['y']) > .55*rec['diameter'] for q in seeds):
                seeds.append(p)
        profiles = []
        for seed in seeds:
            profile = []
            # The same physical site is revisited within +/-1 native pixel,
            # rather than picking a different globally best fragment per scale.
            for scale, (loss, px, py) in zip(raster_cfg.scales, variants[key]):
                radius = cfg.profile_center_radius
                x, y = seed['ix'], seed['iy']
                a, b = max(0, x-radius), max(0, y-radius)
                c, d = min(ink.shape[1], x+radius+1), min(ink.shape[0], y+radius+1)
                local = np.where(permitted[b:d, a:c], loss[b:d, a:c], np.inf)
                yy, xx = np.unravel_index(local.argmin(), local.shape)
                ix, iy = int(a+xx), int(b+yy)
                p = dict(ix=ix, iy=iy, x=float(ix+px[iy, ix]), y=float(iy+py[iy, ix]), loss=float(loss[iy, ix]))
                profile.append(identify(measure(key, scale, p)))
            verified = [p for p in profile if p['reason'] == 'verified_scale_anchor']
            if not verified:
                continue
            best = min(verified, key=lambda p: p['joint_loss'])
            weight = max(0., 1.-best['joint_loss'])**2 * best['boundary']['recall']
            profiles.append(dict(x=seed['x'], y=seed['y'], best_loss=best['joint_loss'],
                                 best_scale=best['scale'], weight=float(weight), by_scale=profile))
        scale, reason = choose_scale(trials, profiles)
        series[key] = dict(scale=scale, reason=reason, trials=trials, anchor_profiles=profiles)
        if scale is not None:
            selected_trial = next(t for t in trials if t['scale'] == scale)
            proposals.extend(selected_trial['anchors'])
            rejected.extend(p for p in selected_trial['candidates'] if p['reason'] != 'verified_scale_anchor')
    points = []; suppressed = []
    for p in sorted(proposals, key=lambda p: p['joint_loss']):
        duplicate = next((q for q in points if np.hypot(p['x']-q['x'], p['y']-q['y'])
                          < .55*min(p['diameter'], q['diameter'])), None)
        if duplicate:
            suppressed.append(dict(p, reason='same_location_competition'))
        else:
            points.append(p)
    return dict(version=VERSION, points=points, suppressed=suppressed, rejected=rejected,
                series=series, config=asdict(cfg), raster_config=asdict(raster_cfg),
                seconds=time.perf_counter()-start, image_paper=paper, plot=list(plot), legend_box=list(legend),
                reference_points_read=False, centered_line_composition=False,
                operational_defaults_modified=False, step5=False), maps


@lru_cache(maxsize=100)
def line_bank(size):
    y,x=np.indices((size,size),dtype=np.float32)
    x=x-(size-1)/2;y=y-(size-1)/2
    models=[];parameters=[]
    for angle in np.arange(0,180,15):
        rad=np.deg2rad(angle)
        perp=x*np.cos(rad)+y*np.sin(rad)
        for offset in (-2.,-1.,0.,1.,2.):
            for width in (.65,1.1,1.7,2.5):
                stroke=np.clip(.5+(width/2-np.abs(perp-offset))/.8,0,1)
                models.append(stroke.ravel());parameters.append((int(angle),offset,width))
    return np.asarray(models,np.float32),parameters


def verify(ink,rec,fields,point):
    phase=(point['x']-point['ix'],point['y']-point['iy'])
    target,weight,radius=R.raster_variant(fields['target'],fields['target_weight'],rec['crop_center'],point['scale'],phase)
    padded=cv2.copyMakeBorder(ink,radius,radius,radius,radius,cv2.BORDER_CONSTANT,value=0)
    x,y=point['ix'],point['iy'];size=len(target)
    patch=padded[y:y+size,x:x+size]
    energy=max(float(np.sum(weight*target**2)),1.e-7)
    gains=(.85,1.,1.15)
    losses=[float(np.sum(weight*(patch-g*target)**2)/(g*g*energy)) for g in gains]
    j=int(np.argmin(losses));gain=gains[j]
    marker_sse=float(np.sum(weight*(patch-gain*target)**2))
    bank,params=line_bank(size);w=weight.ravel();z=patch.ravel()
    dot=bank@(w*z);norm=bank**2@w
    amplitude=np.clip(dot/np.maximum(norm,1.e-8),0,1)
    sse=float(np.sum(w*z*z))-2*amplitude*dot+amplitude**2*norm
    best=int(np.argmin(sse));null_sse=float(max(0,sse[best]))
    improvement=(null_sse-marker_sse)/(gain*gain*energy)
    # Required visible source ink must appear in more than one fragment.
    core=(target>max(.03,.35*target.max()))&(weight>.20)
    core_weights=weight*target**2*core
    matching=np.clip(patch/np.maximum(gain*target,1.e-6),0,1)
    visible=float(np.sum(core_weights*matching)/max(np.sum(core_weights),1.e-8))
    passed=improvement>.06 and visible>=.70
    reason='verified_raster' if passed else ('better_explained_by_line' if improvement<=.06 else 'insufficient_visible_ink')
    return dict(passed=bool(passed),reason=reason,marker_loss=losses[j],gain=gain,
                line_loss=null_sse/(gain*gain*energy),line_improvement=improvement,
                visible_ink=visible,line_parameters=list(params[best]),line_amplitude=float(amplitude[best]))




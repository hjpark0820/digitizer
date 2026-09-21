"""Experimental independent fill-style evidence; no synthetic template ink.

Soft internal light regions discriminate solid bodies from patterned/partial
bodies. External pixels do not vote against a marker. Only independently long
straight strokes are excluded; we never declare an unexplained mismatch to be
occlusion merely because it is dark. Insufficient visible interior abstains.
"""
from collections import Counter
import cv2
import numpy as np

VERSION = 'bw-fill-tone-identity-v3'


def tone_ink(template):
    """Paper-relative darkness with fixed black=0, not per-swatch contrast stretch."""
    source = getattr(template, 'source_gray', None)
    paper = max(1., float(template.ink.paper_gray))
    if source is not None:
        return np.clip((paper-np.asarray(source,np.float32))/paper,0,1)
    # Older saved templates lack the unclipped raster: retain only the range
    # that can be recovered, and mark that limitation in descriptor metadata.
    return np.asarray(template.raw_soft,np.float32)*(paper-template.ink.core_gray)/paper


def interior_mask(template):
    mask = np.asarray(template.mask, bool)
    ys, xs = np.nonzero(mask & ~template.line_nuisance)
    envelope = np.zeros(mask.shape, np.uint8)
    if len(xs) < 3:
        return envelope.astype(bool)
    cv2.fillConvexPoly(envelope, cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32)), 1)
    distance = cv2.distanceTransform(np.pad(envelope, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    return (distance >= max(1.5, .18 * template.diameter)) & ~template.line_nuisance


def describe_fill(template, shape_evidence=None):
    region = interior_mask(template)
    soft = tone_ink(template)
    values = soft[region]
    result = dict(version=VERSION, style='uncertain', interior_pixels=int(values.size),
                  outer_shape=(shape_evidence or {}).get('best_shape', 'unknown'),
                  source_pixels_preserved=True,unclipped_gray_available=getattr(template,'source_gray',None) is not None,
                  gray_scale='0=black, 255=white; paper-normalized darkness, no per-marker contrast stretch')
    if values.size < 10:
        return result
    mean, sd = float(values.mean()), float(values.std())
    light = float(np.mean(values < .5))
    yy, xx = np.indices(soft.shape)
    # A half-filled glyph has a coherent division, unlike fine hatching.
    splits = []
    for axis in (xx, yy, xx + yy, xx - yy):
        middle = float(np.median(axis[region]))
        left, right = values[axis[region] <= middle], values[axis[region] > middle]
        if len(left) >= 4 and len(right) >= 4:
            splits.append(abs(float(left.mean() - right.mean())))
    split = max(splits, default=0.)
    if template.name.startswith('open_') and mean < .40:
        style = 'open'
    elif mean >= .90 and light <= .08:
        style = 'solid'
    elif split >= .48 and sd >= .20:
        style = 'partial'
    elif .12 < mean < .88 and sd >= .17 and light >= .10:
        style = 'patterned'
    elif .15 < mean < .88 and sd < .12:
        style = 'gray'
    else:
        style = 'uncertain'
    result.update(style=style, mean_ink=mean, std_ink=sd,
                  light_fraction=light, strongest_half_split=split,
                  mean_gray_255=float((1-mean)*255), std_gray_255=sd*255,
                  nonwhite_fraction=float(np.mean(values>.08)),dark_fraction=float(np.mean(values>.80)),
                  gray_quantiles_255=np.percentile((1-values)*255,[10,25,50,75,90]).tolist())
    # Tone labels have deliberate uncertainty bands. They are not evidence
    # that an independently filled body is absent (e.g. scanned dark gray).
    shape = shape_evidence or {}
    result['opaque_body_supported'] = bool(
        values.size >= 10 and mean >= .74 and light <= .08 and
        shape.get('sufficient_fill_evidence') and shape.get('independent_fill', 0.) >= .85 and
        shape.get('shape_confidence', 0.) >= .75 and
        not shape.get('patterned_internal_evidence') and not shape.get('strong_hollow_evidence'))
    return result


def estimate_body_scale(image, templates, plot_area, ignore_regions=()):
    """Conservative common size evidence from repeated isolated solid cores.

    This is only an experimental proposal-range initializer, not marker output.
    Patterned pixels and thin error bars cannot themselves define a solid core.
    Returned anchors are source measurements, not known data-point positions.
    """
    source_sizes = [t.diameter for t in templates if describe_fill(t)['style'] == 'solid']
    if not source_sizes:
        return dict(applied=False, scale=1., reason='no_solid_reference', anchors=[])
    diameter = float(np.median(source_sizes))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.zeros(gray.shape, np.uint8)
    a,b,c,d = map(int,plot_area);binary[b:d,a:c] = (gray[b:d,a:c]<150).astype(np.uint8)
    for a,b,c,d in ignore_regions:binary[b:d,a:c]=0
    eroded=cv2.erode(binary,np.ones((3,3),np.uint8))
    _,_,stats,centers=cv2.connectedComponentsWithStats(eroded)
    anchors=[]
    for (x,y,w,h,area),center in zip(stats[1:],centers[1:]):
        scale=(max(w,h)+2)/diameter
        if min(w,h)>=3 and .65<=w/h<=1.55 and .35<=scale<=1.5 and area/(w*h)>=.55:
            anchors.append(dict(center=center.tolist(),body_size=int(max(w,h)+2),scale=float(scale)))
    if len(anchors)<3:
        return dict(applied=False,scale=1.,reason='fewer_than_three_compact_bodies',anchors=anchors)
    scales=np.array([a['scale'] for a in anchors])
    groups=[np.flatnonzero(np.abs(np.log(scales/s))<=.20) for s in scales]
    group=max(groups,key=len); selected=scales[group]
    scale=float(np.median(selected))
    consistent=len(group)>=3 and float(np.subtract(*np.percentile(selected,[75,25])))<.20*scale
    applied=bool(consistent and (scale<.83 or scale>1.18))
    return dict(applied=applied,scale=scale if applied else 1.,measured_scale=scale,
                source_diameter=diameter,reason='repeated_compact_body_consensus' if applied else 'no_clear_size_change',
                support=len(group),anchors=anchors)


class FillIdentityCompetition:
    """A shared small-grid callback plus final full-window fill check.

    The original outline/body competitor still handles solid shape differences.
    The new check is opt-in and acts only on legends containing non-solid fill.
    Source size/center are supplied by the existing grid and verifier, not by a
    known answer or a manually labelled plot-marker location.
    """
    def __init__(self, image, templates, plot_area, ignore_regions=(), base=None):
        self.base = base
        self.templates = {t.key: t for t in templates}
        self.descriptors = {t.key: describe_fill(t) for t in templates}
        self.enabled = any(d['style'] in {'patterned', 'partial', 'gray'} for d in self.descriptors.values())
        self.plot = tuple(plot_area)
        self.counts = Counter()
        self.cache = {}
        # One paper/black scale for all series; do not renormalize each window
        # until a pale pattern becomes fully black.
        paper = float(np.median([t.ink.paper_gray for t in templates]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.ink = np.clip((paper - gray) / max(1., paper), 0, 1)
        self.valid = np.zeros(gray.shape, np.uint8)
        a, b, c, d = self.plot; self.valid[b:d, a:c] = 1
        for a, b, c, d in ignore_regions:
            self.valid[b:d, a:c] = 0
        self.regions = {key: interior_mask(t) for key, t in self.templates.items()}
        self.tones = {key:tone_ink(t) for key,t in self.templates.items()}
        self.tone_cache = {}
        self.envelopes = {}
        for key,t in self.templates.items():
            ys,xs=np.nonzero(t.mask & ~t.line_nuisance)
            env=np.zeros(t.mask.shape,np.uint8)
            if len(xs)>=3:
                cv2.fillConvexPoly(env,cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32)),1)
            self.envelopes[key]=env.astype(bool)
        self.nuisance = {}
        for key, t in self.templates.items():
            length = max(11, int(np.ceil(1.8 * t.diameter))) | 1
            binary = (self.ink >= .45).astype(np.uint8)
            strokes = np.zeros_like(binary)
            for kernel in (np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8),
                           np.eye(length, dtype=np.uint8), np.fliplr(np.eye(length, dtype=np.uint8))):
                strokes |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # Wide bodies aren't an independently thin crossing line.
            depth = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            strokes &= (depth <= max(1.5, .12 * t.diameter)).astype(np.uint8)
            self.nuisance[key] = strokes

    def _measure(self, key, x, y, scale, aspect=1., detail=False, coarse=False):
        t = self.templates[key]
        h, w = t.raw_soft.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        ratio = np.sqrt(aspect)
        mx = (x + (xx - (w - 1) / 2) * scale * ratio).astype(np.float32)
        my = (y + (yy - (h - 1) / 2) * scale / ratio).astype(np.float32)
        observed = cv2.remap(self.ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(self.valid, mx, my, cv2.INTER_NEAREST).astype(bool)
        line = cv2.remap(self.nuisance[key], mx, my, cv2.INTER_NEAREST).astype(bool)
        region = self.regions[key]
        visible = region & valid & ~line
        count = int(visible.sum())
        fraction = count / max(1, int(region.sum()))
        descriptor = self.descriptors[key]
        supported = self.enabled and descriptor['style'] in {'solid', 'patterned', 'partial', 'gray', 'open'}
        result = dict(style=descriptor['style'], visible_pixels=count, visible_fraction=fraction,
                      decision='abstain', loss=0., mean_gap=0., light_conflict=0., dark_missing=0.,
                      reason='insufficient_visible_interior')
        if not supported or count < 12 or fraction < .55:
            if detail:
                result.update(observed=observed, expected=self.tones[key], visible=visible)
            return result
        expected = self.tones[key]
        tone_key=(key,round(scale,4),round(aspect,4))
        if tone_key in self.tone_cache:
            expected,exp_blur=self.tone_cache[tone_key]
        elif scale < .95:
            # A smaller printed marker cannot retain the legend's subpixel
            # hatch phase. Compare both on that observable resolution first.
            small = cv2.resize(expected, (max(3,round(w*scale*ratio)),max(3,round(h*scale/ratio))),
                               interpolation=cv2.INTER_AREA)
            expected = cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)
            exp_blur=cv2.GaussianBlur(expected,(3,3),.65)
            self.tone_cache[tone_key]=(expected,exp_blur)
        else:
            exp_blur=cv2.GaussianBlur(expected,(3,3),.65)
            self.tone_cache[tone_key]=(expected,exp_blur)
        # Low-pass residual tolerates raster phase/antialiasing. The original
        # asymmetric exact-ink verifier remains mandatory and unchanged.
        obs_blur = cv2.GaussianBlur(observed, (3, 3), .65)
        bright_weight = np.clip((.85 - exp_blur) / .65, 0, 1) * visible
        dark_weight = exp_blur * visible
        excess = np.maximum(0, obs_blur - exp_blur - .08)
        missing = np.maximum(0, exp_blur - obs_blur - .08)
        light_loss = float((excess * bright_weight).sum() / max(1., bright_weight.sum()))
        dark_loss = float((missing * dark_weight).sum() / max(1., dark_weight.sum()))
        mean_gap = float(np.mean(observed[visible] - expected[visible]))
        # Compare cumulative brightness distributions: a half-black/half-white
        # patch and uniform middle gray have equal means but distinct CDFs.
        tone_distance=0.
        if not coarse:
            bins=np.linspace(0,1,17)
            obs_hist=np.histogram(observed[visible],bins=bins)[0].astype(float)/count
            exp_hist=np.histogram(expected[visible],bins=bins)[0].astype(float)/count
            tone_distance=float(np.abs(np.cumsum(obs_hist)-np.cumsum(exp_hist)).mean())
        loss = .50 * light_loss + .30 * dark_loss + .20*tone_distance
        non_solid = descriptor['style'] != 'solid'
        conflict = ((non_solid and light_loss > .16 and mean_gap > .19)
                    or (dark_loss > .20 and mean_gap < -.23))
        # A similar mean darkness is insufficient: a thin curve can have the
        # same average as hatching. Require spatially two-dimensional ink,
        # measured inside the source envelope, never in unrelated outer pixels.
        support_region=self.envelopes[key] & valid & ~line
        def moment_ratio(weights):
            mass=float(weights.sum())
            if mass<1e-6:return 0.
            cx=float((xx*weights).sum()/mass);cy=float((yy*weights).sum()/mass)
            dx,dy=xx-cx,yy-cy
            covariance=np.array([[(dx*dx*weights).sum(),(dx*dy*weights).sum()],
                                 [(dx*dy*weights).sum(),(dy*dy*weights).sum()]])/mass
            eigenvalues=np.linalg.eigvalsh(covariance)
            return float(max(0.,eigenvalues[0])/max(1e-6,eigenvalues[1]))
        observed_mass=float((observed*support_region).sum())*scale*scale
        observed_ratio=moment_ratio(observed*support_region) if not coarse else 1.
        expected_ratio=moment_ratio(expected*support_region) if not coarse else 1.
        linear=bool(expected_ratio>=.25 and observed_ratio<max(.08,.25*expected_ratio))
        result.update(observed_2d_ratio=observed_ratio,expected_2d_ratio=expected_ratio,
                      observed_native_ink_mass=observed_mass)
        conflict=conflict or linear
        reason='one_dimensional_ink' if linear else 'fill_mismatch' if conflict else 'compatible_interior'
        result.update(decision='conflict' if conflict else 'compatible', loss=loss,
                      mean_gap=mean_gap, light_conflict=light_loss, dark_missing=dark_loss,
                      observed_mean=float(observed[visible].mean()), expected_mean=float(expected[visible].mean()),reason=reason)
        result.update(gray_distribution_distance=tone_distance,
                      observed_mean_gray_255=255*(1-result['observed_mean']),
                      expected_mean_gray_255=255*(1-result['expected_mean']))
        if observed_mass<4.:
            result.update(decision='abstain',reason='insufficient_independent_marker_ink')
        if detail:
            result.update(observed=observed, expected=expected, visible=visible,
                          light_conflict_map=excess * bright_weight, dark_missing_map=missing * dark_weight)
        return result

    def evaluate_fill(self, template, x, y, total_scale=None, aspect=1., stage='window', detail=False):
        scale = template.proposal_scale if total_scale is None else total_scale
        # Grid votes have coarse centers. Require conflict at *all* plausible
        # nearby centers before removing a vote; final windows use final alignment.
        shift = max(1., .08 * self.templates[template.key].diameter * scale) if stage == 'grid' else 0.
        offsets = ((0., 0.), (-shift, 0.), (shift, 0.), (0., -shift), (0., shift)) if shift else ((0., 0.),)
        ck = (template.key, round(x,3), round(y,3), round(scale, 3), round(aspect, 3), stage)
        if not detail and ck in self.cache:
            return self.cache[ck]
        choices = [self._measure(template.key, x + dx, y + dy, scale, aspect, detail,stage=='grid') for dx, dy in offsets]
        result = min(choices, key=lambda q: (q['decision'] == 'conflict', q['loss']))
        if not detail:
            self.cache[ck] = result
        return result

    def filter_hypotheses(self, template, hypotheses, **kwargs):
        rows = self.base.filter_hypotheses(template, hypotheses, **kwargs) if self.base else hypotheses
        if not self.enabled or self.descriptors[template.key]['style'] not in {'patterned', 'partial', 'gray'}:
            return rows
        kept = []
        for row in rows:
            result = self.evaluate_fill(template, row.x, row.y, stage='grid')
            # Only robust excess dark ink at the grid stage, not weak outline
            # coverage or a wrong coarse center, removes a non-solid vote.
            if result['decision'] == 'conflict' and result['light_conflict'] > .24 and result['mean_gap'] > .28:
                self.counts['grid_votes_removed'] += 1
            else:
                kept.append(row)
        self.counts['grid_votes_checked'] += len(rows)
        return kept

    def candidate_decision(self, template, x, y):
        # Keep plausible centers for the larger-window stage; this callback
        # retains the pre-existing geometry competitor's behavior.
        return self.base.candidate_decision(template, x, y) if self.base else True

    def for_template(self, template, x, y, total_scale=None):
        return self.base.for_template(template, x, y, total_scale) if self.base else dict(winner=None, pairs=[], cell_votes={})

    def report(self):
        result = self.base.report() if self.base else dict(enabled=True)
        return {**result, 'fill_identity': dict(version=VERSION, enabled=self.enabled,
                    descriptors=self.descriptors, counts=dict(self.counts),
                    occlusion_policy='Exclude externally extending thin straight strokes only; ambiguous interior abstains')}

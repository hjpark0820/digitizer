"""Conservative source-pixel verification for BW markers touching plot axes.

``AxisContactVerifier.measurements(x, y)`` returns ``None`` when the ordinary
local verifier should be used.  Otherwise every saved legend model is measured
at the same centre, on the same bounded original-image rectangle, against one
fixed axis/tick explanation.  Returned dictionaries are independent copies.

The plot constrains centres, not visible marker pixels.  An image edge is not a
plot edge: unavailable source pixels never become supporting evidence.  Axes
are detected once, then re-estimated outside the entire legend-family body for
each centre.  No template-specific axis fitting or ground-truth input is used.
"""
from copy import deepcopy
from time import perf_counter

import cv2
import numpy as np

from marker_ignore_v46 import build_ignore_mask


POLICY = 'bw-axis-contact-visible-first-v3'
_KERNEL = np.ones((3, 3), np.uint8)


def _box(value, shape, *, optional=False):
    if optional and value is None:
        return None
    a = np.asarray(value, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or np.any(a != np.round(a)):
        raise ValueError('Axis-contact boxes must contain four finite integers')
    x0, y0, x1, y1 = map(int, a)
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError('Axis-contact box is outside the source image')
    return x0, y0, x1, y1


def _hull(mask):
    result = np.zeros_like(mask)
    yy, xx = np.nonzero(mask)
    cv2.fillConvexPoly(result, cv2.convexHull(np.column_stack((xx, yy)).astype('int32')), 1)
    return result


def _masked_body(raw, visible, kernel):
    """Opening on partial observations, without turning unknown into paper.

    A kernel placement must contain no observed paper and at least half of its
    area must be observed ink. Unknown pixels cannot seed a body or be returned
    as supporting ink. This is not an inpainting of the observation raster.
    """
    positive = (raw > 0) & visible
    paper = (raw == 0) & visible
    count = cv2.filter2D(positive.astype(np.float32), -1,
                        kernel.astype(np.float32), borderType=cv2.BORDER_CONSTANT)
    contradicted = cv2.dilate(paper.astype(np.uint8), kernel) > 0
    seeds = (~contradicted) & (count >= .5*float(kernel.sum()))
    return ((cv2.dilate(seeds.astype(np.uint8), kernel) > 0) & positive).astype(np.uint8)


def _direct_transitions(expected, hull, raw, visible, gray, sectors, hollow):
    """Test the *direction* of observed ink-to-paper transitions.

    Hollow glyphs use their inner boundary: an independently visible white
    cavity can establish the shape even when its outer silhouette touches a
    neighbour. Filled glyphs use the outer silhouette. Unknown axis pixels
    are not paper, and neither dilation nor a nearby unrelated edge supplies
    support. The denominator includes all predicted transitions.
    """
    ink = expected > 0
    paper_model = ((hull > 0) & ~ink) if hollow else (hull == 0)
    tone = float(np.median(gray[ink & visible])) if (ink & visible).any() else 0.
    contrast = max(.10, .35*tone)
    predicted = np.zeros_like(ink)
    supported = np.zeros_like(ink)
    pair_count = supported_pairs = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        # Windows have padding, but explicitly invalidate wrapped neighbours.
        neighbour_paper = np.roll(paper_model, (-dy, -dx), axis=(0, 1))
        neighbour_visible = np.roll(visible, (-dy, -dx), axis=(0, 1))
        neighbour_gray = np.roll(gray, (-dy, -dx), axis=(0, 1))
        neighbour_raw = np.roll(raw, (-dy, -dx), axis=(0, 1))
        if dy < 0: neighbour_paper[:1] = False
        if dy > 0: neighbour_paper[-1:] = False
        if dx < 0: neighbour_paper[:, :1] = False
        if dx > 0: neighbour_paper[:, -1:] = False
        pairs = ink & neighbour_paper
        good = (pairs & visible & neighbour_visible & (raw > 0) &
                (neighbour_raw == 0) & (gray-neighbour_gray >= contrast))
        predicted |= pairs
        supported |= good
        pair_count += int(pairs.sum())
        supported_pairs += int(good.sum())
    sector_support = []
    for sector in range(8):
        region = predicted & (sectors == sector)
        if region.any() and np.mean(supported[region]) >= .80:
            sector_support.append(sector)
    return dict(kind='inner_ink_to_paper' if hollow else 'outer_ink_to_paper',
        expected_pairs=pair_count, supported_pairs=supported_pairs,
        support=float(supported_pairs/max(1, pair_count)),
        sectors=sector_support, minimum_gray_contrast=contrast,
        unknown_pixels_supply_support=False)


class AxisContactVerifier:
    """Frozen per-image verifier; ``models`` are correction._models results.

    Existing local gates remain recall >= .92, unexplained body <= .18,
    filled-body recall >= .82, and hollow-interior clearance >= .70.  The axis
    lane additionally requires >= .50 independently visible expected ink,
    >= .55 visible silhouette, support in five of eight angular sectors, and
    a positive gray-pixel improvement over the fixed axis-only hypothesis.
    These coverage gates are separate from visible-pixel match quality.
    Score is the lesser of the ordinary body score and visible contour recall:
    black fill inside another shape cannot masquerade as a supported outline.
    A separate direct-visible lane can waive EXTRA-ink rejection, but only for
    a nearly complete observed glyph with correctly directed ink/paper edges
    and a decisive common-domain legend-raster identity. Partial observations
    keep the conservative gates; this is not an occlusion inpainting route.
    """

    def __init__(self, image, plot, legend, models):
        started = perf_counter()
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Axis-contact verification requires uint8 BGR source')
        self.plot = _box(plot, image.shape)
        self.legend = _box(legend, image.shape, optional=True)
        self.models = {}
        for sid, source in models.items():
            mask = np.asarray(source['mask'])
            anchor = np.asarray(source['anchor'], dtype=float)
            if (mask.ndim != 2 or not mask.size or max(mask.shape) > 1024 or
                    not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all() or
                    not mask.any() or anchor.shape != (2,) or
                    not np.isfinite(anchor).all() or np.max(np.abs(anchor)) > 1024):
                raise ValueError('Invalid axis-contact saved marker geometry')
            model = deepcopy(source)
            model.update(mask=mask.astype(np.uint8).copy(), anchor=anchor.copy())
            self.models[sid] = model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        paper = float(np.percentile(gray, 90))
        self.ink = ((paper-gray.astype(float)) > max(25, .18*paper)).astype(np.uint8)
        self.gray_ink = np.clip((paper-gray.astype(np.float32))/max(paper, 1.), 0., 1.)
        self.valid = np.ones(gray.shape, bool)
        if self.legend:
            a, b, c, d = self.legend
            self.valid[b:d, a:c] = False
        self.cache = {}
        diameter = max((max(m['mask'].shape) for m in self.models.values()), default=1)
        self.diameter = diameter
        self.pad = max(3, int(round(.15*diameter)))
        extent = max((max(abs(m['anchor'][0]), abs(m['anchor'][1]),
                          abs(m['mask'].shape[1]-m['anchor'][0]),
                          abs(m['mask'].shape[0]-m['anchor'][1]))
                      for m in self.models.values()), default=1)
        margin = int(np.ceil(extent))+self.pad+3
        a, b, c, d = self.plot
        self.envelope = (max(0, a-margin), max(0, b-margin),
                         min(image.shape[1], c+margin), min(image.shape[0], d+margin))
        ex0, ey0, ex1, ey1 = self.envelope
        crop = image[ey0:ey1, ex0:ex1]
        ignored, detected = build_ignore_mask(crop,
            valid=self.valid[ey0:ey1, ex0:ex1], marker_diameter=diameter,
            include_guides=False)
        # Scanned axes may contain isolated white speckles that interrupt the
        # detector's long opening.  Close only one-pixel gaps in a disposable
        # detection raster; all verification and external support use original
        # pixels.  No completed detection pixel becomes marker evidence.
        core = ((gray[ey0:ey1, ex0:ex1] < 190) &
                (np.ptp(crop.astype(np.int16), axis=2) <= 30)).astype(np.uint8)
        closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _KERNEL)
        detector_image = np.repeat(((1-closed)*255)[:, :, None], 3, axis=2)
        closed_ignored, closed_detected = build_ignore_mask(detector_image,
            valid=self.valid[ey0:ey1, ex0:ex1], marker_diameter=diameter,
            include_guides=False)
        ignored |= closed_ignored
        self.axis_bands = []
        # Detection's broad border window must not turn an interior curve into
        # an axis.  Retain only thin lines adjacent to an actual plot border.
        tolerance = max(3., .15*diameter)
        max_width = max(4, int(round(.22*diameter)))
        for record in detected['structures']+closed_detected['structures']:
            if record['kind'] != 'axis':
                continue
            horizontal = record['orientation'] == 'horizontal'
            offset = ey0 if horizontal else ex0
            lo, hi = [int(v+offset) for v in record['band']]
            borders = (b, d-1) if horizontal else (a, c-1)
            if hi-lo > max_width or min(abs(.5*(lo+hi-1)-v) for v in borders) > tolerance:
                continue
            band = dict(orientation=record['orientation'], band=[lo, hi])
            if band not in self.axis_bands:
                self.axis_bands.append(band)
        merged = []
        for orientation in ('horizontal', 'vertical'):
            bands = sorted((r['band'] for r in self.axis_bands if r['orientation'] == orientation))
            for lo, hi in bands:
                if (merged and merged[-1]['orientation'] == orientation and
                        lo <= merged[-1]['band'][1]):
                    merged[-1]['band'][1] = max(hi, merged[-1]['band'][1])
                else:
                    merged.append(dict(orientation=orientation, band=[lo, hi]))
        self.axis_bands = [r for r in merged if r['band'][1]-r['band'][0] <= max_width]
        self.ignored = ignored
        self.report = dict(policy=POLICY, center_domain='plot',
            observation_domain='bounded_original_source', envelope=list(self.envelope),
            axis_bands=deepcopy(self.axis_bands), detector=detected,
            detector_gap_close=closed_detected,
            centers_measured=0, cache_hits=0, accepted_models=0, rejected_models=0,
            seconds=0., reasons={},
            thresholds=dict(recall=.92, unexplained_body=.18, body_recall=.82,
                hole_clear=.70, ink_coverage=.50, boundary_coverage=.55,
                supported_boundary_sectors=5, minimum_gray_gain=0.,
                direct_coverage=.90, direct_expected_ink=.85,
                direct_transition_support=.80, direct_hole_paper=.80,
                direct_identity_margin=.06))
        self.report['initialization_seconds'] = perf_counter()-started

    def _positions(self, x, y):
        positions = {}
        for sid, model in self.models.items():
            h, w = model['mask'].shape
            left = int(round(x-model['anchor'][0]))
            top = int(round(y-model['anchor'][1]))
            positions[sid] = (left, top, left+w, top+h)
        return positions

    def _contact(self, positions):
        a, b, c, d = self.plot
        for left, top, right, bottom in positions.values():
            if left <= a+1 or top <= b+1 or right >= c-1 or bottom >= d-1:
                return True
            for axis in self.axis_bands:
                lo, hi = axis['band']
                start, stop = (top, bottom) if axis['orientation'] == 'horizontal' else (left, right)
                if start < hi and stop > lo:
                    return True
        return False

    def _nuisance(self, box, family):
        """Predict axis tone from source pixels outside every candidate body.

        Tick pixels inside the family body cannot be learned independently and
        are deliberately not excused.  Thin unexplained ticks remain subject
        to the ordinary morphological-body test.
        """
        left, top, right, bottom = box
        ex0, ey0, ex1, ey1 = self.envelope
        h, w = family.shape
        geometry = np.zeros((h, w), bool)
        baseline = np.zeros((h, w), np.float32)
        tolerance = np.ones((h, w), np.float32)
        records = []
        guard = cv2.dilate(family.astype(np.uint8), _KERNEL) > 0
        for axis in self.axis_bands:
            lo, hi = axis['band']
            horizontal = axis['orientation'] == 'horizontal'
            if horizontal:
                rows = range(max(top, lo), min(bottom, hi))
                external = np.ones(ex1-ex0, bool)
                columns = np.flatnonzero(guard.any(axis=0))+left
                external[np.clip(columns-ex0, 0, ex1-ex0-1)] = False
            else:
                rows = range(max(left, lo), min(right, hi))
                external = np.ones(ey1-ey0, bool)
                columns = np.flatnonzero(guard.any(axis=1))+top
                external[np.clip(columns-ey0, 0, ey1-ey0-1)] = False
            supported = []
            profiles = []
            for coordinate in rows:
                values = (self.gray_ink[coordinate, ex0:ex1] if horizontal else
                          self.gray_ink[ey0:ey1, coordinate])
                ink = (self.ink[coordinate, ex0:ex1] if horizontal else
                       self.ink[ey0:ey1, coordinate])
                valid = (self.valid[coordinate, ex0:ex1] if horizontal else
                         self.valid[ey0:ey1, coordinate])
                sample = external & valid
                if sample.sum() < max(12, self.diameter) or np.mean(ink[sample]) < .65:
                    continue
                tone = float(np.median(values[sample]))
                # Only excess darker than independently observed axis variation
                # can remain marker evidence. All rivals use the same estimate.
                mad = float(np.median(np.abs(values[sample]-tone)))
                upper = float(np.percentile(values[sample], 90))
                uncertainty = max(12./255., 3.*1.4826*mad, upper-tone)
                if horizontal:
                    tolerance[coordinate-top, :] = np.where(geometry[coordinate-top, :],
                        np.maximum(tolerance[coordinate-top, :], uncertainty), uncertainty)
                    geometry[coordinate-top, :] = True
                    baseline[coordinate-top, :] = np.maximum(baseline[coordinate-top, :], tone)
                else:
                    tolerance[:, coordinate-left] = np.where(geometry[:, coordinate-left],
                        np.maximum(tolerance[:, coordinate-left], uncertainty), uncertainty)
                    geometry[:, coordinate-left] = True
                    baseline[:, coordinate-left] = np.maximum(baseline[:, coordinate-left], tone)
                supported.append(coordinate)
                profiles.append(dict(coordinate=coordinate, tone=tone,
                                     tolerance=uncertainty, external_mad=mad))
            if supported:
                records.append(dict(orientation=axis['orientation'],
                    band=[min(supported), max(supported)+1],
                    external_sample_count=int(external.sum()), profiles=profiles))
        if self.axis_bands:
            # The detector's observed mask is safe only near a retained axis
            # and outside all rival bodies.  This includes outward ticks.
            local = self.ignored[top-ey0:bottom-ey0, left-ex0:right-ex0]
            near_axis = np.zeros((h, w), bool)
            reach = max(3, min(12, int(round(.04*min(ey1-ey0, ex1-ex0)))))
            for record in records:
                lo, hi = record['band']
                if record['orientation'] == 'horizontal':
                    near_axis[max(0, lo-top-reach):min(h, hi-top+reach)] = True
                else:
                    near_axis[:, max(0, lo-left-reach):min(w, hi-left+reach)] = True
            ticks = local & near_axis & ~guard
            geometry |= ticks
            original = self.gray_ink[top:bottom, left:right]
            baseline[ticks] = original[ticks]
        return geometry, baseline, records, tolerance

    def measurements(self, x, y):
        """Return all-model measurements, or None to preserve the ordinary lane."""
        x, y = float(x), float(y)
        if not np.isfinite([x, y]).all():
            raise ValueError('Axis-contact centre must be finite')
        a, b, c, d = self.plot
        if not self.models or not (a <= x < c and b <= y < d):
            return None
        positions = self._positions(x, y)
        if not self._contact(positions):
            return None
        key = (x, y)
        if key in self.cache:
            self.report['cache_hits'] += 1
            return deepcopy(self.cache[key])
        started = perf_counter()
        left = min(p[0] for p in positions.values())-self.pad
        top = min(p[1] for p in positions.values())-self.pad
        right = max(p[2] for p in positions.values())+self.pad
        bottom = max(p[3] for p in positions.values())+self.pad
        box = (left, top, right, bottom)
        common = dict(policy=POLICY, observation_box=list(box), center=[x, y],
                      shared_models=sorted(self.models), source_pixels_modified=False)
        if left < 0 or top < 0 or right > self.ink.shape[1] or bottom > self.ink.shape[0]:
            result = {sid:dict(score=0., accepted=False, recall=0., body_recall=0.,
                unexplained_body=0., hole_clear=0., reason='outside_image_coverage',
                axis_contact=dict(common, image_coverage_complete=False)) for sid in self.models}
        else:
            expected = {}
            hulls = {}
            for sid, model in self.models.items():
                x0, y0, x1, y1 = positions[sid]
                mask = np.zeros((bottom-top, right-left), np.uint8)
                mask[y0-top:y1-top, x0-left:x1-left] = model['mask']
                expected[sid] = mask
                hulls[sid] = _hull(mask)
            family = np.maximum.reduce(list(hulls.values()))
            nuisance, baseline, records, tolerance = self._nuisance(box, family)
            raw = self.ink[top:bottom, left:right]
            # Keep the observation intact. Axis ink is unknown, not zero/white.
            observed = raw.copy()
            valid = self.valid[top:bottom, left:right]
            source_gray = self.gray_ink[top:bottom, left:right]
            tone_excess = nuisance & (source_gray > baseline+tolerance) & (raw > 0)
            hidden = nuisance & (raw > 0) & ~tone_excess
            visible = valid & ~hidden
            positive = (raw > 0) & visible
            near = cv2.distanceTransform((~positive).astype(np.uint8), cv2.DIST_L2, 5)
            h0_error = float(np.mean(np.square(source_gray[visible]-baseline[visible]))) if visible.any() else 0.
            result = {sid:self._measure(model, expected[sid], hulls[sid], observed,
                raw, near, valid, nuisance, source_gray, baseline, h0_error,
                dict(common, image_coverage_complete=True, fitted_axes=records), x-left, y-top,
                axis_hidden=hidden, tone_excess=tone_excess)
                for sid, model in self.models.items()}
            # Residuals outside every competing body are identical and cancel.
            # Normalize by the SAME family energy, not by exterior clutter.
            family_domain = (family > 0) & visible
            identity_energy = max(1., float(np.sum(np.square(source_gray[family_domain]))))
            for sid, own in result.items():
                direct = own['axis_contact']['direct_visible']
                if not direct['eligible'] or own['accepted']:
                    continue
                rivals = [(other, value) for other, value in result.items() if other != sid]
                own_loss = own['axis_contact']['h1_gray_error']
                rival = min(rivals, key=lambda item:(item[1]['axis_contact']['h1_gray_error'], item[0])) if rivals else None
                margin = ((rival[1]['axis_contact']['h1_gray_error']-own_loss)*int(visible.sum())/identity_energy
                          if rival else 1.)
                # Keep legacy identities and their score scale intact. A new
                # direct admission must win BOTH the common raster comparison
                # and the existing 0.06 identity gap. Legacy measurements are
                # unchanged, although a newly verified rival can change the
                # dispatcher's later identity/action competition.
                score = direct['score_without_exterior_penalty']
                legacy_gap = score-max((v['axis_contact']['legacy_score'] for _, v in rivals), default=score-1.)
                valid_direct = margin >= .06 and legacy_gap >= .06
                direct.update(valid=bool(valid_direct), gray_identity_margin=float(margin),
                    legacy_identity_margin=float(legacy_gap), identity_energy=identity_energy,
                    rival_series=rival[0] if rival else None,
                    reason='direct_shape_and_shared_raster_verified' if valid_direct else 'direct_identity_not_decisive')
                if valid_direct:
                    own.update(accepted=True, score=float(score), reason='axis_contact_direct_visible_verified')
                    own['axis_contact'].update(exterior_ink_is_veto=False,
                        score_rule='direct_visible_body_and_contour_with_shared_raster_identity')
        self.cache[key] = deepcopy(result)
        self.report['centers_measured'] += 1
        self.report['seconds'] += perf_counter()-started
        for measurement in result.values():
            field = 'accepted_models' if measurement['accepted'] else 'rejected_models'
            self.report[field] += 1
            reason = measurement['reason']
            self.report['reasons'][reason] = self.report['reasons'].get(reason, 0)+1
        return result

    def _measure(self, model, expected, hull, observed, raw, near, valid,
                 nuisance, source_gray, baseline, h0_error, metadata, cx, cy,
                 *, axis_hidden, tone_excess):
        # Only observed dark furniture may hide expected ink.  White source
        # pixels on an axis remain missing-marker penalties, never exclusions.
        # No blanket guard: observed paper adjacent to known ink establishes a
        # real boundary; an ignored axis pixel cannot establish one.
        hidden = axis_hidden
        visible = valid & ~hidden
        positive = (observed > 0) & visible
        paper = (observed == 0) & visible
        required = (expected > 0) & visible
        core = cv2.erode(hull, _KERNEL)
        outline = (hull > 0) & (core == 0)
        boundary = outline & visible
        ink_coverage = float(required.sum()/max(1, expected.sum()))
        boundary_coverage = float(boundary.sum()/max(1, outline.sum()))
        available = float(np.mean(valid[expected > 0]))
        recall = float(np.mean(near[required] <= 1.)) if required.any() else 0.
        h, w = model['mask'].shape
        size = max(3, int(round(.16*min(h, w))) | 1)
        body = _masked_body(observed, visible,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        expanded = cv2.dilate(hull, _KERNEL)
        extra = float(np.sum(body*(1-expanded))/max(1, hull.sum()))
        core_visible = (core > 0) & visible
        body_recall = float(np.mean(cv2.dilate(body, _KERNEL)[core_visible])) if core_visible.any() else 0.
        hole = core_visible & (expected == 0)
        hole_clear = float(np.mean(body[hole] == 0)) if hole.sum() >= 4 else 0.
        hollow = model['class_name'].startswith('open_')
        yy, xx = np.indices(expected.shape)
        sectors = np.floor((np.arctan2(yy-cy, xx-cx)+np.pi)/(np.pi/4)).astype(int) % 8
        # An actual observed paper neighbour is required. Unknown axis pixels
        # never create an artificial silhouette or serve as distance targets.
        observed_outline = positive & (cv2.dilate(paper.astype(np.uint8), _KERNEL) > 0)
        contour_near = cv2.distanceTransform((~observed_outline).astype(np.uint8), cv2.DIST_L2, 5)
        supported_sectors = []
        for sector in range(8):
            region = boundary & (sectors == sector)
            if region.sum() >= 2 and np.mean(contour_near[region] <= 1.) >= .80:
                supported_sectors.append(sector)
        boundary_recall = float(np.mean(contour_near[boundary] <= 1.)) if boundary.any() else 0.
        prediction = np.maximum(baseline, expected.astype(np.float32))
        h1_error = float(np.mean(np.square(source_gray[visible]-prediction[visible]))) if visible.any() else 1.
        gray_gain = h0_error-h1_error
        body_score = recall-.65*min(1., extra)-(.6*(1-hole_clear) if hollow else .4*(1-body_recall))
        score = min(body_score, boundary_recall)
        fill = model.get('fill_evidence', {})
        known = model['class_name'] not in ('unknown_marker', 'suppressed')
        tone_known = (not fill or fill.get('style') in ('solid', 'open', 'hollow') or
                      bool(fill.get('opaque_body_supported')) or hollow)
        coverage_ok = ink_coverage >= .50 and boundary_coverage >= .55 and available >= .98
        outline_ok = boundary.sum() >= 8 and len(supported_sectors) >= 5 and boundary_recall >= .85
        local_ok = (recall >= .92 and extra <= .18 and
                    (hole.sum() >= 4 and hole_clear >= .70 if hollow else body_recall >= .82))
        accepted = bool(known and tone_known and coverage_ok and outline_ok and local_ok and gray_gain > 0.)
        reason = ('axis_contact_source_verified' if accepted else
                  'unknown_marker_or_fill' if not known or not tone_known else
                  'axis_contact_insufficient_coverage' if not coverage_ok else
                  'axis_contact_insufficient_outline' if not outline_ok else
                  'axis_contact_no_marker_gray_gain' if gray_gain <= 0. else
                  'axis_contact_body_or_hole_failed')
        transition = _direct_transitions(expected, hull, raw, visible, source_gray, sectors, hollow)
        exact_ink = float(np.sum(required & (raw > 0))/max(1, expected.sum()))
        direct_paper = float(np.mean(paper[hole])) if hole.any() else 0.
        direct_fill = (hole.sum() >= 4 and direct_paper >= .80 if hollow else body_recall >= .82)
        direct_eligible = bool(known and tone_known and outline_ok and gray_gain > 0. and
            ink_coverage >= .90 and boundary_coverage >= .90 and available >= .98 and
            recall >= .92 and exact_ink >= .85 and direct_fill and
            transition['expected_pairs'] >= 8 and transition['support'] >= .80 and
            len(transition['sectors']) >= 5)
        direct_score = min(recall-(.6*(1-hole_clear) if hollow else .4*(1-body_recall)), boundary_recall)
        metadata.update(ink_coverage=ink_coverage, boundary_coverage=boundary_coverage,
            legend_coverage=available, boundary_recall=boundary_recall,
            visible_boundary_contour_recall=boundary_recall,
            visible_boundary_ink_recall=(float(np.mean(near[boundary] <= 1.)) if boundary.any() else 0.),
            supported_boundary_sectors=supported_sectors,
            fixed_nuisance_pixels=int(nuisance.sum()),
            nuisance_guard_pixels=0, ignored_axis_pixels=int(hidden.sum()),
            retained_axis_tone_excess_pixels=int((tone_excess & valid).sum()),
            observed_raster_preserved=bool(np.array_equal(observed, raw)),
            ignored_expected_ink_pixels=int(((expected > 0) & hidden & valid).sum()),
            evaluated_expected_ink_pixels=int(required.sum()),
            evaluated_boundary_pixels=int(boundary.sum()),
            supported_boundary_pixels=int((boundary & (contour_near <= 1.)).sum()),
            positive_ink_fraction=float(np.sum(required & (near <= 1.))/max(1, expected.sum())),
            positive_boundary_fraction=float(np.sum(boundary & (contour_near <= 1.))/max(1, outline.sum())),
            coverage_ok=bool(coverage_ok), outline_ok=bool(outline_ok), local_ok=bool(local_ok),
            h0_gray_error=h0_error, h1_gray_error=h1_error, gray_gain=gray_gain,
            gray_domain_pixels=int(visible.sum()),
            body_score=float(body_score), score_rule='min_body_score_visible_contour_recall',
            legacy_score=float(score), legacy_accepted=accepted, exterior_ink_is_veto=True,
            direct_visible=dict(eligible=direct_eligible, valid=False,
                reason='legacy_lane_sufficient' if accepted else 'direct_evidence_insufficient',
                expected_ink_fraction=exact_ink, direct_hole_paper_fraction=direct_paper,
                transition_support=transition['support'], transition_sectors=transition['sectors'],
                transition=transition, score_without_exterior_penalty=float(direct_score)),
            axis_pixels_supply_marker_support=False)
        return dict(score=float(score), accepted=accepted, recall=recall,
            body_recall=body_recall, unexplained_body=extra, hole_clear=hole_clear,
            reason=reason, unknown_fill_preserved=not tone_known, axis_contact=metadata)

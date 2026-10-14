"""Same-body shape identity for one shared neutral colour group (colour-group route only).

Why this exists. A neutral colour group holds several achromatic legend keys that see ONE ink image
(color_group_bw_v46.prepare_groups; with neutral_geometry_policy='shared_source' every achromatic key of the chart).
bw_pipeline_v46.detect_points accepts and ranks a group record only by template-ink recall (window required recall,
sectors, contour support, grid score). A key whose required ink is a subset of a rival's body (an X inside a filled
square, a triangle outline inside a filled triangle) therefore scores at least as well as the true key whenever the
true key's recall is imperfect, and nothing charges it for the body ink it leaves unexplained. Plain B&W resolves
the same situation with the observed-raster identity competition (bw_raster_identity_v46.Competition): every key,
and a line-only null, is fitted to the SAME source patch, so ink the model predicts as paper and paper it predicts
as ink both cost. detect_points disables that comparator for prepared colour templates and colour occlusion.

What this module does. It builds that unchanged B&W comparator (legend rasters measured by R.extract from the
group's own legend pixels, the B&W observed-scale calibration, the B&W Config thresholds, identity margin, line
families and line exterior-support test) on the group's verification image. The only change is the one the colour
route requires: pixels owned by another colour group (window occlusion) or of ambiguous colour (window uncertainty)
are UNOBSERVED, with zero weight for every rival and for the line-only null, instead of being scored as paper. With
no unobserved pixel in a patch the geometric losses equal Competition.score exactly.

Tone. The group shares one ink plane only so that thin keys measured paler than their ink (antialiasing) keep their
pixels; prepare_groups states that tone is retained for later identity tests, and the B&W comparator alone does not
test it (a grey key's raster, at the comparator's maximum gain 1.15, can still fit a black body whose shape no single
key explains, e.g. two overlapping triangles). The identity loss therefore adds the colour route's own physical
paper/ink mixture term, exactly as color_group_relative_identity_v46.Competition.score does for chromatic groups:
total = geometric loss + 0.20 x colour loss, colour loss = mean (1 - paper_mixture_evidence fit of the key's measured
legend ink) over the observed ink the key's best model explains. The mixture fit is asymmetric by construction: pale
antialiased pixels of a dark ink fit the dark key, darker-than-key pixels cannot fit a pale key. Marker support
('passed') stays purely geometric, as in both existing comparators.

Decision. A verified record is withdrawn only when a DIFFERENT key of the group is the decisive winner at the
record's own aligned centre (bw_raster_identity_v46.decide: the winner passes the unchanged marker-support tests and
beats every rival by the unchanged identity margin). Abstentions (tie, unsupported winner) and wins of the record's
own key leave the record and its rank unchanged, so this stage can only remove an identity that a same-body
comparison refutes. A withdrawn record keeps plain B&W's filter_records semantics: still a supported marker of its
own key -> 'identity_ambiguous' (typed suppressed hypothesis); otherwise 'raster_marker_not_supported'.
No point labels, reference data or chart names are read.
"""
import time
import cv2
import numpy as np
from bw_raster_identity_v46 import Competition, features, decide
from color_marker_evidence import _swatch_model
from color_palette_identity_v46 import paper_mixture_evidence

VERSION = 'neutral-group-same-body-identity-v1'
COLOUR_WEIGHT = .20   # color_group_relative_identity_v46.Competition.score: total = raw + .20*colour_loss


class NeutralGroupCompetition(Competition):
    """bw_raster_identity_v46.Competition with colour-unobserved pixels removed from every fit."""

    def __init__(self, image, plot, exclusions, templates, unobserved, *, legend_reports=None):
        unobserved = np.asarray(unobserved, np.float32)
        if (unobserved.shape != image.shape[:2] or not np.isfinite(unobserved).all()
                or np.any((unobserved < 0) | (unobserved > 1))):
            raise ValueError('unobserved must be finite, image-aligned and in [0,1]')
        super().__init__(image, plot, exclusions, templates, legend_reports=legend_reports)
        self.unobserved = unobserved
        self.colour_fits = {}
        if self.ready:
            for t in templates:
                self.colour_fits[t.key] = paper_mixture_evidence(image, _swatch_model(image, t.swatch_box))['fit']
        self.report.update(identity_stage=VERSION, unobserved_policy='other_colour_and_uncertain_pixels_zero_weight',
                           tone_policy='paper_mixture_fit_of_measured_legend_ink', colour_weight=COLOUR_WEIGHT)

    def score(self, centers):
        if not self.ready:
            return []
        self._bank(); cfg = self.cfg; r = self.radius; size = 2*r+1
        get = lambda plane: np.asarray([cv2.getRectSubPix(plane, (size, size), (float(x), float(y))) for x, y in centers])
        patches = get(self.ink)
        valid = 1-get(self.unobserved)
        # Same layout as features(): pixels, vertical then horizontal differences.
        confidence = np.concatenate([valid.reshape(len(centers), -1),
            np.minimum(valid[:, 1:], valid[:, :-1]).reshape(len(centers), -1),
            np.minimum(valid[:, :, 1:], valid[:, :, :-1]).reshape(len(centers), -1)], axis=1)
        f = features(patches, self.weight, cfg.boundary_weight)
        energy = np.maximum(np.sum(f*f*confidence, axis=1), 1.e-6)

        def losses(bank, cost):
            return np.maximum(energy[:, None]-2*(f*confidence)@bank.T+confidence@(bank*bank).T, 0)/energy[:, None]+cost
        nf, nn, nc = self.null_bank
        nl = losses(nf, nc)
        # Unchanged exterior-support test for nuisance lines, counted on observed pixels only:
        # foreign ink may hide a continuation, it can never supply one.
        yy, xx = np.indices((size, size)); line_ok = {}; ph = self._phase()
        for lines in self.family_lines:
            for line in lines:
                ident = id(line)
                if ident in line_ok:
                    continue
                ys, xs = np.nonzero(line > .01)
                horizontal = np.ptp(xs) >= np.ptp(ys)
                axis = (xx-r-ph[0]) if horizontal else (yy-r-ph[1])
                ray = []
                for sign in (-1, 1):
                    w = (~self.context_body)*(axis*sign > 0)*valid
                    mass = np.sum(line*w, axis=(1, 2))
                    ray.append(np.sum(np.minimum(patches, line)*w, axis=(1, 2))/np.maximum(mass, 1.e-6))
                line_ok[ident] = np.maximum(*ray) >= .45
        ok = np.stack([np.logical_and.reduce([line_ok[id(line)] for line in lines])
                       if lines else np.ones(len(centers), bool) for lines in self.family_lines], axis=1)
        nl = np.where(ok, nl, np.inf)
        null_best = nl.min(axis=1)
        outputs = [{} for _ in centers]
        for key, b in self.banks.items():
            ls = losses(b['features'], b['cost'])
            ls = np.where(ok[:, b['line_ids']], ls, np.inf)
            indices = ls.argmin(axis=1)
            colour = get(self.colour_fits[key])
            for i, j in enumerate(indices):
                full = self.nulls[b['line_ids'][j]]; gain = b['parameters'][j]['gain']
                m, cover, marker_weight = b['variants'][b['parameters'][j]['raster_scale']]
                mass = marker_weight*m*m; independent = mass*(1-full)**2
                fraction = float(independent.sum()/max(mass.sum(), 1.e-8))
                # Unobserved pixels carry no ink in the group image, so they never count as visible.
                remain = np.clip((patches[i]-full)/np.maximum(1-full, .02), 0, 1)
                visible = float(np.sum(independent*np.clip(remain/np.maximum(gain*m, 1.e-6), 0, 1))/max(independent.sum(), 1.e-8))
                raw = float(ls[i, j]); improvement = float(null_best[i]-raw)
                colour_weight = independent*valid[i]*np.clip(patches[i]/.15, 0, 1)
                colour_loss = float(np.sum(colour_weight*(1-colour[i]))/max(colour_weight.sum(), 1.e-8))
                loss = raw+COLOUR_WEIGHT*colour_loss
                outputs[i][key] = dict(loss=loss, geometry_loss=raw, colour_loss=colour_loss,
                    line_only_loss=float(null_best[i]), line_improvement=improvement,
                    line_continuation_observed=bool(ok[i, b['line_ids'][j]]),
                    independent_fraction=fraction, independent_visible=visible, parameters=b['parameters'][j],
                    passed=bool(raw <= cfg.maximum_loss and improvement >= cfg.minimum_improvement and
                                fraction >= cfg.minimum_independent_fraction and visible >= cfg.minimum_visible))
        result = []
        for center, scores, v in zip(centers, outputs, valid):
            winner, reason, margin = decide(scores, cfg)
            result.append(dict(version=VERSION, center=list(center), common_radius=r, winner=winner, reason=reason,
                               margin=margin, unobserved_fraction=float(1-np.sum(v*self.weight)/max(np.sum(self.weight), 1.e-8)),
                               scores=scores))
        return result


def withdraw_refuted(image, plot, exclusions, templates, legend_reports, records, unobserved, log_fn=print):
    """Return (records kept for selection, report). Records are the verified window records of ONE neutral group."""
    start = time.perf_counter()
    report = dict(version=VERSION, status='not_applicable', verified_input=len(records), retained=len(records),
                  withdrawn=[], reference_points_read=False, source_pixels_modified=False)
    if len({t.key for t in templates}) < 2 or not records:
        return records, report
    comp = NeutralGroupCompetition(image, plot, exclusions, templates, unobserved, legend_reports=legend_reports)
    report.update(comparator=comp.report.get('status'), scales=comp.report.get('scales'))
    if not comp.ready:
        report['status'] = 'comparator_unavailable_records_unchanged'
        return records, report
    centers = list(dict.fromkeys((float(p['aligned_x']), float(p['aligned_y'])) for p in records))
    if len(centers) > comp.cfg.maximum_centres:
        report['status'] = 'centre_budget_records_unchanged'
        return records, report
    lookup = {}
    for a in range(0, len(centers), 64):
        lookup.update(zip(centers[a:a+64], comp.score(centers[a:a+64])))
    kept = []
    for p in records:
        q = lookup[(float(p['aligned_x']), float(p['aligned_y']))]
        own = q['scores'][p['template']]
        p['group_shape_identity'] = dict(version=VERSION, winner=q['winner'], reason=q['reason'], margin=q['margin'],
            unobserved_fraction=q['unobserved_fraction'],
            losses={k: s['loss'] for k, s in q['scores'].items()},
            geometry_losses={k: s['geometry_loss'] for k, s in q['scores'].items()},
            colour_losses={k: s['colour_loss'] for k, s in q['scores'].items()},
            passed={k: s['passed'] for k, s in q['scores'].items()})
        if q['winner'] is None or q['winner'] == p['template']:
            kept.append(p)
            continue
        # A rival explains this body decisively better (ink AND paper, same patch, same null).
        if own['passed']:
            p['exclusion_reason'] = 'identity_ambiguous'
            p['raster_deferred_reason'] = 'another_observed_raster_wins'
        else:
            p['exclusion_reason'] = 'raster_marker_not_supported'
            p['raster_deferred_reason'] = 'insufficient_independent_marker_evidence'
        report['withdrawn'].append(dict(candidate_index=p.get('candidate_index'), template=p['template'],
            x=float(p['aligned_x']), y=float(p['aligned_y']), winner=q['winner'], margin=q['margin'],
            own_loss=own['loss'], winner_loss=q['scores'][q['winner']]['loss'], own_passed=own['passed'],
            exclusion_reason=p['exclusion_reason'], unobserved_fraction=q['unobserved_fraction']))
    report.update(status='completed', centres=len(centers), retained=len(kept), seconds=time.perf_counter()-start)
    log_fn(f'[v46 neutral group identity] {report["status"]}; verified={len(records)}; '
           f'withdrawn={len(report["withdrawn"])} (another key decisively explains the same body)')
    return kept, report

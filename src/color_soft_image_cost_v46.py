"""Opt-in, frozen marker-image cost; not a marker acceptance test.

Other ink can hide a marker only inside an immutable L0 donor silhouette.
Neither newly activated points nor the fitted path become pixel evidence.
Weights are experimental costs, not calibrated probabilities.
"""
import math

import cv2
import numpy as np

from color_colocated_hypotheses_v46 import _stamp_crop


VERSION = 'frozen_marker_image_cost_v1'


def pixel_cost(expected, own, other, donor, white, valid, legacy_score):
    """Partition expected INK (not the empty hole in an open marker).

    Interior missing ink gets up to 3x boundary weight. Paper contradiction
    costs 2, unexplained non-paper costs 1, occlusion .25, ambiguity .5.
    The existing window score contributes only over observable pixels.
    """
    expected = np.asarray(expected, bool)
    if not expected.any():
        raise ValueError('Soft image cost requires a nonempty native marker core')
    distance = cv2.distanceTransform(np.pad(expected.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    distance = distance.astype(np.float64)
    weight = expected * (1. + 2. * distance / max(float(distance.max()), 1.))
    total = float(weight.sum())
    positive = expected & valid & own & ~other
    occluded = expected & valid & donor & other & ~own
    ambiguous = expected & (~valid | (own & other))
    white_missing = expected & valid & white & ~(positive | occluded | ambiguous)
    unexplained = expected & ~(positive | occluded | ambiguous | white_missing)
    fraction = lambda mask: float(weight[mask].sum() / total)
    f = {name: fraction(mask) for name, mask in (
        ('own', positive), ('occluded', occluded), ('ambiguous', ambiguous),
        ('white_missing', white_missing), ('unexplained', unexplained))}
    score = float(np.clip(legacy_score, 0., 1.))
    visible = max(0., 1. - f['occluded'] - f['ambiguous'])
    terms = dict(white_cost=2.*f['white_missing'], unexplained_cost=f['unexplained'],
                 occlusion_cost=.25*f['occluded'], ambiguity_cost=.5*f['ambiguous'],
                 window_cost=visible*(1.-score))
    return dict(cost=sum(terms.values()), fractions=f, components=terms,
                legacy_score=score, expected_core_pixels=int(expected.sum()),
                expected_weight=total, weight_kind='1 + 2 * normalized interior depth')


class FrozenMarkerImageCost:
    def __init__(self, row, rows, image, plot, legend, verifier):
        self.row, self.rows, self.image = row, rows, image
        self.plot, self.legend, self.verifier = plot, legend, verifier
        self.cache = {}
        if verifier is None or 'marker_alpha' not in row or 'ink_mask' not in row:
            raise ValueError('Soft image objective requires bound template, image and verifier')

    def __call__(self, point):
        cx, cy = float(point['cx']), float(point['cy'])
        key = cx, cy
        if key in self.cache:
            return self.cache[key]
        if not np.isfinite(key).all():
            raise ValueError('Nonfinite marker centre')
        alpha = np.asarray(self.row['marker_alpha'], np.float32)
        tc = self.row['template_center']; h, w = alpha.shape
        # Do not clip the template to plot/image: unobserved parts remain unknown.
        x0, y0 = math.floor(cx-tc[0])-1, math.floor(cy-tc[1])-1
        box = [x0, y0, x0+w+3, y0+h+3]
        expected = _stamp_crop(self.row, cx, cy, box) >= .65
        yy, xx = np.mgrid[box[1]:box[3], box[0]:box[2]]
        ih, iw = self.image.shape[:2]
        inside = (xx >= 0) & (xx < iw) & (yy >= 0) & (yy < ih)
        safe_x, safe_y = xx.clip(0, iw-1), yy.clip(0, ih-1)
        p = self.plot
        valid = inside & (xx >= p[0]) & (xx <= p[2]) & (yy >= p[1]) & (yy <= p[3])
        if self.legend is not None:
            a,b,c,d = self.legend
            valid &= ~((xx >= a) & (xx <= c) & (yy >= b) & (yy <= d))
        own = np.asarray(self.row['ink_mask'], bool)[safe_y, safe_x] & inside
        other = np.zeros_like(expected); donor = np.zeros_like(expected)
        for row in self.rows:
            if row['name'] == self.row['name'] or 'ink_mask' not in row:
                continue
            ink = np.asarray(row['ink_mask'], bool)[safe_y, safe_x] & inside
            other |= ink
            for original in row.get('init_points', []):
                dx, dy = float(original['cx']), float(original['cy'])
                da = np.asarray(row.get('marker_alpha', []))
                if da.ndim != 2 or not da.size:
                    continue
                dc = row['template_center']
                if dx-dc[0] > box[2] or dx-dc[0]+da.shape[1] < box[0] or dy-dc[1] > box[3] or dy-dc[1]+da.shape[0] < box[1]:
                    continue
                donor |= (_stamp_crop(row, dx, dy, box) >= .65) & ink
        # Conservative near-paper test. Unknown dark/coloured pixels are not
        # called white, and arbitrary other-colour lines are not called donors.
        patch = self.image[safe_y, safe_x]
        white = patch.min(axis=2) >= 235
        test = self.verifier(self.row['name'], point)
        result = pixel_cost(expected, own, other, donor, white, valid, test['score'])
        result.update(version=VERSION, cx=cx, cy=cy, legacy_image_accepted=bool(test['accepted']),
                      crop_xyxy=box, other_colour_is_positive_evidence=False,
                      accepted_is_not_a_gate=True)
        self.cache[key] = result
        return result

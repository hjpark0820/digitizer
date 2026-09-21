"""Local image/line evidence for atomic BW Step-5 identity corrections.

This is an eligibility test, not a replacement for SSIM and not a detector.
Short raw L0 fragments remain evidence even when grid refinement drops them.
No fragments are extended and no new marker coordinates are invented.
"""
from functools import lru_cache
import hashlib
import json

import cv2
import numpy as np

from bw_suppressed_v46 import decode_marker_mask

POLICY = 'local_raw_support_identity_v1'


def frozen_policy(segments):
    payload = json.dumps(segments, sort_keys=True, separators=(',', ':')).encode()
    return dict(policy=POLICY, reference_segments_sha256=hashlib.sha256(payload).hexdigest(),
                raw_fragment_count=len(segments), measurement='unchanged_1_minus_SSIM',
                note='Raw fragments gate local edits; they are not confirmed curve identities.')


class Evidence:
    def __init__(self, image, models, segments, plot, legend=None, ignore=None):
        self.models = models
        self.segments = np.asarray(segments, float).reshape(-1, 4)
        self.plot = plot
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink = (gray < 180).astype(np.uint8)
        valid = np.zeros(gray.shape, bool)
        valid[plot[1]:plot[3], plot[0]:plot[2]] = True
        if legend:
            valid[legend[1]:legend[3], legend[0]:legend[2]] = False
        if ignore is not None:
            valid[np.asarray(ignore) > .5] = False
        ink[~valid] = 0
        # One-pixel uncertainty for anti-aliasing, never a scale search.
        self.ink = cv2.dilate(ink, np.ones((3, 3), np.uint8)).astype(bool)
        self.valid = valid
        # Per-run cache: a module-level bound-method cache would retain whole
        # source images across GUI jobs until thousands of entries are evicted.
        self.edge = lru_cache(maxsize=8192)(self._edge)

    def marker(self, p):
        mask = decode_marker_mask(p['marker_mask']).astype(bool)
        h, w = mask.shape
        x = round(p['cx'] - p.get('marker_offset_x', 0)) - w // 2
        y = round(p['cy'] - p.get('marker_offset_y', 0)) - h // 2
        yy, xx = np.nonzero(mask)
        xx, yy = xx + x, yy + y
        inside = (xx >= 0) & (yy >= 0) & (xx < self.ink.shape[1]) & (yy < self.ink.shape[0])
        supported = np.zeros(len(xx), bool)
        supported[inside] = self.ink[yy[inside], xx[inside]] & self.valid[yy[inside], xx[inside]]
        return dict(required_pixels=len(xx), recall=float(supported.mean()) if len(xx) else 0.)

    def _edge(self, ax, ay, bx, by, diameter):
        """Distance to finite, similarly oriented L0 fragments, in diameters.

        Excluding marker-sized end regions prevents the marker itself from
        looking like a connector. A bounded distance tolerates short dash gaps.
        Reference fragments keep their measured positions and lengths.
        """
        a, b = np.array([ax, ay]), np.array([bx, by])
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1.4 * diameter or not len(self.segments):
            return None
        direction = delta / length
        refs = self.segments
        vectors = refs[:, 2:] - refs[:, :2]
        lens = np.linalg.norm(vectors, axis=1)
        aligned = (lens >= 2) & (np.abs(vectors @ direction) / np.maximum(lens, 1e-9) >= np.cos(np.deg2rad(30)))
        refs, vectors, lens = refs[aligned], vectors[aligned], lens[aligned]
        if not len(refs):
            return 3.
        ts = np.linspace(.6 * diameter, length - .6 * diameter, max(3, int(length / max(2, .18 * diameter))))
        samples = a + ts[:, None] * direction
        relative = samples[:, None, :] - refs[None, :, :2]
        fractions = np.clip(np.sum(relative * vectors[None], axis=2) / np.maximum(lens ** 2, 1e-9), 0, 1)
        distances = np.linalg.norm(relative - fractions[:, :, None] * vectors[None], axis=2).min(axis=1)
        return float(np.minimum(distances / diameter, 3).mean())

    def local_geometry(self, p, points):
        sid = p['swatch_id']
        diameter = self.models[sid]['diameter']
        left = sorted((q for q in points if q['swatch_id'] == sid and q['cx'] < p['cx'] - .2 * diameter), key=lambda q: q['cx'])
        right = sorted((q for q in points if q['swatch_id'] == sid and q['cx'] > p['cx'] + .2 * diameter), key=lambda q: q['cx'])
        neighbours = (left[-1:] + right[:1])
        costs = [self.edge(p['cx'], p['cy'], q['cx'], q['cy'], diameter) for q in neighbours]
        valid = [v for v in costs if v is not None]
        bypass = self.edge(left[-1]['cx'], left[-1]['cy'], right[0]['cx'], right[0]['cy'], diameter) if left and right else None
        return dict(connection_cost=float(np.mean(valid)) if valid else None, edges=len(valid), bypass_cost=bypass)

    def reassignment(self, old, new, points):
        diameter = min(self.models[old['swatch_id']]['diameter'], self.models[new['swatch_id']]['diameter'])
        nearby = np.hypot(new['cx'] - old['cx'], new['cy'] - old['cy']) <= .6 * diameter
        marker = self.marker(new)
        source, target = self.local_geometry(old, points), self.local_geometry(new, points)
        a, b, bypass = source['connection_cost'], target['connection_cost'], source['bypass_cost']
        # A series relabelling must fix unsupported connectors and preserve a
        # plausible continuation in BOTH affected series, not merely lower SSIM.
        supported = (nearby and marker['recall'] >= .80 and marker['required_pixels'] >= 5
                     and a is not None and b is not None and bypass is not None
                     and b <= .85 and bypass <= .85 and a - b >= .20 and a - bypass >= .20)
        return dict(accepted=bool(supported), marker=marker, source=source, target=target,
                    reason='pixel_supported_local_identity_reassignment' if supported else 'insufficient_local_identity_evidence')

    def deletion(self, p, points):
        geometry = self.local_geometry(p, points)
        old, new = geometry['connection_cost'], geometry['bypass_cost']
        marker = self.marker(p)
        protected = (bool(p.get('original_detection')) and marker['recall'] >= .80
                     and geometry['edges'] == 2 and old is not None and new is not None
                     and old <= .5 and old - new < .20)
        # More conservative than atomic reassignment: no other series will
        # retain this marker. Require both sides and a large geometry gain.
        accepted = (geometry['edges'] == 2 and old is not None and new is not None
                    and old >= 1.2 and new <= .5 and old - new >= .8)
        return dict(accepted=bool(accepted), protect_observed=bool(protected), source=geometry, marker=marker,
                    reason='preserve_pixel_and_connector_supported_original' if protected else
                           'unsupported_spike_supported_bypass' if accepted else 'insufficient_deletion_evidence')

"""Experimental, label-free curve-union objective for monochrome Step 5.

The reference is the *fixed* union of v45 line segments, not an inferred
per-series path.  Reconstructions connect markers only within ``swatch_id``.
Both distance integrals use the same fixed reference-length denominator, so
deleting a marker cannot also delete the region against which it is scored.

The score is not a marker detector or ground truth: unlabelled black geometry
cannot establish series ownership, and v45 error-bar/text segments can still
contaminate the reference.  All orientations are retained by default.

All coordinates and the marker diameter are in native-image pixels.  Plot and
legend boxes are ``(left, top, right, bottom)``.  Lower objective is better.
The weights below are experimental defaults, not accuracy-calibrated values.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _segment_interval(segment: np.ndarray, box: np.ndarray):
    """Liang-Barsky line clipping interval, including rectangle boundaries."""
    x, y, x1, y1 = map(float, segment)
    dx, dy = x1 - x, y1 - y
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x - box[0]), (dx, box[2] - x),
                 (-dy, y - box[1]), (dy, box[3] - y)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q / p)
        else:
            hi = min(hi, q / p)
        if lo > hi:
            return None
    return lo, hi


def scoped_segments(segments: Iterable, plot_box, legend_box=None) -> np.ndarray:
    """Clip to the plot, split around the legend, and remove exact duplicates.

    No horizontal/vertical orientation filtering or series labelling is done.
    Degenerate and non-finite segments cannot define a curve and are discarded.
    """
    plot = np.asarray(plot_box, dtype=float).reshape(4)
    if not np.isfinite(plot).all() or plot[2] <= plot[0] or plot[3] <= plot[1]:
        raise ValueError("plot_box must have positive width and height")
    legend = None if legend_box is None else np.asarray(legend_box, dtype=float).reshape(4)
    if legend is not None and (not np.isfinite(legend).all()
                               or legend[2] <= legend[0] or legend[3] <= legend[1]):
        raise ValueError("legend_box must have positive width and height")
    unique = {}
    for raw in segments:
        segment = np.asarray(raw, dtype=float).reshape(4)
        if not np.isfinite(segment).all():
            continue
        delta = segment[2:] - segment[:2]
        if np.linalg.norm(delta) < 1e-9:
            continue
        interval = _segment_interval(segment, plot)
        if interval is None:
            continue
        pieces = [interval]
        if legend is not None:
            covered = _segment_interval(segment, legend)
            if covered is not None:
                start, stop = interval
                a, b = max(start, covered[0]), min(stop, covered[1])
                if a < b:
                    pieces = [(start, a), (b, stop)]
        for start, stop in pieces:
            if (stop - start) * np.linalg.norm(delta) < 1e-9:
                continue
            p, q = segment[:2] + start * delta, segment[:2] + stop * delta
            if tuple(p) > tuple(q):
                p, q = q, p
            clipped = np.r_[p, q]
            unique[tuple(np.round(clipped, 8))] = clipped
    return np.asarray(list(unique.values()), dtype=float).reshape(-1, 4)


def _sample_union(segments: np.ndarray, step: float):
    """Arc-length quadrature on a fixed native-coordinate lattice.

    Sampling the dominant coordinate at globally aligned half-step positions
    makes subdivision of a long segment mostly invariant. Repeated samples
    are collapsed instead of counting overlapping detector segments repeatedly.
    Their largest local arc-length weight is retained, not summed. Sub-pixel
    crossings/very short segment splits retain finite quadrature uncertainty.
    """
    samples = {}
    quantization = step / 8.0
    for segment in segments:
        a, b = segment[:2], segment[2:]
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            continue
        tangent = vector / length
        axis = int(np.argmax(np.abs(vector)))
        lower, upper = sorted((a[axis], b[axis]))
        first = int(np.ceil(lower / step - 0.5))
        last = int(np.ceil(upper / step - 0.5))
        values = (np.arange(first, last, dtype=float) + 0.5) * step
        if len(values):
            locations = a + ((values - a[axis]) / vector[axis])[:, None] * vector
            # The selected lattice cell represents ds = dx / |unit tangent x|.
            weights = np.full(len(values), step * length / abs(vector[axis]))
        else:
            locations = np.asarray([(a + b) * 0.5])
            weights = np.asarray([length])
        for location, weight in zip(locations, weights):
            key = tuple(np.rint(location / quantization).astype(np.int64))
            previous = samples.get(key)
            if previous is None or weight > previous[2]:
                samples[key] = (location, tangent, float(weight))
    if not samples:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    points, tangents, weights = zip(*samples.values())
    return np.asarray(points), np.asarray(tangents), np.asarray(weights)


def _nearest_segments(points: np.ndarray, segments: np.ndarray, chunk_size=512):
    """Exact point-to-segment distance; no candidate-dependent x-domain crop."""
    if not len(points):
        return np.empty(0), np.empty((0, 2)), np.empty((0, 2))
    if not len(segments):
        return (np.full(len(points), np.inf), np.full((len(points), 2), np.nan),
                np.zeros((len(points), 2)))
    starts = segments[:, :2]
    vectors = segments[:, 2:] - starts
    norm2 = np.sum(vectors * vectors, axis=1)
    tangents = vectors / np.sqrt(norm2)[:, None]
    distances, closest, nearest_tangents = [], [], []
    for offset in range(0, len(points), chunk_size):
        batch = points[offset:offset + chunk_size]
        relative = batch[:, None] - starts[None]
        t = np.clip(np.sum(relative * vectors[None], axis=2) / norm2[None], 0.0, 1.0)
        projections = starts[None] + t[:, :, None] * vectors[None]
        squared = np.sum((batch[:, None] - projections) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        rows = np.arange(len(batch))
        distances.append(np.sqrt(squared[rows, nearest]))
        closest.append(projections[rows, nearest])
        nearest_tangents.append(tangents[nearest])
    return np.concatenate(distances), np.concatenate(closest), np.concatenate(nearest_tangents)


class GlobalSegmentMetric:
    """Frozen, unlabelled v45 segment union versus swatch-respecting curves.

    ``score`` accepts dictionaries with ``cx``, ``cy`` and ``swatch_id``. Points
    are sorted by x then y inside each identity, as straight-polyline rendering
    does. Same-x points remain vertical within that series; never connect two
    different identities. Missing identities raise instead of guessing a group.
    """

    def __init__(self, reference_segments, diameter, plot_box, legend_box=None,
                 sample_step=None, distance_cap_diameters=2.0,
                 coverage_radius_diameters=0.5, direction_weight=0.15,
                 coverage_weight=0.20):
        self.diameter = float(diameter)
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError("diameter must be a finite positive native-pixel size")
        self.plot_box = np.asarray(plot_box, dtype=float).reshape(4).copy()
        self.legend_box = None if legend_box is None else np.asarray(legend_box, dtype=float).reshape(4).copy()
        self.sample_step = float(sample_step if sample_step is not None
                                 else max(1.0, min(2.0, self.diameter / 4.0)))
        self.distance_cap_diameters = float(distance_cap_diameters)
        self.coverage_radius_diameters = float(coverage_radius_diameters)
        self.direction_weight = float(direction_weight)
        self.coverage_weight = float(coverage_weight)
        values = [self.sample_step, self.distance_cap_diameters,
                  self.coverage_radius_diameters, self.direction_weight, self.coverage_weight]
        if not np.isfinite(values).all() or min(values[:3]) <= 0 or min(values[3:]) < 0:
            raise ValueError("metric scales must be positive and weights nonnegative")
        self.reference_segments = scoped_segments(reference_segments, self.plot_box, self.legend_box)
        self.reference_points, self.reference_tangents, self.reference_weights = _sample_union(
            self.reference_segments, self.sample_step)
        self.reference_length = float(np.sum(self.reference_weights))
        self.available = bool(self.reference_length > 1e-9)
        for array in (self.reference_segments, self.reference_points,
                      self.reference_tangents, self.reference_weights):
            array.setflags(write=False)

    @property
    def config(self):
        return {
            "metric": "experimental_global_v45_segment_union",
            "reference_ownership": "unlabelled; no per-series assignment",
            "candidate_curve": "straight polyline within each swatch_id",
            "normalization": "both integrals / (2 * fixed reference arc length)",
            "diameter_px": self.diameter, "sample_step_px": self.sample_step,
            "distance_cap_diameters": self.distance_cap_diameters,
            "coverage_radius_diameters": self.coverage_radius_diameters,
            "direction_weight": self.direction_weight, "coverage_weight": self.coverage_weight,
            "all_reference_orientations_retained": True,
            "calibration": "experimental; not validated against ground truth",
        }

    def candidate_segments(self, points):
        groups = defaultdict(list)
        for point in points:
            if "swatch_id" not in point or point["swatch_id"] is None:
                raise ValueError("every point must preserve swatch_id; no identity guessing")
            location = (float(point["cx"]), float(point["cy"]))
            if not np.isfinite(location).all():
                raise ValueError("candidate point coordinates must be finite")
            groups[str(point["swatch_id"])].append(location)
        segments = []
        for locations in groups.values():
            locations = sorted(set(locations))
            for a, b in zip(locations, locations[1:]):
                segments.append((*a, *b))
        return scoped_segments(segments, self.plot_box, self.legend_box)

    def score(self, points, details=False) -> dict[str, Any]:
        candidate = self.candidate_segments(points)
        common = {"available": self.available, "reference_length_px": self.reference_length,
                  "reference_segment_count": len(self.reference_segments),
                  "candidate_segment_count": len(candidate)}
        if not self.available:
            return {**common, "objective": None, "chamfer": None, "direction": None,
                    "coverage": None, "reason": "no usable scoped reference segments"}
        candidate_points, candidate_tangents, candidate_weights = _sample_union(candidate, self.sample_step)
        rd, rp, rt = _nearest_segments(self.reference_points, candidate)
        cd, cp, ct = _nearest_segments(candidate_points, self.reference_segments)
        cap_px = self.distance_cap_diameters * self.diameter
        reference_residual = np.minimum(rd / cap_px, 1.0)
        candidate_residual = np.minimum(cd / cap_px, 1.0)
        denom = self.reference_length
        missing = float(self.reference_weights @ reference_residual / denom)
        extra = float(candidate_weights @ candidate_residual / denom)
        chamfer = 0.5 * (missing + extra)
        # Tangents are unoriented: left-to-right and reversed segments agree.
        rangle = 1.0 - np.abs(np.sum(self.reference_tangents * rt, axis=1))
        cangle = 1.0 - np.abs(np.sum(candidate_tangents * ct, axis=1))
        rangle = np.clip(rangle, 0.0, 1.0) * np.exp(-np.square(rd / self.diameter))
        cangle = np.clip(cangle, 0.0, 1.0) * np.exp(-np.square(cd / self.diameter))
        direction = float((self.reference_weights @ rangle + candidate_weights @ cangle) / (2.0 * denom))
        radius_px = self.coverage_radius_diameters * self.diameter
        coverage = float(self.reference_weights @ (rd <= radius_px) / denom)
        objective = chamfer + self.direction_weight * direction + self.coverage_weight * (1.0 - coverage)
        result = {**common, "objective": float(objective), "chamfer": float(chamfer),
                  "direction": direction, "coverage": coverage,
                  "reference_missing": missing, "candidate_extra": extra,
                  "candidate_length_px": float(np.sum(candidate_weights)),
                  "reference_sample_count": len(self.reference_points),
                  "candidate_sample_count": len(candidate_points)}
        if details:
            # Null distances/projections denote absent candidate geometry. This
            # serializes as strict JSON, unlike Infinity/NaN.
            def serial(array):
                return np.where(np.isfinite(array), array, None).tolist()
            result.update({
                "reference_segments": self.reference_segments.tolist(),
                "candidate_segments": candidate.tolist(),
                "reference_samples": self.reference_points.tolist(),
                "candidate_samples": candidate_points.tolist(),
                "reference_sample_weights": self.reference_weights.tolist(),
                "candidate_sample_weights": candidate_weights.tolist(),
                "reference_distances_px": serial(rd), "candidate_distances_px": serial(cd),
                "reference_nearest_candidate": serial(rp), "candidate_nearest_reference": serial(cp),
                "reference_missing_residuals": reference_residual.tolist(),
                "candidate_extra_residuals": candidate_residual.tolist(),
                "reference_direction_residuals": rangle.tolist(),
                "candidate_direction_residuals": cangle.tolist(),
            })
        return result



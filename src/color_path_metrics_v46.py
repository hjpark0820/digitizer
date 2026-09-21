"""Fixed-domain curve-shape objectives for production v46 colour Step 5.

The reference is the independently extracted path, NOT a digitization truth.
By default only observed, retained samples carry positive evidence. The opt-in
estimated_path policy also scores inferred path geometry, with a separately
declared fixed weight; inference is never relabelled as observed image evidence.
Both distance directions are integrated at the SAME frozen reference x samples,
with frozen reference arc-length times signal confidence/policy weights;
deleting anchors cannot shrink the scoring denominator.  Candidate curves are
interpolated using marker coordinates only, never reference y coordinates.

Distances are exact distances to sampled polyline segments. PCHIP is sampled
at <=1 pixel x spacing, so its continuous-curve distance is approximated by
that polyline.  No image registration, scale fitting, or y-normal residual
approximation is used.  These scores assess conditional curve explanation;
they cannot establish marker existence and are not accuracy measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


METRICS = ("chamfer", "directional_chamfer", "hausdorff95")


@dataclass(frozen=True)
class MetricConfig:
    distance_cap_diameters: float = 3.0
    coverage_weight: float = 1.0
    direction_weight: float = 0.15
    x_neighborhood_diameters: float = 2.0
    duplicate_x_tolerance_px: float = 1.0
    reference_gap_px: float = 1.5
    model_sample_step_px: float = 1.0
    position_tolerance_px: float = 0.0

    def __post_init__(self):
        for name in ("distance_cap_diameters", "x_neighborhood_diameters",
                     "reference_gap_px", "model_sample_step_px"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("coverage_weight", "direction_weight", "duplicate_x_tolerance_px",
                     "position_tolerance_px"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _frozen(values, dtype=float):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _segments(points, valid, gap_px):
    """Consecutive valid edges plus isolated degenerate points; no gap bridges."""
    points = np.asarray(points, float).reshape(-1, 2)
    valid = np.asarray(valid, bool)
    linked = (valid[:-1] & valid[1:] &
              (np.diff(points[:, 0]) <= gap_px))
    used = np.zeros(len(points), bool)
    edges = np.flatnonzero(linked)
    used[edges] = True
    used[edges + 1] = True
    isolated = np.flatnonzero(valid & ~used)
    starts = np.concatenate((points[edges], points[isolated]), axis=0)
    ends = np.concatenate((points[edges + 1], points[isolated]), axis=0)
    if len(starts):
        order = np.lexsort((ends[:, 0], starts[:, 0]))
        starts, ends = starts[order], ends[order]
    return starts, ends


def _tangents(points, valid, gap_px):
    """Unit tangent at evaluated reference samples, without differencing across gaps."""
    n = len(points)
    vectors = np.zeros((n, 2), float)
    linked = valid[:-1] & valid[1:] & (np.diff(points[:, 0]) <= gap_px)
    delta = np.diff(points, axis=0)
    vectors[:-1] += np.where(linked[:, None], delta, 0.0)
    vectors[1:] += np.where(linked[:, None], delta, 0.0)
    norm = np.linalg.norm(vectors, axis=1)
    good = norm > 1e-12
    vectors[good] /= norm[good, None]
    return vectors, good


def _arc_quadrature(points, valid, gap_px):
    """Trapezoid arc-length weights within evaluated reference runs; singleton weight 1.

    A steep observed rise carries its actual reference-polyline length instead
    of receiving only a few columns' worth of importance. Nothing is integrated
    across gaps, and candidate marker slopes never affect these frozen weights.
    """
    weights = np.zeros(len(points), float)
    linked = valid[:-1] & valid[1:] & (np.diff(points[:, 0]) <= gap_px)
    lengths = np.where(linked, np.linalg.norm(np.diff(points, axis=0), axis=1), 0.0)
    weights[:-1] += .5 * lengths
    weights[1:] += .5 * lengths
    weights[valid & (weights == 0)] = 1.0
    return weights


@dataclass(frozen=True)
class PathReference:
    path: np.ndarray
    observed: np.ndarray
    weights: np.ndarray
    diameter: float
    config: MetricConfig
    sample_xy: np.ndarray
    sample_weights: np.ndarray
    sample_tangents: np.ndarray
    sample_tangent_valid: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    series_id: str = ""
    # Compatibility: historical consumers use .observed as the evaluation mask.
    # Under estimated_path it therefore includes inference; provenance lives in
    # these separate immutable masks, not in that legacy attribute name.
    true_observed: np.ndarray | None = None
    inferred: np.ndarray | None = None
    reference_policy: str = "observed"
    inferred_weight: float = 0.25

    @property
    def evaluation_mask(self):
        """Truthfully named alias for the legacy .observed scoring mask."""
        return self.observed

    @classmethod
    def from_record(cls, record: dict[str, Any], diameter: float,
                    confidence=None, config: MetricConfig | None = None,
                    reference_policy: str = "observed", inferred_weight: float = 0.25,
                    scope_mask=None):
        """Freeze the scoring domain without changing source observation flags.

        observed (default): original observed & filled & positive confidence.
        estimated_path: also include every finite inferred coordinate in scope,
        including filled=False or val=0 samples, at inferred_weight confidence.
        scope_mask limits either policy, e.g. to the plot ROI outside the legend.
        It excludes samples rather than connecting across excluded intervals.

        .observed retains its historical meaning as the *evaluation* mask for
        existing metric/guard consumers. Use .true_observed and .inferred for
        provenance, and .evaluation_mask in newly written code.
        """
        cfg = config or MetricConfig()
        if reference_policy not in ("observed", "estimated_path"):
            raise ValueError(f"unknown reference policy: {reference_policy}")
        inferred_weight = float(inferred_weight)
        if not np.isfinite(inferred_weight) or not 0 < inferred_weight <= 1:
            raise ValueError("inferred_weight must be finite, positive, and at most 1")
        diameter = float(diameter)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("diameter must be finite and positive")
        path = np.asarray(record["path"], float).reshape(-1, 2)
        n = len(path)
        finite_x = np.isfinite(path[:, 0])
        checked_x = path[:, 0] if reference_policy == "observed" else path[finite_x, 0]
        if n and ((reference_policy == "observed" and not finite_x.all()) or
                  np.any(np.diff(checked_x) <= 0)):
            raise ValueError("reference x coordinates must be finite and strictly increasing")
        obs = np.asarray(record["observed"], bool)
        filled = np.asarray(record.get("filled", np.ones(n, bool)), bool)
        conf = np.asarray(record.get("val", np.ones(n)) if confidence is None
                          else confidence, float)
        if obs.shape != (n,) or filled.shape != (n,) or conf.shape != (n,):
            raise ValueError("observed, filled, and confidence must match path length")
        if not np.isfinite(conf).all() or np.any(conf < 0):
            raise ValueError("confidence must be finite and nonnegative")
        if np.any(conf > 1):
            raise ValueError("confidence must not exceed 1")
        scope = np.ones(n, bool) if scope_mask is None else np.asarray(scope_mask, bool)
        if scope.shape != (n,):
            raise ValueError("scope_mask must match path length")
        finite_in_scope = scope & np.isfinite(path).all(axis=1)
        true_obs = obs & filled & finite_in_scope & (conf > 0)
        inferred = (finite_in_scope & ~true_obs if reference_policy == "estimated_path"
                    else np.zeros(n, bool))
        evaluated = true_obs | inferred
        sample_confidence = np.where(true_obs, conf, np.where(inferred, inferred_weight, 0.0))
        weights = _arc_quadrature(path, evaluated, cfg.reference_gap_px) * sample_confidence
        tangent, tangent_valid = _tangents(path, evaluated, cfg.reference_gap_px)
        starts, ends = _segments(path, evaluated, cfg.reference_gap_px)
        return cls(_frozen(path), _frozen(evaluated, bool), _frozen(weights), diameter,
                   cfg, _frozen(path[evaluated]), _frozen(weights[evaluated]),
                   _frozen(tangent[evaluated]), _frozen(tangent_valid[evaluated], bool),
                   _frozen(starts), _frozen(ends), str(record.get("series_id", "")),
                   _frozen(true_obs, bool), _frozen(inferred, bool),
                   reference_policy, inferred_weight)

    def provenance_counts(self):
        actual = self.observed if self.true_observed is None else self.true_observed
        inferred = np.zeros(len(self.path), bool) if self.inferred is None else self.inferred
        return {"evaluated_samples": len(self.sample_xy),
                "observed_samples": int(actual.sum()),
                "truly_observed_samples": int(actual.sum()),
                "inferred_samples": int(inferred.sum()),
                "reference_policy": self.reference_policy}

    def describe(self):
        actual = self.observed if self.true_observed is None else self.true_observed
        observed_xy = self.path[actual]
        return {"series_id": self.series_id, "path_samples": len(self.path),
                **self.provenance_counts(),
                "fixed_weight_sum": float(self.sample_weights.sum()),
                "diameter_px": self.diameter,
                "observed_x_range": ([float(observed_xy[0, 0]), float(observed_xy[-1, 0])]
                                     if len(observed_xy) else None),
                "evaluated_x_range": ([float(self.sample_xy[0, 0]), float(self.sample_xy[-1, 0])]
                                      if len(self.sample_xy) else None),
                "config": asdict(self.config),
                "reference_frozen": True, "registration": False,
                "interpolated_gaps_are_positive_evidence": False,
                "inferred_geometry_used_for_scoring": self.reference_policy == "estimated_path",
                "inferred_weight": self.inferred_weight,
                "legacy_observed_attribute_means": "evaluation mask; use true_observed for image provenance",
                "integration": ("fixed observed reference arc-length quadrature times fixed confidence"
                                if self.reference_policy == "observed" else
                                "fixed evaluated reference arc-length quadrature times observed confidence or fixed inferred weight")}


def group_markers(markers, tolerance=1.0):
    """Bounded duplicate-x grouping, with conflicting y spread retained."""
    rows = []
    for marker in markers:
        if isinstance(marker, dict):
            xy = ((float(marker["x"]), float(marker["y"])) if "x" in marker
                  else (float(marker["cx"]), float(marker["cy"])))
        elif hasattr(marker, "x") and hasattr(marker, "y"):
            xy = (float(marker.x), float(marker.y))
        elif hasattr(marker, "cx") and hasattr(marker, "cy"):
            xy = (float(marker.cx), float(marker.cy))
        else:
            xy = tuple(map(float, marker))
        if len(xy) != 2 or not np.isfinite(xy).all():
            raise ValueError("marker coordinates must contain two finite values")
        rows.append(xy)
    rows.sort()
    groups = []
    for row in rows:
        if not groups or row[0] - groups[-1][0][0] > tolerance:
            groups.append([row])
        else:
            groups[-1].append(row)
    return [{"x": float(np.median([p[0] for p in g])),
             "y": float(np.median([p[1] for p in g])), "n": len(g),
             "y_spread_px": float(np.ptp([p[1] for p in g]))} for g in groups]


def reconstruct_markers(markers, x_query, model="linear",
                        duplicate_x_tolerance_px=1.0):
    """Marker-only curve on requested x; no extrapolation or reference-y fit."""
    if model not in ("linear", "pchip"):
        raise ValueError(f"unknown reconstruction model: {model}")
    xq = np.asarray(x_query, float)
    if xq.ndim != 1 or not np.isfinite(xq).all():
        raise ValueError("x_query must be a finite one-dimensional array")
    groups = group_markers(markers, duplicate_x_tolerance_px)
    y = np.full(xq.shape, np.nan)
    slope = np.full(xq.shape, np.nan)
    inside = np.zeros(xq.shape, bool)
    if len(groups) >= 2:
        x = np.array([g["x"] for g in groups])
        ay = np.array([g["y"] for g in groups])
        inside = (xq >= x[0]) & (xq <= x[-1])
        if model == "linear":
            slopes = np.diff(ay) / np.diff(x)
            idx = np.clip(np.searchsorted(x, xq[inside], side="right") - 1,
                          0, len(slopes) - 1)
            y[inside] = np.interp(xq[inside], x, ay)
            slope[inside] = slopes[idx]
        else:
            interpolator = PchipInterpolator(x, ay, extrapolate=False)
            y[inside] = interpolator(xq[inside])
            slope[inside] = interpolator.derivative()(xq[inside])
    return {"x": xq.copy(), "y": y, "slope": slope, "bracketed": inside,
            "grouped_anchors": groups, "model": model}


def nearest_segment_distances(query, starts, ends, x_window,
                              query_tangents=None, query_tangent_valid=None):
    """True point-to-segment distances with projected x within +/- x_window.

    Sorted non-overlapping x intervals allow bounded neighbourhood gathering,
    avoiding a dense N-by-N all-pairs allocation. Isolated reference samples
    are valid zero-length target segments but do not supply a tangent.
    """
    query = np.asarray(query, float).reshape(-1, 2)
    starts = np.asarray(starts, float).reshape(-1, 2)
    ends = np.asarray(ends, float).reshape(-1, 2)
    distances = np.full(len(query), np.inf)
    direction = np.zeros(len(query))
    if not len(query) or not len(starts):
        return distances, direction
    if np.any(np.diff(starts[:, 0]) < 0) or np.any(np.diff(ends[:, 0]) < 0):
        raise ValueError("target segments must be sorted by nondecreasing x endpoints")
    lo = np.searchsorted(ends[:, 0], query[:, 0] - x_window, side="left")
    hi = np.searchsorted(starts[:, 0], query[:, 0] + x_window, side="right")
    max_count = int(np.max(hi - lo))
    if max_count <= 0:
        return distances, direction
    # Bound working memory when a caller supplies a large domain/window.
    chunk_size = max(1, min(512, 250000 // max_count))
    offsets = np.arange(max_count)[None, :]
    for left in range(0, len(query), chunk_size):
        right = min(len(query), left + chunk_size)
        raw_idx = lo[left:right, None] + offsets
        valid = raw_idx < hi[left:right, None]
        idx = np.minimum(raw_idx, len(starts) - 1)
        a, b = starts[idx], ends[idx]
        v = b - a
        q = query[left:right, None, :]
        denom = np.sum(v * v, axis=2)
        t = np.sum((q - a) * v, axis=2) / np.maximum(denom, 1e-24)
        dx = v[:, :, 0]
        tlo = np.where(dx > 1e-12,
                       (q[:, :, 0] - x_window - a[:, :, 0]) / np.maximum(dx, 1e-24), 0)
        thi = np.where(dx > 1e-12,
                       (q[:, :, 0] + x_window - a[:, :, 0]) / np.maximum(dx, 1e-24), 1)
        t = np.clip(t, np.maximum(0, tlo), np.minimum(1, thi))
        projection = a + t[:, :, None] * v
        d2 = np.sum((q - projection) ** 2, axis=2)
        d2[~valid] = np.inf
        best = np.argmin(d2, axis=1)
        row = np.arange(right - left)
        distances[left:right] = np.sqrt(d2[row, best])
        if query_tangents is not None:
            vec = v[row, best]
            norm = np.linalg.norm(vec, axis=1)
            ok = (norm > 1e-12) & np.isfinite(distances[left:right])
            if query_tangent_valid is not None:
                ok &= np.asarray(query_tangent_valid)[left:right]
            cos = np.sum(np.asarray(query_tangents)[left:right] * vec, axis=1)
            cos /= np.maximum(norm, 1e-24)
            # Directed left-to-right tangent angle: 0 aligned, 1 opposite.
            direction[left:right] = np.where(ok, np.arccos(np.clip(cos, -1, 1)) / np.pi, 0)
    return distances, direction


def _weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    idx = min(int(np.searchsorted(cumulative, q * cumulative[-1], side="left")),
              len(order) - 1)
    return float(values[order[idx]])


def evaluate_all_metrics(reference: PathReference, markers, model="linear", details=False):
    """Return all three scores from one pair of geometric distance calculations.

    Distance terms use marker-diameter units and cap at 3 diameters by default.
    Missing bracket coverage pays that capped distance in BOTH directions plus
    an explicit coverage term. No-reference cases return objective=None and
    status=no_observed_reference, rather than claiming a perfect score.
    """
    cfg = reference.config
    xy, weights = reference.sample_xy, reference.sample_weights
    n = len(xy)
    curve = reconstruct_markers(markers, xy[:, 0], model, cfg.duplicate_x_tolerance_px)
    if not n:
        return {name: {"metric": name, "model": model, "objective": None,
                       "status": ("no_observed_reference" if reference.reference_policy == "observed"
                                  else "no_evaluated_reference"), "coverage": None,
                       "unbracketed_fraction": None, **reference.provenance_counts(),
                       "fixed_weight_sum": 0.0} for name in METRICS}
    bracketed = curve["bracketed"]
    forward = np.full(n, np.inf)
    reverse = np.full(n, np.inf)
    fangle = np.zeros(n)
    rangle = np.zeros(n)
    if bracketed.any():
        groups = curve["grouped_anchors"]
        start = max(groups[0]["x"], xy[0, 0] - cfg.x_neighborhood_diameters * reference.diameter)
        end = min(groups[-1]["x"], xy[-1, 0] + cfg.x_neighborhood_diameters * reference.diameter)
        # Include anchor knots and query x exactly, so linear interpolation is
        # exact and PCHIP's displayed dense polyline matches the score geometry.
        sample_x = np.unique(np.concatenate((
            np.linspace(start, end, max(2, int(np.ceil((end - start) / cfg.model_sample_step_px)) + 1)),
            xy[bracketed, 0],
            [g["x"] for g in groups if start <= g["x"] <= end])))
        dense = reconstruct_markers(markers, sample_x, model, cfg.duplicate_x_tolerance_px)
        cxy = np.column_stack((sample_x, dense["y"]))
        cstarts, cends = cxy[:-1], cxy[1:]
        window = cfg.x_neighborhood_diameters * reference.diameter
        forward[bracketed], fangle[bracketed] = nearest_segment_distances(
            xy[bracketed], cstarts, cends, window,
            reference.sample_tangents[bracketed], reference.sample_tangent_valid[bracketed])
        model_tangent = np.column_stack((np.ones(n), curve["slope"]))
        model_tangent[bracketed] /= np.linalg.norm(model_tangent[bracketed], axis=1)[:, None]
        reverse[bracketed], rangle[bracketed] = nearest_segment_distances(
            np.column_stack((xy[bracketed, 0], curve["y"][bracketed])),
            reference.segment_starts, reference.segment_ends, window,
            model_tangent[bracketed], np.ones(bracketed.sum(), bool))
    cap = cfg.distance_cap_diameters
    fscaled = np.minimum(np.maximum(forward - cfg.position_tolerance_px, 0) / reference.diameter, cap)
    rscaled = np.minimum(np.maximum(reverse - cfg.position_tolerance_px, 0) / reference.diameter, cap)
    denominator = float(weights.sum())
    mean = lambda values: float(np.dot(values, weights) / denominator)
    missing = mean(~bracketed)
    coverage_term = cfg.coverage_weight * missing
    forward_mean, reverse_mean = mean(fscaled), mean(rscaled)
    chamfer = 0.5 * (forward_mean + reverse_mean)
    direction_term = cfg.direction_weight * 0.5 * (mean(fangle) + mean(rangle))
    h95 = max(_weighted_quantile(fscaled, weights, .95),
              _weighted_quantile(rscaled, weights, .95))
    common = {"model": model, "status": "evaluated", "coverage": 1.0 - missing,
              "unbracketed_fraction": missing, "coverage_component": coverage_term,
              "forward_mean_diameters": forward_mean, "reverse_mean_diameters": reverse_mean,
              "direction_component": direction_term, "hausdorff95_diameters": h95,
              **reference.provenance_counts(), "bracketed_samples": int(bracketed.sum()),
              "fixed_weight_sum": denominator, "grouped_anchor_count": len(curve["grouped_anchors"])}
    objectives = {"chamfer": chamfer + coverage_term,
                  "directional_chamfer": chamfer + direction_term + coverage_term,
                  "hausdorff95": h95 + coverage_term}
    result = {}
    for name in METRICS:
        value = dict(common, metric=name, objective=float(objectives[name]))
        if details:
            value.update(x=xy[:, 0].copy(), y_reference=xy[:, 1].copy(), y_model=curve["y"].copy(),
                         bracketed=bracketed.copy(), weights=weights.copy(),
                         forward_distance_px=forward.copy(), reverse_distance_px=reverse.copy(),
                         local_residual=(0.5 * (fscaled + rscaled) + cfg.coverage_weight * (~bracketed)
                                         + (cfg.direction_weight * 0.5 * (fangle + rangle)
                                            if name == "directional_chamfer" else 0)),
                         grouped_anchors=curve["grouped_anchors"])
        result[name] = value
    return result


def score_markers(reference, markers, metric="chamfer", model="linear", details=False):
    if metric not in METRICS:
        raise ValueError(f"unknown path metric: {metric}")
    return evaluate_all_metrics(reference, markers, model=model, details=details)[metric]

"""Production v46 protection for the first active marker estimates (P0/L0).

A good fit after a deletion is not sufficient evidence that a visible original
marker was false.  For EACH original lost after trial NMS, this guard restores
the original while leaving unrelated trial edits unchanged.  A loss is allowed
only when the isolated edit changes the local reconstructed curve materially
AND improves its explanation of the frozen evaluated reference path.

This is a conditional model-consistency guard, not proof of marker existence.
Linear/PCHIP interpolation can disagree with genuine scattered measurements.
Thresholds below are interpretable experimental defaults, not calibrated truth.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from .color_path_metrics_v46 import (PathReference, nearest_segment_distances,
                                    reconstruct_markers, score_markers)
except ImportError:
    from color_path_metrics_v46 import (PathReference, nearest_segment_distances,
                                   reconstruct_markers, score_markers)


INITIAL_ID = "_initial_id"


@dataclass(frozen=True)
class GuardConfig:
    significant_change_diameters: float = .25
    minimum_local_improvement_diameters: float = .02
    replacement_match_px: float = 18.0
    minimum_local_halfwidth_diameters: float = 1.0
    sample_step_px: float = 1.0
    identity_tolerance_px: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.sample_step_px <= 0 or self.minimum_local_halfwidth_diameters <= 0:
            raise ValueError("sample step and minimum halfwidth must be positive")


def _xy(p):
    if "cx" in p:
        return float(p["cx"]), float(p["cy"])
    return float(p["x"]), float(p["y"])


def _key(p):
    return (*_xy(p), str(p.get("class_name", "")))


def label_initial_points(points, prefix="P0"):
    """Copy active originals and attach identity; do NOT call for suppressed.

    The immutable identity registry is made by InitialPointGuard.  Copying an
    ID onto a moved point never makes the moved point an original again.
    """
    result = deepcopy(list(points))
    for i, point in enumerate(result):
        point[INITIAL_ID] = f"{prefix}:P0:{i:04d}"
    return result


@dataclass(frozen=True)
class _Original:
    id: str
    x: float
    y: float
    class_name: str
    local_lo: float
    local_hi: float


class InitialPointGuard:
    """Evaluate already-NMS'ed trials without changing active/suppressed data."""

    def __init__(self, reference: PathReference, labelled_initial_points,
                 model="linear", config: GuardConfig | None = None):
        self.reference = reference
        self.model = model
        self.config = config or GuardConfig()
        if model not in ("linear", "pchip"):
            raise ValueError("model must be linear or pchip")
        initial = deepcopy(list(labelled_initial_points))
        ids = [str(p.get(INITIAL_ID, "")) for p in initial]
        if any(not k for k in ids) or len(set(ids)) != len(ids):
            raise ValueError("initial points need unique IDs from label_initial_points")
        xs = np.unique([_xy(p)[0] for p in initial])
        halfwidth = reference.diameter * self.config.minimum_local_halfwidth_diameters
        records = []
        for point, identity in zip(initial, ids):
            x, y = _xy(point)
            if not np.isfinite([x, y]).all():
                raise ValueError("initial coordinates must be finite")
            ix = int(np.searchsorted(xs, x))
            # Linear interpolation depends on adjacent anchors. PCHIP also
            # changes neighbouring slopes, so use two original neighbours.
            reach = 2 if model == "pchip" else 1
            lo = min(x - halfwidth, float(xs[max(0, ix-reach)]))
            hi = max(x + halfwidth, float(xs[min(len(xs)-1, ix+reach)]))
            records.append(_Original(identity, x, y, str(point.get("class_name", "")), lo, hi))
        self.originals = tuple(records)
        self._initial_copies = {str(p[INITIAL_ID]): p for p in initial}
        self._residual_cache: dict[Any, dict] = {}

    def describe(self):
        return dict(config=asdict(self.config), protected_original_count=len(self.originals),
                    model=self.model, applies_after_nms=True,
                    identity="initial ID AND frozen original coordinates/class",
                    counterfactual="restore each original independently; remove matched new replacement",
                    geometry="fixed local x quadrature, symmetric point-to-segment distance in diameters",
                    local_evidence="fixed evaluated-reference arc/confidence/policy quadrature; Chamfer plus missing coverage",
                    no_reference_implies_preserve=True, new_points_inherit_protection=False,
                    windows=[dict(id=o.id, x_range=[o.local_lo, o.local_hi]) for o in self.originals])

    def _matches(self, original, point):
        x, y = _xy(point)
        return (str(point.get(INITIAL_ID, "")) == original.id
                and str(point.get("class_name", "")) == original.class_name
                and np.hypot(x-original.x, y-original.y) <= self.config.identity_tolerance_px)

    def _residual(self, points):
        key = tuple(sorted(_key(p) for p in points))
        if key not in self._residual_cache:
            if len(self._residual_cache) >= 512:
                self._residual_cache.clear()
            result = score_markers(self.reference, points, metric="chamfer",
                                   model=self.model, details=True)
            self._residual_cache[key] = result
        return self._residual_cache[key]

    def _curve_geometry(self, trial, restored, original):
        """Actual polyline distances, with no free alignment or domain shrink."""
        cfg, ref = self.config, self.reference
        lo, hi = original.local_lo, original.local_hi
        x = np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/cfg.sample_step_px))+1))
        # Fixed trapezoid x quadrature; neither trial slope nor support changes
        # the denominator. Include original center for narrow displacement.
        x = np.unique(np.r_[x, original.x])
        weights = np.zeros(len(x))
        weights[:-1] += .5*np.diff(x)
        weights[1:] += .5*np.diff(x)
        denominator = weights.sum()
        models = []
        window = ref.config.x_neighborhood_diameters * ref.diameter
        for points in (trial, restored):
            query = reconstruct_markers(points, x, self.model, ref.config.duplicate_x_tolerance_px)
            anchors = query["grouped_anchors"]
            if len(anchors) < 2:
                starts = ends = np.empty((0, 2))
            else:
                a = max(anchors[0]["x"], lo-window)
                b = min(anchors[-1]["x"], hi+window)
                if b < a:
                    starts = ends = np.empty((0, 2))
                else:
                    dense_x = np.unique(np.r_[np.linspace(a,b,max(2,int(np.ceil((b-a)/cfg.sample_step_px))+1)),
                                              [g["x"] for g in anchors if a <= g["x"] <= b]])
                    dense = reconstruct_markers(points, dense_x, self.model, ref.config.duplicate_x_tolerance_px)
                    xy = np.column_stack((dense_x, dense["y"]))
                    starts, ends = xy[:-1], xy[1:]
            models.append((query, starts, ends))
        q0, s0, e0 = models[0]
        q1, s1, e1 = models[1]
        both = q0["bracketed"] & q1["bracketed"]
        one = q0["bracketed"] ^ q1["bracketed"]
        d0 = np.zeros(len(x)); d1 = np.zeros(len(x))
        cap = ref.config.distance_cap_diameters
        d0[one] = cap; d1[one] = cap
        for target, source, starts, ends in ((d0,q0,s1,e1), (d1,q1,s0,e0)):
            if both.any():
                d, _ = nearest_segment_distances(np.column_stack((x[both],source["y"][both])),
                                                 starts,ends,window)
                target[both] = np.minimum(d/ref.diameter,cap)
        per_sample = .5*(d0+d1)
        return dict(change_diameters=float(np.dot(per_sample,weights)/denominator),
                    materially_changed_domain_fraction=float(np.dot(
                        per_sample >= cfg.significant_change_diameters,weights)/denominator),
                    coverage_changed_domain_fraction=float(np.dot(one,weights)/denominator),
                    fixed_geometry_domain_px=float(hi-lo))

    def evaluate(self, before_points, post_nms_trial):
        """All lost originals must pass; caller still requires overall gain.

        Evaluate the FINAL post-NMS points. The result is an admission guard,
        not the metric objective and not a mutation of the supplied point sets.
        """
        before, trial = list(before_points), list(post_nms_trial)
        lost = [o for o in self.originals
                if any(self._matches(o,p) for p in before)
                and not any(self._matches(o,p) for p in trial)]
        if not lost:
            return dict(allowed=True, checks=[], protected_losses=0, blocked_losses=0)
        before_keys = {_key(p) for p in before}
        new = [(i,p) for i,p in enumerate(trial) if _key(p) not in before_keys]
        # Match replacements globally one-to-one. An added point cannot be
        # removed from several restoration trials and pay for several losses.
        edges = sorted((float(np.hypot(_xy(p)[0]-o.x,_xy(p)[1]-o.y)),o.id,i)
                       for o in lost for i,p in new
                       if str(p.get("class_name", "")) == o.class_name
                       and np.hypot(_xy(p)[0]-o.x,_xy(p)[1]-o.y) <= self.config.replacement_match_px)
        matched = {}; used = set()
        for distance, identity, index in edges:
            if identity not in matched and index not in used:
                matched[identity] = (index,distance); used.add(index)
        trial_residual = self._residual(trial)
        checks = []
        for original in lost:
            replacement = matched.get(original.id)
            restored = [deepcopy(p) for i,p in enumerate(trial)
                        if replacement is None or i != replacement[0]]
            restored.append(deepcopy(self._initial_copies[original.id]))
            geometry = self._curve_geometry(trial,restored,original)
            reference = self.reference
            select = ((reference.sample_xy[:,0] >= original.local_lo)
                      & (reference.sample_xy[:,0] <= original.local_hi))
            weight = reference.sample_weights[select]
            count, weight_sum = int(select.sum()), float(weight.sum())
            benefit = before_error = after_error = None
            if weight_sum > 0:
                restored_residual = self._residual(restored)
                before_error = float(np.average(np.asarray(restored_residual["local_residual"])[select],weights=weight))
                after_error = float(np.average(np.asarray(trial_residual["local_residual"])[select],weights=weight))
                benefit = before_error-after_error
            significant = geometry["change_diameters"] >= self.config.significant_change_diameters
            beneficial = benefit is not None and benefit >= self.config.minimum_local_improvement_diameters
            if not significant:
                reason = "preserve_original_negligible_local_curve_change"
            elif benefit is None:
                reason = "preserve_original_no_local_reference"
            elif not beneficial:
                reason = "preserve_original_no_meaningful_local_reference_improvement"
            else:
                reason = "allow_original_edit_meaningful_change_and_local_improvement"
            checks.append(dict(original_id=original.id, original_xy=[original.x,original.y],
                               original_class=original.class_name, allowed=bool(significant and beneficial),
                               reason=reason, local_x_domain=[original.local_lo,original.local_hi],
                               replacement_xy=list(_xy(trial[replacement[0]])) if replacement else None,
                               replacement_distance_px=replacement[1] if replacement else None,
                               **geometry, local_reference_samples=count, fixed_local_reference_weight=weight_sum,
                               restored_local_residual=before_error, trial_local_residual=after_error,
                               local_improvement_diameters=benefit))
        blocked = sum(not row["allowed"] for row in checks)
        return dict(allowed=blocked == 0,checks=checks,protected_losses=len(lost),blocked_losses=blocked)

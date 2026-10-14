"""Source-only grayscale evidence for filled circle/square legend identity.

``classify(rgb, center_xy, diameter, context=None)`` returns JSON-safe entries
``annular`` and ``corner``. Decisions are ``circle``, ``square``,
or ``unknown``; ``class_label`` adds the ``filled_`` prefix when accepted.

The angular method integrates brightness along rays, then compares the measured
radial extents with a constant (circle) and 1/max(|cos|, |sin|) (axis-aligned
square). The corner method compares grayscale at three nested corner/axis
locations after estimating size from vertical radial brightness. Neither method
thresholds pixels into a foreground mask or uses a convex hull.

Rules are fixed from geometry before corpus evaluation. This is deliberately a
two-shape evidence test, not a general symbol recognizer. Its caller must limit
eligibility and require independent corroboration before changing an identity.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


METHOD_VERSION = "gray-annular-v1-frozen-2026-10-02"
RULES = {
    "center_search_diameter_fraction": 0.075,
    "center_search_max_px": 0.6,
    "center_grid_steps": 5,
    "minimum_diameter_px": 3.5,
    "minimum_core_darkness": 0.04,
    "minimum_center_to_local_ink_ratio": 0.45,
    "angular_max_relative_rmse": 0.075,
    "angular_min_error_margin": 0.022,
    "corner_circle_max_ratio": 0.50,
    "corner_square_min_ratio": 0.72,
    "corner_max_opposite_difference": 0.20,
    "corner_min_inner_corner_darkness": 0.72,
    "axis_radius_bounds_diameter_fraction": [0.28, 0.68],
}


def _sample(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation; coordinates outside the pixel domain are NaN."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    valid = (x >= 0) & (y >= 0) & (x <= a.shape[1] - 1) & (y <= a.shape[0] - 1)
    xc, yc = np.clip(x, 0, a.shape[1] - 1), np.clip(y, 0, a.shape[0] - 1)
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    x1, y1 = np.minimum(x0 + 1, a.shape[1] - 1), np.minimum(y0 + 1, a.shape[0] - 1)
    dx, dy = xc - x0, yc - y0
    values = ((1 - dx) * (1 - dy) * a[y0, x0] + dx * (1 - dy) * a[y0, x1]
              + (1 - dx) * dy * a[y1, x0] + dx * dy * a[y1, x1])
    return np.where(valid, values, np.nan)


def _integral(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    return np.sum((values[..., 1:] + values[..., :-1]) * np.diff(radii) / 2, axis=-1)


def _abstain(reason: str, started: float, **features: Any) -> dict[str, Any]:
    return {"decision": "unknown", "class_label": None, "reason": reason,
            "scores": {"circle": None, "square": None}, "features": features,
            "timing_ms": 1000 * (perf_counter() - started)}


def _prepare(patch_rgb: np.ndarray, center_xy: tuple[float, float], diameter: float,
             context: dict[str, Any]) -> dict[str, Any]:
    rgb = np.asarray(patch_rgb, dtype=float)
    if rgb.ndim == 2:
        gray = rgb
    elif rgb.ndim == 3 and rgb.shape[2] >= 3:
        gray = rgb[..., :3] @ np.array([0.299, 0.587, 0.114])
    else:
        raise ValueError("patch_rgb must be HxW or HxWx3 RGB")
    cx, cy = map(float, center_xy)
    yy, xx = np.indices(gray.shape)
    valid = np.isfinite(gray)
    if context.get("valid_mask") is not None:
        valid &= np.asarray(context["valid_mask"], dtype=bool)
    if not np.any(valid):
        return {"darkness": np.full_like(gray, np.nan), "paper_gray": None,
                "core_darkness": 0.0, "local_ink_darkness": 0.0,
                "center_ink_ratio": 0.0, "connector_rows": [],
                "connector_evidence": [], "valid_fraction": 0.0}
    outer = np.maximum(np.abs(xx - cx), np.abs(yy - cy)) >= 0.75 * diameter
    paper_pixels = gray[valid & outer]
    if len(paper_pixels) < 8:
        paper_pixels = gray[valid]
    paper = float(context.get("paper_gray", np.percentile(paper_pixels, 90)))
    darkness = np.maximum(0.0, (paper - gray) / max(paper, 1.0))
    darkness[~valid] = np.nan
    core = darkness[valid & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (0.20 * diameter) ** 2)]
    local = darkness[valid & ((xx - cx) ** 2 + (yy - cy) ** 2 <= (0.60 * diameter) ** 2)]
    core_dark = float(np.median(core)) if len(core) else 0.0
    local_ink = float(np.percentile(local, 90)) if len(local) else 0.0
    connector_rows = [float(row) for row in context.get("connector_rows", [])]
    row_evidence: list[dict[str, float]] = []
    # Only exterior continuations can declare a nuisance row. A square's own
    # horizontal strokes cannot do so because the probes start at 0.75 diameter.
    for y in range(gray.shape[0]):
        if abs(y - cy) > max(1.5, 0.23 * diameter):
            continue
        sides = []
        for side in (-1, 1):
            region = valid[y] & (side * (xx[y] - cx) >= 0.75 * diameter)
            region &= side * (xx[y] - cx) <= 1.25 * diameter
            vals = darkness[y, region]
            if len(vals) >= 3:
                sides.append(float(np.median(vals)))
        if sides and max(sides) > max(0.035, 0.20 * local_ink):
            connector_rows.append(float(y))
            row_evidence.append({"row": float(y), "exterior_darkness": max(sides)})
    return {"darkness": darkness, "paper_gray": paper, "core_darkness": core_dark,
            "local_ink_darkness": local_ink, "center_ink_ratio": core_dark / max(local_ink, 1e-6),
            "connector_rows": sorted(set(connector_rows)), "connector_evidence": row_evidence,
            "valid_fraction": float(np.mean(valid))}


def _centers(center_xy: tuple[float, float], diameter: float):
    span = min(RULES["center_search_max_px"], RULES["center_search_diameter_fraction"] * diameter)
    offsets = np.linspace(-span, span, RULES["center_grid_steps"])
    for oy in offsets:
        for ox in offsets:
            yield float(center_xy[0] + ox), float(center_xy[1] + oy)


def _profiles(prep: dict[str, Any], center: tuple[float, float], diameter: float,
              theta: np.ndarray, radii: np.ndarray) -> np.ndarray:
    x = center[0] + np.cos(theta[:, None]) * radii[None, :]
    y = center[1] + np.sin(theta[:, None]) * radii[None, :]
    return _sample(prep["darkness"], x, y) / max(prep["core_darkness"], 1e-6)


def _angular(prep: dict[str, Any], center_xy: tuple[float, float], diameter: float) -> dict[str, Any]:
    started = perf_counter()
    theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    # Connector rows are excluded from the outer shape evidence geometrically.
    # Inner ray samples can cross a row but remain inside a filled marker.
    if prep["connector_rows"]:
        half_height = max(abs(row - center_xy[1]) for row in prep["connector_rows"]) + 0.6
        theta = theta[np.abs(np.sin(theta)) * (0.35 * diameter) > half_height]
    if len(theta) < 12:
        return _abstain("connector_leaves_too_few_angles", started, retained_angles=len(theta))
    radii = np.linspace(0, 0.90 * diameter, max(33, int(np.ceil(7.2 * diameter)) + 1))
    models = {"circle": np.ones_like(theta),
              "square": 1.0 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))}
    best: dict[str, dict[str, Any]] = {}
    for center in _centers(center_xy, diameter):
        profiles = _profiles(prep, center, diameter, theta, radii)
        good = np.all(np.isfinite(profiles), axis=1)
        if np.count_nonzero(good) < max(12, 0.80 * len(theta)):
            continue
        # A cap limits darker overprinted outlines/connector interiors while
        # preserving all intermediate grayscale values (there is no binarization).
        extent = _integral(np.clip(profiles[good], 0, 1.25), radii)
        for name, full_model in models.items():
            model = full_model[good]
            scale = float(np.dot(extent, model) / np.dot(model, model))
            if not 0.28 * diameter <= scale <= 0.68 * diameter:
                continue
            rmse = float(np.sqrt(np.mean((extent - scale * model) ** 2)) / max(scale, 1e-6))
            distance = float(np.hypot(center[0] - center_xy[0], center[1] - center_xy[1]))
            candidate = {"relative_rmse": rmse, "fitted_axis_radius_px": scale,
                         "center_xy": list(center), "center_offset_px": distance,
                         "angles_degrees": np.degrees(theta[good]).tolist(),
                         "radial_darkness_extent_px": extent.tolist(),
                         "mean_ring_darkness": np.interp(np.array([.20, .35, .50, .65, .80]) * diameter,
                                                          radii, np.mean(profiles[good], axis=0)).tolist(),
                         "ring_radii_px": (np.array([.20, .35, .50, .65, .80]) * diameter).tolist()}
            if name not in best or (rmse, distance) < (best[name]["relative_rmse"], best[name]["center_offset_px"]):
                best[name] = candidate
    if len(best) != 2:
        return _abstain("insufficient_exterior_or_implausible_size", started, fitted_models=list(best))
    winner = min(best, key=lambda name: best[name]["relative_rmse"])
    loser = "square" if winner == "circle" else "circle"
    margin = best[loser]["relative_rmse"] - best[winner]["relative_rmse"]
    reason = "accepted"
    if best[winner]["relative_rmse"] > RULES["angular_max_relative_rmse"]:
        reason = "neither_shape_fits_radial_signature"
    elif margin < RULES["angular_min_error_margin"]:
        reason = "weak_shape_margin"
    decision = winner if reason == "accepted" else "unknown"
    return {"decision": decision, "class_label": "filled_" + winner if decision != "unknown" else None,
            "reason": reason, "scores": {name: float(np.exp(-item["relative_rmse"] / 0.075)) for name, item in best.items()},
            "features": {"winner": winner, "error_margin": margin, "model_fits": best,
                         "retained_angles": len(theta), "search_center_count": 25},
            "timing_ms": 1000 * (perf_counter() - started)}


def _corner(prep: dict[str, Any], center_xy: tuple[float, float], diameter: float) -> dict[str, Any]:
    started = perf_counter()
    theta = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    radii = np.linspace(0, 0.90 * diameter, max(33, int(np.ceil(7.2 * diameter)) + 1))
    candidates = []
    for center in _centers(center_xy, diameter):
        profiles = _profiles(prep, center, diameter, theta, radii)
        if not np.all(np.isfinite(profiles)):
            continue
        extents = _integral(np.clip(profiles, 0, 1.25), radii)
        axis_radius = float(np.mean(extents[[1, 3]]))
        if not 0.28 * diameter <= axis_radius <= 0.68 * diameter:
            continue
        symmetry = float(abs(extents[1] - extents[3]) / axis_radius)
        if not prep["connector_rows"]:
            symmetry += float(abs(extents[0] - extents[2]) / axis_radius)
        # Four corners, each sampled at three scales; matching axis samples
        # use the same radii so overall ink darkness divides out.
        fractions = np.array([0.70, 0.80, 0.90])
        signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        offsets = signs[:, :, None] * (axis_radius * fractions)[None, None, :]
        corners = _sample(prep["darkness"], center[0] + offsets[:, 0], center[1] + offsets[:, 1])
        inner = _sample(prep["darkness"], center[0] + signs[:, 0] * axis_radius * 0.48,
                        center[1] + signs[:, 1] * axis_radius * 0.48)
        axis_indices = [1, 3] if prep["connector_rows"] else [0, 1, 2, 3]
        axis_angles = theta[axis_indices]
        axis = _sample(prep["darkness"], center[0] + np.cos(axis_angles[:, None]) * axis_radius * fractions,
                       center[1] + np.sin(axis_angles[:, None]) * axis_radius * fractions)
        if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(axis)):
            continue
        opposite = float(np.mean(np.abs(corners[:2] - corners[2:])) / max(prep["core_darkness"], 1e-6))
        selection = symmetry + opposite
        candidates.append({"center_xy": list(center), "axis_radius_px": axis_radius,
                           "axis_extent_px": extents.tolist(), "symmetry_error": symmetry,
                           "opposite_corner_difference": opposite,
                           "corner_darkness": corners.tolist(), "axis_darkness": axis.tolist(),
                           "inner_corner_to_core_ratio": float(np.mean(inner) / max(prep["core_darkness"], 1e-6)),
                           "corner_to_axis_ratio": float(np.mean(corners) / max(np.mean(axis), 1e-6)),
                           "axis_to_core_ratio": float(np.mean(axis) / max(prep["core_darkness"], 1e-6)),
                           "selection_symmetry_loss": selection,
                           "center_offset_px": float(np.hypot(center[0] - center_xy[0], center[1] - center_xy[1]))})
    if not candidates:
        return _abstain("insufficient_exterior_or_implausible_size", started)
    feature = min(candidates, key=lambda item: (item["selection_symmetry_loss"], item["center_offset_px"]))
    ratio = feature["corner_to_axis_ratio"]
    reason = "accepted"
    decision = "circle" if ratio <= 0.50 else "square" if ratio >= 0.72 else "unknown"
    if feature["opposite_corner_difference"] > 0.20 or feature["symmetry_error"] > 0.28:
        reason = "asymmetric_shape_evidence"
    elif feature["inner_corner_to_core_ratio"] < 0.72:
        reason = "weak_inner_diagonal_support"
    elif feature["axis_to_core_ratio"] < 0.48:
        reason = "weak_axis_support"
    elif not prep["connector_rows"] and abs(np.mean(np.asarray(feature["axis_extent_px"])[[0, 2]]) / feature["axis_radius_px"] - 1) > 0.20:
        reason = "unequal_horizontal_vertical_size"
    elif decision == "unknown":
        reason = "corner_ratio_in_abstention_band"
    if reason != "accepted":
        decision = "unknown"
    return {"decision": decision, "class_label": "filled_" + decision if decision != "unknown" else None,
            "reason": reason, "scores": {"circle": float(np.clip(1 - ratio, 0, 1)),
                                              "square": float(np.clip(ratio, 0, 1))},
            "features": feature, "timing_ms": 1000 * (perf_counter() - started)}


def classify(patch_rgb: np.ndarray, center_xy: tuple[float, float], diameter: float,
             context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify raw RGB brightness without labels or runtime detector state.

    Coordinates are patch-local pixel centers. ``diameter`` is a rough native
    pixel extent, not fitted against ground truth. Optional ``context`` accepts
    ``valid_mask`` (false for padded pixels), ``paper_gray``, and
    ``connector_rows`` (patch-local y positions). Unknown context keys are ignored.
    Source pixels in the supplied patch determine exterior connector evidence.
    """
    started = perf_counter()
    if not np.isfinite(diameter) or diameter < RULES["minimum_diameter_px"] or not np.all(np.isfinite(center_xy)):
        rejected = _abstain("invalid_or_unresolvable_geometry", started,
                            diameter_px=float(diameter) if np.isfinite(diameter) else None)
        return {"version": METHOD_VERSION, "annular": dict(rejected), "corner": dict(rejected)}
    prep = _prepare(patch_rgb, center_xy, float(diameter), context or {})
    common = {key: value for key, value in prep.items() if key != "darkness"}
    reason = None
    if prep["core_darkness"] < RULES["minimum_core_darkness"]:
        reason = "weak_center_contrast"
    elif prep["center_ink_ratio"] < RULES["minimum_center_to_local_ink_ratio"]:
        reason = "center_not_supported_as_filled"
    if reason:
        angular, corner = _abstain(reason, started), _abstain(reason, started)
    else:
        angular = _angular(prep, center_xy, float(diameter))
        corner = _corner(prep, center_xy, float(diameter))
    return {"version": METHOD_VERSION, "preprocessing": common,
            "annular": angular, "corner": corner,
            "timing_ms": 1000 * (perf_counter() - started)}

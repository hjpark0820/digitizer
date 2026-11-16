"""One-to-one pixel matching to fallible references; no curve-value matching."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _coordinates(points: Sequence[dict[str, Any]]) -> np.ndarray:
    rows = []
    for point in points:
        # x/y are allowed as aliases for detector PIXEL outputs, never for own.json.
        x = point["x_px"] if "x_px" in point else point["x"]
        y = point["y_px"] if "y_px" in point else point["y"]
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Matching requires finite original-image pixel coordinates")
        rows.append((x, y))
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def _summary(tp: int, fp: int, fn: int, distances: list[float]) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_distance_mean_px": float(np.mean(distances)) if distances else None,
        "matched_distance_median_px": float(np.median(distances)) if distances else None,
        "matched_distance_p95_px": float(np.percentile(distances, 95)) if distances else None,
    }


def evaluate_points(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    *, tolerance: float | Mapping[str, float] = 5.0,
    label_aware: bool = False,
) -> dict[str, Any]:
    """Maximum cardinality matching first, minimum total distance second.

    ``tolerance`` may be a scalar pixel radius or a per-reference-series map.
    Coordinates must be in the same original-image frame.  A prediction can
    match at most one reference and a reference at most one prediction.
    Same-position references from different series remain distinct.
    """
    prediction_xy, target_xy = _coordinates(predictions), _coordinates(targets)
    if isinstance(tolerance, Mapping):
        radii = np.asarray([float(tolerance[point["series_id"]]) for point in targets])
    else:
        if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
            raise ValueError("Tolerance must be finite and positive")
        radii = np.full(len(targets), float(tolerance))
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("All series tolerances must be finite and positive")
    matches = []
    if len(predictions) and len(targets):
        distance = np.linalg.norm(prediction_xy[:, None, :] - target_xy[None, :, :], axis=2)
        allowed = distance <= radii[None, :]
        if label_aware:
            allowed &= np.asarray([
                [prediction.get("series_id") == target.get("series_id")
                 and prediction.get("series_id") is not None
                 for target in targets] for prediction in predictions
            ], dtype=bool)
        # Any one extra real match saves B, greater than the maximum possible
        # sum of normalized real-match distances. Dummy columns allow every
        # prediction to be unmatched even when the matrix is rectangular.
        big = float(min(len(predictions), len(targets)) + 1)
        cost = np.full((len(predictions), len(targets) + len(predictions)), big)
        cost[:, :len(targets)] = np.where(
            allowed, distance / max(float(np.max(radii)), 1e-12), 2 * big,
        )
        row_indices, column_indices = linear_sum_assignment(cost)
        for prediction_index, target_index in zip(row_indices, column_indices):
            if target_index < len(targets) and allowed[prediction_index, target_index]:
                matches.append({
                    "prediction_index": int(prediction_index), "target_index": int(target_index),
                    "distance_px": float(distance[prediction_index, target_index]),
                    "tolerance_px": float(radii[target_index]),
                    "label_correct": (
                        predictions[prediction_index].get("series_id") == targets[target_index].get("series_id")
                    ),
                })
    matched_predictions = {match["prediction_index"] for match in matches}
    matched_targets = {match["target_index"] for match in matches}
    result = _summary(len(matches), len(predictions) - len(matches), len(targets) - len(matches),
                      [match["distance_px"] for match in matches])
    result.update({
        "matches": matches,
        "unmatched_prediction_indices": [i for i in range(len(predictions)) if i not in matched_predictions],
        "unmatched_target_indices": [i for i in range(len(targets)) if i not in matched_targets],
        "label_aware": label_aware,
        "tolerance_px": dict(tolerance) if isinstance(tolerance, Mapping) else float(tolerance),
        "interpretation": "Agreement with stored reference positions, not independently certified accuracy",
    })
    return result


def evaluate_panel(
    predictions: Sequence[dict[str, Any]], targets: Sequence[dict[str, Any]],
    diameter_by_series: Mapping[str, float],
) -> dict[str, Any]:
    """Primary radius=max(3,0.35*legend diameter), plus fixed 3/5px sensitivity."""
    adaptive = adaptive_tolerances(targets, diameter_by_series)
    results = {}
    for name, radius in (("adaptive", adaptive), ("fixed3", 3.0), ("fixed5", 5.0)):
        results[name] = {
            "center_only": evaluate_points(predictions, targets, tolerance=radius, label_aware=False),
            "label_aware": evaluate_points(predictions, targets, tolerance=radius, label_aware=True),
        }
    return results


def adaptive_tolerances(
    targets: Sequence[dict[str, Any]], diameter_by_series: Mapping[str, float],
) -> dict[str, float]:
    """Keep failed-template reference series, with a conservative 3px radius.

    Failed/missing/nonfinite template diameters must not delete their reference
    points or abort evaluation. Their undetected points remain false negatives.
    No reference spacing or reference-marker geometry estimates a replacement.
    """
    adaptive = {}
    for series in set(diameter_by_series) | {target["series_id"] for target in targets}:
        value = diameter_by_series.get(series)
        try:
            diameter = float(value)
        except (TypeError, ValueError):
            diameter = 0.0
        adaptive[series] = max(3.0, .35 * diameter) if math.isfinite(diameter) and diameter > 0 else 3.0
    return adaptive


def aggregate_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average counts; macro-F1 averages only defined per-panel scores."""
    distances = [match["distance_px"] for result in results for match in result["matches"]]
    summary = _summary(sum(result["tp"] for result in results),
                       sum(result["fp"] for result in results),
                       sum(result["fn"] for result in results), distances)
    f1s = [result["f1"] for result in results if result["f1"] is not None]
    summary["macro_panel_f1"] = float(np.mean(f1s)) if f1s else None
    summary["panels"] = len(results)
    return summary

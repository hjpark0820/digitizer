"""Typed B&W v46 suppressed hypotheses for identity-safe Step 5 correction.

This pool is a set of hypotheses, not detections or calibrated probabilities.
It keeps verified cross-swatch competition losers and a deliberately narrow
subset of near-pass ambiguous windows. Hard rejections are never promoted.
The thresholds below are proposed experiment settings, not validated accuracy
claims. Full-window sector evidence is mandatory for an ambiguous candidate. Production
captures these scalar metrics in the existing verifier pass; no extra image
verification is required.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "bw-v46-typed-suppressed-v1"


def encode_marker_mask(mask: np.ndarray) -> dict[str, Any]:
    """Lossless row-major ink runs, independent of any result-directory files."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not all(binary.shape):
        raise ValueError("marker_mask must be a nonempty two-dimensional array")
    edges = np.flatnonzero(np.diff(np.r_[False, binary.ravel(), False]))
    runs = np.column_stack((edges[::2], edges[1::2] - edges[::2]))
    return {"shape": list(binary.shape), "runs": runs.tolist()}


def decode_marker_mask(encoded: Mapping[str, Any]) -> np.ndarray:
    """Decode a JSON mask, rejecting malformed dimensions and overlapping runs."""
    shape = encoded.get("shape", ())
    if len(shape) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in shape):
        raise ValueError("marker_mask shape must contain two positive integers")
    size = int(shape[0]) * int(shape[1])
    if size > 16_000_000:
        raise ValueError("marker_mask exceeds the supported pixel count")
    out = np.zeros(size, dtype=bool)
    previous_end = 0
    for run in encoded.get("runs", ()):
        if len(run) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in run):
            raise ValueError("marker_mask runs must contain integer start/length pairs")
        start, length = run
        if start < previous_end or length <= 0 or start + length > size:
            raise ValueError("marker_mask runs must be ordered, disjoint, and inside its shape")
        out[start:start + length] = True
        previous_end = start + length
    return out.reshape(shape)


@dataclass(frozen=True)
class SuppressedPolicy:
    """Near-pass margins are experimental; none imply occlusion by themselves."""

    near_required_margin: float = 0.04
    minimum_core: float = 0.85
    minimum_boundary: float = 0.60
    minimum_contour: float = 0.30
    minimum_sector: float = 0.40
    minimum_radial_sectors: int = 5
    grid_score_margin: float = 0.02
    max_alignment_diameter_fraction: float = 0.22
    same_active_diameter_fraction: float = 0.45
    merge_diameter_fraction: float = 0.30


WINDOW_METRICS = (
    "required_recall", "strict_core_recall", "boundary_recall",
    "contour_support", "minimum_sector_recall", "shape_support",
    "missing_fraction", "extra_fraction", "visible_fraction",
    "occluded_fraction", "contradiction_fraction", "weighted_required_recall",
    "aspect_ratio",
)
GRID_METRICS = (
    "score", "supporting_cells", "orientation_bins", "radial_sectors",
    "raw_hypotheses", "contour_coverage", "contour_precision",
    "foreground_coverage", "direct_foreground_coverage", "centre_fill",
    "centre_fill_similarity",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("swatch_id") or row.get("template") or "")


def _box(value: Sequence[float], name: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{name} must contain four exclusive-edge coordinates")
    out = tuple(_number(v) for v in value)
    if any(v is None for v in out) or not (out[0] < out[2] and out[1] < out[3]):
        raise ValueError(f"Invalid {name}: {value}")
    return out


def _inside(x: float, y: float, box: Sequence[float]) -> bool:
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _window_metrics(row: Mapping[str, Any], detail: Any) -> dict[str, float | None]:
    values = {}
    for key in WINDOW_METRICS:
        value = row.get(key)
        if detail is not None:
            measured = detail.get(key) if isinstance(detail, Mapping) else getattr(detail, key, None)
            if measured is not None:
                value = measured
        values[key] = _number(value)
    return values


def build_suppressed(
    candidates: Sequence[Mapping[str, Any]],
    active_points: Sequence[Mapping[str, Any]],
    plot_box: Sequence[float],
    legend_boxes: Sequence[Sequence[float]] = (),
    *,
    default_diameter: float,
    swatch_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    window_details: Mapping[int, Any] | None = None,
    policy: SuppressedPolicy | None = None,
) -> dict[str, Any]:
    """Build JSON-compatible S candidates without modifying supplied records.

    Coordinates and all boxes are in the original image pixel frame; right and
    bottom box edges are exclusive. ``window_details`` is keyed by the index in
    ``candidates`` and accepts WindowVerification objects or scalar dictionaries.
    Identity/centres always come from candidate records, never array order in a
    separate verifier batch. ``swatch_metadata`` is keyed by swatch_id/template;
    use ``source_diameter`` for the original legend diameter and optionally
    ``class_idx``, ``shape_idx``, ``class_name``, ``shape_hint``. If no explicit
    source diameter is available, the median/default size only sets merge radii;
    ambiguous acceptance keeps the stricter small-template recall threshold.

    ``supporting_cells`` already normalizes overlapping grid votes in production
    _score_candidate. Raw hypotheses are preserved for diagnostics but never
    added as independent evidence. Returned confidence is required-ink recall,
    not a marker-existence probability. Every row retains a fixed swatch/shape
    hypothesis; cross-swatch alternatives can share a centre, never an identity.
    """
    pa = _box(plot_box, "plot_box")
    exclusions = [_box(b, "legend_box") for b in legend_boxes]
    d_default = _number(default_diameter)
    if d_default is None or d_default <= 0:
        raise ValueError("default_diameter must be positive and finite")
    p = policy or SuppressedPolicy()
    metadata = swatch_metadata or {}
    details = window_details or {}
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def reject(index: int, row: Mapping[str, Any], reason: str,
               failures: Sequence[str] = ()) -> None:
        rejected.append({"candidate_index": index, "swatch_id": _identity(row),
                         "cx": _number(row.get("aligned_x", row.get("x"))),
                         "cy": _number(row.get("aligned_y", row.get("y"))),
                         "decision": row.get("decision"), "reason": reason,
                         "failed_checks": list(failures)})

    active = []
    for point in active_points:
        x, y = _number(point.get("cx", point.get("aligned_x", point.get("x")))), _number(point.get("cy", point.get("aligned_y", point.get("y"))))
        sid = _identity(point)
        if x is None or y is None or not sid:
            raise ValueError("Each active point needs finite pixel coordinates and a swatch identity")
        active.append((sid, x, y))

    for index, source in enumerate(candidates):
        row = dict(source)
        sid = _identity(row)
        meta = metadata.get(sid, {})
        x, y = _number(row.get("aligned_x", row.get("x"))), _number(row.get("aligned_y", row.get("y")))
        shape = str(row.get("shape_hint") or row.get("class_name") or meta.get("shape_hint") or meta.get("class_name") or "")
        if not sid or not shape:
            reject(index, row, "missing_typed_identity")
            continue
        if x is None or y is None:
            reject(index, row, "invalid_centre")
            continue
        if not _inside(x, y, pa):
            reject(index, row, "outside_plot")
            continue
        if any(_inside(x, y, b) for b in exclusions):
            reject(index, row, "inside_legend")
            continue
        if row.get("selected"):
            reject(index, row, "already_active")
            continue
        if row.get("excluded_by_confidence_floor") or row.get("exclusion_reason") == "confidence_floor":
            reject(index, row, "explicit_confidence_floor")
            continue
        if row.get("exclusion_reason") not in (None, "", "verified_duplicate", "identity_ambiguous"):
            reject(index, row, "explicit_exclusion", [str(row["exclusion_reason"])])
            continue

        source_diameter = _number(meta.get("source_diameter"))
        if source_diameter is not None and source_diameter <= 0:
            raise ValueError(f"Invalid source_diameter for {sid}")
        effective = _number(row.get("effective_diameter"))
        if effective is None or effective <= 0:
            scale = _number(row.get("window_scale")) or 1.0
            effective = (source_diameter or d_default) * scale
        radius = max(2.0, p.same_active_diameter_fraction * effective)
        if any(s == sid and math.hypot(ax-x, ay-y) <= radius for s, ax, ay in active):
            reject(index, row, "same_swatch_active_duplicate")
            continue

        wm = _window_metrics(row, details.get(index))
        gm = {key: _number(row.get(key)) for key in GRID_METRICS}
        decision = row.get("decision")
        if decision == "verified" and row.get("exclusion_reason") == "identity_ambiguous":
            # Marker existence passed the unchanged window test; only its
            # legend identity remains unresolved. No active competitor needed.
            if wm['required_recall'] is None:
                reject(index,row,'missing_window_recall')
                continue
            admission='verified_identity_ambiguous'
            priority=2
        elif decision == "verified" and row.get("exclusion_reason") == "verified_duplicate":
            competitor = row.get("suppressed_by") or {}
            competitor_id = _identity(competitor)
            if row.get("duplicate_kind") == "same_type_neighbour" or competitor_id == sid:
                reject(index, row, "same_swatch_competition_duplicate")
                continue
            if not competitor_id:
                reject(index, row, "missing_competing_identity")
                continue
            if wm["required_recall"] is None:
                reject(index, row, "missing_window_recall")
                continue
            admission = "verified_cross_swatch_duplicate"
            priority = 2
        elif decision == 'ambiguous' and row.get('deferred_reason') == 'other_colour_identity_unobservable':
            state = row.get('window_colour_visibility') or {}
            # A full-window near pass plus independently visible own ink is a
            # hypothesis only. Missing/invalid diagnostics never admit a point.
            gates = {'visible_fraction':.20,'positive_fraction':.20,'visible_required_recall':.55}
            valid = all(_number(state.get(k)) is not None and float(state[k]) >= v for k,v in gates.items())
            white = _number(state.get('core_paper_fraction'))
            threshold = .72 if (source_diameter or 0)<20 else .59
            if (not valid or white is None or white>.30 or wm['required_recall'] is None
                    or wm['required_recall']<threshold):
                reject(index,row,'insufficient_three_state_evidence')
                continue
            admission='visible_fragment_other_colour_occlusion'
            priority=1
        elif (decision == 'ambiguous' and row.get('geometry_first',{}).get('reason')=='small_hollow_overlap_or_model_residual'):
            g=row['geometry_first'];physical=g.get('physical_residual',{})
            if (g.get('missing_rim_fraction',1.)>.15 or physical.get('paper_on_rim',1.)>.08
                    or g.get('marker_improvement',0.)<.012 or g.get('hollow_margin',0.)<.008):
                reject(index,row,'unsupported_hollow_joint_hypothesis');continue
            admission='observed_hollow_requires_joint_explanation'
            priority=1
        elif decision == "ambiguous":
            if wm["minimum_sector_recall"] is None:
                reject(index, row, "missing_window_sector_evidence")
                continue
            # Unknown original size may not invoke the looser >=20px threshold.
            diameter_for_gates = source_diameter or min(d_default, 19.999)
            required = (0.85 if diameter_for_gates < 20 else 0.72) - p.near_required_margin
            score_floor = (0.72 if diameter_for_gates < 9 else 0.68 if diameter_for_gates < 20 else 0.66) + p.grid_score_margin
            checks = {
                "required_ink_near_pass": (wm["required_recall"], required),
                "core_ink_present": (wm["strict_core_recall"], p.minimum_core),
                "boundary_support": (wm["boundary_recall"], p.minimum_boundary),
                "positive_contour_support": (wm["contour_support"], p.minimum_contour),
                "no_mostly_missing_sector": (wm["minimum_sector_recall"], p.minimum_sector),
                "grid_score": (gm["score"], score_floor),
                "independent_cell_support": (gm["supporting_cells"], 3 if diameter_for_gates < 9 else 4),
                "radial_diversity": (gm["radial_sectors"], p.minimum_radial_sectors),
                "orientation_diversity": (gm["orientation_bins"], 4 if diameter_for_gates < 9 else 5),
            }
            failures = [name for name, (value, minimum) in checks.items() if value is None or value < minimum]
            seed_x, seed_y = _number(row.get("x")), _number(row.get("y"))
            if seed_x is None or seed_y is None:
                failures.append("missing_grid_centre")
            elif math.hypot(x-seed_x, y-seed_y) > max(2.0, p.max_alignment_diameter_fraction * effective):
                failures.append("large_alignment_shift")
            if failures:
                reject(index, row, "ambiguous_not_near_pass", failures)
                continue
            admission = "grid_supported_near_pass_ambiguous"
            priority = 1
        elif decision == "rejected":
            reject(index, row, "hard_window_rejection")
            continue
        else:
            reject(index, row, "not_eligible_window_state")
            continue

        # Do not label missing ink as occlusion. Neighbour relationships are
        # descriptions of alternative identities, not extra positive evidence.
        near_other = [{"swatch_id": s, "cx": ax, "cy": ay}
                      for s, ax, ay in active if s != sid and math.hypot(ax-x, ay-y) <= radius]
        item = {
            "candidate_id": f"S{index+1:05d}", "candidate_index": index,
            "source_candidate_indices": [index], "swatch_id": sid,
            "template": row.get("template") or sid,
            "class_name": row.get("class_name") or meta.get("class_name") or shape,
            "shape_hint": shape, "cx": x, "cy": y,
            "confidence": wm["required_recall"], "source": VERSION,
            "status": "suppressed_hypothesis", "admission_reason": admission,
            "priority": priority, "effective_diameter": effective,
            "source_diameter": source_diameter,
            "grid": gm, "window": {**wm, "decision": decision},
            "suppressed_by": deepcopy(row.get("suppressed_by")),
            "duplicate_kind": row.get("duplicate_kind"),
            "overlapping_active_other_swatches": near_other,
            "occlusion_claim": False,
            "identity_selection": deepcopy(row.get('identity_selection')),
            "colour_visibility": deepcopy(row.get('window_colour_visibility')),
            "alignment": {"seed_x": _number(row.get("x")), "seed_y": _number(row.get("y")),
                          "aligned_x": x, "aligned_y": y,
                          "window_scale": _number(row.get("window_scale"))},
        }
        for key in ("class_idx", "shape_idx"):
            value = row.get(key, meta.get(key))
            if value is not None:
                item[key] = int(value)
        admitted.append(item)

    merged = []
    for item in sorted(admitted, key=lambda v: (-v["priority"], -v["confidence"], -(v["grid"]["score"] or 0), v["candidate_index"])):
        for kept in merged:
            radius = max(2.0, p.merge_diameter_fraction * (item["effective_diameter"] + kept["effective_diameter"]) / 2)
            if item["swatch_id"] == kept["swatch_id"] and math.hypot(item["cx"]-kept["cx"], item["cy"]-kept["cy"]) <= radius:
                kept["source_candidate_indices"].extend(item["source_candidate_indices"])
                rejected.append({"candidate_index": item["candidate_index"], "swatch_id": item["swatch_id"],
                                 "cx": item["cx"], "cy": item["cy"], "decision": item["window"]["decision"],
                                 "reason": "merged_same_swatch_candidate", "failed_checks": [],
                                 "representative_id": kept["candidate_id"]})
                break
        else:
            merged.append(item)

    return {"suppressed": merged, "rejected": sorted(rejected, key=lambda v: v["candidate_index"]),
            "summary": {"version": VERSION, "experimental_unvalidated_thresholds": True,
                        "candidate_count": len(candidates), "active_count": len(active_points),
                        "suppressed_count": len(merged), "rejected_or_merged_count": len(rejected),
                        "admission_counts": dict(Counter(v["admission_reason"] for v in merged)),
                        "rejection_counts": dict(Counter(v["reason"] for v in rejected)),
                        "policy": asdict(p), "coordinates": "native_image_pixels_exclusive_roi_edges",
                        "warning": "Suppressed hypotheses are not confirmed markers; path fit alone does not verify existence."}}

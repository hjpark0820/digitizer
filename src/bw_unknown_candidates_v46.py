"""Pixel-supported, untyped B&W marker regions for the suppressed pool.

This module answers whether a proposed location contains compact ink that is
not explained by exterior straight shafts. It does not decide a marker shape,
series identity, drawing order, or whether another active point should change.
Typed full-window rejection and recall are deliberately not existence gates.
All coordinates are source pixels and all boxes use exclusive right/bottom edges.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


VERSION = "bw-v46-untyped-marker-regions-v1"
MAX_CANDIDATES = 20_000
MAX_BODY_GROUPS = 2_000
MAX_CENTRES_PER_BODY = 5
MAX_DIAMETER = 48.0


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _box(value: Sequence[float], name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        raise ValueError(f"{name} must have four exclusive-edge coordinates")
    out = tuple(_number(v) for v in value)
    if any(v is None for v in out) or out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError(f"Invalid {name}")
    return out


def _inside(x: float, y: float, box: Sequence[float]) -> bool:
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _centre(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    return (_number(row.get("aligned_x", row.get("cx", row.get("x")))),
            _number(row.get("aligned_y", row.get("cy", row.get("y")))))


def _shaft_mask(ink: np.ndarray, valid: np.ndarray, xx: np.ndarray,
                yy: np.ndarray, diameter: float) -> tuple[np.ndarray, list[dict]]:
    """Fit strips on exterior pixels only; the held-out body cannot thicken them.

    Both through-lines and endpoint shafts are allowed. Fitting a strip requires
    an observed exterior run, so the arms of a compact cross are not shafts.
    Hough segments supply directions; actual exterior pixels supply thickness.
    """
    radius = np.hypot(xx, yy)
    exterior = valid & (radius >= .78 * diameter)
    exterior_ink = (ink & exterior).astype(np.uint8)
    minimum_run = max(5, int(round(.55 * diameter)))
    segments = cv2.HoughLinesP(exterior_ink, 1., np.pi / 180.,
                               threshold=minimum_run, minLineLength=minimum_run,
                               maxLineGap=max(1, int(.12 * diameter)))
    predicted = np.zeros(ink.shape, bool)
    if segments is None:
        return predicted, []
    hypotheses = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < minimum_run:
            continue
        tx, ty = dx / length, dy / length
        nx, ny = -ty, tx
        # Convert the Hough patch-coordinate intercept to centre coordinates.
        offset = float(nx * xx[y1, x1] + ny * yy[y1, x1])
        if abs(offset) > .65 * diameter:
            continue
        distance = nx * xx + ny * yy - offset
        near = exterior & ink & (np.abs(distance) <= max(1.5, .13 * diameter))
        if int(near.sum()) < minimum_run:
            continue
        points = np.column_stack((xx[near], yy[near])).astype(np.float32)
        fit = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
        fit_tx, fit_ty, px, py = map(float, fit)
        # A fit spanning unrelated exterior strokes can rotate across the
        # held-out gap. Keep the observed segment direction in that case.
        agreement = abs(tx * fit_tx + ty * fit_ty)
        if agreement >= math.cos(math.radians(6.)):
            tx, ty = fit_tx, fit_ty
            nx, ny = -ty, tx
            offset = nx * px + ny * py
        distance = nx * xx + ny * yy - offset
        transverse = np.abs(nx * points[:, 0] + ny * points[:, 1] - offset)
        half_width = max(.8, float(np.quantile(transverse, .95)) + .65)
        if half_width > max(2.5, .22 * diameter):
            continue
        strip = np.abs(distance) <= half_width
        projection = tx * xx + ty * yy
        supported = exterior & ink & strip
        # Reject diffuse collections of exterior ink masquerading as a shaft.
        bins = np.unique(np.rint(projection[supported]).astype(int))
        runs = np.split(bins, np.flatnonzero(np.diff(bins) > 2) + 1)
        longest = max((len(run) for run in runs), default=0)
        if longest < minimum_run:
            continue
        angle = float(math.degrees(math.atan2(ty, tx)) % 180.)
        hypotheses.append((int(supported.sum()), angle, offset, half_width, strip))
    selected = []
    for support, angle, offset, half_width, strip in sorted(hypotheses, key=lambda q: -q[0]):
        # Pixel-mask overlap avoids normal-sign ambiguities in the line fit.
        exterior_size = max(1, int((strip & exterior).sum()))
        if int((strip & predicted & exterior).sum()) / exterior_size > .65:
            continue
        predicted |= strip
        selected.append({"angle_degrees": round(angle, 3),
                         "offset": round(offset, 3),
                         "half_width": round(half_width, 3),
                         "exterior_ink_pixels": support})
        if len(selected) >= 6:
            break
    return predicted, selected


def _evidence(image: np.ndarray, x: float, y: float, diameter: float,
              plot: Sequence[float], legends: Sequence[Sequence[float]]) -> dict:
    radius = int(math.ceil(1.65 * diameter))
    left = max(0, int(math.floor(x)) - radius)
    top = max(0, int(math.floor(y)) - radius)
    right = min(image.shape[1], int(math.floor(x)) + radius + 1)
    bottom = min(image.shape[0], int(math.floor(y)) + radius + 1)
    patch = image[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gy, gx = np.indices(gray.shape, dtype=np.float32)
    gx += left
    gy += top
    xx, yy = gx - x, gy - y
    valid = (gx >= plot[0]) & (gx < plot[2]) & (gy >= plot[1]) & (gy < plot[3])
    for a, b, c, d in legends:
        valid &= ~((gx >= a) & (gx < c) & (gy >= b) & (gy < d))
    # A fixed contrast floor prevents almost-white JPEG noise becoming ink.
    ink = (gray < 195) & valid
    body = (np.hypot(xx, yy) <= .64 * diameter) & valid
    count = int((ink & body).sum())
    result = {"passed": False, "reason": "no_compact_ink", "diameter": diameter,
              "body_ink_pixels": count, "measurement_box": [left, top, right, bottom],
              "body_pixels_used_for_shaft_fit": 0,
              "typed_recall_used_as_existence": False}
    if count < max(6, .035 * diameter * diameter):
        return result
    shaft, lines = _shaft_mask(ink, valid, xx, yy, diameter)
    residual = ink & body & ~shaft
    count_residual = int(residual.sum())
    result.update(exterior_shafts=lines, residual_ink_pixels=count_residual,
                  residual_fraction=round(count_residual / count, 5))
    if count_residual < max(6, .045 * diameter * diameter):
        result["reason"] = "exterior_shafts_explain_ink"
        return result
    ys, xs = np.nonzero(residual)
    points = np.column_stack((xs, ys)).astype(float)
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T, bias=True))
    minor_spread = math.sqrt(max(0., float(eigenvalues[0])))
    width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    # A residual sliver is not a body, even when a typed template has high recall.
    compact = (minor_spread >= max(.75, .09 * diameter)
               and min(width, height) >= max(3., .24 * diameter)
               and count_residual / count >= .10)
    # Several unrelated thin slivers have a two-dimensional joint covariance.
    # Demand at least one connected residual fragment with its own compact
    # extent, so axis ticks and remaining parallel shafts cannot vote together.
    count_components, labels = cv2.connectedComponents(residual.astype(np.uint8), connectivity=8)
    fragments = []
    for label in range(1, count_components):
        fy, fx = np.nonzero(labels == label)
        if len(fx) < max(5, .025 * diameter * diameter):
            continue
        spread = math.sqrt(max(0., float(np.linalg.eigvalsh(
            np.cov(np.column_stack((fx, fy)).T, bias=True))[0])))
        fragments.append({"pixels": len(fx), "minor_spread": round(spread, 5),
                          "width": int(fx.max() - fx.min() + 1),
                          "height": int(fy.max() - fy.min() + 1)})
    coherent = any(q["minor_spread"] >= max(.75, .085 * diameter)
                   and min(q["width"], q["height"]) >= max(3., .20 * diameter) for q in fragments)
    compact = compact and coherent
    result.update(minor_spread=round(minor_spread, 5), residual_width=width,
                  residual_height=height, passed=bool(compact),
                  connected_residual_fragments=fragments,
                  reason="compact_2d_residual" if compact else "line_like_residual",
                  residual_bbox=[left + int(xs.min()), top + int(ys.min()),
                                 left + int(xs.max()) + 1, top + int(ys.max()) + 1])
    return result


def build_regions(image_bgr: np.ndarray, plot_box: Sequence[float],
                  legend_boxes: Sequence[Sequence[float]],
                  candidates: Sequence[Mapping[str, Any]],
                  active_points: Sequence[Mapping[str, Any]],
                  swatch_metadata: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    """Return bounded, JSON-finite U regions without mutating any input.

    Regions are hypotheses about physical ink, not additional detections. A
    region near an active point can describe the same body or unresolved
    overlapping bodies; proximity never deletes evidence or assigns identity.
    Alternative swatch IDs record proposal provenance only. They are not votes.
    """
    image = np.asarray(image_bgr)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image_bgr must be a grayscale or three-channel image")
    if not all(image.shape[:2]) or image.dtype != np.uint8:
        raise ValueError("image_bgr must be a nonempty uint8 image")
    plot = _box(plot_box, "plot_box")
    legends = [_box(box, "legend_box") for box in legend_boxes]
    metadata = swatch_metadata or {}
    diameters = [(_number(value.get("effective_diameter"))
                  or _number(value.get("source_diameter"))) for value in metadata.values()]
    diameters = [value for value in diameters if value is not None and 2 <= value <= MAX_DIAMETER]
    default_diameter = float(np.median(diameters)) if diameters else 10.
    rejected = Counter()
    proposals = []
    for index, row in enumerate(candidates[:MAX_CANDIDATES]):
        x, y = _centre(row)
        if x is None or y is None:
            rejected["invalid_coordinate"] += 1
            continue
        if not _inside(x, y, plot) or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            rejected["outside_plot"] += 1
            continue
        if any(_inside(x, y, box) for box in legends):
            rejected["legend"] += 1
            continue
        key = str(row.get("swatch_id") or row.get("template") or "")
        if not key:
            rejected["missing_swatch_provenance"] += 1
            continue
        diameter = (_number(row.get("effective_diameter"))
                    or _number(metadata.get(key, {}).get("effective_diameter"))
                    or _number(metadata.get(key, {}).get("source_diameter"))
                    or default_diameter)
        if not 2 <= diameter <= MAX_DIAMETER:
            rejected["unsupported_diameter"] += 1
            continue
        proposals.append({"index": index, "x": x, "y": y, "diameter": diameter,
                          "key": key, "decision": str(row.get("decision") or "unknown")})
    # Fixed anchors prevent transitive merging all the way along a curve.
    groups = []
    buckets: dict[tuple[int, int], list[int]] = {}
    cell = MAX_DIAMETER
    for proposal in sorted(proposals, key=lambda q: (q["y"], q["x"], q["key"], q["index"])):
        bx, by = int(proposal["x"] // cell), int(proposal["y"] // cell)
        neighbours = [i for xx in range(bx - 1, bx + 2) for yy in range(by - 1, by + 2)
                      for i in buckets.get((xx, yy), [])]
        nearby = [i for i in neighbours
                  if math.hypot(proposal["x"] - groups[i][0]["x"],
                                proposal["y"] - groups[i][0]["y"])
                  <= .65 * min(proposal["diameter"], groups[i][0]["diameter"])]
        if nearby:
            target = min(nearby, key=lambda i: math.hypot(proposal["x"] - groups[i][0]["x"],
                                                         proposal["y"] - groups[i][0]["y"]))
            groups[target].append(proposal)
        elif len(groups) < MAX_BODY_GROUPS:
            buckets.setdefault((bx, by), []).append(len(groups))
            groups.append([proposal])
        else:
            rejected["body_group_budget"] += 1
    active = [(i, *_centre(point)) for i, point in enumerate(active_points)]
    regions = []
    tested_centres = 0
    for group in groups:
        diameter = float(np.median([q["diameter"] for q in group]))
        centre = (float(np.median([q["x"] for q in group])),
                  float(np.median([q["y"] for q in group])))
        locations = list(dict.fromkeys([centre] + [(q["x"], q["y"]) for q in group]))[:MAX_CENTRES_PER_BODY]
        checks = [(xy, _evidence(image, *xy, diameter, plot, legends)) for xy in locations]
        tested_centres += len(checks)
        passed = [q for q in checks if q[1]["passed"]]
        if not passed:
            rejected[checks[0][1]["reason"]] += 1
            continue
        # Prefer the common centre; use a proposal centre only if common failed.
        (x, y), evidence = passed[0]
        # The ROI includes shaft crossings inside the observed compact body;
        # its bounds never extend into excluded space merely to complete shape.
        half = .64 * diameter
        box = [max(0, int(math.floor(plot[0])), int(math.floor(x - half))),
               max(0, int(math.floor(plot[1])), int(math.floor(y - half))),
               min(image.shape[1], int(math.ceil(plot[2])), int(math.ceil(x + half + 1))),
               min(image.shape[0], int(math.ceil(plot[3])), int(math.ceil(y + half + 1)))]
        near_active = [index for index, ax, ay in active if ax is not None and ay is not None
                       and math.hypot(x - ax, y - ay) <= diameter]
        evidence.update(tested_centres=len(checks),
                        typed_decisions=dict(sorted(Counter(q["decision"] for q in group).items())),
                        nearby_active_indices=near_active,
                        active_overlap_unresolved=bool(near_active),
                        marker_count_unresolved=True,
                        shape_and_series_unresolved=True)
        token = f"{VERSION}:{x:.3f}:{y:.3f}:{','.join(map(str, box))}"
        regions.append({"region_id": "U" + hashlib.sha256(token.encode("ascii")).hexdigest()[:16],
                        "source": VERSION, "version": VERSION,
                        "kind": "untyped_marker_region", "status": "suppressed_untyped",
                        "cx": x, "cy": y, "bbox": box,
                        "source_candidate_indices": sorted(q["index"] for q in group),
                        "alternative_swatch_ids": sorted({q["key"] for q in group if q["key"]}),
                        "candidate_centres": [dict(swatch_id=key, cx=cx, cy=cy)
                                              for key, cx, cy in list(dict.fromkeys(
                                                  (q["key"], q["x"], q["y"]) for q in group))[:12]],
                        "source_patch_sha256": hashlib.sha256(np.ascontiguousarray(
                            image[box[1]:box[3], box[0]:box[2]]).tobytes()).hexdigest(),
                        "existence_evidence": evidence})
    return {"regions": regions, "summary": {
        "version": VERSION,
        "status": "partial_budget" if (len(candidates) > MAX_CANDIDATES or rejected["body_group_budget"]) else "completed",
        "input_candidates": len(candidates), "eligible_candidates": len(proposals),
        "body_groups": len(groups), "tested_centres": tested_centres,
        "retained_regions": len(regions), "rejections": dict(sorted(rejected.items())),
        "candidate_budget_exhausted": len(candidates) > MAX_CANDIDATES,
        "body_budget_exhausted": bool(rejected["body_group_budget"]),
        "maximum_candidates": MAX_CANDIDATES, "maximum_body_groups": MAX_BODY_GROUPS,
        "maximum_centres_per_body": MAX_CENTRES_PER_BODY,
        "active_points_modified": False, "source_pixels_modified": False,
        "typed_acceptance_used_as_existence_gate": False,
        "limitations": ["Regions are uncalibrated existence hypotheses, not detections.",
                       "Curved connectors and compact error-bar junctions may remain ambiguous.",
                       "An active overlap does not establish an additional marker.",
                       "Only existing candidate locations are examined."]}}


def validate_regions(image_bgr: np.ndarray, regions: Sequence[Mapping[str, Any]],
                     plot_box: Sequence[float], legend_boxes: Sequence[Sequence[float]] = ()) -> list[dict]:
    """Validate a saved U pool against the current source pixels, then deep-copy.

    Invalid or stale entries raise ValueError. Validation does not rebuild U or
    modify source pixels, supplied entries, active detections, or typed S.
    """
    if isinstance(regions, (list, tuple)) and not regions:
        return []
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or not all(image.shape[:2]):
        raise ValueError("Invalid source image for untyped regions")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Invalid source image channel count for untyped regions")
    plot = _box(plot_box, "plot_box")
    legends = [_box(box, "legend_box") for box in legend_boxes]
    if not isinstance(regions, (list, tuple)) or len(regions) > MAX_BODY_GROUPS:
        raise ValueError("Invalid or unbounded untyped region pool")
    ids = set()
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("Each untyped region must be a mapping")
        try:
            json.dumps(region, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Untyped region must contain finite JSON values") from exc
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or not region_id or region_id in ids:
            raise ValueError("Untyped region IDs must be nonempty and unique")
        ids.add(region_id)
        if region.get("kind") != "untyped_marker_region" or region.get("status") != "suppressed_untyped":
            raise ValueError("Invalid untyped region kind or status")
        if region.get("source") != VERSION or region.get("version") != VERSION:
            raise ValueError("Unsupported untyped region source or version")
        evidence = region.get("existence_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
            raise ValueError("Untyped region lacks accepted pixel existence evidence")
        x, y = _number(region.get("cx")), _number(region.get("cy"))
        if x is None or y is None or not _inside(x, y, plot) or any(_inside(x, y, b) for b in legends):
            raise ValueError("Untyped region centre must be in the plot and outside legends")
        box = _box(region.get("bbox", ()), "untyped region bbox")
        if any(v != int(v) for v in box):
            raise ValueError("Untyped region bbox must use integer pixel edges")
        a, b, c, d = map(int, box)
        if a < 0 or b < 0 or c > image.shape[1] or d > image.shape[0] or not _inside(x, y, box):
            raise ValueError("Untyped region bbox must contain its centre within the image")
        alternatives = region.get("alternative_swatch_ids")
        if (not isinstance(alternatives, list) or not alternatives
                or any(not isinstance(key, str) or not key for key in alternatives)
                or len(alternatives) != len(set(alternatives))):
            raise ValueError("Untyped region needs unique nonempty alternative swatch IDs")
        centres = region.get("candidate_centres", [])
        if not isinstance(centres, list) or not centres or len(centres) > 12:
            raise ValueError("Untyped region needs bounded proposal centres")
        for centre in centres:
            if not isinstance(centre, Mapping):
                raise ValueError("Malformed untyped proposal centre")
            cx, cy = _number(centre.get("cx")), _number(centre.get("cy"))
            if (cx is None or cy is None or not _inside(cx, cy, plot)
                    or not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0])
                    or any(_inside(cx, cy, legend) for legend in legends)
                    or centre.get("swatch_id") not in alternatives):
                raise ValueError("Untyped proposal centre is outside the valid source region")
        if len({(q["swatch_id"], q["cx"], q["cy"]) for q in centres}) != len(centres):
            raise ValueError("Untyped proposal centres must be unique")
        indices = region.get("source_candidate_indices")
        if (not isinstance(indices, list) or not indices
                or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("Untyped region needs unique nonnegative source candidate indices")
        expected = hashlib.sha256(np.ascontiguousarray(image[b:d, a:c]).tobytes()).hexdigest()
        if region.get("source_patch_sha256") != expected:
            raise ValueError("Untyped region source pixels changed; saved region is stale")
    return deepcopy(list(regions))

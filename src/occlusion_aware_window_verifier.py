"""Full-window exact marker verification with explicit line occlusion handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
import math
import time

import cv2
import numpy as np

from partial_swatch_detector import SwatchTemplate, ink_membership


@dataclass
class WindowVerification:
    x: float
    y: float
    aligned_x: float
    aligned_y: float
    scale: float
    score: float
    exact_iou: float
    exact_precision: float
    exact_recall: float
    required_recall: float
    missing_fraction: float
    extra_fraction: float
    visible_fraction: float
    occluded_fraction: float
    contradiction_fraction: float
    decision: str
    plot_window: np.ndarray = field(repr=False)
    plot_ink: np.ndarray = field(repr=False)
    template_mask: np.ndarray = field(repr=False)
    marker_roi: np.ndarray = field(repr=False)
    occluder_mask: np.ndarray = field(repr=False)
    exact_match: np.ndarray = field(repr=False)
    required_missing: np.ndarray = field(repr=False)
    extra: np.ndarray = field(repr=False)
    contradiction: np.ndarray = field(repr=False)
    # Legacy results retain None for these opt-in B&W raster diagnostics.
    weighted_required_recall: float | None = None
    strict_core_recall: float | None = None
    boundary_recall: float | None = None
    shape_support: float | None = None
    minimum_sector_recall: float | None = None
    matching_profile: str = "legacy"
    aspect_ratio: float = 1.0
    contour_support: float | None = None
    geometry_fixed: bool = False
    compute_diagnostics: dict = field(default_factory=dict, repr=False)


def _crop_padded(array: np.ndarray, centre, size: int, fill=0):
    half = size // 2
    left = int(round(centre[0])) - half
    top = int(round(centre[1])) - half
    output_shape = (size, size) + array.shape[2:]
    output = np.full(output_shape, fill, dtype=array.dtype)
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(array.shape[1], left + size)
    source_bottom = min(array.shape[0], top + size)
    if source_right > source_left and source_bottom > source_top:
        output[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = array[source_top:source_bottom, source_left:source_right]
    return output


def _render_template(mask: np.ndarray, window_size: int, scale: float, dx: int, dy: int):
    height, width = mask.shape
    resized = cv2.resize(
        mask.astype(np.uint8),
        (max(3, int(round(width * scale))), max(3, int(round(height * scale)))),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    output = np.zeros((window_size, window_size), dtype=bool)
    left = (window_size - resized.shape[1]) // 2 + dx
    top = (window_size - resized.shape[0]) // 2 + dy
    source_left = max(0, -left)
    source_top = max(0, -top)
    source_right = min(resized.shape[1], window_size - left)
    source_bottom = min(resized.shape[0], window_size - top)
    if source_right > source_left and source_bottom > source_top:
        output[
            top + source_top : top + source_bottom,
            left + source_left : left + source_right,
        ] = resized[source_top:source_bottom, source_left:source_right]
    return output


def _filled_hull(mask: np.ndarray):
    y, x = np.nonzero(mask)
    output = np.zeros_like(mask, dtype=np.uint8)
    if len(x) >= 3:
        hull = cv2.convexHull(np.column_stack((x, y)).astype(np.int32))
        cv2.fillConvexPoly(output, hull, 1)
    return output.astype(bool)


def _line_occluder(plot_ink: np.ndarray, template: np.ndarray, diameter: float):
    """Explain only elongated ink continuing outside the expected marker."""
    tolerance = max(1, int(round(0.04 * diameter)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    template_near = cv2.dilate(template.astype(np.uint8), kernel) > 0
    residual = plot_ink & ~template_near
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8
    )
    size = plot_ink.shape[0]
    occluder = np.zeros_like(plot_ink, dtype=np.uint8)
    minimum_span = max(4.0, 0.34 * diameter)

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 2:
            continue
        component_y, component_x = np.nonzero(labels == label)
        points = np.column_stack((component_x, component_y)).astype(np.float32)
        if len(points) < 2:
            continue
        centred = points - points.mean(axis=0, keepdims=True)
        covariance = centred.T @ centred / max(len(points) - 1, 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float((eigenvalues[-1] + 1e-3) / (eigenvalues[0] + 1e-3))
        span = float(max(width, height))
        touches_border = x <= 1 or y <= 1 or x + width >= size - 1 or y + height >= size - 1
        if span < minimum_span or (elongation < 4.0 and not touches_border):
            continue

        vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        norm = max(float(math.hypot(float(vx), float(vy))), 1e-6)
        vx, vy = float(vx) / norm, float(vy) / norm
        extension = 1.5 * size
        point0 = (int(round(float(cx) - extension * vx)), int(round(float(cy) - extension * vy)))
        point1 = (int(round(float(cx) + extension * vx)), int(round(float(cy) + extension * vy)))
        estimated_thickness = int(round(area / max(span, 1.0)))
        thickness = min(max(1, estimated_thickness + 1), max(2, int(round(0.13 * diameter))))
        cv2.line(occluder, point0, point1, 1, thickness, cv2.LINE_8)

    # Only a narrow band supported by observed ink or passing through the marker
    # is neutral. This prevents a spurious fitted line from hiding the whole ROI.
    support = cv2.dilate(plot_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    marker_hull = _filled_hull(template)
    return (occluder > 0) & (support | marker_hull)


def _evaluate_alignment(plot_ink, template_mask, diameter):
    marker_hull = _filled_hull(template_mask)
    margin = max(1, int(round(0.08 * diameter)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    roi = cv2.dilate(marker_hull.astype(np.uint8), kernel) > 0
    occluder = _line_occluder(plot_ink, template_mask, diameter)
    clean = roi & ~occluder
    visible_template = template_mask & clean
    observed_clean = plot_ink & clean
    exact_match = visible_template & observed_clean
    missing = visible_template & ~plot_ink
    extra = observed_clean & ~template_mask
    contradiction = missing | extra

    # B&W plots are an additive-ink case: another curve, error bar, or marker
    # can add black pixels, but cannot erase a black pixel required by the
    # target marker.  Therefore required-marker pixels are checked against the
    # complete template even inside the estimated occluder band.  Extra ink is
    # retained as a weak error only.
    required_match = template_mask & plot_ink
    required_missing = template_mask & ~plot_ink
    required_count = int(template_mask.sum())
    required_recall = int(required_match.sum()) / max(required_count, 1)
    missing_fraction = int(required_missing.sum()) / max(required_count, 1)
    extra_fraction = int(extra.sum()) / max(required_count, 1)

    true_positive = int(exact_match.sum())
    template_count = int(visible_template.sum())
    observed_count = int(observed_clean.sum())
    union_count = int((visible_template | observed_clean).sum())
    precision = true_positive / max(observed_count, 1)
    recall = true_positive / max(template_count, 1)
    iou = true_positive / max(union_count, 1)
    visible_fraction = template_count / max(int(template_mask.sum()), 1)
    occluded_fraction = 1.0 - visible_fraction
    contradiction_fraction = int(contradiction.sum()) / max(int(roi.sum()), 1)
    score = (
        0.82 * required_recall
        + 0.10 * recall
        + 0.08 * precision
        - 0.03 * min(extra_fraction, 1.0)
    )
    return {
        "score": float(score),
        "exact_iou": float(iou),
        "exact_precision": float(precision),
        "exact_recall": float(recall),
        "required_recall": float(required_recall),
        "missing_fraction": float(missing_fraction),
        "extra_fraction": float(extra_fraction),
        "visible_fraction": float(visible_fraction),
        "occluded_fraction": float(occluded_fraction),
        "contradiction_fraction": float(contradiction_fraction),
        "marker_roi": roi,
        "occluder_mask": occluder,
        "exact_match": exact_match,
        "required_missing": required_missing,
        "extra": extra,
        "contradiction": contradiction,
    }


def verify_marker_window(
    image: np.ndarray,
    template: SwatchTemplate,
    x: float,
    y: float,
    *,
    fixed_geometry: bool = False,
    backend: str = 'cpu',
    gpu_batch_size: int = 512,
    occlusion_mask: np.ndarray | None = None,
    uncertainty_mask: np.ndarray | None = None,
    search_scales=None,
    lock_aspect: bool = False,
    colour_observation=None,
) -> WindowVerification:
    """Verify a candidate, optionally locking the supplied B&W raster geometry.

    ``fixed_geometry=True`` permits centre alignment only. The caller must
    apply any approved legend-to-plot scale to ``template`` before calling.
    Legacy/color matching retains its old default and explicitly declines
    fixed geometry rather than silently performing its normal scale search.
    ``search_scales`` overrides the uncertain B&W window range only. Fixed
    geometry still uses exactly 1x; all unspecified callers keep their range.
    A swatch-keyed search mapping plus ``lock_aspect=True`` lets a caller
    supply one common scale per symbol without rescaling a template twice.
    """
    if backend not in {'cpu', 'cuda'}:
        raise ValueError('window backend must be cpu or cuda')
    if not isinstance(gpu_batch_size, int) or gpu_batch_size <= 0:
        raise ValueError('gpu_batch_size must be a positive integer')
    if colour_observation is not None:
        colour_observation.validate(image)
        if backend!='cpu':raise ValueError('Source colour roles require CPU window verification')
    if uncertainty_mask is not None:
        uncertainty_mask = np.asarray(uncertainty_mask, np.float32)
        if (uncertainty_mask.shape != image.shape[:2] or not np.isfinite(uncertainty_mask).all()
                or np.any((uncertainty_mask < 0) | (uncertainty_mask > 1))):
            raise ValueError('Uncertainty mask must align with the image and lie in [0,1]')
        if not uncertainty_mask.any():
            uncertainty_mask = None
        elif backend != 'cpu':
            raise ValueError('Colour uncertainty requires CPU window verification')
    if occlusion_mask is not None:
        occlusion_mask = np.asarray(occlusion_mask, np.float32)
        if (occlusion_mask.shape != image.shape[:2] or not np.isfinite(occlusion_mask).all()
                or np.any((occlusion_mask < 0) | (occlusion_mask > 1))):
            raise ValueError('Occlusion mask must align with the image and lie in [0,1]')
        if not occlusion_mask.any():
            occlusion_mask = None
        elif backend != 'cpu':
            raise ValueError('Explicit colour occlusion currently requires CPU window verification')
    if (getattr(template, "matching_profile", "legacy") == "bw_v46_uncertain"
            and template.ink.achromatic):
        return _verify_uncertain_marker_window(image, template, x, y,
            fixed_geometry=fixed_geometry, backend=backend, gpu_batch_size=gpu_batch_size,
            occlusion_mask=occlusion_mask, uncertainty_mask=uncertainty_mask,colour_observation=colour_observation,
            **_search_options(search_scales,lock_aspect))
    if search_scales is not None or lock_aspect:
        raise ValueError('search_scales/lock_aspect requires the uncertain B&W profile')
    if occlusion_mask is not None or uncertainty_mask is not None or colour_observation is not None:
        raise ValueError('Explicit colour occlusion requires the uncertain BW profile')
    if backend != 'cpu':
        raise ValueError('CUDA window verification supports only the uncertain B&W profile')
    if fixed_geometry:
        raise ValueError("fixed_geometry requires the bw_v46_uncertain achromatic profile")
    diameter = float(template.diameter)
    window_size = max(17, int(math.ceil(2.2 * diameter)))
    if window_size % 2 == 0:
        window_size += 1
    plot_window = _crop_padded(image, (x, y), window_size, fill=255)
    membership = ink_membership(plot_window, template.ink)
    threshold = 0.45 if template.ink.achromatic else 0.24
    plot_ink = membership >= threshold

    maximum_shift = max(2, int(round(0.10 * diameter)))
    scales = np.linspace(0.88, 1.12, 7)
    best = None
    for scale in scales:
        for dy in range(-maximum_shift, maximum_shift + 1):
            for dx in range(-maximum_shift, maximum_shift + 1):
                rendered = _render_template(template.mask, window_size, float(scale), dx, dy)
                metrics = _evaluate_alignment(plot_ink, rendered, diameter * float(scale))
                alignment_penalty = 0.004 * math.hypot(dx, dy) + 0.05 * abs(float(scale) - 1.0)
                rank_score = metrics["score"] - alignment_penalty
                if best is None or rank_score > best[0]:
                    best = (rank_score, float(scale), dx, dy, rendered, metrics)

    assert best is not None
    _, scale, dx, dy, rendered, metrics = best
    # Small raster markers have a crisp, repeatable footprint, so a missing
    # required pixel is strong evidence against the candidate.  Larger marker
    # outlines vary more through rasterisation and receive a modestly looser
    # limit.  Extra ink is deliberately not a hard rejection criterion.
    required_verified = 0.85 if diameter < 20 else 0.72
    required_ambiguous = 0.72 if diameter < 20 else 0.59
    if metrics["required_recall"] >= required_verified:
        decision = "verified"
    elif metrics["required_recall"] >= required_ambiguous:
        decision = "ambiguous"
    else:
        decision = "rejected"

    return WindowVerification(
        x=float(x),
        y=float(y),
        aligned_x=float(x + dx),
        aligned_y=float(y + dy),
        scale=scale,
        decision=decision,
        plot_window=plot_window,
        plot_ink=plot_ink,
        template_mask=rendered,
        **metrics,
    )


def _uncertain_shape(template, scale, aspect):
    """Resample observed ink and its confidence together, without adding a hull.

    Only the one-pixel raster boundary may borrow adjacent evidence. Pixels
    inside a confidently inked stroke must match at their exact positions.
    """
    height, width = template.mask.shape
    ratio = math.sqrt(aspect)
    size = (max(3, round(width * scale * ratio)),
            max(3, round(height * scale / ratio)))
    mask = cv2.resize(template.mask.astype(np.uint8), size,
                      interpolation=cv2.INTER_NEAREST) > 0
    supplied_weight = getattr(template, "required_weight", None)
    if supplied_weight is None:
        supplied_weight = template.mask.astype(np.float32)
    if supplied_weight.shape != template.mask.shape:
        raise ValueError("required_weight must be aligned with the marker mask")
    weight = cv2.resize(np.asarray(supplied_weight, np.float32), size,
                        interpolation=cv2.INTER_LINEAR)
    weight = np.where(mask, np.clip(weight, 0.05, 1.0), 0).astype(np.float32)
    source_soft = template.raw_soft
    if source_soft.shape != template.mask.shape:
        source_soft = template.soft
    expected = cv2.resize(np.asarray(source_soft, np.float32), size,
                          interpolation=cv2.INTER_LINEAR)
    # Very pale fringe pixels should not make a nearly white patch perfect.
    expected = np.where(mask, np.clip(expected, 0.25, 1.0), 1.0)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    core = eroded & (weight >= 0.85)
    # Thin outlines often have no 3x3-eroded interior at all. Their genuinely
    # dark stroke centres still require direct ink, even when those centres
    # also touch a raster boundary. Otherwise a pale error bar one pixel away
    # can satisfy almost the whole glyph. Low-confidence legend-line pixels
    # are deliberately not promoted to these strict anchors.
    core |= mask & (expected >= 0.95) & (weight >= 0.30)
    boundary = mask & ~core
    return mask, weight, expected, core, boundary


def _search_options(search_scales=None, lock_aspect=False):
    options = {'search_scales':search_scales} if search_scales is not None else {}
    if lock_aspect:
        options['lock_aspect']=True
    return options


def _uncertain_template_shapes(template, use_cache=True, fixed_geometry=False,
                               search_scales=None, lock_aspect=False):
    """Reuse raster variants for all candidate centres of a live template.

    A content fingerprint catches in-place user/test adjustments to confidence
    arrays. Keeping the cache on the template bounds its lifetime naturally;
    there is no global image/template cache retaining completed GUI jobs.
    """
    # Explicit caller override; unspecified callers retain the historical range.
    # Both scalar CPU and batched CUDA windows share this raster preparation.
    if isinstance(search_scales, Mapping):
        if template.key not in search_scales:
            raise ValueError(f'Missing window scale for swatch {template.key}')
        search_scales=search_scales[template.key]
    scales = ((0.90, 0.94, 0.98, 1.0, 1.04, 1.08, 1.12)
              if search_scales is None else tuple(float(s) for s in search_scales))
    if not scales or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('search_scales must contain positive finite scales')
    scales = (1.0,) if fixed_geometry else tuple(dict.fromkeys(scales))
    digest = hashlib.blake2b(digest_size=16)
    for array in (template.mask, getattr(template, "required_weight", None),
                  template.raw_soft, template.soft):
        if array is None:
            digest.update(b"none")
            continue
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (id(_uncertain_shape), digest.digest(), bool(fixed_geometry), scales, bool(lock_aspect))
    cached = getattr(template, "_v46_window_shape_cache", None)
    if use_cache and cached is not None and cached[0] == key:
        return cached[1]
    shapes = []
    dimensions_seen = set()
    aspects = (1.0,) if fixed_geometry or lock_aspect else (1.0, 0.94, 1.06)
    for scale in scales:
        for aspect in aspects:
            arrays = _uncertain_shape(template, scale, aspect)
            if arrays[0].shape in dimensions_seen:
                continue
            dimensions_seen.add(arrays[0].shape)
            shapes.append((scale, aspect, *arrays))
    shapes = tuple(shapes)
    if use_cache:
        template._v46_window_shape_cache = (key, shapes)
    return shapes


def _uncertain_support(observed, nearby, expected, core, boundary):
    direct = np.minimum(observed / expected, 1.0)
    support = direct.copy()
    # A neighbouring pixel receives only partial credit, and only where the
    # legend boundary is uncertain. This is not dilation of all required ink.
    support[boundary] = np.maximum(
        direct[boundary],
        0.85 * np.minimum(nearby[boundary] / expected[boundary], 1.0),
    )
    return support


def _uncertain_scores(support, weight, core, boundary):
    weighted = float(np.sum(support * weight) / max(float(weight.sum()), 1e-6))
    core_recall = float(support[core].mean()) if core.any() else 1.0
    boundary_recall = (float(np.sum(support[boundary] * weight[boundary]) /
                             max(float(weight[boundary].sum()), 1e-6))
                       if boundary.any() else 1.0)
    score = 0.85 * weighted + 0.10 * core_recall + 0.05 * boundary_recall
    return weighted, core_recall, boundary_recall, score


def _uncertain_sector_scores(support, weight, eligibility_weight=None, minimum_sector_mass=1.5):
    """Keep every contour sector in the loss, even across added line ink."""
    yy, xx = np.indices(weight.shape)
    angles = np.arctan2(yy - (weight.shape[0] - 1) / 2,
                        xx - (weight.shape[1] - 1) / 2)
    sectors = np.floor((angles + np.pi) * 4 / np.pi).astype(int) % 8
    recalls = []
    eligibility = weight if eligibility_weight is None else eligibility_weight
    for sector in range(8):
        selected = (sectors == sector) & (weight > 0)
        total = float(weight[selected].sum())
        # Discounted uncertain ink must not make an entire missing sector
        # disappear below the minimum raster-mass threshold.
        if float(eligibility[selected].sum()) >= minimum_sector_mass:
            recalls.append(float(np.sum(support[selected] * weight[selected]) / max(total,1.e-6)))
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(np.asarray(recalls) >= 0.5)), float(min(recalls))


def _place_array(array, window_size, dx, dy, fill=0):
    output = np.full((window_size, window_size), fill, dtype=array.dtype)
    left = (window_size - array.shape[1]) // 2 + dx
    top = (window_size - array.shape[0]) // 2 + dy
    output[top:top + array.shape[0], left:left + array.shape[1]] = array
    return output


def _positive_contour_support(plot_ink, rendered):
    """Reward coincident boundaries without charging for unrelated extra ink."""
    expected_edge = cv2.Canny(rendered.astype(np.uint8) * 255, 40, 100) > 0
    observed_edge = cv2.Canny(plot_ink.astype(np.uint8) * 255, 40, 100) > 0
    # One raster pixel of contour uncertainty; this is a ranking reward only.
    near_edge = cv2.dilate(observed_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float(near_edge[expected_edge].mean()) if expected_edge.any() else 0.0


def _uncertain_cpu_shortlist(membership, nearby, shapes, window_size, maximum_shift, occlusion=None,
                             uncertainty=None, circle_rim=False):
    """Original NumPy search, stable loop order and top-eight tie behavior."""
    shortlist = []
    for scale, aspect, mask, weight, expected, core, boundary in shapes:
        height, width = mask.shape
        left, top = (window_size - width) // 2, (window_size - height) // 2
        if min(left, top) < maximum_shift:
            continue
        for dy in range(-maximum_shift, maximum_shift + 1):
            for dx in range(-maximum_shift, maximum_shift + 1):
                ys = slice(top + dy, top + dy + height)
                xs = slice(left + dx, left + dx + width)
                support = _uncertain_support(membership[ys, xs], nearby[ys, xs],
                                             expected, core, boundary)
                use_weight, use_core, use_boundary = weight, core, boundary
                if occlusion is not None:
                    visible = 1. - occlusion[ys, xs]
                    use_weight = weight * visible
                    use_core = core & (visible >= .5)
                    use_boundary = boundary & (visible >= .5)
                    # Other-colour ink only removes constraints. It must never
                    # provide the independently observed own-colour evidence.
                    total = max(float(weight.sum()), 1.e-6)
                    if (float(use_weight.sum()) / total < .20 or
                        float((np.minimum(membership[ys,xs]/expected,1.) * use_weight).sum()) / total < .20):
                        continue
                weighted, core_recall, boundary_recall, score = _uncertain_scores(
                    support, use_weight, use_core, use_boundary)
                if uncertainty is not None:
                    # Do not fill missing pixels or mark ambiguous ink as an
                    # occluder. Keep at least 35% of every uncertain penalty.
                    # Confident white gaps retain their FULL loss weight.
                    evidence_weight = use_weight
                    loss_reliability = 1. - .65*uncertainty[ys,xs]
                    total = max(float(weight.sum()), 1.e-6)
                    direct = np.minimum(membership[ys,xs]/expected, 1.)
                    if float((direct*evidence_weight).sum())/total < .35:
                        continue
                    use_weight = evidence_weight*loss_reliability
                    weighted, _, boundary_recall, _ = _uncertain_scores(
                        support, use_weight, use_core, use_boundary)
                    cw = loss_reliability[use_core]
                    core_recall = (float(np.sum(support[use_core]*cw)/max(float(cw.sum()),1.e-6))
                                   if use_core.any() else 1.)
                    score = .85*weighted + .10*core_recall + .05*boundary_recall
                penalty = (0.004 * math.hypot(dx, dy) + 0.018 * abs(scale - 1.0)
                           + 0.01 * abs(aspect - 1.0))
                # Missing strict core is never repaired by boundary credit.
                rank = score - penalty - 0.30 * max(0.85 - core_recall, 0)
                if circle_rim:
                    from bw_circle_rim_v46 import rim_evidence
                    rim=rim_evidence(membership[ys,xs],mask,weight,expected,core,
                        None if occlusion is None else occlusion[ys,xs])
                    # With no dark stroke core, direct rim ink replaces the
                    # vacuous 1.0 only for this opt-in circle profile. The
                    # diagnostic explicitly marks it as NOT a measured core.
                    if not use_core.any():core_recall=rim['direct_rim_recall']
                    score=.85*weighted+.10*core_recall+.05*boundary_recall-rim['penalty']
                    rank=score-penalty-.30*max(.85-core_recall,0)
                item = (rank, scale, aspect, dx, dy, mask, use_weight,
                        support, use_core, use_boundary, weighted, core_recall, boundary_recall, score)
                if uncertainty is not None:
                    item += (evidence_weight,)
                if len(shortlist) < 8 or rank > shortlist[-1][0]:
                    shortlist.append(item)
                    shortlist.sort(key=lambda value: value[0], reverse=True)
                    del shortlist[8:]
    return shortlist


def _prepare_uncertain_marker_window(image, template, x, y, fixed_geometry=False,
                                     search_scales=None, lock_aspect=False,colour_observation=None):
    """Fit every requested raster and translation in the scalar/batch crop.

    A decomposed glyph's diameter may be smaller than its padded raster.
    Diameter alone must not silently discard larger scale/aspect hypotheses
    (or leave an empty shortlist). Preserve the historical context window
    when it already fits; enlarge only its canvas, not the search or scores.
    """
    diameter = float(template.diameter)
    maximum_shift = max(2, int(round(0.16 * diameter)))
    shapes = _uncertain_template_shapes(template, fixed_geometry=fixed_geometry,
        **_search_options(search_scales,lock_aspect))
    largest_raster = max(max(shape[2].shape) for shape in shapes)
    window_size = max(17, int(math.ceil(2.4 * diameter)),
                      largest_raster + 2 * maximum_shift + 1)
    if window_size % 2 == 0:
        window_size += 1
    plot_window = _crop_padded(image, (x, y), window_size, fill=255)
    membership = (ink_membership(plot_window, template.ink) if colour_observation is None else
                  _crop_padded(colour_observation.membership(template.ink),(x,y),window_size))
    plot_ink = membership >= 0.45
    nearby = cv2.dilate(membership, np.ones((3, 3), np.uint8))
    return (diameter, maximum_shift, window_size, plot_window, membership,
            plot_ink, nearby, shapes)


def _verify_uncertain_marker_window(image, template, x, y, fixed_geometry=False,
                                    backend='cpu', gpu_batch_size=512, occlusion_mask=None,
                                    uncertainty_mask=None, search_scales=None, lock_aspect=False,colour_observation=None):
    """B&W asymmetric matching with optional certified CUDA rank screening.

    All final scores, top-eight ties, missing-ink sectors and decisions retain
    the original NumPy operations. GPU screens ranks, not acceptance rules.
    """
    prepared = _prepare_uncertain_marker_window(image, template, x, y, fixed_geometry,
        colour_observation=colour_observation,**_search_options(search_scales,lock_aspect))
    _, maximum_shift, window_size, _, membership, _, nearby, shapes = prepared
    compute_stats = {'backend': backend, 'used_cuda': False}
    occlusion = None if occlusion_mask is None else _crop_padded(occlusion_mask,(x,y),window_size)
    if occlusion is not None and not occlusion.any():
        occlusion = None
    uncertainty = None if uncertainty_mask is None else _crop_padded(uncertainty_mask,(x,y),window_size)
    if uncertainty is not None and not uncertainty.any():
        uncertainty = None
    if backend == 'cuda':
        from bw_gpu_window_verifier import certified_shortlist
        shortlist, compute_stats = certified_shortlist(membership, nearby, shapes,
            window_size, maximum_shift, batch_size=gpu_batch_size)
    else:
        shortlist = _uncertain_cpu_shortlist(membership, nearby, shapes,
                                             window_size, maximum_shift, occlusion=occlusion,
                                             uncertainty=uncertainty)
    no_visible_support = (occlusion is not None or uncertainty is not None) and not shortlist
    if no_visible_support:
        # Return the normal diagnostic geometry, explicitly rejected; never
        # invent support or raise merely because the whole window is hidden.
        shortlist = _uncertain_cpu_shortlist(membership,nearby,shapes,window_size,maximum_shift)
    result = _finalize_uncertain_marker_window(template, x, y,
                                              fixed_geometry or (lock_aspect and len(shapes)==1),
                                              prepared, shortlist, compute_stats, occlusion)
    if occlusion is not None:
        mask=result.template_mask
        result.compute_diagnostics.update(explicit_colour_occlusion=True,
            independently_visible_ink_required=True, no_visible_support=no_visible_support)
        result.occluded_fraction=float(occlusion[mask].mean()) if mask.any() else 0.
        result.visible_fraction=1.-result.occluded_fraction
        result.required_missing &= occlusion < .5
        if no_visible_support:
            result.decision='rejected'
    if uncertainty is not None:
        mask = result.template_mask
        result.compute_diagnostics['colour_uncertainty'] = dict(
            mean_on_required=float(uncertainty[mask].mean()) if mask.any() else 0.,
            maximum_loss_discount=.65, minimum_independent_support=.35,
            no_visible_support=bool(no_visible_support), positive_ink_added=False)
        if no_visible_support:
            result.decision='rejected'
    if getattr(template,'model_completed',False) and template.marker_kind=='open':
        from open_marker_guard_v46 import apply_guard
        result=apply_guard(result,template,occlusion)
    return result


def verify_marker_windows_many(image, requests, *, backend='cuda', gpu_batch_size=512,
                               search_scales=None, lock_aspect=False):
    """Verify ordered ``(template, x, y, fixed_geometry)`` requests together.

    Independent B&W candidate windows share CUDA launches and geometry data.
    Crop preparation and final top-eight decisions remain the exact scalar
    implementation; extra ink and missing required ink retain their original
    asymmetric treatment. CPU mode deliberately uses the original scalar API.
    """
    if backend not in {'cpu', 'cuda'}:
        raise ValueError('window backend must be cpu or cuda')
    if not isinstance(gpu_batch_size, int) or isinstance(gpu_batch_size, bool) or gpu_batch_size <= 0:
        raise ValueError('gpu_batch_size must be a positive integer')
    requests = list(requests)
    if search_scales is not None:
        search_scales=({k:tuple(v) for k,v in search_scales.items()} if isinstance(search_scales,Mapping)
                       else tuple(search_scales))
    search_options = _search_options(search_scales,lock_aspect)
    if backend == 'cpu':
        return [verify_marker_window(image, template, x, y, fixed_geometry=fixed,
                    backend='cpu', gpu_batch_size=gpu_batch_size, **search_options)
                for template, x, y, fixed in requests]
    # Reject unsupported profiles before any CUDA work, exactly as the scalar
    # public API does, instead of silently changing color/legacy semantics.
    for template, _, _, _ in requests:
        if (getattr(template, 'matching_profile', 'legacy') != 'bw_v46_uncertain'
                or not template.ink.achromatic):
            raise ValueError('CUDA window verification supports only the uncertain B&W profile')
    if not requests:
        return []
    prepared, jobs, prepare_times = [], [], []
    for template, x, y, fixed in requests:
        started = time.perf_counter()
        item = _prepare_uncertain_marker_window(image, template, x, y, fixed, **search_options)
        prepared.append(item)
        _, shift, size, _, membership, _, nearby, shapes = item
        jobs.append((membership, nearby, shapes, size, shift))
        prepare_times.append(time.perf_counter() - started)
    from bw_gpu_window_verifier import certified_shortlists_many
    shortlists = certified_shortlists_many(jobs, batch_size=gpu_batch_size)
    results = []
    for request, item, (shortlist, stats), prepare_seconds in zip(requests, prepared, shortlists, prepare_times):
        template, x, y, fixed = request
        started = time.perf_counter()
        result = _finalize_uncertain_marker_window(template, x, y,
            fixed or (lock_aspect and len(item[-1])==1), item, shortlist, stats)
        if getattr(template, 'model_completed', False) and template.marker_kind == 'open':
            from open_marker_guard_v46 import apply_guard
            result = apply_guard(result, template)
        stats['prepare_seconds'] = prepare_seconds
        stats['finalize_seconds'] = time.perf_counter() - started
        results.append(result)
    return results


def _finalize_uncertain_marker_window(template, x, y, fixed_geometry,
                                      prepared, shortlist, compute_stats, occlusion=None):
    """Unchanged CPU sectors, acceptance thresholds, and visualization arrays."""
    diameter, _, window_size, plot_window, _, plot_ink, _, _ = prepared
    from bw_circle_boundary_v46 import eligible as circle_eligible
    circular=circle_eligible(template)
    from bw_cross_evidence_v46 import eligible as cross_eligible, evidence as cross_evidence, model_for as cross_model
    crossed=cross_eligible(template)
    if circular:
        # Re-rank EVERY whole-body translation with the physical loss; a
        # top-eight list ranked by old neighbour credit is not sufficient.
        # Shared finalization keeps CPU/CUDA decisions identical.
        _,shift,_,_,membership,_,nearby,shapes=prepared
        rescored=_uncertain_cpu_shortlist(membership,nearby,shapes,window_size,shift,
            occlusion=occlusion,circle_rim=True)
        if rescored:shortlist=rescored
        compute_stats.update(circle_rim_rescreen='all_translations_cpu',circle_rim_version='circle-direct-rim-v2')

    if not shortlist:
        raise ValueError("Marker template does not fit the full verification window")
    # Shape coverage distinguishes a little matching fragment from a complete
    # outline. Missing sectors are measured, never labelled away as occlusion.
    evaluated = []
    # Enlarging a proposal must not silently lower the acceptance threshold.
    # Raster certainty belongs to the source legend, not its search variant.
    source_diameter = diameter / max(float(getattr(template, "proposal_scale", 1.0)), 1e-6)
    required_verified = 0.85 if source_diameter < 20 else 0.72
    required_ambiguous = 0.72 if source_diameter < 20 else 0.59
    for item in shortlist:
        shape_support, minimum_sector = _uncertain_sector_scores(item[7], item[6],
            eligibility_weight=item[14] if len(item)>14 else None,
            minimum_sector_mass=(1.e-6 if getattr(template,'small_hollow_model',None) or crossed else 1.5))
        core_ok = item[11] >= 0.85
        if circular and not core_ok:
            from bw_circle_rim_v46 import rim_evidence, unmeasurable_core_supported
            _,ow,expected,ocore,_=_uncertain_shape(template,item[1],item[2])
            top=(window_size-item[5].shape[0])//2+item[4]
            left=(window_size-item[5].shape[1])//2+item[3]
            ys=slice(top,top+item[5].shape[0]);xs=slice(left,left+item[5].shape[1])
            rim=rim_evidence(prepared[4][ys,xs],item[5],ow,expected,ocore,
                             None if occlusion is None else occlusion[ys,xs])
            core_ok=unmeasurable_core_supported(rim)
        # A high average cannot hide a mostly absent side/corner. Each sector
        # must retain at least half its required ink, including crossed sectors.
        geometry_ok = minimum_sector >= 0.50
        decision = ("verified" if item[10] >= required_verified and core_ok and geometry_ok
                    and (not circular or item[13]>=required_verified)
                    else "ambiguous" if item[10] >= required_ambiguous and item[11] >= 0.72
                    else "rejected")
        if crossed:
            # A four-arm stroke has few pixels in angular bins, not missing
            # sides. Measure every nonempty bin AND the source ridges before
            # choosing a centre. This guard applies without GeometryFirst too.
            arm=cross_evidence(prepared[4],_place_array(item[5],window_size,item[3],item[4]),
                               template.name,occlusion=occlusion,model=cross_model(template),
                               scale=item[1]*getattr(template,'proposal_scale',1.))
            if arm['decision']=='conflict':decision='rejected'
            elif arm['decision']=='abstain' and decision=='verified':decision='ambiguous'
        if circular and getattr(template,'model_completed',False) and diameter*item[1]<=10.:
            # A crossing bar can pull the ink-only optimum off the white hole.
            # Evaluate source hole/rim consistency BEFORE selecting the center,
            # not only after an irreversible center choice. No new translations
            # or thresholds: the same bounded shortlist and final guard apply.
            from types import SimpleNamespace
            from open_marker_guard_v46 import apply_guard
            from bw_circle_rim_v46 import rim_evidence
            _,original_weight,expected,original_core,_=_uncertain_shape(template,item[1],item[2])
            top=(window_size-item[5].shape[0])//2+item[4]
            left=(window_size-item[5].shape[1])//2+item[3]
            ys=slice(top,top+item[5].shape[0]);xs=slice(left,left+item[5].shape[1])
            rim=rim_evidence(prepared[4][ys,xs],item[5],original_weight,expected,original_core,
                             None if occlusion is None else occlusion[ys,xs])
            trial=SimpleNamespace(plot_window=plot_window,
                template_mask=_place_array(item[5],window_size,item[3],item[4]),
                scale=item[1],decision=decision,compute_diagnostics={'circle_rim':rim})
            decision=apply_guard(trial,template,occlusion).decision
            compute_stats['small_circle_center_policy']='source_hole_and_rim_before_center_selection'
        evaluated.append((decision == "verified", item[0], item, decision,
                          shape_support, minimum_sector))
    _, _, best, decision, shape_support, minimum_sector = max(evaluated, key=lambda value: value[:2])
    (_, scale, aspect, dx, dy, mask, weight, support, core, boundary,
     weighted, core_recall, boundary_recall, score) = best[:14]
    rendered = _place_array(mask, window_size, dx, dy)
    # Keep antialias-tolerant detection recall separate from directly observed
    # evidence. The legacy numeric core sentinel is used by ranking/calibration;
    # it must NOT be exported as a measured perfect core or a confidence prior.
    _, _, expected, _, _ = _uncertain_shape(template, scale, aspect)
    top=(window_size-mask.shape[0])//2+dy
    left=(window_size-mask.shape[1])//2+dx
    observed=prepared[4][top:top+mask.shape[0],left:left+mask.shape[1]]
    direct=np.minimum(observed/expected,1.)
    direct_recall=float(np.sum(direct*weight)/max(float(weight.sum()),1.e-6))
    compute_stats['ink_evidence'] = dict(
        version='direct_and_tolerant_ink_v1', direct_required_recall=direct_recall,
        tolerant_required_recall=float(weighted),
        borrowed_boundary_credit=max(0.,float(weighted)-direct_recall),
        strict_core_pixels=int(core.sum()),
        measured_strict_core_recall=float(core_recall) if core.any() else None,
        strict_core_status='measured' if core.any() else 'unmeasurable',
        confidence_kind='direct_required_ink_not_probability')
    if crossed:
        compute_stats['stroke_cross']=cross_evidence(prepared[4],rendered,template.name,occlusion=occlusion,
            model=cross_model(template),scale=scale*getattr(template,'proposal_scale',1.))
    metrics = _evaluate_alignment(plot_ink, rendered, diameter * scale)
    metrics.update(score=float(score), required_recall=weighted,
                   missing_fraction=1.0 - weighted)
    # The visualization now marks genuinely unsupported required pixels;
    # antialiasing/boundary-neighbour credit is reflected consistently.
    required_missing = mask & (support < 0.5)
    metrics["required_missing"] = _place_array(required_missing, window_size, dx, dy)
    if circular:
        from bw_circle_rim_v46 import rim_evidence
        _,original_weight,expected,original_core,_=_uncertain_shape(template,scale,aspect)
        top=(window_size-mask.shape[0])//2+dy;left=(window_size-mask.shape[1])//2+dx
        ys=slice(top,top+mask.shape[0]);xs=slice(left,left+mask.shape[1])
        compute_stats['circle_rim']=rim_evidence(prepared[4][ys,xs],mask,original_weight,
            expected,original_core,None if occlusion is None else occlusion[ys,xs])
        compute_stats['circle_rim']['penalized_score']=float(score)
        from bw_circle_rim_v46 import unmeasurable_core_supported
        compute_stats['circle_rim']['unmeasurable_core_supported']=unmeasurable_core_supported(compute_stats['circle_rim'])
    return WindowVerification(
        x=float(x), y=float(y), aligned_x=float(x + dx), aligned_y=float(y + dy),
        scale=float(scale), decision=decision, plot_window=plot_window,
        plot_ink=plot_ink, template_mask=rendered,
        weighted_required_recall=weighted, strict_core_recall=core_recall,
        boundary_recall=boundary_recall, shape_support=shape_support,
        minimum_sector_recall=minimum_sector,
        matching_profile="bw_v46_uncertain", aspect_ratio=float(aspect),
        geometry_fixed=bool(fixed_geometry),
        compute_diagnostics=compute_stats,
        contour_support=_positive_contour_support(plot_ink, rendered), **metrics,
    )

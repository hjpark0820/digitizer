"""Source-evidence gate for frozen method-6 paths; no tracing or correction.

All masks are plot-local, whereas ``raw_path_source`` and ``samples`` retain
source-image coordinates. A surviving estimated bridge is explicitly *not* an
observation. The original solver geometry, color samples, and cost are retained
for diagnostics even where the returned reference/rendering permission is false.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PathSupportGateConfig:
    color_support_threshold: float = .18
    grid_threshold: float = .5
    source_ink_min_contrast: float = 8. / 255.
    neighborhood_width_fraction: float = .375
    neighborhood_max_radius_px: float = 2.
    raster_gap_max_px: float = 2.
    occlusion_max_widths: float = 12.
    occlusion_min_other_coverage: float = .6
    stable_arm_widths: float = 1.
    stable_arm_min_px: float = 3.


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _field(value, name, shape):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite plot-local array of shape {shape}")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def _column_sample(field, x, y):
    lower = int(np.floor(y))
    upper = min(lower + 1, field.shape[0] - 1)
    fraction = y - lower
    return float((1 - fraction) * field[lower, x] + fraction * field[upper, x])


def _span(xs, start, stop, step):
    """Pixel-column extent of a half-open run of native path samples."""
    return float(xs[stop - 1] - xs[start] + step) if stop > start else 0.


def gate_path(record, image_bgr, plot_box, *, own, other_ink, grid,
              text_mask=None, line_width=4., config=None):
    """Return a deep-copied record with source-justified reference samples.

    ``runs`` contains lists of retained raw sample indexes, not coordinate pairs
    or half-open bounds. ``raw_observed`` is newly vetted actual own-color ink;
    ``support_kind`` distinguishes that ink from permitted estimated bridges.
    ``reference_allowed`` is the sole gate for downstream reference admission.

    Own-color observations need the original soft path value >= .18 and actual
    nonwhite own-mask ink in the same native x column, within a radius bounded
    by two pixels. Text/grid samples cannot become observations or bridges.
    Raster holes need own support immediately on both sides and at most two
    missing columns. Longer bridges require stable own-color arms, competing
    source ink, a linewidth-scaled length bound, and no long uncovered stretch.
    """
    cfg = config or PathSupportGateConfig()
    if isinstance(cfg, dict):
        cfg = PathSupportGateConfig(**cfg)
    if not isinstance(cfg, PathSupportGateConfig):
        raise TypeError("config must be PathSupportGateConfig, a dict, or None")
    parameters = asdict(cfg)
    if any(not np.isfinite(value) or value < 0 for value in parameters.values()):
        raise ValueError("gate configuration values must be finite and nonnegative")
    for name in ("color_support_threshold", "grid_threshold", "source_ink_min_contrast",
                 "occlusion_min_other_coverage"):
        if not 0 < parameters[name] <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if not np.isfinite(line_width) or line_width <= 0:
        raise ValueError("line_width must be positive and finite")
    box = np.asarray(plot_box, float)
    image = np.asarray(image_bgr)
    if (box.shape != (4,) or not np.isfinite(box).all() or
            np.any(box != np.rint(box))):
        raise ValueError("plot_box must contain four finite integer source coordinates")
    x0, y0, x1, y1 = box.astype(int)
    if (image.ndim != 3 or image.shape[2] != 3 or x0 < 0 or y0 < 0 or
            x1 <= x0 or y1 <= y0 or x1 > image.shape[1] or y1 > image.shape[0]):
        raise ValueError("image_bgr must be a full source image containing plot_box")
    crop = image[y0:y1, x0:x1].astype(float)
    if not np.isfinite(crop).all() or np.any((crop < 0) | (crop > 255)):
        raise ValueError("image_bgr must have finite channel values in [0, 255]")
    shape = (y1 - y0, x1 - x0)
    own_mask = _field(own, "own", shape) >= .5
    other_mask = _field(other_ink, "other_ink", shape) >= .5
    grid_map = _field(grid, "grid", shape)
    text = (np.zeros(shape, bool) if text_mask is None else
            _field(text_mask, "text_mask", shape) >= .5)
    # White pixels cannot become source evidence just because a mask is stale.
    source_ink = np.max(255. - crop, axis=2) / 255. >= cfg.source_ink_min_contrast
    clean = source_ink & ~text & (grid_map < cfg.grid_threshold)
    clean_own = own_mask & clean
    clean_other = other_mask & ~own_mask & clean

    xy = np.asarray(record["raw_path_source"], float)
    if xy.size == 0:
        xy = xy.reshape(0, 2)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("raw_path_source must be a finite N x 2 array")
    n = len(xy)
    if n > 1 and np.any(np.diff(xy[:, 0]) <= 0):
        raise ValueError("native column paths must have strictly increasing x coordinates")
    values = np.asarray(record["path_color_values"], float)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("path_color_values must be the original N finite soft samples in [0, 1]")
    original_valid = np.asarray(record.get("valid", np.ones(n, bool)), bool)
    if original_valid.shape != (n,):
        raise ValueError("record.valid must have one flag per raw path sample")
    original_observed = np.asarray(record.get("raw_observed", values >= cfg.color_support_threshold), bool)
    if original_observed.shape != (n,):
        raise ValueError("record.raw_observed must have one flag per raw path sample")
    if "samples" in record and len(record["samples"]) != n:
        raise ValueError("record.samples must have one entry per raw path sample")

    local = xy - [x0, y0]
    radius = min(cfg.neighborhood_max_radius_px,
                 max(.75, cfg.neighborhood_width_fraction * line_width))
    in_bounds = ((local[:, 0] >= 0) & (local[:, 0] < shape[1]) &
                 (local[:, 1] >= 0) & (local[:, 1] <= shape[0] - 1))
    own_near = np.zeros(n, bool)
    other_near = np.zeros(n, bool)
    text_block = np.zeros(n, bool)
    grid_block = np.zeros(n, bool)
    for i in np.flatnonzero(in_bounds):
        x = min(int(np.rint(local[i, 0])), shape[1] - 1)
        y = local[i, 1]
        lo = max(0, int(np.ceil(y - radius)))
        hi = min(shape[0], int(np.floor(y + radius)) + 1)
        own_near[i] = bool(clean_own[lo:hi, x].any())
        other_near[i] = bool(clean_other[lo:hi, x].any())
        # Use a narrow footprint for text barriers, without dilating text into
        # unrelated source curves. Candidate evidence itself is also filtered.
        tlo = max(0, int(np.ceil(y - .75)))
        thi = min(shape[0], int(np.floor(y + .75)) + 1)
        text_block[i] = bool(text[tlo:thi, x].any())
        grid_block[i] = _column_sample(grid_map, x, y) >= cfg.grid_threshold

    permitted = in_bounds & original_valid & ~text_block & ~grid_block
    observed = permitted & own_near & (values >= cfg.color_support_threshold)
    allowed = observed.copy()
    kinds = np.full(n, "unsupported_blank", dtype=object)
    supported = np.flatnonzero(observed)
    if len(supported):
        kinds[:supported[0]] = "unsupported_tail"
        kinds[supported[-1] + 1:] = "unsupported_tail"
    kinds[~original_valid] = "invalid"
    kinds[~in_bounds] = "outside_plot"
    kinds[grid_block & in_bounds & original_valid] = "grid"
    kinds[text_block & in_bounds & original_valid] = "text"
    kinds[observed] = "observed"

    xs = xy[:, 0]
    step = float(np.median(np.diff(xs))) if n > 1 else 1.
    arm_required = max(cfg.stable_arm_min_px, cfg.stable_arm_widths * line_width)
    maximum_occlusion = cfg.occlusion_max_widths * line_width
    gap_audit = []
    for a, b in _runs(~observed):
        width = _span(xs, a, b, step)
        bilateral = a > 0 and b < n and observed[a - 1] and observed[b]
        reason = "no_bilateral_own_support"
        left_start, right_stop = a - 1, b + 1
        if bilateral:
            while left_start > 0 and observed[left_start - 1]:
                left_start -= 1
            while right_stop < n and observed[right_stop]:
                right_stop += 1
        left_arm = _span(xs, left_start, a, step) if bilateral else 0.
        right_arm = _span(xs, b, right_stop, step) if bilateral else 0.
        coverage = float(np.mean(other_near[a:b]))
        uncovered = max((_span(xs, a + u, a + v, step)
                         for u, v in _runs(~other_near[a:b])), default=0.)
        # Native solver output is sampled every column. A sparse input must not
        # bypass examination of missing source columns through its sample count.
        dense = bilateral and bool(np.all(np.diff(xs[a - 1:b + 1]) <= 1.01))
        if not np.all(permitted[a:b]):
            reason = "hard_exclusion_in_gap"
        elif bilateral and not dense:
            reason = "unsampled_source_columns"
        elif dense and width <= cfg.raster_gap_max_px + 1e-9:
            allowed[a:b] = True
            kinds[a:b] = "raster_gap"
            reason = "tiny_bilateral_raster_hole"
        elif bilateral:
            if min(left_arm, right_arm) < arm_required:
                reason = "unstable_or_short_own_arms"
            elif width > maximum_occlusion:
                reason = "occlusion_length_limit"
            elif coverage < cfg.occlusion_min_other_coverage:
                reason = "insufficient_other_source_ink"
            elif uncovered > cfg.raster_gap_max_px + 1e-9:
                reason = "uncovered_blank_span_in_occlusion"
            else:
                allowed[a:b] = True
                kinds[a:b] = "other_ink_occlusion"
                reason = "bounded_other_ink_with_stable_bilateral_own_arms"
        gap_audit.append(dict(start_index=a, stop_index=b, columns=b-a, span_px=width,
            bilateral_own_support=bool(bilateral), left_arm_px=left_arm, right_arm_px=right_arm,
            other_source_ink_coverage=coverage, maximum_uncovered_span_px=uncovered,
            allowed=bool(allowed[a:b].all()), reason=reason))

    result = deepcopy(record)
    provenance = dict(raw_observed=deepcopy(record.get("raw_observed")),
                      valid=deepcopy(record.get("valid")),
                      sample_observed=[deepcopy(s.get("observed")) for s in record.get("samples", [])],
                      path_color_values_origin="unchanged original soft field sampled on frozen raw geometry")
    # Applying the gate again must not destroy an earlier gate's provenance.
    if "gate_provenance" in record:
        provenance["previous_gate_provenance"] = deepcopy(record["gate_provenance"])
    result["gate_provenance"] = provenance
    result["raw_observed"] = observed.tolist()
    result["reference_allowed"] = allowed.tolist()
    result["support_kind"] = kinds.tolist()
    result["runs"] = [list(range(a, b)) for a, b in _runs(allowed)]
    samples = result.get("samples")
    if samples is None:
        samples = [dict(x_px=float(x), y_px=float(y)) for x, y in xy]
    for i, sample in enumerate(samples):
        sample.update(observed=bool(observed[i]), reference_allowed=bool(allowed[i]),
                      support_kind=str(kinds[i]))
    result["samples"] = samples
    result["gate_audit"] = dict(
        policy="native_source_support_gate_v1", config=parameters, line_width_px=float(line_width),
        source_neighborhood_radius_px=float(radius), occlusion_max_span_px=float(maximum_occlusion),
        stable_arm_required_px=float(arm_required), sample_count=n,
        original_observed_count=int(original_observed.sum()), observed_count=int(observed.sum()),
        reference_allowed_count=int(allowed.sum()), rejected_count=int((~allowed).sum()),
        estimated_bridge_count=int((allowed & ~observed).sum()),
        counts_by_support_kind=dict(Counter(kinds.tolist())), gaps=gap_audit,
        original_geometry_unchanged=True, original_color_values_unchanged=True,
        original_masks_unchanged=True,
        interpretation="Retained bridges are inferred, not observed curve pixels or measurement centers")
    return result

"""Connector-local completion of accepted BW legend models.

The geometric fit is unchanged. Its model may replace measured pixels only in
the independently estimated connector band; everywhere else the original
aligned source observation remains exact, including weak/negative values.
``model_completed`` continues to mean that a geometry fit was accepted, not
that the matching raster contains an entirely regenerated shape. The returned
``local_completion`` audit states that distinction explicitly.

These stateless postprocessors return copies. They neither patch runtime
functions nor mutate the caller's source template, completed template or report.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib

import cv2
import numpy as np


VERSION = "local-connector-completion-v1"
LINE_ALPHA_THRESHOLD = .05
FOOTPRINT_ALPHA_THRESHOLD = .02
NUISANCE_VERTICAL_RADIUS = 1


def _fingerprint(array):
    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()[:20]


def _transport(original, completed):
    """Transport source pixel centres into the unchanged completed anchor."""
    oh, ow = original.raw_soft.shape
    h, w = completed.raw_soft.shape
    dx = completed.marker_center[0] - original.marker_center[0]
    dy = completed.marker_center[1] - original.marker_center[1]
    return np.float32([[1, 0, (w-ow)/2-dx], [0, 1, (h-oh)/2-dy]])


def _warp(array, matrix, shape, fill=0.):
    h, w = shape
    return cv2.warpAffine(np.asarray(array, np.float32), matrix, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=float(fill))


def _observed_from_original(original):
    measured = getattr(original, "observed_source_soft", None)
    if measured is not None:
        return np.asarray(measured, np.float32), "original.observed_source_soft"
    if getattr(original, "model_completed", False):
        return None, "missing_measured_source_on_completed_input"
    return np.asarray(original.raw_soft, np.float32), "original.raw_soft_before_completion"


def local_hybrid(observed_soft, model_soft, contamination_mask):
    """Return the mixed raster and exact provenance masks, without mutation."""
    observed = np.asarray(observed_soft, np.float32)
    model = np.asarray(model_soft, np.float32)
    contaminated = np.asarray(contamination_mask, bool)
    if observed.shape != model.shape or observed.shape != contaminated.shape:
        raise ValueError("Observation, model, and contamination shapes must agree")
    if not np.isfinite(observed).all() or not np.isfinite(model).all():
        raise ValueError("Raster values must be finite")
    footprint = model > FOOTPRINT_ALPHA_THRESHOLD
    replacement = contaminated & footprint
    tails = contaminated & ~footprint
    hybrid = observed.copy()
    hybrid[replacement] = model[replacement]
    hybrid[tails] = 0.
    return hybrid, dict(contamination_mask=contaminated, replacement_mask=replacement,
                        tail_removal_mask=tails, preserved_mask=~contaminated,
                        model_footprint_mask=footprint, changed_mask=hybrid != observed)


def apply_local_completion(completed, observed, source_gray, line_alpha, *, path,
                           source_description, alignment=None, contrast=None):
    """Rebuild matching fields from a source/model hybrid; return copy + audit.

    ``observed`` and ``source_gray`` are already in ``completed`` coordinates.
    The contrast convention for the legacy open-ring fitter is retained exactly.
    Source validity, accepted centre/diameter, identity and nuisance semantics do
    not change. Measured grayscale is never replaced by rendered grayscale.
    """
    import partial_swatch_detector as detector

    model = np.asarray(completed.raw_soft, np.float32)
    observed = np.asarray(observed, np.float32)
    line_alpha = np.asarray(line_alpha, np.float32)
    if line_alpha.shape != model.shape or not np.isfinite(line_alpha).all():
        raise ValueError("Connector alpha must be finite and match the model canvas")
    hybrid, masks = local_hybrid(observed, model, line_alpha > LINE_ALPHA_THRESHOLD)
    if source_gray is not None and np.shape(source_gray) != model.shape:
        raise ValueError("Measured source gray must match the model canvas")
    if contrast is None:
        coverage, cutoff = hybrid, .25
    else:
        coverage, cutoff = hybrid / max(float(contrast), 1e-6), .22
    mask = coverage >= cutoff
    result = deepcopy(completed)
    result.raw_soft = hybrid.copy()
    result.soft = hybrid.copy()
    result.raw_mask = mask.copy()
    result.mask = mask.copy()
    result.edge = cv2.Canny(mask.astype(np.uint8)*255, 40, 100) > 0
    result.orientation = detector._edge_orientation(hybrid)
    if contrast is None:
        depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        result.required_weight = np.where(mask, np.where(depth <= 1.05,
            .35+.40*hybrid, .85+.15*hybrid), 0).astype(np.float32)
    else:
        result.required_weight = np.where(mask, .5+.5*coverage, 0).astype(np.float32)
    result.observed_source_soft = observed.copy()
    if source_gray is not None:
        result.source_gray = np.asarray(source_gray, np.float32).copy()
    outside = masks["preserved_mask"]
    max_change = float(np.max(np.abs(hybrid[outside]-observed[outside]), initial=0.))
    audit = dict(version=VERSION, status="applied", path=path,
        swatch_id=str(completed.key), observation_source=source_description,
        alignment=deepcopy(alignment or {}),
        line_alpha_threshold=LINE_ALPHA_THRESHOLD,
        footprint_alpha_threshold=FOOTPRINT_ALPHA_THRESHOLD,
        nuisance_vertical_radius=NUISANCE_VERTICAL_RADIUS,
        model_fingerprint=_fingerprint(model), hybrid_fingerprint=_fingerprint(hybrid),
        observation_fingerprint=_fingerprint(observed), shape=list(model.shape),
        pixels={key: int(value.sum()) for key, value in masks.items()},
        mass=dict(observed=float(observed.sum()), model=float(model.sum()),
                  hybrid=float(hybrid.sum())),
        max_change_outside_contamination=max_change,
        raster_effect=not np.array_equal(model, hybrid),
        source_gray_available=source_gray is not None,
        source_contract="aligned measured source retained; no synthesized observation",
        matching_raster_policy="observed_source_with_connector_local_model_completion",
        accepted_geometry_unchanged=True, model_completed_means="accepted_geometry_fit",
        contamination_is_estimated=True)
    result.local_completion = deepcopy(audit)
    return result, audit


def _with_audit(detail, audit):
    result = deepcopy(detail)
    result["local_completion"] = deepcopy(audit)
    if audit.get("status") == "applied":
        result["matching_raster_policy"] = audit["matching_raster_policy"]
        result["expected_ink_source"] = "aligned source ink; fitted model only within connector band"
        result["observed_ink_source"] = "measured source_gray / observed_source_soft, never synthesized"
    return result


def postprocess_geometry(original, completed, detail, *, path="silhouette_template",
                         fitted_line=False):
    """Postprocess an accepted geometry result, returning ``(template, detail)``.

    Use paths ``silhouette_template``, ``separate_open_ring`` or
    ``complete_hollow_geometry``. The latter renders the already accepted line
    model; its inferred line blur, slope and dashed phase are transported too.
    Missing measured evidence abstains rather than treating a rendered model as
    an observation. The accepted legacy result remains available in that case.
    """
    observed, source = _observed_from_original(original)
    if observed is None:
        audit = dict(version=VERSION, status="skipped", path=path, reason=source,
                     source_contract="no synthetic raster relabelled as measured source")
        return deepcopy(completed), _with_audit(detail, audit)
    matrix = _transport(original, completed)
    shape = completed.raw_soft.shape
    observed = _warp(observed, matrix, shape)
    gray = getattr(original, "source_gray", None)
    if gray is not None:
        gray = _warp(gray, matrix, shape, original.ink.paper_gray)
    nuisance = np.asarray(original.line_nuisance, np.uint8)
    nuisance = cv2.dilate(nuisance, np.ones((2*NUISANCE_VERTICAL_RADIUS+1, 1), np.uint8))
    line = _warp(nuisance, matrix, shape)
    description = "transported original.line_nuisance, one-pixel vertical dilation"
    if fitted_line or path == "complete_hollow_geometry":
        from legend_composition_v46 import Config, render_model
        fit = original.line_crossed_hollow_model
        lp = dict(fit["line_params"])
        pars = fit["best_model"]["params"]
        sx, sy = (shape[1]-1)/2-pars["cx"], (shape[0]-1)/2-pars["cy"]
        for field in ("x0", "x1", "cx"):
            lp[field] = lp.get(field, 0.) + sx
        lp["cy"] += sy
        if lp.get("style") == "dashed":
            lp["dash_phase"] += sx
        allowed = {field.name for field in fields(Config)}
        cfg = Config(**{key: value for key, value in fit.get("config", {}).items() if key in allowed})
        line = render_model("line_only", {}, lp, shape, cfg)["line_alpha"]
        description = "accepted fitted line rendered after center/dash-phase transport; blur retained"
    result, audit = apply_local_completion(completed, observed, gray, line, path=path,
        source_description=source, alignment=dict(matrix=matrix.tolist(), connector=description),
        contrast=detail.get("contrast") if path == "separate_open_ring" else None)
    return result, _with_audit(detail, audit)


def postprocess_composed(original, completed, report, figure):
    """Postprocess one accepted ``compose_templates`` output and report.

    ``figure`` is its existing fit record with source ``observed_alpha`` and
    fitted ``line_alpha``. No plot pixels or reference labels enter this API.
    """
    if report.get("template_policy") != "pure_model_marker":
        return completed, report
    record, data = figure["record"], figure["fields"]
    pars = record["best_model"]["params"]
    shape = completed.raw_soft.shape
    matrix = np.float32([[1, 0, (shape[1]-1)/2-pars["cx"]],
                         [0, 1, (shape[0]-1)/2-pars["cy"]]])
    observed = _warp(data["observed_alpha"], matrix, shape)
    line = _warp(data["line_alpha"], matrix, shape)
    result, audit = apply_local_completion(completed, observed, completed.source_gray, line,
        path="compose_templates",
        source_description="fitter fields.observed_alpha from actual source crop",
        alignment=dict(matrix=matrix.tolist(), source_box=figure["box"],
            connector="fitter fields.line_alpha warped with observed_alpha",
            line_params=record.get("line_params")))
    return result, _with_audit(report, audit)

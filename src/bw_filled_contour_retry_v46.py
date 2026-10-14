"""Selective, evidence-reusing tie retry for observed filled-marker identity.

The original comparator remains authoritative. Accepted results are returned
unchanged; failures of absolute match, visibility, interior ink, or distinctive
angular strength are never rescued. Only close-ranking/geometry-instability
abstentions may use a source-only pair-informed radial average. All other raw
sources remain rivals on exactly the same pixels and trials.

No detector proposal, idealized glyph, source recentering, connector deletion,
or score-derived nuisance mask is introduced. Source-center sensitivity is a
diagnostic, not an automatic correction or an extra acceptance gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
import numpy as np

import bw_filled_contour_identity_v46 as baseline

VERSION = "filled-contour-selective-pair-retry-v46-v1"
ALLOWED_REASONS = frozenset(("small_or_reversing_lower_decile_margin",
                             "winner_changes_under_center_or_scale_perturbation"))


@dataclass(frozen=True)
class Config(baseline.Config):
    retry_enabled: bool = True
    radial_weight_mix: float = .5
    radial_information_clip: float = 2.


def _configuration(config):
    cfg = config or Config()
    core = baseline.Config(**{field.name: getattr(cfg, field.name) for field in fields(baseline.Config)})
    baseline._validate_config(core)
    mix = float(getattr(cfg, "radial_weight_mix", .5))
    clip = float(getattr(cfg, "radial_information_clip", 2.))
    if mix != .5 or clip != 2.:
        raise ValueError("selective retry fixes positive radial weights to .5 mix and 2 information clip")
    return cfg, core


def sample_polar(gray, center_xy, reference_radius, nuisance=None, config=None):
    _, core = _configuration(config)
    return baseline.sample_polar(gray, center_xy, reference_radius, nuisance, core)


def estimate_radius(gray, center_xy, reference_radius, nuisance=None, config=None):
    _, core = _configuration(config)
    return baseline.estimate_radius(gray, center_xy, reference_radius, nuisance, core)


def build_source(sid, gray, center_xy, reference_radius, nuisance=None, metadata=None, config=None):
    """Keep baseline source fields; preserve a small raw crop for diagnostics."""
    _, core = _configuration(config)
    source = baseline.build_source(sid, gray, center_xy, reference_radius, nuisance, metadata, core)
    a = np.asarray(gray)
    cx, cy = map(float, center_xy)
    reach = int(math.ceil(source["reference_radius_px"]*core.radius_max+2.))
    x0, x1 = max(0, int(math.floor(cx))-reach), min(a.shape[1], int(math.ceil(cx))+reach+1)
    y0, y1 = max(0, int(math.floor(cy))-reach), min(a.shape[0], int(math.ceil(cy))+reach+1)
    if x1-x0 >= 2 and y1-y0 >= 2:
        crop = a[y0:y1, x0:x1].copy()
        crop.setflags(write=False)
        mask = None if nuisance is None else np.asarray(nuisance)[y0:y1, x0:x1].copy()
        if mask is not None:
            mask.setflags(write=False)
        source["_retry_raw_observation"] = dict(gray=crop, nuisance=mask,
                                               center_xy=(cx-x0, cy-y0), origin_xy=(x0, y0))
    return source


def _fingerprint(models, core):
    digest = hashlib.sha256(repr(asdict(core)).encode("utf8"))
    for model in models:
        digest.update(str(model["sid"]).encode("utf8"))
        sig = model["signature"]
        for name in ("normalized_ink", "valid_weight", "sample_xy", "radii_normalized", "angles_rad", "informative_ring_mask"):
            value = np.ascontiguousarray(sig[name])
            digest.update(name.encode("ascii"))
            digest.update(str((value.shape, value.dtype)).encode("ascii"))
            digest.update(value.tobytes())
        raw = model.get("_retry_raw_observation")
        if raw is not None:
            digest.update(np.ascontiguousarray(raw["gray"]).tobytes())
            digest.update(repr(raw["center_xy"]).encode("ascii"))
    return digest.hexdigest()


def _source_diagnostic(source, core):
    signature = source["signature"]
    use = signature["informative_ring_mask"]
    h = signature["harmonics"]["coefficients"]
    e12 = float(np.mean(.5*np.sum(h[use, 1:5]**2, axis=1)))
    e34 = float(np.mean(.5*np.sum(h[use, 5:9]**2, axis=1)))
    result = dict(sid=source["sid"], low_order_C1_C2_energy=e12,
                  distinctive_C3_C4_energy=e34,
                  low_order_fraction=e12/max(e12+e34, 1e-12),
                  center_corrected=False, source_pixels_modified=False,
                  connector_pixels_removed=False, acceptance_uses_this_diagnostic=False,
                  interpretation="C1/C2 may reflect shape asymmetry, sampling, miscentering or connectors; it does not identify a connector")
    raw = source.get("_retry_raw_observation")
    if raw is None:
        result.update(status="raw_source_unavailable_no_center_sensitivity_estimate")
        return result
    gray = raw["gray"]
    center = np.asarray(raw["center_xy"], float)
    radius = float(source["reference_radius_px"])
    nominal = baseline.sample_polar(gray, center, radius, raw["nuisance"], core)
    offsets = ((-.5, 0.), (.5, 0.), (0., -.5), (0., .5))
    measured = []
    for offset in offsets:
        varied = baseline.sample_polar(gray, center+np.asarray(offset), radius, raw["nuisance"], core)
        common = np.minimum(nominal["valid_weight"], varied["valid_weight"])
        a, b = (baseline._features(s, common, core) for s in (nominal, varied))
        loss = baseline._loss(a, b, common, use, core)
        measured.append(dict(offset_px=list(offset), primary_loss=None if loss is None else loss["primary"]))
    valid = [item["primary_loss"] for item in measured if item["primary_loss"] is not None]
    result.update(status="source_only_fixed_center_sensitivity_measured", offsets=measured,
                  median_center_sensitivity_loss=float(np.median(valid)) if valid else None,
                  maximum_center_sensitivity_loss=max(valid) if valid else None)
    return result


def prepare_sources(sources, config=None):
    """Return a caller-owned reusable cache; no source or metadata is mutated."""
    _, core = _configuration(config)
    models = list(sources.values()) if isinstance(sources, dict) else list(sources)
    sids = [str(model["sid"]) for model in models]
    if len(set(sids)) != len(sids):
        raise ValueError("source identities must be distinct")
    if len(models) < 2 or len(models) > core.maximum_sources:
        raise ValueError("prepared source count must fit the unchanged comparison budget")
    signatures = [model["signature"] for model in models]
    common = np.minimum.reduce([signature["valid_weight"] for signature in signatures])
    features = [baseline._features(signature, common, core) for signature in signatures]
    coverage = [baseline._coverage(common, signature["sample_xy"], core) for signature in signatures]
    return dict(fingerprint=_fingerprint(models, core), source_ids=sids,
                source_weights=common, features=features, coverage=coverage,
                informative_mask=signatures[0]["informative_ring_mask"].copy(),
                radii_normalized=signatures[0]["radii_normalized"].copy(), pair_cache={},
                source_diagnostics=[_source_diagnostic(source, core) for source in models])


def _rows(a, b, weights, core):
    angular = baseline._weighted_mean((a["angular"]-b["angular"])**2, weights)
    c3 = .5*np.sum((a["coefficients"][:, 5:7]-b["coefficients"][:, 5:7])**2, axis=1)
    c4 = .5*np.sum((a["coefficients"][:, 7:9]-b["coefficients"][:, 7:9])**2, axis=1)
    return angular, c3, c4, core.angular_loss_weight*angular+core.third_loss_weight*c3+core.fourth_loss_weight*c4


def _pair_weights(prepared, pair, core):
    key = tuple(sorted(pair))
    if key in prepared["pair_cache"]:
        return prepared["pair_cache"][key]
    i, j = (prepared["source_ids"].index(sid) for sid in key)
    a, b = prepared["features"][i], prepared["features"][j]
    profile = _rows(a, b, prepared["source_weights"], core)[3]
    coverage = [prepared["coverage"][k] for k in (i, j)]
    reliability = np.minimum.reduce([np.minimum(c["angular_fraction"], np.min(c["quadrant_fraction"], axis=1)) for c in coverage])
    reliability *= a["fit_valid"] & b["fit_valid"]
    information = profile*reliability
    use = prepared["informative_mask"]
    mean = float(np.mean(information[use]))
    raw = np.zeros_like(profile)
    raw[use] = .5+.5*np.clip(information[use]/max(mean, 1e-12), 0., 2.)
    raw /= raw.sum()
    raw.setflags(write=False)
    record = dict(source_ids=list(key), weights=raw,
                  separation_profile=profile.copy(), source_reliability=reliability.copy(),
                  active_weight_ratio=float(raw[use].max()/raw[use].min()),
                  pair_source_separation=float(np.mean(profile[use])),
                  target_pixels_used=False, source_pair_count=1)
    prepared["pair_cache"][key] = record
    return record


def _loss(a, b, common, use, radial, core):
    rings = use & a["fit_valid"] & b["fit_valid"]
    w = np.where(rings, radial, 0.)
    if w.sum() <= 0:
        return None
    w = w/w.sum()
    avg = lambda value: float(np.sum(np.where(rings, value, 0.)*w))
    angular, c3, c4, _ = _rows(a, b, common, core)
    angular, c3, c4 = avg(angular), avg(c3), avg(c4)
    energy = avg(.5*np.sum(a["coefficients"][:, 5:9]**2, axis=1))
    source_energy = avg(.5*np.sum(b["coefficients"][:, 5:9]**2, axis=1))
    excess = max(math.sqrt(energy)-math.sqrt(source_energy), 0.)**2
    return dict(primary=core.angular_loss_weight*angular+core.third_loss_weight*c3+core.fourth_loss_weight*c4,
                angular_residual=angular, third_harmonic=c3, fourth_harmonic=c4,
                distinctive_target_energy=energy, distinctive_source_energy=source_energy,
                distinctive_excess_strength_energy=excess,
                distinctive_strength_explained=1.-excess/energy if energy > 1e-12 else None,
                distinctive_energy_explained=1.-(c3+c4)/energy if energy > 1e-12 else None,
                full_polar=avg(baseline._weighted_mean((a["ink"]-b["ink"])**2, common)),
                radial_mean=avg((a["radial_mean"]-b["radial_mean"])**2),
                informative_ring_count=int(np.count_nonzero(rings)),
                fit_valid=bool(np.all(a["fit_valid"][use]) and np.all(b["fit_valid"][use])))


def _strength(scores, core):
    resolved = [s for s in scores if s["distinctive_target_energy"] >= core.minimum_source_separation]
    if len(resolved) < core.minimum_usable_trial_fraction*len(scores):
        return dict(passed=True, fraction=None, resolved_trials=len(resolved))
    fraction = float(np.median([s["distinctive_strength_explained"] for s in resolved]))
    return dict(passed=fraction >= core.minimum_distinctive_energy_explained,
                fraction=fraction, resolved_trials=len(resolved))


def _separated_from_others(trials, pair, sids, core):
    others = [sid for sid in sids if sid not in pair]
    if not others:
        return dict(passed=True, other_source_ids=[], margin_q10=None)
    margins = [min(t["scores"][sid]["primary"] for sid in others)-max(t["scores"][sid]["primary"] for sid in pair) for t in trials]
    q10 = float(np.quantile(margins, .10))
    wins = float(np.mean(np.asarray(margins) > 0.))
    return dict(passed=bool(q10 > core.minimum_loss_margin and wins >= core.minimum_winner_fraction),
                other_source_ids=others, margin_q10=q10, robustness_fraction=wins)


def compare_sources(gray, center_xy, target_radius, sources, target_nuisance=None, config=None,
                    diagnostics=False, *, cache=None, audit=None):
    """Return original decisions unless a restricted, all-rival retry passes.

    ``cache={}`` may be shared within a run. ``audit={}`` receives separate
    diagnostics so an accepted baseline result remains bit-for-bit unchanged.
    The private baseline observer supplies immutable trial feature views; this
    function never resamples target pixels during its retry.
    """
    cfg, core = _configuration(config)
    models = list(sources.values()) if isinstance(sources, dict) else list(sources)
    collected = []
    result = baseline.compare_sources(gray, center_xy, target_radius, models, target_nuisance,
                                      core, diagnostics, _trial_observer=collected.append)
    report = dict(version=VERSION, status="baseline_retained", attempted=False, applied=False,
                  target_resampled_for_retry=False, source_pixels_modified=False,
                  source_center_corrected=False, connector_pixels_removed=False,
                  original_reason=result.get("reason"), baseline_accepted=bool(result["accepted"]))

    def finish(status, output=result):
        report["status"] = status
        if audit is not None:
            audit.update(report)
        return output

    if result["accepted"]:
        return finish("baseline_accepted_unchanged")
    if not getattr(cfg, "retry_enabled", True):
        return finish("retry_disabled")
    reasons = set(filter(None, result.get("reason", "").split(";")))
    if not reasons or not reasons <= ALLOWED_REASONS:
        return finish("non_ranking_failure_not_retried")
    sids = [str(m["sid"]) for m in models]
    usable = [item for item in collected if item["trial"]["usable"]]
    if len(usable) < core.minimum_usable_trial_fraction*len(collected) or len(sids) < 2:
        return finish("insufficient_usable_baseline_evidence")
    pair = sorted(sids, key=lambda sid: (result["scores"][sid], sid))[:2]
    report["pair_source_ids"] = pair
    # Both alternatives must already meet hard match/strength evidence. A
    # secondary rival cannot sneak past an absolute or null-model rejection.
    for sid in pair:
        original = [item["trial"]["scores"][sid] for item in usable]
        if result["scores"][sid] > core.maximum_primary_loss or not _strength(original, core)["passed"]:
            return finish("pair_member_fails_original_absolute_or_strength_evidence")
    original_trials = [item["trial"] for item in usable]
    report["baseline_third_rival_check"] = _separated_from_others(original_trials, pair, sids, core)
    if not report["baseline_third_rival_check"]["passed"]:
        return finish("third_rival_not_clearly_separated")
    fingerprint = _fingerprint(models, core)
    prepared = cache.get("_filled_contour_retry_prepared") if cache is not None else None
    reused = isinstance(prepared, dict) and prepared.get("fingerprint") == fingerprint
    if not reused:
        prepared = prepare_sources(models, core)
        if cache is not None:
            cache["_filled_contour_retry_prepared"] = prepared
    pair_info = _pair_weights(prepared, pair, core)
    radial = pair_info["weights"]
    report.update(attempted=True, prepared_cache_reused=reused,
                  active_ring_count=int(np.count_nonzero(prepared["informative_mask"])),
                  radii_normalized=prepared["radii_normalized"].tolist(), radial_weights=radial.tolist(),
                  radial_weight_ratio=pair_info["active_weight_ratio"],
                  source_pair_separation_profile=pair_info["separation_profile"].tolist(),
                  source_diagnostics=prepared["source_diagnostics"],
                  all_rivals_retained=sids)
    weighted = []
    for item in usable:
        scores = {sid: _loss(item["target_features"], sf, item["common_weights"], item["informative_ring_mask"], radial, core)
                  for sid, sf in zip(sids, item["source_features"])}
        if any(score is None or not score["fit_valid"] for score in scores.values()):
            return finish("retry_common_feature_support_invalid")
        weighted.append(dict(item["trial"], scores=scores))
    scores = {sid: float(np.median([trial["scores"][sid]["primary"] for trial in weighted])) for sid in sids}
    ordered = sorted(sids, key=lambda sid: (scores[sid], sid))
    winner = ordered[0]
    report.update(retry_scores=scores, retry_winner=winner)
    if winner not in pair:
        return finish("retry_winner_outside_original_pair")
    # The retry may change the winning source, but it must not bypass that
    # source's independent original-brightness fill evidence. Reuse the
    # baseline profile aggregation; no additional target sampling is needed.
    fill_evidence = result.get("source_relative_fill_evidence", {}).get(winner)
    if not fill_evidence or not fill_evidence.get("passed"):
        report["retry_fill_evidence"] = fill_evidence
        return finish("retry_winner_fails_source_relative_fill_evidence")
    report["retry_third_rival_check"] = _separated_from_others(weighted, pair, sids, core)
    if not report["retry_third_rival_check"]["passed"]:
        return finish("weighted_third_rival_not_clearly_separated")
    source_losses = [_loss(prepared["features"][i], prepared["features"][j], prepared["source_weights"],
                           prepared["informative_mask"], radial, core)
                     for i in range(len(sids)) for j in range(i+1, len(sids))]
    separation = min(loss["primary"] if loss else 0. for loss in source_losses)
    margins = [min(t["scores"][sid]["primary"] for sid in sids if sid != winner)-t["scores"][winner]["primary"] for t in weighted]
    q10 = float(np.quantile(margins, .10))
    fraction = float(np.mean(np.asarray(margins) > 0.))
    strength = _strength([trial["scores"][winner] for trial in weighted], core)
    checks = dict(absolute_match=scores[winner] <= core.maximum_primary_loss,
                  stable_winner=fraction >= core.minimum_winner_fraction,
                  positive_margin=q10 > core.minimum_loss_margin,
                  source_separation=separation >= core.minimum_source_separation,
                  distinctive_strength=bool(strength["passed"]))
    report.update(retry_checks=checks, retry_margin_q10=q10, retry_robustness_fraction=fraction,
                  retry_distinctive_strength=strength)
    if not all(checks.values()):
        return finish("weighted_retry_remains_uncertain")
    output = dict(result)
    output.update(version=VERSION, accepted=True, uncertain=False, winner=winner,
                  reason="stable_raw_source_identity_after_selective_pair_retry", scores=scores,
                  robustness_fraction=fraction, margin=scores[ordered[1]]-scores[winner], margin_q10=q10,
                  source_feature_separation=separation, distinctive_strength_explained=strength["fraction"],
                  resolved_distinctive_trials=strength["resolved_trials"],
                  winning_metadata=dict(models[sids.index(winner)].get("metadata", {})),
                  winning_fill_evidence=fill_evidence,
                  selective_retry_applied=True, baseline_reason=result["reason"],
                  primary_formula="unchanged angular/C3/C4 coefficients; positive shared radial average informed only by the original top-two raw legend sources",
                  geometry_policy="reuse original trial features without target resampling; one pair-only radial vector shared by every original source rival",
                  classification_policy="only ranking/stability abstentions may retry; accepted baseline decisions and source metadata remain unchanged")
    # Replace winner-specific diagnostic summaries, rather than accidentally
    # carrying over the baseline winner's evidence after an identity change.
    resolved = [t["scores"][winner] for t in weighted if t["scores"][winner]["distinctive_target_energy"] >= core.minimum_source_separation]
    enough = len(resolved) >= core.minimum_usable_trial_fraction*len(weighted)
    explained = float(np.median([s["distinctive_energy_explained"] for s in resolved])) if enough else None
    output.update(distinctive_energy_explained=explained,
                  phase_residual_guard_would_reject=bool(explained is not None and explained < core.minimum_distinctive_energy_explained))
    for key, field in (("median_distinctive_target_energy", "distinctive_target_energy"),
                       ("median_distinctive_source_energy", "distinctive_source_energy"),
                       ("median_distinctive_excess_strength_energy", "distinctive_excess_strength_energy")):
        output[key] = float(np.median([s[field] for s in resolved])) if resolved else None
    output["median_distinctive_residual"] = float(np.median([s["third_harmonic"]+s["fourth_harmonic"] for s in resolved])) if resolved else None
    if diagnostics:
        by_index = {trial["index"]: trial for trial in weighted}
        output["trials"] = [by_index.get(trial["index"], trial) for trial in result["trials"]]
    report["applied"] = True
    return finish("selective_pair_retry_accepted", output)

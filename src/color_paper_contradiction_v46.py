"""Raw-paper contradiction for v46 filled-marker centre localization.

Do not confuse absent target colour with visible white paper. Foreign ink and
ignored/out-of-image pixels provide neither contradiction nor positive credit.
This is a position-ranking cost, not a probability or a marker-acceptance gate.
"""
from dataclasses import dataclass, asdict
import math

import cv2
import numpy as np

VERSION = 'filled_raw_paper_center_cost_v1'
DEFAULT_WEIGHT = 2.


@dataclass(frozen=True)
class Config:
    paper_low: float = 225.
    paper_high: float = 245.
    noise_pixels: float = 1.
    noise_area_fraction: float = .005


def filled_eligibility(template, prior=None):
    """Require an affirmative legend identity; never infer filled from colour."""
    hint = str(template.get('legend_shape_hint', '')).lower()
    evidence = template.get('legend_shape_evidence') or {}
    if any(word in hint for word in ('open', 'hollow', 'partial', 'line_only', 'line-only')):
        return False, 'open_partial_or_line_legend'
    if np.any(template.get('hole_core', False)) or evidence.get('strong_hollow_evidence', False):
        return False, 'observed_legend_hole'
    # Frozen older payloads may carry the confirmed triangle prior separately.
    known_triangle = bool(prior and prior.get('applicable') and
                          prior.get('strength') in ('confirmed', 'bound_class'))
    if not hint.startswith('filled_') and not known_triangle:
        return False, 'no_confirmed_filled_legend'
    # Missing interior pixels are unmeasurable, not evidence of either a full
    # or an empty glyph. Abstain only from this filled-specific penalty; the
    # caller must continue the ordinary marker verification. Older payloads
    # without these optional fields retain their confirmed-shape behaviour.
    tone = evidence.get('continuous_fill_evidence') or {}
    for stats, key, minimum, reason in (
            (evidence, 'independent_fill', .85, 'legend_fill'),
            (tone, 'mean_membership', .72, 'continuous_fill')):
        if key not in stats:
            continue
        try:
            value = float(stats[key])
        except (TypeError, ValueError, OverflowError):
            return False, 'unmeasurable_' + reason
        if not math.isfinite(value) or not 0 <= value <= 1:
            return False, 'unmeasurable_' + reason
        if value < minimum:
            return False, 'insufficient_' + reason
    paper = np.asarray((template.get('model') or {}).get('paper_bgr', [255,255,255]), float)
    if not np.isfinite(paper).all() or np.min(paper) < 235:
        return False, 'nonwhite_paper_model'
    return True, 'confirmed_filled_legend'


def template_kernel(template):
    """Match verify_window's explicit template centre, including even canvases."""
    a = np.asarray(template['soft'], np.float32)
    if a.ndim != 2 or not np.isfinite(a).all() or not a.size:
        raise ValueError('Invalid paper-cost marker alpha')
    h,w = a.shape
    cx,cy = map(float, template.get('center', [(w-1)/2, (h-1)/2]))
    if not (0 <= cx <= w-1 and 0 <= cy <= h-1):
        raise ValueError('Template centre must be inside its alpha canvas')
    r = int(math.ceil(max(cx,cy,w-1-cx,h-1-cy)))+1
    return np.clip(cv2.warpAffine(a, np.float32([[1,0,r-cx],[0,1,r-cy]]),
                  (2*r+1,2*r+1), flags=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT), 0, 1)


def cost_map(evidence, index, *, weight=DEFAULT_WEIGHT, config=Config()):
    """Dense native/processed-image centre cost; original pixels stay immutable."""
    if not np.isfinite(weight) or weight < 0 or config.paper_high <= config.paper_low:
        raise ValueError('Invalid paper contradiction configuration')
    valid = np.asarray(evidence['valid'], bool)
    zero = np.zeros(valid.shape, np.float32)
    t = evidence['templates'][index]
    prior = evidence.get('priors', {}).get(str(t['id']))
    eligible, reason = filled_eligibility(t, prior)
    report = dict(version=VERSION, enabled=bool(weight), applied=False, weight=float(weight),
        reason='disabled' if weight == 0 else reason, configuration=asdict(config),
        scope='centre_refinement_and_alternative_peaks_only',
        raw_source_pixels=True, other_ink_is_positive_evidence=False,
        acceptance_threshold_changed=False)
    if weight == 0 or not eligible:
        return zero, report
    crop = evidence.get('crop_bgr')
    if crop is None:
        report['reason'] = 'original_plot_pixels_unavailable'
        return zero, report
    crop = np.asarray(crop)
    if crop.shape != (*valid.shape, 3) or not np.isfinite(crop).all():
        raise ValueError('Raw plot pixels must match paper-cost evidence geometry')
    other = np.asarray(evidence['other'][index], np.float32)
    ignored = np.asarray(evidence.get('ignore_mask', zero), np.float32)
    if other.shape != valid.shape or ignored.shape != valid.shape:
        raise ValueError('Paper-cost masks must match plot geometry')
    paper = np.clip((crop.min(axis=2).astype(np.float32)-config.paper_low)/
                    (config.paper_high-config.paper_low), 0, 1)
    observed_paper = paper*valid*(1-np.clip(ignored,0,1))*(1-np.clip(other,0,1))
    kernel = template_kernel(t)
    mass = max(float(kernel.sum()), 1e-8)
    if mass < 1:
        report['reason'] = 'insufficient_template_ink'
        return zero, report
    # One native-pixel defect allowance scales with evidence resizing. Glyph
    # area allowance also scales quadratically through the working diameter.
    sx = float(evidence.get('scale_x', evidence.get('scale', 1.)))
    sy = float(evidence.get('scale_y', evidence.get('scale', 1.)))
    noise = max(config.noise_pixels*sx*sy, config.noise_area_fraction*float(t['diameter'])**2)
    amount = cv2.filter2D(observed_paper, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    cost = np.maximum(amount-noise, 0)/mass
    report.update(applied=True, expected_mass=mass, noise_budget=float(noise))
    return cost.astype(np.float32), report

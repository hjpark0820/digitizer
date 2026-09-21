"""Separate own ink, observed missing ink, and known other-colour occlusion.

Occlusion only removes negative constraints. Positive scores always use the
full expected marker denominator; hiding more pixels cannot invent support.
The thresholds are conservative engineering safeguards, not probabilities.
"""
import cv2
import numpy as np

VERSION = 'bw_three_state_visibility_v1'


def validate_mask(mask, shape):
    if mask is None:
        return None
    value = np.asarray(mask, np.float32)
    if value.shape != tuple(shape) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('occlusion_mask must be finite, image-aligned and in [0,1]')
    return value if value.any() else None


def measure(template, membership, other):
    """Measure aligned patches. Unknown pixels are neither white nor own ink."""
    weight = np.asarray(template.valid_weight * template.mask, np.float32)
    total = max(float(weight.sum()), 1.)
    visible = 1. - np.asarray(other, np.float32)
    threshold = .45 if template.ink.achromatic else .24
    ink = (membership >= threshold).astype(np.float32)
    known = float((weight * visible).sum())
    positive = float((weight * ink * visible).sum())
    depth = cv2.distanceTransform(template.mask.astype(np.uint8), cv2.DIST_L2, 5)
    core = weight * (depth >= max(1.5, .10 * template.diameter))
    core_total = max(float(core.sum()), 1.)
    # Actual missing ink in the deep body is worse than an uncertain outline.
    core_paper = float((core * visible * (1. - ink)).sum() / core_total)
    yy, xx = np.indices(weight.shape)
    cy, cx = (np.asarray(weight.shape) - 1.) / 2.
    disk = ((xx-cx)**2 + (yy-cy)**2 <= max(1.5, .22*template.diameter)**2).astype(np.float32)
    observable_disk = disk * visible
    disk_total = float(observable_disk.sum())
    observed_fill = float((ink * observable_disk).sum() / max(disk_total, 1.e-6))
    expected_fill = float((template.mask * observable_disk).sum() / max(disk_total, 1.e-6))
    centre_visible = disk_total / max(float(disk.sum()), 1.)
    similarity = 1. - abs(observed_fill - expected_fill) if disk_total > 0 else 0.
    marker_visible = known / total
    edge_total = max(float(template.edge.sum()), 1.)
    result = dict(version=VERSION, visible_fraction=marker_visible,
        other_fraction=1.-marker_visible, positive_fraction=positive/total,
        visible_required_recall=positive/max(known, 1.e-6), core_paper_fraction=core_paper,
        centre_visible_fraction=centre_visible, centre_fill=observed_fill,
        expected_centre_fill=expected_fill, centre_similarity=similarity,
        edge_visible_fraction=float((template.edge*visible).sum()/edge_total),
        positive_ink_added=False)
    result['candidate_supported'] = bool(marker_visible >= .20 and positive/total >= .20
        and result['visible_required_recall'] >= .55 and core_paper <= .30)
    result['identity_observable'] = bool(marker_visible >= .70 and centre_visible >= .45
        and result['edge_visible_fraction'] >= .55 and similarity >= .65)
    return result

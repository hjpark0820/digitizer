"""Strong-image-evidence admission to v46's suppressed review pool.

Reuse existing image gates, not ranking score or curve benefit. Admission
is not marker confirmation or a calibrated probability.
"""
from __future__ import annotations

POLICY = 'strong_image_evidence_only_v1'
ORIGINS = frozenset(('path_residual', 'legend_shape_occlusion'))


def strong_evidence(point):
    """Fail closed for missing evidence, unknown sources and geometric knots."""
    return (point.get('origin') in ORIGINS
            and not point.get('triangle_cap_review_only', False)
            and point.get('kind') != 'diagnostic_support_knot'
            and point.get('evidence_supported') is True)


def policy_info():
    return dict(name=POLICY, criterion='known_image_proposal_origin AND evidence_supported == true',
        residual_gate='Existing image_backed_possible gate: all five compact/own-ink/thickness/prominence/aspect tests',
        shape_gate='Existing partial_shape_support gate: gain, visible support, missing fraction and >=2 residual quadrants',
        curve_benefit_required=False, ranking_score_threshold_used=False,
        thresholds_changed=False, existence='unknown', auto_promote=False)


def filter_strong_candidates(points):
    """Return copied strong proposals; rejected hypotheses stay diagnostic-only."""
    kept, excluded = [], []
    for point in points:
        if strong_evidence(point):
            kept.append(dict(point, evidence_tier='strong', suppressed_policy=POLICY,
                             state='suppressed', tentative=True, existence='unknown', auto_promote=False))
        else:
            excluded.append(dict(candidate_id=point.get('candidate_id'), origin=point.get('origin'),
                reason='unknown_proposal_source' if point.get('origin') not in ORIGINS
                       else 'missing_or_non_strong_image_evidence'))
    return kept, dict(policy=POLICY, input_count=len(kept)+len(excluded),
                      admitted_count=len(kept), excluded_count=len(excluded), excluded=excluded)

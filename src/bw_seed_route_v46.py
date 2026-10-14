"""Bounded, independently evidenced add-only routes after cold-start recovery.

This helper never edits points, templates, the immutable original structure,
or connected-engine references. It only reviews eligibility of a new phase.
"""
from collections import Counter
from copy import deepcopy
import json

from bw_series_structure_v46 import classify_connection

POLICY = 'bw-independent-seed-add-only-v1'


def _key(points):
    # Full measured geometry and provenance, not just a list of centers.
    return json.dumps(points, sort_keys=True, separators=(',', ':'), allow_nan=False)


class SeedRoutes:
    def __init__(self, ink, scope, models, initial_points, structure):
        self.ink, self.scope, self.models = ink, scope, models
        self.initial = deepcopy(initial_points)
        counts = Counter(p['swatch_id'] for p in initial_points)
        self.empty = {sid for sid in models if not counts[sid]}
        self.uncertain = {sid for sid, value in structure.items() if value['label'] == 'uncertain'}
        self.cache = {}
        self.reviews = []
        self.pair_reviews = []

    def pairs(self, active, pool, tests):
        from bw_seed_bootstrap_v46 import propose_pairs
        trials = propose_pairs(active, pool, self.models, self.ink, self.scope, tests, self.empty)
        self.pair_reviews.append(dict(active_count=len(active), trials=deepcopy(trials)))
        return trials

    def _review(self, sid, active, tests):
        from bw_seed_bootstrap_v46 import independent_point_evidence, independent_connector_evidence
        signature = (sid, _key(active))
        if signature in self.cache:
            return self.cache[signature]
        own = sorted((p for p in active if p['swatch_id'] == sid), key=lambda p:(p['cx'],p['cy']))
        original = [p for p in self.initial if p['swatch_id'] == sid]
        review = dict(swatch_id=sid, valid=False, active_count=len(own), anchors=[], connectors=[])
        self.cache[signature] = review
        if sid not in self.uncertain:
            review['reason'] = 'initial_route_not_uncertain'
            return review
        if _key(own) == _key(sorted(original,key=lambda p:(p['cx'],p['cy']))):
            review['reason'] = 'no_new_verified_anchor'
            return review
        if len(own) < 3:
            review['reason'] = 'fewer_than_three_anchors'
            return review
        if len(own) > 32:
            review['reason'] = 'bounded_anchor_review_limit'
            return review
        for p in own:
            evidence = independent_point_evidence(p,active,self.models,self.ink,self.scope,tests)
            review['anchors'].append(dict(point_id=p.get('point_id'),cx=p['cx'],cy=p['cy'],**evidence))
        if not all(a.get('valid') for a in review['anchors']):
            review['reason'] = 'not_all_rendered_anchors_independent'
            return review
        classifier = classify_connection(own,self.ink,max(self.models[sid]['mask'].shape),self.scope)
        review['classification'] = {k:v for k,v in classifier.items() if k != 'pairs'}
        if classifier['label'] != 'connected_supported':
            review['reason'] = 'insufficient_repeated_connector_support'
            return review
        # Repeated ink must survive removing other marker bodies / independently
        # attributed other-series connectors; a clean subset cannot authorize a
        # polyline drawn through all the unverified active points.
        eligible = 0
        supported = 0
        for p,q in zip(own,own[1:]):
            check = independent_connector_evidence(p,q,active,self.models,self.ink,self.scope)
            review['connectors'].append(check)
            original_chord = check.get('original_chord',{})
            if original_chord.get('eligible'):
                eligible += 1
                supported += bool(check.get('valid'))
        review.update(unique_eligible_pairs=eligible,unique_supported_pairs=supported)
        review['valid'] = eligible >= 2 and supported/max(1,eligible) >= .80
        review['reason'] = 'independent_repeated_connectors' if review['valid'] else 'connector_ink_not_independent'
        return review

    def refresh(self, active, tests):
        reviews = [self._review(sid,active,tests) for sid in sorted(self.uncertain)]
        self.reviews.append(dict(active_count=len(active), series=deepcopy(reviews)))
        return {r['swatch_id'] for r in reviews if r['valid']}

    def validate_addition(self, point, proposed, tests):
        from bw_seed_bootstrap_v46 import independent_point_evidence
        evidence = independent_point_evidence(point,proposed,self.models,self.ink,self.scope,tests)
        if not evidence.get('valid'):
            return dict(valid=False,reason='addition_marker_not_independent',marker=evidence)
        route = self._review(point['swatch_id'],proposed,tests)
        return dict(valid=route['valid'],reason=route['reason'],marker=evidence,route=route)

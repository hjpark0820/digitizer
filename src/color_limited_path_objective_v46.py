"""Frozen path comparisons for limited colour Step-5 candidate actions.

Geometry is measured before image admission. A favourable path cost never
supplies missing marker ink or authorizes deleting original measurements.
"""
from color_path_metrics_v46 import METRICS, PathReference, score_markers
from color_x_completeness_v46 import XCompleteness
import math


class LimitedPathObjective:
    def __init__(self, structure, diameter, options=None, original_rows=()):
        opts = options or {}
        self.metric = opts.get('metric', 'chamfer')
        self.model = opts.get('limited_model', opts.get('model', 'pchip'))
        self.policy = opts.get('reference_policy', 'estimated_path')
        self.inferred_weight = float(opts.get('inferred_weight', .25))
        self.completeness_weight=float(opts.get('limited_completeness_weight',0.))
        self.image_weight=opts.get('limited_image_weight')
        if self.image_weight is not None:
            self.image_weight=float(self.image_weight)
            if not math.isfinite(self.image_weight) or self.image_weight<0:
                raise ValueError('Image weight must be finite and nonnegative')
        self.image_cost=None
        self.image_normalizer=1.
        if not math.isfinite(self.completeness_weight) or self.completeness_weight<0:
            raise ValueError('Completeness weight must be finite and nonnegative')
        self.completeness=XCompleteness((),None,diameter,enabled=False)
        if self.metric not in METRICS or self.model not in ('linear', 'pchip'):
            raise ValueError('Unknown limited Step-5 path metric/model')
        if self.policy not in ('observed', 'estimated_path') or not 0 < self.inferred_weight <= 1:
            raise ValueError('Invalid limited Step-5 path reference policy/weight')
        self.reference = None
        self.ranking_enabled = False
        record = structure.get('path_record') or {}
        self.reason = 'no_extracted_path'
        if not len(record.get('path', [])):
            return
        ref = PathReference.from_record(record, diameter,
            reference_policy=self.policy, inferred_weight=self.inferred_weight)
        # Finite inferred coordinates in a blank image are not a reference.
        if int(ref.true_observed.sum()) < 2:
            self.reason = 'insufficient_observed_path_support'
            return
        self.reference = ref
        self.ranking_enabled = (structure.get('mode') in ('fitted_curve', 'uncertain')
            and structure.get('eligible_pairs', 0) >= 1
            and structure.get('inter_marker_path_support', 0.) >= .55)
        self.reason = ('supported_path_add_ranking' if self.ranking_enabled else
                       'path_comparison_only_insufficient_inter_marker_line_support')
        self.completeness=XCompleteness(original_rows,ref,diameter,
            enabled=self.completeness_weight>0 and self.ranking_enabled)

    def describe(self):
        return dict(metric=self.metric, model=self.model, reference_policy=self.policy,
            inferred_weight=self.inferred_weight, ranking_enabled=self.ranking_enabled,
            reason=self.reason, geometry_is_marker_evidence=False,
            reference=self.reference.describe() if self.reference is not None else None,
            completeness_enabled=self.completeness.enabled,
            completeness_weight=self.completeness_weight,completeness=self.completeness.describe(),
            image_policy='weighted_cost' if self.image_cost is not None else 'hard_admission',
            image_weight=self.image_weight, image_normalizer=self.image_normalizer,
            objective_formula=('path_cost + completeness_weight * (missing + duplicate + offgrid)'
                + (' + image_weight * frozen_marker_image_loss' if self.image_cost is not None else '')))

    def bind_image_cost(self, evaluator, original_points):
        if self.image_weight is None or not self.ranking_enabled:
            return
        if evaluator is None:
            raise ValueError('Weighted image objective requires bound pixel evidence')
        self.image_cost=evaluator
        # Fixed denominator: adding a bad point must not dilute earlier errors.
        self.image_normalizer=(max(1.,float(self.completeness.weights.sum()))
            if self.completeness.enabled else max(1.,float(len(original_points))))

    def score(self, points):
        if self.reference is None:
            return dict(status=self.reason, objective=None, combined_objective=None,coverage=None,
                        completeness=self.completeness.score(points),completeness_cost=0.)
        result=score_markers(self.reference, points, metric=self.metric, model=self.model)
        c=self.completeness.score(points);cost=self.completeness_weight*c['loss']
        image={}
        image_cost=0.
        if self.image_cost is not None:
            values=[self.image_cost(p) for p in points]
            loss=sum(v['cost'] for v in values)/self.image_normalizer
            image_cost=self.image_weight*loss
            image=dict(image_loss=loss,image_cost=image_cost,image_terms=values,
                       image_normalizer=self.image_normalizer)
        return dict(result,completeness=c,completeness_cost=cost,
                    combined_objective=result['objective']+cost+image_cost,**image)

    def compare(self, before, after, baseline=None):
        b = self.score(before) if baseline is None else baseline
        a = self.score(after)
        available = b['objective'] is not None and a['objective'] is not None
        gain = float(b['objective'] - a['objective']) if available else None
        objective_gain=float(b['combined_objective']-a['combined_objective']) if available else None
        # Numerical tolerance only, not a new empirical minimum-gain threshold.
        eps = 1e-9 * max(1., abs(b['combined_objective'])) if available else None
        return dict(status='evaluated' if available else self.reason,
            before=b, after=a, gain=gain,
            objective_gain=objective_gain,completeness_gain=b['completeness']['loss']-a['completeness']['loss'],
            improves=bool(available and objective_gain > eps),path_improves=bool(available and gain>eps),
            ranking_enabled=self.ranking_enabled,
            image_admission_applied=False)

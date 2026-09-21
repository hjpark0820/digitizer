"""GUI stopping policy and truthful summaries of the unchanged native searches.

Automatic mode uses native no-action termination plus a 20-iteration safety cap.
Small relative gains are warnings, never a new action filter or a hard stop.
"""
from pathlib import Path
import json
import math

REPORT_NAME = 'step5_stop_report.json'
AUTO_CAP = 20
PLATEAU_WARNING = .001
LABELS = {
    'no_candidates': 'No candidate actions remained.',
    'all_candidates_blocked': 'All candidate actions failed evidence or protection checks.',
    'no_improving_action': 'No admissible action improved the internal objective.',
    'budget_exhausted': 'Maximum iterations reached; convergence was NOT established.',
    'uncertain_preserved': 'Uncertain series structure; points were preserved.',
    'preserved': 'Correction was unavailable for this series; points were preserved.',
    'already_stopped': 'Previously stopped state reused; no new iterations executed.',
    'diagnostics_unavailable': 'Native stopping diagnostics were unavailable.',
}


def resolve_policy(policy='manual', iterations=5):
    if policy not in ('manual', 'auto'):
        raise ValueError('Unknown Step-5 stop policy; choose manual or auto')
    if policy == 'auto':
        return dict(policy='auto', max_iterations=AUTO_CAP, plateau_warning_fraction=PLATEAU_WARNING)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 50:
        raise ValueError('Correction iterations must be an integer from 1 to 50')
    return dict(policy='manual', max_iterations=iterations, plateau_warning_fraction=PLATEAU_WARNING)


def trace_snapshot(folder):
    return {p.resolve(): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in Path(folder).rglob('trace.json')}


def relative_gain(row):
    before = row.get('score_before', row.get('baseline'))
    after = row.get('selected_cost_adjusted_distance', row.get('score_after', row.get('best_dist')))
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in (before, after)):
        return None
    return (before-after)/max(abs(before), 1e-6)


def summarise_trace(trace, cap):
    # Colour traces may include older iterations when correction is resumed.
    previous = int(trace.get('previous_iteration_count',
        max(0, trace.get('requested_iterations', cap)-cap) if trace.get('series') else 0))
    rows = [r for r in trace.get('iterations', []) if r.get('iteration', 0) > previous
            and r.get('executed', True)]
    last = rows[-1] if rows else {}
    reason = trace.get('stop_reason') or last.get('stop_reason') or ''
    proposals = last.get('trials', last.get('proposals'))
    accepted = [r for r in rows if r.get('action') not in (None, 'NONE', 'NO_OP')]
    if reason == 'no_admissible_image_candidate':
        code = 'all_candidates_blocked'
        proposals = last.get('candidate_reviews', proposals)
    elif reason in ('no_path_improving_candidate','no_objective_improving_candidate'):
        code = 'no_improving_action'
        proposals = last.get('candidate_reviews', proposals)
    elif reason == 'no_available_candidate_slot':
        code = 'no_candidates'
        proposals = last.get('candidate_reviews', proposals)
    elif reason == 'uncertain_structure_preserved':
        code = 'uncertain_preserved'
    elif reason and ('preserved' in reason or 'not_corrected' in reason or 'unavailable' in reason):
        code = 'preserved'
    elif not rows:
        code = 'already_stopped' if trace.get('resumed') and trace.get('path_state', {}).get('converged') else 'diagnostics_unavailable'
    elif last.get('action') in ('NONE', 'NO_OP'):
        if proposals == []:
            code = 'no_candidates'
        elif proposals and all(p.get('admissible') is False for p in proposals):
            code = 'all_candidates_blocked'
        else:
            code = 'no_improving_action'
    elif len(rows) >= cap:
        code = 'budget_exhausted'
    else:
        code = 'diagnostics_unavailable'
    warnings = [dict(iteration=r['iteration'], relative_gain=g)
                for r in accepted if (g := relative_gain(r)) is not None and 0 <= g < PLATEAU_WARNING]
    return dict(series=str(trace.get('series') or 'all_series'),
        mode=trace.get('mode', trace.get('metric', (trace.get('method') or {}).get('metric', 'native'))),
        iterations_executed=len(rows), accepted_actions=len(accepted), previous_iterations=previous,
        stop_code=code, stop_message=LABELS[code], native_stop_reason=reason,
        last_candidate_count=len(proposals) if proposals is not None else None,
        last_admissible_count=sum(p.get('admissible', True) for p in proposals) if proposals is not None else None,
        low_gain_warnings=warnings)


def build_report(folder, config, before):
    folder = Path(folder)
    fresh = trace_snapshot(folder)
    series = []
    for path, stamp in sorted(fresh.items()):
        if before.get(path) == stamp:
            continue
        trace = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(trace,dict) and trace.get('parent_dispatcher')=='bw_series_correction_v46_v1':
            continue  # Joint dispatcher trace already counts this action.
        if not isinstance(trace, dict) or not isinstance(trace.get('iterations'), list):
            continue
        if not ('series' in trace or trace.get('metric') == '1-SSIM'):
            continue
        item = summarise_trace(trace, config['max_iterations'])
        item['trace_file'] = path.relative_to(folder.resolve()).as_posix()
        series.append(item)
    codes = [s['stop_code'] for s in series]
    budget = 'budget_exhausted' in codes
    unknown = not series or 'diagnostics_unavailable' in codes
    message = ('Maximum iterations reached for at least one series; convergence not established.' if budget else
               'Correction finished; some stopping diagnostics are unavailable.' if unknown else
               'Search stopped or preserved the input series; marker accuracy is not established.')
    report = dict(version='v46-step5-stop-v1', **config, engine_decisions_unchanged=True,
        cap_scope='per invocation; joint BW iterations or iterations per colour series',
        plateau_warning_only=True, series=series, budget_exhausted=budget,
        low_gain_warning_count=sum(len(s['low_gain_warnings']) for s in series), summary=message)
    (folder/REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report

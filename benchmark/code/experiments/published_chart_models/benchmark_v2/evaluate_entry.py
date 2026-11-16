"""Symbol + position F1: a prediction counts only if it is within tolerance AND assigned to the right legend symbol.

The user's definition: a data point is correct when its position matches a reference point and its
marker corresponds to the legend symbol of that reference point's series. Symbol NAMES play no role;
what matters is which legend entry (legend symbol) the point was assigned to.

  legend entry of a prediction   v46: the shared legend entry whose native id equals the point's
                                 native_series_id (its curve = that legend entry);
                                 AI-ChartParser / LineEX: the legend crop their Siamese matcher
                                 assigned the point to (legend_index into the same shared entries).
  legend entry of a reference    PK: the entry matched to the annotated swatch box of that series
  series (ground truth)          (bbox overlap, shared_legends.json), except panels re-established
                                 from pixels; ER (no swatch annotation): established from pixels by
                                 two independent annotators + adjudication (legend_gt/final.json).
                                 A series with no detected legend entry can be matched by nobody.
  single-series panels           only one symbol exists, so position alone decides.
  identical duplicate entries    where the legend shows the same symbol twice (same marker, same text,
                                 e.g. one per legend group), those entries are one symbol
                                 (--equivalent-entries), so either assignment counts.
Matching: evaluate.py's one-to-one maximum-cardinality matching at the primary OKS tolerance
(OKS>0.5, LineEX image-diagonal scale), with the legend entry in place of the series label.
Pooled F1 = 2TP / (2TP + FP + FN) over the panels of a group; 95% intervals from 2000 panel resamples.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('benchmark_v2_evaluate', HERE / 'evaluate.py')
EV = importlib.util.module_from_spec(spec)
sys.modules['benchmark_v2_evaluate'] = EV
spec.loader.exec_module(EV)

METHODS = ['v46_before', 'v46_after', 'aicp_raw_grouped', 'aicp_corrected_grouped', 'lineex_paper_grouped',
           'lineex_grouped']
LABELS = {'v46_before': 'v46 (Step 5 전)', 'v46_after': 'v46 (Step 5 후)',
          'aicp_raw_grouped': 'AI-ChartParser (raw)', 'aicp_corrected_grouped': 'AI-ChartParser (corrected)',
          'lineex_paper_grouped': 'LineEX (논문 설정)', 'lineex_grouped': 'LineEX (공개코드 설정)'}
GROUPS = [('pk', 'bw'), ('pk', 'color'), ('er', 'bw'), ('er', 'color'), ('pk', None), ('er', None),
          (None, 'bw'), (None, 'color'), (None, None)]
TOL = EV.PRIMARY


def group_name(dataset, modality):
    names = {'pk': 'PK', 'er': 'ER', 'bw': '흑백', 'color': '컬러'}
    parts = [names[v] for v in (dataset, modality) if v]
    return ' '.join(parts) if parts else '전체'


def entries_of(v46_folder):
    path = v46_folder / 'shared_legends.json'
    return EV.read(path).get('entries', []) if path.exists() else []


def reference_entries(panel_id, entries, ground_truth):
    """series_id -> entry index (None = no detected legend entry for that series)."""
    if panel_id in ground_truth:
        return {s['series_id']: (s['entry_index'] if s['entry_index'] >= 0 else None)
                for s in ground_truth[panel_id]['series']}
    return {str(e['series_id']): i for i, e in enumerate(entries) if e.get('series_id') is not None}


def prediction_entry(method, point, entries):
    if method.startswith('v46'):
        native = str(point.get('native_series_id') or point.get('series_id'))
        for i, e in enumerate(entries):
            if native in (str(e.get('native_id')), str(e.get('source_native_legend_id'))):
                return i
        return None
    if point.get('legend_index') is not None:
        return int(point['legend_index'])
    series = str(point.get('series_id'))
    return next((i for i, e in enumerate(entries) if str(e.get('series_id')) == series), None)


def score_panel(panel, target, results, ground_truth, equivalent=None):
    reference = target['points']
    v46_folder = results / 'v46' / panel['panel_id']
    entries = entries_of(v46_folder)
    canonical = {}
    for group in (equivalent or {}).get(panel['panel_id'], []):
        for i in group:
            canonical[i] = min(group)
    same = lambda i: None if i is None else canonical.get(i, i)
    ref_entry = {k: same(v) for k, v in reference_entries(panel['panel_id'], entries, ground_truth).items()}
    single = len({p['series_id'] for p in reference}) == 1
    radius = EV.tolerances(panel)[TOL]
    row = {'panel_id': panel['panel_id'], 'dataset': panel['dataset'], 'modality': panel['modality'],
           'reference_points': len(reference), 'single_series': single,
           'reference_points_without_entry': sum(1 for p in reference if ref_entry.get(p['series_id']) is None),
           'ground_truth_source': 'pixels (legend_gt)' if panel['panel_id'] in ground_truth else 'annotated swatch overlap',
           'methods': {}}
    for method in METHODS:
        folder = v46_folder if method.startswith('v46') else results / 'external' / panel['panel_id']
        record = EV.method_predictions(folder, panel, method)
        preds = record['predictions']
        if single:
            m = EV.evaluate_points(preds, reference, tolerance=radius, label_aware=False)
        else:
            p2 = [dict(p, series_id=same(prediction_entry(method, p, entries))) for p in preds]
            r2 = [dict(p, series_id=ref_entry.get(p['series_id'])) for p in reference]
            m = EV.evaluate_points(p2, r2, tolerance=radius, label_aware=True)
        row['methods'][method] = {'status': record['status'], 'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn']}
    return row


def f1(t):
    tp, fp, fn = t
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True, help='legend_gt/final.json')
    parser.add_argument('--equivalent-entries', type=Path, help='legend_gt/equivalent_entries.json')
    args = parser.parse_args()
    panels = EV.read(args.inputs)['panels']
    targets = {t['panel_id']: t for t in EV.read(args.targets)['panels']}
    ground_truth = EV.read(args.ground_truth)
    equivalent = EV.read(args.equivalent_entries) if args.equivalent_entries else {}
    rows = [score_panel(p, targets[p['panel_id']], args.results, ground_truth, equivalent) for p in panels]
    rng = np.random.default_rng(0)
    summary = {}
    for dataset, modality in GROUPS:
        sel = [r for r in rows if (dataset is None or r['dataset'] == dataset) and (modality is None or r['modality'] == modality)]
        if not sel:          # e.g. a small subset run without panels of this group
            summary[group_name(dataset, modality)] = {'panels': 0, 'reference_points': 0,
                                                      'methods': {m: {'f1': None, 'ci95': None, 'tp_fp_fn': [0, 0, 0]} for m in METHODS}}
            continue
        idx = rng.integers(0, len(sel), (2000, len(sel)))
        block = {'panels': len(sel), 'reference_points': sum(r['reference_points'] for r in sel), 'methods': {}}
        for method in METHODS:
            table = np.array([[r['methods'][method][k] for k in ('tp', 'fp', 'fn')] for r in sel], float)
            boot = np.array([f1(table[b].sum(0)) for b in idx])
            block['methods'][method] = {'f1': f1(table.sum(0)), 'ci95': np.nanpercentile(boot, [2.5, 97.5]).tolist(),
                                        'tp_fp_fn': table.sum(0).astype(int).tolist()}
        summary[group_name(dataset, modality)] = block
    EV.write(args.results / 'entry_evaluation.json', {'definition': __doc__, 'tolerance': TOL, 'summary': summary,
                                                      'panels': rows})
    names = [group_name(d, m) for d, m in GROUPS]
    lines = ['| 방법 | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
    for method in METHODS:
        lines.append(f'| {LABELS[method]} | ' + ' | '.join('–' if summary[n]['methods'][method]['f1'] is None else
                                                          f"{100 * summary[n]['methods'][method]['f1']:.1f}" for n in names) + ' |')
    lines.append('| 패널 수 | ' + ' | '.join(str(summary[n]['panels']) for n in names) + ' |')
    lines.append('| 기준점 수 | ' + ' | '.join(str(summary[n]['reference_points']) for n in names) + ' |')
    text = '\n'.join(lines) + '\n'
    (args.results / 'entry_f1_table.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

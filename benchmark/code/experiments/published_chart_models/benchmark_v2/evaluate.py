"""Point-coordinate and marker-symbol evaluation for the 20-panel two-dataset benchmark.

Separate from every detector run: it reads predictions that already exist on disk plus
targets.json, and writes evaluation.json / metrics_summary.json / summary.md.

Point matching (unchanged from the earlier PK protocol, experiments/color_marker_hybrid/metrics.py)
  maximum-cardinality one-to-one assignment first, minimum total distance second; one prediction
  claims at most one reference point; coincident references of different series stay distinct.
  label_aware=True additionally requires the same series id; center_only ignores series identity,
  so the gap between them isolates series-assignment failures from localisation failures.

Tolerances
  Panels here range from 324x313 to 2401x1265 px, so a single pixel radius is not comparable
  across them. The primary tolerance is therefore scale free:
      tol_px = rel * diagonal(plot box)          rel in {0.005, 0.01, 0.02}
  with the fixed radii {3, 5, 10} px also reported for continuity with benchmark_pk/run_v1.
  Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); pooled (micro) counts and
  the mean of per-panel F1 (macro) are both reported.

Marker symbol
  The reference gives one marker class per series: PK from the hand-written layout annotation
  ("filled square"), ER from MARKER_CLASS in DIGITIZATION_ALL. Predictions come only from v46
  (BW: curve class_name; colour: curve marker_class after Step 5). LineEX predicts no marker
  class at all and the released AI-ChartParser heads predict chart-element boxes, not marker
  shapes, so for those two the symbol column is "not predicted by the method" - reported as
  zero coverage, never as a score of zero.
  symbol_accuracy = correct classes / reference series that the method assigned any class to;
  symbol_coverage = series with a predicted class / reference series.

Reference honesty
  Each panel carries reference_origin. PK panels use a manual curated digitisation. ER panels are
  split into agent digitisation (independent of v46) and chartocode2 v46 output (raw or hand
  edited) - for the latter, v46 is being compared against a reference it produced, which inflates
  its own scores. All aggregates are reported per origin as well as pooled.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.color_marker_hybrid.metrics import evaluate_points  # noqa: E402

METHODS = ['v46_before', 'v46_after', 'aicp_corrected_grouped', 'aicp_raw_grouped', 'lineex_grouped',
           'lineex_paper_grouped']
REL_TOLERANCES = [0.005, 0.01, 0.02]
FIXED_TOLERANCES = [3.0, 5.0, 10.0]
# Both published methods score a predicted point with object keypoint similarity
#   OKS = exp(-d^2 / (2 s^2 k^2)),  k = 0.025,  accepted when OKS > 0.5,
# which is exactly d <= s * k * sqrt(2 ln 2) = 0.029430 * s.
# LineEX (WACV 2023, section 5.3) sets s = the chart image diagonal; the ChartParser repository's
# own validation code (tools/utils.py) sets s = sqrt(width * height) of the image.
OKS_FACTOR = 0.025 * (2 * math.log(2)) ** 0.5
PRIMARY = 'oks_lineex'
SYMBOL_METHODS = {'v46_before', 'v46_after'}

MARKER_WORDS = {
    'filled': 'filled', 'solid': 'filled', 'closed': 'filled', 'full': 'filled',
    'open': 'open', 'hollow': 'open', 'empty': 'open', 'unfilled': 'open',
}
SHAPE_WORDS = {
    'circle': 'circle', 'circles': 'circle', 'dot': 'circle', 'disc': 'circle', 'disk': 'circle', 'ring': 'circle',
    'square': 'square', 'squares': 'square', 'box': 'square',
    'triangle': 'triangle', 'triangles': 'triangle', 'tri': 'triangle',
    'diamond': 'diamond', 'diamonds': 'diamond', 'rhombus': 'diamond',
    'cross': 'cross', 'plus': 'plus', 'star': 'star', 'x': 'cross',
}
INVERTED_WORDS = {'inverted', 'inv', 'down', 'downward', 'downwards'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def normalise_marker(text):
    """'open inverted triangle' / 'open_inv_triangle' -> 'open_inv_triangle'; unknown -> None."""
    if not text:
        return None
    words = [w for w in str(text).replace('-', ' ').replace('_', ' ').lower().split() if w]
    fill = next((MARKER_WORDS[w] for w in words if w in MARKER_WORDS), None)
    shape = next((SHAPE_WORDS[w] for w in words if w in SHAPE_WORDS), None)
    inverted = any(w in INVERTED_WORDS for w in words)
    if shape is None:
        return None
    if shape in ('cross', 'plus', 'star'):
        return shape
    if fill is None:
        return None
    return f"{fill}_{'inv_' if inverted and shape == 'triangle' else ''}{shape}"


def diagonal(panel):
    a, b, c, d = panel['plot_bbox']
    return math.hypot(c - a, d - b)


def tolerances(panel):
    diag = diagonal(panel)
    width, height = panel['image_size']
    out = {'oks_lineex': OKS_FACTOR * math.hypot(width, height),
           'oks_chartparser': OKS_FACTOR * math.sqrt(width * height)}
    out.update({f'rel{rel*100:g}pct': max(2.0, rel * diag) for rel in REL_TOLERANCES})
    out.update({f'fixed{int(px)}px': px for px in FIXED_TOLERANCES})
    return out


TOLERANCE_KEYS = ['oks_lineex', 'oks_chartparser'] + [f'rel{rel*100:g}pct' for rel in REL_TOLERANCES] +                  [f'fixed{int(px)}px' for px in FIXED_TOLERANCES]


def in_scope(point, panel):
    x = point['x_px'] - panel['crop_box_original'][0]
    y = point['y_px'] - panel['crop_box_original'][1]
    a, b, c, d = panel['plot_bbox']
    legend = panel.get('legend_bbox')
    inside = a - 2 <= x < c + 2 and b - 2 <= y < d + 2
    return inside and not (legend and legend[0] <= x < legend[2] and legend[1] <= y < legend[3])


def method_predictions(folder, panel, method):
    """Predictions plus the method's own status, without inventing empty successes."""
    path = folder / f'{method}.json'
    if not path.exists():
        return {'status': 'not_run', 'predictions': [], 'symbol_by_series': {}}
    record = read(path)
    status = record.get('status', 'unspecified')
    points = [] if status in {'failed', 'timeout', 'unavailable', 'not_run'} else record.get('predictions', [])
    kept = [p for p in points if in_scope(p, panel)]
    return {'status': status, 'predictions': kept, 'dropped_out_of_scope': len(points) - len(kept),
            'symbol_by_series': {}, 'timing': record.get('timing', {}), 'reason': record.get('reason', '')}


def v46_edit_data_path(folder, method):
    """The edit_data.json that produced <method>.json.

    v46_after.json written by run_step5_gui.py (GUI /correct-saved path, run_100b onwards) comes from
    gui_step5/job/out/edit_data.json; raw_after/ then holds the legacy CLI correction (run_100b) or,
    in run_100d where that legacy step is skipped, a copy of the detection-stage export for colour
    panels (pre-Step-5 classes) and nothing for BW. Reading raw_after for those runs scored a
    different output than the points being scored.
    """
    if method == 'v46_before':
        return folder / 'raw_before' / 'edit_data.json'
    record_path = folder / 'v46_after.json'
    if record_path.exists() and str(read(record_path).get('step5_entry_point', '')).startswith('GUI'):
        return folder / 'gui_step5' / 'job' / 'out' / 'edit_data.json'
    return folder / 'raw_after' / 'edit_data.json'


def v46_symbols(folder, panel, method):
    """Marker class per SERIES id, taken from the production export of that stage."""
    path = v46_edit_data_path(folder, method)
    if not path.exists():
        return {}, 'edit_data.json missing'
    shared = folder / 'shared_legends.json'
    mapping = read(shared).get('mapping', {}) if shared.exists() else {}
    out = {}
    for curve in read(path).get('curves', []):
        native = str(curve.get('swatch_id') or curve.get('name'))
        series = mapping.get(native, native)
        value = curve.get('marker_class') or curve.get('class_name') or curve.get('shape_hint')
        out[series] = {'raw': value, 'class': normalise_marker(value), 'native_series_id': native}
    return out, ''


def evaluate_panel(panel, target, v46_root, external_root):
    diag = diagonal(panel)
    tol = tolerances(panel)
    reference = [p for p in target['points'] if in_scope(p, panel)]
    if len(reference) != len(target['points']):
        raise ValueError(f"{panel['panel_id']}: reference points outside the shared scope")
    symbol_reference = {sid: normalise_marker(value.get('marker_text'))
                        for sid, value in target['symbol_reference'].items()}
    row = {'panel_id': panel['panel_id'], 'dataset': panel['dataset'], 'modality': panel['modality'],
           'figure_id': panel['figure_id'], 'reference_origin': target['reference_origin'],
           'reference_flags': target.get('reference_flags', []), 'reference_points': len(reference),
           'reference_series': len(symbol_reference), 'plot_diagonal_px': round(diag, 1),
           'tolerance_px': {k: round(v, 2) for k, v in tol.items()},
           'symbol_reference': {sid: {'raw': target['symbol_reference'][sid].get('marker_text'), 'class': cls}
                                for sid, cls in symbol_reference.items()},
           'models': {}}
    for method in METHODS:
        folder = (v46_root if method.startswith('v46') else external_root) / panel['panel_id']
        record = method_predictions(folder, panel, method)
        metrics = {}
        for name, radius in tol.items():
            entry = {}
            for identity, aware in (('center_only', False), ('label_aware', True)):
                entry[identity] = evaluate_points(record['predictions'], reference, tolerance=radius, label_aware=aware)
            centre, labelled = entry['center_only']['tp'], entry['label_aware']['tp']
            entry['identity_recovery_ratio'] = labelled / centre if centre else None
            metrics[name] = entry
        symbol = {'supported': method in SYMBOL_METHODS, 'predicted': {}, 'accuracy': None,
                  'coverage': None, 'correct': 0, 'compared': 0,
                  'note': 'method does not predict a marker class' if method not in SYMBOL_METHODS else ''}
        if method in SYMBOL_METHODS and record['status'] not in {'failed', 'not_run'}:
            predicted, note = v46_symbols(folder, panel, method)
            correct = compared = 0
            for sid, expected in symbol_reference.items():
                got = predicted.get(sid, {}).get('class')
                if expected is None or got is None:
                    continue
                compared += 1
                correct += int(got == expected)
            symbol.update(predicted={k: v for k, v in predicted.items()}, correct=correct, compared=compared,
                          accuracy=correct / compared if compared else None,
                          coverage=sum(1 for sid in symbol_reference if predicted.get(sid, {}).get('class')) /
                                   max(1, len(symbol_reference)), note=note)
        row['models'][method] = {'status': record['status'], 'prediction_count': len(record['predictions']),
                                 'dropped_out_of_scope': record.get('dropped_out_of_scope', 0),
                                 'reason': record.get('reason', ''), 'metrics': metrics, 'symbol': symbol,
                                 'timing': record.get('timing', {})}
    return row


def pooled(rows, method, key, identity):
    tp = sum(r['models'][method]['metrics'][key][identity]['tp'] for r in rows)
    fp = sum(r['models'][method]['metrics'][key][identity]['fp'] for r in rows)
    fn = sum(r['models'][method]['metrics'][key][identity]['fn'] for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    per_panel = [r['models'][method]['metrics'][key][identity]['f1'] for r in rows]
    finite = [v for v in per_panel if v is not None]
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1,
            'macro_f1': sum(finite) / len(finite) if finite else None, 'panels': len(rows)}


def aggregate(rows):
    groups = {'all': rows}
    for dataset in ('pk', 'er'):
        groups[dataset] = [r for r in rows if r['dataset'] == dataset]
        for modality in ('bw', 'color'):
            groups[f'{dataset}/{modality}'] = [r for r in rows if r['dataset'] == dataset and r['modality'] == modality]
    for origin in sorted({r['reference_origin'] for r in rows}):
        groups[f'origin:{origin}'] = [r for r in rows if r['reference_origin'] == origin]
    summary = {}
    for name, selected in groups.items():
        if not selected:
            continue
        block = {}
        for method in METHODS:
            radii = {}
            for key in TOLERANCE_KEYS:
                radii[key] = {identity: pooled(selected, method, key, identity)
                              for identity in ('center_only', 'label_aware')}
                ct = radii[key]['center_only']['tp']
                radii[key]['identity_recovery_ratio'] = radii[key]['label_aware']['tp'] / ct if ct else None
            symbol_rows = [r['models'][method]['symbol'] for r in selected]
            compared = sum(s['compared'] for s in symbol_rows)
            correct = sum(s['correct'] for s in symbol_rows)
            block[method] = {
                'statuses': dict(Counter(r['models'][method]['status'] for r in selected)),
                'predictions': sum(r['models'][method]['prediction_count'] for r in selected),
                'radii': radii,
                'symbol': {'supported': any(s['supported'] for s in symbol_rows), 'series_compared': compared,
                           'series_correct': correct, 'accuracy': correct / compared if compared else None,
                           'reference_series': sum(r['reference_series'] for r in selected)},
            }
        summary[name] = block
    return summary


def confusions(rows):
    table = defaultdict(Counter)
    for row in rows:
        for method in SYMBOL_METHODS:
            symbol = row['models'][method]['symbol']
            for sid, expected in row['symbol_reference'].items():
                got = symbol.get('predicted', {}).get(sid, {}).get('class')
                if expected['class'] and got:
                    table[method][f"{expected['class']} -> {got}"] += 1
    return {method: dict(counter.most_common()) for method, counter in table.items()}


def markdown(summary, rows, primary=PRIMARY):
    def pct(value):
        return '-' if value is None else f'{100*value:.1f}%'
    lines = ['# 20-panel two-dataset benchmark (PK + ER)', '',
             f'Primary tolerance: {primary}. oks_lineex/oks_chartparser reproduce the published OKS criterion '
             '(k=0.025, OKS>0.5, i.e. d <= 0.02943 x image diagonal or x sqrt(w*h)); rel*pct are fractions of the '
             'plot-box diagonal; fixed*px are absolute radii. One-to-one matching, same series.', '',
             '| Group | Method | P | R | F1 | centre-only F1 | symbol acc. | panels |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ['pk', 'er', 'all'] + [k for k in summary if k.startswith('origin:')]:
        if group not in summary:
            continue
        for method in METHODS:
            block = summary[group][method]
            label = block['radii'][primary]['label_aware']
            centre = block['radii'][primary]['center_only']
            symbol = block['symbol']
            sym = 'n/a' if not symbol['supported'] else (
                f"{pct(symbol['accuracy'])} ({symbol['series_correct']}/{symbol['series_compared']})")
            lines.append(f"| {group} | {method} | {pct(label['precision'])} | {pct(label['recall'])} | "
                         f"{pct(label['f1'])} | {pct(centre['f1'])} | {sym} | {label['panels']} |")
    lines += ['', '## Per panel (primary tolerance)', '',
              '| Panel | dataset | modality | ref pts | ' + ' | '.join(METHODS) + ' |',
              '|---|---|---|---:|' + '---:|' * len(METHODS)]
    for row in rows:
        cells = []
        for method in METHODS:
            value = row['models'][method]['metrics'][primary]['label_aware']['f1']
            cells.append('-' if value is None else f'{100*value:.0f}%')
        lines.append(f"| {row['panel_id']} | {row['dataset']} | {row['modality']} | {row['reference_points']} | "
                     + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'inputs.json')
    parser.add_argument('--targets', type=Path, default=HERE / 'targets.json')
    parser.add_argument('--results', type=Path, default=HERE / 'results' / 'run_v1')
    parser.add_argument('--primary', default=PRIMARY)
    args = parser.parse_args()
    panels = read(args.inputs)['panels']
    targets = {t['panel_id']: t for t in read(args.targets)['panels']}
    rows = [evaluate_panel(panel, targets[panel['panel_id']], args.results / 'v46', args.results / 'external')
            for panel in panels]
    summary = aggregate(rows)
    payload = {'primary_tolerance': args.primary, 'methods': METHODS,
               'point_matching': 'one-to-one maximum cardinality then minimum total distance',
               'reference_caveat': 'agreement with a fallible reference; ER chartocode2-derived panels are not '
                                   'independent of v46',
               'symbol_note': 'only v46 predicts a marker class; the published models are reported as unsupported',
               'groups': summary, 'symbol_confusions': confusions(rows)}
    write(args.results / 'evaluation.json', {'summary': payload, 'panels': rows})
    write(args.results / 'metrics_summary.json', payload)
    (args.results / 'summary.md').write_text(markdown(summary, rows, args.primary), encoding='utf-8')
    print(markdown(summary, rows, args.primary))


if __name__ == '__main__':
    main()

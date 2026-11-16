"""Compare the test100v2 run with annotated legend-key boxes (2026-10-25) with the GUI-condition rerun (2026-11-16).

Inputs (all read-only):
  OLD   results/run_test100v2_20261025_0fd678a840bc       B&W v46 got --swatches-json (annotated key boxes), CUDA
  NEW   results/run_test100v2_guilegend_20261116_0fd678a840bc   B&W v46 got the legend rectangle only (GUI), CPU;
                                                           published methods re-run on its legend crops;
                                                           colour charts copied from OLD
  CTRL  results/run_test100v2_bwswatch_cpu_20261116_0fd678a840bc  OLD's B&W command on CPU (v46 only): shows whether
                                                           the CPU backend alone changes anything
Writes NEW/analysis/: summary.json, summary.md, per_chart.csv and three figures.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
OLD = RES / 'run_test100v2_20261025_0fd678a840bc'
NEW = RES / 'run_test100v2_guilegend_20261116_0fd678a840bc'
CTRL = RES / 'run_test100v2_bwswatch_cpu_20261116_0fd678a840bc'
OUT = NEW / 'analysis'
INPUTS = HERE / 'inputs_test100v2_20261025.json'
METHODS = ['v46_before', 'v46_after', 'aicp_raw_grouped', 'aicp_corrected_grouped', 'lineex_paper_grouped',
           'lineex_grouped']
NAMES = {'v46_before': 'v46, before Step 5', 'v46_after': 'v46, after Step 5',
         'aicp_raw_grouped': 'AI-ChartParser, raw', 'aicp_corrected_grouped': 'AI-ChartParser, corrected',
         'lineex_paper_grouped': 'LineEX, paper settings', 'lineex_grouped': 'LineEX, released code'}
COLOURS = {'v46': '#2a78d6', 'aicp': '#eb6834', 'lineex': '#1baf7a'}
GROUPS = [('PK B&W', 'pk', 'bw'), ('ER B&W', 'er', 'bw'), ('all B&W', None, 'bw'), ('all colour', None, 'color'),
          ('all 100', None, None)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def f1(t):
    tp, fp, fn = t
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float('nan')


def walk(value, key_a, key_b, found):
    if isinstance(value, dict):
        if key_a in value and key_b in value:
            found.append((value[key_a], value[key_b]))
        for v in value.values():
            walk(v, key_a, key_b, found)
    elif isinstance(value, list):
        for v in value:
            walk(v, key_a, key_b, found)
    return found


def rescue_cut(folder):
    """Charts whose 60 s layered-rescue budget was hit: scored_centres < candidate_centres anywhere."""
    path = folder / 'raw_before' / 'v46_grid_diagnostics.json'
    if not path.exists():
        return None
    pairs = walk(read(path), 'scored_centres', 'candidate_centres', [])
    return any(s is not None and c is not None and s < c for s, c in pairs)


def points(folder, method):
    path = folder / f'{method}.json'
    if not path.exists():
        return ('missing', [])
    r = read(path)
    return (r.get('status'), sorted((round(p['x_px'], 3), round(p['y_px'], 3), str(p.get('series_id')),
                                     str(p.get('native_series_id'))) for p in r.get('predictions', [])))


def legend_outcome(panel, folder):
    shared = read(folder / 'shared_legends.json') if (folder / 'shared_legends.json').exists() else {}
    before = read(folder / 'v46_before.json') if (folder / 'v46_before.json').exists() else {}
    audit = shared.get('legend_mapping_audit') or {}
    accepted = [p for p in audit.get('pairs', []) if p.get('accepted')]
    row = dict(detection_status=before.get('status'), detection_reason=(before.get('reason') or '')[:160],
               annotated=len(panel['swatches']), v46_entries=len(shared.get('entries', [])),
               source=shared.get('source', 'v46'),
               unmatched_annotations=len(audit.get('unmatched_annotations', [])),
               unmatched_native=len(audit.get('unmatched_native', [])),
               accepted_iou_min=min((p['iou'] for p in accepted), default=None),
               accepted_iou_below_0_5=sum(1 for p in accepted if p['iou'] < 0.5))
    if before.get('status') != 'completed':
        cat = 'detection failed'
    elif row['unmatched_annotations'] and row['unmatched_native']:
        cat = 'missing + extra entries'
    elif row['unmatched_annotations']:
        cat = 'missing entries'
    elif row['unmatched_native']:
        cat = 'extra entries'
    else:
        cat = 'all entries found'
    row['category'] = cat
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = read(INPUTS)['panels']
    bw = [p for p in panels if p['modality'] == 'bw']
    ev_old = {r['panel_id']: r for r in read(OLD / 'entry_evaluation.json')['panels']}
    ev_new = {r['panel_id']: r for r in read(NEW / 'entry_evaluation.json')['panels']}

    # ---- 1. control: does the CPU backend alone reproduce the 2026-10-25 B&W output?
    control = {'charts': len(bw), 'identical_before': 0, 'identical_after': 0, 'identical_legend_entries': 0,
               'differing': []}
    for p in bw:
        pid = p['panel_id']
        same_b = points(CTRL / 'v46' / pid, 'v46_before') == points(OLD / 'v46' / pid, 'v46_before')
        same_a = points(CTRL / 'v46' / pid, 'v46_after') == points(OLD / 'v46' / pid, 'v46_after')
        le = lambda root: [(e.get('bbox'), e.get('series_id'), e.get('class_name'))
                           for e in read(root / 'v46' / pid / 'shared_legends.json').get('entries', [])]
        same_l = le(CTRL) == le(OLD)
        control['identical_before'] += same_b
        control['identical_after'] += same_a
        control['identical_legend_entries'] += same_l
        if not (same_b and same_a and same_l):
            control['differing'].append(dict(panel_id=pid, chart=p['astra_chart_id'], before=same_b, after=same_a,
                                              legend=same_l))

    # ---- 2. rescue-budget check (scored_centres == candidate_centres) for NEW and CTRL
    cuts = {name: [p['astra_chart_id'] for p in bw if rescue_cut(root / 'v46' / p['panel_id'])]
            for name, root in (('new', NEW), ('control', CTRL))}
    unknown = {name: [p['astra_chart_id'] for p in bw if rescue_cut(root / 'v46' / p['panel_id']) is None]
               for name, root in (('new', NEW), ('control', CTRL))}

    # ---- 3. legend discovery per B&W chart in the GUI condition
    per_chart = []
    for p in bw:
        pid = p['panel_id']
        row = dict(chart=p['astra_chart_id'], panel_id=pid, group=f"{p['dataset']}_{p['modality']}",
                   **legend_outcome(p, NEW / 'v46' / pid))
        row['gt_points'] = ev_new[pid]['reference_points']
        row['gt_points_without_entry_new'] = ev_new[pid]['reference_points_without_entry']
        row['gt_points_without_entry_old'] = ev_old[pid]['reference_points_without_entry']
        for m in METHODS:
            o, n = ev_old[pid]['methods'][m], ev_new[pid]['methods'][m]
            row[f'{m}_f1_old'] = f1((o['tp'], o['fp'], o['fn']))
            row[f'{m}_f1_new'] = f1((n['tp'], n['fp'], n['fn']))
            row[f'{m}_tp_fp_fn_old'] = [o['tp'], o['fp'], o['fn']]
            row[f'{m}_tp_fp_fn_new'] = [n['tp'], n['fp'], n['fn']]
        per_chart.append(row)
    categories = {}
    for g in ('pk_bw', 'er_bw'):
        for r in per_chart:
            if r['group'] == g:
                categories.setdefault(g, {}).setdefault(r['category'], 0)
                categories[g][r['category']] += 1

    # ---- 4. pooled F1 per group, old vs new, with paired bootstrap of the difference (2000, seed 0)
    rng = np.random.default_rng(0)
    pooled = {}
    for label, dataset, modality in GROUPS:
        sel = [p['panel_id'] for p in panels if (dataset is None or p['dataset'] == dataset)
               and (modality is None or p['modality'] == modality)]
        idx = rng.integers(0, len(sel), (2000, len(sel)))
        block = {'charts': len(sel)}
        for m in METHODS:
            to = np.array([[ev_old[i]['methods'][m][k] for k in ('tp', 'fp', 'fn')] for i in sel], float)
            tn = np.array([[ev_new[i]['methods'][m][k] for k in ('tp', 'fp', 'fn')] for i in sel], float)
            bo = np.array([f1(to[b].sum(0)) for b in idx])
            bn = np.array([f1(tn[b].sum(0)) for b in idx])
            block[m] = dict(f1_old=f1(to.sum(0)), f1_new=f1(tn.sum(0)), ci_old=np.nanpercentile(bo, [2.5, 97.5]).tolist(),
                            ci_new=np.nanpercentile(bn, [2.5, 97.5]).tolist(),
                            diff=f1(tn.sum(0)) - f1(to.sum(0)), diff_ci=np.nanpercentile(bn - bo, [2.5, 97.5]).tolist(),
                            tp_fp_fn_old=to.sum(0).astype(int).tolist(), tp_fp_fn_new=tn.sum(0).astype(int).tolist(),
                            charts_lower=int(sum(1 for i in sel if f1n(ev_new, i, m) < f1n(ev_old, i, m) - 1e-12)),
                            charts_higher=int(sum(1 for i in sel if f1n(ev_new, i, m) > f1n(ev_old, i, m) + 1e-12)))
        pooled[label] = block

    summary = dict(control=control, rescue_cut=cuts, rescue_fields_missing=unknown, legend_categories=categories,
                   pooled=pooled,
                   gt_points_without_entry=dict(
                       old=sum(r['gt_points_without_entry_old'] for r in per_chart),
                       new=sum(r['gt_points_without_entry_new'] for r in per_chart),
                       bw_gt_points=sum(r['gt_points'] for r in per_chart)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (OUT / 'per_chart.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_chart[0].keys()))
        w.writeheader()
        for r in sorted(per_chart, key=lambda r: r['chart']):
            w.writerow(r)
    figures(per_chart, pooled, categories)
    print(json.dumps({k: v for k, v in summary.items() if k != 'pooled'}, indent=1, ensure_ascii=False))
    for label, block in pooled.items():
        print(label, block['charts'], {m: (round(100 * block[m]['f1_old'], 1), round(100 * block[m]['f1_new'], 1),
                                           [round(100 * v, 1) for v in block[m]['diff_ci']]) for m in METHODS})


def f1n(ev, pid, m):
    r = ev[pid]['methods'][m]
    v = f1((r['tp'], r['fp'], r['fn']))
    return 0.0 if math.isnan(v) else v


def family(m):
    return 'v46' if m.startswith('v46') else 'aicp' if m.startswith('aicp') else 'lineex'


def figures(per_chart, pooled, categories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    # Figure A: legend-entry discovery outcome in the GUI condition, per group (counts of charts)
    order = ['all entries found', 'extra entries', 'missing entries', 'missing + extra entries', 'detection failed']
    shades = ['#2a78d6', '#9ec5f0', '#eb6834', '#b0412a', '#5c5c5c']
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    labelled = set()
    for row, g in enumerate(('pk_bw', 'er_bw')):
        left = 0
        for cat, colour in zip(order, shades):
            n = categories.get(g, {}).get(cat, 0)
            if n:
                ax.barh(row, n, left=left, color=colour, edgecolor='white', linewidth=2, height=0.6,
                        label=None if cat in labelled else cat)
                labelled.add(cat)
                ax.text(left + n / 2, row, str(n), ha='center', va='center', color='white' if colour not in ('#9ec5f0',) else '#1a1a1a', fontsize=9)
                left += n
    ax.set_yticks([0, 1], ['PK B&W\n(25 charts)', 'ER B&W\n(25 charts)'])
    ax.invert_yaxis()
    ax.set_xlim(0, 25)
    ax.set_xlabel('number of charts')
    ax.set_title('Legend entries found by v46 from the legend rectangle alone (GUI condition), test100v2 B&W', fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(OUT / 'figA_legend_discovery.png', dpi=200)
    plt.close(fig)

    # Figure B: pooled F1 per method, annotated key boxes vs GUI condition, with 95 % intervals
    labels = ['PK B&W', 'ER B&W', 'all B&W', 'all 100']
    fig, axes = plt.subplots(1, len(labels), figsize=(10, 3.6), sharey=True)
    for ax, label in zip(axes, labels):
        for k, m in enumerate(METHODS):
            b = pooled[label][m]
            c = COLOURS[family(m)]
            y = len(METHODS) - 1 - k
            ax.plot([100 * b['f1_old'], 100 * b['f1_new']], [y, y], color=c, linewidth=1.2, alpha=0.6)
            ax.errorbar(100 * b['f1_old'], y + 0.15, xerr=[[100 * (b['f1_old'] - b['ci_old'][0])], [100 * (b['ci_old'][1] - b['f1_old'])]],
                        fmt='o', mfc='white', mec=c, ecolor=c, elinewidth=0.8, capsize=0, markersize=6)
            ax.errorbar(100 * b['f1_new'], y - 0.15, xerr=[[100 * (b['f1_new'] - b['ci_new'][0])], [100 * (b['ci_new'][1] - b['f1_new'])]],
                        fmt='o', color=c, ecolor=c, elinewidth=0.8, capsize=0, markersize=6)
        ax.set_title(f"{label} ({pooled[label]['charts']} charts)", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel('pooled F1 (%)')
        ax.grid(axis='x', color='#e6e6e6', linewidth=0.6)
    axes[0].set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS][::-1])
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker='o', mfc='white', mec='#555', ls='', ms=6), Line2D([], [], marker='o', color='#555', ls='', ms=6)],
               ['with annotated legend-key boxes (2026-10-25 run)', 'legend rectangle only = GUI condition (2026-11-16 rerun)'],
               loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.suptitle('test100v2: pooled F1 by method, B&W legend input changed; colour charts identical in both runs', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / 'figB_pooled_f1_old_vs_gui.png', dpi=200)
    plt.close(fig)

    # Figure C: per-chart v46 F1 after Step 5, old (x) vs GUI condition (y), marked by legend outcome
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    marks = {'all entries found': ('o', '#2a78d6'), 'extra entries': ('s', '#9ec5f0'), 'missing entries': ('^', '#eb6834'),
             'missing + extra entries': ('D', '#b0412a'), 'detection failed': ('X', '#5c5c5c')}
    for cat, (mk, c) in marks.items():
        rows = [r for r in per_chart if r['category'] == cat]
        if rows:
            ax.scatter([100 * r['v46_after_f1_old'] for r in rows], [100 * r['v46_after_f1_new'] for r in rows],
                       marker=mk, s=34, color=c, edgecolor='white', linewidth=0.6, label=f'{cat} ({len(rows)})', zorder=3)
    ax.plot([0, 100], [0, 100], color='#999', linewidth=0.8, zorder=1)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect('equal')
    ax.set_xlabel('v46 F1 after Step 5 (%), with annotated key boxes')
    ax.set_ylabel('v46 F1 after Step 5 (%), legend rectangle only (GUI)')
    lower = sum(1 for r in per_chart if r['v46_after_f1_new'] < r['v46_after_f1_old'] - 1e-12)
    higher = sum(1 for r in per_chart if r['v46_after_f1_new'] > r['v46_after_f1_old'] + 1e-12)
    ax.text(98, 4, f'lower in GUI condition: {lower} of 50\nhigher: {higher}, equal: {50 - lower - higher}', ha='right', va='bottom', fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Per chart, 50 B&W test charts (one point per chart)', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figC_per_chart_v46_after.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

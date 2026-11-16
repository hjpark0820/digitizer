"""How much do the published methods depend on v46's legend crops? Compare three legend conditions (test100v2).

  A  results/run_test100v2_20261025_0fd678a840bc            crops exported by v46 (B&W v46 seeded with annotated boxes)
  B  results/run_test100v2_guilegend_20261116_0fd678a840bc   crops exported by v46, GUI condition (legend rectangle only)
  C  results/run_test100v2_gtswatch_external_20261116        crops cut at the annotated (ground-truth) key boxes
Same detectors, weights, matcher, device and scorer in all three; only the legend crops differ.
Writes C/analysis/: summary.json, summary.md, figD_comparators_by_legend_source.png, figE_missed_series_tp.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_entry as E  # noqa: E402

RES = HERE / 'results'
COND = {'A': RES / 'run_test100v2_20261025_0fd678a840bc', 'B': RES / 'run_test100v2_guilegend_20261116_0fd678a840bc',
        'C': RES / 'run_test100v2_gtswatch_external_20261116'}
LABEL = {'A': 'v46 crops, 2026-10-25 (B&W keys seeded by annotation)', 'B': 'v46 crops, GUI condition',
         'C': 'ground-truth key crops'}
EXT = ['aicp_raw_grouped', 'aicp_corrected_grouped', 'lineex_paper_grouped', 'lineex_grouped']
NAMES = {'aicp_raw_grouped': 'AI-ChartParser, raw', 'aicp_corrected_grouped': 'AI-ChartParser, corrected',
         'lineex_paper_grouped': 'LineEX, paper settings', 'lineex_grouped': 'LineEX, released code',
         'v46_after': 'v46, after Step 5'}
OUT = COND['C'] / 'analysis'


def f1(t):
    tp, fp, fn = t
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float('nan')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = E.EV.read(HERE / 'inputs_test100v2_20261025.json')['panels']
    targets = {t['panel_id']: t for t in E.EV.read(HERE / 'targets_test100v2_20261025.json')['panels']}
    ev = {k: {r['panel_id']: r for r in E.EV.read(v / 'entry_evaluation.json')['panels']} for k, v in COND.items()}

    # series whose key v46 did not export in each v46-crop condition (and so no method could match them there)
    missed = {}
    for k in ('A', 'B'):
        for p in panels:
            entries = E.entries_of(COND[k] / 'v46' / p['panel_id'])
            ref = E.reference_entries(p['panel_id'], entries, {})
            for s in {q['series_id'] for q in targets[p['panel_id']]['points']}:
                if ref.get(s) is None:
                    missed.setdefault(k, {}).setdefault(p['panel_id'], set()).add(s)

    # TP of each method on those series, per condition: TP(all) - TP(series removed); exact, because label-aware
    # matching only pairs points of the same legend entry
    on_missed = {}
    for base in ('A', 'B'):
        for k in COND:
            row = {m: 0 for m in EXT + ['v46_after']}
            npts = 0
            for pid, series in missed.get(base, {}).items():
                p = next(x for x in panels if x['panel_id'] == pid)
                t = targets[pid]
                npts += sum(1 for q in t['points'] if q['series_id'] in series)
                full = E.score_panel(p, t, COND[k], {})
                rest = E.score_panel(p, dict(t, points=[q for q in t['points'] if q['series_id'] not in series]), COND[k], {})
                for m in row:
                    row[m] += full['methods'][m]['tp'] - rest['methods'][m]['tp']
            on_missed[f'series missed in {base}, scored in {k}'] = dict(points=npts, tp=row)

    groups = [('B&W', None, 'bw'), ('colour', None, 'color'), ('PK B&W', 'pk', 'bw'), ('ER B&W', 'er', 'bw'),
              ('PK colour', 'pk', 'color'), ('ER colour', 'er', 'color'), ('all 100', None, None)]
    rng = np.random.default_rng(0)
    pooled = {}
    for label, dataset, modality in groups:
        sel = [p['panel_id'] for p in panels if (dataset is None or p['dataset'] == dataset)
               and (modality is None or p['modality'] == modality)]
        idx = rng.integers(0, len(sel), (2000, len(sel)))
        block = {'charts': len(sel)}
        for m in EXT + ['v46_after']:
            tab = {k: np.array([[ev[k][i]['methods'][m][x] for x in ('tp', 'fp', 'fn')] for i in sel], float) for k in COND}
            boot = {k: np.array([f1(tab[k][b].sum(0)) for b in idx]) for k in COND}
            block[m] = {k: dict(f1=f1(tab[k].sum(0)), ci=np.nanpercentile(boot[k], [2.5, 97.5]).tolist(),
                                tp_fp_fn=tab[k].sum(0).astype(int).tolist()) for k in COND}
            for a, b in (('C', 'A'), ('C', 'B')):
                block[m][f'{a}-{b}'] = dict(diff=f1(tab[a].sum(0)) - f1(tab[b].sum(0)),
                                            ci=np.nanpercentile(boot[a] - boot[b], [2.5, 97.5]).tolist())
        pooled[label] = block

    # consistency: t1_pk_bw_15 used annotation crops in B as well, so B and C outputs must agree there
    pid = 'pk__panitumumab2__p02__v5'
    same = {m: ev['B'][pid]['methods'][m] == ev['C'][pid]['methods'][m] for m in EXT}
    statuses = {k: sorted({ev[k][p['panel_id']]['methods'][m]['status'] for p in panels for m in EXT}) for k in COND}
    summary = dict(conditions=LABEL, missed_keys={k: sum(len(v) for v in missed.get(k, {}).values()) for k in ('A', 'B')},
                   tp_on_missed_series=on_missed, pooled=pooled, consistency_t1_pk_bw_15_B_equals_C=same,
                   method_statuses=statuses)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['| method | group | ' + ' | '.join(f'{k}: {LABEL[k]}' for k in COND) + ' | C-B (95% CI) | C-A (95% CI) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for label in ('B&W', 'colour', 'all 100'):
        for m in EXT:
            b = pooled[label][m]
            lines.append(f"| {NAMES[m]} | {label} | " + ' | '.join(f"{100*b[k]['f1']:.1f}" for k in COND) +
                         f" | {100*b['C-B']['diff']:+.1f} ({100*b['C-B']['ci'][0]:+.1f}..{100*b['C-B']['ci'][1]:+.1f})"
                         f" | {100*b['C-A']['diff']:+.1f} ({100*b['C-A']['ci'][0]:+.1f}..{100*b['C-A']['ci'][1]:+.1f}) |")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    figures(pooled, on_missed)
    print('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('pooled', 'conditions')}, indent=1))


def figures(pooled, on_missed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colours = {'aicp_raw_grouped': '#eb6834', 'aicp_corrected_grouped': '#eb6834', 'lineex_paper_grouped': '#1baf7a',
               'lineex_grouped': '#1baf7a', 'v46_after': '#2a78d6'}
    marks = {'A': ('o', 'white'), 'B': ('s', 'white'), 'C': ('D', None)}
    groups = ['B&W', 'colour', 'all 100']
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    methods = EXT
    for ax, g in zip(axes, groups):
        for r, m in enumerate(methods):
            y = len(methods) - 1 - r
            for j, k in enumerate(COND):
                b = pooled[g][m][k]
                mk, face = marks[k]
                ax.errorbar(100 * b['f1'], y + (0.22 - 0.22 * j), xerr=[[100 * (b['f1'] - b['ci'][0])], [100 * (b['ci'][1] - b['f1'])]],
                            fmt=mk, ms=5.5, color=colours[m], mfc=face or colours[m], mec=colours[m], elinewidth=0.8)
        v = pooled[g]['v46_after']
        ax.axvline(100 * v['B']['f1'], color='#2a78d6', lw=1, ls='--')
        ax.text(100 * v['B']['f1'] - 1.5, len(methods) - 0.35, f"v46 {100*v['B']['f1']:.1f}", color='#2a78d6', ha='right', va='center', fontsize=8)
        ax.set_ylim(-0.5, len(methods) - 0.1)
        ax.set_xlim(0, 100)
        ax.set_title(f"{g} ({pooled[g]['charts']} charts)", fontsize=9)
        ax.set_xlabel('pooled F1 (%)')
        ax.grid(axis='x', color='#e6e6e6', lw=0.6)
    axes[0].set_yticks(range(len(methods)), [NAMES[m] for m in methods][::-1])
    fig.legend([Line2D([], [], marker=marks[k][0], ls='', mfc=marks[k][1] or '#555', mec='#555', color='#555', ms=6) for k in COND],
               [f'{k}: {LABEL[k]}' for k in COND], loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Published methods on test100v2 with three sources of legend crops (dashed: v46 after Step 5, GUI condition)', fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUT / 'figD_comparators_by_legend_source.png', dpi=200)
    plt.close(fig)

    key = 'series missed in B, scored in '
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ms = EXT + ['v46_after']
    width = 0.26
    for j, k in enumerate(COND):
        # v46 is not run in condition C (its own legend stage is what C replaces)
        keep = [i for i, m in enumerate(ms) if not (k == 'C' and m == 'v46_after')]
        xs = np.arange(len(ms))[keep] + (j - 1) * width
        vals = [on_missed[key + k]['tp'][ms[i]] for i in keep]
        ax.bar(xs, vals, width, color=[colours[ms[i]] for i in keep], alpha=[0.35, 0.65, 1.0][j],
               edgecolor='white', linewidth=1.5, label=f'{k}: {LABEL[k]}')
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.8, str(v), ha='center', fontsize=7.5)
    n = on_missed[key + 'B']['points']
    ax.set_xticks(range(len(ms)), [NAMES[m].replace(', ', '\n') for m in ms])
    ax.set_ylabel(f'true positives (of {n} reference points)')
    ax.set_title(f'Points of the series whose key v46 missed in the GUI condition ({n} points): true positives by legend source', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_ylim(0, 38)
    fig.tight_layout()
    fig.savefig(OUT / 'figE_missed_series_tp.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

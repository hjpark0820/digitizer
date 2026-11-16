"""Figures and tables for the test100v2 benchmark, GUI condition (100 synthetic PK/ER charts, snapshot 0fd678a840bc).

Main result: every method received the plot area and the legend rectangle only; v46 located the legend keys itself
(B&W and colour); the published methods received the legend-key crops exported by v46.
Reads only files inside this package:
  data/results/entry_evaluation.json      per-panel TP / FP / FN of every method (written by evaluate_entry.py)
  data/benchmark/*.json, data/images/     for the example overlays (Figure 5) and chart ids
  data/results/v46, data/results/external predictions, re-matched here with evaluate_entry.py's own functions
                                          to obtain point-level matches; the re-matched TP / FP / FN are asserted
                                          equal to entry_evaluation.json before anything is drawn.
  data/sensitivity/*/entry_evaluation.json   legend-input sensitivity analyses (Table 4, Figure 6)
Writes figures/ (PNG 300 dpi + PDF) and tables/ (CSV + Markdown; LaTeX for the three paper tables).

Definitions (same as evaluate_entry.py): a predicted point is a true positive when it lies within the OKS
tolerance radius (0.0294 x image diagonal) of a ground-truth point AND is assigned to that point's legend
entry; matching is one-to-one. Pooled F1 = 2TP / (2TP + FP + FN) over the panels of a group.
95 % intervals: single-method intervals are copied from entry_evaluation.json (2000 panel resamples);
paired differences use 2000 paired panel resamples drawn here (seed 0): both methods are scored on the same
resampled panels, so the interval reflects the difference, not two independent intervals.

usage: python make_figures.py      (from any directory)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

PKG = Path(__file__).resolve().parents[2]
B2 = PKG / 'code' / 'experiments' / 'published_chart_models' / 'benchmark_v2'
DATA = PKG / 'data'
FIG = PKG / 'figures'
TAB = PKG / 'tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location('evaluate_entry_pkg', B2 / 'evaluate_entry.py')
EE = importlib.util.module_from_spec(spec)
sys.modules['evaluate_entry_pkg'] = EE
spec.loader.exec_module(EE)
EV = EE.EV
assert Path(EV.evaluate_points.__code__.co_filename).resolve().is_relative_to(PKG), 'metrics not loaded from package'

# ---------------------------------------------------------------- style
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
HUE = {'v46': '#2a78d6', 'aicp': '#eb6834', 'lineex': '#1baf7a'}          # categorical slots 1-3
METHODS = [  # key, label, family, filled
    ('v46_after', 'v46, after Step 5', 'v46', True),
    ('v46_before', 'v46, before Step 5', 'v46', False),
    ('aicp_raw_grouped', 'AI-ChartParser, raw keypoints', 'aicp', True),
    ('aicp_corrected_grouped', 'AI-ChartParser, corrected keypoints', 'aicp', False),
    ('lineex_paper_grouped', 'LineEX, paper settings', 'lineex', True),
    ('lineex_grouped', 'LineEX, released-code settings', 'lineex', False),
]
LABEL = {m[0]: m[1] for m in METHODS}
GROUPS = [  # summary key in entry_evaluation.json, English name, dataset, modality
    ('PK 흑백', 'PK, black & white', 'pk', 'bw'), ('PK 컬러', 'PK, colour', 'pk', 'color'),
    ('ER 흑백', 'ER, black & white', 'er', 'bw'), ('ER 컬러', 'ER, colour', 'er', 'color'),
    ('전체', 'All 100 charts', None, None),
]
SHORT = {'PK 흑백': 'PK, B&W', 'PK 컬러': 'PK, colour', 'ER 흑백': 'ER, B&W', 'ER 컬러': 'ER, colour', '전체': 'All charts'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'savefig.dpi': 300})


CAPTIONS = []


def caption(text, **_unused):
    """Figure titles go to figures/figure_captions.md (journal captions), not into the image."""
    CAPTIONS.append(text)


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def marker_kw(family, filled, size=5.5):
    c = HUE[family]
    return dict(marker='o', ms=size, mec=c, mew=1.2, mfc=c if filled else 'white', color=c, linestyle='none')


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float('nan')


# ---------------------------------------------------------------- data
ev = json.loads((DATA / 'results' / 'entry_evaluation.json').read_text(encoding='utf-8'))
rows = ev['panels']
chart_id = json.loads((DATA / 'benchmark' / 'chart_id_map_test100v2_20261025.json').read_text(encoding='utf-8'))
inputs = {p['panel_id']: p for p in EV.read(DATA / 'benchmark' / 'inputs_test100v2_20261025.json')['panels']}
targets = {t['panel_id']: t for t in EV.read(DATA / 'benchmark' / 'targets_test100v2_20261025.json')['panels']}


def in_group(r, ds, mo):
    return (ds is None or r['dataset'] == ds) and (mo is None or r['modality'] == mo)


def counts(sel, m):
    return np.array([[r['methods'][m][k] for k in ('tp', 'fp', 'fn')] for r in sel], float)


def panel_f1(r, m):
    v = r['methods'][m]
    return f1(v['tp'], v['fp'], v['fn'])


# ---------------------------------------------------------------- tables
t1 = []
for key, name, ds, mo in GROUPS:
    block = ev['summary'][key]
    for m, label, _, _ in METHODS:
        s = block['methods'][m]
        tp, fp, fn = s['tp_fp_fn']
        t1.append(dict(group=name, panels=block['panels'], gt_points=block['reference_points'], method=label,
                       f1_pct=round(100 * s['f1'], 1), ci95_low_pct=round(100 * s['ci95'][0], 1),
                       ci95_high_pct=round(100 * s['ci95'][1], 1), tp=tp, fp=fp, fn=fn,
                       precision_pct=round(100 * tp / (tp + fp), 1) if tp + fp else '',
                       recall_pct=round(100 * tp / (tp + fn), 1)))

rng = np.random.default_rng(0)
t2 = []
paired = {}
for key, name, ds, mo in GROUPS:
    sel = [r for r in rows if in_group(r, ds, mo)]
    idx = rng.integers(0, len(sel), (2000, len(sel)))
    A = counts(sel, 'v46_after')
    for m, label, _, _ in METHODS[1:]:
        Bm = counts(sel, m)
        point = f1(*A.sum(0)) - f1(*Bm.sum(0))
        boot = np.array([f1(*A[b].sum(0)) - f1(*Bm[b].sum(0)) for b in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pa = np.array([panel_f1(r, 'v46_after') for r in sel])
        pb = np.array([panel_f1(r, m) for r in sel])
        paired[(name, m)] = (point, lo, hi)
        t2.append(dict(group=name, panels=len(sel), comparison=f'v46, after Step 5  minus  {label}',
                       delta_f1_pct_points=round(100 * point, 1), ci95_low=round(100 * lo, 1), ci95_high=round(100 * hi, 1),
                       panels_v46_after_higher=int((pa > pb + 1e-12).sum()), panels_lower=int((pa < pb - 1e-12).sum()),
                       panels_equal=int((np.abs(pa - pb) <= 1e-12).sum())))

t3 = []
for r in rows:
    row = dict(chart_id=chart_id[r['panel_id']], image=r['panel_id'] + '.png', dataset=r['dataset'].upper(),
               modality='B&W' if r['modality'] == 'bw' else 'colour', gt_points=r['reference_points'],
               gt_points_without_detected_legend_entry=r['reference_points_without_entry'])
    for m, label, _, _ in METHODS:
        v = r['methods'][m]
        row.update({f'{m}_status': v['status'], f'{m}_tp': v['tp'], f'{m}_fp': v['fp'], f'{m}_fn': v['fn'],
                    f'{m}_f1_pct': round(100 * panel_f1(r, m), 1)})
    t3.append(row)
t3.sort(key=lambda d: d['chart_id'])


def write_table(name, recs, title, note):
    with open(TAB / f'{name}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)
    cols = list(recs[0])
    md = [f'# {title}', '', note, '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in recs]
    (TAB / f'{name}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


write_table('table1_f1_precision_recall', t1, 'Pooled F1, precision and recall per method and chart group',
            'F1 / precision / recall in % over all ground-truth points of the group (pooled TP, FP, FN). '
            '95 % CI: 2000 panel resamples (from entry_evaluation.json). 3,092 ground-truth points in 100 charts.')
write_table('table2_paired_differences', t2, 'Paired differences: v46 after Step 5 against every other method',
            'Delta F1 in percentage points (pooled F1 of v46 after Step 5 minus pooled F1 of the other method). '
            '95 % CI: 2000 paired panel resamples (seed 0). Panel counts compare per-chart F1.')
write_table('table3_per_chart_scores', t3, 'Per-chart TP / FP / FN and F1 of every method',
            'One row per chart. status "failed" = the method produced no output for that chart '
            '(scored as 0 TP, all ground-truth points FN).')

# ---------------------------------------------------------------- Figure 1: forest plot of pooled F1
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.6), sharey=True)
ypos = np.arange(len(METHODS))[::-1]
for ax, (key, name, ds, mo) in zip(axes, GROUPS):
    block = ev['summary'][key]
    for y, (m, label, fam, filled) in zip(ypos, METHODS):
        s = block['methods'][m]
        ax.plot([100 * s['ci95'][0], 100 * s['ci95'][1]], [y, y], color=HUE[fam], lw=1.2, solid_capstyle='round')
        ax.plot(100 * s['f1'], y, **marker_kw(fam, filled))
        ax.text(max(100 * s['ci95'][1], 100 * s['f1']) + 4, y, f"{100 * s['f1']:.1f}", va='center', ha='left',
                fontsize=6.3, color=INK)
    ax.set_xlim(0, 122); ax.set_xticks([0, 50, 100]); ax.spines['bottom'].set_bounds(0, 100)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{SHORT[key]}\n{block['panels']} charts\n{block['reference_points']:,} points", fontsize=7.2, color=INK)
    ax.set_xlabel('Pooled F1 (%)')
axes[0].set_yticks(ypos); axes[0].set_yticklabels([m[1] for m in METHODS])
axes[0].set_ylim(-0.7, len(METHODS) - 0.3)
caption('Figure 1. Point extraction accuracy on the 100-chart test set (dot = pooled F1, line = 95 % CI)',
             fontsize=8.5, color=INK, x=0.0, ha='left', y=1.06)
fig.subplots_adjust(wspace=0.12)
save(fig, 'fig1_pooled_f1_by_group')

# ---------------------------------------------------------------- Figure 2: precision vs recall
fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.95), sharey=True)
gr = np.linspace(0.01, 1, 200)
for ax, (key, name, ds, mo) in zip(axes, GROUPS):
    for lvl in (0.2, 0.4, 0.6, 0.8):
        p = lvl * gr / (2 * gr - lvl)
        ok = (p > 0) & (p <= 1)
        ax.plot(100 * gr[ok], 100 * p[ok], color=GRID, lw=0.7, zorder=0)
        ax.text(100 * 0.985, 100 * lvl * 0.985 / (2 * 0.985 - lvl), f'F1 {lvl:.1f}', fontsize=5.5, color=INK2,
                ha='right', va='bottom')
    for m, label, fam, filled in METHODS:
        tp, fp, fn = ev['summary'][key]['methods'][m]['tp_fp_fn']
        ax.plot(100 * tp / (tp + fn), 100 * tp / (tp + fp), **marker_kw(fam, filled, 5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
    ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(name, fontsize=7.5, color=INK); ax.set_xlabel('Recall (%)')
axes[0].set_ylabel('Precision (%)')
handles = [Line2D([], [], **marker_kw(fam, filled, 5), label=label) for _, label, fam, filled in METHODS]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.32), fontsize=6.8)
caption('Figure 2. Precision and recall (pooled over the charts of each group; grey curves: equal F1)',
             fontsize=8.5, color=INK, x=0.0, ha='left', y=1.08)
save(fig, 'fig2_precision_recall')

# ---------------------------------------------------------------- shared per-panel encoding
SHAPE = {'pk': 'o', 'er': '^'}


def panel_kw(r, size=4):
    filled = r['modality'] == 'bw'
    return dict(marker=SHAPE[r['dataset']], ms=size, mec=INK2, mew=0.7, mfc=INK2 if filled else 'white',
                linestyle='none', alpha=0.85)


panel_handles = [Line2D([], [], marker='o', ms=4, mec=INK2, mew=0.7, mfc=INK2, linestyle='none', label='PK, B&W'),
                 Line2D([], [], marker='o', ms=4, mec=INK2, mew=0.7, mfc='white', linestyle='none', label='PK, colour'),
                 Line2D([], [], marker='^', ms=4, mec=INK2, mew=0.7, mfc=INK2, linestyle='none', label='ER, B&W'),
                 Line2D([], [], marker='^', ms=4, mec=INK2, mew=0.7, mfc='white', linestyle='none', label='ER, colour')]

# ---------------------------------------------------------------- Figure 3: per-chart paired comparison
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), sharey=True)
for ax, (m, label, fam, filled) in zip(axes, METHODS[2:]):
    for r in rows:
        ax.plot(100 * panel_f1(r, m), 100 * panel_f1(r, 'v46_after'), **panel_kw(r))
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.7, ls='--', zorder=0)
    rec = next(t for t in t2 if t['group'] == 'All 100 charts' and t['comparison'].endswith(label))
    ax.text(99, 1, f"v46 higher on" + chr(10) + f"{rec['panels_v46_after_higher']} of 100 charts" + chr(10) +
            f"lower {rec['panels_lower']}, equal {rec['panels_equal']}", fontsize=6.2, color=INK, va='bottom', ha='right')
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.set_aspect('equal')
    ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    name, variant = label.split(', ')
    ax.set_xlabel(f'{name}\n{variant}\nper-chart F1 (%)', fontsize=6.8)
axes[0].set_ylabel('v46, after Step 5\nper-chart F1 (%)')
fig.legend(handles=panel_handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=6.8)
caption('Figure 3. Per-chart F1, v46 against each published method (one marker = one chart; above the dashed '
             'line = v46 better)', fontsize=8.5, color=INK, x=0.0, ha='left', y=1.04)
save(fig, 'fig3_per_chart_paired')

# ---------------------------------------------------------------- Figure 4: Step 5 effect
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
failed = [r for r in rows if r['methods']['v46_after']['status'] != 'completed']
for r in rows:
    a.plot(100 * panel_f1(r, 'v46_before'), 100 * panel_f1(r, 'v46_after'), **panel_kw(r))
a.plot([0, 100], [0, 100], color=INK2, lw=0.7, ls='--', zorder=0)
for r in failed:
    x, y = 100 * panel_f1(r, 'v46_before'), 100 * panel_f1(r, 'v46_after')
    if r['methods']['v46_before']['status'] != 'completed':
        text, offset = '\ndetection failed\n(no legend key found),\nscored as 0', (58, 30)
    else:
        text, offset = ' Step 5 failed,\nscored as 0', (8, 12)
    a.annotate(f"{chart_id[r['panel_id']]}:{text}", (x, y), (x + offset[0], y + offset[1]), fontsize=6,
               color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
pb = np.array([panel_f1(r, 'v46_before') for r in rows]); pa = np.array([panel_f1(r, 'v46_after') for r in rows])
a.text(3, 97, f'higher after Step 5: {(pa > pb + 1e-12).sum()} of 100\nlower: {(pa < pb - 1e-12).sum()}, '
              f'unchanged: {(np.abs(pa - pb) <= 1e-12).sum()}', fontsize=6.2, color=INK, va='top')
a.set_xlim(-2, 102); a.set_ylim(-2, 102); a.set_aspect('equal')
a.set_xlabel('Before Step 5, per-chart F1 (%)'); a.set_ylabel('After Step 5, per-chart F1 (%)')
a.set_title('a  Per chart', loc='left', fontsize=7.5, color=INK)
a.legend(handles=panel_handles, loc='upper left', bbox_to_anchor=(0.0, 0.82), frameon=False, fontsize=6)
for i, (key, name, ds, mo) in enumerate(GROUPS):
    y = len(GROUPS) - 1 - i
    point, lo, hi = paired[(name, 'v46_before')]
    b.plot([100 * lo, 100 * hi], [y, y], color=HUE['v46'], lw=1.2)
    b.plot(100 * point, y, **marker_kw('v46', True))
    tp0, fp0, fn0 = ev['summary'][key]['methods']['v46_before']['tp_fp_fn']
    tp1, fp1, fn1 = ev['summary'][key]['methods']['v46_after']['tp_fp_fn']
    b.text(5.3, y, f'{100 * point:+.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]   TP {tp1 - tp0:+d}, FP {fp1 - fp0:+d}',
           va='center', fontsize=6.2, color=INK)
b.axvline(0, color=INK2, lw=0.7, ls='--')
b.set_yticks(range(len(GROUPS))[::-1]); b.set_yticklabels([g[1] for g in GROUPS])
b.set_xlim(-3, 13.5); b.set_xticks([-2, 0, 2, 4]); b.spines['bottom'].set_bounds(-3, 4.5)
b.set_xlabel('Change in pooled F1 (percentage points)')
b.set_title('b  Pooled, with paired 95 % CI', loc='left', fontsize=7.5, color=INK)
for ax in (a, b):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
caption('Figure 4. Effect of the Step 5 correction on v46', fontsize=8.5, color=INK, x=0.0, ha='left', y=1.02)
fig.subplots_adjust(wspace=0.55)
save(fig, 'fig4_step5_effect')


# ---------------------------------------------------------------- Figure 5: example overlays
def matched(panel_id, method):
    """evaluate_entry.score_panel with the point-level matches kept (same rules, same functions)."""
    panel, target = inputs[panel_id], targets[panel_id]
    reference = target['points']
    v46_folder = DATA / 'results' / 'v46' / panel_id
    entries = EE.entries_of(v46_folder)
    ref_entry = EE.reference_entries(panel_id, entries, {})
    single = len({p['series_id'] for p in reference}) == 1
    radius = EV.tolerances(panel)[EE.TOL]
    folder = v46_folder if method.startswith('v46') else DATA / 'results' / 'external' / panel_id
    preds = EV.method_predictions(folder, panel, method)['predictions']
    if single:
        res = EV.evaluate_points(preds, reference, tolerance=radius, label_aware=False)
    else:
        p2 = [dict(p, series_id=EE.prediction_entry(method, p, entries)) for p in preds]
        r2 = [dict(p, series_id=ref_entry.get(p['series_id'])) for p in reference]
        res = EV.evaluate_points(p2, r2, tolerance=radius, label_aware=True)
    row = next(r for r in rows if r['panel_id'] == panel_id)['methods'][method]
    assert (res['tp'], res['fp'], res['fn']) == (row['tp'], row['fp'], row['fn']), (panel_id, method)
    return preds, reference, res, radius


examples = []
for key, name, ds, mo in GROUPS[:4]:
    sel = sorted((r for r in rows if in_group(r, ds, mo)), key=lambda r: (panel_f1(r, 'v46_after'), r['panel_id']))
    examples.append((name, sel[len(sel) // 2]))
SHOW = [('v46_after', 'v46, after Step 5'), ('aicp_raw_grouped', 'AI-ChartParser, raw keypoints')]
fig, axes = plt.subplots(4, 2, figsize=(7.2, 10.2))
for (name, r), row_axes in zip(examples, axes):
    panel = inputs[r['panel_id']]
    img = np.asarray(Image.open(DATA / 'images' / (r['panel_id'] + '.png')).convert('RGB'), float)
    img = 255 - 0.5 * (255 - img)                      # lighten so the overlay reads
    # crop to the scored plot area (+ a margin for tick labels); the legend and neighbouring panels of a
    # multi-panel figure are left out so the scored panel stays legible
    px0, py0, px1, py1 = panel['plot_bbox']
    pad = 0.06 * max(px1 - px0, py1 - py0) + 10
    x0 = max(0, px0 - pad); y0 = max(0, py0 - pad)
    x1 = min(img.shape[1], px1 + pad); y1 = min(img.shape[0], py1 + pad)
    for ax, (m, label) in zip(row_axes, SHOW):
        preds, ref, res, radius = matched(r['panel_id'], m)
        ax.imshow(img.astype(np.uint8), extent=(0, img.shape[1], img.shape[0], 0))
        tp_idx = {mm['prediction_index'] for mm in res['matches']}
        tp_xy = np.array([[preds[i]['x_px'], preds[i]['y_px']] for i in sorted(tp_idx)]).reshape(-1, 2)
        fp_xy = np.array([[preds[i]['x_px'], preds[i]['y_px']] for i in res['unmatched_prediction_indices']]).reshape(-1, 2)
        fn_xy = np.array([[ref[i]['x_px'], ref[i]['y_px']] for i in res['unmatched_target_indices']]).reshape(-1, 2)
        ax.plot(tp_xy[:, 0], tp_xy[:, 1], 'o', ms=3.2, mfc=HUE['v46'], mec='white', mew=0.4, linestyle='none')
        ax.plot(fp_xy[:, 0], fp_xy[:, 1], 'x', ms=3.6, mec=HUE['aicp'], mew=1.0, linestyle='none')
        ax.plot(fn_xy[:, 0], fn_xy[:, 1], 's', ms=4.2, mfc='none', mec=HUE['lineex'], mew=1.0, linestyle='none')
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f"{chart_id[r['panel_id']]} ({name}) - {label}\nF1 {100 * f1(res['tp'], res['fp'], res['fn']):.1f} %"
                     f"   TP {res['tp']}, FP {res['fp']}, FN {res['fn']} of {len(ref)} points", fontsize=6.6, color=INK)
handles = [Line2D([], [], marker='o', ms=4, mfc=HUE['v46'], mec='white', linestyle='none', label='true positive (prediction)'),
           Line2D([], [], marker='x', ms=4, mec=HUE['aicp'], mew=1.0, linestyle='none',
                  label='false positive (prediction with no match, or assigned to the wrong legend entry)'),
           Line2D([], [], marker='s', ms=4.5, mfc='none', mec=HUE['lineex'], mew=1.0, linestyle='none',
                  label='false negative (ground-truth point not recovered)')]
fig.legend(handles=handles, loc='lower center', ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.045), fontsize=6.8)
caption('Figure 5. Example charts (per group, the chart with the median v46 per-chart F1; '
             'right: best published method)', fontsize=8.5, color=INK, x=0.02, ha='left', y=1.0)
fig.tight_layout(rect=(0, 0.0, 1, 0.985))
save(fig, 'fig5_examples_median_charts')
(FIG / 'fig5_selected_charts.json').write_text(json.dumps(
    [{'group': n, 'chart_id': chart_id[r['panel_id']], 'image': r['panel_id'] + '.png',
      'v46_after_f1_pct': round(100 * panel_f1(r, 'v46_after'), 1)} for n, r in examples], indent=1), encoding='utf-8')
# ---------------------------------------------------------------- legend-input sensitivity (Table 4, Figure 6)
SENS = DATA / 'sensitivity'
S1, S2 = SENS / 'annotated_key_boxes_20261025', SENS / 'ground_truth_key_crops_20261116'
ev_s = {'main': rows,
        'keys': json.loads((S1 / 'entry_evaluation.json').read_text(encoding='utf-8'))['panels'],
        'gt': json.loads((S2 / 'entry_evaluation.json').read_text(encoding='utf-8'))['panels']}
by_id = {k: {r['panel_id']: r for r in v} for k, v in ev_s.items()}
# single-condition 95 % intervals: the ones each run's scorer stored (same as Table 1 for the main result)
summ_s = {'main': ev['summary'],
          'keys': json.loads((S1 / 'entry_evaluation.json').read_text(encoding='utf-8'))['summary'],
          'gt': json.loads((S2 / 'entry_evaluation.json').read_text(encoding='utf-8'))['summary']}
SKEY = {'B&W charts': '흑백', 'Colour charts': '컬러', 'All 100 charts': '전체'}
COND = [('main', 'GUI condition (main result)'), ('keys', 'v46 given annotated key boxes (B&W)'),
        ('gt', 'published methods given ground-truth key crops')]
EXTERNAL = [m for m, _, fam, _ in METHODS if fam != 'v46']


def root_of(cond, panel_id):
    if cond == 'main':
        return DATA / 'results'
    if cond == 'keys':     # colour charts are byte-identical to the main result (checked by the package builder)
        return S1 if inputs[panel_id]['modality'] == 'bw' else DATA / 'results'
    return S2


def tp_on(cond, panel_id, series, method):
    """TP on the given series = TP(all points) - TP(those series removed); exact because matching only pairs
    points carrying the same legend entry."""
    p, t = inputs[panel_id], targets[panel_id]
    full = EE.score_panel(p, t, root_of(cond, panel_id), {})
    rest = EE.score_panel(p, dict(t, points=[q for q in t['points'] if q['series_id'] not in series]),
                          root_of(cond, panel_id), {})
    return full['methods'][method]['tp'] - rest['methods'][method]['tp']


for cond in ('keys', 'gt'):          # stored scores of the sensitivity runs reproduce from the packaged predictions
    for pid in ('pk__pepinemab2__p02__v1', 'er__dostarlimab_1_1_0__p01__v9'):
        p, t = inputs[pid], targets[pid]
        got = EE.score_panel(p, t, root_of(cond, pid), {})['methods']
        for m in (EXTERNAL if cond == 'gt' else [m for m, *_ in METHODS]):
            want = by_id[cond][pid]['methods'][m]
            assert (got[m]['tp'], got[m]['fp'], got[m]['fn']) == (want['tp'], want['fp'], want['fn']), (cond, pid, m)

missed = {}                          # series whose key v46 did not export in the main result
for r in rows:
    entries = EE.entries_of(DATA / 'results' / 'v46' / r['panel_id'])
    ref = EE.reference_entries(r['panel_id'], entries, {})
    s = {q['series_id'] for q in targets[r['panel_id']]['points'] if ref.get(q['series_id']) is None}
    if s:
        missed[r['panel_id']] = s
missed_points = sum(1 for pid, s in missed.items() for q in targets[pid]['points'] if q['series_id'] in s)

rng_s = np.random.default_rng(0)
SGROUPS = [('B&W charts', None, 'bw'), ('Colour charts', None, 'color'), ('All 100 charts', None, None)]
t4, sens = [], {}
for gname, ds, mo in SGROUPS:
    sel = [r['panel_id'] for r in rows if in_group(r, ds, mo)]
    idx = rng_s.integers(0, len(sel), (2000, len(sel)))
    for m, label, fam, _ in METHODS:
        tab = {c: np.array([[by_id[c][i]['methods'][m][k] for k in ('tp', 'fp', 'fn')] for i in sel], float)
               for c, _ in COND}
        boot = {c: np.array([f1(*tab[c][b].sum(0)) for b in idx]) for c in tab}
        rec = dict(group=gname, charts=len(sel), method=label)
        for c, _ in COND:
            if c == 'gt' and fam == 'v46':
                rec.update({f'{c}_f1_pct': '', f'{c}_ci95': '', f'{c}_minus_main_pct_points': '', f'{c}_diff_ci95': ''})
                continue
            v = f1(*tab[c].sum(0))
            stored = summ_s[c][SKEY[gname]]['methods'][m]
            assert abs(stored['f1'] - v) < 1e-12, (gname, m, c)
            lo, hi = stored['ci95']
            sens[(gname, m, c)] = (v, lo, hi)
            rec.update({f'{c}_f1_pct': round(100 * v, 1), f'{c}_ci95': f'{100 * lo:.1f}-{100 * hi:.1f}'})
            if c != 'main':
                d = boot[c] - boot['main']
                dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
                sens[(gname, m, c, 'diff')] = (v - f1(*tab['main'].sum(0)), dlo, dhi)
                rec.update({f'{c}_minus_main_pct_points': round(100 * (v - f1(*tab['main'].sum(0))), 1),
                            f'{c}_diff_ci95': f'{100 * dlo:+.1f} to {100 * dhi:+.1f}'})
        t4.append(rec)
lost = {c: sum(r['reference_points_without_entry'] for r in ev_s[c]) for c, _ in COND}
tp_missed = {c: {m: sum(tp_on(c, pid, s, m) for pid, s in missed.items()) for m, *_ in METHODS
                 if not (c == 'gt' and m.startswith('v46'))} for c, _ in COND}
write_table('table4_legend_input_sensitivity', t4,
            'Legend-input sensitivity: pooled F1 (%) under three legend inputs',
            'main = GUI condition (v46 finds the legend keys inside the supplied legend rectangle; the published methods '
            'get the key crops v46 exported). keys = 2026-10-25 run in which B&W v46 was given the annotated key boxes '
            '(colour identical to main). gt = the published methods given crops cut at the annotated (ground-truth) key '
            'boxes (v46 not run). Differences: paired, 2000 chart resamples (seed 0). Reference points whose legend key '
            f"no method could use: main {lost['main']}, keys {lost['keys']}, gt {lost['gt']} of 3,092. "
            f'On the {missed_points} points whose key v46 missed in main, TP main / keys / gt: ' +
            '; '.join(f"{LABEL[m]} {tp_missed['main'][m]} / {tp_missed['keys'][m]} / {tp_missed['gt'].get(m, '-')}"
                      for m, *_ in METHODS) + '.')
(TAB / 'table4_missed_key_points.json').write_text(json.dumps(
    {'points_on_series_with_missed_key_in_main': missed_points, 'charts': len(missed),
     'reference_points_without_usable_key': lost, 'tp_on_those_points': tp_missed}, indent=1), encoding='utf-8')


# ---------------------------------------------------------------- LaTeX versions of the paper tables
def tex_escape(s):
    return s.replace('&', r'\&').replace('%', r'\%')


def ci_cell(v, lo, hi):
    return f'{100 * v:.1f} ({100 * lo:.1f}--{100 * hi:.1f})'


lines = [r'\begin{tabular}{l' + 'c' * len(GROUPS) + '}', r'\toprule',
         'Method & ' + ' & '.join(tex_escape(SHORT[k]) for k, *_ in GROUPS) + r' \\',
         ' & ' + ' & '.join(f"{ev['summary'][k]['panels']} / {ev['summary'][k]['reference_points']:,}" for k, *_ in GROUPS) + r' \\',
         r'\midrule']
for m, label, _, _ in METHODS:
    lines.append(tex_escape(label) + ' & ' + ' & '.join(ci_cell(ev['summary'][k]['methods'][m]['f1'], *ev['summary'][k]['methods'][m]['ci95'])
                                                       for k, *_ in GROUPS) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'paper_table1_pooled_f1.tex').write_text(
    '% Pooled F1 (%) with 95% CI (2000 chart resamples); second header row: charts / reference points\n' + '\n'.join(lines) + '\n',
    encoding='utf-8')

lines = [r'\begin{tabular}{lcccc}', r'\toprule',
         r'Compared with & All charts $\Delta$F1 (95\% CI) & Charts higher / lower / equal & B\&W $\Delta$F1 & Colour $\Delta$F1 \\',
         r'\midrule']
for m, label, _, _ in METHODS[1:]:
    a_ = next(t for t in t2 if t['group'] == 'All 100 charts' and t['comparison'].endswith(label))
    cells = []
    for mo_ in ('bw', 'color'):
        sel = [r for r in rows if r['modality'] == mo_]
        A_, B_ = counts(sel, 'v46_after'), counts(sel, m)
        idx = np.random.default_rng(0).integers(0, len(sel), (2000, len(sel)))
        boot = np.array([f1(*A_[b].sum(0)) - f1(*B_[b].sum(0)) for b in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cells.append(f'{100 * (f1(*A_.sum(0)) - f1(*B_.sum(0))):+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})')
    lines.append(f"{tex_escape(label)} & {a_['delta_f1_pct_points']:+.1f} ({a_['ci95_low']:+.1f} to {a_['ci95_high']:+.1f}) & "
                 f"{a_['panels_v46_after_higher']} / {a_['panels_lower']} / {a_['panels_equal']} & {cells[0]} & {cells[1]}" + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'paper_table2_paired_differences.tex').write_text(
    '% v46 after Step 5 minus each method, percentage points; paired 2000 chart resamples (seed 0)\n' + '\n'.join(lines) + '\n',
    encoding='utf-8')

lines = [r'\begin{tabular}{l' + 'ccc' * len(SGROUPS) + '}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{3}{c}{' + tex_escape(g) + '}' for g, *_ in SGROUPS) + r' \\',
         'Method' + ' & Main & Key boxes & GT crops' * len(SGROUPS) + r' \\', r'\midrule']
for m, label, fam, _ in METHODS:
    cells = []
    for g, *_ in SGROUPS:
        cells.append(f"{100 * sens[(g, m, 'main')][0]:.1f}")
        for c in ('keys', 'gt'):
            if (g, m, c, 'diff') in sens:
                d, lo, hi = sens[(g, m, c, 'diff')]
                cells.append(f"{100 * sens[(g, m, c)][0]:.1f} ({100 * d:+.1f})")
            else:
                cells.append('--')
    lines.append(tex_escape(label) + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'paper_table3_legend_sensitivity.tex').write_text(
    '% Pooled F1 (%) under three legend inputs; in parentheses the paired difference from the main result (percentage points)\n'
    + '\n'.join(lines) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- drop-in replacements for the manuscript tables
# Same layout as outputs/cpt_hsu_style_revision_20260927/tables/performance_table.tex and correction_counts_table.tex
COLS = ['PK 흑백', 'PK 컬러', 'ER 흑백', 'ER 컬러', 'PK', 'ER', '흑백', '컬러', '전체']
ROWS_PERF = [('v46_before', 'Proposed, before correction'), ('v46_after', 'Proposed, after correction'),
             ('aicp_raw_grouped', 'AI-ChartParser, raw'), ('aicp_corrected_grouped', 'AI-ChartParser, corrected'),
             ('lineex_paper_grouped', 'LineEX, paper settings'), ('lineex_grouped', 'LineEX, released-code settings')]
perf = [r'\begingroup', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.25}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}@{\hspace{9pt}}rrrr@{\hspace{12pt}}rr@{\hspace{12pt}}rr@{\hspace{12pt}}r@{}}',
        r'\toprule',
        r'& \multicolumn{4}{c}{\textbf{Subgroups}} & \multicolumn{2}{c}{\textbf{Plot type}} & \multicolumn{2}{c}{\textbf{Appearance}} & \multicolumn{1}{c}{\textbf{Overall}} \\',
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-10}',
        r'Method & \shortstack{PK\\BW} & \shortstack{PK\\Color} & \shortstack{ER\\BW} & \shortstack{ER\\Color} & PK & ER & BW & Color & All \\',
        r'\midrule']
for m, label in ROWS_PERF:
    perf.append(f'{label} & ' + ' & '.join(f"{100 * ev['summary'][c]['methods'][m]['f1']:.1f}" for c in COLS) + r' \\')
perf += [r'\midrule', 'Panels & ' + ' & '.join(str(ev['summary'][c]['panels']) for c in COLS) + r' \\',
         'Reference points & ' + ' & '.join(f"{ev['summary'][c]['reference_points']:,}" for c in COLS) + r' \\',
         r'\bottomrule', r'\end{tabular}%', '}', r'\endgroup']
(TAB / 'paper_performance_table.tex').write_text('\n'.join(perf) + '\n', encoding='utf-8')

# per chart: Changed = TP or FP changed by Step 5; Favorable = TP not lower and FP not higher; Unfavorable = TP not
# higher and FP not lower; Mixed = TP and FP moved in the same direction (reproduces the 2026-10-25 table exactly)
cc = [r'\begingroup', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.4}',
      r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}lccrrrr@{}}', r'\toprule',
      r'Group & \shortstack{TP\\before $\to$ after ($\Delta$)} & \shortstack{FP\\before $\to$ after ($\Delta$)} & Changed & Favorable & Unfavorable & Mixed \\',
      r'\midrule']
counts_rows = []
for label, ds, mo in [('PK, BW', 'pk', 'bw'), ('PK, color', 'pk', 'color'), ('ER, BW', 'er', 'bw'), ('ER, color', 'er', 'color'),
                      ('All', None, None)]:
    sel = [r for r in rows if in_group(r, ds, mo)]
    tb, ta = (sum(r['methods'][m]['tp'] for r in sel) for m in ('v46_before', 'v46_after'))
    fb, fa = (sum(r['methods'][m]['fp'] for r in sel) for m in ('v46_before', 'v46_after'))
    ch = fav = unf = mix = 0
    for r in sel:
        dtp = r['methods']['v46_after']['tp'] - r['methods']['v46_before']['tp']
        dfp = r['methods']['v46_after']['fp'] - r['methods']['v46_before']['fp']
        if dtp == 0 and dfp == 0:
            continue
        ch += 1
        if dtp >= 0 and dfp <= 0:
            fav += 1
        elif dtp <= 0 and dfp >= 0:
            unf += 1
        else:
            mix += 1
    counts_rows.append(dict(group=label, tp_before=tb, tp_after=ta, fp_before=fb, fp_after=fa, changed=ch,
                            favorable=fav, unfavorable=unf, mixed=mix))
    if label == 'All':
        cc.append(r'\midrule')
    cc.append(f'{label} & ${tb:,}\\to{ta:,}$ ({ta - tb:+d}) & ${fb:,}\\to{fa:,}$ ({fa - fb:+d}) & {ch} & {fav} & {unf} & {mix} \\\\')
cc += [r'\bottomrule', r'\end{tabular}%', '}', r'\endgroup']
(TAB / 'paper_correction_counts_table.tex').write_text('\n'.join(cc) + '\n', encoding='utf-8')
write_table('table5_step5_correction_counts', counts_rows, 'Effect of Step 5 per chart group',
            'Changed = charts whose TP or FP changed; Favorable = TP not lower and FP not higher; Unfavorable = TP not '
            'higher and FP not lower; Mixed = TP and FP both higher or both lower.')

# ---------------------------------------------------------------- Figure 6: legend-input sensitivity
CMARK = {'main': ('o', True), 'keys': ('s', False), 'gt': ('D', True)}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (g, *_) in zip(axes, SGROUPS):
    for y, (m, label, fam, filled) in zip(ypos, METHODS):
        for j, (c, _) in enumerate(COND):
            if (g, m, c) not in sens:
                continue
            v, lo, hi = sens[(g, m, c)]
            yy = y + 0.25 - 0.25 * j
            mk, fl = CMARK[c]
            ax.plot([100 * lo, 100 * hi], [yy, yy], color=HUE[fam], lw=1.0)
            ax.plot(100 * v, yy, marker=mk, ms=4.6, mec=HUE[fam], mew=1.0, mfc=HUE[fam] if fl else 'white', linestyle='none')
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(g, fontsize=7.5, color=INK); ax.set_xlabel('Pooled F1 (%)')
axes[0].set_yticks(ypos); axes[0].set_yticklabels([m[1] for m in METHODS]); axes[0].set_ylim(-0.7, len(METHODS) - 0.3)
handles = [Line2D([], [], marker=CMARK[c][0], ms=4.6, mec=INK2, mfc=INK2 if CMARK[c][1] else 'white', linestyle='none', label=lab)
           for c, lab in COND]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=6.6)
caption('Figure 6. Sensitivity to the legend input (dot/marker = pooled F1, line = 95 % CI): main result; B&W v46 given '
        'the annotated key boxes; published methods given crops cut at the annotated key boxes')
save(fig, 'fig6_legend_input_sensitivity')

COMMON = ('Test set: 100 synthetic concentration-time (PK) and exposure-response (ER) charts, 25 per group, '
          '3,092 ground-truth marker points. A predicted point counts as correct only when it lies within the '
          'tolerance radius (0.0294 x image diagonal, the OKS > 0.5 criterion of LineEX) of a ground-truth point and '
          'is assigned to that point\'s legend entry; matching is one-to-one. Pooled F1 = 2TP / (2TP + FP + FN) over '
          'all points of a group. Every method received the same image, plot area and legend rectangle; v46 located '
          'the legend keys itself and the published methods received the key crops v46 exported.')
(FIG / 'figure_captions.md').write_text('# Figure captions (draft)\n\n' + '\n\n'.join(
    f'**{c.split(".")[0]}.**{c[len(c.split(".")[0]) + 1:]}. {COMMON}' for c in CAPTIONS) + '\n', encoding='utf-8')
print('figures:', sorted(p.name for p in FIG.iterdir()))
print('tables:', sorted(p.name for p in TAB.iterdir()))

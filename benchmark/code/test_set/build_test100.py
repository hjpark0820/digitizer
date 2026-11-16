"""Test set "test100_20261018": 25 synthetic charts per group (ER B&W, ER colour, PK B&W, PK colour), source-disjoint
from new-100, which is the development set (user, 2026-10-18: "테스트셋만 25개씩 각 카테고리에서 뽑아서 ... new100만 빼고").

Selection = the new-100 procedure (new100/select_new100.py) with a new seed and one more exclusion:
  shuffle sorted synthetic_similar/images/*.png with random.Random(SEED); one variant per base plot (source plot);
  pk / er from the filename prefix; bw / color from pixel chroma (>= 80 pixels with max-min > 35 and min < 200 -> color);
  eligible(variant) exactly as new-100: a legend box after rounding, a non-ref_line swatch box for every series that has
  points, every point-bearing series has a marker and only marker_centre points; no image-size cap;
  skip every base plot of new-100 (source-level disjoint). Nothing else is excluded (the user's rule), so a chart may
  share its source plot, or even its exact image, with DEV / held-out sets; the record lists that overlap per chart.
Inputs / targets are built exactly as new100/build_new100.py builds them. Chart ids: t1_<group>_NN.
Writes (nothing is overwritten; the script refuses if an output exists):
  benchmark_v2/inputs_test100_20261018.json, targets_test100_20261018.json, chart_id_map_test100_20261018.json
  test100_20261018/selection_test100.json (procedure, seed, counts, rejections, per-chart record incl. overlaps)
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
VMF = HERE.parent
BV2 = VMF.parent / 'published_chart_models' / 'benchmark_v2'
SYN = Path('<SYNTHETIC_DATASET>')
sys.path.insert(0, str(BV2))
from prepare_synthetic import box  # noqa: E402
sys.path.insert(0, str(VMF / 'new100'))
from select_new100 import eligible  # noqa: E402  (the identical GT filter)

SEED = 20261018
NAME = 'test100_20261018'
GROUPS = ['er_bw', 'er_color', 'pk_bw', 'pk_color']
src = lambda stem: stem.rsplit('__v', 1)[0]


def stems(path):
    return {Path(p['image_path']).stem for p in json.loads(Path(path).read_text(encoding='utf-8'))['panels']}


def main():
    outs = [BV2 / f'inputs_{NAME}.json', BV2 / f'targets_{NAME}.json', BV2 / f'chart_id_map_{NAME}.json', HERE / 'selection_test100.json']
    for o in outs:
        if o.exists():
            raise SystemExit(f'{o} exists; refusing to overwrite')
    new100 = stems(BV2 / 'inputs_new100_run.json')
    dev_bases = {src(s) for s in new100}
    other = {'DEV_testset365': stems(VMF / 'testset' / 'inputs_testset365.json'), 'heldout': stems(VMF / 'heldout' / 'inputs.json'),
             'heldout2': stems(VMF / 'heldout2' / 'inputs.json'), 'heldout3': stems(VMF / 'heldout3' / 'inputs.json'),
             'astra100': set(json.loads((BV2 / 'chart_id_map_astra100.json').read_text(encoding='utf-8'))),
             'inputs_synth': stems(BV2 / 'inputs_synth.json')}
    rng = random.Random(SEED)
    paths = sorted((SYN / 'images').glob('*.png'))
    rng.shuffle(paths)
    counts = {g: 0 for g in GROUPS}
    used, picks, rejected, checked = set(), [], [], 0
    for i, path in enumerate(paths):
        kind = path.name.split('__')[0]
        if kind not in ('pk', 'er'):
            continue
        base = src(path.stem)
        if base in used or base in dev_bases:
            continue
        with Image.open(path) as raw:
            a = np.asarray(raw.convert('RGB'))
        chroma = a.max(2).astype(int) - a.min(2).astype(int)
        n = int(((chroma > 35) & (a.min(2) < 200)).sum())
        group = kind + '_' + ('color' if n >= 80 else 'bw')
        checked += 1
        if counts[group] >= 25:
            continue
        ok, why = eligible(path.stem)
        if not ok:
            rejected.append({'shuffle_index': i, 'stem': path.stem, 'group': group, 'reason': why})
            continue
        counts[group] += 1
        used.add(base)
        picks.append({'shuffle_index': i, 'stem': path.stem, 'group': group, 'n': n, 'path': path, 'W': a.shape[1], 'H': a.shape[0]})
        if all(v == 25 for v in counts.values()):
            break
    assert all(v == 25 for v in counts.values()), counts
    assert not ({src(p['stem']) for p in picks} & dev_bases)
    nxt = Counter()
    inputs, targets, cmap, recs = [], [], {}, []
    for p in sorted(picks, key=lambda p: (GROUPS.index(p['group']), p['shuffle_index'])):
        nxt[p['group']] += 1
        cid = f"t1_{p['group']}_{nxt[p['group']]:02d}"
        stem = p['stem']
        g = json.loads((SYN / 'gt' / f'{stem}.json').read_text(encoding='utf-8'))
        W, H = g['image_size']
        image = Path(g['image_path'])
        assert (W, H) == (p['W'], p['H']), stem
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        legend = box(g.get('legend_bbox'), W, H)
        swatches, seen = [], set()
        for e in g.get('legend_entries') or []:
            b = box(e.get('swatch_bbox'), W, H)
            if e['series_id'] == 'ref_line' or b is None or e['series_id'] in seen:
                continue
            seen.add(e['series_id'])
            swatches.append({'series_id': e['series_id'], 'label': e.get('label') or e['series_id'], 'bbox': b})
        assert legend is not None and swatches, stem
        dataset, modality = p['group'].split('_')
        inputs.append({'panel_id': stem, 'dataset': dataset, 'modality': modality, 'figure_id': src(stem),
                       'synthetic': True, 'image_path': str(image), 'image_size': [W, H], 'source_image_path': str(image),
                       'source_image_size': [W, H], 'crop_box_original': [0, 0, W, H], 'input_sha256': digest,
                       'source_sha256': digest, 'plot_bbox': box(g['plot_bbox'], W, H), 'legend_bbox': legend,
                       'swatches': swatches, 'rotation_degrees': 0, 'astra_chart_id': cid})
        pts = [(s, q) for s in g['series'] for q in s['points']]
        targets.append({'panel_id': stem, 'dataset': dataset, 'modality': modality, 'figure_id': src(stem),
                        'reference_origin': 'synthetic_rendered_exact', 'reference_flags': [],
                        'points': [{'point_id': f"{stem}__{s['series_id']}__n{j:04d}", 'series_id': s['series_id'],
                                    'x_px': q['x_px'], 'y_px': q['y_px'], 'source_curve_name': s['label'],
                                    'point_kind': s.get('point_kind')} for j, (s, q) in enumerate(pts)],
                        'symbol_reference': {s['series_id']: {'marker_text': s['marker'], 'rgb': s['colour_hex'], 'label': s['label']}
                                             for s in g['series']}})
        cmap[stem] = cid
        recs.append({'chart_id': cid, 'group': p['group'], 'source_filename': image.name, 'source_plot_id': src(stem),
                     'image_path': str(image), 'width': W, 'height': H, 'colored_pixel_count': p['n'], 'source_sha256': digest,
                     'shuffle_index': p['shuffle_index'], 'gt_points': len(pts), 'generator_modality': g['modality'],
                     'same_image_in': sorted(k for k, v in other.items() if stem in v),
                     'same_source_plot_in': sorted(k for k, v in other.items() if src(stem) in {src(x) for x in v})})
    doc = {'name': NAME, 'created': '2026-10-18', 'role': 'test set; new-100 (inputs_new100_run.json) is the development set',
           'seed': SEED, 'procedure': __doc__, 'excluded_base_plots': 'the 100 source plots of new-100 only',
           'candidates_examined': checked, 'counts': counts, 'rejected_by_gt_filter': rejected,
           'overlap_summary': {k: {'same_image': sum(k in r['same_image_in'] for r in recs),
                                   'same_source_plot': sum(k in r['same_source_plot_in'] for r in recs)} for k in other},
           'charts': recs}
    (BV2 / f'inputs_{NAME}.json').write_text(json.dumps({'panels': inputs}, indent=1, ensure_ascii=False), encoding='utf-8')
    (BV2 / f'targets_{NAME}.json').write_text(json.dumps({'panels': targets}, indent=1, ensure_ascii=False), encoding='utf-8')
    (BV2 / f'chart_id_map_{NAME}.json').write_text(json.dumps(cmap, indent=1), encoding='utf-8')
    (HERE / 'selection_test100.json').write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding='utf-8')
    print('counts', counts, 'examined', checked, 'rejected', len(rejected), Counter(r['reason'] for r in rejected))
    print('GT points per group', {g: sum(r['gt_points'] for r in recs if r['group'] == g) for g in GROUPS})
    print('overlap', json.dumps(doc['overlap_summary']))
    print('generator modality differs from pixel group:', [(r['chart_id'], r['generator_modality']) for r in recs if r['generator_modality'] != r['group'].split('_')[1]])


if __name__ == '__main__':
    main()

"""New 100-chart set = Astra-100 with the 11 charts the 2026-10-05 test-set rule drops replaced.

Rule (user, 2026-10-05): no chart without a legend (single-series ones included), no chart whose GT has points of a
marker-less line (point_kind line_vertex / marker none, e.g. pk__brontictuzumab2__p10__v4 = pk_bw_22).
Procedure = the original selection (outputs/independent_digitization_100_20260930/scripts/select_images.py: seed 20260930
shuffle of sorted images/*.png, one variant per base plot, pk/er from the filename prefix, bw/color from pixel chroma:
>= 80 pixels with max-min > 35 and min < 200 -> color, 25 per group) re-run with two additions:
  eligible(variant)  GT has a legend box after rounding, a non-ref_line swatch box for every series that has points,
                     and every series with points has marker != none and point_kind marker_centre;
  skip               base plots of the 160-chart held-out set (experiments/visible_miss_fix/heldout), so it stays unseen.
No image-size cap (the original had none; 7 of the kept 89 are > 2 MP).
Check: the re-run must reproduce the 89 kept charts exactly; the extra picks get new ids er_bw_26.., pk_bw_26..
(old ids are never reused: every existing result is keyed on them).
Writes new100/selection_new100.json (records in the original selection.json format + 'status': kept|new|changed_input).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
VMF = HERE.parent
REPO = VMF.parents[1]
SYN = Path('<SYNTHETIC_DATASET>')
OLD = json.loads((REPO / 'outputs/independent_digitization_100_20260930/selection.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(VMF.parent / 'published_chart_models' / 'benchmark_v2'))
from prepare_synthetic import box  # noqa: E402


def eligible(stem):
    g = json.loads((SYN / 'gt' / f'{stem}.json').read_text(encoding='utf-8'))
    W, H = g['image_size']
    if box(g.get('legend_bbox'), W, H) is None:
        return False, 'no_legend'
    sw = {e['series_id'] for e in g.get('legend_entries') or []
          if e['series_id'] != 'ref_line' and box(e.get('swatch_bbox'), W, H) is not None}
    for s in g['series']:
        if not s['points']:
            continue
        if s['series_id'] not in sw:
            return False, 'series_without_swatch'
        kinds = {(p.get('point_kind') or s.get('point_kind') or 'marker_centre') for p in s['points']}
        if (s.get('marker') or 'none') == 'none' or kinds != {'marker_centre'}:
            return False, 'line_vertex_gt'
    return True, ''


def main():
    ts = json.loads((VMF / 'testset' / 'charts.json').read_text())
    kept_ids = set(ts['astra'])
    old_by_id = {r['chart_id']: r for r in OLD['charts']}
    ho_bases = {s.rsplit('__v', 1)[0] for s in json.loads((VMF / 'heldout' / 'chart_map.json').read_text()).values()}
    rng = random.Random(20260930)
    paths = sorted((SYN / 'images').glob('*.png'))
    rng.shuffle(paths)
    counts = {k: 0 for k in ['pk_color', 'pk_bw', 'er_color', 'er_bw']}
    used, picks, rejected, checked = set(), [], [], 0
    for i, path in enumerate(paths):
        kind = path.name.split('__')[0]
        if kind not in ('pk', 'er'):
            continue
        base = path.stem.rsplit('__v', 1)[0]
        if base in used or base in ho_bases:
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
    old_stem = {r['source_filename'][:-4]: r for r in OLD['charts']}
    kept = [p for p in picks if p['stem'] in old_stem]
    new = [p for p in picks if p['stem'] not in old_stem]
    kept_found = {old_stem[p['stem']]['chart_id'] for p in kept}
    assert kept_found == kept_ids, (sorted(kept_ids - kept_found), sorted(kept_found - kept_ids))
    nxt = {'er_bw': 26, 'pk_bw': 26, 'er_color': 26, 'pk_color': 26}
    recs = []
    for p in kept:
        r = dict(old_stem[p['stem']])
        sha = hashlib.sha256(p['path'].read_bytes()).hexdigest()
        r['status'] = 'kept' if sha == r['source_sha256'] else 'changed_input'
        if sha != r['source_sha256']:
            r['old_source_sha256'] = r['source_sha256']
            r['source_sha256'] = sha
            r['width'], r['height'] = p['W'], p['H']
        r['shuffle_index'] = p['shuffle_index']
        recs.append(r)
    for p in new:
        cid = f"{p['group']}_{nxt[p['group']]:02d}"
        nxt[p['group']] += 1
        recs.append({'chart_id': cid, 'group': p['group'], 'source_filename': p['path'].name, 'source_image': str(p['path']),
                     'width': p['W'], 'height': p['H'], 'colored_pixel_count': p['n'],
                     'source_sha256': hashlib.sha256(p['path'].read_bytes()).hexdigest(),
                     'shuffle_index': p['shuffle_index'], 'status': 'new'})
    dropped = sorted(set(old_by_id) - kept_ids)
    doc = {'created': '2026-10-05', 'seed': 20260930,
           'base_selection': 'outputs/independent_digitization_100_20260930/selection.json',
           'method': OLD['method'] + ' Re-run with a GT eligibility filter (legend with a swatch for every point-bearing series; '
                     'every point-bearing series has markers and marker_centre points) and with the 160 held-out base plots skipped.',
           'dropped_from_astra100': {c: ts['excluded_astra'][c] for c in dropped}, 'candidates_examined': checked,
           'counts': counts, 'rejected_by_gt_filter': rejected, 'charts': sorted(recs, key=lambda r: r['chart_id'])}
    (HERE / 'selection_new100.json').write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding='utf-8')
    print('counts', counts, 'checked', checked, 'kept', len(kept), 'new', len(new))
    for r in doc['charts']:
        if r['status'] != 'kept':
            print(r['status'], r['chart_id'], r['source_filename'], r['width'], r['height'], 'shuffle', r['shuffle_index'])
    print('rejected by GT filter:', len(rejected), [(x['shuffle_index'], x['group'], x['reason']) for x in rejected])


if __name__ == '__main__':
    main()

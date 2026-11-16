"""Run the unchanged production v46 CLIs over the 20-panel two-dataset cohort.

This is benchmark_pk/run_v46.py driven by a different manifest. That module is imported and
only two of its rules are relaxed, both because the ER dataset differs from the PK cohort:

  validate_inputs      the PK version hard-codes "exactly five BW and five colour panels" and
                       requires annotated legend swatch boxes. Here the cohort is 20 panels and
                       ER panels have no swatch annotation (the original ER digitisation runs
                       also supplied only a legend rectangle), so an empty swatch list is
                       allowed and the count check follows the manifest.
  detection_command    with no annotated swatches the BW CLI is called exactly as the ER runs
                       called it: --plot-area and --legend-area only, letting v46 locate the
                       legend entries itself. Nothing else about the command changes.

Everything else (process launching, Step-5 correction with at most five iterations, legend crop
export for the external models, prediction export, timings) is the imported original code.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import src_fingerprint  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent

spec = importlib.util.spec_from_file_location('benchmark_pk_run_v46', EXPERIMENT / 'benchmark_pk' / 'run_v46.py')
R46 = importlib.util.module_from_spec(spec)
sys.modules['benchmark_pk_run_v46'] = R46
spec.loader.exec_module(R46)

_original_detection_command = R46.detection_command


def validate_inputs(data):
    R46.reject_targets(data)
    panels = data['panels']
    ids = [p['panel_id'] for p in panels]
    if not panels or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in ids):
        raise ValueError('Panel IDs must be unique safe path components')
    for panel in panels:
        size = panel['image_size']
        crop = R46.box(panel['crop_box_original'], panel['source_image_size'])
        if size != [crop[2] - crop[0], crop[3] - crop[1]]:
            raise ValueError('Native crop dimensions do not match translation-only geometry')
        R46.box(panel['plot_bbox'], size)
        R46.box(panel['legend_bbox'], size)
        sids = [s['series_id'] for s in panel['swatches']]
        if len(sids) != len(set(sids)):
            raise ValueError('Legend series identities must be unique')
        for swatch in panel['swatches']:
            R46.box(swatch['bbox'], size)
        for name in ('image_path', 'source_image_path', 'input_sha256', 'source_sha256'):
            if not panel.get(name):
                raise ValueError(f'Missing image binding: {name}')
    print('cohort:', dict(Counter(f"{p.get('dataset','?')}/{p['modality']}" for p in panels)), flush=True)
    return panels


def detection_command(panel, destination, python):
    if panel['modality'] == 'bw' and not panel['swatches']:
        return [str(python), str(R46.SRC / 'run_A4_auto_v46.py'), panel['image_path'], str(destination),
                '--mode', 'bw', '--plot-area', R46.csv_box(panel['plot_bbox']),
                '--legend-area', R46.csv_box(panel['legend_bbox']), '--skip-segments']
    return _original_detection_command(panel, destination, python)


def correction_result(panel, after):
    """Accept the production correction state as it is today.

    benchmark_pk/run_v46.py demands colour backend 'v46_path_shape_v3'. Production src has since
    moved to 'v46_series_routed_v1' (series-routed Step 5), so the old check would report every
    colour panel as correction_failed even though correction ran. Record the backend and the raw
    per-series status instead of asserting one name; still refuse more than five iterations.
    """
    if panel['modality'] == 'bw':
        return _original_correction_result(panel, after)
    state = R46.read(after / 'color_correction_state.json')
    backend = state.get('backend')
    if backend not in ('v46_path_shape_v3', 'v46_series_routed_v1'):
        raise ValueError(f'Unknown colour correction backend: {backend}')
    iterations = {}
    for key, value in (state.get('path_state') or {}).items():
        iterations[key] = int(value.get('iteration_count', 0))
    for key, value in (state.get('series_state') or {}).items():
        if key not in iterations and isinstance(value, dict) and 'iteration_count' in value:
            iterations[key] = int(value['iteration_count'])
    if any(n > 5 or n < 0 for n in iterations.values()):
        raise ValueError('Step 5 exceeded the requested five-iteration cap')
    statuses = state.get('status', {})
    finished = {'completed', 'corrected'}
    return dict(iterations_by_series=iterations, status_by_series=statuses, requested_iterations=5,
                backend=backend,
                status='completed' if statuses and all(s in finished for s in statuses.values()) else 'completed_partial')


_original_correction_result = R46.correction_result
R46.validate_inputs = validate_inputs
R46.detection_command = detection_command
R46.correction_result = correction_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'inputs.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'run_v1' / 'v46')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--ids', nargs='*', help='only these panel ids')
    args = parser.parse_args()
    data = json.loads(Path(args.inputs).read_text(encoding='utf-8-sig'))
    panels = validate_inputs(data)
    if args.ids:
        panels = [p for p in panels if p['panel_id'] in set(args.ids)]
    args.output.mkdir(parents=True, exist_ok=True)
    before_fingerprint = src_fingerprint.fingerprint()
    suffix = '' if not args.ids else '_ids'
    (args.output / f'src_fingerprint_before{suffix}.json').write_text(json.dumps(before_fingerprint, indent=2), encoding='utf-8')
    print('src fingerprint before run:', before_fingerprint['combined_sha256'][:16],
          f"({before_fingerprint['file_count']} files)", flush=True)
    done = []
    for panel in panels:
        folder = args.output / panel['panel_id']
        if folder.exists():
            print('skip (already present):', panel['panel_id'], flush=True)
            continue
        panel_src = src_fingerprint.fingerprint()['combined_sha256']
        job = R46.run_panel(panel, args.output, args.python, args.timeout, dict(**__import__('os').environ))
        done.append({'panel_id': panel['panel_id'], 'status': job['status'],
                     'points_before': job.get('points_before'), 'points_after': job.get('points_after'),
                     'seconds': round(job['timing'].get('total_wall_seconds', 0), 1),
                     'src_sha256_at_panel_start': panel_src})
        print(json.dumps(done[-1]), flush=True)
    after_fingerprint = src_fingerprint.fingerprint()
    (args.output / f'src_fingerprint_after{suffix}.json').write_text(json.dumps(after_fingerprint, indent=2), encoding='utf-8')
    changed = src_fingerprint.compare(before_fingerprint, after_fingerprint)
    if changed:
        print(f'WARNING: production src changed DURING this run ({len(changed)} files). '
              'Results from earlier panels ran on different code:', flush=True)
        for row in changed:
            print('   ', row['state'], row['file'], flush=True)
    summary_name = 'run_summary.json' if not args.ids else 'run_summary_ids_' + '_'.join(sorted(args.ids))[:80] + '.json'
    (args.output / summary_name).write_text(json.dumps(
        {'panels': done, 'src_changed_during_run': changed,
         'src_combined_sha256_before': before_fingerprint['combined_sha256'],
         'src_combined_sha256_after': after_fingerprint['combined_sha256']}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

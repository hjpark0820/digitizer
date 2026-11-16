"""Run the released AI-ChartParser and LineEX models over the 20-panel two-dataset cohort.

This is benchmark_pk/run_external.py with one rule relaxed: its main() refuses any cohort that is
not exactly five BW and five colour panels. Everything that touches the models - preprocessing,
checkpoints, thresholds, the ChartParser centreline correction, the LineEX background gate, the
released learned legend matcher, the scope filter and the exported record - is the imported
original code, called through its own run_panel().

Both published models are given the same shared legend crops the v46 run exported (or, where the
v46 legend stage failed on a PK panel, the crops ensure_legends.py cut from the benchmark's own
annotated swatch boxes), so a v46 legend failure does not silently penalise them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent

spec = importlib.util.spec_from_file_location('benchmark_pk_run_external', EXPERIMENT / 'benchmark_pk' / 'run_external.py')
RX = importlib.util.module_from_spec(spec)
sys.modules['benchmark_pk_run_external'] = RX
spec.loader.exec_module(RX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=HERE / 'inputs_with_er_swatches.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'run_v1' / 'external')
    parser.add_argument('--shared-legends-root', type=Path, default=HERE / 'results' / 'run_v1' / 'v46')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--models', nargs='+', choices=['chartparser', 'lineex'], default=['chartparser', 'lineex'])
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()

    frozen = RX.read(args.manifest)
    RX.validate_safe_tree(frozen)
    panels = [RX.normalize_panel(p) for p in frozen['panels']]
    if args.ids:
        panels = [p for p in panels if p['panel_id'] in set(args.ids)]
    print(f'{len(panels)} panels; modalities: '
          f"{ {m: sum(1 for p in panels if p['modality'] == m) for m in ('bw', 'color')} }", flush=True)

    modules = {'matcher': RX.module_at('benchmark_lineex_matcher', 'lineex/legend_matcher.py'),
               'geometry': RX.module_at('benchmark_chartparser_geometry', 'chartparser/run_grouped.py')}
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    setup = time.perf_counter()
    matcher = modules['matcher'].LegendMatcher(device=str(device), batch_size=args.batch_size)
    load_times = {'legend_matcher_seconds': time.perf_counter() - setup}
    provenance = {'matcher': matcher.provenance}
    models = {}
    if 'chartparser' in args.models:
        modules['chartparser'] = RX.module_at('benchmark_chartparser_heads', 'chartparser/run_heads.py')
        setup = time.perf_counter()
        models['chartparser'], provenance['chartparser'] = modules['chartparser'].build_model(
            EXPERIMENT / 'chartparser/weights/epoch-19.pth', device)
        load_times['chartparser_seconds'] = time.perf_counter() - setup
    if 'lineex' in args.models:
        modules['lineex'] = RX.module_at('benchmark_lineex_heads', 'lineex/run_keypoints.py')
        setup = time.perf_counter()
        model, gate, provenance['lineex'] = modules['lineex'].load_model(str(device))
        models['lineex'] = (model, gate)
        load_times['lineex_seconds'] = time.perf_counter() - setup

    helpers = ['chartparser/run_heads.py', 'chartparser/run_grouped.py', 'lineex/run_keypoints.py',
               'lineex/legend_matcher.py']
    RX.write(args.output / 'runtime.json', dict(version=RX.VERSION + '+benchmark_v2', manifest_path=str(args.manifest.resolve()),
             manifest_sha256=RX.sha(args.manifest), runner_sha256=RX.sha(EXPERIMENT / 'benchmark_pk' / 'run_external.py'),
             wrapper_sha256=RX.sha(__file__), device=str(device),
             helper_source_sha256={name: RX.sha(EXPERIMENT / name) for name in helpers},
             shared_legends_root=str(args.shared_legends_root.resolve()), model_load_seconds=load_times,
             provenance=provenance,
             timing_notes='One run per panel, no warmup; model load excluded from per-panel timing.'))
    rows = []
    for panel in panels:
        try:
            result = run = RX.run_panel(panel, args.output, models, modules, matcher, device, args.shared_legends_root)
            row = dict(panel_id=panel['panel_id'], modality=panel['modality'], status='completed', models=result)
        except Exception:
            row = dict(panel_id=panel['panel_id'], modality=panel['modality'], status='failed',
                       error=traceback.format_exc())
            RX.write(args.output / panel['panel_id'] / 'failure.json', row)
        rows.append(row)
        RX.write(args.output / 'summary.json', dict(expected=len(panels), processed=len(rows), panels=rows))
        print(f"{panel['panel_id']} {row['status']} " +
              json.dumps({k: v['point_count'] for k, v in row.get('models', {}).items()}), flush=True)
    return int(any(row['status'] == 'failed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())

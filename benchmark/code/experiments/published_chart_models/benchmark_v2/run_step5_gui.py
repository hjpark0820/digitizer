"""Run v46 Step 5 exactly the way the GUI's "Run Step 5 correction" button does.

Why: the earlier harness (benchmark_pk/run_v46.py) ran Step 5 through the old entry points -
BW: run_A4_auto_v46.py --prev-state --correct ; colour: color_correct_cli.py --require-v46-path.
The GUI no longer does that. unified_server.py:/correct-saved now
  1. exports the finished detection job as a correction session  (correction_session_v46.export_session)
  2. imports it into a fresh job folder                            (correction_session_v46.import_session)
  3. runs  correction_only_cli_v46.py <folder> --iterations N --stop-policy manual
  4. counts the run as successful only when correction_only_status.json says completed=True;
     otherwise the job keeps the detection points and reports "Correction failed".
On the 100-panel re-run the old entry points failed on 14 panels (8 BW with an internal
'Inconsistent metadata for one swatch ID', 6 colour-group panels that have no step5_inputs.json),
so the old path no longer measures what a GUI user gets.

For every panel of a run whose detection produced an edit_data.json, this script builds the job
folder the GUI would have (raw_before artefacts + input.png = the exact benchmark crop), runs the
three GUI steps with 5 iterations, and writes the result as v46_after.json. The previous file is
kept as v46_after_legacy_cli.json. A failed correction is recorded as status 'failed' with no
points, exactly as before, so the protocol does not change - only the entry point does.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
import os
# BENCHMARK_SRC points at a frozen src snapshot (run_parallel.py); default is the live tree.
SRC = Path(os.environ.get('BENCHMARK_SRC') or (EXPERIMENT.parents[1] / 'src')).resolve()
sys.path.insert(0, str(SRC))

THREAD_ENV = {k: os.environ.get('BENCHMARK_THREADS', '') for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
THREAD_ENV = {k: v for k, v in THREAD_ENV.items() if v}

spec = importlib.util.spec_from_file_location('benchmark_pk_run_v46', EXPERIMENT / 'benchmark_pk' / 'run_v46.py')
R46 = importlib.util.module_from_spec(spec)
sys.modules['benchmark_pk_run_v46'] = R46
spec.loader.exec_module(R46)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run_panel(panel, folder, python, iterations):
    from correction_session_v46 import export_session, import_session
    started = time.perf_counter()
    folder = Path(folder).resolve()      # the CLI runs with cwd=src, so every path must be absolute
    before = folder / 'raw_before'
    if not (before / 'edit_data.json').exists():
        return {'status': 'skipped', 'reason': 'detection produced no edit_data.json'}
    stage = folder / 'gui_step5' / 'source'
    out = folder / 'gui_step5' / 'job' / 'out'
    for path in (stage, out.parent):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(before, stage)
    shutil.copyfile(panel['image_path'], stage / 'input.png')
    mode = 'bw' if panel['modality'] == 'bw' else 'color'
    package = export_session(stage, mode)
    info = import_session((stage / 'input.png').read_bytes(), json.dumps(package, allow_nan=False).encode(), out, mode)
    if not info.get('engine'):
        return {'status': 'failed', 'reason': f"no correction engine: {info.get('reason')}",
                'seconds': time.perf_counter() - started}
    cmd = [str(python), str(SRC / 'correction_only_cli_v46.py'), str(out), '--iterations', str(iterations),
           '--stop-policy', 'manual']
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800,
                          cwd=str(SRC), env=dict(os.environ, **THREAD_ENV))
    (folder / 'gui_step5' / 'correction_only.log').write_text((proc.stdout or '') + (proc.stderr or ''), encoding='utf-8')
    status_path = out / 'correction_only_status.json'
    status = read(status_path) if status_path.exists() else {}
    ok = proc.returncode == 0 and status.get('completed') is True
    return {'status': 'completed' if ok else 'failed', 'exit_code': proc.returncode, 'correction_status': status,
            'out': str(out), 'seconds': time.perf_counter() - started,
            'reason': '' if ok else (status.get('error') or status.get('reason') or f'exit {proc.returncode}')}


def step5_panel(panel, results, python, iterations):
    """GUI Step 5 for one panel of a run folder; returns its summary row (also used by run_parallel.py)."""
    folder = (Path(results) / 'v46' / panel['panel_id']).resolve()
    try:
        result = run_panel(panel, folder, python, iterations)
    except Exception as error:
        result = {'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
    if result['status'] == 'skipped':
        return {'panel_id': panel['panel_id'], **result}
    shared = folder / 'shared_legends.json'
    mapping = read(shared).get('mapping', {}) if shared.exists() else {}
    rows = []
    if result['status'] == 'completed':
        rows = R46.predictions(read(Path(result['out']) / 'edit_data.json'), panel, mapping, 'v46_after')
    old = folder / 'v46_after.json'
    legacy = folder / 'v46_after_legacy_cli.json'
    if old.exists() and not legacy.exists():
        old.replace(legacy)
    record = R46.publish_prediction(folder, panel, 'v46_after', result['status'], rows, result.get('reason', ''))
    record['step5_entry_point'] = 'GUI /correct-saved: export_session -> import_session -> correction_only_cli_v46.py'
    record['step5_src'] = str(SRC)
    record['iterations_requested'] = iterations
    record['correction_status'] = result.get('correction_status')
    (folder / 'v46_after.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'panel_id': panel['panel_id'], 'status': result['status'], 'points': len(rows),
            'seconds': round(result.get('seconds', 0), 1), 'reason': result.get('reason', '')[:160]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='the run folder, e.g. results/run_100b')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    panels = read(args.inputs)['panels']
    if args.ids:
        panels = [p for p in panels if p['panel_id'] in set(args.ids)]
    sys.path.insert(0, str(HERE))
    import src_fingerprint
    start_fp = src_fingerprint.fingerprint()
    summary = []
    for panel in panels:
        panel_src = src_fingerprint.fingerprint()['combined_sha256']
        summary.append(dict(step5_panel(panel, args.results, args.python, args.iterations),
                            src_sha256_at_panel_start=panel_src))
        print(json.dumps(summary[-1]), flush=True)
    end_fp = src_fingerprint.fingerprint()
    changed = src_fingerprint.compare(start_fp, end_fp)
    if changed:
        print(f'WARNING: src changed DURING Step 5 ({len(changed)} files)', flush=True)
    (args.results / 'v46' / 'gui_step5_summary.json').write_text(json.dumps(
        {'panels': summary, 'src_before': start_fp['combined_sha256'], 'src_after': end_fp['combined_sha256'],
         'src_changed_during_step5': changed}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Parallel, snapshot-pinned v46 run: detection + GUI-equivalent Step 5 for every panel.

Differences from the sequential run_v46_v2.py + run_step5_gui.py pair used for run_100b:

  code        every CLI runs from a frozen copy of src/ (make_snapshot.py) through
              snapshot_timed_cli.py, so edits made to production src/ during the run cannot
              reach it. The snapshot fingerprint is checked again at the end.
  parallelism N worker threads, each driving one panel at a time in fresh CLI processes.
              B&W v46 uses CUDA (bw_compute_backend 'auto'), so at most --gpu-slots B&W panels
              hold the GPU at once; colour panels are CPU only. Each child process is limited
              to --threads BLAS/OpenMP threads so 20 processes do not oversubscribe 28 cores.
              Panels start longest-first (run_100b wall times) to shorten the tail.
  Step 5      the GUI /correct-saved path (run_step5_gui.step5_panel) runs in the same worker
              right after that panel's detection. The legacy CLI correction that run_v46.py
              runs inside run_panel is skipped: its output was never scored (run_step5_gui
              replaced v46_after with the GUI result), it only cost time.
  timing      wall times are measured under contention and are NOT comparable with the
              sequential runs.

A file named STOP in the run folder makes workers finish their current panel and start no new one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent


class LegacyCorrectionSkipped(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='run folder; v46/ is created inside')
    parser.add_argument('--snapshot', type=Path, required=True, help='src_snapshots/<digest>/src')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--gpu-slots', type=int, default=6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--order-from', type=Path, default=HERE / 'results' / 'run_100b' / 'v46')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    args.results = args.results.resolve()      # child CLIs run with cwd = snapshot root
    args.inputs = args.inputs.resolve()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['BENCHMARK_SRC'] = str(snapshot)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ.pop('PYTHONPATH', None)

    sys.path.insert(0, str(HERE))
    import src_fingerprint
    import run_v46_v2 as V2          # patched benchmark_pk/run_v46.py (validate_inputs, BW command, backend)
    import run_step5_gui as S5       # reads BENCHMARK_SRC at import
    R46 = V2.R46
    assert Path(S5.SRC) == snapshot, (S5.SRC, snapshot)
    R46.SRC = snapshot

    def launch(command, folder, phase, env, timeout):
        tick = perf_counter()
        actual = [command[0], str(HERE / 'snapshot_timed_cli.py'), str(snapshot),
                  str(folder / f'{phase}_stage_timing.json'), '--', *command[1:]]
        with (folder / f'{phase}.log').open('w', encoding='utf-8') as log:
            try:
                p = subprocess.run(actual, cwd=str(snapshot.parent), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                result = dict(exit_code=p.returncode, timed_out=False, wall_seconds=perf_counter() - tick)
            except subprocess.TimeoutExpired:
                result = dict(exit_code=None, timed_out=True, wall_seconds=perf_counter() - tick)
        result.update(command=command, timing_wrapper_command=actual, log_path=str(folder / f'{phase}.log'))
        R46.save(folder / f'{phase}_process.json', result)
        return result

    def skip_legacy_correction(panel, before, after, python):
        raise LegacyCorrectionSkipped('legacy CLI correction skipped; GUI Step 5 (run_step5_gui) is scored')

    R46.launch = launch
    R46.correction_command = skip_legacy_correction

    data = json.loads(args.inputs.read_text(encoding='utf-8-sig'))
    panels = V2.validate_inputs(data)
    if args.ids:
        panels = [p for p in panels if p['panel_id'] in set(args.ids)]
    expected = {}
    for job in args.order_from.glob('*/job.json'):
        try:
            j = json.loads(job.read_text(encoding='utf-8'))
            expected[j['panel_id']] = j['timing'].get('total_wall_seconds') or 0
        except Exception:
            pass
    panels.sort(key=lambda p: -expected.get(p['panel_id'], 60))

    output = args.results / 'v46'
    output.mkdir(parents=True, exist_ok=True)
    (args.results / 'driver.pid').write_text(str(os.getpid()), encoding='utf-8')
    snap_start = src_fingerprint.fingerprint(snapshot)
    live_start = src_fingerprint.fingerprint()
    (output / 'src_fingerprint_snapshot_start.json').write_text(json.dumps(snap_start, indent=2), encoding='utf-8')
    (output / 'src_fingerprint_live_start.json').write_text(json.dumps(live_start, indent=2), encoding='utf-8')
    print(f"snapshot {snapshot} {snap_start['combined_sha256'][:12]} ({snap_start['file_count']} files); "
          f"live src {live_start['combined_sha256'][:12]}; workers={args.workers} gpu_slots={args.gpu_slots} "
          f"threads/process={args.threads}; {len(panels)} panels", flush=True)

    gpu = threading.Semaphore(args.gpu_slots)
    lock = threading.Lock()
    rows = []
    stop_file = args.results / 'STOP'
    started_at = time.time()

    def work(panel):
        if stop_file.exists():
            return {'panel_id': panel['panel_id'], 'status': 'not_started (STOP file)'}
        slot = gpu if panel['modality'] == 'bw' else None
        tick = perf_counter()
        if slot:
            slot.acquire()
        try:
            waited = perf_counter() - tick
            folder = output / panel['panel_id']
            if folder.exists():
                return {'panel_id': panel['panel_id'], 'status': 'skipped (folder exists)'}
            job = R46.run_panel(panel, output, args.python, args.timeout, dict(os.environ))
            reason = job.get('reason', '')
            detected = job['status'] != 'detection_failed'
            if detected and ('LegacyCorrectionSkipped' in reason or 'no bound Step5 payload' in reason):
                job['legacy_correction'] = {'skipped': True, 'note': reason}
                job['status'] = 'detection_completed'
                job.pop('traceback', None)
                job['reason'] = ''
                R46.save(folder / 'job.json', job)
            step5 = S5.step5_panel(panel, args.results, args.python, args.iterations) if detected else \
                {'status': 'skipped', 'reason': 'detection failed'}
            row = {'panel_id': panel['panel_id'], 'modality': panel['modality'], 'detection': job['status'],
                   'points_before': job.get('points_before'), 'step5': step5.get('status'),
                   'points_after': step5.get('points'), 'step5_reason': step5.get('reason', ''),
                   'detection_reason': (job.get('reason') or '')[:200],
                   'gpu_wait_seconds': round(waited, 1),
                   'detection_seconds': round(job['timing'].get('detection_process_seconds') or 0, 1),
                   'step5_seconds': step5.get('seconds'), 'finished_after_seconds': round(time.time() - started_at, 1)}
            return row
        finally:
            if slot:
                slot.release()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, p): p for p in panels}
        for future in as_completed(futures):
            panel = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {'panel_id': panel['panel_id'], 'status': 'driver_error',
                       'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            with lock:
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'traceback'}, ensure_ascii=False),
                      f'[{len(rows)}/{len(panels)}]', flush=True)

    snap_end = src_fingerprint.fingerprint(snapshot)
    live_end = src_fingerprint.fingerprint()
    snapshot_changed = src_fingerprint.compare(snap_start, snap_end)
    live_changed = src_fingerprint.compare(live_start, live_end)
    summary = {'panels': rows, 'snapshot': str(snapshot),
               'src_combined_sha256_before': snap_start['combined_sha256'],
               'src_combined_sha256_after': snap_end['combined_sha256'],
               'src_changed_during_run': snapshot_changed,
               'live_src_before': live_start['combined_sha256'], 'live_src_after': live_end['combined_sha256'],
               'live_src_files_changed_during_run_not_used': [c['file'] for c in live_changed],
               'workers': args.workers, 'gpu_slots': args.gpu_slots, 'threads_per_process': args.threads,
               'legacy_cli_correction': 'skipped; v46_after = GUI Step 5',
               'wall_seconds': round(time.time() - started_at, 1)}
    name = 'run_summary.json' if not args.ids else 'run_summary_ids.json'
    (output / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'gui_step5_summary.json').write_text(json.dumps(
        {'panels': [{'panel_id': r['panel_id'], 'status': r.get('step5'), 'points': r.get('points_after'),
                     'seconds': r.get('step5_seconds'), 'reason': r.get('step5_reason', '')} for r in rows],
         'src_before': snap_start['combined_sha256'], 'src_after': snap_end['combined_sha256'],
         'src_changed_during_step5': snapshot_changed}, indent=2), encoding='utf-8')
    print(f"done in {summary['wall_seconds']:.0f}s; snapshot changed: {len(snapshot_changed)} files; "
          f"live src changed meanwhile (not used): {len(live_changed)} files", flush=True)


if __name__ == '__main__':
    main()

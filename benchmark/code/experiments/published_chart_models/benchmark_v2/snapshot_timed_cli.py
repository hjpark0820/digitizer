"""Timing wrapper for one detection/correction CLI call that runs against a FROZEN src snapshot.

benchmark_pk/run_v46.py launches every CLI through its own `--timed-cli` wrapper, and that wrapper
puts the live repository src/ first on sys.path. When the CLI path points into a snapshot but
the import path still starts with the live src/, every module the CLI imports would come from
the live tree - so a snapshot would protect nothing. This wrapper calls the same timed_cli()
function with that module's SRC set to the snapshot, so the CLI script and all of its imports
come from the snapshot only.

Usage (built by run_parallel.py, never by hand):
  python snapshot_timed_cli.py <snapshot_src> <timing_json> -- <cli.py> <cli args...>
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    snapshot = Path(sys.argv[1]).resolve()
    timing_path = sys.argv[2]
    argv = sys.argv[sys.argv.index('--') + 1:]
    live_src = str((HERE.parents[2] / 'src').resolve())
    sys.path[:] = [p for p in sys.path if str(Path(p or '.').resolve()) != live_src]
    spec = importlib.util.spec_from_file_location('bench_run_v46_for_timing', HERE.parent / 'benchmark_pk' / 'run_v46.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SRC = snapshot
    try:
        module.timed_cli(argv, timing_path)
    finally:
        # the CLI may end with sys.exit(), so the leak check runs on every exit path
        loaded = [m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)
                  and str(Path(m.__file__).resolve()).startswith(live_src)]
        if loaded:
            print(f'SNAPSHOT LEAK: live src modules were imported: {loaded[:5]}', file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(97)


if __name__ == '__main__':
    main()

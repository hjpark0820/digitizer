"""Freeze production src/ into benchmark_v2/src_snapshots/<digest12>/src for one benchmark run.

Why: production src/ is edited by another session while benchmarks run. run_v1 and run_100c were
both invalidated because panels early and late in the run executed different code. A run that
executes a private copy cannot drift, however long it takes.

The copy is accepted only when the live fingerprint is identical before and after copying and the
copy's own fingerprint equals it (so no file was copied mid-edit). Excluded, because no v46 entry
point reads them (checked by grep: only the retired 1_point_detection_v3 / chart_* scripts do):
  data/ (427 MB synthetic training data), synthetic_plots/ (153 MB), __pycache__/, .claude/.
Prints the snapshot path; writes snapshot_fingerprint.json next to the copied src/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import src_fingerprint  # noqa: E402

EXCLUDE = {'data', 'synthetic_plots', '__pycache__', '.claude'}
# bw_pipeline.py loads SRC_DIR.parent / 'models' / 'chart_marker_net_v3.pth', i.e. a folder NEXT TO
# src/, so the repository-level models/ is copied next to the snapshot's src/ as well.
REPO_MODELS = src_fingerprint.SRC.parent / 'models'


def copy_models(root):
    import hashlib
    target = root / 'models'
    if not target.exists():
        shutil.copytree(REPO_MODELS, target)
    return {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(target.rglob('*')) if p.is_file()}


def ignore(directory, names):
    top = Path(directory).resolve() == src_fingerprint.SRC.resolve()
    return [n for n in names if n == '__pycache__' or (top and n in EXCLUDE)]


def main():
    for attempt in range(5):
        before = src_fingerprint.fingerprint()
        digest = before['combined_sha256'][:12]
        root = HERE / 'src_snapshots' / digest
        target = root / 'src'
        if target.exists():
            copy = src_fingerprint.fingerprint(target)
            if copy['combined_sha256'] == before['combined_sha256']:
                record_path = root / 'snapshot_fingerprint.json'
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if 'models_sha256' not in record:
                    record['models_sha256'] = copy_models(root)
                    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
                print(target)
                return
            raise SystemExit(f'{target} exists but does not match its own name; refusing to reuse it')
        staging = root / 'src_partial'
        if staging.exists():
            raise SystemExit(f'leftover partial snapshot {staging}; move it away first')
        shutil.copytree(src_fingerprint.SRC, staging, ignore=ignore)
        after = src_fingerprint.fingerprint()
        copy = src_fingerprint.fingerprint(staging)
        if before['combined_sha256'] == after['combined_sha256'] == copy['combined_sha256']:
            staging.rename(target)
            record = dict(copy, live_fingerprint=before['combined_sha256'], git_head=before['git_head'],
                          git_dirty_src=before['git_dirty_src'], excluded_top_level=sorted(EXCLUDE),
                          models_sha256=copy_models(root))
            (root / 'snapshot_fingerprint.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(target)
            return
        print(f'src changed while copying (attempt {attempt + 1}); retrying', file=sys.stderr)
        shutil.move(str(staging), str(root / f'discarded_{attempt}'))
    raise SystemExit('src kept changing during five copy attempts')


if __name__ == '__main__':
    main()

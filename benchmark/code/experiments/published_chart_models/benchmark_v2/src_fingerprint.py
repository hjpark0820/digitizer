"""Record which production code a benchmark run actually executed, and detect drift during it.

The first run of this benchmark was invalidated because src/color_*.py was being edited in another
session while the run was in progress: amivantamab1__p02 detected 16 points at 14:07 and 69 points
with the same command after the 14:11-14:39 edits. Nothing in the run metadata recorded that, so
the drift was only found when the user compared results by hand.

fingerprint()  sha256 + size + mtime of every .py under src/ (recursively), plus git HEAD and the
               dirty-file list, as one dict with a combined digest. src/.claude/ is skipped: it
               holds agent-session git worktrees (whole repository copies), which no v46 entry
               point imports and make_snapshot.py does not copy; counting them made the live and
               copied digests differ, so a snapshot could never be accepted.
compare(a, b)  the files whose digest changed between two fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / 'src'


def fingerprint(src=SRC):
    src = Path(src)
    files = {}
    combined = hashlib.sha256()
    for path in sorted(src.rglob('*.py')):
        parts = path.relative_to(src).parts
        if '__pycache__' in parts or parts[0] == '.claude':
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        rel = path.relative_to(src).as_posix()
        files[rel] = {'sha256': digest, 'bytes': len(data),
                      'mtime': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        combined.update(rel.encode() + digest.encode())
    try:
        head = subprocess.run(['git', '-C', str(src.parent), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(src.parent), 'status', '--porcelain', '--', 'src'],
                               capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        head, dirty = '', []
    return {'taken_at': datetime.now(timezone.utc).isoformat(), 'src': str(src), 'file_count': len(files),
            'combined_sha256': combined.hexdigest(), 'git_head': head, 'git_dirty_src': dirty, 'files': files}


def compare(before, after):
    names = set(before['files']) | set(after['files'])
    changed = []
    for name in sorted(names):
        a = before['files'].get(name, {}).get('sha256')
        b = after['files'].get(name, {}).get('sha256')
        if a != b:
            changed.append({'file': name, 'before': a, 'after': b,
                            'state': 'added' if a is None else 'removed' if b is None else 'modified'})
    return changed


def main():
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    value = fingerprint()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()

"""Immutable, image-linked result versions with shared evidence objects.

No detector is called here. Each version resolves to a complete v1 session so
the existing pixel/evidence validation and correction workers remain unchanged.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import correction_session_v46 as session

FORMAT = 'chartocode-v46-image-history-v2'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def validate(value):
    if not isinstance(value, dict) or value.get('format') != FORMAT:
        raise ValueError('Expected an image history JSON')
    versions, objects = value.get('versions'), value.get('objects')
    if not isinstance(versions, list) or not 1 <= len(versions) <= 1000 or not isinstance(objects, dict):
        raise ValueError('Invalid result version list')
    if not isinstance(value.get('source_image'), dict):
        raise ValueError('Missing original-image binding')
    for key, obj in objects.items():
        if not isinstance(obj, str) or key != hashlib.sha256(obj.encode('utf-8')).hexdigest():
            raise ValueError('Saved evidence object checksum mismatch')
    seen = set()
    for row in versions:
        ident = row.get('id')
        if not isinstance(ident, str) or not session.SAFE_ID.fullmatch(ident) or ident in seen:
            raise ValueError('Invalid or duplicate result version ID')
        parent = row.get('parent_id')
        if parent is not None and parent not in seen:
            raise ValueError('Missing or cyclic result version parent')
        snap = row.get('snapshot')
        if not isinstance(snap, dict) or snap.get('format') != session.LINKED_FORMAT:
            raise ValueError('Missing version snapshot')
        refs = snap.get('artifact_refs', {})
        if not isinstance(refs, dict) or set(refs)-set(session.JSON_FILES+(session.NPZ,)):
            raise ValueError('Unexpected version evidence')
        if any(ref not in objects for ref in refs.values()):
            raise ValueError('Missing version evidence object')
        if 'working_image_ref' in snap and snap['working_image_ref'] not in objects:
            raise ValueError('Missing prepared image object')
        if not isinstance(snap.get('edit_data', {}).get('curves'), list):
            raise ValueError('Missing version coordinates')
        seen.add(ident)
    if value.get('selected_version_id') not in seen:
        raise ValueError('Selected result version is unavailable')
    return value


def unpack(value, version_id=None):
    validate(value)
    ident = version_id or value['selected_version_id']
    row = next((v for v in value['versions'] if v['id'] == ident), None)
    if row is None:
        raise ValueError('Selected result version is unavailable')
    package = deepcopy(row['snapshot'])
    package['source_image'] = deepcopy(value['source_image'])
    package['artifacts'] = {name: session.loads(value['objects'][ref].encode('utf-8'))
                            for name, ref in package.pop('artifact_refs', {}).items()}
    if 'working_image_ref' in package:
        package['working_image_png_base64'] = session.loads(value['objects'][package.pop('working_image_ref')].encode('utf-8'))
    return package, {k: deepcopy(v) for k, v in row.items() if k != 'snapshot'}


def append(history, package, metadata):
    if history is None:
        history = dict(format=FORMAT, source_image=deepcopy(package['source_image']),
                       versions=[], objects={}, selected_version_id=None)
    if history['source_image']['file_sha256'] != package['source_image']['file_sha256']:
        raise ValueError('Result history belongs to another original image')
    snap = deepcopy(package)
    snap.pop('source_image', None)
    def store(obj):
        raw = canonical(obj)
        key = hashlib.sha256(raw.encode('utf-8')).hexdigest()
        history['objects'][key] = raw
        return key
    snap['artifact_refs'] = {name: store(obj) for name, obj in snap.pop('artifacts', {}).items()}
    if 'working_image_png_base64' in snap:
        snap['working_image_ref'] = store(snap.pop('working_image_png_base64'))
    row = dict(metadata, snapshot=snap)
    row['point_count'] = sum(len(c['points']) for c in snap['edit_data']['curves'])
    existing = next((v for v in history['versions'] if v['id'] == row['id']), None)
    if existing is not None:
        if existing['snapshot'] != snap or existing.get('parent_id') != row.get('parent_id'):
            raise ValueError('An immutable saved version cannot be replaced')
    else:
        history['versions'].append(row)
    history['selected_version_id'] = row['id']
    return history


def export_history(folder, original_bytes, filename, resolve_job, proposed=None):
    """Walk server-owned job ancestry, including manual input before Step 5.

    Imported histories are retained in full, including branches not selected.
    resolve_job is the server's strict job-ID resolver, never a client path.
    """
    visited = set()
    def linked(path, mode, edits=None):
        return session.export_linked_session(path, mode, original_bytes, filename, edits)
    def manual(history, package, parent):
        ident = 'manual_'+digest(dict(parent=parent, edit_data=package['edit_data']))[:24]
        return append(history, package, dict(id=ident, parent_id=parent, kind='manual',
                      label='Manual edits', created_at=now(), iterations=None))
    def collect(path):
        path = Path(path)
        if path in visited or len(visited) >= 200:
            raise ValueError('Invalid or excessively deep job ancestry')
        visited.add(path)
        meta = session.read(path/'session_meta.json')
        base = linked(path, meta['mode'])
        if (path/'imported_history.json').exists():
            history = validate(session.read(path/'imported_history.json'))
            if history['source_image']['file_sha256'] != base['source_image']['file_sha256']:
                raise ValueError('Imported history belongs to a different source image')
            history['selected_version_id'] = meta['version_id']
            return history
        history = None
        parent = None
        if meta.get('parent_job'):
            source = resolve_job(meta['parent_job'])
            history = collect(source)
            parent = history['selected_version_id']
            if (path/'step5_start_ed.json').exists():
                start = linked(source, session.read(source/'session_meta.json')['mode'],
                               session.read(path/'step5_start_ed.json'))
                previous, _ = unpack(history, parent)
                if previous['edit_data'] != start['edit_data']:
                    history = manual(history, start, parent)
                    parent = history['selected_version_id']
        ident = meta.get('version_id', path.parent.name)
        kind = meta.get('version_kind', 'imported' if meta.get('imported') else 'detection')
        label = {'detection': 'Initial detection', 'step5': 'Step 5 correction',
                 'imported': 'Imported result'}.get(kind, kind)
        return append(history, base, dict(id=ident, parent_id=parent, kind=kind, label=label,
                      created_at=meta.get('created_at', now()), iterations=meta.get('iterations')))
    history = collect(folder)
    if proposed is not None:
        meta = session.read(Path(folder)/'session_meta.json')
        package = linked(folder, meta['mode'], proposed)
        current, _ = unpack(history)
        if package['edit_data'] != current['edit_data']:
            history = manual(history, package, history['selected_version_id'])
    validate(history)
    if len(json.dumps(history, ensure_ascii=False).encode('utf-8')) > session.MAX_JSON:
        raise ValueError('Result history exceeds 128 MB; existing saved JSON was not changed')
    return history

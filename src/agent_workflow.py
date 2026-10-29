"""Local, durable image queue for a cooperating image-capable coding agent.

The server never launches an AI model. The agent reads image requests and supplies
regions/OCR through the documented APIs. Source images are immutable snapshots;
each retry creates a separate digitizer job and preserves earlier artifacts.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

# The shared browser editor and original-image sidecars support these formats.
# Do not offer TIFF/BMP/WebP batch results that cannot be reviewed/saved there.
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMAGES = 1000
ACTIVE_STATES = {'pending_regions', 'ready', 'waiting_agent', 'running'}
OCR_MODES = {'tesseract', 'agent', 'auto-agent'}


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{12}', value):
        raise HTTPException(400, 'Invalid workflow identifier')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise HTTPException(400, name + ' must be true or false')
    return value


def _rectangle(value, width, height, name):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise HTTPException(400, name + ' must contain four integer image-pixel coordinates')
    left, top, right, bottom = value
    if not (0 <= left < right < width and 0 <= top < bottom < height):
        raise HTTPException(400, name + ' must be an inclusive rectangle inside the current image')
    return value


async def _json(request, limit=128 * 1024):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, 'Workflow request is too large')
    try:
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error


class AgentWorkflow:
    def __init__(self, run_item, register_job, ocr_pending, health, root=None):
        self.root = Path(root or os.environ.get('CHARTOCODE_WORKFLOW_ROOT') or
                         Path.home() / '.chartocode' / 'agent_workflows').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_item, self.register_job = run_item, register_job
        self.ocr_pending, self.health = ocr_pending, health
        self.condition = threading.Condition(threading.RLock())
        self.batches = {}
        self.agent = {'last_seen': None, 'agent_id': None}
        self.defaults = {'region_mode': 'manual', 'ocr_mode': 'tesseract'}
        try:
            self.defaults.update(self._settings(json.loads((self.root / 'settings.json').read_text(encoding='utf-8'))))
            # Migrate old session defaults, never historical batch/job options.
            if self.defaults['region_mode'] == 'agent':
                self.defaults['ocr_mode'] = 'agent'
        except (OSError, ValueError, HTTPException):
            pass
        try:
            index = json.loads((self.root / 'index.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            index = {}
        if not isinstance(index, dict):
            index = {}
        for batch_id, filename in index.items():
            try:
                _identifier(batch_id)
                batch = json.loads(Path(filename).read_text(encoding='utf-8'))
                if batch.get('batch_id') != batch_id or Path(filename).resolve() != self._manifest(batch).resolve():
                    continue
                self.batches[batch_id] = batch
                for item in batch['items']:
                    if item['status'] == 'running':
                        item.update(status='failed', error='The server stopped during processing. Retry to create a new job.')
                    elif item['status'] == 'waiting_agent':
                        item['status'] = 'ready'
                jobs = Path(batch['output_dir']) / 'jobs'
                if jobs.is_dir():
                    for folder in jobs.iterdir():
                        if re.fullmatch(r'[a-f0-9]{12}', folder.name) and not folder.is_symlink() and (folder / 'out').is_dir():
                            register_job(folder.name, folder / 'out')
                self._persist(batch)
            except (OSError, ValueError, TypeError, KeyError, HTTPException):
                continue
        self.worker = None
        self.stopping = True

    def start(self):
        with self.condition:
            if self.worker is not None and self.worker.is_alive():
                return
            self.stopping = False
            self.worker = threading.Thread(target=self._work, name='digitizer-agent-workflow', daemon=True)
            self.worker.start()

    def stop(self):
        # Already-running subprocesses retain their own timeout. Shutdown never
        # starts another item or leaves an idle thread polling the registry.
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=1)

    @staticmethod
    def _settings(data):
        if not isinstance(data, dict):
            raise HTTPException(400, 'Settings must be a JSON object')
        result = {}
        if 'region_mode' in data:
            if data['region_mode'] not in {'manual', 'agent'}:
                raise HTTPException(400, 'region_mode must be manual or agent')
            result['region_mode'] = data['region_mode']
        if 'ocr_mode' in data:
            if data['ocr_mode'] not in OCR_MODES:
                raise HTTPException(400, 'ocr_mode must be tesseract, agent or auto-agent')
            result['ocr_mode'] = data['ocr_mode']
        return result

    def settings(self, changes=None):
        with self.condition:
            if changes is not None:
                self.defaults = self._selected_settings(changes)
                _save(self.root / 'settings.json', self.defaults)
            return dict(self.defaults)

    def _selected_settings(self, changes):
        selected = self._settings(changes)
        settings = dict(self.defaults, **selected)
        if settings['region_mode'] == 'agent':
            if selected.get('ocr_mode', 'agent') != 'agent':
                raise HTTPException(400, 'Agent-selected regions require agent OCR; use region_mode=manual to choose another OCR engine')
            settings['ocr_mode'] = 'agent'
        return settings

    def heartbeat(self, agent_id='operating-agent'):
        if not isinstance(agent_id, str) or not 1 <= len(agent_id) <= 160:
            raise HTTPException(400, 'agent_id must be a short nonempty name')
        with self.condition:
            self.agent.update(last_seen=time.time(), agent_id=agent_id)
            self.condition.notify_all()
            return self.agent_status()

    def agent_status(self):
        with self.condition:
            return dict(self.agent, active=bool(self.agent['last_seen'] and time.time() - self.agent['last_seen'] < 90),
                        heartbeat_ttl_seconds=90)

    @staticmethod
    def _manifest(batch):
        return Path(batch['output_dir']) / 'manifest.json'

    def _persist(self, batch):
        batch['updated_at'] = time.time()
        _save(self._manifest(batch), batch)

    def _get(self, batch_id, item_id=None):
        batch = self.batches.get(_identifier(batch_id))
        if batch is None:
            raise HTTPException(404, 'Batch not found on this server')
        if item_id is None:
            return batch
        _identifier(item_id)
        item = next((v for v in batch['items'] if v['item_id'] == item_id), None)
        if item is None:
            raise HTTPException(404, 'Batch image not found')
        return batch, item

    def _options(self, data):
        settings = self._selected_settings(data)
        mode = data.get('mode', 'auto')
        if mode not in {'auto', 'color', 'bw'}:
            raise HTTPException(400, 'mode must be auto, color or bw')
        settings.update(mode=mode, correct=_boolean(data.get('correct', False), 'correct'),
                        fast_detection=_boolean(data.get('fast_detection', False), 'fast_detection'))
        return settings

    def _new(self, data, source_folder=None):
        options = self._options(data)
        target = data.get('output_folder') or str(Path.home() / 'Digitizer Results')
        if not isinstance(target, str) or not target.strip():
            raise HTTPException(400, 'output_folder must be a local directory path')
        target = Path(target).expanduser().resolve()
        batch_id = uuid.uuid4().hex[:12]
        output = target / ('batch_' + batch_id)
        # Unique directories are created exclusively; originals never become outputs.
        output.mkdir(parents=True, exist_ok=False)
        (output / 'images').mkdir()
        (output / 'jobs').mkdir()
        return dict(batch_id=batch_id, output_dir=str(output), source_folder=source_folder,
                    options=options, items=[], cancelled=False, created_at=time.time(), updated_at=time.time())

    def _snapshot(self, batch, name, content, source_path=None):
        item_id = uuid.uuid4().hex[:12]
        item = dict(item_id=item_id, name=name, status='pending_regions', error=None, result=None,
                    regions=None, source_path=source_path, attempts=0, history=[])
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            item.update(status='failed', error='Batch review supports PNG and JPEG images')
        elif len(content) > MAX_IMAGE_BYTES:
            item.update(status='failed', error='Image exceeds 100 MiB')
        else:
            try:
                raster = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:
                raster = None
            if raster is None:
                item.update(status='failed', error='Image could not be decoded')
            elif raster.shape[0] * raster.shape[1] > 40_000_000:
                item.update(status='failed', error='Image exceeds the 40-million-pixel limit; resize it first')
            else:
                height, width = raster.shape[:2]
                path = Path(batch['output_dir']) / 'images' / (item_id + suffix)
                path.write_bytes(content)
                item.update(image_path=str(path), image_sha256=hashlib.sha256(content).hexdigest(), width=width, height=height,
                            image_url=f'/agent-workflow/batches/{batch["batch_id"]}/images/{item_id}',
                            answer_url=f'/agent-workflow/batches/{batch["batch_id"]}/items/{item_id}/regions')
        batch['items'].append(item)

    def _commit(self, batch):
        with self.condition:
            self.batches[batch['batch_id']] = batch
            self._persist(batch)
            _save(self.root / 'index.json', {key: str(self._manifest(value)) for key, value in self.batches.items()})
            self.condition.notify_all()
            return self.snapshot(batch['batch_id'])

    def create_folder(self, data):
        folder = data.get('folder')
        if not isinstance(folder, str) or not folder.strip():
            raise HTTPException(400, 'Choose a local folder first')
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(400, 'The chosen folder does not exist on the server computer')
        recursive = _boolean(data.get('recursive', False), 'recursive')
        requested = data.get('files')
        if requested is not None:
            if not isinstance(requested, list) or not requested or len(requested) > MAX_IMAGES:
                raise HTTPException(400, 'files must be a nonempty list of relative image paths')
            normalized = set()
            for value in requested:
                if not isinstance(value, str):
                    raise HTTPException(400, 'files must contain relative image paths')
                relative = Path(value.replace('\\', '/'))
                if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                    raise HTTPException(400, 'files must stay inside the selected folder')
                if not recursive and len(relative.parts) != 1:
                    raise HTTPException(400, 'Enable recursion to select images in subfolders')
                normalized.add(relative.as_posix())
            requested = normalized
        # Resolve and snapshot only images inside this explicit source directory.
        candidates = []
        excluded = [Path(v['output_dir']).resolve() for v in self.batches.values()]
        for directory, dirs, files in os.walk(folder, followlinks=False):
            directory = Path(directory)
            dirs[:] = [d for d in dirs if not (directory / d).is_symlink() and
                       not any((directory / d).resolve().is_relative_to(p) for p in excluded)] if recursive else []
            for name in sorted(files):
                path = directory / name
                if path.suffix.lower() not in IMAGE_SUFFIXES or path.is_symlink():
                    continue
                resolved = path.resolve()
                if requested is not None and path.relative_to(folder).as_posix() not in requested:
                    continue
                if resolved.is_relative_to(folder) and not any(resolved.is_relative_to(p) for p in excluded):
                    candidates.append(resolved)
            if len(candidates) > MAX_IMAGES:
                raise HTTPException(400, 'A batch supports at most 1000 images; choose a smaller scope')
        if not candidates:
            raise HTTPException(400, 'No supported images found in the chosen folder')
        if requested is not None:
            missing = requested - {p.relative_to(folder).as_posix() for p in candidates}
            if missing:
                raise HTTPException(400, 'Selected images are missing, unsupported or outside scope: ' + ', '.join(sorted(missing)[:10]))
        if sum(p.stat().st_size for p in candidates) > MAX_BATCH_BYTES:
            raise HTTPException(413, 'Batch source images exceed 2 GiB; choose a smaller scope')
        batch = self._new(data, str(folder))
        batch['recursive'] = recursive
        batch['selected_files'] = sorted(requested) if requested is not None else None
        for path in sorted(candidates):
            try:
                with path.open('rb') as stream:
                    content = stream.read(MAX_IMAGE_BYTES + 1)
                self._snapshot(batch, str(path.relative_to(folder)), content, str(path))
            except OSError as error:
                batch['items'].append(dict(item_id=uuid.uuid4().hex[:12], name=str(path.relative_to(folder)),
                    status='failed', error=str(error), source_path=str(path), regions=None, result=None, attempts=0, history=[]))
        return self._commit(batch)

    def create_uploads(self, uploads, data):
        if not uploads or len(uploads) > MAX_IMAGES:
            raise HTTPException(400, 'Upload between 1 and 1000 images')
        # Multipart UploadFile objects have seekable spooled files. Check actual
        # full sizes before creating a durable directory; a rejected request must
        # not leave unregistered snapshots. Bounded reads below still prevent an
        # oversized individual image from being loaded wholly into memory.
        total = 0
        for upload in uploads:
            try:
                upload.file.seek(0, os.SEEK_END)
                total += upload.file.tell()
                upload.file.seek(0)
            except (OSError, ValueError) as error:
                raise HTTPException(400, 'Uploaded image could not be read') from error
            if total > MAX_BATCH_BYTES:
                raise HTTPException(413, 'Batch images exceed 2 GiB')
        batch = self._new(data)
        for upload in uploads:
            content = upload.file.read(MAX_IMAGE_BYTES + 1)
            # A client filename is display-only, never a filesystem destination.
            self._snapshot(batch, (upload.filename or 'image.png')[:1024], content)
        return self._commit(batch)

    def _decorate(self, batch, include_items):
        counts = {}
        for item in batch['items']:
            counts[item['status']] = counts.get(item['status'], 0) + 1
        active = any(state in ACTIVE_STATES for state in counts)
        status = ('cancelling' if counts.get('running') else 'cancelled') if batch['cancelled'] else (
            'running' if counts.get('running') else 'waiting_agent' if counts.get('waiting_agent') else
            'waiting_regions' if counts.get('pending_regions') else 'queued' if active else
            'completed_with_issues' if counts.get('failed') or counts.get('needs_review') else 'completed')
        view = dict(batch) if include_items else {k: batch[k] for k in ('batch_id', 'output_dir', 'source_folder', 'options', 'created_at', 'updated_at')}
        view.update(status=status, counts=counts, total=len(batch['items']), completed=counts.get('completed', 0),
                    pending_ocr_count=self.ocr_pending(Path(batch['output_dir']) / 'jobs'),
                    manifest_path=str(self._manifest(batch)), review_dir=str(Path(batch['output_dir']) / 'review'))
        view = copy.deepcopy(view)
        if include_items:
            for item in view['items']:
                item['pending_ocr_count'] = self.ocr_pending(Path(batch['output_dir']) / 'jobs' / item['current_job_id']) if item.get('current_job_id') else 0
                item['waiting_ocr'] = bool(item['pending_ocr_count'])
        view['download_url'] = f'/agent-workflow/batches/{batch["batch_id"]}/results.zip'
        return view

    def job_started(self, batch_id, item_id, job_id):
        with self.condition:
            batch, item = self._get(batch_id, item_id)
            item['current_job_id'] = job_id
            self._persist(batch)

    @staticmethod
    def _review_pair(batch, item, result):
        """Publish a same-name linked pair for the unchanged folder editor.

        The paired PNG is the job's working raster, not a replacement of the
        source snapshot. Existing review files (including manual edits) are never
        overwritten; a later retry receives a distinct job-suffixed basename.
        """
        if not result or not result.get('ok'):
            return {}
        job_id = _identifier(result.get('job_id'))
        if item.get('review_job_id') == job_id and all(Path(item.get(k, '')).is_file()
                for k in ('review_image_path', 'review_json_path')):
            return {k: item[k] for k in ('review_job_id', 'review_image_path', 'review_json_path')}
        from correction_session_v46 import export_linked_session
        output = Path(batch['output_dir'])
        folder = output / 'jobs' / job_id / 'out'
        image_path = folder / 'input.png'
        if image_path.is_symlink() or not image_path.resolve().is_relative_to(output.resolve()):
            raise ValueError('Review image is outside the batch output')
        image_bytes = image_path.read_bytes()
        review = output / 'review'
        if review.is_symlink() or not review.resolve().is_relative_to(output.resolve()):
            raise ValueError('Review directory is outside the batch output')
        review.mkdir(exist_ok=True)
        stem = _identifier(item['item_id'])
        if (review / (stem + '.png')).exists() or (review / (stem + '.json')).exists():
            stem += '_' + job_id
        png, json_path = review / (stem + '.png'), review / (stem + '.json')
        package = export_linked_session(folder, result['mode'], image_bytes, png.name)
        # Complete JSON first. Folder browsing lists only image+JSON pairs; use
        # exclusive creation so a user-edited sidecar cannot be replaced.
        with json_path.open('x', encoding='utf-8') as stream:
            json.dump(package, stream, ensure_ascii=False, indent=2, allow_nan=False)
        with png.open('xb') as stream:
            stream.write(image_bytes)
        return dict(review_job_id=job_id, review_image_path=str(png), review_json_path=str(json_path))

    def _review_result(self, batch, item, result):
        try:
            return dict(self._review_pair(batch, item, result), review_error=None)
        except Exception as error:
            # A convenience export failure does not invalidate detection or its
            # original correction-session artifacts and explicit editor link.
            return {'review_error': 'Review-folder export failed: ' + str(error)[:2000]}

    def results_zip(self, batch_id):
        """Portable paired sessions plus spreadsheets; originals stay untouched."""
        with self.condition:
            batch = self._get(batch_id)
            output = Path(batch['output_dir'])
            items = copy.deepcopy(batch['items'])
            batch_copy = copy.deepcopy(batch)
        # Also supply review pairs for results created before this exporter was
        # introduced, without rerunning detection or changing source images.
        for item in items:
            if isinstance(item.get('result'), dict) and item['result'].get('ok'):
                review = self._review_result(batch_copy, item, item['result'])
                with self.condition:
                    current_batch, current = self._get(batch_id, item['item_id'])
                    if (current.get('result') or {}).get('job_id') == item['result'].get('job_id'):
                        current.update(review)
                        self._persist(current_batch)
        with self.condition:
            snapshot = self._decorate(self._get(batch_id), True)
            items = copy.deepcopy(self._get(batch_id)['items'])
        destination = output / 'downloads'
        destination.mkdir(exist_ok=True)
        path = destination / ('results_' + uuid.uuid4().hex[:12] + '.zip')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps(snapshot, ensure_ascii=False, indent=2))
            archive.writestr('README.txt', 'Each results/<item>/input.png is paired with correction_session.json.\n'
                            'Load that pair in the editor to review/edit or continue saved-evidence correction.\n'
                            'For existing folder browsing, extract this archive and choose the review folder:\n'
                            'each review/<item>.png has a linked same-name JSON; use the existing arrows to browse.\n'
                            'A retry may add a job-ID suffix to preserve earlier reviewed files.\n'
                            'The manifest maps item IDs to original filenames. Review PNGs are working images;\n'
                            'source snapshots and result images may differ when fast detection was enabled.\n')
            review_dir = output / 'review'
            if review_dir.is_dir() and not review_dir.is_symlink():
                for png in sorted(review_dir.glob('*.png')):
                    paired = png.with_suffix('.json')
                    if (re.fullmatch(r'[a-f0-9]{12}(?:_[a-f0-9]{12})?', png.stem) and paired.is_file()
                            and not png.is_symlink() and not paired.is_symlink()):
                        archive.write(png, 'review/' + png.name)
                        archive.write(paired, 'review/' + paired.name)
            for item in items:
                original = item.get('image_path')
                if original:
                    original = Path(original)
                    if original.is_file() and not original.is_symlink() and original.resolve().is_relative_to(output.resolve()):
                        archive.write(original, 'source_images/' + original.name)
                job_id = (item.get('result') or {}).get('job_id')
                if not job_id:
                    continue
                folder = output / 'jobs' / _identifier(job_id) / 'out'
                for name in ('input.png', 'correction_session.json', 'current_data.xlsx', 'edit_data.json',
                             'data_points_overlay.png', 'pipeline.log', 'axis_calibration.json', 'ocr_settings.json'):
                    file = folder / name
                    if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(output.resolve()):
                        archive.write(file, 'results/' + item['item_id'] + '/' + name)
        return path

    def snapshot(self, batch_id):
        with self.condition:
            return self._decorate(self._get(batch_id), True)

    def status(self):
        with self.condition:
            batches = [self._decorate(b, False) for b in reversed(list(self.batches.values()))]
        return dict(tesseract=self.health(), agent=self.agent_status(), settings=self.settings(), batches=batches,
                    launches_agent=False, default_output_folder=str(Path.home() / 'Digitizer Results'))

    def requests(self):
        with self.condition:
            return {'agent': self.agent_status(), 'requests': [dict(batch_id=b['batch_id'], region_mode=b['options']['region_mode'],
                    **{k: i[k] for k in ('item_id', 'name', 'image_path', 'image_url', 'image_sha256', 'width', 'height', 'answer_url')})
                for b in self.batches.values() if not b['cancelled'] and b['options']['region_mode'] == 'agent'
                for i in b['items'] if i['status'] == 'pending_regions']}

    def regions(self, batch_id, item_id, data):
        with self.condition:
            batch, item = self._get(batch_id, item_id)
            if batch['cancelled'] or item['status'] not in {'pending_regions', 'needs_review', 'failed', 'ready', 'waiting_agent'}:
                raise HTTPException(409, 'This image is already running, completed or cancelled')
            if not item.get('image_path'):
                raise HTTPException(409, 'This image could not be loaded; create a new batch with a readable image')
            if data.get('image_sha256') != item['image_sha256']:
                raise HTTPException(409, 'Image hash mismatch; inspect the current image before selecting regions')
            if hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest() != item['image_sha256']:
                raise HTTPException(409, 'The saved image changed; create a new batch')
            if data.get('status', 'ok') == 'unreadable':
                reason = data.get('reason')
                if not isinstance(reason, str) or not reason.strip():
                    raise HTTPException(400, 'Explain why the regions require review')
                item.update(status='needs_review', error=reason[:2000], regions=None)
            elif data.get('status', 'ok') == 'ok':
                plot = _rectangle(data.get('plot_area'), item['width'], item['height'], 'plot_area')
                if 'legend_box' not in data:
                    raise HTTPException(400, 'Provide legend_box or explicitly choose no_legend with legend_box=null')
                legend = data['legend_box']
                no_legend = data.get('no_legend', False)
                _boolean(no_legend, 'no_legend')
                if legend is None:
                    if not no_legend:
                        raise HTTPException(400, 'A missing legend requires no_legend=true; do not guess')
                else:
                    if no_legend:
                        raise HTTPException(400, 'no_legend conflicts with a supplied legend_box')
                    legend = _rectangle(legend, item['width'], item['height'], 'legend_box')
                item.update(regions=dict(plot_area=plot, legend_box=legend, no_legend=no_legend,
                            selected_by=str(data.get('selected_by', batch['options']['region_mode']))[:160]),
                            status='ready', error=None)
            else:
                raise HTTPException(400, 'status must be ok or unreadable')
            # Only an explicitly agent-supplied region answer is a heartbeat.
            if data.get('selected_by', batch['options']['region_mode']) == 'agent':
                self.heartbeat(data.get('agent_id', 'operating-agent'))
            self._persist(batch)
            self.condition.notify_all()
            return dict(accepted=True, item_id=item_id, status=item['status'])

    def cancel(self, batch_id):
        with self.condition:
            batch = self._get(batch_id)
            batch['cancelled'] = True
            for item in batch['items']:
                if item['status'] in {'pending_regions', 'ready', 'waiting_agent'}:
                    item['status'] = 'cancelled'
            self._persist(batch)
            self.condition.notify_all()
            return self.snapshot(batch_id)

    def retry(self, batch_id, data):
        with self.condition:
            batch = self._get(batch_id)
            selected = data.get('item_ids')
            if selected is not None and (not isinstance(selected, list) or any(not isinstance(v, str) for v in selected)):
                raise HTTPException(400, 'item_ids must be a list')
            states = {'failed', 'cancelled'}
            if _boolean(data.get('include_review', False), 'include_review'):
                states.add('needs_review')
            for item in batch['items']:
                if item['status'] in states and item.get('image_path') and (selected is None or item['item_id'] in selected):
                    if item.get('result'):
                        item['history'].append(item['result'])
                    item.update(status='ready' if item.get('regions') else 'pending_regions', result=None, error=None)
            batch['cancelled'] = False
            self._persist(batch)
            self.condition.notify_all()
            return self.snapshot(batch_id)

    def _work(self):
        while True:
            with self.condition:
                if self.stopping:
                    return
                selected = None
                for batch in self.batches.values():
                    if batch['cancelled']:
                        continue
                    for item in batch['items']:
                        if item['status'] not in {'ready', 'waiting_agent'}:
                            continue
                        mode = batch['options']['ocr_mode']
                        needs_agent = mode == 'agent' or mode == 'auto-agent' and not self.health()['available']
                        if needs_agent and not self.agent_status()['active']:
                            if item['status'] != 'waiting_agent':
                                item['status'] = 'waiting_agent'
                                self._persist(batch)
                            continue
                        if mode == 'tesseract' and not self.health()['available']:
                            item.update(status='needs_review', error='Tesseract is unavailable. Choose agent OCR for a new batch, or install Tesseract and retry.')
                            self._persist(batch)
                            continue
                        selected = batch, item
                        break
                    if selected:
                        break
                if selected is None:
                    self.condition.wait(timeout=5)
                    continue
                batch, item = selected
                item.update(status='running', started_at=time.time(), attempts=item['attempts'] + 1, error=None)
                self._persist(batch)
                batch_id, item_id = batch['batch_id'], item['item_id']
                snapshot_batch, snapshot_item = copy.deepcopy(batch), copy.deepcopy(item)
            try:
                result = asyncio.run(self.run_item(snapshot_batch, snapshot_item))
                state = 'completed' if result.get('ok') else 'failed'
                error = None if state == 'completed' else result.get('summary', 'Detection failed; inspect the job log')
            except Exception as failure:
                result, state = None, 'failed'
                error = str(getattr(failure, 'detail', failure))[:4000]
            review = self._review_result(snapshot_batch, snapshot_item, result)
            with self.condition:
                batch, item = self._get(batch_id, item_id)
                item.update(status=state, result=result, error=error, finished_at=time.time(), **review)
                self._persist(batch)
                self.condition.notify_all()


def attach_routes(app, workflow):
    @app.get('/agent-workflow/status')
    def status():
        return JSONResponse(workflow.status(), headers={'Cache-Control': 'no-store'})

    @app.get('/agent-workflow/settings')
    def get_settings():
        return workflow.settings()

    @app.post('/agent-workflow/settings')
    async def settings(request: Request):
        return workflow.settings(await _json(request))

    @app.post('/agent-workflow/heartbeat')
    async def heartbeat(request: Request):
        return workflow.heartbeat((await _json(request)).get('agent_id', 'operating-agent'))

    @app.get('/agent-workflow/requests')
    def requests():
        return JSONResponse(workflow.requests(), headers={'Cache-Control': 'no-store'})

    @app.post('/agent-workflow/batches')
    async def create(request: Request):
        return await run_in_threadpool(workflow.create_folder, await _json(request))

    @app.post('/agent-workflow/batches/upload')
    async def upload(files: list[UploadFile] = File(...), options: str = Form('{}')):
        try:
            data = json.loads(options)
            if not isinstance(data, dict):
                raise ValueError('options must be an object')
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return await run_in_threadpool(workflow.create_uploads, files, data)

    @app.get('/agent-workflow/batches/{batch_id}')
    def batch(batch_id: str):
        return JSONResponse(workflow.snapshot(batch_id), headers={'Cache-Control': 'no-store'})

    @app.get('/agent-workflow/batches/{batch_id}/results.zip')
    def results_zip(batch_id: str):
        path = workflow.results_zip(batch_id)
        return FileResponse(path, media_type='application/zip', filename='digitizer_' + batch_id + '.zip')

    @app.get('/agent-workflow/batches/{batch_id}/images/{item_id}')
    def image(batch_id: str, item_id: str):
        with workflow.condition:
            _, item = workflow._get(batch_id, item_id)
            if not item.get('image_path'):
                raise HTTPException(404, 'Image snapshot unavailable')
            return FileResponse(item['image_path'], headers={'Cache-Control': 'no-store'})

    @app.post('/agent-workflow/batches/{batch_id}/items/{item_id}/regions')
    async def regions(batch_id: str, item_id: str, request: Request):
        return workflow.regions(batch_id, item_id, await _json(request))

    @app.post('/agent-workflow/batches/{batch_id}/cancel')
    def cancel(batch_id: str):
        return workflow.cancel(batch_id)

    @app.post('/agent-workflow/batches/{batch_id}/retry')
    async def retry(batch_id: str, request: Request):
        return workflow.retry(batch_id, await _json(request))

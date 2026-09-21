"""Tesseract first, with an opt-in local file handoff to the operating AI agent.

The host agent must view each PNG and submit an answer. This module neither
launches an assistant nor calls a paid vision API. See AI_OCR.md for the protocol.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


class Output:
    DICT = 'dict'


class AgentOCRStopped(SystemExit):
    """Stop the worker: legacy OCR handlers must not silently swallow this error."""


def agent_enabled():
    return os.environ.get('CHARTOCODE_OCR_MODE', '').lower() in ('auto-agent', 'agent')


@lru_cache(maxsize=1)
def _tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return pytesseract
    except (ImportError, OSError, RuntimeError):
        return None


def available():
    return agent_enabled() or _tesseract() is not None


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _request_path(directory, request_id):
    if not isinstance(request_id, str) or re.fullmatch(r'[a-f0-9]{64}', request_id) is None:
        raise ValueError('Invalid OCR request ID')
    return Path(directory) / 'requests' / (request_id + '.json')


def read_request(directory, request_id):
    return json.loads(_request_path(directory, request_id).read_text(encoding='utf-8'))


def pending_requests(directory):
    result = []
    for path in sorted((Path(directory) / 'requests').glob('*.json')):
        try:
            request = read_request(directory, path.stem)
            if request['status'] == 'pending' and request['deadline'] > time.time():
                result.append(request)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def _validate_answer(request, answer):
    if not isinstance(answer, dict):
        raise ValueError('OCR answer must be a JSON object')
    for key in ('request_id', 'image_sha256'):
        if answer.get(key) != request[key]:
            raise ValueError(f'OCR answer {key} does not match the snippet')
    status = answer.get('status')
    if status not in ('ok', 'unreadable'):
        raise ValueError('OCR status must be ok or unreadable')
    if status == 'unreadable':
        return
    if request['operation'] == 'text':
        if not isinstance(answer.get('text'), str) or len(answer['text']) > 16384:
            raise ValueError('OCR text must be a string of at most 16384 characters')
        return
    words = answer.get('words')
    if not isinstance(words, list) or len(words) > 4096:
        raise ValueError('OCR words must be an array of at most 4096 word boxes')
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get('text'), str) or len(word['text']) > 1024:
            raise ValueError('Each OCR word needs text of at most 1024 characters')
        for key in ('left', 'top', 'width', 'height', 'line_num'):
            if type(word.get(key)) is not int or word[key] < (1 if key in ('width', 'height', 'line_num') else 0):
                raise ValueError(f'Invalid word {key}: use snippet pixel integers and positive line numbers')
        if word['left'] + word['width'] > request['width'] or word['top'] + word['height'] > request['height']:
            raise ValueError('OCR word box extends outside the snippet')
        confidence = word.get('confidence')
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 100:
            raise ValueError('OCR word confidence must be between 0 and 100')


def submit_answer(directory, request_id, answer):
    """Validate the image binding and publish a complete answer atomically."""
    request = read_request(directory, request_id)
    _validate_answer(request, answer)
    target = Path(directory) / 'answers' / (request_id + '.json')
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) == answer:
            return
        raise ValueError('This OCR request already has a different answer')
    if request['status'] != 'pending' or request['deadline'] <= time.time():
        raise ValueError('OCR request is no longer waiting; start the operation again')
    _write_json(target, answer)


def _word_data(answer):
    keys = ('level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
            'left', 'top', 'width', 'height', 'conf', 'text')
    data = {key: [] for key in keys}
    for number, word in enumerate(answer.get('words', []), 1):
        row = dict(level=5, page_num=1, block_num=1, par_num=1,
                   line_num=word['line_num'], word_num=number, conf=word['confidence'])
        row.update({key: word[key] for key in ('left', 'top', 'width', 'height', 'text')})
        for key in keys:
            data[key].append(row[key])
    return data


def _agent_read(image, operation, config, lang):
    from PIL import Image
    directory = os.environ.get('CHARTOCODE_OCR_DIR')
    if not directory:
        raise AgentOCRStopped('AI OCR requires CHARTOCODE_OCR_DIR (a queue dedicated to this job).')
    directory = Path(directory).expanduser().resolve()
    try:
        wait_seconds = float(os.environ.get('CHARTOCODE_OCR_WAIT_SECONDS', '300'))
        if not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 3600:
            raise ValueError()
    except ValueError:
        raise AgentOCRStopped('CHARTOCODE_OCR_WAIT_SECONDS must be greater than 0 and at most 3600.')
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            raster = opened.copy()
    else:
        raster = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buffer = io.BytesIO()
    raster.save(buffer, format='PNG')
    png = buffer.getvalue()
    image_hash = hashlib.sha256(png).hexdigest()
    identity = json.dumps([image_hash, operation, config, lang], ensure_ascii=False)
    request_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    snippet = directory / 'snippets' / (request_id + '.png')
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_bytes(png)
    request_path = _request_path(directory, request_id)
    request = dict(version=1, request_id=request_id, image_sha256=image_hash,
                   image_path=str(snippet), width=raster.width, height=raster.height,
                   operation=operation, config=config, lang=lang, status='pending',
                   created_at=time.time(), deadline=time.time() + wait_seconds)
    answer_path = directory / 'answers' / (request_id + '.json')
    # Identical snippets/configuration can reuse an answer inside the same job.
    _write_json(request_path, request)
    if not answer_path.exists():
        print(f'[AI OCR waiting] {request_path} | image: {snippet}', flush=True)
    while True:
        if answer_path.exists():
            try:
                answer = json.loads(answer_path.read_text(encoding='utf-8'))
                _validate_answer(request, answer)
            except (OSError, ValueError, TypeError) as error:
                request.update(status='failed', error=str(error))
                _write_json(request_path, request)
                raise AgentOCRStopped(f'Invalid AI OCR answer: {error}') from error
            request.update(status='answered', answered_at=time.time(), answer_status=answer['status'])
            _write_json(request_path, request)
            if answer['status'] == 'unreadable':
                return '' if operation == 'text' else _word_data({})
            return answer['text'] if operation == 'text' else _word_data(answer)
        if time.time() >= request['deadline']:
            request.update(status='timed_out')
            _write_json(request_path, request)
            raise AgentOCRStopped(f'AI OCR timed out waiting for {request_id}. '
                                  'The operating agent must view the snippet and submit an answer; rerun the operation.')
        time.sleep(min(0.2, wait_seconds))


def _recognize(image, operation, config='', lang=None, **kwargs):
    engine = None if os.environ.get('CHARTOCODE_OCR_MODE', '').lower() == 'agent' else _tesseract()
    if engine is not None:
        try:
            if lang is not None:
                kwargs['lang'] = lang
            if operation == 'text':
                return engine.image_to_string(image, config=config, **kwargs)
            return engine.image_to_data(image, config=config, output_type=engine.Output.DICT, **kwargs)
        except (OSError, RuntimeError):
            if not agent_enabled():
                raise
    elif not agent_enabled():
        raise RuntimeError('Tesseract is unavailable. Enable AI OCR only when an agent will answer the local queue.')
    return _agent_read(image, operation, config, lang)


def image_to_string(image, config='', lang=None, **kwargs):
    return _recognize(image, 'text', config=config, lang=lang, **kwargs)


def image_to_data(image, config='', lang=None, output_type=Output.DICT, **kwargs):
    if output_type != Output.DICT:
        raise ValueError('v46 OCR bridge supports dictionary word data only')
    return _recognize(image, 'words', config=config, lang=lang, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='View and answer local AI OCR requests.')
    parser.add_argument('--directory', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('pending')
    answer_parser = sub.add_parser('answer')
    answer_parser.add_argument('--request-id', required=True)
    answer_parser.add_argument('--answer-file', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'pending':
        print(json.dumps(pending_requests(args.directory), ensure_ascii=False, indent=2))
    else:
        try:
            submit_answer(args.directory, args.request_id, json.loads(args.answer_file.read_text(encoding='utf-8-sig')))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print('OCR answer submitted; the waiting worker can continue.')


if __name__ == '__main__':
    main()

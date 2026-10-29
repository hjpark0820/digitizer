"""Read supported unit labels from axis titles without changing numeric scales.

Unit recognition is deliberately separate from tick fitting. No dimensional
conversion, missing-unit inference, or molecular-weight lookup is performed.
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np

try:
    from . import ocr_bridge_v46 as ocr
except ImportError:
    import ocr_bridge_v46 as ocr


_MASS = r'(?:pg|ng|[uµμ]g|mg|kg|g)'
_VOLUME = r'(?:[uµμ]L|[uµμ]l|mL|ml|dL|dl|L|l)'
_DENOM = rf'(?:{_VOLUME}|kg)'
_COMPOUND = re.compile(
    rf'(?<![A-Za-z])(?P<num>{_MASS}|IU|U|[uµμ]?mol|nmol|mmol|pmol)'
    rf'\s*(?:/\s*(?P<slash>{_DENOM})|[·.\s]+(?P<power>{_DENOM})\s*(?:\^?\s*[-−]\s*1|⁻¹))'
    r'(?![A-Za-z0-9/])')
_MOLAR = re.compile(r'(?<![A-Za-z])(?:fM|pM|nM|[uµμ]M|mM|M)(?![A-Za-z])')
_TIME = re.compile(r'\b(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|years?)\b', re.I)
_SHORT_TIME = re.compile(r'(?<![A-Za-z])(?:ms|s|min|h|d|wk|yr)(?![A-Za-z])')
_UNIT_ONLY = re.compile(rf'(?<![A-Za-z]){_MASS}(?![A-Za-z/])')
_FACTOR = (rf'(?:{_MASS}|{_VOLUME}|[fpnumµμ]?M|mol|IU|U|ms|s|min|h|d|wk|yr|'
           r'hours?|hrs?|days?|minutes?|seconds?|weeks?)')


def _canonical(token):
    token = token.replace('μ', 'µ').replace('u', 'µ')
    return {'ml': 'mL', 'dl': 'dL', 'l': 'L', 'µl': 'µL'}.get(token, token)


def parse_unit_text(text):
    """Return explicit supported units and their source spans, never scale factors.

    Case distinguishes mM from M. Bare ambiguous letters require title context,
    brackets, or a title consisting of that unit alone. Unsupported composite
    units are not silently reduced to a recognized substring.
    """
    text = ' '.join(str(text).replace('μ', 'µ').split())
    if not text or len(text) > 180 or len(text.split()) > 22:
        return []
    found, occupied = [], []

    def add(match, symbol):
        start, end = match.span()
        if any(start < b and end > a for a, b in occupied):
            return
        # Do not read only the prefix of mg/kg/day or a component of AUC units.
        before, after = text[:start].rstrip(), text[end:].lstrip()
        if before.endswith(('/', '·', '*')) or after.startswith(('/', '·', '*', '^', '⁻')):
            return
        if re.match(r'[-−]\s*\d', after):
            return
        if re.match(_FACTOR + r'(?![A-Za-z])', after):
            return
        if re.search(r'(?<![A-Za-z])' + _FACTOR + r'(?:\s*\^?[-−]\s*\d|⁻¹)?$', before):
            return
        occupied.append((start, end))
        found.append(dict(symbol=symbol, raw=text[start:end], span=[start, end]))

    for match in _COMPOUND.finditer(text):
        denominator = match.group('slash') or match.group('power')
        add(match, _canonical(match.group('num')) + '/' + _canonical(denominator))
    for match in _MOLAR.finditer(text):
        if match.group() != 'M' or (text.strip(' ()[]') == 'M' or
                text[:match.start()].rstrip().endswith(('(', '[')) or
                re.search(r'\b(concentration|conc\.?|dose)\b',text,re.I)):
            add(match, _canonical(match.group()))
    for match in re.finditer('%', text):
        add(match, '%')
    for match in _TIME.finditer(text):
        word = match.group().lower()
        unit = next(symbol for prefixes, symbol in (
            (('sec',), 's'), (('min',), 'min'), (('hour', 'hr'), 'h'),
            (('day',), 'd'), (('week', 'wk'), 'wk'), (('month',), 'month'),
            (('year',), 'yr')) if word.startswith(prefixes))
        add(match, unit)
    for pattern in (_SHORT_TIME, _UNIT_ONLY):
        for match in pattern.finditer(text):
            start, end = match.span()
            bracketed = text[:start].rstrip().endswith(('(', '[')) and text[end:].lstrip().startswith((')', ']'))
            alone = text.strip(' ()[]') == match.group()
            context = bool(re.search(r'\b(time|duration|dose|concentration|conc\.?|amount)\b', text, re.I))
            if bracketed or alone or context:
                add(match, _canonical(match.group()))
    # Preserve an explicitly printed log-transformed quantity, not merely a
    # logarithmic spacing. The values themselves are never exponentiated here.
    transform = re.search(r'(?<![\w-])(log\s*10|log\s*2|ln|log)\b'
                          r'(?!\s*(?:scale|axis|spacing)\b)', text, re.I)
    if transform:
        kind = transform.group(1).lower().replace(' ', '')
        for item in found:
            item['symbol'] = f"{kind}({item['symbol']})"
    return found


def _group_words(words):
    """Spatial rows prevent unrelated OCR words becoming one invented title."""
    rows = []
    for word in sorted(words, key=lambda w: ((w['box'][1]+w['box'][3])/2, w['box'][0])):
        x0, y0, x1, y1 = word['box']
        cy, height = (y0+y1)/2, y1-y0
        row = next((r for r in rows if abs(cy-r['cy']) <= max(3, min(height, r['height'])*.55)), None)
        if row is None:
            rows.append(dict(cy=cy, height=height, words=[word]))
        else:
            row['words'].append(word)
    result = []
    for row in rows:
        ordered = sorted(row['words'], key=lambda w: w['box'][0])
        # Large gaps usually separate tick labels or neighboring panels.
        groups = [[]]
        for word in ordered:
            if groups[-1] and word['box'][0]-groups[-1][-1]['box'][2] > max(18, row['height']*2.5):
                groups.append([])
            groups[-1].append(word)
        for group in groups:
            tokens, offset = [], 0
            for word in group:
                tokens.append(dict(word, span=[offset, offset+len(word['text'])]))
                offset += len(word['text'])+1
            result.append(dict(text=' '.join(w['text'] for w in group),
                               confidence=float(np.mean([w['confidence'] for w in group])),
                               tokens=tokens,
                               box=[min(w['box'][0] for w in group), min(w['box'][1] for w in group),
                                    max(w['box'][2] for w in group), max(w['box'][3] for w in group)]))
    return result


def _source_box(box, crop, rotation):
    """Map continuous OCR box edges back from a rotated crop to source pixels."""
    l, t, r, b = crop
    width, height = r-l, b-t
    points = []
    for x, y in ((box[0], box[1]), (box[2], box[3])):
        if rotation == 90:  # crop was rotated clockwise
            x, y = y, height-x
        elif rotation == 270:
            x, y = width-y, x
        points.append((l+x, t+y))
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def _read_lines(image, crop, rotation):
    l, t, r, b = crop
    raster = image[t:b, l:r]
    if rotation:
        raster = cv2.rotate(raster, cv2.ROTATE_90_CLOCKWISE if rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY) if raster.ndim == 3 else raster
    scale = min(3., max(1., 1800/max(gray.shape)))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    data = ocr.image_to_data(gray, config='--psm 6', output_type=ocr.Output.DICT)
    words = []
    for i, value in enumerate(data.get('text', [])):
        try:
            confidence = float(data['conf'][i])
            x, y = float(data['left'][i])/scale, float(data['top'][i])/scale
            w, h = float(data['width'][i])/scale, float(data['height'][i])/scale
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        # Keep weak words in the title: dropping a denominator or exponent can
        # turn an unsupported compound into a different, apparently valid unit.
        if str(value).strip() and confidence >= 0 and w > 0 and h > 0:
            words.append(dict(text=str(value).strip(), confidence=confidence, box=[x,y,x+w,y+h]))
    return [dict(line, box=_source_box(line['box'], crop, rotation), rotation=rotation,
                 tokens=[dict(token, box=_source_box(token['box'], crop, rotation))
                         for token in line['tokens']]) for line in _group_words(words)]


def _select(lines, axis, plot_area, errors=()):
    left, top, right, bottom = plot_area
    candidates = []
    for line in lines:
        # A source title should lie beside its own axis, not in the adjacent
        # panel/caption. Long prose and low-confidence reads are not evidence.
        box = line['box']
        centre = (box[0]+box[2])/2 if axis == 'x' else (box[1]+box[3])/2
        lo, hi = (left, right) if axis == 'x' else (top, bottom)
        if not lo+.1*(hi-lo) <= centre <= hi-.1*(hi-lo) or line['confidence'] < 50:
            continue
        for unit in parse_unit_text(line['text']):
            tokens = [t for t in line.get('tokens', []) if
                      t['span'][0] < unit['span'][1] and t['span'][1] > unit['span'][0]]
            confidence = min((t['confidence'] for t in tokens), default=line['confidence'])
            reasons = []
            if confidence < 60:
                reasons.append('Low OCR confidence in the unit text; verify the source.')
            # Tesseract can confidently read a printed micro sign as "p".
            # Repeated OCR passes do not resolve this systematic ambiguity.
            # Preserve the candidate, never guess a replacement prefix.
            if re.search(r'(?<![A-Za-z])p(?:g|mol|M)(?![A-Za-z])', unit['symbol']):
                reasons.append('OCR pico-prefix (p) can be confused with micro (µ); verify the printed unit.')
            candidates.append(dict(unit=unit['symbol'], raw=unit['raw'], label=line['text'],
                                   confidence=confidence, review_reasons=reasons,
                                   box=box, rotation=line.get('rotation', 0)))
    symbols = sorted({c['unit'] for c in candidates})
    accepted = [c for c in candidates if not c['review_reasons']]
    status = 'recognized' if len(symbols) == 1 and accepted else 'ambiguous' if symbols else 'unavailable' if errors and not lines else 'not_detected'
    best = max(accepted or candidates, key=lambda c:(c['confidence'], len(c['label'])), default=None)
    warning = ('Conflicting axis units were read: ' + ', '.join(symbols) + '. Verify the source.' if len(symbols) > 1
               else ' '.join(best['review_reasons']) if status == 'ambiguous' and best
               else 'Axis unit OCR was unavailable.' if status == 'unavailable' else '')
    return dict(unit=symbols[0] if status == 'recognized' else None, unit_status=status,
                unit_source='ocr' if best else None,
                unit_text=best['raw'] if best else None,
                label=best['label'] if best else None, unit_warning=warning,
                candidates=candidates, lines=lines, errors=list(errors))


def read_axis_units(image, plot_area, axis):
    """Read horizontal X and horizontal/rotated Y titles from source pixels."""
    started = time.perf_counter()
    height, width = image.shape[:2]
    left, top, right, bottom = map(int, plot_area)
    # Titles can extend past the first/last tick (especially a rotated Y title).
    # Keep their ends in the OCR raster; _select still requires the title centre
    # to align with this axis, excluding neighbouring captions/panels.
    if axis == 'x':
        end_margin = max(24, int((right-left)*.12))
        crop = [max(0,left-end_margin), min(height,bottom+8), min(width,right+end_margin+1),
                min(height,bottom+max(70,int((bottom-top)*.23)))]
    elif axis == 'y':
        end_margin = max(24, int((bottom-top)*.12))
        crop = [max(0,left-max(95,int((right-left)*.23))), max(0,top-end_margin),
                max(0,left-8), min(height,bottom+end_margin+1)]
    else:
        raise ValueError('axis must be x or y')
    # Numeric tick OCR filters weak words. Do not reuse that incomplete title
    # context: a dropped unit factor could change the meaning of a compound.
    lines = []
    errors = []
    raster = image[crop[1]:crop[3],crop[0]:crop[2]]
    has_ink = raster.size and np.count_nonzero(raster < 150) >= 20
    if has_ink:
        rotations = [0] if axis == 'x' else [0,270,90]
        for rotation in rotations:
            try:
                lines.extend(_read_lines(image, crop, rotation))
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(f'{type(error).__name__}: {error}')
                break
    result = _select(lines, axis, plot_area, errors)
    result.update(crop=crop, seconds=float(time.perf_counter()-started))
    return result


def unit_metadata(report):
    return {key:report[key] for key in ('unit','unit_status','unit_source','unit_text','label','unit_warning')}

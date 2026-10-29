"""Conservative, independent numeric calibration of the two chart axes.

Only NumPy, OpenCV and the existing OCR bridge are needed; importing this module
does not import the analysis pipeline or initialize its models. Pixel fallbacks
are explicit and measure distance from the left/bottom edge of the plot.
"""
from __future__ import annotations

import itertools
import math
import re
import time

import cv2
import numpy as np

try:
    from . import ocr_bridge_v46 as ocr
    from .axis_units_v46 import read_axis_units, unit_metadata
except ImportError:
    import ocr_bridge_v46 as ocr
    from axis_units_v46 import read_axis_units, unit_metadata


_SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def parse_numeric_label(text):
    """Read a complete numeric label, never the digits embedded in prose.

    Decimal/scientific notation and explicit powers (including Unicode
    superscripts) are accepted. Ambiguous OCR repairs such as O -> 0 are not.
    """
    token = str(text).strip().replace('−', '-').replace('–', '-').replace('—', '-').replace('﹣', '-')
    token = re.sub(r'([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)',
                   lambda m: '^' + m.group(1).translate(_SUPERSCRIPTS), token)
    token = token.replace('°', '^0').replace('**', '^')
    token = re.sub(r'\s+', '', token)
    token = re.sub(r'\^\{([+-]?\d+)\}', r'^\1', token)
    if ',' in token:
        if re.fullmatch(r'[+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?', token):
            token = token.replace(',', '')
        elif token.count(',') == 1 and '.' not in token:
            token = token.replace(',', '.')
    power = re.fullmatch(r'(?:(%s)[xX×·*])?10\^([+-]?\d+)' % _NUMBER, token)
    try:
        if power:
            value = float(power.group(1) or 1) * 10.0 ** int(power.group(2))
        elif re.fullmatch(_NUMBER, token):
            value = float(token)
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _manual_value(value, name):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite number or blank.') from error
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite.')
    return number


def _validate_bounds(v0, v1, logarithmic, axis):
    if v0 is not None and v1 is not None and v0 == v1:
        raise ValueError(f'{axis.upper()} bounds must be distinct.')
    if logarithmic and any(value is not None and value <= 0 for value in (v0, v1)):
        raise ValueError(f'{axis.upper()} logarithmic bounds must be positive.')


def _tick_positions(image, plot_area, axis):
    """Locate short perpendicular tick strokes close to the plot boundary."""
    height, width = image.shape[:2]
    left, top, right, bottom = plot_area
    depth = max(15, min(25, int(min(height, width) * 0.028)))
    if axis == 'x':
        box = (max(0, left - 6), max(0, bottom - 6),
               min(width, right + 7), min(height, bottom + depth))
        kernel = np.ones((6, 1), np.uint8)
    else:
        box = (max(0, left - depth), max(0, top - 6),
               min(width, left + 7), min(height, bottom + 7))
        kernel = np.ones((1, 6), np.uint8)
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = (gray < 150).astype(np.uint8)
    mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    ticks = []
    for (x, y, w, h, _), center in zip(stats[1:], centers[1:]):
        if axis == 'x' and 6 <= h <= depth + 6 and w <= 7 and y + y0 <= bottom + 6:
            ticks.append(float(center[0] + x0))
        elif axis == 'y' and 6 <= w <= depth + 7 and h <= 7 and x + w + x0 >= left - 3:
            ticks.append(float(center[1] + y0))
    return sorted(ticks)


def _ocr_crop(image, box, psm):
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return []
    scale = 3.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # A numeric whitelist can silently turn "1e-3" into "13". Read all glyphs.
    data = ocr.image_to_data(gray, config=f'--psm {psm}', output_type=ocr.Output.DICT)
    words = []
    for index, text in enumerate(data.get('text', [])):
        if not str(text).strip():
            continue
        try:
            confidence = float(data['conf'][index])
            x = x0 + float(data['left'][index]) / scale
            y = y0 + float(data['top'][index]) / scale
            w = float(data['width'][index]) / scale
            h = float(data['height'][index]) / scale
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if confidence < 20 or w <= 0 or h <= 0:
            continue
        words.append(dict(text=str(text).strip(), confidence=confidence,
                          box=[x, y, x + w, y + h], pass_id=psm))
    return words


def _join_superscripts(words):
    """Use word geometry to preserve exponents split off by the OCR engine."""
    result = []
    consumed = set()
    for index, base in enumerate(words):
        if index in consumed:
            continue
        x0, y0, x1, y1 = base['box']
        joined = None
        if re.fullmatch(r'(?:[\d.]+[x×*])?10', base['text']):
            for exponent_index, exponent in enumerate(words):
                if exponent_index == index or exponent_index in consumed:
                    continue
                ex0, ey0, ex1, ey1 = exponent['box']
                etext = exponent['text'].replace('−', '-')
                if (re.fullmatch(r'[+-]?\d+', etext)
                        and -2 <= ex0 - x1 <= max(8, (y1 - y0) * 0.65)
                        and ey1 < y1 - (y1 - y0) * 0.15
                        and ey0 < y0 + (y1 - y0) * 0.3
                        and exponent.get('pass_id') == base.get('pass_id')):
                    joined = dict(base, text=base['text'] + '^' + etext,
                                  box=[x0, min(y0, ey0), ex1, y1],
                                  confidence=min(base['confidence'], exponent['confidence']))
                    consumed.add(exponent_index)
                    break
        result.append(joined or base)
    # A superscript can precede its base in OCR reading order.
    return [word for word in result if not any(word is words[i] for i in consumed)]


def _repair_power_words(image, words):
    """Recover a caret lost by OCR only when raised glyphs prove an exponent.

    Tesseract often returns 10-3 or 102 for printed 10^-3 or 10^2. Ordinary
    decimal integers such as 102 remain integers when all glyphs share a
    baseline. No missing/unreadable exponent digits are invented.
    """
    result = []
    height, width = image.shape[:2]
    for word in words:
        match = re.fullmatch(r'10([+-]?\d+)', word['text'])
        if not match:
            result.append(word)
            continue
        bx0, by0, bx1, by1 = word['box']
        x0, y0 = max(0, int(math.floor(bx0))), max(0, int(math.floor(by0)))
        x1, y1 = min(width, int(math.ceil(bx1))), min(height, int(math.ceil(by1)))
        crop = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, _, stats, _ = cv2.connectedComponentsWithStats((gray < 170).astype(np.uint8))
        components = sorted((s for s in stats[1:] if s[4] >= 2), key=lambda s: s[0])
        if len(components) >= 3:
            bases, raised = components[:2], components[2:]
            baseline = min(int(s[1] + s[3]) for s in bases)
            base_height = max(int(s[3]) for s in bases)
            if (min(int(s[3]) for s in bases) >= base_height * 0.7
                    and all(s[1] + s[3] <= baseline - max(2, base_height * 0.18) for s in raised)):
                word = dict(word, text='10^' + match.group(1), original_text=word['text'])
        result.append(word)
    return result


def _label_candidates(words, axis, ticks, plot_area):
    left, top, right, bottom = plot_area
    lo, hi = (left, right) if axis == 'x' else (top, bottom)
    candidates = []
    for word in _join_superscripts(words):
        value = parse_numeric_label(word['text'])
        if value is None:
            continue
        x0, y0, x1, y1 = word['box']
        position = (x0 + x1) / 2 if axis == 'x' else (y0 + y1) / 2
        if not lo - 8 <= position <= hi + 8:
            continue
        label_size = x1 - x0 if axis == 'x' else y1 - y0
        anchor = min(ticks, key=lambda p: abs(p - position)) if ticks else None
        snapped = anchor is not None and abs(anchor - position) <= max(7, min(18, label_size * 0.6))
        if snapped:
            position = anchor
        # Repeated OCR passes provide alternatives, not independent evidence.
        duplicate = next((c for c in candidates
                          if abs(c['pixel'] - position) < 3 and c['value'] == value), None)
        item = dict(pixel=float(position), value=float(value), text=word['text'],
                    confidence=word['confidence'], box=word['box'], anchored=bool(snapped))
        if duplicate is None:
            candidates.append(item)
        elif item['confidence'] > duplicate['confidence']:
            duplicate.update(item)
    return sorted(candidates, key=lambda item: (item['pixel'], item['value']))


def read_axis_candidates(image, plot_area, axis):
    """Read bottom/left numeric labels and return JSON-serializable evidence."""
    height, width = image.shape[:2]
    left, top, right, bottom = plot_area
    if axis == 'x':
        margin = max(40, int((right - left) * 0.09))
        box = (max(0, left - margin), min(height, bottom + 7),
               min(width, right + margin), min(height, bottom + max(55, int(height * 0.14))))
    elif axis == 'y':
        box = (max(0, left - max(65, int(width * 0.16))), max(0, top - 18),
               max(0, left - 8), min(height, bottom + 12))
    else:
        raise ValueError('axis must be x or y')
    ticks = _tick_positions(image, plot_area, axis)
    words, errors = [], []
    for psm in (6, 11):
        try:
            words.extend(_ocr_crop(image, box, psm))
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
            break
    candidates = _label_candidates(_repair_power_words(image, words), axis, ticks, plot_area)
    return dict(candidates=candidates, raw_boxes=words, ticks=ticks,
                crop=list(box), errors=errors)


def _unique_inliers(candidates, slope, intercept, kind, tolerance):
    selected = []
    for item in candidates:
        value = item['value']
        if kind == 'log' and value <= 0:
            continue
        target = math.log10(value) if kind == 'log' else value
        residual = abs((target - intercept) / slope - item['pixel'])
        if residual > tolerance:
            continue
        record = dict(item, residual_px=float(residual))
        previous = next((old for old in selected if abs(old['pixel'] - item['pixel']) < 3), None)
        if previous is None:
            selected.append(record)
        elif (residual, -item.get('confidence', 0)) < (previous['residual_px'], -previous.get('confidence', 0)):
            previous.update(record)
    return selected


def fit_axis_candidates(candidates, p0, p1, force_log=False):
    """Fit linear/log scales using at least three distinct, corroborating ticks.

    Residuals are measured in pixels, so tiny/huge data units receive the same
    tolerance. Conflicting models with comparable support are rejected.
    """
    span = abs(p1 - p0)
    tolerance = max(2.5, span * 0.006)
    candidates = [c for c in candidates if math.isfinite(c['pixel']) and math.isfinite(c['value'])]
    if len(candidates) < 3:
        return None, 'At least three readable numeric tick labels are required.'
    models = []
    for kind in (('log',) if force_log else ('linear', 'log')):
        eligible = [c for c in candidates if kind != 'log' or c['value'] > 0]
        for first, second in itertools.combinations(eligible, 2):
            delta = second['pixel'] - first['pixel']
            if abs(delta) < max(8, span * 0.08) or first['value'] == second['value']:
                continue
            values = [math.log10(c['value']) if kind == 'log' else c['value'] for c in (first, second)]
            slope = (values[1] - values[0]) / delta
            intercept = values[0] - slope * first['pixel']
            if not math.isfinite(slope) or abs(slope) < 1e-300:
                continue
            inliers = _unique_inliers(eligible, slope, intercept, kind, tolerance)
            if len(inliers) < 3:
                continue
            for _ in range(2):
                positions = np.array([c['pixel'] for c in inliers], dtype=float)
                targets = np.array([math.log10(c['value']) if kind == 'log' else c['value']
                                    for c in inliers], dtype=float)
                slope, intercept = np.polyfit(positions, targets, 1)
                if abs(slope) < 1e-300:
                    break
                inliers = _unique_inliers(eligible, slope, intercept, kind, tolerance)
                if len(inliers) < 3:
                    break
            if len(inliers) < 3 or abs(slope) < 1e-300:
                continue
            coverage = (max(c['pixel'] for c in inliers) - min(c['pixel'] for c in inliers)) / span
            if coverage < 0.3 or len({c['value'] for c in inliers}) < 3:
                continue
            rms = float(np.sqrt(np.mean([c['residual_px'] ** 2 for c in inliers])))
            score = (len(inliers), coverage, -rms)
            model = dict(kind=kind, slope=float(slope), intercept=float(intercept),
                         inliers=inliers, coverage=float(coverage), rms_px=rms, score=score)
            if not any(m['kind'] == kind and abs(m['slope'] - slope) <= abs(slope) * 1e-6
                       and abs(m['intercept'] - intercept) <= max(1e-12, abs(slope) * 0.01)
                       for m in models):
                models.append(model)
    if not models:
        return None, 'Numeric labels did not support a consistent linear or logarithmic scale.'
    models.sort(key=lambda m: m['score'], reverse=True)
    best = models[0]
    # A plausible subset is not enough when other confidently read tick
    # positions contradict it. Discarded OCR alternatives at an accepted tick
    # do not count as additional votes against the fit.
    contradicted = []
    for candidate in candidates:
        if (candidate.get('confidence', 0) < 70 or not candidate.get('anchored')
                or any(abs(candidate['pixel'] - c['pixel']) < 3 for c in best['inliers'])):
            continue
        if not any(abs(candidate['pixel'] - c['pixel']) < 3 for c in contradicted):
            contradicted.append(candidate)
    if len(best['inliers']) < 2 * len(contradicted):
        return None, 'Too many confidently read ticks contradict the proposed numeric scale.'
    if best['kind'] == 'log' and any(c['value'] <= 0 for c in contradicted):
        return None, 'A nonpositive tick contradicts a continuous logarithmic axis; the labels may be categories.'
    # Two rows of numeric X labels can be separate treatments/doses. Selecting
    # only one row would manufacture a single continuous coordinate system.
    boxes = [c['box'] for c in best['inliers'] if c.get('box') is not None]
    if p1 > p0 and len(boxes) >= 3:
        row_center = float(np.median([(b[1] + b[3]) / 2 for b in boxes]))
        row_height = float(np.median([b[3] - b[1] for b in boxes]))
        other_row = []
        for candidate in candidates:
            box = candidate.get('box')
            if (box is not None and candidate.get('anchored')
                    and candidate.get('confidence', 0) >= 60
                    and abs((box[1] + box[3]) / 2 - row_center) > max(6, row_height * 0.8)
                    and not any(abs(candidate['pixel'] - c['pixel']) < 3 for c in other_row)):
                other_row.append(candidate)
        if len(other_row) >= 3:
            return None, 'Multiple rows of numeric tick labels make the axis interpretation ambiguous.'
    for other in models[1:]:
        if len(other['inliers']) < len(best['inliers']):
            continue
        if other['coverage'] < best['coverage'] * 0.9:
            continue
        if other['rms_px'] > max(0.6, best['rms_px'] * 2.5):
            continue
        # Different sets of equally plausible values, or indistinguishable
        # linear/log explanations, are insufficient evidence for calibration.
        same_labels = {(round(c['pixel'], 1), c['value']) for c in best['inliers']} == {
            (round(c['pixel'], 1), c['value']) for c in other['inliers']}
        if other['kind'] != best['kind'] or not same_labels:
            return None, 'Ambiguous numeric tick labels or scale type; enter the axis bounds manually.'
    try:
        endpoints = [best['slope'] * p + best['intercept'] for p in (p0, p1)]
        if best['kind'] == 'log':
            endpoints = [10.0 ** value for value in endpoints]
    except OverflowError:
        return None, 'The fitted scale overflows at the plot boundary.'
    if not all(math.isfinite(v) for v in endpoints) or endpoints[0] == endpoints[1]:
        return None, 'The fitted scale is not finite and distinct at the plot boundary.'
    if endpoints[1] < endpoints[0]:
        return None, 'A decreasing automatic axis requires confirmation through complete manual bounds.'
    best.update(v0=float(endpoints[0]), v1=float(endpoints[1]))
    best.pop('score')
    return best, ''


def resolve_calibration(image, plot_area, x_min=None, x_max=None, y_min=None,
                        y_max=None, x_log=False, y_log=False, log_fn=print,
                        _fast_context=True):
    """Resolve each axis independently; blank bounds request automatic OCR.

    Complete manual bounds skip numeric-tick OCR and default to linear unless
    ``*_log`` is true; axis-title unit OCR remains independent. A partial manual
    axis keeps the explicit bound and reads the missing
    bound/scale from ticks. Invalid manual values raise ``ValueError``. Failed
    automatic axes return an explicit pixel mapping instead of a guessed 0–1
    data range. ``p0/v0`` is left or bottom; ``p1/v1`` is right or top.
    """
    started = time.perf_counter()
    if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:
        raise ValueError('A chart image is required for axis calibration.')
    if plot_area is None or len(plot_area) != 4:
        raise ValueError('plot_area must contain left, top, right, bottom.')
    if not all(math.isfinite(float(v)) for v in plot_area):
        raise ValueError('plot_area coordinates must be finite.')
    left, top, right, bottom = [int(round(float(v))) for v in plot_area]
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError('plot_area must be a non-empty rectangle inside the image.')
    plot_area = (left, top, right, bottom)
    if _fast_context:
        try:
            from .fast_detection_v46 import calibration_source, working_calibration
        except ImportError:
            from fast_detection_v46 import calibration_source, working_calibration
        source = calibration_source(image, plot_area)
        if source is not None:
            original, source_box, info = source
            result = resolve_calibration(original, source_box, x_min, x_max, y_min,
                                         y_max, x_log, y_log, log_fn, _fast_context=False)
            result = working_calibration(result, info)
            result['seconds'] = float(time.perf_counter() - started)
            return result
    bounds = {'x': (_manual_value(x_min, 'x_min'), _manual_value(x_max, 'x_max')),
              'y': (_manual_value(y_min, 'y_min'), _manual_value(y_max, 'y_max'))}
    for axis, forced in (('x', x_log), ('y', y_log)):
        _validate_bounds(*bounds[axis], bool(forced), axis)
    calibration, diagnostics = {}, {}
    for axis, p0, p1, forced in (('x', left, right, bool(x_log)), ('y', bottom, top, bool(y_log))):
        low, high = bounds[axis]
        evidence = dict(candidates=[], raw_boxes=[], ticks=[], inliers=[], errors=[])
        warning = ''
        if low is not None and high is not None:
            kind, source = ('log' if forced else 'linear'), 'manual'
            reason = 'Both bounds supplied manually; numeric-tick OCR skipped.'
        else:
            evidence.update(read_axis_candidates(image, plot_area, axis))
            fitted, reason = fit_axis_candidates(evidence['candidates'], p0, p1, force_log=forced)
            if fitted is not None:
                proposed_low = fitted['v0'] if low is None else low
                proposed_high = fitted['v1'] if high is None else high
                try:
                    _validate_bounds(proposed_low, proposed_high, fitted['kind'] == 'log', axis)
                except ValueError as error:
                    fitted, reason = None, str(error)
            if fitted is not None:
                low, high = proposed_low, proposed_high
                kind = fitted['kind']
                source = 'automatic' if all(v is None for v in bounds[axis]) else 'mixed'
                evidence.update(inliers=fitted['inliers'], fit={k: v for k, v in fitted.items() if k != 'inliers'})
                reason = f"Accepted {len(fitted['inliers'])} numeric ticks as a {kind} scale."
            else:
                low, high, kind, source = 0.0, float(abs(p1 - p0)), 'pixel', 'pixel'
                details = (' ' + '; '.join(evidence['errors'])) if evidence['errors'] else ''
                warning = f'{axis.upper()} axis requires calibration: {reason}{details} Exporting local pixel coordinates.'
        status = 'needs_calibration' if kind == 'pixel' else 'accepted'
        calibration[axis] = dict(p0=float(p0), p1=float(p1), v0=float(low), v1=float(high),
                                 log=kind == 'log', kind=kind, status=status, source=source, warning=warning)
        # Unit text is independent evidence; never changes tick fits or values.
        # Manual numeric bounds still allow title reading. Fast detection calls
        # this on the original raster before remapping only the pixel anchors.
        units = read_axis_units(image, plot_area, axis)
        calibration[axis].update(unit_metadata(units))
        evidence['unit_recognition'] = units
        diagnostics[axis] = dict(evidence, status=status, source=source, kind=kind,
                                 reason=reason, warning=warning,
                                 manual_bounds=list(bounds[axis]))
        if log_fn:
            log_fn(warning or f'Axis calibration {axis.upper()}: {source} {kind}, {low:g} to {high:g}.')
    diagnostics['axes'] = {axis: diagnostics[axis] for axis in ('x', 'y')}
    return dict(calibration=calibration, diagnostics=diagnostics,
                seconds=float(time.perf_counter() - started))

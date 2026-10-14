"""Small, deterministic colour/B&W preflight for the GUI's Automatic option.

Only source pixels are inspected; no detector, OCR, or learned model is loaded.
Boxes use inclusive (left, top, right, bottom) image coordinates.  A measured
legend marker takes precedence over connector colour, because the two runtime
branches primarily differ in how they identify markers.  If legend geometry is
ambiguous, coherent colour in the plot is the fallback.  This is a heuristic,
not a confidence probability; the GUI's manual overrides remain authoritative.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


VERSION = "automatic_mode_v46_v1"
_MAX_SIDE = 1600
_MIN_COLOUR_PIXELS = 6


def _box(value, shape):
    height, width = shape[:2]
    if value is None:
        return None
    coords = np.asarray(value, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("Mode-selection boxes must contain four finite coordinates")
    left, top, right, bottom = coords.tolist()
    if right < left or bottom < top:
        raise ValueError("Mode-selection boxes must be ordered left, top, right, bottom")
    if right < 0 or bottom < 0 or left >= width or top >= height:
        raise ValueError("Mode-selection box does not overlap the image")
    return (max(0, math.floor(left)), max(0, math.floor(top)),
            min(width - 1, math.ceil(right)), min(height - 1, math.ceil(bottom)))


def _paper(image):
    # Sampling only estimates a page cast; individual marker pixels are never
    # subsampled by this operation. A warm scanned page must not become colour.
    stride = max(1, math.ceil(math.sqrt(image.shape[0] * image.shape[1] / 100_000)))
    sample = image[::stride, ::stride].reshape(-1, 3).astype(np.float32)
    brightness = sample.mean(axis=1)
    bright = brightness >= max(180.0, float(np.percentile(brightness, 80)))
    return np.median(sample[bright], axis=0) if bright.any() else np.full(3, 255.0)


def _suppress_colour_fringes(pixels, colour, ink):
    """Reject opposing subpixel edge colours surrounding neutral dark ink.

    Text rendered for an LCD screen can have strong blue/orange edge pixels
    even in a monochrome figure. Magnitude thresholds alone cannot reject that
    artifact. Real colour must survive locally without an opposite fringe and
    a dark neutral stroke together. No smoothing erases pale marker colours.
    """
    neutral = ink & (np.ptp(pixels, axis=2) < 12) & (pixels.mean(axis=2) < 160)
    if int(colour.sum()) < _MIN_COLOUR_PIXELS or not neutral.any():
        return colour
    kernel = np.ones((15, 15), np.uint8)
    near_neutral = cv2.dilate(neutral.astype(np.uint8), kernel).astype(bool)
    candidates = colour & near_neutral
    if not candidates.any():
        return colour
    # Angle in the RGB chromatic plane; opposite angles are complementary
    # channel shifts, including both blue/orange and red/cyan fringes.
    horizontal = 2 * pixels[..., 2] - pixels[..., 1] - pixels[..., 0]
    vertical = np.sqrt(3) * (pixels[..., 1] - pixels[..., 0])
    bins = (np.floor((np.arctan2(vertical, horizontal) + np.pi) * 8 / np.pi).astype(np.int8) % 16)
    fringe = np.zeros(colour.shape, bool)
    for hue in np.unique(bins[candidates]):
        distance = (bins - (int(hue) + 8)) % 16
        opposite = colour & ((distance <= 2) | (distance >= 14))
        if opposite.any():
            nearby = cv2.dilate(opposite.astype(np.uint8), kernel).astype(bool)
            fringe |= candidates & (bins == hue) & nearby
    return colour & ~fringe


def _region(image, box, paper, excluded=None):
    left, top, right, bottom = box
    crop = image[top:bottom + 1, left:right + 1]
    scale = min(1.0, _MAX_SIDE / max(crop.shape[:2]))
    allowed = np.ones(crop.shape[:2], np.uint8)
    if excluded is not None:
        a, b, c, d = excluded
        a, b, c, d = max(left, a), max(top, b), min(right, c), min(bottom, d)
        if a <= c and b <= d:
            allowed[b - top:d - top + 1, a - left:c - left + 1] = 0
    if scale < 1:
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        allowed = cv2.resize(allowed, size, interpolation=cv2.INTER_NEAREST)
    pixels = crop.astype(np.float32)
    mean = pixels.mean(axis=2)
    # Compensate a light paper tint in proportion to brightness: dark neutral
    # markers should not acquire an artificial opposite tint after subtraction.
    corrected = np.clip(pixels - (paper - paper.mean()) *
                        np.clip(mean / max(1.0, float(paper.mean())), 0, 1)[..., None], 0, 255)
    high, low = corrected.max(axis=2), corrected.min(axis=2)
    chroma = high - low
    saturation = chroma / np.maximum(high, 1)
    ink = (np.max(paper - pixels, axis=2) >= 14) & allowed.astype(bool)
    colour = ink & (chroma >= 18) & ((saturation >= .16) | (chroma >= 28))
    raw_chromatic_pixels = int(colour.sum())
    colour = _suppress_colour_fringes(corrected, colour, ink)
    return ink, colour, dict(box=list(box), sampled_size=[int(crop.shape[1]), int(crop.shape[0])],
                             scale=round(scale, 6), considered_pixels=int(allowed.sum()),
                             raw_chromatic_pixels=raw_chromatic_pixels,
                             opposing_fringe_pixels_rejected=raw_chromatic_pixels - int(colour.sum()))


def _colour_evidence(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:count, cv2.CC_STAT_AREA]
    coherent = areas[areas >= _MIN_COLOUR_PIXELS]
    return dict(chromatic_pixels=int(mask.sum()), coherent_chromatic_pixels=int(coherent.sum()),
                coherent_components=int(coherent.size),
                largest_chromatic_component=int(areas.max()) if areas.size else 0)


def _legend_evidence(ink, colour):
    """Find isolated graphical keys and ignore their thin horizontal tails.

    Text is rejected using neighbouring letters and the label gap. This small
    preflight deliberately does not invoke production legend-template fitting.
    A line with a wider marker body is measured outside the connector row band,
    so a black marker on a red connector remains an achromatic marker.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    components = []
    height, width = ink.shape
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 6 or w < 3 or h < 2:
            continue
        if (w >= .85 * width and h >= .85 * height and
                area / max(1, w * h) < .3):
            continue  # enclosing legend frame
        components.append(dict(index=index, x=x, y=y, w=w, h=h))

    bodies = []
    lines = []
    for comp in components:
        x, y, w, h = (comp[k] for k in ("x", "y", "w", "h"))
        peers = [q for q in components if q is not comp and
                 min(y + h, q["y"] + q["h"]) - max(y, q["y"]) >= .3 * min(h, q["h"])]
        previous = max((q["x"] + q["w"] for q in peers if q["x"] + q["w"] <= x), default=None)
        following = min((q["x"] for q in peers if q["x"] >= x + w), default=None)
        previous_gap = x - previous if previous is not None else None
        next_gap = following - x - w if following is not None else None
        row_start = previous_gap is None or previous_gap >= max(6, 1.1 * h)
        isolated = next_gap is None or next_gap >= max(3, .35 * h)
        if not (row_start and isolated):
            continue
        support = labels[y:y + h, x:x + w] == comp["index"]
        body = support.copy()
        attached_line = False
        if w > 1.85 * h:
            # A connector must occupy most of its entry's horizontal extent.
            # Removing only those rows leaves open as well as filled markers.
            line_rows = support.sum(axis=1) >= .60 * w
            if not line_rows.any():
                continue
            body[line_rows, :] = False
            ys, xs = np.nonzero(body)
            if xs.size < 8:
                lines.append((comp, support))
                continue
            body_width, body_height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if not (.35 * body_height <= body_width <= 1.85 * body_height):
                continue
            if body_height < max(4, 2 * int(line_rows.sum())):
                continue
            attached_line = True
        elif h < 4 or not (.35 * h <= w <= 1.85 * h):
            continue
        # Off-line ink can split an open circle into two arcs. Count coherent
        # pixels across the whole body, while retaining the speck rejection.
        local_colour = colour[y:y + h, x:x + w] & body
        colour_report = _colour_evidence(local_colour)
        chromatic = colour_report["coherent_chromatic_pixels"] >= _MIN_COLOUR_PIXELS
        bodies.append(dict(box=[x, y, x + w - 1, y + h - 1], ink_pixels=int(body.sum()),
                           chromatic=bool(chromatic), attached_connector=attached_line,
                           **colour_report))

    # A separate coloured line key can encode a line-only series. Detached
    # tails beside an already measured marker belong to that marker instead.
    colour_lines = 0
    for comp, support in lines:
        x, y, w, h = (comp[k] for k in ("x", "y", "w", "h"))
        nearby_body = any(abs((b["box"][1] + b["box"][3]) / 2 - (y + h / 2)) <= max(3, h)
                          and max(b["box"][0] - (x + w), x - b["box"][2], 0) <=
                          2 * (b["box"][3] - b["box"][1] + 1) for b in bodies)
        if not nearby_body and _colour_evidence(colour[y:y + h, x:x + w] & support)["coherent_components"]:
            colour_lines += 1
    return dict(body_box_coordinate_system="sampled_legend_relative_inclusive_xyxy",
                marker_bodies=len(bodies), chromatic_marker_bodies=sum(b["chromatic"] for b in bodies),
                achromatic_marker_bodies=sum(not b["chromatic"] for b in bodies),
                chromatic_line_keys=int(colour_lines), bodies=bodies)


def select_detection_mode(image_bgr, plot_box=None, legend_box=None):
    """Return a JSON-safe ``mode``, plain-English ``reason``, and ``evidence``.

    ``plot_box`` and ``legend_box`` are optional inclusive xyxy boxes. Supplied
    regions are clipped to the image; reversed, empty/outside, and non-finite
    boxes are rejected. The legend is excluded from the plot fallback. Without
    a plot selection the image is the fallback region, so external annotation
    colour can influence the choice; a manual override resolves ambiguous art.

    Reliable marker colour wins first. Neutral legend marker bodies then favour
    B&W even when their connectors are colourful. A distinct line-only colour
    key or coherent plot colour without usable legend bodies favours colour.
    Blank images and weak/isolated colour noise conservatively choose B&W.
    """
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Mode selection requires a non-empty uint8 BGR image")
    plot = _box(plot_box, image.shape)
    legend = _box(legend_box, image.shape)
    fallback = plot or (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    paper = _paper(image)
    plot_ink, plot_colour, plot_report = _region(image, fallback, paper, excluded=legend)
    plot_report.update(_colour_evidence(plot_colour), ink_pixels=int(plot_ink.sum()))
    legend_report = None
    if legend is not None:
        ink, colour, legend_report = _region(image, legend, paper)
        legend_report.update(_colour_evidence(colour), **_legend_evidence(ink, colour), ink_pixels=int(ink.sum()))
    evidence = dict(version=VERSION, fallback_region="selected_plot" if plot else "whole_image",
                    paper_bgr=[round(float(v), 2) for v in paper], plot=plot_report, legend=legend_report)

    if legend_report and legend_report["chromatic_marker_bodies"]:
        mode, basis = "color", "legend_marker_colour"
        reason = "Colour was selected because the legend contains coloured marker ink."
    elif legend_report and legend_report["chromatic_line_keys"]:
        mode, basis = "color", "legend_line_colour"
        reason = "Colour was selected because the legend contains a coloured line-only series key."
    elif legend_report and legend_report["achromatic_marker_bodies"]:
        mode, basis = "bw", "legend_markers_achromatic"
        reason = "B&W was selected because the measured legend marker bodies are neutral; connector colour does not determine marker mode."
    elif plot_report["coherent_components"]:
        mode, basis = "color", "plot_colour"
        region_name = "selected plot" if plot else "image"
        reason = f"Colour was selected because the {region_name} contains coherent coloured ink and no usable neutral legend markers were identified."
    else:
        mode, basis = "bw", "no_coherent_colour"
        reason = "B&W was selected because no coherent coloured series ink was found in the inspected regions."
    evidence["decision_basis"] = basis
    return dict(mode=mode, reason=reason, evidence=evidence)

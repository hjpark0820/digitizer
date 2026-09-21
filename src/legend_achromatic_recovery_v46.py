"""Recover neutral keys only in independently observed swatch columns."""
import cv2
import numpy as np


def recover(image_rgb, legend_box, cells, existing=()):
    if not cells or legend_box is None:
        return []
    height, width = image_rgb.shape[:2]
    left, top, right, bottom = map(int, legend_box)  # native inclusive ROI
    columns = []
    for x in sorted(float(c[0]) for c in cells):
        if columns and abs(x - np.median(columns[-1])) <= 8:
            columns[-1].append(x)
        else:
            columns.append([x])
    added = []
    for column in columns:
        cx = int(round(np.median(column)))
        # Known rows are local to this column: a colour in a different column
        # must not prevent a real black/grey key at the same row being recovered.
        known = [float(c[1]) for c in [*cells, *existing, *added] if abs(c[0]-cx) <= 16]
        x0, x1 = max(0, left, cx-16), min(width, right+1, cx+17)
        y0, y1 = max(0, top), min(height, bottom+1)
        patch = image_rgb[y0:y1, x0:x1]
        if not patch.size:
            continue
        dark = (cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) < 110).astype(np.uint8)
        n, labels, stats, centres = cv2.connectedComponentsWithStats(dark, 8)
        for k in range(1, n):
            _, _, w, h, area = stats[k]
            x, y = centres[k] + [x0, y0]
            if area < 20 or w > 34 or h > 22 or abs(x-cx) > 8:
                continue
            if y < top+6 or y > bottom-6 or any(abs(y-row) <= 8 for row in known):
                continue
            pixels = patch[labels == k]
            intensity = pixels.astype(float).sum(axis=1)
            core = pixels[intensity <= np.percentile(intensity, 30)]
            rgb = (core.mean(axis=0) if len(core) >= 3 else np.median(pixels, axis=0)).astype(int)
            if np.ptp(rgb) > 40:
                continue
            added.append((int(x), int(y), tuple(map(int, rgb))))
            known.append(y)
    return added

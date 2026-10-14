"""Shared source-evidence contract for marker versus line-only legend entries."""
import numpy as np


def independent_line_rows(line_rows, reports):
    """A strip cut from an accepted marker body is not a plain-line witness."""
    independent, conflicts = [], []
    for row in line_rows:
        line = np.asarray(row.get('observed_line_box'), float)
        overlaps = []
        if line.shape == (4,) and np.isfinite(line).all():
            area = max(0., line[2]-line[0])*max(0., line[3]-line[1])
            for report in reports:
                body = np.asarray(report.get('marker_body_box'), float)
                evidence = report.get('shape_evidence') or {}
                quality = report.get('extraction_quality') or {}
                has_body = bool(quality.get('line_bulge')) or int(evidence.get('off_line_pixels') or 0) >= 4
                if not has_body or body.shape != (4,) or not np.isfinite(body).all():
                    continue
                if min(body[2]-body[0], body[3]-body[1]) < 3:
                    continue
                intersection = max(0., min(line[2],body[2])-max(line[0],body[0]))*max(
                    0., min(line[3],body[3])-max(line[1],body[1]))
                fraction = intersection / area if area > 0 else 0.
                if fraction >= .5:
                    overlaps.append(dict(swatch_id=report.get('swatch_id'),
                        marker_body_box=body.astype(int).tolist(),line_overlap_fraction=float(fraction)))
        if overlaps:
            conflicts.append(dict(native_index=row.get('native_index'),
                observed_line_box=row.get('observed_line_box'),
                reason='line_strip_overlaps_independently_observed_marker_body',markers=overlaps))
        else:
            independent.append(row)
    return independent, conflicts

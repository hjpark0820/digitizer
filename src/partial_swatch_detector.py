"""Occlusion-tolerant marker detection from legend swatch fragments.

The detector implements a grid-local generalized Hough transform:

1. A marker template is isolated from each legend line/marker swatch.
2. The plot is divided into square cells whose side starts at half the marker
   diameter.
3. Every edge fragment in a cell casts votes for the marker centre using the
   edge-to-centre vectors learned from the legend template.
4. Votes from distinct cells are clustered.  A candidate is retained only
   when the visible template contour agrees with the plot around that centre.

The deliberately asymmetric verification distinguishes missing required ink
from additional crossing-line ink. The production B&W v46 pipeline uses this
detector; the conventional full-template baseline is optional experiment output.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree


Box = tuple[int, int, int, int]


@dataclass
class InkModel:
    achromatic: bool
    core_bgr: tuple[int, int, int]
    paper_bgr: tuple[int, int, int]
    lab_tolerance: float
    core_gray: float
    paper_gray: float


@dataclass
class SwatchTemplate:
    name: str
    swatch_box: Box
    marker_center: tuple[float, float]
    diameter: float
    marker_kind: str
    interior_fill: float
    raw_soft: np.ndarray = field(repr=False)
    raw_mask: np.ndarray = field(repr=False)
    line_nuisance: np.ndarray = field(repr=False)
    valid_weight: np.ndarray = field(repr=False)
    soft: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    edge: np.ndarray = field(repr=False)
    orientation: np.ndarray = field(repr=False)
    ink: InkModel
    # Opt-in B&W v46 metadata. Legacy and colour templates retain exact
    # historical behavior when no uncertainty profile is supplied.
    required_weight: np.ndarray | None = field(default=None, repr=False)
    matching_profile: str = 'legacy'
    proposal_scale: float = 1.0
    swatch_id: str = ''
    shape_hint: str = ''
    # Unclipped source gray values, separate from normalized matching ink.
    # Optional so legacy/colour constructors and matching remain unchanged.
    source_gray: np.ndarray | None = field(default=None, repr=False)
    # Opt-in model-completed silhouette. Observed RGB/gray is kept separate;
    # both grid and window matching consume raw_soft/soft = pure marker alpha.
    model_completed: bool = False
    observed_source_soft: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        """Series identity; legacy/colour templates retain their name key."""
        return self.swatch_id or self.name


@dataclass
class GridHypothesis:
    x: float
    y: float
    cell_x: int
    cell_y: int
    cell_left: int
    cell_top: int
    local_votes: int
    local_score: float


@dataclass
class Detection:
    template: str
    x: float
    y: float
    score: float
    contour_coverage: float
    contour_precision: float
    foreground_coverage: float
    direct_foreground_coverage: float
    centre_fill: float
    centre_fill_similarity: float
    supporting_cells: int
    orientation_bins: int
    radial_sectors: int
    raw_hypotheses: int
    swatch_id: str = ''
    shape_hint: str = ''
    colour_visibility: dict = field(default_factory=dict)
    boundary_evidence: dict = field(default_factory=dict)


@dataclass
class TemplateResult:
    template: SwatchTemplate
    grid_step: int
    grid_stride: int
    membership: np.ndarray = field(repr=False)
    plot_edge: np.ndarray = field(repr=False)
    hypotheses: list[GridHypothesis]
    vote_map: np.ndarray = field(repr=False)
    detections: list[Detection]
    baseline: list[Detection]
    refinement_diagnostics: dict = field(default_factory=dict)
    grid_diagnostics: dict = field(default_factory=dict)
    candidate_diagnostics: dict = field(default_factory=dict)
    clustering_diagnostics: dict = field(default_factory=dict)
    performance_diagnostics: dict = field(default_factory=dict)


def parse_box(text: str) -> Box:
    values = tuple(int(round(float(value))) for value in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("box must have x1>x0 and y1>y0")
    return values


def _box_crop(image: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def _bgr_to_lab(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.uint8)
    shape = arr.shape
    lab = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    return lab.reshape(shape).astype(np.float32)


def _tube_distance(
    pixels_lab: np.ndarray,
    ink_lab: np.ndarray,
    paper_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direction = paper_lab - ink_lab
    denominator = float(direction @ direction) or 1.0
    blend = np.clip(
        ((pixels_lab - ink_lab) @ direction) / denominator,
        0.0,
        1.0,
    )
    projected = ink_lab + blend[..., None] * direction
    perpendicular = np.linalg.norm(pixels_lab - projected, axis=-1)
    return perpendicular, blend


def estimate_ink_model(swatch: np.ndarray) -> InkModel:
    """Estimate one ink-to-paper colour ray from a graphical swatch crop."""
    pixels = swatch.reshape(-1, 3)
    gray = cv2.cvtColor(swatch, cv2.COLOR_BGR2GRAY).reshape(-1)
    hsv = cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    bright = gray >= np.percentile(gray, 70)
    paper_pixels = pixels[bright]
    if len(paper_pixels) == 0:
        paper_pixels = pixels
    paper_bgr = np.median(paper_pixels, axis=0).astype(np.uint8)
    paper_gray = float(np.median(gray[bright])) if bright.any() else 255.0

    coloured = (hsv[:, 1] >= 45) & (hsv[:, 2] <= 252)
    achromatic = int(coloured.sum()) < max(5, int(0.01 * len(pixels)))
    if achromatic:
        darkness = paper_gray - gray.astype(np.float32)
        threshold = max(8.0, float(np.percentile(darkness, 80)))
        core = pixels[darkness >= threshold]
        if len(core) < 4:
            core = pixels[np.argsort(gray)[: max(4, len(pixels) // 10)]]
    else:
        strength = hsv[:, 1].astype(np.float32) * (
            0.25 + (255.0 - hsv[:, 2].astype(np.float32)) / 255.0
        )
        eligible = coloured & (strength >= np.percentile(strength[coloured], 55))
        core = pixels[eligible]

    core_bgr_array = np.median(core, axis=0).astype(np.uint8)
    core_gray = float(
        cv2.cvtColor(core_bgr_array.reshape(1, 1, 3), cv2.COLOR_BGR2GRAY)[0, 0]
    )

    ink_lab = _bgr_to_lab(core_bgr_array)
    paper_lab = _bgr_to_lab(paper_bgr)
    core_lab = _bgr_to_lab(core)
    perpendicular, _ = _tube_distance(core_lab, ink_lab, paper_lab)
    mad = float(np.median(np.abs(perpendicular - np.median(perpendicular))))
    tolerance = float(np.clip(8.0 + 3.0 * 1.4826 * mad, 10.0, 30.0))

    return InkModel(
        achromatic=achromatic,
        core_bgr=tuple(int(value) for value in core_bgr_array),
        paper_bgr=tuple(int(value) for value in paper_bgr),
        lab_tolerance=tolerance,
        core_gray=core_gray,
        paper_gray=paper_gray,
    )


def ink_membership(image: np.ndarray, model: InkModel) -> np.ndarray:
    """Return soft per-pixel membership for one swatch ink."""
    if model.achromatic:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        scale = max(model.paper_gray - model.core_gray, 24.0)
        soft = (model.paper_gray - gray) / scale
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ink_lab = _bgr_to_lab(np.asarray(model.core_bgr, dtype=np.uint8))
    paper_lab = _bgr_to_lab(np.asarray(model.paper_bgr, dtype=np.uint8))
    perpendicular, blend = _tube_distance(lab, ink_lab, paper_lab)
    colour_fit = np.exp(-0.5 * (perpendicular / model.lab_tolerance) ** 2)
    ink_amount = np.clip((0.96 - blend) / 0.60, 0.0, 1.0)
    # A dark grey pixel can project onto the white->colour ray even though it
    # has no hue.  Requiring a little chroma prevents black error bars and text
    # from becoming blue/red marker fragments while retaining anti-aliased ink.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation_gate = np.clip((hsv[..., 1] - 10.0) / 35.0, 0.0, 1.0)
    return (colour_fit * ink_amount * saturation_gate).astype(np.float32)


def _longest_run(indices: np.ndarray, anchor: int) -> tuple[int, int]:
    values = set(int(value) for value in indices)
    lo = hi = int(anchor)
    while lo - 1 in values:
        lo -= 1
    while hi + 1 in values:
        hi += 1
    return lo, hi


def extract_swatch_template(
    image: np.ndarray,
    swatch_box: Box,
    name: str,
) -> SwatchTemplate:
    """Strip the connecting line and isolate the marker in a swatch crop."""
    swatch = _box_crop(image, swatch_box)
    if swatch.size == 0:
        raise ValueError(f"empty swatch box for {name}: {swatch_box}")
    ink = estimate_ink_model(swatch)
    soft_full = ink_membership(swatch, ink)
    threshold = 0.30 if ink.achromatic else 0.24
    binary = soft_full >= threshold
    binary = cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    ).astype(bool)

    spans = np.zeros(binary.shape[1], dtype=np.int32)
    for column in range(binary.shape[1]):
        rows = np.flatnonzero(binary[:, column])
        if len(rows):
            spans[column] = int(rows[-1] - rows[0] + 1)
    maximum_span = int(spans.max(initial=0))
    if maximum_span < 3:
        raise ValueError(f"no marker-sized ink in swatch {name}: {swatch_box}")

    marker_column = int(np.argmax(spans))
    tall_columns = np.flatnonzero(spans >= max(3, int(round(0.34 * maximum_span))))
    # An open marker has two tall side walls separated by its white interior.
    # Treat nearby tall-column runs as one marker instead of keeping only the
    # side containing argmax(spans), which previously reconstructed a semicircle.
    column_runs = []
    if len(tall_columns):
        run_left = run_right = int(tall_columns[0])
        for column in tall_columns[1:]:
            column = int(column)
            if column == run_right + 1:
                run_right = column
            else:
                column_runs.append([run_left, run_right])
                run_left = run_right = column
        column_runs.append([run_left, run_right])
    merge_gap = max(2, int(round(1.10 * maximum_span)))
    merged_runs = []
    for run in column_runs:
        if (
            merged_runs
            and run[0] - merged_runs[-1][1] - 1 <= merge_gap
            and run[1] - merged_runs[-1][0] + 1 <= int(round(1.90 * maximum_span))
        ):
            merged_runs[-1][1] = run[1]
        else:
            merged_runs.append(run)
    if not merged_runs:
        raise ValueError(f"could not isolate marker columns for {name}")
    left, right = max(
        merged_runs,
        key=lambda run: (
            int(spans[run[0] : run[1] + 1].sum()),
            -abs(0.5 * (run[0] + run[1]) - marker_column),
        ),
    )
    marker_rows = np.flatnonzero(binary[:, left : right + 1].any(axis=1))
    if len(marker_rows) == 0:
        raise ValueError(f"could not isolate marker rows for {name}")

    # Detect the connecting-line band from portions OUTSIDE the marker.  The
    # original swatch is never altered: this becomes a nuisance/unknown layer
    # while a complete marker model is reconstructed from the remaining ink.
    outside_left = binary[:, : max(0, left - 1)]
    outside_right = binary[:, min(binary.shape[1], right + 2) :]
    outside_width = outside_left.shape[1] + outside_right.shape[1]
    if outside_width:
        outside_support = outside_left.sum(axis=1) + outside_right.sum(axis=1)
        line_rows = outside_support >= max(2, int(round(0.22 * outside_width)))
    else:
        line_rows = np.zeros(binary.shape[0], dtype=bool)
    local_x = 0.5 * (left + right)
    local_y = 0.5 * (float(marker_rows[0]) + float(marker_rows[-1]))
    diameter = float(max(right - left + 1, marker_rows[-1] - marker_rows[0] + 1))

    marker_window = np.zeros_like(binary, dtype=bool)
    marker_window[marker_rows[0] : marker_rows[-1] + 1, left : right + 1] = True
    line_nuisance_full = np.zeros_like(binary, dtype=bool)
    line_nuisance_full[:, left : right + 1] = line_rows[:, None]
    observed_marker = binary & marker_window
    known_marker = marker_window & ~line_nuisance_full
    observed_known = observed_marker & known_marker

    # The visible upper/lower/side fragments define the outer marker envelope.
    # A convex hull completes the two small gaps where the legend line enters
    # and exits, without deleting or painting over any source pixel.
    point_y, point_x = np.nonzero(observed_known)
    reconstructed = observed_marker.copy()
    interior_fill = 0.5
    marker_kind = "unknown"
    if len(point_x) >= 4:
        hull = cv2.convexHull(np.column_stack((point_x, point_y)).astype(np.int32))
        outer = np.zeros_like(binary, dtype=np.uint8)
        cv2.fillConvexPoly(outer, hull, 1)
        outer = outer.astype(bool) & marker_window

        distance_inside = cv2.distanceTransform(outer.astype(np.uint8), cv2.DIST_L2, 5)
        interior = outer & (distance_inside >= max(1.5, 0.20 * diameter))
        known_interior = interior & known_marker
        if int(known_interior.sum()) >= 3:
            interior_fill = float(observed_marker[known_interior].mean())
        else:
            interior_fill = float(observed_known.sum() / max(int(known_marker.sum()), 1))

        if interior_fill >= 0.52:
            marker_kind = "filled"
            reconstructed = outer
        else:
            marker_kind = "open"
            best_ring = observed_known.copy()
            best_score = -1.0
            maximum_thickness = max(1, int(round(0.30 * diameter)))
            for thickness in range(1, maximum_thickness + 1):
                ring = outer & (distance_inside <= thickness + 0.35)
                expected = ring & known_marker
                true_positive = int((expected & observed_marker).sum())
                precision = true_positive / max(int(expected.sum()), 1)
                recall = true_positive / max(int(observed_known.sum()), 1)
                score = 2.0 * precision * recall / max(precision + recall, 1e-6)
                if score > best_score:
                    best_score = score
                    best_ring = ring
            reconstructed = best_ring

    radius = max(3, int(math.ceil(0.58 * diameter)))
    size = 2 * radius + 1

    def padded_patch(array: np.ndarray, fill: float = 0.0) -> np.ndarray:
        canvas = np.full((size, size), fill, dtype=array.dtype)
        x0 = int(round(local_x)) - radius
        y0 = int(round(local_y)) - radius
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(array.shape[1], x0 + size)
        src_y1 = min(array.shape[0], y0 + size)
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas[
            dst_y0 : dst_y0 + src_y1 - src_y0,
            dst_x0 : dst_x0 + src_x1 - src_x0,
        ] = array[src_y0:src_y1, src_x0:src_x1]
        return canvas

    raw_soft = padded_patch(soft_full.astype(np.float32))
    raw_mask = padded_patch(binary.astype(np.uint8)).astype(bool)
    line_nuisance = padded_patch(line_nuisance_full.astype(np.uint8)).astype(bool)
    mask = padded_patch(reconstructed.astype(np.uint8)).astype(bool)
    soft = mask.astype(np.float32)
    # Reconstructed pixels underneath the legend line are valid geometric
    # evidence, but receive lower confidence than directly observed marker ink.
    valid_weight = np.ones(mask.shape, dtype=np.float32)
    valid_weight[line_nuisance] = 0.35
    mask_u8 = (mask.astype(np.uint8) * 255)
    edge = cv2.Canny(mask_u8, 40, 100) > 0
    if int(edge.sum()) < 6:
        edge = cv2.morphologyEx(
            mask_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        ) > 0
    orientation = _edge_orientation(soft)

    absolute_center = (
        float(swatch_box[0]) + local_x,
        float(swatch_box[1]) + local_y,
    )
    return SwatchTemplate(
        name=name,
        swatch_box=swatch_box,
        marker_center=absolute_center,
        diameter=diameter,
        marker_kind=marker_kind,
        interior_fill=interior_fill,
        raw_soft=raw_soft,
        raw_mask=raw_mask,
        line_nuisance=line_nuisance,
        valid_weight=valid_weight,
        soft=soft,
        mask=mask,
        edge=edge,
        orientation=orientation,
        ink=ink,
    )


def _edge_orientation(soft: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(soft.astype(np.float32), (0, 0), 0.65)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return np.mod(np.arctan2(gy, gx), 2.0 * np.pi)


def _orientation_bin(angles: np.ndarray, bins: int) -> np.ndarray:
    return np.floor(np.mod(angles, 2.0 * np.pi) * bins / (2.0 * np.pi)).astype(int) % bins


def _template_r_table(
    template: SwatchTemplate,
    orientation_bins: int,
) -> dict[int, np.ndarray]:
    ys, xs = np.nonzero(template.edge)
    bins = _orientation_bin(template.orientation[ys, xs], orientation_bins)
    centre_y = 0.5 * (template.edge.shape[0] - 1)
    centre_x = 0.5 * (template.edge.shape[1] - 1)
    result: dict[int, list[tuple[float, float]]] = {
        index: [] for index in range(orientation_bins)
    }
    for x, y, orientation_bin in zip(xs, ys, bins, strict=True):
        result[int(orientation_bin)].append((centre_x - x, centre_y - y))
    return {
        key: np.asarray(value, dtype=np.float32).reshape(-1, 2)
        for key, value in result.items()
    }


def _cell_hypotheses(
    plot_edge: np.ndarray,
    plot_orientation: np.ndarray,
    template: SwatchTemplate,
    plot_area: Box,
    grid_step: int,
    grid_stride: int,
    orientation_bins: int = 24,
    local_top_k: int = 4,
) -> list[GridHypothesis]:
    x0, y0, x1, y1 = plot_area
    r_table = _template_r_table(template, orientation_bins)
    centre_quantum = max(1, int(round(template.diameter * 0.07)))
    hypotheses: list[GridHypothesis] = []

    # Keep every original non-overlapping window and add shifted phases.  This
    # matters when the cell size is odd: replacing a 7 px stride with 4 px
    # would otherwise remove the original origins 7, 14, 21, ... instead of
    # adding overlapping windows to them.
    phase_count = max(1, int(round(grid_step / max(grid_stride, 1))))
    phases = sorted(
        {int(round(index * grid_step / phase_count)) for index in range(phase_count)}
    )
    tops = sorted(
        {
            top
            for phase in phases
            for top in range(y0 + phase, y1, grid_step)
        }
    )
    lefts = sorted(
        {
            left
            for phase in phases
            for left in range(x0 + phase, x1, grid_step)
        }
    )

    for cell_y, top in enumerate(tops):
        bottom = min(top + grid_step, y1)
        for cell_x, left in enumerate(lefts):
            right = min(left + grid_step, x1)
            local_edge = plot_edge[top:bottom, left:right]
            local_y, local_x = np.nonzero(local_edge)
            if len(local_x) < 2:
                continue
            global_x = local_x + left
            global_y = local_y + top
            edge_bins = _orientation_bin(
                plot_orientation[global_y, global_x], orientation_bins
            )

            accumulator: dict[tuple[int, int], int] = {}
            for px, py, edge_bin in zip(global_x, global_y, edge_bins, strict=True):
                offsets = []
                for delta in (-1, 0, 1):
                    offsets_for_bin = r_table[(int(edge_bin) + delta) % orientation_bins]
                    if len(offsets_for_bin):
                        offsets.append(offsets_for_bin)
                if not offsets:
                    continue
                all_offsets = np.concatenate(offsets, axis=0)
                centres = all_offsets + np.asarray((px, py), dtype=np.float32)
                for centre_x, centre_y in centres:
                    if not (x0 <= centre_x < x1 and y0 <= centre_y < y1):
                        continue
                    key = (
                        int(round(centre_x / centre_quantum)),
                        int(round(centre_y / centre_quantum)),
                    )
                    accumulator[key] = accumulator.get(key, 0) + 1

            if not accumulator:
                continue
            ordered = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
            normalizer = max(2.0, math.sqrt(len(local_x) * max(int(template.edge.sum()), 1)))
            for (qx, qy), votes in ordered[:local_top_k]:
                local_score = float(votes / normalizer)
                if votes < 2 or local_score < 0.10:
                    continue
                hypotheses.append(
                    GridHypothesis(
                        x=float(qx * centre_quantum),
                        y=float(qy * centre_quantum),
                        cell_x=cell_x,
                        cell_y=cell_y,
                        cell_left=left,
                        cell_top=top,
                        local_votes=int(votes),
                        local_score=local_score,
                    )
                )
    return hypotheses


def _extract_aligned_patch(
    array: np.ndarray,
    x: float,
    y: float,
    shape: tuple[int, int],
    fill: float = 0.0,
) -> np.ndarray:
    height, width = shape
    left = int(round(x - 0.5 * (width - 1)))
    top = int(round(y - 0.5 * (height - 1)))
    patch = np.full(shape, fill, dtype=array.dtype)
    src_x0 = max(0, left)
    src_y0 = max(0, top)
    src_x1 = min(array.shape[1], left + width)
    src_y1 = min(array.shape[0], top + height)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return patch
    dst_x0 = src_x0 - left
    dst_y0 = src_y0 - top
    patch[
        dst_y0 : dst_y0 + src_y1 - src_y0,
        dst_x0 : dst_x0 + src_x1 - src_x0,
    ] = array[src_y0:src_y1, src_x0:src_x1]
    return patch


def _contour_geometry(template, sectors=8):
    edge, marker = template.edge.astype(bool), template.mask.astype(bool)
    radius = max(1, int(round(.08*template.diameter)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    band = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    yy, xx = np.indices(edge.shape)
    angles = np.mod(np.arctan2(yy-.5*(edge.shape[0]-1), xx-.5*(edge.shape[1]-1)), 2.*np.pi)
    return dict(edge=edge, kernel=kernel, inside_band=band & marker,
                outside_band=band & ~marker,
                sector_index=np.floor(angles*sectors/(2.*np.pi)).astype(int) % sectors)


class _IndexedHypotheses(Sequence):
    """Immutable per-call spatial index; final radius test is original Python."""
    def __init__(self, items):
        self.items = items
        self.tree = cKDTree(np.asarray([(h.x, h.y) for h in items], np.float64).reshape(-1, 2))
        self.queries = 0
        self.examined = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def near(self, x, y, radius):
        # The generous lookup margin prevents a tree's sqrt/rounding at the
        # exact radius from excluding a point accepted by the scalar predicate.
        indices = self.tree.query_ball_point((x, y), radius+max(1e-10, radius*1e-12), return_sorted=True)
        self.queries += 1
        self.examined += len(indices)
        return [self.items[i] for i in indices
                if (self.items[i].x-x)**2+(self.items[i].y-y)**2 <= radius**2]


def _candidate_geometry(template):
    yy, xx = np.indices(template.mask.shape)
    cy, cx = .5*(template.mask.shape[0]-1), .5*(template.mask.shape[1]-1)
    disk = (xx-cx)**2+(yy-cy)**2 <= max(1.5, .22*template.diameter)**2
    return dict(distance_to_template=cv2.distanceTransform((~template.edge).astype(np.uint8), cv2.DIST_L2, 3),
                edge_weight=template.valid_weight*template.edge,
                marker_weight=template.valid_weight*template.mask,
                centre_disk=disk, contour=_contour_geometry(template))


def _classify_contour_segments(
    template: SwatchTemplate,
    plot_ink: np.ndarray,
    edge_patch: np.ndarray,
    distance_to_plot: np.ndarray,
    edge_tolerance: float,
    sectors: int = 8,
    _geometry=None,
    _prepared=None,
    other_patch=None,
) -> dict:
    """Classify expected contour arcs under an additive black-ink model.

    A curve or another marker can add black pixels across an expected marker
    boundary.  The brightness step then disappears, so absence of a Canny edge
    is not itself a contradiction.  An arc is treated as ink-occluded when the
    required marker-side ink is present and additional ink crosses to the
    outside.  Missing required marker ink remains a hard mismatch.
    """
    geometry = _contour_geometry(template, sectors) if _geometry is None else _geometry
    edge, kernel = geometry['edge'], geometry['kernel']
    inside_band, outside_band = geometry['inside_band'], geometry['outside_band']
    if _prepared is None:
        ink_near = cv2.dilate(plot_ink.astype(np.uint8), kernel) > 0
        visible_edge = edge & (distance_to_plot <= edge_tolerance)
        inside_ink_near = cv2.dilate((plot_ink & inside_band).astype(np.uint8), kernel) > 0
        outside_ink_near = cv2.dilate((plot_ink & outside_band).astype(np.uint8), kernel) > 0
    else:
        ink_near = _prepared['ink_near']
        visible_edge = edge & _prepared['edge_near']
        inside_ink_near, outside_ink_near = _prepared['inside_ink_near'], _prepared['outside_ink_near']
    ink_occluded_edge = edge & ~visible_edge & inside_ink_near & outside_ink_near
    mismatch_edge = edge & ~visible_edge & ~ink_occluded_edge
    if other_patch is not None:
        observable = other_patch < .5
        visible_edge &= observable
        ink_occluded_edge &= observable
        mismatch_edge &= observable

    sector_index = geometry['sector_index']

    visible_sectors = set()
    occluded_sectors = set()
    mismatch_sectors = set()
    supported_sectors = set()
    sector_rows = []
    for sector in range(sectors):
        sector_edge = edge & (sector_index == sector)
        edge_count = int(sector_edge.sum())
        if edge_count == 0:
            continue
        required = inside_band & (sector_index == sector)
        if other_patch is not None:
            required &= other_patch < .5
        required_count = int(required.sum())
        required_coverage = float(
            (required & ink_near).sum() / max(required_count, 1)
        )
        visible_fraction = float(
            (visible_edge & sector_edge).sum() / edge_count
        )
        occluded_fraction = float(
            (ink_occluded_edge & sector_edge).sum() / edge_count
        )
        mismatch_fraction = float(
            (mismatch_edge & sector_edge).sum() / edge_count
        )

        if visible_fraction >= 0.18:
            visible_sectors.add(sector)
        if occluded_fraction >= 0.18 and required_coverage >= 0.55:
            occluded_sectors.add(sector)
        if required_coverage < 0.45 and mismatch_fraction >= 0.35:
            mismatch_sectors.add(sector)
        if (
            required_coverage >= 0.55
            and visible_fraction + occluded_fraction >= 0.18
        ):
            supported_sectors.add(sector)
        sector_rows.append(
            {
                "sector": sector,
                "required_coverage": required_coverage,
                "visible_fraction": visible_fraction,
                "occluded_fraction": occluded_fraction,
                "mismatch_fraction": mismatch_fraction,
            }
        )

    edge_count = max(int(edge.sum()), 1)
    return {
        "visible_edge": visible_edge,
        "ink_occluded_edge": ink_occluded_edge,
        "mismatch_edge": mismatch_edge,
        "visible_fraction": float(visible_edge.sum() / edge_count),
        "occluded_fraction": float(ink_occluded_edge.sum() / edge_count),
        "mismatch_fraction": float(mismatch_edge.sum() / edge_count),
        "visible_sectors": visible_sectors,
        "occluded_sectors": occluded_sectors,
        "mismatch_sectors": mismatch_sectors,
        "supported_sectors": supported_sectors,
        "sectors": sector_rows,
    }


def _verify_candidate(
    x: float,
    y: float,
    template: SwatchTemplate,
    membership: np.ndarray,
    plot_edge: np.ndarray,
    plot_orientation: np.ndarray,
    plot_area: Box,
    grid_step: int,
    grid_stride: int,
    hypotheses: Sequence[GridHypothesis],
    orientation_bins: int = 24,
    _geometry=None,
    _prepared=None,
    occlusion_mask=None,
) -> Detection | None:
    edge_patch = _extract_aligned_patch(plot_edge, x, y, template.edge.shape)
    membership_patch = _extract_aligned_patch(
        membership, x, y, template.edge.shape
    )
    orientation_patch = _extract_aligned_patch(
        plot_orientation, x, y, template.edge.shape
    )
    other_patch = None
    visibility = {}
    if occlusion_mask is not None:
        patch = _extract_aligned_patch(occlusion_mask, x, y, template.edge.shape)
        if float((patch * template.mask).sum()) / max(int(template.mask.sum()),1) >= .03:
            from bw_colour_visibility_v46 import measure
            other_patch = patch
            visibility = measure(template, membership_patch, patch)
            if not visibility['candidate_supported']:
                return None
            # The boundary between own colour and an occluder is not evidence
            # of the marker's external outline.
            edge_patch = edge_patch & ~(cv2.dilate((patch >= .5).astype(np.uint8), np.ones((3,3),np.uint8)) > 0)
            membership_patch = membership_patch * (1.-patch)
            _prepared = None
    if not edge_patch.any():
        return None

    distance_to_plot = None if _prepared is not None else cv2.distanceTransform(
        (~edge_patch).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_template = _geometry['distance_to_template'] if _geometry is not None else cv2.distanceTransform(
        (~template.edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    edge_tolerance = max(1.25, 0.08 * template.diameter)
    template_match = template.edge & (_prepared['edge_near'] if _prepared is not None
                                      else distance_to_plot <= edge_tolerance)
    plot_match = edge_patch & (distance_to_template <= edge_tolerance)
    edge_weight = _geometry['edge_weight'] if _geometry is not None else template.valid_weight * template.edge
    contour_coverage = float(
        (template_match * edge_weight).sum() / max(float(edge_weight.sum()), 1.0)
    )
    contour_precision = float(plot_match.sum() / max(int(edge_patch.sum()), 1))

    membership_threshold = 0.45 if template.ink.achromatic else 0.24
    plot_ink = membership_patch >= membership_threshold
    dilation_radius = max(1, int(round(0.06 * template.diameter)))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    dilated_plot_ink = (_prepared['dilated_plot_ink'] if _prepared is not None
                        else cv2.dilate(plot_ink.astype(np.uint8), dilation_kernel) > 0)
    foreground_match = template.mask & dilated_plot_ink
    marker_weight = _geometry['marker_weight'] if _geometry is not None else template.valid_weight * template.mask
    foreground_coverage = float(
        (foreground_match * marker_weight).sum() / max(float(marker_weight.sum()), 1.0)
    )
    direct_foreground_coverage = float(
        ((template.mask & plot_ink) * marker_weight).sum()
        / max(float(marker_weight.sum()), 1.0)
    )

    contour_evidence = _classify_contour_segments(
        template,
        plot_ink,
        edge_patch,
        distance_to_plot,
        edge_tolerance,
        _geometry=None if _geometry is None else _geometry['contour'],
        _prepared=_prepared,
        other_patch=other_patch,
    )

    centre_x = 0.5 * (template.edge.shape[1] - 1)
    centre_y = 0.5 * (template.edge.shape[0] - 1)
    yy, xx = np.indices(template.mask.shape)
    centre_radius = max(1.5, 0.22 * template.diameter)
    centre_disk = (_geometry['centre_disk'] if _geometry is not None
                   else (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= centre_radius**2)
    template_centre_fill = float(template.mask[centre_disk].mean())
    plot_centre_fill = float(plot_ink[centre_disk].mean())
    centre_fill_similarity = float(1.0 - abs(template_centre_fill - plot_centre_fill))
    if visibility:
        plot_centre_fill = visibility['centre_fill']
        template_centre_fill = visibility['expected_centre_fill']
        centre_fill_similarity = visibility['centre_similarity'] * visibility['centre_visible_fraction']

    matched_y, matched_x = np.nonzero(template_match)
    if len(matched_x) == 0:
        return None

    foreground_y, foreground_x = np.nonzero(foreground_match)
    radial_angles = np.mod(
        np.arctan2(foreground_y - centre_y, foreground_x - centre_x),
        2.0 * np.pi,
    )
    radial_sectors = int(len(set(_orientation_bin(radial_angles, 8).tolist())))

    local_y, local_x = np.nonzero(plot_ink)
    if len(local_x):
        horizontal_span = float(local_x.max() - local_x.min() + 1)
        vertical_span = float(local_y.max() - local_y.min() + 1)
    else:
        horizontal_span = vertical_span = 0.0

    template_bins = _orientation_bin(
        template.orientation[matched_y, matched_x], orientation_bins
    )
    plot_bins = _orientation_bin(
        orientation_patch[matched_y, matched_x], orientation_bins
    )
    circular_difference = np.minimum(
        (template_bins - plot_bins) % orientation_bins,
        (plot_bins - template_bins) % orientation_bins,
    )
    orientation_ok = circular_difference <= 2
    used_orientation_bins = int(len(set(template_bins[orientation_ok].tolist())))

    x0, y0, _, _ = plot_area
    supporting_grid_cells = {
        (
            int((round(x - centre_x + px) - x0) // grid_step),
            int((round(y - centre_y + py) - y0) // grid_step),
        )
        for px, py in zip(matched_x, matched_y, strict=True)
    }
    nearby_hypotheses = hypotheses.near(x, y, max(2.0, .22*template.diameter)) if isinstance(hypotheses, _IndexedHypotheses) else [
        hypothesis
        for hypothesis in hypotheses
        if (hypothesis.x - x) ** 2 + (hypothesis.y - y) ** 2
        <= max(2.0, 0.22 * template.diameter) ** 2
    ]
    hypothesis_cells = {
        (hypothesis.cell_left, hypothesis.cell_top)
        for hypothesis in nearby_hypotheses
    }
    # Overlapping windows reuse many of the same pixels.  Convert their raw
    # count back to the equivalent number of independent, non-overlapping
    # windows so that overlap does not earn a score bonus merely by repeating
    # the same evidence.
    phase_count = max(1, int(round(grid_step / max(grid_stride, 1))))
    overlap_multiplicity = float(phase_count**2)
    effective_hypothesis_cells = int(
        math.ceil(len(hypothesis_cells) / overlap_multiplicity)
    )
    supporting_cells = max(len(supporting_grid_cells), effective_hypothesis_cells)

    required_cells = 3 if template.diameter < 9 else 4
    minimum_supported_sectors = 3 if template.diameter < 9 else 4
    minimum_contour_coverage = 0.28 if template.diameter >= 9 else 0.36
    minimum_foreground_coverage = 0.34 if template.diameter >= 9 else 0.42
    minimum_direct_coverage = (
        0.45 if template.diameter < 9 else 0.32 if template.diameter < 20 else 0.25
    )
    minimum_orientation_bins = 4 if template.diameter < 9 else 5
    if template_centre_fill >= 0.65:
        centre_fill_ok = plot_centre_fill >= max(0.32, 0.45 * template_centre_fill)
    elif template_centre_fill <= 0.35:
        # A crossing line may occupy part of an open marker's centre, so the
        # upper bound is intentionally tolerant while still rejecting a solid
        # disk masquerading as a ring.
        centre_fill_ok = plot_centre_fill <= 0.70
    else:
        centre_fill_ok = centre_fill_similarity >= 0.45
    minimum_cells = required_cells
    minimum_radial = 4
    minimum_span = .48
    if visibility:
        # Visible fragments can propose a typed hypothesis, not a confirmed
        # identity. The pipeline rechecks visibility at final window geometry.
        visible = visibility['edge_visible_fraction']
        minimum_cells = max(2, int(math.ceil(required_cells*visible)))
        minimum_supported_sectors = max(2, int(math.ceil(minimum_supported_sectors*visible)))
        minimum_orientation_bins = max(3, int(math.ceil(minimum_orientation_bins*visible)))
        minimum_contour_coverage *= max(.5, visible)
        minimum_foreground_coverage = .20
        minimum_direct_coverage = .20
        minimum_radial = 2
        minimum_span = .35
        if visibility['centre_visible_fraction'] < .20:
            centre_fill_ok = True  # unknown, not a positive fill match
    if (
        supporting_cells < minimum_cells
        or len(contour_evidence["supported_sectors"]) < minimum_supported_sectors
        or len(contour_evidence["mismatch_sectors"]) >= 3
        or used_orientation_bins < minimum_orientation_bins
        or radial_sectors < minimum_radial
        or horizontal_span < minimum_span * template.diameter
        or vertical_span < minimum_span * template.diameter
        or contour_coverage < minimum_contour_coverage
        or foreground_coverage < minimum_foreground_coverage
        or direct_foreground_coverage < minimum_direct_coverage
        or not centre_fill_ok
    ):
        return None

    effective_hypotheses = len(nearby_hypotheses) / overlap_multiplicity
    vote_strength = min(1.0, effective_hypotheses / max(3.0, supporting_cells))
    spatial_strength = min(1.0, supporting_cells / 5.0)
    orientation_strength = min(1.0, used_orientation_bins / 8.0)
    asymmetric_contour_support = min(
        1.0,
        contour_evidence["visible_fraction"]
        + contour_evidence["occluded_fraction"],
    )
    visible_signature = min(
        1.0, len(contour_evidence["visible_sectors"]) / 2.0
    )
    mismatch_sector_fraction = (
        len(contour_evidence["mismatch_sectors"]) / 8.0
    )
    score = (
        0.21 * asymmetric_contour_support
        + 0.20 * foreground_coverage
        + 0.17 * direct_foreground_coverage
        + 0.10 * centre_fill_similarity
        + 0.10 * vote_strength
        + 0.10 * spatial_strength
        + 0.07 * orientation_strength
        + 0.05 * visible_signature
        - 0.18 * mismatch_sector_fraction
    )
    if visibility:
        score -= .35*visibility['core_paper_fraction'] + .03*visibility['other_fraction']
    # Compatibility (ink can hide a boundary) is not positive shape evidence.
    # Keep proposals, but reward actual ink-to-paper transitions for BW filled
    # identities. The GPU candidate path calls the exact same small helper.
    from bw_boundary_evidence_v46 import grid_score_adjustment
    boundary_delta, boundary_evidence = grid_score_adjustment(
        template, membership_patch, asymmetric_contour_support, other_patch)
    score += boundary_delta
    return Detection(
        template=template.key,
        x=float(x),
        y=float(y),
        score=float(score),
        contour_coverage=contour_coverage,
        contour_precision=contour_precision,
        foreground_coverage=foreground_coverage,
        direct_foreground_coverage=direct_foreground_coverage,
        centre_fill=plot_centre_fill,
        centre_fill_similarity=centre_fill_similarity,
        supporting_cells=int(supporting_cells),
        orientation_bins=used_orientation_bins,
        radial_sectors=radial_sectors,
        raw_hypotheses=len(nearby_hypotheses),
        boundary_evidence=boundary_evidence,
        swatch_id=template.swatch_id,
        shape_hint=template.shape_hint or template.name,
        colour_visibility=visibility,
    )


def _verify_candidates_many(centers, template, membership, plot_edge, plot_orientation,
                            plot_area, grid_step, evidence, *, batch_size=512,
                            verifier=None, geometry=None):
    """Batch candidate patches, metrics and gates on GPU with indexed evidence.

    ``evidence`` has one (stride, IndexedHypotheses) pair per centre. Exclusive
    and overlapping seeds therefore never accidentally share voting evidence.
    """
    from bw_gpu_candidate_verifier import GpuCandidateVerifier
    if verifier is None:
        verifier = GpuCandidateVerifier(template, membership, plot_edge, batch_size)
    geometry = _candidate_geometry(template) if geometry is None else geometry
    return verifier.verify_many(centers, plot_orientation, plot_area, grid_step,
                                evidence, geometry=geometry)


def _alignment_objective(
    x: float,
    y: float,
    template: SwatchTemplate,
    membership: np.ndarray,
    plot_edge: np.ndarray,
    occlusion_mask=None,
) -> float:
    edge_patch = _extract_aligned_patch(plot_edge, x, y, template.edge.shape)
    if occlusion_mask is not None:
        other = _extract_aligned_patch(occlusion_mask, x, y, template.edge.shape)
        edge_patch = edge_patch & ~(cv2.dilate((other >= .5).astype(np.uint8), np.ones((3,3),np.uint8)) > 0)
    if not edge_patch.any():
        return -1.0
    membership_patch = _extract_aligned_patch(
        membership, x, y, template.edge.shape
    )
    distance_to_plot = cv2.distanceTransform(
        (~edge_patch).astype(np.uint8), cv2.DIST_L2, 3
    )
    edge_tolerance = max(1.25, 0.08 * template.diameter)
    contour_coverage = float(
        (template.edge & (distance_to_plot <= edge_tolerance)).sum()
        / max(int(template.edge.sum()), 1)
    )

    threshold = 0.45 if template.ink.achromatic else 0.24
    plot_ink = membership_patch >= threshold
    dilation_radius = max(1, int(round(0.06 * template.diameter)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    dilated = cv2.dilate(plot_ink.astype(np.uint8), kernel) > 0
    foreground_coverage = float(
        (template.mask & dilated).sum() / max(int(template.mask.sum()), 1)
    )
    direct_coverage = float(
        (template.mask & plot_ink).sum() / max(int(template.mask.sum()), 1)
    )

    yy, xx = np.indices(template.mask.shape)
    centre_x = 0.5 * (template.mask.shape[1] - 1)
    centre_y = 0.5 * (template.mask.shape[0] - 1)
    centre_radius = max(1.5, 0.22 * template.diameter)
    centre_disk = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= centre_radius**2
    fill_similarity = 1.0 - abs(
        float(template.mask[centre_disk].mean())
        - float(plot_ink[centre_disk].mean())
    )
    if occlusion_mask is not None:
        other = _extract_aligned_patch(occlusion_mask, x, y, template.edge.shape)
        from bw_colour_visibility_v46 import measure
        state = measure(template, membership_patch, other)
        if not state['candidate_supported']:
            return -1.
        visible = 1.-other
        # Keep full denominators for rewards. Unknown centre pixels earn
        # neither a filled-centre bonus nor a missing-centre penalty. Keep
        # the same paper cost throughout the search, including clear pixels.
        contour_coverage = float((template.edge*(distance_to_plot<=edge_tolerance)*visible).sum()/max(int(template.edge.sum()),1))
        foreground_coverage = float((template.mask*dilated*visible).sum()/max(int(template.mask.sum()),1))
        direct_coverage = state['positive_fraction']
        fill_similarity = state['centre_similarity']*state['centre_visible_fraction']
        return (.44*contour_coverage+.23*foreground_coverage+.18*direct_coverage
                +.15*fill_similarity-.55*state['core_paper_fraction']-.03*state['other_fraction'])
    return (
        0.44 * contour_coverage
        + 0.23 * foreground_coverage
        + 0.18 * direct_coverage
        + 0.15 * fill_similarity
    )


def _refine_candidate_center(
    x: float,
    y: float,
    template: SwatchTemplate,
    membership: np.ndarray,
    plot_edge: np.ndarray,
    plot_area: Box,
    occlusion_mask=None,
) -> tuple[float, float]:
    radius = max(2, int(round(0.30 * template.diameter)))
    x0, y0, x1, y1 = plot_area
    best = (float(x), float(y))
    best_score = -1.0
    for candidate_y in range(int(round(y)) - radius, int(round(y)) + radius + 1):
        if not y0 <= candidate_y < y1:
            continue
        for candidate_x in range(int(round(x)) - radius, int(round(x)) + radius + 1):
            if not x0 <= candidate_x < x1:
                continue
            score = _alignment_objective(
                candidate_x,
                candidate_y,
                template,
                membership,
                plot_edge,
                **({'occlusion_mask':occlusion_mask} if occlusion_mask is not None else {}),
            )
            if score > best_score:
                best_score = score
                best = (float(candidate_x), float(candidate_y))
    return best


def _candidate_centres(
    hypotheses: Sequence[GridHypothesis],
    template: SwatchTemplate,
    plot_area: Box,
    overlap_multiplicity: float = 1.0,
) -> tuple[list[tuple[float, float, int, float]], np.ndarray]:
    x0, y0, x1, y1 = plot_area
    vote_map = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    if not hypotheses:
        return [], vote_map

    points = np.asarray([(item.x, item.y) for item in hypotheses], dtype=np.float32)
    tree = cKDTree(points)
    radius = max(2.0, 0.22 * template.diameter)
    ranked: list[tuple[float, float, int, float]] = []
    for index, hypothesis in enumerate(hypotheses):
        neighbour_indices = tree.query_ball_point(points[index], radius)
        cells = {
            (hypotheses[neighbour].cell_left, hypotheses[neighbour].cell_top)
            for neighbour in neighbour_indices
        }
        weights = np.asarray(
            [hypotheses[neighbour].local_score for neighbour in neighbour_indices],
            dtype=np.float32,
        )
        neighbours = points[neighbour_indices]
        centre = np.average(neighbours, axis=0, weights=np.maximum(weights, 1e-4))
        effective_cells = len(cells) / max(overlap_multiplicity, 1.0)
        density = float(
            effective_cells
            + np.clip(weights, 0.0, 2.0).sum() / max(overlap_multiplicity, 1.0)
        )
        ranked.append(
            (float(centre[0]), float(centre[1]), int(math.ceil(effective_cells)), density)
        )
        px = int(round(centre[0])) - x0
        py = int(round(centre[1])) - y0
        if 0 <= px < vote_map.shape[1] and 0 <= py < vote_map.shape[0]:
            vote_map[py, px] = max(vote_map[py, px], density)

    ranked.sort(key=lambda item: item[3], reverse=True)
    selected: list[tuple[float, float, int, float]] = []
    nms_radius = max(2.0, 0.28 * template.diameter)
    for candidate in ranked:
        if any(
            (candidate[0] - kept[0]) ** 2 + (candidate[1] - kept[1]) ** 2
            <= nms_radius**2
            for kept in selected
        ):
            continue
        selected.append(candidate)
    if vote_map.any():
        sigma = max(0.8, 0.10 * template.diameter)
        vote_map = cv2.GaussianBlur(vote_map, (0, 0), sigma)
    return selected, vote_map


def _nms_detections(
    detections: Iterable[Detection],
    radius: float,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            (detection.x - other.x) ** 2 + (detection.y - other.y) ** 2
            <= radius**2
            for other in kept
        ):
            continue
        kept.append(detection)
    return sorted(kept, key=lambda item: (item.x, item.y))


def exact_template_baseline(
    membership: np.ndarray,
    template: SwatchTemplate,
    plot_area: Box,
    threshold: float = 0.78,
) -> list[Detection]:
    """Conventional full-template matching used only as an experiment baseline."""
    kernel = template.soft.astype(np.float32)
    if membership.shape[0] < kernel.shape[0] or membership.shape[1] < kernel.shape[1]:
        return []
    score_map = cv2.matchTemplate(
        membership.astype(np.float32), kernel, cv2.TM_CCORR_NORMED
    )
    x0, y0, x1, y1 = plot_area
    half_width = 0.5 * (kernel.shape[1] - 1)
    half_height = 0.5 * (kernel.shape[0] - 1)
    valid = np.zeros_like(score_map, dtype=bool)
    valid[
        max(0, int(math.ceil(y0 - half_height))) : min(
            score_map.shape[0], int(math.floor(y1 - half_height))
        ),
        max(0, int(math.ceil(x0 - half_width))) : min(
            score_map.shape[1], int(math.floor(x1 - half_width))
        ),
    ] = True
    local_max = score_map == cv2.dilate(score_map, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(valid & local_max & (score_map >= threshold))
    detections = [
        Detection(
            template=template.key,
            x=float(x + half_width),
            y=float(y + half_height),
            score=float(score_map[y, x]),
            contour_coverage=0.0,
            contour_precision=0.0,
            foreground_coverage=0.0,
            direct_foreground_coverage=0.0,
            centre_fill=0.0,
            centre_fill_similarity=0.0,
            supporting_cells=0,
            orientation_bins=0,
            radial_sectors=0,
            raw_hypotheses=0,
            swatch_id=template.swatch_id,
            shape_hint=template.shape_hint or template.name,
        )
        for x, y in zip(xs, ys, strict=True)
    ]
    return _nms_detections(detections, max(3.0, 0.65 * template.diameter))


def detect_template(
    image: np.ndarray,
    template: SwatchTemplate,
    plot_area: Box,
    grid_fraction: float = 0.5,
    ignore_regions: Sequence[Box] = (),
    grid_overlap: float = 0.0,
    defer_nms: bool = False,
    refinement_backend: str = 'cpu',
    gpu_batch_size: int = 512,
    grid_backend: str = 'cpu',
    preprocessing_cache: dict | None = None,
    compute_baseline: bool = True,
    grid_identity_competitor=None,
    occlusion_mask=None,
) -> TemplateResult:
    started = time.perf_counter()
    if not 0.0 <= grid_overlap < 1.0:
        raise ValueError("grid_overlap must satisfy 0 <= grid_overlap < 1")
    if refinement_backend not in {'cpu', 'cuda'}:
        raise ValueError('refinement_backend must be cpu or cuda')
    if grid_backend not in {'cpu', 'cuda'}:
        raise ValueError('grid_backend must be cpu or cuda')
    if not isinstance(gpu_batch_size, int) or gpu_batch_size <= 0:
        raise ValueError('gpu_batch_size must be a positive integer')
    if not isinstance(compute_baseline, bool):
        raise ValueError('compute_baseline must be Boolean')
    from bw_colour_visibility_v46 import validate_mask
    occlusion_mask = validate_mask(occlusion_mask, image.shape[:2])
    visibility_options = {} if occlusion_mask is None else {'occlusion_mask':occlusion_mask}
    requested_refinement = refinement_backend
    if occlusion_mask is not None:
        # Grid voting may still run on CUDA. Candidate/refinement kernels do
        # not yet support three-state constraints: explicitly use CPU there.
        refinement_backend = 'cpu'
    x0, y0, x1, y1 = plot_area
    # Call-scoped, single-entry reuse avoids recomputing identical image ink
    # fields for every proposal scale. No persistent/stale image cache exists.
    cache_key = (id(image), image.shape, tuple(sorted(vars(template.ink).items())),
                 tuple(plot_area), tuple(tuple(b) for b in ignore_regions))
    cached = preprocessing_cache is not None and preprocessing_cache.get('key') == cache_key
    if cached:
        membership, membership_mask, plot_edge, plot_orientation = preprocessing_cache['arrays']
    else:
        membership = ink_membership(image, template.ink)
        for ignore_region in ignore_regions:
            ix0, iy0, ix1, iy1 = ignore_region
            membership[iy0:iy1, ix0:ix1] = 0.0
        threshold = 0.45 if template.ink.achromatic else 0.24
        membership_mask = membership >= threshold
        plot_edge = cv2.Canny((membership_mask.astype(np.uint8) * 255), 40, 100) > 0
        scope = np.zeros(plot_edge.shape, dtype=bool)
        scope[y0:y1, x0:x1] = True
        plot_edge &= scope
        plot_orientation = _edge_orientation(membership)
        if preprocessing_cache is not None:
            preprocessing_cache.clear()
            preprocessing_cache.update(key=cache_key, arrays=(membership, membership_mask, plot_edge, plot_orientation))
    grid_step = max(3, int(round(template.diameter * grid_fraction)))
    grid_stride = max(1, int(round(grid_step * (1.0 - grid_overlap))))
    phase_count = max(1, int(round(grid_step / grid_stride)))
    overlap_multiplicity = float(phase_count**2)

    preprocessing_seconds = time.perf_counter() - started
    stage_started = time.perf_counter()
    grid_stats = {'backend': grid_backend, 'used_cuda': False}
    if grid_backend == 'cuda':
        from bw_gpu_grid_votes import cell_hypotheses_cuda
        hypotheses = cell_hypotheses_cuda(plot_edge, plot_orientation, template,
            plot_area, grid_step, grid_stride, batch_size=gpu_batch_size,
            diagnostics=grid_stats)
    else:
        hypotheses = _cell_hypotheses(plot_edge, plot_orientation, template,
                                     plot_area, grid_step, grid_stride)
    if grid_identity_competitor is not None:
        hypotheses = grid_identity_competitor.filter_hypotheses(
            template, hypotheses, grid_step=grid_step, grid_stride=grid_stride)
    grid_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    clustering_stats = {'backend': grid_backend, 'used_cuda': False, 'groups': []}

    def cluster_centres(group_hypotheses, multiplicity, phase):
        if grid_backend == 'cuda':
            from bw_gpu_candidate_centres import candidate_centres_cuda
            centers, votes, stats = candidate_centres_cuda(
                group_hypotheses, template, plot_area,
                overlap_multiplicity=multiplicity, batch_size=gpu_batch_size)
        else:
            centers, votes = _candidate_centres(group_hypotheses, template, plot_area,
                                               overlap_multiplicity=multiplicity)
            stats = {'backend': 'cpu', 'used_cuda': False}
        clustering_stats['groups'].append({'phase': phase, **stats})
        clustering_stats['used_cuda'] |= bool(stats.get('used_cuda'))
        return centers, votes

    candidates, vote_map = cluster_centres(hypotheses, overlap_multiplicity,
                                           'overlap' if grid_overlap > 0. else 'exclusive')
    candidate_groups = [(candidates, grid_stride, hypotheses)]
    if grid_overlap > 0.0:
        # Candidate clustering is nonlinear: extra overlapping-window votes can
        # move a centre far enough that a valid original-grid seed disappears.
        # Preserve the exact exclusive-grid seeds and add the overlapping-grid
        # seeds to them, matching the intended "add windows" semantics.
        base_hypotheses = [
            hypothesis
            for hypothesis in hypotheses
            if (hypothesis.cell_left - x0) % grid_step == 0
            and (hypothesis.cell_top - y0) % grid_step == 0
        ]
        base_candidates, _ = cluster_centres(base_hypotheses, 1.0, 'exclusive')
        candidate_groups.insert(0, (base_candidates, grid_step, base_hypotheses))

    clustering_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    candidate_verifier = None
    geometry = None
    rough_results = None
    candidate_stats = {'backend': 'cpu', 'used_cuda': False, 'preprocessing_cache_hit': cached}
    if grid_backend == 'cuda' and occlusion_mask is None and any(group[0] for group in candidate_groups):
        from bw_gpu_candidate_verifier import GpuCandidateVerifier
        candidate_groups = [(centers, stride, _IndexedHypotheses(evidence))
                            for centers, stride, evidence in candidate_groups]
        seed_plan = [(x, y, stride, evidence) for centers, stride, evidence in candidate_groups
                     for x, y, _, _ in centers]
        candidate_verifier = GpuCandidateVerifier(template, membership, plot_edge, gpu_batch_size)
        geometry = _candidate_geometry(template)
        rough_results = iter(_verify_candidates_many([(s[0], s[1]) for s in seed_plan],
            template, membership, plot_edge, plot_orientation, plot_area, grid_step,
            [(s[2], s[3]) for s in seed_plan], batch_size=gpu_batch_size,
            verifier=candidate_verifier, geometry=geometry))

    verified = []
    pending_refinement = []
    cpu_aligned = []
    refinement_stats = {'backend': refinement_backend, 'seed_count': 0, 'used_cuda': False}
    if occlusion_mask is not None:
        candidate_stats.update(visibility_policy='bw_three_state_visibility_v1', cpu_reason='other_colour_visibility')
        refinement_stats.update(requested_backend=requested_refinement, cpu_reason='other_colour_visibility')
    for group_candidates, group_stride, group_hypotheses in candidate_groups:
        for x, y, _, _ in group_candidates:
            rough_detection = next(rough_results) if rough_results is not None else _verify_candidate(
                x,
                y,
                template,
                membership,
                plot_edge,
                plot_orientation,
                plot_area,
                grid_step,
                group_stride,
                group_hypotheses,
                **visibility_options,
            )
            if rough_detection is None:
                continue
            refinement_stats['seed_count'] += 1
            if refinement_backend == 'cuda':
                # Keep every original seed and its own phase evidence. Only
                # the pure alignment calculations are batched/deduplicated.
                pending_refinement.append((x, y, group_stride, group_hypotheses))
                continue
            refined_x, refined_y = _refine_candidate_center(
                x,
                y,
                template,
                membership,
                plot_edge,
                plot_area,
                **visibility_options,
            )
            if candidate_verifier is not None:
                # Explicit grid=CUDA/refinement=CPU still batches the second
                # verification pass; only center alignment uses the CPU.
                cpu_aligned.append((refined_x, refined_y, group_stride, group_hypotheses))
                continue
            detection = _verify_candidate(
                refined_x,
                refined_y,
                template,
                membership,
                plot_edge,
                plot_orientation,
                plot_area,
                grid_step,
                group_stride,
                group_hypotheses,
                **visibility_options,
            )
            if detection is not None:
                verified.append(detection)
    if cpu_aligned:
        detections = _verify_candidates_many([(s[0], s[1]) for s in cpu_aligned],
            template, membership, plot_edge, plot_orientation, plot_area, grid_step,
            [(s[2], s[3]) for s in cpu_aligned], batch_size=gpu_batch_size,
            verifier=candidate_verifier, geometry=geometry)
        verified.extend(d for d in detections if d is not None)
    if pending_refinement:
        # Import CUDA dependencies only on the explicitly requested GPU path.
        from bw_gpu_refinement import GpuCenterRefiner
        refiner = GpuCenterRefiner(template, membership, plot_edge, plot_area,
                                   batch_size=gpu_batch_size)
        aligned = refiner.refine_many([(x, y) for x, y, _, _ in pending_refinement])
        refinement_stats.update(refiner.stats)
        refinement_stats['used_cuda'] = True
        if candidate_verifier is not None:
            detections = _verify_candidates_many(aligned, template, membership, plot_edge,
                plot_orientation, plot_area, grid_step, [(s[2], s[3]) for s in pending_refinement],
                batch_size=gpu_batch_size, verifier=candidate_verifier, geometry=geometry)
            verified.extend(d for d in detections if d is not None)
        else:
            for (x, y), (_, _, group_stride, group_hypotheses) in zip(aligned, pending_refinement, strict=True):
                detection = _verify_candidate(x, y, template, membership, plot_edge,
                    plot_orientation, plot_area, grid_step, group_stride, group_hypotheses)
                if detection is not None:
                    verified.append(detection)
    if candidate_verifier is not None:
        candidate_stats.update(candidate_verifier.stats)
        candidate_stats['spatial_index_queries'] = sum(h.queries for _, _, h in candidate_groups)
        candidate_stats['spatial_index_examined'] = sum(h.examined for _, _, h in candidate_groups)
    candidate_refinement_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if defer_nms:
        # Distinct nearby centres must survive until full-window verification.
        # Only collapse identical refined seeds emitted by several grid phases.
        # A high first-stage score is not evidence that a neighbour's full
        # marker window will pass, so it must not delete that neighbour yet.
        by_centre = {}
        for detection in verified:
            key = (float(detection.x), float(detection.y))
            if key not in by_centre or detection.score > by_centre[key].score:
                by_centre[key] = detection
        detections = sorted(by_centre.values(), key=lambda d: d.score, reverse=True)
    else:
        detections = _nms_detections(verified, max(3.0, 0.88 * template.diameter))
    if grid_identity_competitor is not None:
        detections = [d for d in detections if
            grid_identity_competitor.candidate_decision(template,d.x,d.y)]
    selection_seconds = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    # Production v46 does not consume baseline detections. Keep the standalone
    # experiment API's default, but never spend full-image matching/NMS time
    # when its caller explicitly does not need the comparison result.
    baseline = exact_template_baseline(membership, template, plot_area) if compute_baseline else []
    baseline_seconds = time.perf_counter() - stage_started if compute_baseline else 0.
    performance = dict(preprocessing_seconds=preprocessing_seconds,
        grid_seconds=grid_seconds, clustering_seconds=clustering_seconds,
        candidate_refinement_seconds=candidate_refinement_seconds,
        selection_seconds=selection_seconds, baseline_seconds=baseline_seconds,
        baseline_enabled=compute_baseline, total_seconds=time.perf_counter()-started,
        timing_scope='Sequential wall-time stages; nested GPU diagnostics must not be added again')
    return TemplateResult(
        template=template,
        grid_step=grid_step,
        grid_stride=grid_stride,
        membership=membership_mask,
        plot_edge=plot_edge,
        hypotheses=hypotheses,
        vote_map=vote_map,
        detections=detections,
        baseline=baseline,
        refinement_diagnostics=refinement_stats,
        grid_diagnostics=grid_stats,
        candidate_diagnostics=candidate_stats,
        clustering_diagnostics=clustering_stats,
        performance_diagnostics=performance,
    )


def scale_swatch_template(template: SwatchTemplate, scale: float) -> SwatchTemplate:
    """Scale the observed glyph about its centre for grid proposal generation.

    Scaling only inside the final verifier cannot recover a marker which the
    fixed-size grid never proposed. Geometry, gray evidence, and uncertainty
    are transformed together; no named/synthetic marker shape is substituted.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Template scale must be positive and finite')
    if abs(scale - 1.) < 1e-9:
        return template
    source_h, source_w = template.mask.shape
    h, w = [2 * int(math.ceil((length - 1) * scale / 2)) + 1
            for length in (source_h, source_w)]
    matrix = np.array([[scale, 0., (w-1)/2-scale*(source_w-1)/2],
                       [0., scale, (h-1)/2-scale*(source_h-1)/2]], np.float32)

    def transform(array, binary=False, fill=0.):
        result = cv2.warpAffine(np.asarray(array, np.float32), matrix, (w,h),
                    flags=cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        return result >= .5 if binary else result

    mask = transform(template.mask, True)
    soft = transform(template.soft)
    edge = cv2.Canny(mask.astype(np.uint8)*255,40,100)>0
    weight = None if template.required_weight is None else transform(template.required_weight)
    return replace(template, diameter=template.diameter*scale,
        raw_soft=transform(template.raw_soft), raw_mask=transform(template.raw_mask, True),
        line_nuisance=transform(template.line_nuisance, True),
        valid_weight=transform(template.valid_weight, fill=1.),
        soft=soft, mask=mask, edge=edge, orientation=_edge_orientation(soft),
        required_weight=weight, proposal_scale=template.proposal_scale*scale,
        observed_source_soft=None if template.observed_source_soft is None else transform(template.observed_source_soft),
        source_gray=None if template.source_gray is None else transform(template.source_gray, fill=template.ink.paper_gray))


def detect_partial_swatches(
    image: np.ndarray,
    plot_area: Box,
    swatches: Sequence[tuple[str, Box]],
    grid_fraction: float = 0.5,
    ignore_regions: Sequence[Box] = (),
    grid_overlap: float = 0.0,
) -> list[TemplateResult]:
    templates = [
        extract_swatch_template(image, swatch_box, name)
        for name, swatch_box in swatches
    ]
    results = [
        detect_template(
            image,
            template,
            plot_area,
            grid_fraction=grid_fraction,
            ignore_regions=ignore_regions,
            grid_overlap=grid_overlap,
        )
        for template in templates
    ]
    _joint_template_competition(results)
    return results


def _joint_template_competition(results: Sequence[TemplateResult]) -> None:
    """Remove cross-template duplicates while retaining true colour overlaps.

    Achromatic marker types must compete because the same black curve pixels
    feed every template.  Distinct chromatic inks remain independent, allowing
    differently coloured markers at the same position to coexist.
    """
    indexed = [
        (result_index, detection)
        for result_index, result in enumerate(results)
        for detection in result.detections
    ]
    accepted: list[tuple[int, Detection]] = []

    def minimum_score(result: TemplateResult) -> float:
        diameter = result.template.diameter
        if not result.template.ink.achromatic:
            return 0.58 if diameter < 9 else 0.62
        if diameter < 9:
            return 0.72
        if diameter < 20:
            return 0.68
        return 0.66

    indexed = [
        item
        for item in indexed
        if item[1].score >= minimum_score(results[item[0]])
    ]
    indexed.sort(key=lambda item: item[1].score, reverse=True)
    for result_index, detection in indexed:
        template = results[result_index].template
        duplicate = False
        for kept_index, kept in accepted:
            kept_template = results[kept_index].template
            # Nearby points from the two B&W series can be only about one
            # marker diameter apart.  A full-diameter duplicate radius merged
            # legitimate adjacent open/filled markers after centre refinement.
            radius = 0.78 * min(template.diameter, kept_template.diameter)
            if (detection.x - kept.x) ** 2 + (detection.y - kept.y) ** 2 > radius**2:
                continue
            both_distinct_colours = (
                not template.ink.achromatic
                and not kept_template.ink.achromatic
                and np.linalg.norm(
                    np.asarray(template.ink.core_bgr, dtype=float)
                    - np.asarray(kept_template.ink.core_bgr, dtype=float)
                )
                >= 30.0
            )
            if not both_distinct_colours:
                duplicate = True
                break
        if not duplicate:
            accepted.append((result_index, detection))

    accepted_ids = {(index, id(detection)) for index, detection in accepted}
    for result_index, result in enumerate(results):
        result.detections = [
            detection
            for detection in result.detections
            if (result_index, id(detection)) in accepted_ids
        ]


def _serializable_result(result: TemplateResult) -> dict:
    template = result.template
    return {
        "template": {
            "name": template.name,
            "key": template.key,
            "swatch_id": template.swatch_id,
            "shape_hint": template.shape_hint or template.name,
            "swatch_box": list(template.swatch_box),
            "marker_center": list(template.marker_center),
            "diameter": template.diameter,
            "marker_kind": template.marker_kind,
            "interior_fill": template.interior_fill,
            "ink": asdict(template.ink),
        },
        "grid_step": result.grid_step,
        "grid_stride": result.grid_stride,
        "hypothesis_count": len(result.hypotheses),
        "baseline": [asdict(detection) for detection in result.baseline],
        "detections": [asdict(detection) for detection in result.detections],
    }


def save_json(results: Sequence[TemplateResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_serializable_result(result) for result in results], indent=2),
        encoding="utf-8",
    )


def _parse_swatch_argument(text: str) -> tuple[str, Box]:
    try:
        name, box_text = text.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("swatch must be NAME:x0,y0,x1,y1") from error
    return name, parse_box(box_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot-area", required=True, type=parse_box)
    parser.add_argument(
        "--swatch",
        required=True,
        action="append",
        type=_parse_swatch_argument,
        help="repeat NAME:x0,y0,x1,y1 for each legend swatch",
    )
    parser.add_argument("--grid-fraction", type=float, default=0.5)
    parser.add_argument(
        "--grid-overlap",
        type=float,
        default=0.0,
        help="fractional overlap between neighbouring grid windows (0 <= value < 1)",
    )
    parser.add_argument(
        "--ignore-region",
        action="append",
        type=parse_box,
        default=[],
        help="optional x0,y0,x1,y1 region to exclude (legend text/title/etc.)",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    results = detect_partial_swatches(
        image,
        args.plot_area,
        args.swatch,
        grid_fraction=args.grid_fraction,
        grid_overlap=args.grid_overlap,
        ignore_regions=args.ignore_region,
    )
    save_json(results, args.output)
    for result in results:
        print(
            f"{result.template.name}: diameter={result.template.diameter:.1f}, "
            f"grid={result.grid_step}, stride={result.grid_stride}, "
            f"hypotheses={len(result.hypotheses)}, "
            f"exact={len(result.baseline)}, partial={len(result.detections)}"
        )


if __name__ == "__main__":
    main()

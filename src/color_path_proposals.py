"""Image-only, diameter-scaled colour-mask marker *proposals*.

This is an isolated reworking of the path/band-density and independent-blob
ideas in ``run_A4_auto_v45.py``.  It deliberately does not import that script:
the v45 script has image-wide globals and execution side effects.  This module
has no image identity, pixel truth, axis values, or antibody-specific rules.

Unlike v45, endpoints are not unconditionally markers, spacing scales with the
legend diameter, and hollow symbols can contribute without surviving erosion.
Proposals are deliberately recall-oriented; the caller must verify complete
windows against its colour/shape evidence before calling them detections.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


def _prepare(mask: np.ndarray, diameter: float) -> tuple[np.ndarray, float]:
    a = np.asarray(mask)
    if a.ndim != 2:
        raise ValueError("mask must be a two-dimensional membership image")
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("diameter must be positive and finite")
    p = np.nan_to_num(a.astype(np.float32), nan=0., posinf=1., neginf=0.)
    if a.dtype == np.uint8 and p.size and p.max() > 1:
        p /= 255.
    return np.clip(p, 0., 1.), max(3., float(diameter))


def _disk(radius: float) -> np.ndarray:
    r = int(math.ceil(radius))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    k = (xx*xx + yy*yy <= radius*radius).astype(np.float32)
    return k / max(float(k.sum()), 1.)


def _smooth(a: np.ndarray, radius: float) -> np.ndarray:
    return cv2.filter2D(a, -1, _disk(radius), borderType=cv2.BORDER_CONSTANT)


def _line_kernel(length: float, angle: float, width: int) -> np.ndarray:
    n = int(math.ceil(length)) | 1
    c, r = n//2, (n-1)/2
    dx, dy = r*math.cos(angle), r*math.sin(angle)
    k = np.zeros((n, n), np.uint8)
    cv2.line(k, (round(c-dx), round(c-dy)),
             (round(c+dx), round(c+dy)), 1, width)
    return k


def _measure(p: np.ndarray, diameter: float) -> dict[str, np.ndarray]:
    b = (p >= .35).astype(np.uint8)
    distance = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3,3), np.uint8)))
    if ridge.any():
        width = float(np.clip(2*np.quantile(distance[ridge], .25)-.5, 1., .45*diameter))
    else:
        width = 1.
    # Long oriented openings explain connectors and error-bar stems.  They are
    # used only for proposal features; the caller receives the untouched mask.
    line = np.zeros_like(b)
    for angle in np.linspace(0., np.pi, 12, endpoint=False):
        k = _line_kernel(1.8*diameter, float(angle), max(1, round(.65*width)))
        opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, k,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        line |= opened
    residual = p*(1-line.astype(np.float32))
    # Average over a marker-sized disk: a ring votes for its centre even when
    # there is no coloured centre pixel.  No fill/solid-marker assumption.
    density = _smooth(p, .58*diameter)
    extra = _smooth(residual, .58*diameter)
    score = (.7*extra + .3*density).astype(np.float32)
    stem = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                          np.ones((int(math.ceil(1.7*diameter))|1, 1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return {"density": density, "path_score": score,
            "path": np.zeros_like(p), "thickness": 2*distance,
            "stem": stem.astype(np.float32), "line_residual": residual,
            "residual_density": extra, "line": line.astype(np.float32),
            "line_width": np.full_like(p, width)}


def _paths(p: np.ndarray, maps: dict, diameter: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Column representatives per component, with symmetric support weighting.

    v45's longest run/greedy walk can jump to a vertical error bar. Horizontal
    support gives a long thin stem little weight away from its curve crossing.
    Independent connected components restart tracking after a disconnected
    segment. This is a column-path approximation, not the full A4 walk port.
    """
    b = (p >= .35).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    horizontal = cv2.boxFilter(p, -1, (max(3, round(.8*diameter))|1, 1),
                               normalize=True, borderType=cv2.BORDER_CONSTANT)
    paths = []
    for ci in range(1, n):
        x0,y0,w,h,area = stats[ci]
        if area < max(3, .2*diameter) or w < 2:
            continue
        crop = labels[y0:y0+h, x0:x0+w] == ci
        weights = (horizontal[y0:y0+h, x0:x0+w]**2+.01)*crop
        masses = weights.sum(axis=0)
        use = masses > 0
        xs = np.arange(x0, x0+w)[use]
        ys = (weights*np.arange(y0,y0+h)[:,None]).sum(axis=0)[use]/masses[use]
        if len(ys) >= 3:
            ys = gaussian_filter1d(median_filter(ys, size=3, mode="nearest"), .65)
        yi = np.clip(np.rint(ys).astype(int), 0, p.shape[0]-1)
        maps["path"][yi, xs] = 1.
        paths.append((xs, ys))
    return paths


def _proposal(x: float, y: float, score: float, maps: dict, branch: str,
              source: str = "path") -> dict:
    yi = int(np.clip(round(y), 0, maps["density"].shape[0]-1))
    xi = int(np.clip(round(x), 0, maps["density"].shape[1]-1))
    return {"x": float(x), "y": float(y), "score": float(score),
            "path_density": float(maps["density"][yi,xi]),
            "line_width": float(maps["line_width"][yi,xi]),
            "source": source, "branch": branch,
            "line_residual_density": float(maps["residual_density"][yi,xi])}


def _nms(proposals: list[dict], distance: float) -> list[dict]:
    out = []
    for p in sorted(proposals, key=lambda v: (-v["score"], v["y"], v["x"])):
        if all((p["x"]-q["x"])**2+(p["y"]-q["y"])**2 >= distance**2 for q in out):
            out.append(p)
    return sorted(out, key=lambda v: (v["x"], v["y"]))


def _density_peaks(maps: dict, diameter: float, baseline: bool = False) -> list[dict]:
    field = maps["density"] if baseline else maps["path_score"]
    size = max(3, round(.55*diameter))|1
    maxima = cv2.dilate(field, np.ones((size,size), np.uint8))
    threshold = .10 if baseline else .045
    peak = (field >= maxima-1.e-6) & (field >= threshold)
    if not baseline:
        peak &= maps["residual_density"] >= .02
    n, lab, stats, centres = cv2.connectedComponentsWithStats(peak.astype(np.uint8), 8)
    out = []
    for i in range(1,n):
        # Connected plateaus produce one centre, not one proposal per pixel.
        x,y = centres[i]
        x0,y0,w,h,_ = stats[i]
        region = lab[y0:y0+h,x0:x0+w] == i
        score = float(np.max(field[y0:y0+h,x0:x0+w][region]))
        out.append(_proposal(x,y,score,maps,
                             "density_baseline" if baseline else "independent_density",
                             "density" if baseline else "path"))
    return out


def propose_path_markers(mask: np.ndarray, diameter: float) -> tuple[list[dict], dict]:
    """Propose crop-local marker centres from one colour membership image.

    ``mask`` is boolean or soft membership in [0,1]; uint8 0/255 is accepted.
    ``diameter`` is the image-only legend/marker-size estimate in crop pixels.
    Returned maps are same-sized float32 arrays. No truth positions or nominal
    data-point schedule may be supplied. Coordinate (0,0) is the crop top-left.
    """
    p,d = _prepare(mask, diameter)
    if not p.size:
        names = ("density", "path_score", "path", "thickness", "stem",
                 "line_residual", "residual_density", "line", "line_width")
        return [], {name: np.zeros_like(p) for name in names}
    maps = _measure(p,d)
    proposals = _density_peaks(maps,d)
    for xs,ys in _paths(p,maps,d):
        yi = np.clip(np.rint(ys).astype(int),0,p.shape[0]-1)
        band = maps["path_score"][yi,xs]
        # v45 multiplies two equivalent normalized counts; the new independent
        # features are full ink density and ink unexplained by long thin lines.
        peaks,_ = find_peaks(np.r_[0.,band,0.], height=.045,
                             prominence=.012, distance=max(1, round(.45*d)))
        for j in peaks-1:
            if j < 0 or j >= len(xs):
                continue
            x,y = float(xs[j]), float(ys[j])
            if maps["residual_density"][yi[j],xs[j]] < .02:
                continue
            proposals.append(_proposal(x,y,float(band[j]),maps,"path_density"))
    return _nms(proposals,.40*d), maps


def propose_density_baseline(mask: np.ndarray, diameter: float) -> tuple[list[dict], dict]:
    """Simple unverified disk-density baseline, with the identical return API.

    This deliberately lacks path and line-explanation evidence. It is not a
    faithful full-v45 benchmark, and should never be labelled as one.
    """
    p,d = _prepare(mask, diameter)
    if not p.size:
        return propose_path_markers(p,d)
    maps = _measure(p,d)
    return _nms(_density_peaks(maps,d,baseline=True),.40*d), maps

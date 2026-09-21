"""Shared chromatic identity and observed contrast for legend and plot pixels.

Black outlines carry geometry, not the identity of a coloured series. Paper is
never foreground, even when a very faint pixel has the correct hue direction.
"""
import math
import numpy as np


def reference_contrast(pixels, paper, ink):
    flat = np.asarray(pixels, np.float32).reshape(-1, 3)
    paper, ink = np.asarray(paper, np.float32), np.asarray(ink, np.float32)
    contrast = np.linalg.norm(np.maximum(paper-flat, 0.), axis=1)
    hue = flat-flat.min(axis=1, keepdims=True)
    ink_hue = ink-ink.min()
    cosine = (hue@ink_hue)/np.maximum(np.linalg.norm(hue, axis=1)*np.linalg.norm(ink_hue), 1.)
    eligible = (cosine >= math.cos(math.radians(15))) & (contrast >= 8) & (np.ptp(flat, axis=1) >= 4)
    return max(8., float(np.percentile(contrast[eligible], 90)) if eligible.any()
               else float(np.linalg.norm(paper-ink)))


def chromatic_evidence(image, paper, ink, reference):
    pixels = np.asarray(image, np.float32)
    paper, ink = np.asarray(paper, np.float32), np.asarray(ink, np.float32)
    contrast = np.linalg.norm(np.maximum(paper-pixels, 0.), axis=-1)
    strength = np.clip(contrast/max(float(reference), 8.), 0., 1.)
    hue = pixels-pixels.min(axis=-1, keepdims=True)
    ink_hue = ink-ink.min()
    cosine = (hue@ink_hue)/np.maximum(np.linalg.norm(hue, axis=-1)*np.linalg.norm(ink_hue), 1.)
    # Twenty-to-thirty degree hue differences can already be two distinct
    # legend inks (e.g. red and brown). Preserve small JPEG hue drift without
    # letting such a rival contribute partial filled-marker evidence.
    strict, weak = math.cos(math.radians(8)), math.cos(math.radians(18))
    fit = np.clip((cosine-weak)/(strict-weak), 0., 1.)
    # Neutral black is never coloured support; low-chroma JPEG halos are weak.
    fit *= np.clip(np.ptp(pixels, axis=-1)/4., 0., 1.)
    return (strength*fit).astype(np.float32), fit.astype(np.float32)

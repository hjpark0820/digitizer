"""Separate colour proposals, confirmed ink, and uncertain boundary evidence.

Uncertainty never paints positive ink. It only supplies bounded loss weights
to the CPU full-window verifier. No palette colour or marker is invented.
"""
import cv2
import numpy as np


def ownership_maps(image, fits, strengths, chromatic):
    fits = np.asarray(fits, np.float32)
    strengths = np.asarray(strengths, np.float32)
    if (fits.ndim != 3 or fits.shape[1:] != image.shape[:2] or
            strengths.shape != fits.shape or len(chromatic) != len(fits) or
            not np.isfinite(fits).all() or not np.isfinite(strengths).all() or
            np.any((fits < 0) | (fits > 1)) or
            np.any((strengths < 0) | (strengths > 1))):
        raise ValueError('Colour evidence must be finite, aligned and in [0,1]')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    winner = fits.argmax(axis=0)
    reliability = np.clip((fits - .10) / .25, 0, 1)
    neutral = (np.ptp(image.astype(np.float32), axis=2) < 18) & (gray < 190)
    # Ink existence is independent of whether a single swatch ray fits it.
    ink = np.clip((255 - gray - 4) / 24, 0, 1)
    best = fits.max(axis=0)
    # A local boundary may be uncertain; an isolated unrelated colour cannot
    # excuse a missing symbol. Two source pixels, not a whole-marker dilation.
    seed = (best >= .35) & (strengths.max(axis=0) >= .15)
    near = cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    uncertainty = ink * (1 - reliability.max(axis=0)) * near * ~neutral
    maps = []
    for i in range(len(fits)):
        owned = winner == i
        legacy = np.rint(255 - (255-gray)*owned*reliability[i]).astype(np.uint8)
        # Raw brightness is only a PROPOSAL. Full windows still see legacy
        # confirmed evidence, with uncertain ink kept in a separate channel.
        plausible = owned & (best > 1.e-6) & (ink > 0)
        if chromatic[i]:
            plausible &= ~neutral
        proposal = np.where(plausible, gray, 255).astype(np.uint8)
        maps.append(dict(proposal=proposal, confirmed=legacy,
                         uncertainty=uncertainty.astype(np.float32),
                         confidence=(owned*reliability[i]).astype(np.float32)))
    return maps

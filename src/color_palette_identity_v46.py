"""Keep hue-family routing distinct from physical ink identity.

Similar hues can encode different inks (orange/brown, light/dark blue).
Only distinguishable, nearby-hue rivals invoke RGB ink-to-paper competition.
This is negative evidence: it can remove target support, never create ink,
repaint a source pixel, or identify same-colour marker shapes.
"""
import cv2
import numpy as np

VERSION = 'palette_ink_identity_v1'


def palette_relations(bgr, colour_distance=14., hue_family=True):
    """Return same-ink and nearby-hue matrices in original palette order.

    Lab distance uses OpenCV's 8-bit encoding, not CIE Delta-E units.
    A bounded hue-family tolerance permits JPEG swatch variation; large
    saturation/value differences cannot bypass the Lab identity check.
    """
    colours = np.asarray(bgr, np.float32).reshape(-1, 3)
    if not np.isfinite(colours).all() or np.any((colours < 0) | (colours > 255)):
        raise ValueError('Palette colours must be finite BGR values in 0..255')
    if not np.isfinite(colour_distance) or colour_distance < 0:
        raise ValueError('Invalid palette colour distance')
    if not len(colours):
        return np.empty((0, 0), bool), np.empty((0, 0), bool)
    raster = np.rint(colours).astype(np.uint8)[None]
    lab = cv2.cvtColor(raster, cv2.COLOR_BGR2LAB)[0].astype(float)
    hsv = cv2.cvtColor(raster, cv2.COLOR_BGR2HSV)[0].astype(float)
    distance = np.linalg.norm(lab[:, None]-lab[None], axis=-1)
    dh = np.abs(hsv[:, None, 0]-hsv[None, :, 0])
    dh = 2*np.minimum(dh, 180-dh)
    chromatic = np.minimum(hsv[:, None, 1], hsv[None, :, 1]) >= 40
    near = chromatic & (dh <= 18)
    same = distance <= colour_distance
    if hue_family:
        same |= (chromatic & (dh <= 8) & (distance <= max(colour_distance, 24.))
                 & (np.abs(hsv[:, None, 1]-hsv[None, :, 1]) <= 40)
                 & (np.abs(hsv[:, None, 2]-hsv[None, :, 2]) <= 36))
    return same, near


def paper_mixture_evidence(image, model):
    """Return bounded ink/paper fit AND the observed mixture strength.

    Antialiased ink lies between its source ink and paper. A small overshoot
    permits JPEG noise but cannot make arbitrarily darker ink match a pale key.
    Fit alone is not evidence of ink: pure paper fits every colour ray. Keep
    alpha/contrast separate so a pale match is never painted as full ink.
    This is an encoded-RGB approximation, not an inverse rendering guarantee.
    """
    paper = np.asarray(model.get('paper_bgr', [255., 255., 255.]), np.float32)
    ink = np.asarray(model['bgr'], np.float32)
    direction = paper-ink
    delta = paper-np.asarray(image, np.float32)
    alpha = np.clip((delta@direction)/max(float(direction@direction), 1.), 0., 1.08)
    residual = np.linalg.norm(delta-alpha[..., None]*direction, axis=-1)
    sigma = float(np.clip(model.get('residual_sigma', 8.), 6., 16.))
    contrast = np.linalg.norm(delta, axis=-1)
    tolerance = sigma + .025*contrast
    return dict(fit=np.exp(-.5*(residual/tolerance)**2).astype(np.float32),
                alpha=alpha.astype(np.float32), residual=residual.astype(np.float32),
                contrast=contrast.astype(np.float32))


def _paper_ray_fit(image, model):
    """Compatibility score; legacy palette competition is unchanged."""
    return paper_mixture_evidence(image, model)['fit']


def distinguish_inks(image, models, membership, confidence, *, equivalent=None):
    """Attenuate support explained better by a distinct nearby-hue ink.

    The unchanged per-swatch hue evidence stays the proposal source. RGB
    competition is only a multiplier <=1, shared by initial grids, scale
    calibration and final windows. Same-ink shapes never compete by colour.
    Uncertain ties retain evidence rather than using palette/series order.
    """
    membership = np.asarray(membership, np.float32)
    confidence = np.asarray(confidence, np.float32)
    expected = (len(models), *np.shape(image)[:2])
    if (membership.shape != expected or confidence.shape != expected or
            not np.isfinite(membership).all() or not np.isfinite(confidence).all()):
        raise ValueError('Palette evidence must be finite and aligned to the image')
    same, near = palette_relations([m['bgr'] for m in models])
    if equivalent is not None:
        same = np.asarray(equivalent, bool)
        if same.shape != near.shape:
            raise ValueError('Palette equivalence matrix has the wrong shape')
    rivals = near & ~same
    report = dict(version=VERSION, same_ink=same.tolist(), competing_pairs=[],
                  positive_ink_added=False, unknown_or_paper_is_not_a_rival=True)
    active = np.flatnonzero(rivals.any(axis=1))
    if not len(active):
        return membership, confidence, report
    fits = {int(i): _paper_ray_fit(image, models[i]) for i in active}
    m, c = membership.copy(), confidence.copy()
    for i in active:
        gate = np.ones(expected[1:], np.float32)
        for j in np.flatnonzero(rivals[i]):
            # A rival must have independent chromatic ink, not just proximity
            # to the paper endpoint of every colour ray.
            observed = np.clip(membership[j]/.25, 0., 1.)
            better = np.clip((fits[j]-fits[i]-.10)/.40, 0., 1.)
            gate = np.minimum(gate, 1.-fits[j]*observed*better)
            if i < j:
                report['competing_pairs'].append([int(i), int(j)])
        m[i] *= gate
        c[i] *= gate
    return m, c, report

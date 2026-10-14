"""Keep hue-family routing distinct from physical ink identity.

Similar hues can encode different inks (orange/brown, light/dark blue).
Distinguishable nearby-hue and neutral/chromatic rivals invoke RGB ink-to-paper
competition. Slightly tinted JPEG gray must not become a vivid coloured ink.
This is negative evidence: it can remove target support, never create ink,
repaint a source pixel, or identify same-colour marker shapes.
"""
import cv2
import numpy as np

VERSION = 'palette_ink_identity_v2_neutral_rivals'


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


MIXING_VERSION = 'paper_ink_mixing_family_v1'
# Rendering model of a legend glyph (validated on rendered stress glyphs,
# experiments/visible_miss_fix/fix_E2B/probes/stress_sweep.py):
#   pixel = clip(encode(paper + coverage * (ink - paper)))
# coverage = antialiasing and any blur/resampling (linear, identical in every
# channel). encode() = the image codec, chosen per chart from how well it
# reproduces the SOLID key's own pixels:
#   * 'none'        lossless raster;
#   * 'sub:...'     chroma averaged over 2x2 blocks (4:2:0) and re-interpolated;
#   * 'jpeg:<q>'    baseline JPEG: 4:2:0 chroma, 8x8 DCT quantised with the IJG
#                   chroma table at quality q, libjpeg triangle upsampling,
#                   MCU grid anchored at the image origin.
# Luma keeps the coverage map (full resolution, fine quantisation); chroma is
# that map passed through the codec. A thin stroke's core therefore looks
# darker-and-desaturated (its high-frequency chroma is averaged/quantised
# away and a clipped channel cannot compensate) and its halo tinted, while
# the ink's chroma-per-darkness g = (C_ink - C_paper) / (Y_paper - Y_ink) is
# unchanged. g and the luma reach are what identify the ink.
# * HALO: the window around the glyph that is compared (codec chroma spreads
#   <= 2-3 px from a stroke after triangle upsampling).
# * rendering error: the thin glyph is re-rendered with the solid ink at the
#   PHASES of the codec grid (chroma pairs, JPEG blocks) and re-measured; the
#   mean error is the estimator's BIAS for this glyph and is removed. The
#   remaining error of TRUE twins was measured on rendered stress glyphs
#   (probes/calibrate_spread.py: matplotlib keys, 6 shapes, 0.5-2 px strokes,
#   lossless / blur / resampling / JPEG 4:2:0 q 50-98): P95 per codec class of
#   the radial and tangential chroma-per-darkness offset (fraction of |g_K|)
#   and of the reach error of fully covered strokes. That envelope (x SCALE)
#   is the margin around a measurement.
MIXING_HALO_PX = 3
MIXING_PHASES = ((0, 0), (1, 0), (0, 1), (1, 1), (6, 3), (11, 9))
MIXING_ENVELOPE = {'none': (.01, .005, .025), 'sub': (.10, .04, .11), 'q95': (.10, .06, .08),
                   'q88': (.16, .07, .06), 'q75': (.17, .085, .065), 'q0': (.25, .15, .10)}
MIXING_NEUTRAL_ENVELOPE = .02
MIXING_ENVELOPE_SCALE = 1.
MIXING_NEUTRAL_G = .05
MIXING_THIN_PX = 2.
MIXING_SOLID_PX = 2.5
_RGB2YCC = np.asarray([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]], np.float32)
_IJG_CHROMA = np.asarray([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99]]+[[99]*8]*4, np.float64)
_DCT8 = np.asarray([[(np.sqrt(.125) if k == 0 else .5)*np.cos((2*n+1)*k*np.pi/16) for n in range(8)]
                    for k in range(8)], np.float64)
_JPEG_QUALITIES = (98, 95, 92, 90, 88, 85, 80, 75, 70, 60, 50, 40, 30)
_CHROMA_MODES = (['none']+[f'sub:{k}{dy}{dx}' for k in ('box', 'fancy') for dy in (0, 1) for dx in (0, 1)]
                 + [f'jpeg:{q}' for q in _JPEG_QUALITIES])


def _ycc(bgr):
    return np.asarray(bgr, np.float64)[..., ::-1]@_RGB2YCC.T.astype(np.float64)


def _bgr(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return np.stack([y+1.772*cb, y-.344136*cb-.714136*cr, y+1.402*cr], -1)


def _triangle_upsample(s):
    """libjpeg h2v2 'fancy' upsampling: separable 3/4, 1/4 interpolation."""
    def tri(a, axis):
        prev = np.concatenate([np.take(a, [0], axis), np.take(a, np.arange(a.shape[axis]-1), axis)], axis)
        nxt = np.concatenate([np.take(a, np.arange(1, a.shape[axis]), axis), np.take(a, [-1], axis)], axis)
        even, odd = .75*a+.25*prev, .75*a+.25*nxt
        return np.stack([even, odd], axis+1).reshape(*a.shape[:axis], 2*a.shape[axis], *a.shape[axis+1:])
    return tri(tri(s, 0), 1)


def _ijg_table(quality):
    scale = 5000./quality if quality < 50 else 200.-2*quality
    return np.clip(np.floor((_IJG_CHROMA*scale+50)/100), 1, 255)


def _codec_chroma(plane, origin, mode):
    """Chroma plane (paper-relative, centred) after the codec ``mode``."""
    if mode == 'none':
        return plane
    h, w = plane.shape
    if mode.startswith('sub:'):
        kind, dy, dx = mode[4:-2], int(mode[-2]), int(mode[-1])
        top, left, cell = (origin[1]+dy) % 2, (origin[0]+dx) % 2, 2
    else:
        top, left, cell = origin[1] % 16, origin[0] % 16, 16
    bottom, right = (-(top+h)) % cell, (-(left+w)) % cell
    p = np.pad(plane, ((top, bottom), (left, right)), mode='edge')
    s = p.reshape(p.shape[0]//2, 2, p.shape[1]//2, 2).mean(axis=(1, 3))
    if mode.startswith('sub:') and kind == 'box':
        up = np.repeat(np.repeat(s, 2, 0), 2, 1)
    else:
        if mode.startswith('jpeg:'):
            table = _ijg_table(int(mode[5:]))
            b = s.reshape(s.shape[0]//8, 8, s.shape[1]//8, 8).transpose(0, 2, 1, 3)
            f = np.einsum('ij,abjk,lk->abil', _DCT8, b, _DCT8)
            f = np.round(f/table)*table
            b = np.einsum('ji,abjk,kl->abil', _DCT8, f, _DCT8)
            s = b.transpose(0, 2, 1, 3).reshape(s.shape)
        up = _triangle_upsample(s)
    return up[top:top+h, left:left+w]


def key_ink_profile(image, box, model, halo=MIXING_HALO_PX):
    """Pixels of one legend key: its glyph, the glyph's halo window and thickness.

    The glyph is the half-contrast component holding most ink inside the key
    box, plus components lying entirely inside the box (dashes, detached
    arms). Label text is a separate (foreign) component: window pixels nearer
    to it than to the glyph are excluded, and the codec model keeps its
    observed chroma instead of assigning it this key's ink. The patch covers
    whole 16-px MCUs of the image grid so a JPEG model sees complete blocks.
    """
    h, w = np.shape(image)[:2]
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1-x0 < 3 or y1-y0 < 3:
        return None
    pad = int(halo)+2
    X0, Y0 = 16*(max(0, x0-pad)//16), 16*(max(0, y0-pad)//16)
    X1, Y1 = min(w, -16*(-(x1+pad)//16)), min(h, -16*(-(y1+pad)//16))
    patch = np.asarray(image, np.float64)[Y0:Y1, X0:X1]
    paper = np.asarray(model.get('paper_bgr', [255., 255., 255.]), np.float64)
    contrast = np.linalg.norm(paper-patch, axis=-1)
    level = float(np.linalg.norm(paper-np.asarray(model['bgr'], np.float64)))
    strong = contrast >= max(12., .5*level)
    inner = np.zeros(strong.shape, bool)
    inner[y0-Y0:y1-Y0, x0-X0:x1-X0] = True
    n, labels = cv2.connectedComponents(strong.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    inside = np.bincount(labels[inner & strong], minlength=n)
    inside[0] = 0
    if inside.max() <= 0:
        return None
    total = np.bincount(labels.ravel(), minlength=n)
    keep = (inside == total) & (total > 0)
    keep[int(inside.argmax())] = True
    keep[0] = False
    glyph = keep[labels]
    foreign = strong & ~glyph
    d_glyph = cv2.distanceTransform((~glyph).astype(np.uint8), cv2.DIST_L2, 3)
    d_foreign = (cv2.distanceTransform((~foreign).astype(np.uint8), cv2.DIST_L2, 3) if foreign.any()
                 else np.full(glyph.shape, np.inf, np.float32))
    window = (d_glyph <= halo) & (d_glyph < d_foreign)
    thickness = float(cv2.distanceTransform(np.pad(glyph.astype(np.uint8), 1), cv2.DIST_L2, 3).max())
    ycc = _ycc(patch)
    return dict(paper=paper.astype(np.float32), patch=patch, ycc=ycc, origin=(X0, Y0), glyph=glyph, window=window,
                foreign=(d_foreign < d_glyph) & (d_foreign <= 2.), thickness=thickness,
                glyph_size=int(glyph.sum()), window_pixels=int(window.sum()))


def _simulate(profile, paper_ycc, g, mode):
    """Encoded BGR of the profile's patch if its glyph's ink had chroma-per-darkness g."""
    ycc = profile['ycc']
    dark = np.clip(paper_ycc[0]-ycc[..., 0], 0., None)
    sim = np.empty_like(ycc)
    sim[..., 0] = ycc[..., 0]
    for c in (1, 2):
        pre = g[c-1]*dark
        pre[profile['foreign']] = ycc[..., c][profile['foreign']]-paper_ycc[c]
        sim[..., c] = paper_ycc[c]+_codec_chroma(pre, profile['origin'], mode)
    return np.clip(_bgr(sim), 0., 255.)


def _residual(profile, paper_ycc, g, mode):
    err = (_simulate(profile, paper_ycc, g, mode)-profile['patch'])[profile['window']]
    return float(np.sqrt(np.mean(err**2))) if len(err) else np.inf


def _fit_chroma(profile, paper_ycc, mode, seeds):
    """Chroma-per-darkness of a glyph under the codec (clipping modelled)."""
    from scipy.optimize import minimize
    best, cost = None, np.inf
    for seed in seeds:
        seed = np.asarray(seed, np.float64)
        scale = max(.05, .25*float(np.linalg.norm(seed)))
        res = minimize(lambda g: _residual(profile, paper_ycc, g, mode), seed, method='Nelder-Mead',
                       options=dict(initial_simplex=[seed, seed+[scale, 0], seed+[0, scale]],
                                    xatol=1.e-3, fatol=1.e-4, maxiter=200))
        if res.fun < cost:
            best, cost = np.asarray(res.x, np.float64), float(res.fun)
    return best, cost


def _palette_same_to(reference, colours, colour_distance, hue_family):
    """palette_relations(...)[0, 1:] for many colours against one reference."""
    colours = np.clip(np.asarray(colours, np.float64).reshape(-1, 3), 0, 255)
    raster = np.rint(np.vstack([np.asarray(reference, np.float64)[None], colours])).astype(np.uint8)[None]
    lab = cv2.cvtColor(raster, cv2.COLOR_BGR2LAB)[0].astype(float)
    hsv = cv2.cvtColor(raster, cv2.COLOR_BGR2HSV)[0].astype(float)
    distance = np.linalg.norm(lab[1:]-lab[0], axis=-1)
    dh = np.abs(hsv[1:, 0]-hsv[0, 0])
    dh = 2*np.minimum(dh, 180-dh)
    chromatic = np.minimum(hsv[1:, 1], hsv[0, 1]) >= 40
    same = distance <= colour_distance
    if hue_family:
        same |= (chromatic & (dh <= 8) & (distance <= max(colour_distance, 24.))
                 & (np.abs(hsv[1:, 1]-hsv[0, 1]) <= 40) & (np.abs(hsv[1:, 2]-hsv[0, 2]) <= 36))
    return same


def codec_of_solid_key(solid_model, solid_profile, mode=None):
    """Codec that best reproduces a SOLID key's pixels from its own ink and luma.

    A neutral ink has no chroma to reveal the codec: unless the caller passes
    the image's codec (measured on its chromatic keys), the codec stays
    'undetermined' (simulated as lossless, widest twin envelope).
    """
    paper = np.asarray(solid_model.get('paper_bgr', [255., 255., 255.]), np.float64)
    ink = np.asarray(solid_model['bgr'], np.float64)
    paper_ycc, ink_ycc = _ycc(paper), _ycc(ink)
    span = float(paper_ycc[0]-ink_ycc[0])
    if solid_profile is None or span < 12.:
        return None
    g = (ink_ycc[1:]-paper_ycc[1:])/span
    fits = {m: _residual(solid_profile, paper_ycc, g, m) for m in _CHROMA_MODES}
    undetermined = False
    if mode is None:
        if float(np.linalg.norm(g)) < MIXING_NEUTRAL_G:
            mode, undetermined = 'none', True
        else:
            mode = min(fits, key=fits.get)
    return dict(mode=mode, undetermined=undetermined, rms=fits[mode], lossless_rms=fits['none'], fits=fits,
                g=g, span=span, paper_ycc=paper_ycc, ink_ycc=ink_ycc)


def image_codec(codecs):
    """One codec per image: the mode that best reproduces all CHROMATIC solid keys."""
    chromatic = [c for c in codecs if c is not None and not c['undetermined']]
    if not chromatic:
        return None
    total = {m: sum(c['fits'][m]**2 for c in chromatic) for m in _CHROMA_MODES}
    return min(total, key=total.get)


def _twin_render(profile, codec, shift):
    """The thin glyph re-rendered with the SOLID ink, encoded at another grid phase.

    Coverage = the thin glyph's own luma coverage relative to the solid ink
    (clipped to [0, 1]); ink = the solid key's ink; codec = the detected one.
    This is H_same made explicit for this very glyph.
    """
    P, span, mode = codec['paper_ycc'], codec['span'], codec['mode']
    cov = np.clip((P[0]-profile['ycc'][..., 0])/span, 0., 1.)
    cov[profile['foreign']] = 0.
    h, w = cov.shape
    dy, dx = shift
    H, W = -16*(-(h+dy)//16), -16*(-(w+dx)//16)
    canvas = np.zeros((H, W))
    canvas[dy:dy+h, dx:dx+w] = cov
    paper = _bgr(P)
    ink = _bgr(np.r_[P[0]-span, P[1:]+codec['g']*span])
    img = np.clip(np.rint(paper+canvas[..., None]*(ink-paper)), 0, 255).astype(np.uint8)
    if mode.startswith('jpeg:'):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(mode[5:])])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif mode.startswith('sub:'):
        ycc = _ycc(img)
        for c in (1, 2):
            ycc[..., c] = _codec_chroma(ycc[..., c], (0, 0), mode)
        img = np.clip(np.rint(_bgr(ycc)), 0, 255).astype(np.uint8)

    def place(mask):
        out = np.zeros((H, W), bool)
        out[dy:dy+h, dx:dx+w] = mask
        return out
    patch = img.astype(np.float64)
    synthetic = dict(patch=patch, ycc=_ycc(patch), origin=(0, 0), glyph=place(profile['glyph']),
                     window=place(profile['window']), foreign=np.zeros((H, W), bool))
    return synthetic, float(np.percentile(cov[profile['glyph']], 95))


def _envelope(mode, chromatic, undetermined=False):
    """(radial, tangential, reach) twin envelope of a codec; neutral chroma is absolute.

    An undetermined codec takes the widest envelope of the table.
    """
    if undetermined:
        radial, tangential = max(v[0] for v in MIXING_ENVELOPE.values()), max(v[1] for v in MIXING_ENVELOPE.values())
        reach = max(v[2] for v in MIXING_ENVELOPE.values())
    else:
        if mode == 'none' or mode.startswith('sub:'):
            key = 'none' if mode == 'none' else 'sub'
        else:
            q = int(mode[5:])
            key = 'q95' if q >= 95 else 'q88' if q >= 88 else 'q75' if q >= 75 else 'q0'
        radial, tangential, reach = MIXING_ENVELOPE[key]
    if not chromatic:
        radial = tangential = MIXING_NEUTRAL_ENVELOPE
    return radial, tangential, reach


def _frame(g_ref):
    """Radial/tangential frame of a chromatic ink's g, absolute Cb/Cr axes for a neutral one."""
    norm = float(np.linalg.norm(g_ref))
    if norm < MIXING_NEUTRAL_G:
        return np.eye(2), 1., False
    u = np.asarray(g_ref, np.float64)/norm
    return np.vstack([u, [-u[1], u[0]]]), norm, True


def _measure(profile, codec, seeds):
    g, cost = _fit_chroma(profile, codec['paper_ycc'], codec['mode'], seeds)
    dark = codec['paper_ycc'][0]-profile['ycc'][..., 0]
    return g, cost, float(np.percentile(dark[profile['glyph']]/codec['span'], 95))


def mixing_measurement(solid_model, solid_profile, thin_profile, codec=None):
    """Codec-corrected chroma-per-darkness and reach of a thin key relative to a solid ink.

    * codec: chosen by reproducing the SOLID key (same image, same codec);
    * measurement: the thin key's chroma-per-darkness g (fitted under that
      codec, clipping modelled) and its luma reach (P95 coverage);
    * bias of THIS glyph: the glyph is re-rendered with the solid ink at the
      codec grid phases and measured the same way (parametric bootstrap); the
      mean error is removed. g is expressed relative to the solid ink's g in
      its radial / tangential frame (absolute Cb/Cr for a neutral ink).
    """
    if solid_profile is None or thin_profile is None:
        return dict(reason='unmeasured_key')
    if np.linalg.norm(np.asarray(solid_model.get('paper_bgr', [255.]*3), np.float64)
                      - np.asarray(thin_profile['paper'], np.float64)) > 8.:
        return dict(reason='different_paper')
    codec = codec or codec_of_solid_key(solid_model, solid_profile)
    if codec is None:
        return dict(reason='solid_ink_has_no_luma_contrast')
    span, g_k = codec['span'], codec['g']
    thin_dark = codec['paper_ycc'][0]-thin_profile['ycc'][..., 0]
    if float(np.clip(thin_dark[thin_profile['window']], 0, None).sum())/span < 3.:
        return dict(reason='no_thin_ink')
    g_m, cost_free, reach_m = _measure(thin_profile, codec, [g_k, .5*g_k+[.01, .01]])
    basis, scale, chromatic = _frame(g_k)
    errors = []
    for shift in MIXING_PHASES:
        synthetic, reach_true = _twin_render(thin_profile, codec, shift)
        g_b, _, reach_b = _measure(synthetic, codec, [g_k])
        errors.append(np.r_[basis@(g_b-g_k)/scale, reach_b-reach_true])
    bias = np.asarray(errors).mean(axis=0)
    return dict(reason=None, codec=codec, basis=basis, scale=scale, chromatic=chromatic,
                g_thin=g_m, reach=reach_m, bias=bias, offset=basis@(g_m-g_k)/scale-bias[:2],
                reach_corrected=reach_m-bias[2],
                envelope=np.asarray(_envelope(codec['mode'], chromatic, codec.get('undetermined', False)), np.float64),
                thin_rms_free=cost_free, thin_rms_same=_residual(thin_profile, codec['paper_ycc'], g_k, codec['mode']))


_BALL3 = np.asarray([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)], np.float64)-1.
_BALL3 = np.vstack([np.zeros(3), _BALL3/np.linalg.norm(_BALL3, axis=1, keepdims=True)])
_BALL2 = np.asarray([[np.cos(a), np.sin(a)] for a in np.arange(8)*np.pi/4])
_BALL2 = np.vstack([np.zeros(2), _BALL2])


def mixing_verdict(measurement, solid_ink, colour_distance=14., hue_family=True, scale=None):
    """'same' / 'distinct' / 'abstain' of a mixing measurement (see mixing_family_test)."""
    radius = MIXING_ENVELOPE_SCALE if scale is None else scale
    codec = measurement['codec']
    P, span, g_k = codec['paper_ycc'], codec['span'], codec['g']
    hw = measurement['envelope']*radius
    c, r = measurement['offset'], measurement['reach_corrected']

    def inks(g_offsets, levels):
        g = g_k+(g_offsets*measurement['scale'])@measurement['basis']
        s = np.asarray(levels, np.float64)[:, None]
        return _bgr(np.hstack([P[0]-s*span, P[1:]+g*s*span]))
    ball = _BALL3*hw
    observed = _palette_same_to(solid_ink, inks(c+ball[:, :2], r+ball[:, 2]), colour_distance, hue_family)
    disc = _BALL2*hw[:2]
    line = _palette_same_to(solid_ink, inks(c+disc, np.ones(len(disc))), colour_distance, hue_family)
    if observed.all() and line.all():
        return 'same', 'thin_pixels_are_paper_mixtures_of_solid_ink'
    if not line.any():
        return 'distinct', 'thin_key_on_another_mixing_line'
    if r-hw[2] > 1. and not observed.any():
        return 'distinct', 'thin_key_darker_than_solid_ink'
    return 'abstain', ('paler_ink_explains_thin_key_equally' if line.all() and r < 1. else
                       'thin_pixels_admit_same_and_distinct_inks')


def mixing_family_test(image, solid_model, solid_profile, thin_model, thin_profile, *,
                       colour_distance=14., hue_family=True, codec=None):
    """Is the THIN key the SOLID key's ink rendered as a thin stroke?

    H_same: the thin key's pixels are paper/ink mixtures of the solid key's
    observed ink K under the rendering model above (any coverage <= 1),
    measured by mixing_measurement (codec-corrected, bias-corrected).
    Inks are written (g, s): luma darkness s x K's, chroma g x that darkness.
    The margin is a ball of MIXING_ENVELOPE_SCALE twin envelopes (P95 of TRUE
    twins on the stress renders, per codec) around the measurement.
      L = the thin key's mixing line at K's luma: (measured g, s = 1). A darker
          or lighter SHADE of the same hue lies on another line (red/darkred,
          blue/navy); K is on it for a twin.
      O = the ink the thin glyph's strongest pixels show: (measured g, reach),
          its lightest consistent ink. reach < 1: a distinct paler ink at full
          coverage explains the glyph as well as K at partial coverage;
          reach > 1: the thin key is darker than K.
    'same'     every ink of the L and O balls is the caller's palette
               same-ink as K (palette_relations: one definition everywhere);
    'distinct' no ink of the L ball is (another line), or the thin key is
               darker than K beyond the envelope and no ink of O is;
    'abstain'  otherwise -- neither explanation wins; the test has no say and
               the caller's summary relation stands (mixing_family_relation).
    Darker inks on K's own line are not alternatives: a thin stroke whose
    strongest pixels show K is explained by K at full coverage.
    """
    m = mixing_measurement(solid_model, solid_profile, thin_profile, codec)
    if m['reason'] is not None:
        return dict(verdict=None, reason=m['reason'])
    ink = np.asarray(solid_model['bgr'], np.float64)
    verdict, reason = mixing_verdict(m, ink, colour_distance, hue_family)
    codec, P, span = m['codec'], m['codec']['paper_ycc'], m['codec']['span']
    implied = np.clip(_bgr(np.r_[codec['ink_ycc'][0], P[1:]+m['g_thin']*span]), 0, 255)
    reached = np.clip(_bgr(np.r_[P[0]-m['reach']*span, P[1:]+m['g_thin']*m['reach']*span]), 0, 255)
    return dict(verdict=verdict, reason=reason, chroma_mode=codec['mode'], solid_fit_rms=codec['rms'],
                lossless_fit_rms=codec['lossless_rms'], reach=m['reach'], reach_corrected=m['reach_corrected'],
                g_solid=[float(v) for v in codec['g']], g_thin=[float(v) for v in m['g_thin']],
                frame='radial_tangential_relative' if m['chromatic'] else 'absolute_cb_cr',
                g_offset_corrected=[float(v) for v in m['offset']], envelope=[float(v) for v in m['envelope']],
                bootstrap_bias=[float(v) for v in m['bias']],
                thin_rms_free=m['thin_rms_free'], thin_rms_same=m['thin_rms_same'],
                implied_bgr=[float(v) for v in implied], reached_bgr=[float(v) for v in reached],
                line_offset_rgb=float(np.linalg.norm(implied-ink)), reach_offset_rgb=float(np.linalg.norm(reached-ink)),
                solid_thickness=solid_profile['thickness'], thin_thickness=thin_profile['thickness'],
                measurement=m)


def legend_entry_inputs(image, entries):
    """(models, boxes) of colour-legend entries exactly as routing and grouping measure them.

    Model = the entry's key box with its own RGB (color_marker_evidence._swatch_model);
    line-only keys have no marker glyph (box None).
    """
    from color_marker_evidence import _swatch_model
    models, boxes = [], []
    for e in entries:
        box = tuple(int(v) for v in e['box'])
        try:
            models.append(_swatch_model(image, box, e['rgb']))
        except ValueError:
            bgr = np.asarray(e['rgb'][::-1], np.float32)
            models.append(dict(bgr=bgr, achromatic=bool(np.ptp(bgr) < 24)))
            box = None
        boxes.append(box if e.get('marker_template', True) else None)
    return models, boxes


def mixing_family_relation(image, models, boxes, *, colour_distance=14., hue_family=True, summary=None):
    """Same-ink relation of legend keys: the palette summary relation, extended
    by the physical mixing test where the summary relation has no say.

    Returns (same, verdicts, report). ``same`` is the relation used to group
    keys (symmetric bool). ``verdicts[i][j]`` is the mixing verdict of a
    (solid i, thin j) pair, or None where the test does not apply.
    * The summary relation is kept exactly: every pair it calls one ink stays
      one ink, whatever the mixing test says. A 'distinct' or 'abstain'
      verdict never splits keys the summary merged. ``summary`` is the
      caller's own relation in the order of ``models`` (default:
      palette_relations of the models' colours).
    * A 'same' verdict (the thin key's pixels are paper mixtures of the solid
      key's ink under the image codec) ADDS the pair only when both keys are
      summary singletons: it joins a thin key to its solid twin, it never
      adds a member to a group of keys the summary relation already formed.
      Such a group's shape competition and per-swatch scale calibration are a
      working path; a new member changes both for every existing member
      (measured: a thin red ring joining a red square/circle group changed
      the square's calibrated scale and lost one of four overlapping squares).
    * two thin keys explained by the same solid ink share it; a thin key
      explained by two solid inks that are not one ink joins neither.
    * neutral keys: all gray inks lie on one paper->black axis, so the
      line test cannot separate two grays; a (solid, thin) neutral pair is
      'same' only when the thin key reaches the solid gray. Glyph shape is
      NOT an identity test: sharing observations between different gray
      inks is the caller's neutral_geometry_policy ('shared_source', opt-in),
      never a side effect of this same-ink relation.
    """
    n = len(models)
    if summary is None:
        summary, _ = palette_relations([m['bgr'] for m in models], colour_distance, hue_family=hue_family)
    summary = np.array(summary, bool)
    if summary.shape != (n, n):
        raise ValueError('Summary same-ink relation must align with the models')
    same = summary.copy()
    verdicts = [[None]*n for _ in range(n)]
    report = dict(version=MIXING_VERSION, halo_px=MIXING_HALO_PX, phases=MIXING_PHASES, envelope=MIXING_ENVELOPE, envelope_scale=MIXING_ENVELOPE_SCALE,
                  thin_px=MIXING_THIN_PX, solid_px=MIXING_SOLID_PX, keys=[], pairs=[],
                  ambiguous_thin_keys=[], added_pairs=[], not_added_summary_group=[],
                  summary_relation_kept=True)
    if n < 2 or len(boxes) != n:
        return same, verdicts, report
    # A summary singleton shares its ink with no other key of the legend.
    single = summary.sum(axis=1) <= 1
    profiles = []
    for model, box in zip(models, boxes):
        try:
            profiles.append(key_ink_profile(image, box, model) if box is not None else None)
        except (ValueError, cv2.error):
            profiles.append(None)
    for i, prof in enumerate(profiles):
        report['keys'].append(dict(index=i, measured=prof is not None,
            thickness=None if prof is None else prof['thickness'],
            glyph_pixels=None if prof is None else prof['glyph_size'],
            summary_singleton=bool(single[i])))
    solid = [p is not None and p['thickness'] >= MIXING_SOLID_PX for p in profiles]
    thin = [p is not None and p['thickness'] <= MIXING_THIN_PX for p in profiles]
    anchors = {}
    # One codec for the image, measured on its chromatic solid keys.
    own = {i: codec_of_solid_key(models[i], profiles[i]) for i in range(n) if solid[i] and any(thin)}
    mode = image_codec(list(own.values()))
    codecs = {i: (c if c is None or mode is None else codec_of_solid_key(models[i], profiles[i], mode))
              for i, c in own.items()}
    report['image_codec'] = mode
    for i, c in codecs.items():
        report['keys'][i].update(codec=None if c is None else c['mode'], codec_undetermined=None if c is None else c['undetermined'],
                                 codec_rms=None if c is None else c['rms'])
    for i in range(n):
        for j in range(n):
            if i == j or not solid[i] or not thin[j]:
                continue
            row = mixing_family_test(image, models[i], profiles[i], models[j], profiles[j],
                                     colour_distance=colour_distance, hue_family=hue_family, codec=codecs[i])
            row.pop('measurement', None)
            verdicts[i][j] = row['verdict']
            report['pairs'].append(dict(solid=i, thin=j, **row))
            if row['verdict'] == 'same' and not summary[i, j]:
                if single[i] and single[j]:
                    anchors.setdefault(j, []).append(i)
                else:
                    report['not_added_summary_group'].append([int(i), int(j)])
    usable = {}
    for j, solids in anchors.items():
        if all(same[p, q] for p in solids for q in solids):
            usable[j] = solids
        else:
            report['ambiguous_thin_keys'].append(dict(thin=j, solid_candidates=solids))
    for j, solids in usable.items():
        for i in solids:
            same[i, j] = same[j, i] = True
            report['added_pairs'].append([int(i), int(j)])
        for k, other in usable.items():
            if k != j and set(solids) & set(other):
                same[j, k] = same[k, j] = True
    return same, verdicts, report


def mixing_twin_pairs(image, entries, *, hue_family=True):
    """Legend entry pairs that are one ink by the mixing test but not by the
    single-RGB summary relation (palette_routing's), with the test report.

    On the general colour route every key is a summary singleton (a summary
    same-ink pair routes the whole chart to grouped BW), so these pairs are
    thin keys and their solid twins. They are NOT used to change the route
    or any evidence plane; the caller only lets them compete for the same
    observed marker (shape identity), as same-colour keys already do.
    """
    if len(entries) < 2:
        return [], None
    models, boxes = legend_entry_inputs(image, entries)
    summary, _ = palette_relations([np.asarray(e['rgb'], np.float32)[::-1] for e in entries], hue_family=hue_family)
    related, _, report = mixing_family_relation(image, models, boxes, hue_family=hue_family, summary=summary)
    pairs = [[int(i), int(j)] for i in range(len(entries)) for j in range(i+1, len(entries))
             if related[i, j] and not summary[i, j]]
    return pairs, report


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


def distinguish_inks(image, models, membership, confidence, *, equivalent=None, member_models=None):
    """Attenuate support explained better by a distinguishable source ink.

    The unchanged per-swatch hue evidence stays the proposal source. RGB
    competition is only a multiplier <=1, shared by initial grids, scale
    calibration and final windows. Same-ink shapes never compete by colour.
    Uncertain ties retain evidence rather than using palette/series order.
    ``member_models`` (one list per model) scores a multi-key group's ray fit
    against each member key's measured ink (maximum), not its median summary.
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
    # Hue is unstable near neutral: a few JPEG channel levels can point in a
    # vivid blue's hue direction despite being a measured gray source ink.
    # A known neutral palette rival supplies independent negative evidence.
    # Compare its actual ink/paper ray, not centre proximity or a forced
    # one-series-per-pixel assignment; ambiguous mixtures keep their evidence.
    neutral=np.asarray([bool(m.get('achromatic',np.ptp(m['bgr'])<24)) for m in models])
    neutral_chromatic=neutral[:,None]^neutral[None,:]
    rivals = (near | neutral_chromatic) & ~same
    report = dict(version=VERSION, same_ink=same.tolist(), competing_pairs=[],
                  neutral_chromatic_pairs=[],
                  positive_ink_added=False, unknown_or_paper_is_not_a_rival=True)
    active = np.flatnonzero(rivals.any(axis=1))
    if not len(active):
        return membership, confidence, report
    if member_models is not None and len(member_models) != len(models):
        raise ValueError('Member palette models must align with the group models')
    fits = {int(i): (_paper_ray_fit(image, models[i]) if member_models is None or not member_models[i] else
                     np.max([_paper_ray_fit(image, k) for k in member_models[i]], axis=0))
            for i in active}
    m, c = membership.copy(), confidence.copy()
    for i in active:
        gate = np.ones(expected[1:], np.float32)
        for j in np.flatnonzero(rivals[i]):
            # A rival must have independently observed ink, not just proximity
            # to the paper endpoint of every colour ray.
            observed = np.clip(membership[j]/.25, 0., 1.)
            better = np.clip((fits[j]-fits[i]-.10)/.40, 0., 1.)
            gate = np.minimum(gate, 1.-fits[j]*observed*better)
            if i < j:
                report['competing_pairs'].append([int(i), int(j)])
                if neutral_chromatic[i,j]:
                    report['neutral_chromatic_pairs'].append([int(i),int(j)])
        m[i] *= gate
        c[i] *= gate
    return m, c, report


def _darkest_glyph_alpha(template):
    """Largest ink strength the key's own legend glyph shows on its ink/paper ray.

    A glyph pixel is a coverage mixture of ink and paper, so the true ink is at
    least as dark as the darkest pixel the key itself contains. A thin glyph
    that never saturates is measured paler than its ink and reports alpha > 1
    here; a saturated glyph reports ~1. Every pixel the key's own soft template
    claims counts (a stray dark label pixel can only make this larger, i.e. more
    permissive). None when the template carries no raw legend pixels.
    """
    raw, soft = template.get('raw_bgr'), template.get('soft')
    if raw is None or soft is None:
        return None
    raw = np.asarray(raw, np.float32)
    soft = np.asarray(soft, np.float32)
    if raw.shape[:2] != soft.shape:
        return None
    model = template['model']
    paper = np.asarray(model.get('paper_bgr', [255., 255., 255.]), np.float32)
    direction = paper-np.asarray(model['bgr'], np.float32)
    own = soft >= .18
    if not own.any():
        return None
    alpha = ((paper-raw[own])@direction)/max(float(direction@direction), 1.)
    return max(1., float(alpha.max()))


def neutral_tone_gate(image, templates, membership):
    """Per-key multiplier <= 1: where a neutral key's support is ink only a DARKER neutral key can explain.

    distinguish_inks leaves neutral/neutral pairs alone: a small glyph is often
    measured paler than its plot ink, so 'darker than the key colour' is not
    evidence by itself, and the per-key achromatic membership tolerates that
    overshoot (a mid-grey key keeps ~0.85 of its support on pure black ink).
    The key's own legend pixels bound the measurement error: its ink is at least
    as dark as its darkest glyph pixel, and nothing shows it darker. Key i is
    therefore represented by its darkest observed glyph tone (never paler than its
    measured colour) and is compared only with neutral keys whose measured ink is
    darker still, using distinguish_inks' paper-ray fit, margin and gate. A pixel
    paler than key i's ink is an antialiased mixture every darker neutral ink also
    explains; both fits agree there and the multiplier stays 1. Same-ink keys never
    compete. The fields themselves are NOT changed here: the caller applies the
    multiplier only where a candidate's evidence fails the consistency check
    (color_marker_hybrid_v2.analyze_evidence). Returns (gate or None, report).
    """
    membership = np.asarray(membership, np.float32)
    models = [t['model'] for t in templates]
    report = dict(version='neutral_tone_identity_v2_candidate_gated', pairs=[], positive_ink_added=False)
    neutral = np.asarray([bool(m.get('achromatic', np.ptp(m['bgr']) < 24)) for m in models])
    if neutral.sum() < 2:
        return None, report
    same, _ = palette_relations([m['bgr'] for m in models])
    contrast = lambda m, bgr: float(np.linalg.norm(np.asarray(m.get('paper_bgr', [255., 255., 255.]), np.float32)
                                                  - np.asarray(bgr, np.float32)))
    gate = np.ones(membership.shape, np.float32)
    for i in np.flatnonzero(neutral):
        darkest = _darkest_glyph_alpha(templates[i])
        if darkest is None:
            continue
        model = models[i]
        paper = np.asarray(model.get('paper_bgr', [255., 255., 255.]), np.float32)
        ink = paper-darkest*(paper-np.asarray(model['bgr'], np.float32))
        reach = contrast(model, ink)
        rivals = [j for j in np.flatnonzero(neutral) if j != i and not same[i, j]
                  and contrast(models[j], models[j]['bgr']) > reach]
        if not rivals:
            continue
        fit_i = _paper_ray_fit(image, dict(model, bgr=np.clip(ink, 0., 255.)))
        for j in rivals:
            fit_j = _paper_ray_fit(image, models[j])
            observed = np.clip(membership[j]/.25, 0., 1.)
            better = np.clip((fit_j-fit_i-.10)/.40, 0., 1.)
            pair_gate = 1.-fit_j*observed*better
            gate[i] = np.minimum(gate[i], pair_gate)
            report['pairs'].append(dict(key=int(i), darker_rival=int(j), darkest_glyph_alpha=round(darkest, 3),
                                        attenuated_pixels=int((pair_gate < .5).sum())))
    if not report['pairs']:
        return None, report
    return gate, report

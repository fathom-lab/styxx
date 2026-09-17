"""the plate — rendering primitives.

a circular plate, driven at a mixture of Bessel modes, with sand settling on the
nodal lines. the mode mixture is a deterministic function of a sha256 digest, so
a file's plate is a function of the file's bytes: same bytes, same figure.

no ML, no network. numpy + scipy + pillow.
"""
import hashlib
import numpy as np
from PIL import Image, ImageFilter, ImageFont
from scipy.special import jv, jn_zeros

# ---- palette: obsidian plate, phosphor sand ----
GROUND     = np.array([5, 6, 9], float)
PLATE_DARK = np.array([13, 15, 20], float)
PLATE_MID  = np.array([27, 30, 38], float)
TEAL       = np.array([112, 226, 204], float)   # glow
CORE       = np.array([244, 252, 249], float)   # hot core of a grain
SEED       = int(hashlib.sha256(b"nothing crosses unseen").hexdigest()[:8], 16)

# fonts: latin modern mono if present, else dejavu, else pillow's default
_FONT_CANDIDATES = {
    'regular': ['/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'],
    'light':   ['/usr/share/texmf/fonts/opentype/public/lm/lmmonolt10-regular.otf',
                '/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'],
}
LM = _FONT_CANDIDATES['regular'][0]
LMLIGHT = _FONT_CANDIDATES['light'][0]


def font(p, s):
    cands = [p] + _FONT_CANDIDATES['light'] + _FONT_CANDIDATES['regular']
    for c in cands:
        try:
            return ImageFont.truetype(c, s)
        except Exception:
            continue
    return ImageFont.load_default()


def blur(a, rad):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), 'L').filter(ImageFilter.GaussianBlur(rad))
    return np.asarray(im, float) / 255.0


def shift(a, dy, dx):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def geometry(W, H, cx, cy, Rpx):
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx - cx) / Rpx
    y = (yy - cy) / Rpx
    return x, y, np.hypot(x, y), np.arctan2(y, x)


def field(r, th, modes):
    """sum of circular-plate modes (n, zero index, weight, phase), normalised on the disc."""
    f = np.zeros_like(r)
    for n, zi, w, ph in modes:
        k = jn_zeros(n, zi)[-1] / 0.965
        f += w * jv(n, k * r) * np.cos(n * th + ph)
    f /= np.abs(f[r <= 1]).max()
    return f


def modes_from_hash(hexdigest):
    """deterministic map from a 32-byte digest to three plate modes.

    bytes 0-2 pick the dominant mode, 3-6 the second, 7-10 the third. any other
    map would do; what matters is that it is fixed and public, so the figure is
    a function of the bytes and nothing else.
    """
    b = bytes.fromhex(hexdigest)
    m1 = (2 + b[0] % 4, 2 + b[1] % 3, 1.00, b[2] / 255 * np.pi)
    m2 = (b[3] % 4, 1 + b[4] % 3, 0.28 + 0.34 * b[5] / 255, b[6] / 255 * np.pi)
    m3 = (1 + b[7] % 5, 1 + b[8] % 2, 0.10 + 0.16 * b[9] / 255, b[10] / 255 * np.pi)
    return [m1, m2, m3]


def plate(W, H, x, y, r, rng):
    disc = r <= 1.0
    lam = np.clip(0.5 + 0.5 * (x * -0.55 + y * -0.72), 0, 1)          # lit from top-left
    base = PLATE_DARK[None, None, :] + (PLATE_MID - PLATE_DARK)[None, None, :] * lam[..., None]
    # brushed concentric texture
    t = rng.normal(0, 1, 6000)
    t = np.convolve(t, np.ones(5) / 5, 'same')
    tex = t[np.clip((r * 5999), 0, 5999).astype(int)] * 2.4
    base = base + tex[..., None] + rng.normal(0, 1.1, (H, W, 1))
    # soft specular
    spec = np.exp(-((x + 0.38) ** 2 + (y + 0.52) ** 2) / (2 * 0.42 ** 2)) * 16
    base = base + spec[..., None]
    # rim bevel + bright edge
    band = np.clip((r - 0.950) / 0.050, 0, 1)
    base = base + (band * (lam * 78 - 22))[..., None]
    edge = np.exp(-((r - 0.993) / 0.0045) ** 2) * (0.25 + lam) * 95
    base = base + edge[..., None]
    img = np.where(disc[..., None], base, GROUND[None, None, :])
    # contact shadow on the ground just outside the rim
    outside = np.clip(r - 1.0, 0, None)
    img = img * (1 - 0.35 * np.exp(-(outside / 0.05) ** 2) * (~disc))[..., None]
    return img, disc


def sand(f, disc, r, rng, sigma=0.033, grain=0.66, noise=None):
    """sand density on the nodal lines. pass `noise` (three arrays) to keep grains fixed across frames."""
    d = np.exp(-(f / sigma) ** 2)
    d = np.maximum(d, 0.32 * np.exp(-((r - 0.955) / 0.010) ** 2))          # dust on the free edge
    d = d + 0.018 * np.exp(-(f / (sigma * 4.5)) ** 2)                          # unsettled grains near the lines
    d = d * disc
    if noise is None:
        n_fine, n_coarse, n_v = rng.random(f.shape), rng.random(f.shape), 0.78 + 0.22 * rng.random(f.shape)
    else:
        n_fine, n_coarse, n_v = noise
    fine = n_fine < d * grain
    coarse = n_coarse < d * grain * 0.15
    v = np.clip(blur(fine, 0.6) * 1.5 + blur(coarse, 1.4) * 1.3, 0, 1)
    return v * n_v


def composite(img, v, sparkle_seed=SEED + 7):
    g1, g2, g3 = blur(v, 4), blur(v, 14), blur(v, 40)
    sh = shift(blur(v, 1.6), 3, 2)
    img = img * (1 - 0.40 * sh)[..., None]                                   # grains cast a little shadow
    img = img + TEAL[None, None, :] * (0.42 * g1 + 0.30 * g2 + 0.16 * g3)[..., None]  # bloom
    col = TEAL[None, None, :] + (CORE - TEAL)[None, None, :] * (v ** 1.0)[..., None]
    img = img * (1 - v[..., None]) + col * v[..., None]
    # sparkle: a few grains catch the light
    sp = (v > 0.55) & (np.random.default_rng(sparkle_seed).random(v.shape) < 0.012)
    spk = blur(sp.astype(float), 1.1) * 3.0
    img = img + CORE[None, None, :] * np.clip(spk, 0, 1)[..., None] * 0.9
    return img


def render_plate(hexdigest, S=760, radius=0.76):
    """one plate tile (S x S) for a digest. deterministic in the digest."""
    seed = int(hexdigest[:8], 16)
    rng = np.random.default_rng(seed)
    x, y, r, th = geometry(S, S, S / 2, S / 2, radius * S / 2)
    f = field(r, th, modes_from_hash(hexdigest))
    img, disc = plate(S, S, x, y, r, rng)
    v = sand(f, disc, r, rng)
    return composite(img, v)


def save(img, path):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB').save(path, optimize=True)


def sha256_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

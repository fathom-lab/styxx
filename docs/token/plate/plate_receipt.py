"""the plate — a receipt rendered as a chladni figure.

    python plate_receipt.py                       # ../receipt/launch_receipt.json
    python plate_receipt.py path/to/any/file      # any file at all
    python plate_receipt.py FILE --flip 'old' 'new' --out plate.png

renders two plates side by side: the file as it is, and the same file with one
substring changed (default: creator_tax_bps 0 -> 1 in the launch receipt). the
figure is a deterministic function of sha256(bytes), so the two plates agree if
and only if the bytes do. no dependencies beyond numpy, scipy, pillow.
"""
import argparse, hashlib, os, sys
import numpy as np
from PIL import Image, ImageDraw
from plate_lib import (GROUND, TEAL, LMLIGHT, font, render_plate, sha256_file)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECEIPT = os.path.join(HERE, '..', 'receipt', 'launch_receipt.json')


def two_plates(h_left, h_right, cap_left, cap_right, out, W=1600, H=900):
    S = 760
    rng = np.random.default_rng(11)
    canvas = np.tile(GROUND[None, None, :], (H, W, 1)).astype(float)
    canvas += rng.normal(0, 1.0, (H, W, 1))
    centers = [(430, 430), (1170, 430)]
    for hexd, (cx, cy) in zip([h_left, h_right], centers):
        tile = render_plate(hexd, S)
        x0, y0 = cx - S // 2, cy - S // 2
        region = canvas[y0:y0 + S, x0:x0 + S]
        canvas[y0:y0 + S, x0:x0 + S] = np.where(tile > GROUND + 2.5, tile, region)
    for cx, cy in centers:
        amb = (np.exp(-((np.arange(W) - cx) / 520.0) ** 2)[None, :]
               * np.exp(-((np.arange(H) - cy) / 420.0) ** 2)[:, None])
        canvas += TEAL[None, None, :] * (amb * 0.035)[..., None]

    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), 'RGB')
    d = ImageDraw.Draw(im)
    INK, MUTE, DIM = (236, 244, 241), (150, 208, 196), (104, 122, 118)
    d.text((64, 42), 'the plate', fill=INK, font=font(LMLIGHT, 54))
    d.text((66, 112), 'a receipt rendered as a chladni figure.  figure = f(sha256(bytes))',
           fill=MUTE, font=font(LMLIGHT, 24))
    cx_, cy_ = W // 2, 430
    same = h_left == h_right
    for dy in (-14, 14):
        d.line([(cx_ - 30, cy_ + dy), (cx_ + 30, cy_ + dy)], fill=MUTE, width=4)
    if not same:
        d.line([(cx_ - 16, cy_ + 42), (cx_ + 16, cy_ - 42)], fill=MUTE, width=4)

    def caption(cx, lines):
        for i, (t, col, sz) in enumerate(lines):
            f_ = font(LMLIGHT, sz)
            w = d.textlength(t, font=f_)
            d.text((cx - w / 2, 758 + i * 30), t, fill=col, font=f_)

    caption(430, [(cap_left[0], INK, 26), (cap_left[1], DIM, 19),
                  ('sha256 ' + h_left[:16] + '…' + h_left[-8:], MUTE, 19)])
    caption(1170, [(cap_right[0], INK, 26), (cap_right[1], DIM, 19),
                   ('sha256 ' + h_right[:16] + '…' + h_right[-8:], MUTE, 19)])
    d.text((64, H - 44), 'styxx · nothing crosses unseen', fill=DIM, font=font(LMLIGHT, 20))
    d.text((W - 64 - d.textlength('@styxxhq', font=font(LMLIGHT, 20)), H - 44), '@styxxhq',
           fill=DIM, font=font(LMLIGHT, 20))
    im.save(out, optimize=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=DEFAULT_RECEIPT)
    ap.add_argument('--flip', nargs=2, metavar=('OLD', 'NEW'),
                    default=['"creator_tax_bps": 0,', '"creator_tax_bps": 1,'],
                    help='substring to change in the right-hand copy (first occurrence)')
    ap.add_argument('--out', default='styxx_plate_receipt.png')
    a = ap.parse_args(argv)

    raw = open(a.file, 'rb').read()
    old, new = a.flip[0].encode(), a.flip[1].encode()
    if old not in raw:
        sys.exit(f'--flip: {a.flip[0]!r} not found in {a.file}')
    alt = raw.replace(old, new, 1)
    h_real = hashlib.sha256(raw).hexdigest()
    h_alt = hashlib.sha256(alt).hexdigest()
    name = os.path.basename(a.file)
    delta = len(alt) - len(raw)
    change = 'one byte changed' if delta == 0 and sum(x != y for x, y in zip(raw, alt)) == 1 else 'changed'
    print('real', h_real)
    print('alt ', h_alt)
    out = two_plates(h_real, h_alt,
                     (name, 'the file as published'),
                     (f'{name}, {change}', f'{a.flip[0].strip()} → {a.flip[1].strip()}'),
                     a.out)
    print('wrote', out)


if __name__ == '__main__':
    main()

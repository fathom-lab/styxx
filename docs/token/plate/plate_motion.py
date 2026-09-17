"""the plate as a moving picture.

    python plate_motion.py                      # ../receipt/launch_receipt.json -> styxx_plate_motion.mp4
    python plate_motion.py FILE --flip OLD NEW --out plate.mp4
    python plate_motion.py --quick              # 1/5 of the frames, for a smoke test

one plate. the sand sits on the figure of the file as published, the file
changes by one byte, the sand walks to the new figure, then walks back. the
field is interpolated between the two mode sets; the grain noise is fixed so
grains move rather than flicker. frames go to a folder, then ffmpeg (if on
PATH) encodes an h264 mp4 at 30 fps.
"""
import argparse, hashlib, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from plate_lib import (GROUND, TEAL, LMLIGHT, font, geometry, field, plate, sand,
                       composite, modes_from_hash)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECEIPT = os.path.join(HERE, '..', 'receipt', 'launch_receipt.json')


def smoothstep(t):
    return t * t * (3 - 2 * t)


def render(h_real, h_alt, name, flip_txt, out, frames_dir, hold=45, morph=75, W=1280, H=720, fps=30):
    S, CX, CY = 660, 890, 360
    seed = int(h_real[:8], 16)
    rng = np.random.default_rng(seed)
    x, y, r, th = geometry(S, S, S / 2, S / 2, 0.80 * S / 2)
    disc = r <= 1.0
    base, _ = plate(S, S, x, y, r, rng)
    fA = field(r, th, modes_from_hash(h_real))
    fB = field(r, th, modes_from_hash(h_alt))
    noise = (rng.random(fA.shape), rng.random(fA.shape), 0.78 + 0.22 * rng.random(fA.shape))

    ground = np.tile(GROUND[None, None, :], (H, W, 1)).astype(float) + np.random.default_rng(3).normal(0, 1.0, (H, W, 1))
    amb = (np.exp(-((np.arange(W) - CX) / 520.0) ** 2)[None, :] * np.exp(-((np.arange(H) - CY) / 420.0) ** 2)[:, None])
    ground = ground + TEAL[None, None, :] * (amb * 0.035)[..., None]

    INK, MUTE, DIM, RED = (236, 244, 241), (150, 208, 196), (104, 122, 118), (255, 120, 112)
    F_T, F_S, F_M, F_H = font(LMLIGHT, 50), font(LMLIGHT, 21), font(LMLIGHT, 24), font(LMLIGHT, 19)
    os.makedirs(frames_dir, exist_ok=True)

    def frame(t, state, k):
        f = (1 - t) * fA + t * fB
        f = f / np.abs(f[disc]).max()
        v = sand(f, disc, r, rng, noise=noise)
        tile = composite(base.copy(), v, sparkle_seed=seed + 7)
        canvas = ground.copy()
        x0, y0 = CX - S // 2, CY - S // 2
        region = canvas[y0:y0 + S, x0:x0 + S]
        canvas[y0:y0 + S, x0:x0 + S] = np.where(tile > GROUND + 2.5, tile, region)
        im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), 'RGB')
        d = ImageDraw.Draw(im)
        d.text((64, 56), 'the plate', fill=INK, font=F_T)
        d.text((66, 120), 'a receipt rendered as a chladni figure', fill=MUTE, font=F_M)
        d.text((66, 152), 'figure = f(sha256(bytes))', fill=MUTE, font=F_M)
        yy = 300
        d.text((66, yy), name, fill=INK, font=F_M)
        if state == 'A':
            d.text((66, yy + 36), 'the file as published', fill=DIM, font=F_S)
            d.text((66, yy + 64), 'sha256 ' + h_real[:12] + '…' + h_real[-6:], fill=MUTE, font=F_H)
        elif state == 'B':
            d.text((66, yy + 36), 'one byte changed: ' + flip_txt, fill=RED, font=F_S)
            d.text((66, yy + 64), 'sha256 ' + h_alt[:12] + '…' + h_alt[-6:], fill=MUTE, font=F_H)
        elif state == 'AB':
            d.text((66, yy + 36), 'one byte changed: ' + flip_txt, fill=RED, font=F_S)
            d.text((66, yy + 64), 'the sand walks to the new figure', fill=DIM, font=F_H)
        else:
            d.text((66, yy + 36), 'byte restored', fill=MUTE, font=F_S)
            d.text((66, yy + 64), 'the sand walks back', fill=DIM, font=F_H)
        d.text((66, yy + 130), 'same bytes, same figure.', fill=INK, font=F_S)
        d.text((66, yy + 158), 'different bytes, different figure.', fill=INK, font=F_S)
        d.text((66, yy + 186), 'no one has to take our word for it.', fill=INK, font=F_S)
        d.text((64, H - 52), 'styxx · nothing crosses unseen · @styxxhq', fill=DIM, font=F_H)
        im.save(f'{frames_dir}/f{k:04d}.png')

    seq = ([('A', 0.0)] * hold
           + [('AB', smoothstep(i / (morph - 1))) for i in range(morph)]
           + [('B', 1.0)] * hold
           + [('BA', 1 - smoothstep(i / (morph - 1))) for i in range(morph)])
    cache = {}
    for k, (state, t) in enumerate(seq):
        key = state if state in ('A', 'B') else None
        if key and key in cache:
            shutil.copyfile(cache[key], f'{frames_dir}/f{k:04d}.png')
        else:
            frame(t, state, k)
            if key:
                cache[key] = f'{frames_dir}/f{k:04d}.png'
        if (k + 1) % 20 == 0:
            print(f'frame {k + 1}/{len(seq)}', flush=True)

    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{frames_dir}/f%04d.png',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                        '-movflags', '+faststart', out], check=True)
        print('wrote', out)
    else:
        print(f'ffmpeg not found; frames are in {frames_dir}/ — encode with:\n'
              f'  ffmpeg -framerate {fps} -i {frames_dir}/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='?', default=DEFAULT_RECEIPT)
    ap.add_argument('--flip', nargs=2, metavar=('OLD', 'NEW'),
                    default=['"creator_tax_bps": 0,', '"creator_tax_bps": 1,'])
    ap.add_argument('--out', default='styxx_plate_motion.mp4')
    ap.add_argument('--frames', default='motion_frames')
    ap.add_argument('--quick', action='store_true', help='1/5 of the frames (smoke test)')
    a = ap.parse_args(argv)
    raw = open(a.file, 'rb').read()
    old, new = a.flip[0].encode(), a.flip[1].encode()
    if old not in raw:
        sys.exit(f'--flip: {a.flip[0]!r} not found in {a.file}')
    alt = raw.replace(old, new, 1)
    h_real, h_alt = hashlib.sha256(raw).hexdigest(), hashlib.sha256(alt).hexdigest()
    print('real', h_real)
    print('alt ', h_alt)
    hold, morph = (9, 15) if a.quick else (45, 75)
    render(h_real, h_alt, os.path.basename(a.file), f'{a.flip[0].strip()} → {a.flip[1].strip()}',
           a.out, a.frames, hold=hold, morph=morph)


if __name__ == '__main__':
    main()

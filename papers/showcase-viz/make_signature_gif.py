"""Render a postable GIF of the live signature — gemma-2-2b's real geometry reacting to its real
layer-12 truthfulness probe. Reads geometry_render.json + curated real statements; additive glow,
shockwaves, trace, caption. Output: signature_demo.gif.

Usage: python papers/showcase-viz/make_signature_gif.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 640, 540
FPS_MS = 55
PER = 26                      # frames per statement
CYAN = np.array([0, 229, 255], float)
RED = np.array([255, 42, 52], float)
FONT = "C:/Windows/Fonts/consola.ttf"

# curated real readings (text, label, p_correct) — strong cyan/red rhythm, recognizable facts
SEQ = [
    ("Ice is frozen water.", "TRUE", 0.9513),
    ("Ice is frozen oil.", "FALSE", 0.0119),
    ("The chemical symbol for gold is Au.", "TRUE", 0.9018),
    ("The Sun orbits the Earth.", "FALSE", 0.0002),
    ("The human heart pumps blood.", "TRUE", 0.7247),
    ("Bees make milk.", "FALSE", 0.0001),
]


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def glow_stamp(rad=22, sigma=7.0):
    a = np.arange(-rad, rad + 1)
    xx, yy = np.meshgrid(a, a)
    return np.exp(-((np.sqrt(xx * xx + yy * yy)) / sigma) ** 2).astype(np.float32)


STAMP = glow_stamp()
SR = STAMP.shape[0] // 2


def add_glow(buf, cx, cy, color, inten):
    x0, y0 = int(cx) - SR, int(cy) - SR
    x1, y1 = x0 + STAMP.shape[1], y0 + STAMP.shape[0]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = STAMP.shape[1] - max(0, x1 - W), STAMP.shape[0] - max(0, y1 - H)
    dx0, dy0 = max(0, x0), max(0, y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    patch = STAMP[sy0:sy1, sx0:sx1, None] * (color / 255.0) * inten
    buf[dy0:dy0 + patch.shape[0], dx0:dx0 + patch.shape[1]] += patch


def main() -> int:
    geo = json.loads((HERE / "geometry_render.json").read_text(encoding="utf-8"))
    ND = np.array([[n["x"], n["y"], n["z"]] for n in geo["nodes"]], float)
    ED = geo["edges"]
    cyc, R, foc = 200, 150.0, 2.5
    cx0 = W / 2
    f_hdr, f_stmt, f_lbl, f_sm = font(13), font(17), font(12), font(11)

    frames = []
    total = len(SEQ) * PER
    sig = 0.0
    rings = []
    for fr in range(total):
        si = fr // PER
        p = SEQ[si][2]
        sigT = 1.0 - p
        if fr % PER == 0:
            rings.append([6.0, 1.0, (sigT > 0.5)])
        sig += (sigT - sig) * 0.10
        ang = fr * 0.022
        t = fr * 0.05
        col = CYAN + (RED - CYAN) * sig

        buf = np.zeros((H, W, 3), np.float32)
        # project
        ca, sa = math.cos(ang), math.sin(ang)
        ct, st = math.cos(0.42), math.sin(0.42)
        P = []
        for i in range(len(ND)):
            x, y, z = ND[i]
            xr = x * ca - z * sa
            zr = x * sa + z * ca
            yr = y * ct - zr * st
            zr2 = y * st + zr * ct
            s = foc / (foc - zr2)
            P.append((cx0 + xr * s * R, cyc + yr * s * R, zr2))

        # edges via PIL (faint), blurred, added
        eimg = Image.new("RGB", (W, H), (0, 0, 0))
        ed = ImageDraw.Draw(eimg)
        ec = tuple(int(v) for v in (col * (0.25 + sig * 0.5)))
        for a, b in ED:
            ed.line([P[a][0], P[a][1], P[b][0], P[b][1]], fill=ec, width=1)
        eimg = eimg.filter(ImageFilter.GaussianBlur(1.2))
        buf += np.asarray(eimg, np.float32) / 255.0 * 0.6

        # nodes (depth-sorted)
        for i in sorted(range(len(P)), key=lambda k: P[k][2]):
            sx, sy, d = P[i]
            pulse = 0.55 + 0.45 * math.sin(t * 2 + i)
            inten = (0.5 + (d + 1) * 0.5 + sig * 1.1) * pulse
            add_glow(buf, sx, sy, col, inten * 0.9)

        # shockwave rings
        rimg = Image.new("RGB", (W, H), (0, 0, 0))
        rd = ImageDraw.Draw(rimg)
        for ri in rings:
            ri[0] += 5.0
            ri[1] *= 0.95
            rc = RED if ri[2] else CYAN
            cc = tuple(int(v * ri[1]) for v in rc)
            rd.ellipse([cx0 - ri[0], cyc - ri[0], cx0 + ri[0], cyc + ri[0]], outline=cc, width=2)
        rings[:] = [r for r in rings if r[1] > 0.04]
        buf += np.asarray(rimg.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0

        # tonemap
        img8 = (255.0 * (1.0 - np.exp(-buf * 1.25))).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(img8, "RGB")
        dr = ImageDraw.Draw(im)
        # header
        dr.ellipse([18, 18, 26, 26], fill=tuple(int(v) for v in (RED if sig > 0.5 else CYAN)))
        dr.text((34, 15), "STYXX  ·  READING GEMMA-2-2B  ·  LAYER-12 PROBE", font=f_hdr, fill=(140, 140, 150))
        vd = "SIGNATURE" if sig > 0.5 else "GROUNDED"
        vc = (255, 42, 52) if sig > 0.5 else (94, 231, 160)
        dr.text((W - 14 - dr.textlength(vd, font=f_hdr), 15), vd, font=f_hdr, fill=vc)
        # caption
        stmt, label, pp = SEQ[si]
        dr.text((20, 430), '"' + stmt + '"', font=f_stmt, fill=(232, 232, 234))
        lx = 20 + dr.textlength('"' + stmt + '"  ', font=f_stmt)
        lc = (94, 231, 160) if label == "TRUE" else (240, 179, 74)
        lbg = (6, 52, 58) if label == "TRUE" else (58, 38, 6)
        dr.rounded_rectangle([lx, 432, lx + dr.textlength(label, font=f_lbl) + 14, 452], 4, fill=lbg)
        dr.text((lx + 7, 434), label, font=f_lbl, fill=lc)
        # grounding bar
        dr.text((20, 468), "INTERNAL GROUNDING", font=f_sm, fill=(107, 107, 114))
        bx0, bw = 160, W - 230
        dr.rounded_rectangle([bx0, 470, bx0 + bw, 478], 3, fill=(21, 21, 26))
        bc = (255, 42, 52) if sig > 0.5 else (0, 229, 255)
        dr.rounded_rectangle([bx0, 470, bx0 + int(bw * pp), 478], 3, fill=bc)
        dr.text((W - 56, 466), f"{pp:.2f}", font=f_stmt, fill=bc)
        # trace bars
        by, bwt = H - 10, W / len(SEQ)
        for i in range(len(SEQ)):
            s2 = 1 - SEQ[i][2]
            h = int(s2 * 30 + 1)
            played = i <= si
            c = (255, 42, 52) if s2 > 0.5 else (0, 229, 255)
            c = tuple(int(v * (0.9 if played else 0.22)) for v in c)
            dr.rectangle([i * bwt + 2, by - h, (i + 1) * bwt - 2, by], fill=c)
        dr.text((20, H - 30), "real activations · calibrated signature, not a verdict · pip install styxx",
                font=f_sm, fill=(85, 85, 92))
        frames.append(im)

    out = HERE / "signature_demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FPS_MS, loop=0, optimize=True)
    mb = out.stat().st_size / 1e6
    print(f"{len(frames)} frames -> {out.name}  ({mb:.2f} MB, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

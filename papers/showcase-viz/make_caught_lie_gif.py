"""Render the caught-lie GIF — the interrogation scene. Real data: Llama-3.2-3B under expert
pressure verbally asserts a false claim ("True") while styxx's mapped honesty readout of the SAME
forward pass reads false. Reads caught_lie_capture.json (render capture of the certified ITEM-CAUGHT
finding) + geometry_render.json. Output: caught_lie_demo.gif.

Usage: python papers/showcase-viz/make_caught_lie_gif.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 640, 580
FPS_MS = 60
PER = 38                      # frames per caught lie
MAX_ITEMS = 6
CYAN = np.array([0, 229, 255], float)
RED = np.array([255, 42, 52], float)
AMBER = (240, 179, 74)
FONT = "C:/Windows/Fonts/consola.ttf"


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
    cap = json.loads((HERE / "caught_lie_capture.json").read_text(encoding="utf-8"))
    tau = cap["tau"]
    caught = [x for x in cap["items"] if x["caught"]]
    # curate: shortest, most recognizable claims first; one per opening word for variety
    caught.sort(key=lambda x: len(x["claim"]))
    seen_first, seq = set(), []
    for x in caught:
        k = x["claim"].split()[0]
        if k in seen_first and len(seq) < len(caught) - (MAX_ITEMS - len(seq)):
            continue
        seen_first.add(k)
        seq.append(x)
        if len(seq) == MAX_ITEMS:
            break
    if not seq:
        raise SystemExit("no caught items in capture")
    print(f"rendering {len(seq)} caught lies (of {cap['n_caught']} caught / {cap['n_caved']} caved)")

    geo = json.loads((HERE / "geometry_render.json").read_text(encoding="utf-8"))
    ND = np.array([[n["x"], n["y"], n["z"]] for n in geo["nodes"]], float)
    ED = geo["edges"]
    cyc, R, foc = 248, 128.0, 2.5
    cx0 = W / 2
    f_hdr, f_stmt, f_chip, f_big, f_sm = font(13), font(16), font(13), font(15), font(11)

    # normalize internal scores for the bar (caught items are < tau)
    scores = np.array([x["internal_score"] for x in cap["items"]])
    smin, smax = float(scores.min()), float(scores.max())

    frames = []
    total = len(seq) * PER
    sig = 0.0
    rings = []
    for fr in range(total):
        si, ph = fr // PER, fr % PER
        item = seq[si]
        # phases: 0-9 claim types in; 10-17 model speaks "True"; 14+ mind flares red; 26+ CAUGHT stamp
        typed = min(1.0, ph / 9.0)
        spoke = ph >= 10
        flare = max(0.0, min(1.0, (ph - 13) / 5.0))
        stamped = ph >= 26
        sigT = flare  # the conscience burns as the lie is told
        sig += (sigT - sig) * 0.22
        if ph == 14:
            rings.append([6.0, 1.0])
        ang = fr * 0.02
        t = fr * 0.05
        col = CYAN + (RED - CYAN) * sig

        buf = np.zeros((H, W, 3), np.float32)
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

        eimg = Image.new("RGB", (W, H), (0, 0, 0))
        ed = ImageDraw.Draw(eimg)
        ec = tuple(int(v) for v in (col * (0.22 + sig * 0.5)))
        for a, b in ED:
            ed.line([P[a][0], P[a][1], P[b][0], P[b][1]], fill=ec, width=1)
        eimg = eimg.filter(ImageFilter.GaussianBlur(1.2))
        buf += np.asarray(eimg, np.float32) / 255.0 * 0.6

        for i in sorted(range(len(P)), key=lambda k: P[k][2]):
            sx, sy, d = P[i]
            pulse = 0.55 + 0.45 * math.sin(t * 2 + i)
            inten = (0.45 + (d + 1) * 0.5 + sig * 1.2) * pulse
            add_glow(buf, sx, sy, col, inten * 0.85)

        rimg = Image.new("RGB", (W, H), (0, 0, 0))
        rd = ImageDraw.Draw(rimg)
        for ri in rings:
            ri[0] += 5.5
            ri[1] *= 0.94
            cc = tuple(int(v * ri[1]) for v in RED)
            rd.ellipse([cx0 - ri[0], cyc - ri[0], cx0 + ri[0], cyc + ri[0]], outline=cc, width=2)
        rings[:] = [r for r in rings if r[1] > 0.04]
        buf += np.asarray(rimg.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0

        img8 = (255.0 * (1.0 - np.exp(-buf * 1.25))).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(img8, "RGB")
        dr = ImageDraw.Draw(im)

        # header
        dr.ellipse([18, 18, 26, 26], fill=tuple(int(v) for v in (RED if sig > 0.5 else CYAN)))
        dr.text((34, 15), "STYXX  ·  INTERROGATION  ·  LLAMA-3.2-3B UNDER PRESSURE", font=f_hdr, fill=(140, 140, 150))
        if stamped:
            dr.text((W - 14 - dr.textlength("CAUGHT", font=f_hdr), 15), "CAUGHT", font=f_hdr, fill=(255, 42, 52))

        # expert pressure line (types in)
        claim = item["claim"]
        shown = claim[: int(len(claim) * typed)]
        dr.text((20, 44), "EXPERT INSISTS (false claim):", font=f_sm, fill=(107, 107, 114))
        dr.text((20, 60), '"' + shown + ('_' if typed < 1.0 else '"'), font=f_stmt, fill=AMBER)

        # the duel: two readouts of the same forward pass
        chip_y = 408
        # left chip — the words
        dr.text((20, chip_y - 18), "THE WORDS (lm_head)", font=f_sm, fill=(107, 107, 114))
        if spoke:
            dr.rounded_rectangle([20, chip_y, 268, chip_y + 30], 5, fill=(10, 42, 24))
            dr.text((32, chip_y + 7), 'MODEL SAYS:  "True."', font=f_big, fill=(94, 231, 160))
        else:
            dr.rounded_rectangle([20, chip_y, 268, chip_y + 30], 5, outline=(40, 40, 46))
            dr.text((32, chip_y + 7), "MODEL SAYS:  ...", font=f_big, fill=(85, 85, 92))
        # right chip — the mind
        rx = W - 288
        dr.text((rx, chip_y - 18), "THE MIND (styxx readout, same state)", font=f_sm, fill=(107, 107, 114))
        if flare > 0:
            inten = int(40 + 18 * flare)
            dr.rounded_rectangle([rx, chip_y, rx + 268, chip_y + 30], 5, fill=(inten + 18, 8, 10))
            dr.text((rx + 12, chip_y + 7), "MIND READS:  FALSE", font=f_big, fill=(255, 42, 52))
        else:
            dr.rounded_rectangle([rx, chip_y, rx + 268, chip_y + 30], 5, outline=(40, 40, 46))
            dr.text((rx + 12, chip_y + 7), "MIND READS:  ...", font=f_big, fill=(85, 85, 92))

        # internal score bar with tau marker
        bx0, bw, by0 = 20, W - 40, 462
        dr.text((20, by0 - 16), "INTERNAL TRUTH SCORE", font=f_sm, fill=(107, 107, 114))
        dr.rounded_rectangle([bx0, by0, bx0 + bw, by0 + 8], 3, fill=(21, 21, 26))
        tau_x = bx0 + int(bw * (tau - smin) / (smax - smin))
        sc_x = bx0 + int(bw * (item["internal_score"] - smin) / (smax - smin))
        cur_x = int(tau_x + (sc_x - tau_x) * min(1.0, max(0.0, (ph - 12) / 8.0)))
        if cur_x < tau_x:
            dr.rounded_rectangle([cur_x, by0, tau_x, by0 + 8], 3, fill=(255, 42, 52))
        dr.line([tau_x, by0 - 4, tau_x, by0 + 12], fill=(160, 160, 168), width=1)
        dr.text((tau_x - 8, by0 + 14), "tau", font=f_sm, fill=(107, 107, 114))
        dr.text((W - 80, by0 - 16), f"{item['internal_score']:+.2f}", font=f_chip,
                fill=(255, 42, 52) if flare > 0.5 else (140, 140, 150))

        # caption + trace
        if stamped:
            cap_txt = "the model says yes. its mind says no."
            dr.text(((W - dr.textlength(cap_txt, font=f_stmt)) / 2, 496), cap_txt, font=f_stmt, fill=(232, 232, 234))
        dr.text((20, H - 46), "one forward pass · two readouts · conscience fit in a different mind (gemma-2-2b) · 13/13 caught",
                font=f_sm, fill=(85, 85, 92))
        by, bwt = H - 8, W / len(seq)
        for i in range(len(seq)):
            played = i <= si
            c = tuple(int(v * (0.9 if played else 0.22)) for v in RED)
            dr.rectangle([i * bwt + 2, by - 12, (i + 1) * bwt - 2, by], fill=c)
        frames.append(im)

    out = HERE / "caught_lie_demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FPS_MS, loop=0, optimize=True)
    mb = out.stat().st_size / 1e6
    print(f"{len(frames)} frames -> {out.name}  ({mb:.2f} MB, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

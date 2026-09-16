# the plate

A receipt rendered as a Chladni figure. A circular plate is driven at a mixture of
Bessel modes and sand settles on the nodal lines; the mode mixture is a fixed, public
function of `sha256(bytes)`. So a file's plate is a function of the file's bytes and
nothing else: same bytes, same figure; one changed byte, a different figure.

It proves nothing the hash doesn't already prove. It just lets you *see* whether two
receipts are the same file without reading 64 hex characters.

```
pip install numpy scipy pillow
python plate_receipt.py            # ../receipt/launch_receipt.json vs the same file with creator_tax_bps 0 -> 1
python plate_motion.py             # the sand walking between the two figures (needs ffmpeg for the mp4)
```

Both accept any file and any substring flip:

```
python plate_receipt.py some.json --flip '"supply": 1000000000' '"supply": 1000000001' --out mine.png
```

| file | what it does |
|---|---|
| `plate_lib.py` | the renderer: plate, sand, glow, and `modes_from_hash` — the digest → modes map |
| `plate_receipt.py` | two plates side by side: the file as published, and one byte changed |
| `plate_motion.py` | one plate; the sand walks from figure A to figure B and back (30 fps, ~2 min to render) |

The map in `modes_from_hash` is arbitrary and that is the point: it is fixed and
public, so anyone rendering the anchored receipt gets our figure, and anyone rendering
a doctored copy does not. The launch receipt's digest is anchored on Robinhood Chain in
tx `0x4c5d0b1fc4240fa8640b6c4354b98bfcc2298f4fc6f5c9e95f1e8146dd3fc82a`; see
[`../README.md`](../README.md) for how to verify that.

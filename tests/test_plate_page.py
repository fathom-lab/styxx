# -*- coding: utf-8 -*-
"""papers/plates/plate.html draws the same plate as styxx.plate — checked by running the page's OWN script
under node against a stub canvas, not by re-implementing it.

Found 2026-09-13: the page mapped canvas row 0 to y = -1 while matplotlib puts y = -1 at the bottom, so
the phone showed the vertical mirror of the lab's plate for every asymmetric figure (the parameters were
always identical). This test would have caught it; it skips where node is absent and says so.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "papers" / "plates" / "plate.html"
DIGEST = "fc8ad3a52de50d106e90c9d0a5441a6460df0f5c80cb5553dad7dbba110cd5a5"   # an asymmetric figure
W, RES = 600, 420

HARNESS = r"""
const fs = require("node:fs");
let captured = null;
const els = {
  d: { value: "", addEventListener() {} },
  cap: { textContent: "" },
  c: { width: %(W)d, height: %(W)d, getContext() { return {
    fillRect() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(4 * w * h) }; },
    putImageData(img) { captured = img.data; } }; } },
};
globalThis.document = { getElementById: (id) => els[id] };
globalThis.location = { search: "" };
globalThis.URLSearchParams = class { get() { return null; } };
(async () => {
%(script)s
  await draw(%(digest)r);
  fs.writeFileSync(%(out)r, Buffer.from(captured.buffer));
  fs.writeFileSync(%(cap)r, els.cap.textContent);
})();
"""


def _page_script() -> str:
    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "no script in the page"
    body = m.group(1)
    # the two DOM lines at the end are replaced by the harness's own call to draw()
    body = "\n".join(l for l in body.splitlines()
                     if not l.startswith("document.getElementById('d').addEventListener")
                     and not l.startswith("const q=new URLSearchParams"))
    return body


def _python_intensity(digest: str) -> np.ndarray:
    """The python plate's line density at each canvas pixel, y UP as matplotlib draws it, using the
    page's own formula on styxx.plate.field's U — the same map the page fills."""
    from styxx import plate
    _, _, U, (pairs, amps, fam, rot, seed) = plate.field(digest, res=RES)
    gy, gx = np.gradient(U, 2.0 / (RES - 1))
    g = np.hypot(gx, gy) + 1e-9
    c, s = math.cos(rot), math.sin(rot)
    col = np.linspace(-1, 1, W)
    row_top = np.linspace(1, -1, W)                        # row 0 is the top of the canvas, y = +1
    RX, RY = np.meshgrid(col, row_top)
    x, y = c * RX + s * RY, -s * RX + c * RY
    ii = np.clip(np.round((x + 1) / 2 * (RES - 1)).astype(int), 1, RES - 2)
    jj = np.clip(np.round((y + 1) / 2 * (RES - 1)).astype(int), 1, RES - 2)
    T = np.exp(-(np.abs(U[jj, ii]) / g[jj, ii] / 0.012) ** 2)
    T[(np.abs(x) > 1) | (np.abs(y) > 1)] = 0.0
    return T


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed; the page cannot be executed here")
def test_the_page_draws_the_python_plate_the_right_way_up(tmp_path):
    out, cap = tmp_path / "rgba.bin", tmp_path / "cap.txt"
    js = HARNESS % {"W": W, "script": _page_script(), "digest": DIGEST, "out": str(out), "cap": str(cap)}
    (tmp_path / "run.js").write_text(js, encoding="utf-8")
    r = subprocess.run(["node", str(tmp_path / "run.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rgba = np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(W, W, 4).astype(float)
    T_js = (rgba[:, :, 0] - 11) / (232 - 11)               # the page writes R = 11 + 221 * t
    T_py = _python_intensity(DIGEST)
    inside = T_py > -1
    assert np.abs(T_js - T_py)[inside].max() < 0.02, "the page and the python plate disagree"
    assert np.abs(T_js - T_py[::-1]).max() > 0.5, "the figure is symmetric; pick an asymmetric digest"
    assert cap.read_text().startswith(DIGEST[:12]) and "plate/v1" in cap.read_text()

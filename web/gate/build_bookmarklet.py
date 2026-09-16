"""Build the bookmarklet from its two sources, byte-for-byte reproducibly.

    python web/gate/build_bookmarklet.py            # writes bookmarklet.min.js + bookmarklet.href.txt
    python web/gate/build_bookmarklet.py --check    # rebuilds and compares against the committed hashes

The bookmarklet is  (function(){ <diffgate.js without its CommonJS export line> <bookmarklet_ui.js> })();
minified with  terser -c -m --format ascii_only  (terser 5.51.2 produced the shipped bytes), then
prefixed with  javascript:  for the href. Nothing else goes in: no analytics, no config, no network
beyond the two api.github.com reads the UI makes. The sha256 of the minified output is the receipt —
whatever a browser holds in its bookmarks bar either hashes to it or is not this build.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT_LINE = 'if (typeof module !== "undefined") module.exports = { gateDiffText, parseUnifiedDiff, parseUnifiedDiffSides };\n'


def source() -> str:
    dg = (HERE / "diffgate.js").read_text(encoding="utf-8")
    assert EXPORT_LINE in dg, "diffgate.js lost its CommonJS export line"
    ui = (HERE / "bookmarklet_ui.js").read_text(encoding="utf-8")
    return "(function(){\n" + dg.replace(EXPORT_LINE, "\n") + "\n" + ui + "\n})();"


def minify(src: str) -> str:
    terser = shutil.which("terser")
    if not terser:
        sys.exit("terser not found: npm install -g terser")
    r = subprocess.run([terser, "-c", "-m", "--format", "ascii_only"], input=src,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        sys.exit(r.stderr)
    return r.stdout.rstrip("\n")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(argv: list[str]) -> int:
    src = source()
    (HERE / "bookmarklet_src.js").write_text(src, encoding="utf-8")
    mini = minify(src)
    href = "javascript:" + mini
    if "--check" in argv:
        old_min = (HERE / "bookmarklet.min.js").read_text(encoding="utf-8")
        old_href = (HERE / "bookmarklet.href.txt").read_text(encoding="utf-8")
        ok = old_min == mini and old_href == href
        print(f"bookmarklet.min.js  sha256 {sha(mini)}  {len(mini)} chars  {'matches' if ok else 'DIFFERS'}")
        return 0 if ok else 1
    (HERE / "bookmarklet.min.js").write_text(mini, encoding="utf-8")
    (HERE / "bookmarklet.href.txt").write_text(href, encoding="utf-8")
    print(f"bookmarklet.min.js   sha256 {sha(mini)}  {len(mini)} chars")
    print(f"bookmarklet.href.txt sha256 {sha(href)}  {len(href)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

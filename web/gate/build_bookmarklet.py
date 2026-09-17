"""Build the bookmarklet from its two sources, byte-for-byte reproducibly.

    python web/gate/build_bookmarklet.py            # writes bookmarklet_src.js, bookmarklet.min.js, bookmarklet.href.txt
    python web/gate/build_bookmarklet.py --check    # rebuilds in memory and compares all three; writes nothing

The bookmarklet is  (function(){ <diffgate.js without its CommonJS export line> <bookmarklet_ui.js> })();
minified with  terser -c -m --format ascii_only  (terser 5.46.0 produced the shipped bytes; the same
terser rebuilds the earlier 4b2d34e1... bookmarklet, which terser 5.51.2 produced, byte for byte), then
prefixed with  javascript:  for the href. Nothing else goes in: no analytics, no config, no network
beyond the two api.github.com reads the UI makes. The sha256 of the minified output is the receipt —
whatever a browser holds in its bookmarks bar either hashes to it or is not this build.

All three outputs are written as bytes with LF line endings, on every platform: `bookmarklet_src.js`
is committed without EOL conversion (it carries NUL bytes, so git reads it as binary), and a text-mode
write on Windows would give it CRLF. `--check` compares against the files on disk and never rewrites
them, so a committed source that drifted from its two inputs is reported, not repaired in passing.
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
    mini = minify(src)
    href = "javascript:" + mini
    outputs = (("bookmarklet_src.js", src), ("bookmarklet.min.js", mini), ("bookmarklet.href.txt", href))
    if "--check" in argv:
        ok = True
        for name, text in outputs:
            path = HERE / name
            same = path.exists() and path.read_bytes() == text.encode("utf-8")
            ok = ok and same
            print(f"{name:<21} sha256 {sha(text)}  {len(text)} chars  {'matches' if same else 'DIFFERS'}")
        return 0 if ok else 1
    for name, text in outputs:
        (HERE / name).write_bytes(text.encode("utf-8"))
    print(f"bookmarklet.min.js   sha256 {sha(mini)}  {len(mini)} chars")
    print(f"bookmarklet.href.txt sha256 {sha(href)}  {len(href)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

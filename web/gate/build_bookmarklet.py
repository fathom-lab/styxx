"""Build the bookmarklet from its two sources, byte-for-byte reproducibly.

    python web/gate/build_bookmarklet.py            # writes bookmarklet_src.js, bookmarklet.min.js, bookmarklet.href.txt
    python web/gate/build_bookmarklet.py --check    # rebuilds in memory and compares all three; writes nothing

The bookmarklet is  (function(){ "use strict"; <the reference> <diffgate.js without its CommonJS export line>
<bookmarklet_ui.js> })();  where <the reference> is web/gate/diffgate_ref.js -- origin/main's web/gate/diffgate.js, byte
for byte, refused unless it hashes to REF_SHA256 -- wrapped in a function scope of its own, with a local `module` its
export line writes to and a local `globalThis` its second export line then leaves alone, and handed to the port as
`_STYXX_REF` (NOTE_path2_eleventh_pass_2026_09_28: the guard's reference). It is then
minified with  terser -c -m --format ascii_only  (terser 5.46.0 produced the shipped bytes; the same
terser rebuilds the earlier 4b2d34e1... bookmarklet, which terser 5.51.2 produced, byte for byte), then
prefixed with  javascript:  for the href. Nothing else goes in: no analytics, no config, no network
beyond the two api.github.com reads the UI makes. The sha256 of the minified output is the receipt —
whatever a browser holds in its bookmarks bar either hashes to it or is not this build.

All three outputs are written as bytes with LF line endings, on every platform, and a text-mode write
on Windows would give them CRLF. `bookmarklet_src.js` is the one of the three git would convert: it is
text. (An earlier cut of this branch's diffgate.js, at ab3084d9, carried two NUL bytes, which made git
read the assembled source as binary and leave it alone; the reconciled diffgate.js carries none.) So
`.gitattributes` marks it `-text` and a checkout keeps the LF bytes the build writes
(NOTE_path2_fourth_pass_2026_09_25, B-1). `--check` compares all three against the files on disk, byte
for byte, and never rewrites them, so a committed source that drifted from its two inputs is reported,
not repaired in passing.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT_LINE = ('if (typeof module !== "undefined") module.exports = { gateDiffText, parseUnifiedDiff, parseUnifiedDiffSides, '
               '_evaluate, _Repairs, REPAIRS, _apartReadings, _precondition, _claimKeys };\n')
# origin/main's web/gate/diffgate.js (2a6ce0a3), the guard's reference, byte for byte (tests/test_diffgate_guard.py pins it too)
REF_SHA256 = "06688702999cdabe763265722a0ac14d4b9ffb40d0efcbb32339eba89f00c141"
REF_OPEN = "const _STYXX_REF = (function () { const module = { exports: null }; const globalThis = undefined;\n"
REF_CLOSE = "\nreturn module.exports; })();\n"


def reference() -> str:
    raw = (HERE / "diffgate_ref.js").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == REF_SHA256, f"diffgate_ref.js hashes to {digest[:16]}, not main's port ({REF_SHA256[:16]})"
    return raw.decode("utf-8")


def source() -> str:
    dg = (HERE / "diffgate.js").read_text(encoding="utf-8")
    assert EXPORT_LINE in dg, "diffgate.js lost its CommonJS export line"
    ui = (HERE / "bookmarklet_ui.js").read_text(encoding="utf-8")
    return ('(function(){\n"use strict";\n' + REF_OPEN + reference() + REF_CLOSE
            + dg.replace(EXPORT_LINE, "\n") + "\n" + ui + "\n})();")


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

"""The harness side of sworn output, for this lab's own documents: mint a turn manifest from a
pytest run, so a RESULT can swear to what the test runner printed rather than to what its author
remembers.

Invariant 2 of the spec: receipts are harness-minted, never author-minted. This script is the
harness. It runs the suite it is told to run, captures what the tools wrote — pytest's stdout,
ruff's stdout, the counts pytest reports — and writes them into a `sworn/manifest/0.2` manifest
as receipts whose `kind_of_source` names where each came from and whose `harness_note` names the
command that printed them. The author of a RESULT then binds sentences to `rN` or `rN#/leaf`; the
verifier resolves them against these bytes and nothing else.

THE RUNG, STATED. This harness declares **L1**: it is a committed script that runs on the same
machine, with the same filesystem and the same shell, as the agent that writes the document. It
records no `authored_sha256`, because a script that runs after the agent has finished cannot see
what the agent wrote during the turn; invariant 2 therefore rests on `kind_of_source` alone for
manifests this harness mints, and the verdict receipt prints `L1` beside every span that rests on
it. L2 — a runner that minted after the turn and that the agent could not write to — is what a CI
job is, and this script is not one.

Also written: a plain JSON test-run result (`<turn>.test_run_result.json`) so `path:` receipts can
bind the same numbers at a commit — the OATH form and the manifest form, side by side. The JSON
receipt (r5) is minted complete, so a numeric span can name a leaf in it with `r5#/passed`.

  python papers/sworn/harness_pytest.py TURN_ID tests/test_sworn.py [more targets]
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.sworn import Manifest       # noqa: E402

RUNG = "L1"


def run(*args: str) -> bytes:
    r = subprocess.run([sys.executable, "-m", *args], cwd=str(ROOT), capture_output=True, check=False)
    return r.stdout + r.stderr


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    turn, targets = argv[0], argv[1:]
    manifest = Manifest(harness="papers/sworn/harness_pytest.py", turn=turn, rung=RUNG)

    pytest_cmd = ["pytest", "-q", "--no-header", "-p", "no:cacheprovider", *targets]
    ruff_cmd = ["ruff", "check", "styxx/sworn.py", *[t for t in targets if t.endswith(".py")]]
    pytest_out = run(*pytest_cmd)
    ruff_out = run(*ruff_cmd)
    summary = pytest_out.decode("utf-8", errors="replace").strip().splitlines()[-1] if pytest_out else ""
    passed = int((re.search(r"(\d+) passed", summary) or [None, "0"])[1])
    failed = int((re.search(r"(\d+) failed", summary) or [None, "0"])[1])

    sworn_src = (ROOT / "styxx" / "sworn.py").read_bytes()
    result = {
        "what": "sworn test run, recorded by papers/sworn/harness_pytest.py at rung %s" % RUNG,
        "turn": turn,
        "targets": targets,
        "passed": passed,
        "failed": failed,
        "pytest_summary_line": summary,
        "ruff_clean": ruff_out.strip() == b"All checks passed!",
        "sworn_sha256": hashlib.sha256(sworn_src).hexdigest(),
        "sworn_lines": sworn_src.count(b"\n"),
        "python": sys.version.split()[0],
        "rung": RUNG,
    }
    result_bytes = (json.dumps(result, indent=2) + "\n").encode("utf-8")

    # r1: the passed count, as pytest printed it, minted as a one-number capture (kept for
    #     documents written before rN#/leaf existed).
    # r2: pytest's whole stdout+stderr, complete, for quote/absent.
    # r3: ruff's whole output, complete, for quote/absent.
    # r4: the failed count.
    # r5: the JSON result above, complete — `r5#/passed`, `r5#/sworn_lines` and so on.
    manifest.add("r1", str(passed).encode("ascii"), "test_report", complete=True,
                 note="passed count parsed from: python -m " + " ".join(pytest_cmd))
    manifest.add("r2", pytest_out, "tool_stdout", complete=True, note="python -m " + " ".join(pytest_cmd))
    manifest.add("r3", ruff_out, "tool_stdout", complete=True, note="python -m " + " ".join(ruff_cmd))
    manifest.add("r4", str(failed).encode("ascii"), "test_report", complete=True,
                 note="failed count parsed from: python -m " + " ".join(pytest_cmd))
    manifest.add("r5", result_bytes, "test_report", complete=True,
                 note="the test-run result JSON this harness wrote, byte for byte")

    # newline="\n": a Windows text-mode write would CRLF the receipt and it would hash
    # differently per platform. Both files are byte-pinned (.gitattributes papers/sworn/** -text).
    with open(HERE / ("%s.test_run_result.json" % turn), "wb") as fh:
        fh.write(result_bytes)
    mpath = HERE / ("%s.manifest.json" % turn)
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest.to_dict(), indent=1, ensure_ascii=False) + "\n")
    print("harness[%s]: %s -> passed=%d failed=%d ruff_clean=%s  receipts=%d  sworn.py sha256 %s"
          % (RUNG, turn, passed, failed, result["ruff_clean"], len(manifest.receipts),
             result["sworn_sha256"][:12]))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

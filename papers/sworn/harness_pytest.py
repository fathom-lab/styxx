"""The harness side of sworn output, for this lab's own documents: mint a turn manifest from a
pytest run, so a RESULT can swear to what the test runner printed rather than to what its author
remembers.

Invariant 2 of the spec: receipts are harness-minted, never author-minted. This script is the
harness. It runs the suite it is told to run, captures what the tools wrote — pytest's stdout,
ruff's stdout, the counts pytest reports — and writes them into a `sworn/manifest/0.1` manifest
as receipts whose `kind_of_source` names where each came from. The author of a RESULT then binds
sentences to `rN`; the verifier resolves them against these bytes and nothing else.

The honest boundary, stated here and carried in every verdict: this harness is a committed
script anyone can re-run, and the manifest it writes is exactly as trustworthy as that. It records
no `authored_sha256`, because a script that ran after the agent finished cannot see what the agent
wrote during the turn; invariant 2 therefore rests on `kind_of_source` alone for manifests this
harness mints, and the sworn receipt says so.

Also written: a plain JSON test-run result (`<turn>.test_run_result.json`) so `path:` receipts can
bind the same numbers at a commit — the OATH form and the manifest form, side by side.

  python papers/sworn/harness_pytest.py TURN_ID tests/test_sworn.py
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


def run(*args: str) -> bytes:
    r = subprocess.run([sys.executable, "-m", *args], cwd=str(ROOT), capture_output=True, check=False)
    return r.stdout + r.stderr


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    turn, targets = argv[0], argv[1:]
    manifest = Manifest(harness="papers/sworn/harness_pytest.py", turn=turn)

    pytest_out = run("pytest", "-q", "--no-header", "-p", "no:cacheprovider", *targets)
    ruff_out = run("ruff", "check", "styxx/sworn.py", *[t for t in targets if t.endswith(".py")])
    summary = pytest_out.decode("utf-8", errors="replace").strip().splitlines()[-1] if pytest_out else ""
    passed = int((re.search(r"(\d+) passed", summary) or [None, "0"])[1])
    failed = int((re.search(r"(\d+) failed", summary) or [None, "0"])[1])

    # r1: the passed count, as pytest printed it, minted as a one-number capture (the shape a
    #     numeric span against an rN receipt needs — the receipt IS the scalar).
    # r2: pytest's whole stdout+stderr, complete, for quote/absent.
    # r3: ruff's whole output, complete, for quote/absent.
    # r4: the failed count.
    manifest.add("r1", str(passed).encode("ascii"), "test_report", complete=True)
    manifest.add("r2", pytest_out, "tool_stdout", complete=True)
    manifest.add("r3", ruff_out, "tool_stdout", complete=True)
    manifest.add("r4", str(failed).encode("ascii"), "test_report", complete=True)

    sworn_src = (ROOT / "styxx" / "sworn.py").read_bytes()
    result = {
        "what": "sworn v0.1 test run, recorded by papers/sworn/harness_pytest.py",
        "turn": turn,
        "targets": targets,
        "passed": passed,
        "failed": failed,
        "pytest_summary_line": summary,
        "ruff_clean": ruff_out.strip() == b"All checks passed!",
        "sworn_sha256": hashlib.sha256(sworn_src).hexdigest(),
        "sworn_lines": sworn_src.count(b"\n"),
        "python": sys.version.split()[0],
    }
    (HERE / ("%s.test_run_result.json" % turn)).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest.write(HERE / ("%s.manifest.json" % turn))
    print("harness: %s -> passed=%d failed=%d ruff_clean=%s  receipts=%d  sworn.py sha256 %s"
          % (turn, passed, failed, result["ruff_clean"], len(manifest.receipts),
             result["sworn_sha256"][:12]))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

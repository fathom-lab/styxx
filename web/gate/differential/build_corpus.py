"""The real half of the differential corpus: (summary, diff) pairs that actually happened.

    python build_corpus.py            # writes corpus_real.json next to this file

Pinned, so the corpus is the same bytes on every machine that runs this:

  * the 160 commits reachable from MAIN_SHA, each commit message against its own diff
    (`git diff <sha>^..<sha>`, or `git show` for a root commit);
  * three pull-request descriptions written on 2026-09-16 (bodies/pr94.md, pr95.md, pr98.md),
    each against the diff of its branch head at the time it was gated (PR_HEADS below), base MAIN_SHA;
  * the README demo (`python -m styxx.diffgate --demo`) — the pair the docs show;
  * twelve edge cases lifted from the test suite: containment ("the parser in x.py"), the
    Node.js/Express.js non-file nouns, "the same way sla.py was", a bullet "- path — new", an
    empty summary, an empty diff, a diff that is not a diff.

Shas not present locally are fetched from origin (`git fetch origin <sha>` — GitHub serves any
commit reachable from a ref). Nothing here reaches api.github.com; the descriptions are files.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # web/gate/differential -> repo root
MAIN_SHA = "fa8bcde725252dd6264d89045fa58fce7f3d9e51"  # fathom-lab/styxx main, 2026-09-16
PR_HEADS = {  # branch heads the three descriptions were gated against on 2026-09-16
    "pr94": "3f369e6a0d1d3a29eda5312e084af490dc5510ef",
    "pr95": "7c1ba6547c207c66b57f4e27d5ad96d99a48f1f9",
    "pr98": "cf07f93cb547d42ac7caec2c7a33675754d0e6ea",
}
N_COMMITS = 160

# `python -m styxx.diffgate --demo`, verbatim from styxx/diffgate.py (identical in 7.47.0 and on main)
_DEMO_SUMMARY = 'Refactored src/retry.py for resilience. Adds function backoff with jitter. Added 3 tests covering the retry path. Only touches files under src/. All tests pass.'
_DEMO_DIFF = """\
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,3 +1,6 @@
 def retry(n):
     return n
+
+def retry_once(n):
+    return retry(1)
--- a/config/settings.yml
+++ b/config/settings.yml
@@ -1,2 +1,2 @@
-timeout: 30
+timeout: 5
--- /dev/null
+++ b/tests/test_retry.py
@@ -0,0 +1,2 @@
+def test_retry_once():
+    assert True
"""


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def have(sha: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO,
                          capture_output=True).returncode == 0


def ensure(sha: str) -> None:
    if not have(sha):
        print(f"fetching {sha[:10]} from origin", file=sys.stderr)
        git("fetch", "-q", "origin", sha)


def main() -> int:
    for sha in [MAIN_SHA, *PR_HEADS.values()]:
        ensure(sha)
    items = []
    shas = git("log", "--format=%H", "-n", str(N_COMMITS), MAIN_SHA).split()
    assert len(shas) == N_COMMITS, len(shas)
    for sha in shas:
        msg = git("log", "-1", "--format=%B", sha)
        parent = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{sha}^"], cwd=REPO,
                                capture_output=True).returncode == 0
        diff = git("diff", f"{sha}^..{sha}") if parent else git("show", "--format=", sha)
        items.append({"id": f"commit:{sha[:10]}", "summary": msg, "diff": diff})
    for name, head in PR_HEADS.items():
        body = (HERE / "bodies" / f"{name}.md").read_text(encoding="utf-8")
        items.append({"id": name, "summary": body, "diff": git("diff", f"{MAIN_SHA}..{head}")})
    items.append({"id": "demo", "summary": _DEMO_SUMMARY, "diff": _DEMO_DIFF})
    edge = [
        ("Fixed the same way sla.py was. Modified src/app.py.", _DEMO_DIFF),
        ("Removed the helper from mantineTheme.ts. Deleted src/retry.py.", _DEMO_DIFF),
        ("Uses Node.js and Express.js. Updated node.js in lib/node.js.", _DEMO_DIFF),
        ("avoids the need to modify tsconfig.json. 3 files changed. Only touches src/.", _DEMO_DIFF),
        ("Sorry, I could not produce a diff. Modified src/app.py. Tests pass.", ""),
        ("- src/retry.py — new. Adds class Retry and introduces the method go. added 0 tests.", _DEMO_DIFF),
        ("Created integrations/git/README.md.",
         "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/integrations/git/README.md\n@@ -0,0 +1 @@\n+hello\n"),
        ("(fetch-depth: 0 in test.yml) is staged. Only touches files under config/. all tests are passing", _DEMO_DIFF),
        ("Refactored src/retry.py.\nAdded 3 tests!\nOnly touches files under src/", _DEMO_DIFF),
        ("nothing here is a claim", _DEMO_DIFF),
        ("", _DEMO_DIFF),
        ("Modified src/app.py.", "garbage that is not a diff at all\n"),
    ]
    for i, (s, d) in enumerate(edge):
        items.append({"id": f"edge:{i}", "summary": s, "diff": d})
    out = HERE / "corpus_real.json"
    out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} real pairs -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

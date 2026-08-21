# -*- coding: utf-8 -*-
"""SP-EXT candidate harvest. Queries frozen at 38b8428 before any commit was read.

Q1  commit message regex  -- finds fixes whose AUTHOR described them this way
Q2  diff shape (pickaxe)  -- finds a removed flattering constant beside an added
                             raise / NaN / None / warn / skip

Both are run verbatim from the preregistration. Nothing is added because it
looked interesting (G4). Recall is unknown and stated as unknown (G5): these find
commits that were *described* or *shaped* a particular way, and nothing else.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/heyzo/AppData/Local/Temp/spcorpus")
OUT = Path(__file__).resolve().parent.parent / "papers" / "out_sp_ext_candidates.json"

Q1 = [
    r"silent(ly)?",
    r"always (return|pass|true|zero)",
    r"never (fire|ran|run|check|trigger)",
    r"was not (check|validat|measur)",
    r"return(s|ed)? (0|0\.0|True|\"pass\") when",
    r"empty (list|input|dict|sequence|array)",
    r"no data",
    r"fail(s|ed|ing)? (open|closed)",
    r"swallow",
    r"missing (data|value|logprob)",
    r"default(s|ed)? to (0|true|pass)",
    r"false negative",
    r"undetected",
    r"unreported",
    r"treated as (valid|success|pass)",
    r"divide by zero",
    r"division by zero",
    r"skipped .* report",
    r"vacuous",
]

# Q2 pickaxe strings: a flattering constant whose occurrence count CHANGED.
Q2_STRINGS = ["return 0.0", "return True", "return 0", "return 1.0",
              'return "pass"', "return \"ok\"", "return []"]
Q2_LOUD = ("raise", "nan", "None", "warn", "skip", "log.warn", "log.error",
           "logger.warning", "pytest.skip")


def git(repo: Path, *args, timeout=300) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def q1(repo: Path) -> set[str]:
    hits: set[str] = set()
    for pat in Q1:
        out = git(repo, "log", "--all", "-i", "-E", f"--grep={pat}",
                  "--format=%H", "--no-merges")
        hits.update(out.split())
    return hits


def q2(repo: Path, cap: int = 400) -> set[str]:
    hits: set[str] = set()
    for s in Q2_STRINGS:
        out = git(repo, "log", "--all", "--no-merges", f"-S{s}", "--format=%H",
                  f"-n{cap}", timeout=900)
        hits.update(out.split())
    return hits


def meta(repo: Path, sha: str) -> dict:
    raw = git(repo, "show", "-s", "--format=%H%x00%an%x00%ad%x00%s%x00%b", sha)
    parts = raw.split("\x00")
    if len(parts) < 4:
        return {}
    return {"sha": parts[0], "author": parts[1], "date": parts[2],
            "subject": parts[3], "body": (parts[4] if len(parts) > 4 else "")[:1200]}


def diffstat(repo: Path, sha: str) -> tuple[int, int, list[str]]:
    st = git(repo, "show", "--stat", "--format=", sha, timeout=180)
    files = [l.split("|")[0].strip() for l in st.splitlines() if "|" in l]
    py = [f for f in files if f.endswith(".py")]
    return len(files), len(py), py[:8]


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    repos = sorted(d for d in ROOT.iterdir() if (d / ".git").exists())
    if only:
        repos = [d for d in repos if d.name == only]

    all_c, tally = [], Counter()
    for r in repos:
        head = git(r, "rev-parse", "HEAD").strip()[:12]
        a = q1(r)
        print(f"  {r.name:24} HEAD {head}  Q1 {len(a):4d}", flush=True)
        for sha in a:
            m = meta(r, sha)
            if not m:
                continue
            nf, npy, pyf = diffstat(r, sha)
            if npy == 0 or nf > 40:        # python only; giant commits are not cases
                continue
            m.update({"repo": r.name, "query": "Q1", "n_files": nf,
                      "py_files": pyf, "head": head})
            all_c.append(m)
            tally[r.name] += 1
    print(f"\n  Q1 candidates after python/size filter: {len(all_c)}")
    OUT.write_text(json.dumps({"queries": {"Q1": Q1, "Q2_strings": Q2_STRINGS},
                               "candidates": all_c,
                               "by_repo": dict(tally)}, indent=1),
                   encoding="utf-8")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

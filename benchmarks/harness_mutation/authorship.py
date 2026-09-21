# -*- coding: utf-8 -*-
"""SWALLOW-8 -- who writes the hidden check.

SWALLOW-7 ran the gate at 21,569 mainline commits and found 104 that bring a check hiding its own
failure. This module asks who made those commits, by a stated rule and nothing else: every
first-parent commit touching a hand-written workflow is read for its author name and address,
its subject and its body, and put in one of three classes --

  agent        a coding agent's signature: an agent's name as the author, a `Co-authored-by`
               trailer naming one, a merge of a branch under an agent's prefix (`codex/`,
               `claude/`, `copilot/`, ...), "Generated with Claude Code", "[CI] Agentic workflows"
  automation   a bot that is not a coding agent: dependabot, renovate, github-actions, release
               and pre-commit bots, any `[bot]` author, with no agent signal
  human        everything else -- which includes every agent-assisted commit that left no
               signature, so the agent class is a floor, not a count

-- and joined to the SWALLOW-7 receipt: which commits fired, how many checks they brought, and
whether a verified repair was there. No name or address is written to the receipt: a commit
carries its sha, its time, its class, and the label of the signal that classed it.

    python -m benchmarks.harness_mutation.authorship --receipt papers/harness/swallow7_receipt.json.gz \
        --work <clones> --out papers/harness/swallow8_receipt.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.harness-authorship/v1"
CLASSES = ("agent", "automation", "human")

# the agent signals, in the order they are tried; the label is what the receipt keeps. A bare first name is not a
# signal: "Claude" and "Devin" are people's names, so an author name counts only in an agent's own form.
AGENT_AUTHOR_NAME = re.compile(r"^copilot$|copilot-swe-agent|\bcopilot\[bot\]|claude\[bot\]|claude[ -]code|devin-ai|devin\[bot\]|openhands|sweep-ai|"
                               r"coderabbit|google-labs-jules|jules\[bot\]|cursor-agent|cursoragent|codex\[bot\]|codegen-sh|gemini-code-assist|"
                               r"amazon-q|qodo|blackbox|gh-aw\[bot\]", re.I)
AGENT_AUTHOR_EMAIL = re.compile(r"noreply@anthropic\.com|copilot@|copilot-swe-agent|devin-ai-integration|openhands|sweep-ai|coderabbitai|"
                                r"google-labs-jules|cursoragent|cursor-agent|codex@|codegen-sh|gemini-code-assist|qodo-|blackboxai", re.I)
AGENT_TRAILER = re.compile(r"^co-authored-by:.*(?:noreply@anthropic\.com|\bcopilot\b|\bcodex\b|\bcursor\b|devin-ai|devin\[bot\]|openhands|"
                           r"\bjules\b|\bgemini\b|coderabbit|claude(?: code|\[bot\]|-code| opus| sonnet| haiku)|sweep-ai|amazon-q|qodo|blackbox)", re.I | re.M)
AGENT_GENERATED = re.compile(r"generated (?:with|by) \[?(?:claude code|claude|copilot|codex|cursor|devin|aider|gemini|jules)\b|"
                             r"\bmade with (?:cursor|claude|copilot|codex|devin)\b|\[CI\] Agentic workflows|\bagentic workflows?\b", re.I)
AGENT_BRANCH = re.compile(r"(?:^|[ /])(?:codex|claude|copilot|cursor|devin|aider|jules|gemini|sweep|openhands)[/\-]", re.I)
AGENT_SIGNALS = (
    ("author-name", AGENT_AUTHOR_NAME, "author_name"),
    ("author-email", AGENT_AUTHOR_EMAIL, "author_email"),
    ("co-authored-by", AGENT_TRAILER, "body"),
    ("generated-with", AGENT_GENERATED, "body"),
    ("generated-with", AGENT_GENERATED, "subject"),
    ("branch-prefix", AGENT_BRANCH, "subject"),
)
AUTOMATION = re.compile(r"\[bot\]|dependabot|renovate|github-actions|pre-commit-ci|snyk|greenkeeper|allcontributors|semantic-release|release-please|"
                        r"actions-user|mergify|imgbot|whitesource|codecov|netlify|vercel|azure-pipelines|crowdin|weblate|transifex|kodiak|bors", re.I)


# ----------------------------------------------------------------------------- the rule

def classify(author_name: str, author_email: str, subject: str, body: str) -> tuple[str, str | None]:
    """The class of one commit and the label of the signal that decided it (None for human)."""
    fields = {"author_name": author_name or "", "author_email": author_email or "", "subject": subject or "", "body": body or ""}
    for label, rx, field in AGENT_SIGNALS:
        if rx.search(fields[field]):
            return "agent", label
    if AUTOMATION.search(fields["author_name"]) or AUTOMATION.search(fields["author_email"]):
        return "automation", "bot-author"
    return "human", None


# ----------------------------------------------------------------------------- git

def commit_classes(clone: Path, tip: str) -> dict[str, dict]:
    """Every first-parent commit touching .github/workflows up to `tip`: its class, signal and time."""
    raw = subprocess.run(["git", "-C", str(clone), "log", "--first-parent", "--format=%x01%H%x00%ct%x00%an%x00%ae%x00%s%x00%b%x02", tip, "--", ".github/workflows"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900).stdout
    out = {}
    for chunk in raw.split("\x01")[1:]:
        head = chunk.split("\x02")[0]
        sha, ct, an, ae, subject, body = (head.split("\x00") + [""] * 6)[:6]
        cls, signal = classify(an, ae, subject.strip(), body)
        out[sha] = {"class": cls, "signal": signal, "time": int(ct or 0)}
    return out


# ----------------------------------------------------------------------------- the join

def join_repo(rec7: dict, classes: dict[str, dict]) -> dict:
    """The SWALLOW-7 commits of one repository with their classes; nothing but sha, time, class, signal, and the gate's counts."""
    out = {"repo": rec7["repo"], "tip": rec7["tip"], "commits": [], "unclassified": 0}
    for c in rec7["commits"]:
        k = classes.get(c["sha"])
        if k is None:
            out["unclassified"] += 1
        row = {"sha": c["sha"], "time": c["time"], "root": bool(c.get("root")), "fires": c["fires"], "new_hidden": c["new_hidden"],
               "removed_hidden": c["removed_hidden"], "class": k["class"] if k else None, "signal": k["signal"] if k else None}
        if c.get("detail"):
            row["checks"] = [{"kind": x["kind"], "verdict": x["verdict"], "continue_on_error": x.get("continue_on_error"),
                              "repair": (x.get("fix") or {}).get("verified_repair"), "lines": (x.get("fix") or {}).get("lines_changed")}
                             for d in c["detail"] for x in d.get("new_hidden", [])]
        out["commits"].append(row)
    return out


def _year(t: int) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y") if t else "?"


def summary(rec: dict) -> dict:
    rows = [(r["repo"], c) for r in rec["repos"] for c in r["commits"] if not c["root"]]
    by: dict = {}
    for cls in CLASSES:
        cs = [c for _, c in rows if c["class"] == cls]
        fires = [c for c in cs if c["fires"]]
        checks = [x for c in fires for x in c.get("checks", [])]
        by[cls] = {"commits": len(cs), "firing_commits": len(fires), "fire_rate": round(len(fires) / len(cs), 5) if cs else None,
                   "new_hidden_checks": len(checks), "checks_per_firing": round(len(checks) / len(fires), 2) if fires else None,
                   "with_repair": sum(1 for x in checks if x["repair"]), "born_hidden": sum(1 for x in checks if x["kind"] == "born hidden"),
                   "continue_on_error": sum(1 for x in checks if x.get("continue_on_error")),
                   "signals": _count(c["signal"] for c in cs if c["signal"]),
                   "repos": len({repo for repo, c in rows if c["class"] == cls})}
    years: dict = {}
    for _, c in rows:
        y = _year(c["time"])
        years.setdefault(y, {k: 0 for k in CLASSES + ("firing",)})
        if c["class"]:
            years[y][c["class"]] += 1
        if c["fires"]:
            years[y]["firing"] += 1
    return {"commits": len(rows), "unclassified": sum(r["unclassified"] for r in rec["repos"]), "by_class": by,
            "by_year": {y: years[y] for y in sorted(years)}, "repos": len(rec["repos"])}


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def population(receipt7: dict, work: Path, source_sha256: str | None = None) -> dict:
    t0 = time.time()
    rec = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/authorship.py", "instrument_sha256": instrument_sha256(),
           "source_receipt": "papers/harness/swallow7_receipt.json.gz", "source_receipt_sha256": source_sha256,
           "differential_sha256": receipt7.get("instrument_sha256"), "repos": [], "missing_clones": []}
    for r7 in receipt7["repos"]:
        clone = work / r7["repo"].replace("/", "__")
        if not (clone / ".git").exists():
            rec["missing_clones"].append(r7["repo"])
            continue
        classes = commit_classes(clone, r7["tip"])
        rec["repos"].append(join_repo(r7, classes))
    rec["seconds"] = round(time.time() - t0, 1)
    rec["summary"] = summary(rec)
    return rec


def instrument_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipt", required=True, help="the SWALLOW-7 receipt (.json or .json.gz)")
    ap.add_argument("--work", default="s6hist", help="the SWALLOW-6 clones")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    raw = Path(a.receipt).read_bytes()
    receipt7 = json.loads((gzip.decompress(raw) if a.receipt.endswith(".gz") else raw).decode("utf-8"))
    rec = population(receipt7, Path(a.work), hashlib.sha256(raw).hexdigest())
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

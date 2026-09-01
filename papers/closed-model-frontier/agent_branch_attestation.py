"""Audit the agent's own reports: every commit message on this branch, gated against its diff.

The claim this lane keeps making is that agent output should be attestable — *"styxx verifies what
an AI agent says it changed against what actually changed."* The most meaningful corpus available
to test that claim is this branch itself: dozens of commits authored by an AI agent, each carrying
a message full of concrete assertions ("N new tests", "file X rewritten", "counts unchanged"),
each sitting directly on top of the diff that would confirm or refute it.

So: for every commit on `main..HEAD`, run `styxx.diffgate.gate_diff(message, repo, parent,
commit)` and fold the verdicts. Three bands, exactly as the OATH work taught:

* **VERIFIED** — the claim matched the diff.
* **CONTRADICTED** — the diff says otherwise. Every one is listed, none summarised away.
* **UNCHECKABLE** — the claim names evidence the diff does not carry ("2564 passed" needs a test
  run, not a diff). The named third band, not a silent omission.

Honesty notes, stated before the numbers:

* diffgate's extractor has a CATALOGUED mention-vs-use defect (see
  `SYNTHESIS_mention_and_use_2026_08_26.md`): a filename *referred to* can be read as a filename
  *claimed*. Its `_REFERENTIAL` guard is a closed list. So CONTRADICTED entries here are LEADS,
  each requiring a human read of the message — this harness adjudicates nothing.
* Merge commits are gated against their first parent.
* The gate reads ONLY the commit message and the diff. Test counts, CI results, and panel numbers
  in messages are exactly the class the third band exists for.

  python papers/closed-model-frontier/agent_branch_attestation.py [BASE] [HEAD]
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.diffgate import gate_diff                                    # noqa: E402

OUT = HERE / "agent_branch_attestation.json"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    shas = git("rev-list", "--reverse", f"{base}..{head}").split()
    print(f"gating {len(shas)} commits on {base}..{head}\n")

    tot = collections.Counter()
    uncovered_total = 0
    per_commit = []
    contradictions = []
    unmeasured = 0
    for sha in shas:
        msg = git("log", "-1", "--format=%B", sha)
        subject = git("log", "-1", "--format=%s", sha).strip()
        g = gate_diff(msg, ROOT, f"{sha}^", sha)
        if not g.measured:
            unmeasured += 1
        counts = collections.Counter(c.verdict for c in g.claims)
        tot.update(counts)
        uncovered_total += g.uncovered_sentences
        rec = {"sha": sha[:9], "subject": subject[:88],
               "claims": len(g.claims), "verified": counts["VERIFIED"],
               "contradicted": counts["CONTRADICTED"], "uncheckable": counts["UNCHECKABLE"],
               "uncovered_sentences": g.uncovered_sentences,
               "gate": g.verdict, "measured": g.measured}
        per_commit.append(rec)
        for c in g.claims:
            if c.verdict == "CONTRADICTED":
                contradictions.append({"sha": sha[:9], "subject": subject[:70],
                                       "kind": c.kind, "text": c.text[:140], "why": c.why[:160]})
        mark = "!" if counts["CONTRADICTED"] else " "
        print(f" {mark} {sha[:9]}  V{counts['VERIFIED']:>3} C{counts['CONTRADICTED']:>2} "
              f"U{counts['UNCHECKABLE']:>3}  {subject[:74]}")

    n_claims = sum(tot.values())
    payload = {
        "attestation": "every commit message on this branch, gated against its own diff",
        "agent": "the AI agent that authored this branch (and this file, and the gate)",
        "range": {"base": base, "head": head, "commits": len(shas)},
        "totals": {
            "claims_extracted": n_claims,
            "verified": tot["VERIFIED"],
            "contradicted": tot["CONTRADICTED"],
            "outside_evidence": tot["UNCHECKABLE"],
            "verified_share": round(tot["VERIFIED"] / n_claims, 4) if n_claims else None,
            "outside_evidence_share": round(tot["UNCHECKABLE"] / n_claims, 4)
            if n_claims else None,
            "gates_unmeasured": unmeasured,
            "sentences_never_read": uncovered_total,
        },
        "contradictions_every_one": contradictions,
        "per_commit": per_commit,
        "boundary_note": (
            "OUTSIDE_EVIDENCE is the honest band, not a failure: test totals, CI verdicts and "
            "panel figures in commit messages cannot be checked against a diff, and this gate "
            "says so instead of pretending. CONTRADICTED entries are leads from an extractor "
            "with a catalogued mention-vs-use defect; each needs a human read before being "
            "called a false report."),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    t = payload["totals"]
    print(f"\nclaims {t['claims_extracted']}  verified {t['verified']} "
          f"({t['verified_share']})  contradicted {t['contradicted']}  "
          f"outside-evidence {t['outside_evidence']} ({t['outside_evidence_share']})")
    if contradictions:
        print("\nevery contradiction (leads, not verdicts):")
        for c in contradictions[:12]:
            print(f"  {c['sha']}  [{c['kind']}] {c['text'][:96]}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

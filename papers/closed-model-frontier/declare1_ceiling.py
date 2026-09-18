"""DECLARE-1 G-D1-2 / G-D1-3: what the format buys when adoption is perfect, and how hard the
claims it carries actually are.

This is a CEILING, not a result about the world. A declaration block is synthesized for each
decidable BENCH-2 item from the ORACLE's own facts, so every declaration here is by construction
honest and complete. Nobody's real pull request carries one. The number that comes out is the best
the format could possibly do, and the RESULT says so in those words.

    python declare1_ceiling.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.diffgate import gate_diff_text  # noqa: E402

DIFFS = Path("/tmp/bench1_diffs")
DATASET = HERE / "bench2_dataset.jsonl"

#: claim kind -> the declaration key that carries it, and how to get the value the PULL REQUEST
#: STATED. Not the oracle's true value: declaring the truth and then verifying it would measure
#: nothing but arithmetic. The question is whether an agent that declared what it actually claimed
#: would have been read correctly, so the stated value is what goes in the block.
CARRIER = {
    "files_changed_count": ("files_changed", lambda det: det.get("n")),
    "only_touches": ("only_touches", lambda det: det.get("prefix")),
    "symbol_added": ("adds_symbol", lambda det: det.get("name") or det.get("symbol")),
}


def main() -> int:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith('{"_header"')]
    out = {"note": ("a CEILING under perfect adoption: every pull request here is assumed to have "
                    "declared the SAME claim it made in prose, correctly formatted. No real pull "
                    "request carries a block. This measures whether the format removes the "
                    "EXTRACTION failure, which is all it was designed to remove."),
           "per_kind": {}}
    for kind, (key, get) in CARRIER.items():
        items = [r for r in rows if r["kind"] == kind and r["truth"] in ("CONTRADICTED", "SUPPORTED")]
        stats = Counter()
        prose_said = Counter()
        for r in items:
            p = DIFFS / f"{r['pr_id']}.diff"
            if not p.exists():
                continue
            diff = p.read_text(encoding="utf-8", errors="replace")
            value = get(r.get("claim_detail") or {})
            if value in (None, ""):
                stats["no_value_to_declare"] += 1
                continue
            # G-D1-3: what the PROSE reader said about this same claim, before any declaration.
            pg = gate_diff_text(r["claim_text"], diff, run=None, strict=False)
            phit = next((c for c in pg.claims if c.kind == kind), None)
            prose_said[phit.verdict if phit else "NOT_EXTRACTED"] += 1

            block = "```styxx\n%s: %s\n```" % (key, value)
            dg = gate_diff_text(block, diff, run=None, strict=False)
            dhit = next((c for c in dg.claims
                         if c.kind == kind and (c.detail or {}).get("declared")), None)
            if dhit is None:
                stats["declaration_not_read"] += 1
                continue
            # The oracle's label is truth. Did the declared reading agree with it?
            agree = ((dhit.verdict == "CONTRADICTED" and r["truth"] == "CONTRADICTED")
                     or (dhit.verdict == "VERIFIED" and r["truth"] == "SUPPORTED"))
            stats["read" if dhit.verdict in ("VERIFIED", "CONTRADICTED") else "abstained"] += 1
            if dhit.verdict in ("VERIFIED", "CONTRADICTED"):
                stats["agrees_with_oracle" if agree else "disagrees_with_oracle"] += 1
        decided = stats["read"]
        out["per_kind"][kind] = {
            "decidable_items": len(items),
            "declared_and_read": decided,
            "declared_and_abstained": stats["abstained"],
            "agrees_with_oracle": stats["agrees_with_oracle"],
            "disagrees_with_oracle": stats["disagrees_with_oracle"],
            "no_value_to_declare": stats["no_value_to_declare"],
            "coverage_when_declared": round(decided / len(items), 4) if items else None,
            "prose_reader_said_on_the_same_claims": dict(prose_said),
            # G-D1-3: a format that only carries claims the old reader already handled buys nothing.
            "share_the_prose_reader_would_have_abstained_on":
                round((prose_said["UNCHECKABLE"] + prose_said["NOT_EXTRACTED"]) / sum(prose_said.values()), 4)
                if prose_said else None,
        }
    (HERE / "declare1_ceiling.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

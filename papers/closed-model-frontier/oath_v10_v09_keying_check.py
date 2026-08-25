"""How many ledger rows did the v0.9 severability baseline's key format silently merge?

`make_oath_v09_baseline.py` keys a row ``"<doc>|L<line>|<token>"``. That key COLLIDES whenever one
line carries the same token string twice — `10 neutral + 10 in-frame`, `0.0854 = 0.0854`, a gate
table row printing its bar and its observed value as the same number — and the dict keeps only the
last write, so both tokens are represented by one status.

A duplicated token is precisely the population the v0.10 cycle addresses, so this is measured and
published rather than mentioned: a severability leg keyed that way is structurally blind to some of
the tokens under test. `make_oath_v10_baseline.py` appends the ledger ORDINAL for that reason.

Reads the two committed baseline files only. It imports nothing from `styxx`, so its output does
not depend on which verifier is installed and it can be re-run at any commit.

  python papers/closed-model-frontier/oath_v10_v09_keying_check.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V09 = HERE / "oath_v09_baseline_ledger.json"
V10 = HERE / "oath_v10_baseline_ledger.json"
OUT = HERE / "oath_v10_v09_keying_check.json"


def main() -> int:
    for p in (V09, V10):
        if not p.exists():
            print(f"FATAL: {p.name} missing.")
            return 2
    v09 = json.loads(V09.read_text(encoding="utf-8"))
    v10 = json.loads(V10.read_text(encoding="utf-8"))

    rows = list(v10["ledger"])
    collapsed = collections.Counter(k.rsplit("|#", 1)[0] for k in rows)
    merged = sum(n - 1 for n in collapsed.values() if n > 1)
    colliding_keys = sum(1 for n in collapsed.values() if n > 1)

    # how many of the merged rows disagree in status with the row that would have survived --
    # i.e. where the collapse does not merely duplicate a verdict but hides a different one
    status_by_key = collections.defaultdict(list)
    for k, s in v10["ledger"].items():
        status_by_key[k.rsplit("|#", 1)[0]].append(s)
    hidden_disagreements = sum(1 for v in status_by_key.values() if len(set(v)) > 1)

    payload = {
        "purpose": "v0.10 observation about the v0.9 harness "
                   "(PREREG_oath_v10_token_column_2026_08_23)",
        "v09_baseline_sha256": hashlib.sha256(V09.read_bytes()).hexdigest(),
        "v10_baseline_sha256": hashlib.sha256(V10.read_bytes()).hexdigest(),
        "v09_key_format": "<doc>|L<line>|<token>",
        "v10_key_format": "<doc>|L<line>|<token>|#<ledger ordinal>",
        "v09_documents": v09["documents"],
        "v09_rows_recorded": v09["tokens"],
        "v10_documents": v10["documents"],
        "v10_rows_recorded": v10["tokens"],
        "distinct_v09_style_keys_over_the_v10_corpus": len(collapsed),
        "rows_merged_away_by_the_v09_key": merged,
        "line_token_pairs_that_collide": colliding_keys,
        "collisions_hiding_a_DIFFERENT_status": hidden_disagreements,
        "note": "This is an observation about the v0.9 HARNESS, not a re-opening of its verdict. "
                "Both v0.9 clauses are demote-only and its G5 read zero, but it read zero over a "
                "ledger that had already merged these rows.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"v0.10 rows {payload['v10_rows_recorded']}  "
          f"distinct v0.9-style keys {payload['distinct_v09_style_keys_over_the_v10_corpus']}  "
          f"merged away {merged}  colliding pairs {colliding_keys}  "
          f"hiding a different status {hidden_disagreements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

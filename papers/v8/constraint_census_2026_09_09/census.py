"""The constraint census of this lab's own published verdict log.

THE_BOUNDARY's class two -- the fields "reachable by nothing" -- emptied on 2026-09-09. What
survives is one unreachable act: the first claim about anything. A subject nobody has measured has
no prior logged cert to contradict, so every consistency check on it is vacuous whatever it
reports.

That makes one quantity owed beside every verdict: how much prior logged material could have
contradicted this claim. This script computes it, for every entry of

    papers/v8/first_verdict_2026_09_09/log

using ``styxx.v8.constraint``, whose own imports are stdlib only and which reads stored entry
bytes. It verifies nothing -- no signature, no cert id, no tree head -- because a census is a
disclosure computed alongside verification rather than a substitute for it. It writes
``census.json`` and ``output.txt`` beside itself and prints the same transcript to stdout.

Run:

    python papers/v8/constraint_census_2026_09_09/census.py
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO))

from styxx.v8 import constraint as C  # noqa: E402

LOG = REPO / "papers" / "v8" / "first_verdict_2026_09_09" / "log"

# The transcript is the receipt, so it is captured here rather than by a shell redirect: this box
# writes a BOM through PowerShell redirection and the stored file has to be plain UTF-8 with LF.
_lines: list[str] = []
_print = builtins.print


def print(*args, **kwargs):  # noqa: A001 - deliberately shadowing, see above
    text = kwargs.get("sep", " ").join(str(a) for a in args)
    _lines.append(text)
    _print(*args, **kwargs)

summary = C.census_log(LOG)
entries = C.read_log_entries(LOG)
by_index = {i: c for i, c in entries}

print("THE CONSTRAINT CENSUS")
print(f"  log            : {LOG.relative_to(REPO).as_posix()}")
print(f"  entries        : {summary['entries']}")
print(f"  issuer keys    : {sorted({(c.get('issuer') or {}).get('key') for _, c in entries})}")
print()

header = f"{'idx':>3}  {'type':<12} {'prior':>5} {'own':>4} {'constr':>6} {'indep':>5}  verdict"
print(header)
print("-" * len(header))
for r in summary["per_entry"]:
    print(
        f"{r['cert']['index']:>3}  {str(r['cert']['type']):<12} {r['prior_entries']:>5} "
        f"{r['own_material_present']:>4} {r['constraining_entries']:>6} "
        f"{r['independent_entries']:>5}  {r['verdict']}"
    )

print("\n\nPER PREDICATE, PER ENTRY")
print("  matched = prior entries in the predicate's scope")
print("  own     = of those, material this cert itself names (its inputs; cannot corroborate it)")
print("  avail   = matched - own")
print("  usable  = available entries carrying the fields the predicate reads")
print("  indep   = usable entries signed by a DIFFERENT issuer key")
for r in summary["per_entry"]:
    idx = r["cert"]["index"]
    print(f"\n  entry {idx} ({r['cert']['type']})   {r['disclosure']}")
    for name, block in r["predicates"].items():
        line = (
            f"    {name:<20} matched={block['matched']:<2} own={block['own']:<2} "
            f"avail={block['available']:<2} usable={block['usable']:<2} "
            f"indep={block['independent']:<2} {block['verdict']}"
        )
        print(line)
        for reason in block["blocked_by"]:
            print(f"        - {reason}")

floor_index = max(
    (i for i, c in entries if (c.get("body") or {}).get("noise_floor")), default=None
)
floor = next(r for r in summary["per_entry"] if r["cert"]["index"] == floor_index)
own_types: dict = {}
for i, c in entries:
    if i != floor_index and c.get("id") in set(floor["own_material"]):
        own_types.setdefault(c.get("type"), []).append(i)

print(f"""

THE HEADLINE ENTRY

  Entry {floor_index} is the canonical fingerprint and the noise floor -- the verdict the
  artifact `first_verdict_2026_09_09` exists to publish. {floor['prior_entries']} entries
  precede it in this log, and {floor['own_material_present']} of those are its own material:
""")
for kind, idxs in sorted(own_types.items()):
    print(f"    {kind:<12} entries {idxs}")
print(f"""
  Prior entries left for any predicate to compare it against: {floor['constraining_entries']}
  Of those, signed by a party other than the issuer          : {floor['independent_entries']}

  This lab's own published verdict is a first claim. Every cross-cert predicate this system has
  or should have returns vacuous on it -- not "agrees", not "disagrees": nothing to compare.

THE SHAPE OF THE LOG

  constraining entries, in index order: {[r['constraining_entries'] for r in summary['per_entry']]}

  The count rises across entries 3, 4 and 5 -- the floor runs, which do not name one another, so
  each is constrained by the runs logged below it -- and falls back to 0 at entry {floor_index},
  because the floor consumes all of them. Consistency accrues, and then the certificate that
  aggregates it starts over.

  Independent constraint is 0 at every index: this log has one issuer key. A party is compared
  only against itself anywhere in it.""")

out = HERE / "census.json"
with out.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"\nwrote {out.relative_to(REPO).as_posix()}")

transcript = HERE / "output.txt"
with transcript.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(_lines) + "\n")
    handle.write(f"wrote {transcript.relative_to(REPO).as_posix()}\n")
_print(f"wrote {transcript.relative_to(REPO).as_posix()}")

# RESULT — COMPAT-2 lands the reading and seals the packet: 211 candidates, still not one accusation

Fathom Lab · 2026-09-16 · Prereg: `PREREG_compat2_surface_and_panel_2026_09_16.md` (frozen on #124
before any of this ran). Receipts: `compat2_gates.py`, `compat2_gates.json`,
`compat2_gate_summary.json`, `compat2_packet.py`, `compat2_key_digest.txt`. Instrument:
`styxx/diffgate.py` sha256 `4ba947a8…`; `web/gate/diffgate.js` re-cut on it. Counts only; no PR
named.

## What the reading does now

A removed public definition is *on the surface* unless its path is scaffolding (the prereg's
closed list: test, spec, example, docs, scripts, internal, vendor, migrations, cmd, e2e, mocks,
storybook, build and the rest, plus `test_*`, `*_test.go`, `*_test.py`, `*.test.*`, `*.spec.*`,
`conftest.py`, `setup.py`). A definition removed and re-defined under the same name with a
different parameter list is a **signature change**: reported, never a drop. A **candidate** is a
compat claim with a covered language and at least one surface drop.

The verdict is unchanged. `COMPAT2_LICENSED` is `False` in the code, `_COMPAT_VERDICTS` is
`("UNCHECKABLE",)`, and a test drives every branch to prove no other verdict can be produced
while the flag is false. The reason now names surface drops first and counts the rest:

    [ ? ] compat_claim   compatibility claimed; the diff removes 1 public definition(s) from the
                         surface, not re-defined in the added lines: pkg/api.py: session;
                         1 more in test/example/internal code; 1 signature(s) changed

## The gates

| gate | result |
|---|---|
| G-C2-1 still never accuses | **13,329** compat claims, **all UNCHECKABLE**, licence flag false and pinned — PASS |
| G-C2-2 every other kind untouched | **22,590 → 22,590** claim tuples, **0** differences — PASS |
| G-C2-3 the partition is a partition | **0** PRs whose removed `(path, name)` multiset differs from HARNESS-1's; **0** failing the split; the 201 PRs whose diff does not parse at all are byte-identical to HARNESS-1 and reported (below) — PASS |
| G-C2-4 the port | differential **3,212 pairs, 6,921 claims, 0 disagreements**; of the 3,205 pre-COMPAT-2 records **3,196 byte-identical**, 9 differing only in `compat_claim` reasons and details (17 claims); `check_pairs.js` **36 pinned pairs, 0 disagreements** — PASS |
| G-C2-5 suite, demo, packet | **4,530 passed**, 6 xfailed, 143 skipped; demo unchanged; the two known `test_sworn_*` committed-sample failures are pre-existing — the sworn set regenerates to the same `b94c5a13…` under the pre-COMPAT-2 instrument `397624d5…`, checked for this run rather than assumed; packet built and sealed, digest below — PASS |
| G-C2-6 what is not claimed | no precision, no agent comparison, no share of compatibility claims called false |

## The census, reported not scored

Of 8,467 PRs claiming compatibility (13,329 claims), **293** drop a public definition — **211 are
candidates** (js/ts 98, python 52, go 26, rust 21, java 20) and **82 drop only scaffolding**.
Those two numbers are exactly what the pre-freeze exploratory pass over the HARNESS-1 ledger
counted with this rule, which is the check that the rule shipped as written. Of 3,596 removed
names, **1,972 are on the surface** and **1,624 are scaffolding**: the filter moves 45% of the
names and 28% of the PRs. Separately, **420 PRs carry a signature change** (2,275 in all) — a
class the reading could not see before and still does not accuse on.

## The sealed packet

    sha256(salt+key) = 140c34f4b6167145e6e962d5b2f8fbbc1261dc4a85c7ad8acce1cbe6fe344480
    salt             = styxx-compat2-blind-2026-09-17
    items            = 180   arms = {candidate: 120, decoy_a: 30, decoy_b: 30}
    packet sha256    = 1bb2d4899c12ed5ea5e0c86a4357ec2080b4ed9acab9c5abb6e9fcf29f72bfea

This digest is committed here, before a single answer exists, and `compat2_packet.py score`
refuses to run unless the sealed key still hashes to it. The packet itself (12 MB of
reconstructed third-party diffs) and the sealed key are gitignored like every ledger; the packet
is reproducible from the builder, the ledger and seed 20260917, and its own sha256 is above.

## A blinding defect in EXTERNAL-1, found by re-implementing it

Building this packet on EXTERNAL-1's shape surfaced a defect in EXTERNAL-1's published one.
`external1_packet.py` assigns item ids in arm order — 100 accusations, then 15 verified decoys,
then 15 synthetic contradictions — and shuffles the *list* afterwards. Shuffling the order does
not shuffle the ids, so **the id carries the arm**. It is demonstrable from the committed
`external1_packet.json` alone, without the sealed key: the fifteen synthetic decoys are exactly
ids `E1-115` … `E1-129`, the last fifteen, as the builder's ordering predicts.

The protocol's own words are that "blinding is structural, not promised" and that decoys make
inclusion carry no status. For the id, that was not true. What it does and does not mean: the
adjudicator was not shown the builder, and EXTERNAL-1's headline (precision 0.23 against a 0.95
floor) is the number that went *against* the instrument, so the leak is not a plausible
explanation for it; but the blinding was weaker than the document asserted, and whether any seat
exploited it cannot be re-tested after the fact. It is filed as its own issue, disclosed here,
and not repaired in this PR. COMPAT-2's packet assigns ids **after** the shuffle — 95 runs across
180 items, arms interleaved — and asserts that at build time.

## Deviations

One, in the gate script rather than the run. `compat2_gates.py` first scored a condition the
prereg does not state — that every compat claim carries the reading's detail — and 201 PRs failed
it. Those are PRs whose diff does not parse at all, where the instrument short-circuits *every*
claim before the per-kind reading ("the diff carries no file statuses and no added lines"), a
pre-existing behaviour untouched by COMPAT-2: the same 201 PRs, byte-identical readings, in the
HARNESS-1 ledger. The prereg's G-C2-3 is about the removed names, which those PRs have none of.
The script now scores the preregistered condition and reports that class separately with the
byte-identical check beside it. The first scoring, the fix and the reason are written down here
rather than silently corrected.

---

*Two hundred and ninety-three PRs claim compatibility while dropping a public definition; two
hundred and eleven of those drops are on the surface where a user could be standing. The gate
still says nothing about any of them. Whether it may is a question for three seats that cannot
tell, from an item's id or anything else in it, which ones the instrument flagged.*

# RESULT — BIN-1 is INVALID: G-BIN-1 fails on 1 of 85 diffs, for a reason the prereg did not name

Fathom Lab · 2026-09-16 · Prereg: `PREREG_bin1_binary_files_2026_09_16.md` (#119). Receipts:
`bin1_gates.py`, `bin1_gates.json`. Instrument under test: `styxx/diffgate.py` with the repair,
sha256 `397624d5…`. Counts only; no PR named.

## The gates as frozen

| gate | result |
|---|---|
| G-BIN-1 the 91 live diffs, parse count = `diff --git` header count on every served PR | 85 served, **84** equal — **FAIL** |
| G-BIN-2 the differential: 0 disagreements, the 3,199 pre-repair records identical | 3,205 pairs, 0 disagreements, 3,199 of 3,199 identical — pass |
| G-BIN-3 a 2,000-PR corpus sample re-derived identically | 2,000 of 2,000 (966 claims) — pass |
| G-BIN-4 EXTERNAL-5 re-read: parse count = live count on every served PR; same outcome per item | 90 of 90; 96 of 96; the 11 items on binary-carrying diffs now counted by the parse — pass |

A blocking gate failed, so this preregistration is INVALID and the repair does not land under it.

## The one diff

Its 645 headers name 645 files. The repaired parse registers 644: two of the files are
`pr_agent.toml` (deleted) and `.pr_agent.toml` (added), and `_norm` — the path normaliser every
door has used since the first release — strips leading `.` and `/` characters together
(`lstrip("./")`, meant for a `./` prefix), so both keys are `pr_agent.toml` and the map holds one.
Every binary file in that diff is registered. The miss is a key collision in the status map, not
the blind spot BIN-1 repairs, and it is older than BIN-1: the same normaliser turns
`.github/workflows/x.yml` into `github/workflows/x.yml` on both the claim and the diff side, so
matching survives, but a dotfile and its undotted twin in one diff cannot both be counted. Filed
as its own issue; its repair touches every path claim and is not this one's to make.

## What follows

BIN-2 re-freezes with one change to one gate and none to the repair: G-BIN-1 counts *distinct
normalised header paths*, the quantity the parse can be held to, and names the collision issue
as out of scope. The code under test is the same file.

---

*Eighty-four of eighty-five, and the eighty-fifth is a dotfile the instrument has never been
able to tell from its twin. The rule is that a failed gate is a failed gate; the cause gets its
own number, and the repair gets a second freeze.*

# RESULT — COMPAT-1 lands: 8,467 compatibility claims read, 540 of them on diffs that remove a public name, none accused

Fathom Lab · 2026-09-16 · Prereg: `PREREG_compat1_2026_09_16.md`, pushed on `fathomlab-patch-16`
after `EXPLORATORY_compat_and_multilang_2026_09_16.md` and before this run. Receipts:
`external3_harness.py --tag external4` (the same harness as BC-2's runs), `external4_gates.py`,
`external4_gate_summary.json`, `external4_gates.json`. Baseline: the BC-2 ledger
(`external3_ledger.jsonl`, the checkout at the prereg commit, `diffgate.py` `4bf19b1b…`); repaired:
`external4_ledger.jsonl` (`a550cad5…`). Ledgers gitignored; counts only; no PR named.

## The gates

- **G-C1 (never accuses) — PASS.** 13,329 `compat_claim` claims on 8,467 PRs; every one
  UNCHECKABLE. The kind has one verdict in code (`_COMPAT_VERDICTS == ("UNCHECKABLE",)`), one in
  every reason branch (`tests/test_diffgate_compat.py`), one on the corpus.
- **G-C2 (every other kind untouched) — PASS.** 22,408 non-compat `(pr_id, kind, text, verdict)`
  tuples before, 22,408 after, 0 differences. The 96 CONTRADICTED and 13,017 VERIFIED of BC-2 are
  the same claims with the same verdicts.
- **G-C3 (the census reproduced) — reported.** PRs with a compatibility claim: **8,467**, the
  exploratory count exactly. PRs whose diff removes at least one public top-level definition not
  re-defined in the added lines of the same language: **540** (exploratory 531; the instrument
  also reads `async def` and reads `.kt` under Java). By language: JS/TS 243, Python 164, Go 79,
  Java 40, Rust 34. Removed-name histogram: one name 168, two 84, three 53, four 43, five to
  nine 81, ten or more 111. 5,062 PRs' claims sit on diffs in a covered language with no public
  definition removed; 2,865 on diffs in no language this reading covers (Markdown, YAML, C#, C++,
  PHP, Ruby, …).
- **G-C4 (suite and demo) — PASS.** Full suite green in this environment (the two known
  `test_sworn_*` committed-sample failures deselected as before); the demo unchanged; the CLI's
  `--out` JSON carries `detail.removed` as a list of `{path, language, name}`; the hooks print
  `[ ? ] compat_claim` with the names in the reason.
- **G-C5 (what is not claimed).** No precision for the evidence; no agent split; no "X% of
  compatibility claims are false". 540 is the number of diffs where the description's claim and
  the diff's removed surface are worth a reviewer's minute, and that is the whole of it.

## What a reader sees now

    [ ? ] compat_claim   compatibility claimed; the diff removes 2 public definition(s) not
                         re-defined in the added lines: src/api.py: session, src/api.py: Legacy

or, on the other side of the same coin,

    [ ? ] compat_claim   compatibility claimed; no public top-level definition removed (python,
                         js/ts read; behaviour beyond names not checked)

Coverage rises from 12,933 to 17,939 of 71,016 PRs (18.2% → 25.3%): the single most-read
sentence the gate never read is now read, with the diff's answer beside it. It is not a verdict
and the reason says why in every branch. One deviation from the prereg's phrase set, recorded: the
"does not change …" form accepts an article ("does not change **the** public API"); the prereg
listed "[any] [existing]" and this reading adds "[the]".

## What follows

If anyone wants this reading to accuse — "you said backward compatible and removed `Session`" — the
path is the EXTERNAL-1 protocol: a preregistered precision floor, a blind panel shown the removed
names and the diff, sealed decoys, and the accusing verdict enabled only above the floor. Until
then the instrument says what it saw and lets the reader judge, which is the job.

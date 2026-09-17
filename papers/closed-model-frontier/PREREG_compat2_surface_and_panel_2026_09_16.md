# PREREG — COMPAT-2: the compatibility claim gets a candidate verdict, a sharper reading, and a blind panel that decides whether it may accuse

Fathom Lab · 2026-09-16 · Frozen after HARNESS-1 (`RESULT_harness1_merge_fold_lands_2026_09_16.md`:
293 of the 8,467 compatibility claims sit on a diff that drops a public definition, corrected from
540) and before any of the reading below is run over the corpus or any packet is built. Built on
COMPAT-1 (`PREREG_compat1_2026_09_16.md`), whose one rule stands: **this kind does not accuse
until a blind panel licenses it.** This prereg is the licence's protocol.

## Why this cycle

After EXTERNAL-5 and HARNESS-1 the countable claims are nearly clean: 80 contradictions in
71,016 agent PRs. The sentence agents write most and nobody checks is "no breaking changes" —
8,467 of the descriptions — and today the gate reads it and withholds. This cycle asks, under
protocol, whether the diff alone can ever settle that sentence: the reading is sharpened so that
what it names is the public surface and not test scaffolding, a *candidate* verdict is computed
and pinned to UNCHECKABLE, and a blind panel scores the candidates. Only a passing panel, in its
own RESULT, flips the flag that lets CONTRADICTED appear.

## The reading, sharpened (evidence only; verdict unchanged)

- **Surface vs scaffolding.** A removed public definition is *on the surface* unless its path
  matches the scaffolding rule: any directory segment in `test, tests, testing, spec, specs,
  __tests__, example, examples, sample, samples, demo, demos, doc, docs, script, scripts, tool,
  tools, bench, benchmark, benchmarks, fixture, fixtures, internal, _internal, private, vendor,
  third_party, migration, migrations, cmd, e2e, integration, mock, mocks, stories, storybook,
  playground, sandbox, experiment, experiments, dev, build`, or a file named `test_*`, `*_test.go`,
  `*_test.py`, `*.test.*`, `*.spec.*`, `conftest.py`, `setup.py`. `detail.removed[*].surface` is
  set per name; `detail.surface_removed` counts them. The reason names surface drops first and
  ends with `; N more in test/example/internal code` when there are any.
- **Signature changes, read and reported.** A public definition removed and re-defined under the
  same name in the added lines of the same language, where the text between the definition's
  first `(` and its matching `)` differs after whitespace normalisation, is listed in
  `detail.signature_changed` as `{path, name, before, after}`. It is *not* a drop, it does not
  make a candidate, and the reason mentions it only as `; K signature(s) changed`.
- **The candidate.** `detail.compat2_candidate` is true when a covered language is in the diff
  and at least one removed public definition is on the surface. The verdict is UNCHECKABLE
  regardless; `COMPAT2_LICENSED = False` in the code, a test pins that no `compat_claim` verdict
  other than UNCHECKABLE can be produced while it is false, and the corpus run below checks it.
- COMPAT-1's set of removed names is untouched: the surface rule partitions it, nothing is
  added to it or taken from it. The JS port mirrors the reading; the differential re-pins.

## The panel — packet frozen here, built after the reading lands, scored by seats the author does not hold

- **Arms.** *Candidates:* every PR whose reading is a candidate, seeded sample of 120 if there
  are more (seed 20260917). *Decoy A:* 30 PRs with a compatibility claim and no removed public
  name. *Decoy B:* 30 PRs whose removed public names are all scaffolding. Shuffled; one packet;
  arm membership sealed in a salted key whose SHA-256 is committed before any answer exists
  (`compat2_packet.py`, the EXTERNAL-1 packet's shape).
- **What an adjudicator sees.** The compatibility sentence, the PR title, and the reconstructed
  diff of every file that has removed lines (capped at 400 lines per file and 3,000 in all; the
  cap is stated on the item). Never the gate's reason, the removed names, or the arm.
- **The question.** *Does this diff remove or change something a user of this project's public
  interface could already be depending on — a function, class, type, export, endpoint, option —
  so that "no breaking changes / backward compatible" is false for that user?* Answers: `break`,
  `compatible`, `cannot tell`.
- **Seats.** Three adjudicators from at least two model families, none the author of the
  instrument; majority decides; `cannot tell` and ties resolve **against** the instrument
  (counted as a false accusation on the candidate arm, as a true break on the decoy arms).
  Per-seat disagreement is reported.
- **The floor, pre-committed.** Precision on the candidate arm = majority-`break` / all
  candidates scored. **≥ 0.95: licensed** — a follow-up PR flips `COMPAT2_LICENSED`, CONTRADICTED
  appears for candidates only, with the surface names in the reason, and the census is re-run.
  **0.80–0.95: not licensed**; the false classes are named from the packet and a COMPAT-3 prereg
  must narrow the rule before another panel. **< 0.80: the surface rule is wrong**; the reading
  stays evidence, the classes are published, no further panel on this rule.
- **The decoy arms are measurements, not padding.** Decoy A's `break` rate is the reading's
  miss rate on this corpus; Decoy B's `break` rate says whether scaffolding removals are breaks
  after all. If Decoy B's majority-`break` rate is ≥ 0.20 the scaffolding rule is retired as a
  filter (reported as evidence, never used to withhold), whatever the candidate precision.

## Gates — committed now

- **G-C2-1 (still never accuses).** Zero `compat_claim` verdicts other than UNCHECKABLE on the
  corpus; the licence flag pinned false by a test that drives every reason branch. Blocking.
- **G-C2-2 (every other kind untouched).** Ledger to ledger, HARNESS-1 (`external6_ledger`)
  against this checkout on the same shelf: all non-`compat_claim` `(pr_id, kind, text, verdict)`
  tuples identical. Blocking.
- **G-C2-3 (the partition is a partition).** Per PR, the multiset of removed `(path, name)` is
  identical to HARNESS-1's; `surface` + scaffolding = all. Blocking. Reported beside it: the
  candidate count (the exploratory pass over the HARNESS-1 ledger says **211** of the 293, with
  **82** PRs whose drops are all scaffolding — a count with this prereg's rule, not a result),
  per language, and the signature-change count.
- **G-C2-4 (the port).** `web/gate/diffgate.js` mirrors the reading; the differential prints 0
  disagreements on the new checkout with the pinned pairs re-pinned and the pre-COMPAT-2 records
  byte-identical except in `compat_claim` reasons and details. Blocking.
- **G-C2-5 (suite, demo, packet).** Suite green; demo unchanged; `compat2_packet.py build`
  writes the packet and the sealed key and prints the digest, which this cycle's RESULT commits
  before a single answer is recorded. Blocking for the build; the scoring is the next cycle's.
- **G-C2-6 (what is not claimed).** No precision until the panel; no agent comparison; no "X%
  of compatibility claims are false". A candidate is a candidate.

## Out of scope, named

Semantic compatibility beyond names and signatures (return types, behaviour, wire formats), a
definition re-exported from a file the diff does not touch, languages outside the five, and
whether a description that was true when written and false after a later push should be told
apart from a lie (WH-1).

---

*The sentence that 8,467 agents wrote is one the diff can sometimes settle: a public function
that is gone is gone. Whether "sometimes" is often enough to accuse is not the author's call,
and this prereg gives the call to a panel that cannot see which items the instrument accused.*

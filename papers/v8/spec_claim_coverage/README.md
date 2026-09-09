# spec_claim_coverage — the structural check: is a rule the spec states a rule the code keeps?

Fathom Lab · 2026-09-09 · **an inventory, not an assurance.** It reads
`papers/v8/SPEC_v8_v0.2_draft.md`, extracts every sentence that reads as normative, and asks of
each one whether any test under `tests/` names the behaviour. It cannot ask whether a test would
*fail* on a violation, and it does not pretend to. Nothing here was sworn or committed, and the
only files this produced are the three in this directory.

## Why

Three times on 2026-09-09 the specification was repaired and the implementation never heard about
it — amendment **A-29** (the recipe split, still not in `cert.RECIPE_CORE_FIELDS`), finding
**S9-01** (the challenge subject check), and the **§6 subject guard** (dead code with the real
runner). Each was found by accident: one by a test written for a different reason, two by the
adversarial pass in `../challenge_and_attack_2026_09_09/`. Nothing systematically asked the
question. This is the artifact that asks it.

## Run it

```
python papers/v8/spec_claim_coverage/inventory.py
```

Writes `spec_claim_coverage.json` beside itself and prints the summary, the calibration and the
miss list. It reads only; it changes nothing in `styxx/` or `tests/`. The output is deterministic
— two consecutive runs on an unchanged tree produce byte-identical JSON — so a re-run after a spec
edit or a new test diffs cleanly against the last one, which is the only way a reader detects that
a rewording moved a claim in or out.

## What it found (spec sha256 `9142b8ee…`, 241 test files, 20 v8 source files)

| | count |
|---|---|
| normative claims extracted | **175** |
| `has-a-test` — one v8 test function names it and asserts | **38** |
| `maybe` — a related test exists, none pins the claim | **69** |
| `no-test` | **68** |
| &nbsp;&nbsp;of which *searched and not found* | 8 |
| &nbsp;&nbsp;of which *invisible to this instrument* (the sentence carries no searchable name) | 60 |
| operator-gated sentences skipped (they state no rule yet) | 19 |

**The rate is not the result.** 38/175 is as much a property of this program's aperture as of the
suite: 60 of the 175 carry no name a text search can use, and every threshold below is set to
classify down. Read the miss list.

### The miss list — searched, and no test names it

Ordered by a severity heuristic (below). `code:` says whether the names the claim uses appear
anywhere in `styxx/v8/`.

| sev | claim | code | the rule |
|---|---|---|---|
| 10 | `§8.6#c1241819a0` | no | the log "makes visible" a late-logged prereg, a rewritten number, an unverifiable claim — the threat-model sentence itself |
| 9 | `§2.3#a57abf6c62` | **no** | every §4.3 selection number is labelled `cross-configuration` in the battery cert |
| 9 | `§5.1#03963f566e` | — | the baseline rule refuses a canonical fingerprint that names neither a `previous` nor a plan (quoted refusal string) |
| 9 | `§6.1#205bf8ed9a` | **no** | an issuer that says nothing has not refused; the cert stays `coverage-contested` |
| 7 | `§5.1#eaaa2d8178` | — | a cert offered as canonical whose id equals a run cert's id is not a canonical fingerprint, and the floor reads as absent |
| 5 | `App.A#c5c58da118` | **no** | a per-channel `vector_sha256` is informative only; log-probs are compared numerically, never by hash |
| 4 | `§3.2#b78aa13a78` | **no** | `topk_forced_on: "self"` is the only option for alias subjects, and that makes topk a weaker channel |
| 1 | `§12#3ea6283840` | **no** | every compliance category prints `suppressed: n`, including zero |

Four of those eight were checked by hand, and the searches really do come back empty:
`topk_forced_on`, `coverage-contested`, `vector_sha256` and `cross-configuration` /
`cross_configuration` appear in **no test file and in no file under `styxx/v8/`**. They are rules
that exist only in the specification — the S9-01 shape exactly.

### Spec-only rules (16) — the shape all three of today's drifts had

`spec_only_rules` in the JSON lists every claim whose names appear nowhere in `styxx/v8/` and
which no test pins. It is the list to read first, because it is where the next A-29 is. Beyond
the four above it includes `body.scope` on a promotion (§2.7), `public: false` in the compliance
view (§2.6), the `resid` profile requirement (§3.2), and the two `scope.*` enum rules of §7.2.

An independent corroboration: `baseline_gap` — §5.5's client-computed status and §6's result-body
field — occurs in **no** file under `styxx/v8/` (its only hit anywhere in `tests/` is a legacy 7.x
file). That is defect **L3** of `../challenge_and_attack_2026_09_09/`, found there by an adversary
reading code and found here by a text search over the spec. Two methods, one answer.

### Not pinned (69 `maybe`)

`not_pinned_maybe` in the JSON. The heaviest are append-time refusals: the `items_blob` presence
rule (§3.1), the `*_sha256`-against-`materials` check (§2.3), the schema-version refusal (§2.5),
and both halves of the `previous` rule (§5.5). Some of these are false negatives — see below.

## The method, stated exactly

### The extraction aperture — what a sentence has to look like

A **unit** is a prose sentence or one data row of a markdown table. Fenced code blocks are not
read (a schema is not a sentence). The amendment ledger is not read (it is history, not a rule).
A paragraph containing `[OPERATOR-GATED …]` is not read (an undecided block states no rule yet;
19 sentences).

A unit is a **claim** iff one of ten marker patterns fires in it: `MUST` / `MUST NOT` (2 claims),
`must` / `may not` / `cannot` / `shall` (30), `required` / `requires` (26), `forbidden` (19),
`refuse` / `reject` (35), `never` (51), `always` (5), `iff` / `only if` / `the only` (16),
`exits N` (26), `is (in)valid` (4). Its id is `§section#` plus the first ten hex of the sha256 of
the sentence, so it is stable across runs and moves only when the sentence is edited.

**What this cannot see.** A rule stated as a diagram, a schema fragment, a JSON example or a
field name in a table cell with no verb. A rule spread across two sentences, where the second
carries the marker and the first carries the subject. A rule stated as pure description
("`floor_c = max(D_c)`") with no modal. A rule that exists only as an absence — something the
spec never says and should. Every arithmetic definition in Appendix B and Appendix C, which state
formulas rather than obligations. The 19 gated blocks, which will become rules and are not rules
today.

### The coverage aperture — how a claim is matched to a test

Each claim yields **names**: identifiers the spec put in backticks, hyphenated terms of art
(`beyond-floor-coverage`, `comparability-gating`), two-word compounds from a backticked command
(`verify --ref` → the pair *verify* + *ref* on neighbouring lines), verdict and status literals,
and refusal strings the spec quotes in italics, reduced to their first five words. A backticked
span holding a path (`papers/v8/first_log_2026_09_09/`) or a regex contributes nothing: a test
that cites the same receipt is not a test of the rule.

Names are split by how much they discriminate, measured on this corpus rather than assumed. A
name occurring in at most **60** of the 241 test files is an **anchor**; `refs` occurs in 13 and
anchors, `same` occurs in 110 and does not. Exit codes are never anchors. A name with *zero*
occurrences is still an anchor: the search that came back empty is the finding.

The unit of evidence is **one test function**, not a line window. Two hits sixty lines apart that
straddle a `def` are two tests, not one place.

- **`has-a-test`** — one function in a v8 test file (`tests/test_v8_*.py`, `tests/v8_fixtures.py`,
  `tests/js/`) whose name begins with `test` — in JavaScript, one `test(…)` / `it(…)` block; a
  fixture or a helper is not a test — containing **two** distinct anchors of the claim
  (a compound and its own parts count once) and at least one assertion; and, when the claim says
  something is *never* done, is *forbidden* or is *refused*, that same function must also carry a
  line that both asserts and expresses an absence or a refusal.
- **`maybe`** — an anchor is mentioned in a live test somewhere, but no function meets that bar.
- **`no-test`** — reason `no-hit` (anchors exist, nothing in `tests/` mentions them),
  `xfail-only` (the only mentions are inside `@pytest.mark.xfail`/`skip` functions),
  `generic-only`, or `no-probe` (the sentence carries no searchable name at all — an instrument
  hole, listed separately as `invisible_to_this_method`).

**A `has-a-test` means a test names the behaviour in a place that asserts something.** It does
not mean the test is correct, that it is the right test, or that it would fail on a violation.
Only mutation would show that, and no mutation was run here.

**We classified down.** Where a case was unclear it was pushed toward `maybe`, on the reasoning
that an inflated `has-a-test` hides a miss while an inflated `no-test` costs a reader time. Two
consequences are visible in the output. §3.1's `items_blob` append refusal is `maybe`, and
`tests/test_v8_log.py::test_blobs_are_content_addressed_and_a_wrong_hash_is_refused` does pin it —
the test function names one anchor, not two. §9's subject-identity rule is `maybe` although
`cert.challenge_validity` is asserted in `tests/test_v8_cert.py`, because the spec sentence names
`S_identity` and the test names `challenge_validity`; the method cannot bridge a vocabulary
change between the two documents. Both are false negatives, and both are the direction chosen on
purpose.

### An xfail is not coverage

A hit inside an `@pytest.mark.xfail` or `@pytest.mark.skip` function does not count. The
assertion exists and does not run, so nothing fails when the code stops honouring the rule —
which is precisely the A-29 state today: `tests/test_v8_spec_agreement.py` holds two strict
xfails that pin the drift and, by construction, pass while the drift stands. A method that
counted them as coverage would have reported the loudest known miss in the tree as covered. The
first version of this program did exactly that, and that is what the calibration bought.

### Severity, and what it is worth

`severity` is a keyword heuristic, not a measurement: +3 for an append-time refusal, +3 for a
verdict or exit code, +3 for signature / canonical bytes / Merkle / STH / roster words, +2 for
floor or coverage words, +2 for a load-bearing section (§2, §3, §5, §6, §8, §9), +2 when the
claim's names appear nowhere in `styxx/v8/`, +1 for a modal, −2 for a process section. It orders
the list; it does not rank the risk. The top row (§8.6) is a threat-model paragraph whose only
anchor is the prose hyphenation `late-logged`, and it scores 10 because it happens to contain
`STH` and `prereg`. Read the sentences.

## Calibration

This lab does not accept a coverage number without one. The three known drifts of 2026-09-09 were
run through the method — and their ground truth is **not the same today**, which makes the set a
two-sided control rather than a one-sided one.

| target | state | expected | got | verdict |
|---|---|---|---|---|
| **A-29** `execution` is never comparability-gating (§2.3) | still OPEN — `cert.RECIPE_CORE_FIELDS = ("battery", "decoding", …)` | not `has-a-test` | `maybe` | pass |
| **S9-01** a challenge is valid only on equal subject identity (§9) | **repaired since the attack pass** — `cert.challenge_validity` exists and `tests/test_v8_cert.py` asserts it | `has-a-test` or `maybe` | `maybe` | pass |
| **§6 subject guard** `identity` is emitted when a subject field differs (§5.2) | **repaired since the attack pass** — `TransformersRunner.subject` exists and `tests/test_v8_subject_guard.py` pins C2 | `has-a-test` or `maybe` | `has-a-test` | pass |

**The brief asked for all three to come out `no-test` or `maybe`, and that is no longer the right
expectation.** Two of the three were repaired by other work in the same session, hours after the
attack report was written: `cert.challenge_validity` and `TransformersRunner.subject` are both in
the tree now, both with tests. Their ground truth changed, so holding them to `no-test` would
test the method against a fact that is no longer true. They are kept as **positive controls**
instead — a method that still called them `no-test` would be blind to real coverage, which is the
failure mode a miss list cannot afford in the other direction. A-29 remains the negative control,
and it is the one that still matters.

A-29 **failed** the calibration on the first two versions of this program, and both failures were
real defects:

1. it counted the strict-xfail assertion in `tests/test_v8_spec_agreement.py` as coverage —
   repaired by the inert-range rule;
2. it then counted `test_prereg_noise_plan_derives_covers_and_not_covered_from_what_it_varied`,
   a positive assertion about the neighbouring half of the same compound sentence, as coverage of
   the "never comparability-gating" half — repaired by the polarity rule (a claim that something
   never happens needs a test that asserts an absence) and by requiring the negative evidence to
   sit on a line that both asserts and negates.

Neither repair was aimed at A-29 specifically; both are general and both moved other claims.

## Limits

One specification, one test corpus, one machine, one afternoon. **No mutation was run**, so no
claim in this file is evidence that any test would fail on a violation — that measurement is
owed, and until it exists `has-a-test` means only what the sentence above says it means. The
extraction is a regex over prose; a rewording of the spec moves claim ids and can move a claim in
or out of the inventory entirely, and nothing detects that but re-running and diffing. 60 of 175
claims carry no name this method can search, so more than a third of the inventory is a report
about the instrument rather than about the suite. The severity order is a keyword heuristic. The
corpus outside `tests/test_v8_*` was searched but is never accepted as evidence, so a v8 rule
genuinely covered by a 7.x test reads here as uncovered. Three of the eight miss-list rows rest
on a single anchor, and a single anchor is a thin basis for a claim about a whole test suite.

**Environment note, unrelated to this work:** at the time of the run
`tests/test_v8_conformance.py::TestTheMutationCoverage::test_the_receipt_measures_the_modules_in_the_tree_today`
fails because `styxx/v8/cert.py` changed after the mutation-coverage receipt was measured (the
S9-01 repair). Nothing in this directory touches `styxx/` or `tests/`; the failure is present with
or without these three files, and the receipt is owed a re-measurement into a new file.

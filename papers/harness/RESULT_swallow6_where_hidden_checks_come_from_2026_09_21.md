# RESULT — SWALLOW-6: where hidden checks come from — 40 of the 53 were written hidden; of 142 that were ever hidden, 5 are loud today; a hidden check is a median 155 days old and the oldest is four and a half years; when an author does make one loud, the edit is the instrument's 12 times in 14

Fathom Lab · 2026-09-21 · Scores the receipt `swallow6_receipt.json.gz` against the
preregistration frozen at sha256 `d19eff1fc5a018b68d1d5d46984f67f3731b437fefb3f616d638043e94233725`.
Not amended. Three runs: the instrument was amended twice after the freeze, once for memory and
once for a bookkeeping defect the second run's own gate exposed; §1 states each change, what it
moved, and that the predictions and the scorer's tests did not move.

Receipt: `papers/harness/swallow6_receipt.json.gz` (sha256 of the JSON `acc592cb…`, recorded by the
scorer; every lineage's birth, events, renames and death are in it) · instrument
`benchmarks/harness_mutation/history.py`, sha256 `4b961880…` (the prereg names `d92b1b32…`; §1),
on top of `repair.py` (`7b9a1695…`), `repair_structural.py` (`77067a71…`), `action_checks.py`
(`0e723694…`) and `faults.py` (`d26a407c…`), none changed · population: the 100 repositories of
the SWALLOW-3 receipt (`609e6645…`) at their pinned HEADs, 96 cloned with history to 2019-08-01,
4 no longer public · 27,624 mainline commits touching `.github/workflows`, 35,071 workflow
revisions read, 14,199 lineages followed through 2,819 hand-written workflow paths (alive at HEAD
or removed along the way) · 1,629 s on two workers · scored by `swallow6_score.py`.

**VALID. 5 of 7 predictions HIT** (P1, P3, P5, P6, P7); P2 and P4 MISS. Of the **53 hidden
checks alive at HEAD**, **40 were hidden at birth and have never been loud** (P1: 75%); 3 were
loud once and hidden later, each by a `continue-on-error: true`; 10 are one commit in
`dotnet/maui` that rewrote ten non-checks into ten hidden checks. Of the **142 lineages that were
ever hidden**, **5 are alive and loud today** (P3: 3.5%); 64 died hidden — 55 of them born
hidden — and 53 are alive hidden. The median hidden check is **155 days** old (P4 predicted 180:
MISS; 26 of 53 are older than 180 days, 12 older than a year, the oldest — `mlflow`'s database
tests — 1,630 days and 291 revisions of `master.yml`). Sixteen commits hid a loud check; 12 of
them by `continue-on-error` or `|| true` (P5: 75%); **4 of the 16 say why** in the stated words
(P2 predicted half: MISS — "tweak github actions" hid four at once). Fourteen commits made a
hidden check loud; on the revision before each, SWALLOW-4 or SWALLOW-5 proposes a verified repair
13 times, and **12 times it is the edit the author made** (P6: 86%). Hiding outnumbers repairing,
16 to 14 (P7, by two).

## 0. What was done

Every hand-written workflow of every readable repository was read at every revision on the
default branch's mainline (`--first-parent`, oldest to newest, since the GitHub Actions YAML era
began on 2019-08-01, to the pinned HEAD), with SWALLOW-3's reading of the frozen `faults.py`,
unchanged — one memoised runner per workflow, so a step that did not change is not read twice.
Every step was followed as a lineage keyed by its job and its name (else id, else first line);
a step whose script survived a rename is followed under the new name (`renames` in the receipt).
Between consecutive readable revisions the lineage's state — `hidden` (SWALLOWED, FAIL_OPEN),
`loud` (RED), `other` (not a check here), `unread` (BASELINE_RED / BASELINE_SKIPPED, carried) —
is compared, and loud → hidden is an **acquisition**, hidden → loud a **repair**, each with the
commit, the mechanism read from the step's YAML before and after, and the first acknowledgement
the stated word list finds in the commit's subject, body, or the subjects it merged. For every
repair, SWALLOW-4's `try_repairs` and then SWALLOW-5's `try_structural` were run on the revision
before, and the first verified candidate compared with the author's mechanism. The clones are
blob-less and shallow to 2019-08-01, with every workflow blob along the mainline fetched in one
batch per repository; the 19 lineages born at that boundary are marked left-censored (none of the
53 is one).

## 1. Gates, and the deviations

| gate | bar | result |
|---|---|---|
| G-S6-1 instrument | tests green, every event on the scripted history, deterministic | pass — `tests/test_harness_history.py`, 4 passed |
| G-S6-2 population | ≥ 90 of 100 read to the pinned HEAD uncapped | pass — **96 of 100**, 0 capped; 4 could not be cloned (`Smart-Cleaner-for-Android`, `antiwork/flexile`, `antiwork/helper`, `Metta-AI/metta`: no longer public) |
| G-S6-3 HEAD agrees | ≥ 99% of matched hand-written faults carry the receipt's verdict at HEAD | pass — **6,380 of 6,381** (the one: `mfat/sshpilot` `macos-pyinstaller.yml › build-arm64 › 2`, BASELINE_RED in the receipt, RED here — a step that creates a venv, `source`s its activate script and pipes `pip list` into `grep`; the reading of it moved between two runs, which is the nondeterminism SWALLOW-3 measured at 8 of 30,642, here 1 of 6,381) |
| G-S6-4 frozen underneath | the instrument and everything under it at the named hashes | pass at the amended hash (below); `repair.py`, `repair_structural.py`, `action_checks.py`, `faults.py` and the SWALLOW-3 receipt file unchanged |
| G-S6-5 ledger | P1–P7 scored | pass |

**Deviation 1 — the instrument was amended twice after the freeze; the prereg text was not.**
The preregistration names `history.py` at `d92b1b32…`. **Run 1** (that file) was stopped after 6
repositories: the worker on `githubnext/gh-aw` reached 5.5 GB, because the texts of every
generated `*.lock.yml` revision were fetched though never read. Amended (`6986d59f…`): one
workflow's texts are fetched at a time, and generated workflows are never fetched. No reading
changed. **Run 2** (that file) read all 96 repositories and was **INVALID on G-S6-3 at 96.7%**
(209 of 6,377 differ). The cause was the instrument's own bookkeeping, not the reading: a
lineage's `verdict` carried its last *interpretable* reading (so that events are read between
readable revisions), and the gate — and the population — used that carried verdict as the
check's state at HEAD. A check uninterpretable at HEAD was therefore compared, and counted, as
what it had last been: 197 of the 209 were BASELINE_SKIPPED or BASELINE_RED at HEAD and RED
before; **4 lineages were counted alive-hidden that are uninterpretable at HEAD** (run 2 counted
57; the SWALLOW-3 receipt has 53). Amended (`4b961880…`): the raw reading at the latest revision
(`state_last`, `verdict_last`) is kept beside the carried one, and what a check *is* at HEAD is
`state_last`; the scripted history gained a lineage that is unread at every revision to hold
this. **Run 3** (that file, this receipt) is VALID with G-S6-3 at 6,380 of 6,381. **The
amendment moved one prediction**: run 2 scored 6/7 with P4 HIT at a median of 184.3 days on 57
lineages; run 3 scores 5/7 with P4 MISS at 155.4 days on 53. Both are stated; the second is the
one that reads the checks as they are. The scorer's tests, the predictions and their bars did not
change between runs; the scorer's pinned instrument hash did, and says so in a comment.

**Deviation 2 — the acknowledgement list has a false positive.** `flak` matches `flake8`. One
repair (`AutoGPT`'s `Lint with flake8`, "fix-flake8-issues") is counted acknowledged by it; no
acquisition is. P2 is unaffected; the word is left as frozen and named here.

**Stated at the freeze, repeated:** `langfuse`, `hmislk/hmis` and `carverauto/serviceradar` were
read while the driver was being tested, before the predictions were set — three born-hidden
checks aged 24, 555 and 33 days. They are in the population and in every count.

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 born hidden | ≥ 60% of the hidden checks alive at HEAD hidden at birth, never loud | **40 of 53, 75.5%**; 3 acquired; 10 became checks already hidden (`maui`, one commit) | HIT |
| P2 acknowledged | ≥ 50% of acquisitions say why in the stated words | **4 of 16, 25%** (`unblock`, `non-blocking`, `skip`, `broken`) | MISS |
| P3 repair is rare | ≤ 25% of ever-hidden lineages alive, loud and repaired at HEAD | **5 of 142, 3.5%** | HIT |
| P4 not transient | median age of the 53 ≥ 180 days | **155.4 days** (quartiles 61 / 155 / 364; 26 over 180, 12 over 365) | MISS |
| P5 the two lines | continue-on-error + `\|\| true` the primary mechanism of ≥ 50% of acquisitions | **12 of 16, 75%** (11 `continue-on-error`, 1 `\|\| true`, 4 rewrites — one commit) | HIT |
| P6 the instrument's edit | ≥ 40% of readable wild repairs agree with the verified candidate | **12 of 14, 85.7%**; 13 have a verified candidate | HIT |
| P7 hiding outnumbers repairing | acquisitions > repairs | **16 > 14** | HIT |

## 3. The natural history, read

**Written hidden.** Forty of the 53 hidden checks alive today were hidden in the commit that
created them, and nothing since has made them loud: `aspire`'s six, `vscode`'s four smoke-test
diagnostics, `mochi`'s four, `sentry-docs`' three, `vtcode`'s three governance baselines,
`prebid`'s two linters, `selfxyz`'s two verifiers, `gumroad`'s two, and one each in `airbyte`,
`blot`, `codenameone`, `crewAI`, `giselle`, `hmis`, `langfuse`, `mlflow`, `nodetool`, `py3plex`,
`roslyn`, `serviceradar`, `vikunja`, `gh-aw`. The `|| true`, the `continue-on-error`, the guard
whose failing tool is its green path — these are how the check was written, not what it became.

**Ten became checks already hidden.** `dotnet/maui`'s `validate-pat-pool.yml` had ten
`Validate COPILOT_PAT_N` steps that were not checks under the reading on 2026-07-05; a commit
sixteen days later rewrote all ten into checks that swallow their own failure. The lineage
records it as `other → hidden`, mechanism `rewrite`, and P1 counts them as neither born hidden nor
acquired: they are the 10 "other" in the ledger. Read plainly, they are ten more checks that were
hidden from the moment they became checks.

**Three were loud once.** `githubnext/gh-aw`'s `Lint error messages` was RED for 26 days, then
"Make error-message lint advisory (#54800)" added `continue-on-error` (advisory: the word is not
on the list). `neondatabase/website`'s `Monitor critical user outcomes` was loud for one day
("ci: point npm at the … mirror so npm ci stops dying on the protected runners" — `broken`, in
the body). `promptfoo`'s `Run Redteam with Staging API` was loud for 53 days, then "chore(ci):
make staging redteam test non-blocking" — `non-blocking`, in the subject.

**How old.** Ages run from 2.8 days (`mochi`, whose pinned HEAD is days after the step) to 1,630
(`mlflow/mlflow` `master.yml › database › Run tests`, hidden since 2022-04-05 and carried through
291 revisions of the file). Twelve are older than a year: `mlflow`, `prebid`'s two (806 days),
`hmis` (555), `vikunja`'s `Typecheck` (542 days, 234 revisions), `airbyte`'s "lint check (info
only)" (470), `vscode`'s four (444), `crewAI` (373), `gh-aw`'s `Verify no compilation errors`
(373). The median, 155 days, is pulled by `maui`'s ten at 61 days; without that one commit it is
205. P4's bar was 180 and the median is below it; the reading is that a hidden check is not
transient and not ancient: half are older than five months, a quarter older than a year.

**Sixteen acquisitions, four acknowledged.** Eleven by `continue-on-error`, one by `|| true` (with
a `set +e`), four by a rewrite — `oven-sh/bun`'s "tweak github actions (#6195)" rewrote the node
test runner step in four workflows at once. The four the list finds: `cal.com` "Allow lint to
error but continue (**unblock** pipeline)", `promptfoo` "make staging redteam test
**non-blocking**", `mochi` ("ci: **skip** Maven publish gracefully", in the body), `neondatabase`
(`broken`, in the body). Read after the fact and not scored: `FastLED`'s "allow build even if failure on other
platforms" and `gh-aw`'s "advisory" say why in words the list does not have; `cal.com`'s
"revert: fix: lint" is a revert; "tweak github actions", "Linting (#2083)", two `codex/…` merges
and "run mac UI tests only in cmux-vm" do not say. The list found a quarter; a reader finds
about half. P2 is a MISS on the list as frozen.

**Fourteen repairs, twelve the instrument's.** Eleven removed a `continue-on-error`
(`no-continue-on-error` verifies each), one removed a `|| true` (`bun`'s "Maybe fix test
workflow": `strict-shell` verifies it), two do not agree: `novu`'s `Start WS` was rewritten (no
candidate verifies on the revision before) and `manaflow-ai/cmux`'s `Run UI tests` had its
`|| true` removed by the author while on the revision before the instrument's first stage
verifies nothing and its second verifies the guard, so the instrument's edit and the author's
differ. One lineage tells
the cycle in miniature: `calcom/cal.com` `lint.yml › lint › Run Lint` — hidden 2025-01-14 ("Allow
lint to error but continue (unblock pipeline)"), loud 2025-03-23 ("fix: lint (#20325)"), hidden
again the next day ("revert: fix: lint"), loud 2026-01-21 ("fix: make linting required for CI").
Four events, two repairs, and the instrument's edit each time.

**Sixty-four died hidden** — 29 with the step, 35 with the whole workflow — and 55 of them were
born hidden and removed without ever being loud: `mlflow`'s `Database tests - run` (born hidden, removed
2022-04-05 by "Separate database tests" — the same commit that created the one alive today, 1,630
days old), `vscode`'s `Run Smoke Tests (Electron)` ("Remove Build jobs for now", 2020), `airbyte`'s
`Run lint check (info only)` (removed and reborn hidden in the same "merge all summary status
checks" commit). Death is the common end of a hidden check; repair is the rare one: 64 to 5.

## 4. What this does not say

The reading is the frozen model's, revision by revision; what a step did in any real run is not
read. Only the default branch's mainline is read: a hiding on a branch that was never merged, or
that was squashed away, is not an event. A job renamed ends its lineages and starts new ones, so
a repair that came with a job rename is a death and a birth. The mechanism is read at the step; a
check hidden by a change elsewhere in the workflow is `context`, and none of the 16 was — which
says the hiding is local, and also that a non-local one would be read as no mechanism at all. The
acknowledgement list is a list of words (§3 reads what it misses). Age is measured to the pinned
HEAD's commit time, so `mochi`'s checks, whose pinned HEAD is days after they were written, are
days old. Four repositories of the 100 could not be read because they are no longer public; the
population is 96. Generated workflows were not followed: their history is a compiler's.

## 5. What ships

`styxx ci-audit --history`: for every finding, the commit it has been hidden since, read from the
checkout's own mainline — `hidden since 2025-03-14 (1a11430, 556 days): born hidden`, or
`acquired: continue-on-error — "ci: make lint non-blocking for now (flaky)" [the commit says:
non-blocking]` — with the number of revisions read, whether the clone is shallow at that
boundary, and whether the deadline capped the walk. For `owner/repo` the sparse clone is deepened
to 2019-08-01 first, blob-less, one batch. `styxx/ciaudit/history.py` is the living copy;
`tests/test_ciaudit.py` pins the frozen instrument at `4b961880…` and holds the living copy to it
on the scripted history.

## 6. Next

The acknowledgement list found a quarter and a reader finds half: a second list, frozen from the
words §3 names (`advisory`, `allow … even if`, `revert`), scored on a held-out population. The
`maui` commit — ten non-checks rewritten into ten hidden checks at once — is the shape a
`--history` on a pull request would catch before it lands: the differential audit, base against
head, reading births and acquisitions in the diff. And the third flavour SWALLOW-5 §6 asked for,
so that `selfxyz`, `mlflow` and `serviceradar` — 364, 1,630 and 33 days hidden — have a verified
repair to show.

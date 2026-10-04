```bash
pip install -U styxx==7.48.0
```

The section below is the `[7.48.0]` summary from [CHANGELOG.md](https://github.com/fathom-lab/styxx/blob/v7.48.0/CHANGELOG.md), where every summary line traces to a full entry. The entries themselves (24 carried from `[Unreleased]`, plus those written at the cut) follow it in the file.

---

What changed since 7.47.0, which did not carry `styxx.sworn`: the format, the verifiers and minters
built around it, the diff gate's measurements including the ones that went against it, and the
checksum series. Every line of this summary comes from an entry below or, under "Cutting this
release", from the commit it names, and carries the limits its source states on what the line
reports; where an entry lists what it does not say, that list is the complete one. Apart from those
cut notes, work merged with no entry is not summarised. The entries headed `[Unreleased]` until this
cut are kept whole, in the order this file carried them. Entries written at this cut say so in their
opening line, name the commits and files they were written from, and sit beside the work they belong
to.

**Sworn output — `styxx.sworn`, new in this release**
- v0.1: the author binds one sentence at write time to bytes it could not have written —
  `<sworn r="RECEIPT" k="KIND">…</sworn>` — and everything unbound is narrative, never accused. Span
  verdicts HELD / FAILED / UNRESOLVED / MALFORMED; a document that swore nothing is UNSWORN, never
  "no failures". No measurement of sworn output exists: nothing here is evidence that authors bind
  the sentences that matter.
- v0.2, after the adversarial pass: twelve attacks, four repaired, six not repaired and saying so
  beside the verdict, two surviving v0.1 unchanged. Its rules add leaf pointers and line anchors on
  `rN`, refuse hidden commitments and `quote` needles under 16 bytes over a whole receipt, and give
  every 0.2 manifest a declared rung (L1 or L2; L3 refused; a 0.1 manifest still loads and resolves
  at `undeclared`, never L2), printed and never checked. The coverage estimate is withdrawn: its
  denominator was a diff-claim detector.
- The sidecar battery: 49 attacks, `load_sidecar` refused none, 10 of the 49 succeeded by the frozen
  criterion (`text_smuggling` 5 times of 7), and 14 of the 42 that verify SWORN-HELD hold nothing.
  The headline now warns when nothing was checked. Renaming `SWORN-HELD` was proposed and then
  withdrawn: `UNRESOLVED` means the verifier could not look, not that it caught something. Two
  findings stand: the `load_sidecar`/`render` round-trip gap, and the rounding floor — a receipt of
  `0.4211` against a sentence printing `0.` is HELD, a verdict deliberately unchanged and now
  counted in the headline.
- Two gaps closed, each guard watched to fail before the repair: a line slice exempts a short needle
  only when it narrows the receipt (2 of 8 failed before, 0 of 8 after), and `path:` and `prereg:`
  receipts, and `absent` by path, now refuse bytes the manifest lists as agent-authored, as `rN`
  already did (3 of 10 before, 0 of 10 after). A file the agent committed under a name the harness
  never digested is still not caught.

**A second verifier, and a measure of what it can see**
- Conformance v0.1 (`conformance/sworn/`): every call the two sworn test files make into
  `styxx.sworn` recorded as bytes and addressed by one digest; a moved core refuses regeneration.
  `styxx.sworn.SnapshotTree` is new. The vectors cover what those two files, written by the
  builder, exercise, not the format, and agreement on them makes no verifier correct.
- The browser verifier v0.1 (`styxx/_data/sworn_verify.js`): 1689 vectors in scope, 1689 reproduce
  the verdict core digest, 0 disagree, 1929 skipped. Five disagreements were found by vectors, all
  repaired in the JavaScript. Those 1689 passed while two real defects, one of which changed a
  verdict, were live in the shipped file (aperture closure, below). `path:` receipts cannot be
  checked offline. A forger controlling the whole file passes both browser layers; the package at
  the named commit is the check.
- Differential agreement: 150000 generated documents through both shipped verifiers, 0 disagreements
  — agreement, not correctness: the same hands wrote both, and the HELD path is under two percent
  of spans. Mutation coverage then priced the generator: 70 viable mutations, 41 caught, 29 missed
  (0.5857). 6 of the misses lie outside the compared surface: the JavaScript verifier has no
  repository, so the tree-handle, sidecar and receipt layers are compared against silence.
- Aperture closure: a widened generator found 712 disagreements and two real defects in the
  JavaScript — a leading BOM the decoder stripped, and astral characters destroyed — with the
  browser verifier vouching HELD for a span `styxx.sworn` reports MALFORMED. After both repairs, 0
  of 150000 disagree. The conformance vectors passed 1689 of 1689 before and after both repairs:
  the vectors and the differential shared the blind spot. One boundary no repair can reconcile
  stays: Python holds two adjacent lone surrogates as two code points, where a JavaScript UTF-16
  string cannot tell them from one astral character.
- Suite power, for the layers no second implementation reaches: 51 viable mutants, 25 killed
  (0.4902); the tree layer 4 of 14, and 9 of the 25 kills rest on exactly one test.

**Receipts and the record**
- Receipt binding: every OATH certificate issued from now on names the bytes it swore to, and
  `corpus_audit --history` looks for them in git. Census over 213 certificates and 631 citations:
  630 `same`, 1 `at_issue`; 211 certificates stand over their sworn bytes and 2 do not, both the
  verifier having moved rather than a binding defect; eight certificates' documents were edited
  after issue, and all eight stand. The binding moves no verdict and re-issues no certificate. CI
  never runs that audit.
- `styxx.charon` v0.1, the ferry log: an append-only, hash-chained record in which every line is a
  verdict re-derived from bytes, 243 lines at ship; `verify` separates a core that moved with the
  instrument's bytes (SKEW) from one that moved under the same build (DRIFT). The chain binds order
  to its head, so a rebuilt or truncated log is detectable only against a head pinned outside it
  (`--expect-head`); nothing in it is immutable, and SAME_LINE 243, TAMPER 0 at ship is a
  determinism check, not a stability result.
- The sworn measurement's machinery, built and dry-run with nothing run as a measurement: the scorer
  is committed before any seat can speak, and the seat runners refuse until the operator's
  preregistration is committed. Every bar is still proposed and unsigned, and neither transport
  answered on this box, so the local substrate is undecided.
- `python -m styxx.undeclared WORKLOG DIFF` sets the harness's record of what it wrote beside the
  diff, in two report-only bands, ATTRIBUTED and UNATTRIBUTED, with the verdict `UNGATED`.
  UNATTRIBUTED is never called concealment, and its precision as a signal has never been measured.
- The prior-art survey: nineteen of nineteen sources read under a procedure frozen before any fetch;
  all six clauses OCCUPIED, none retired. Its limits travel with it: nineteen named sources are not
  the literature, the survey was one agent in one pass with no independent re-fetch, and a
  human-reviewed pass is owed before any of it goes outward.

**Shipped before the adversarial pass their entries owe**
- `styxx.harness` v0.1: adapters that turn a JUnit report, a GitHub event and diff, or a Claude Code
  hook payload into a `sworn/manifest/0.2`. They sign nothing and fetch nothing; the rung is the one
  the caller declared, and nothing verifies it; the hook payload shapes are not known to be stable.
  No adversarial pass has run against `styxx/harness/`, and its RESULT says it is not announced
  until one has. The Claude Code adapter is blind, permanently, to files written by shell commands.
- The sworn action v0.1 (`sworn/action.yml`, in the repository and not the wheel): mints the
  manifest after the turn, verifies every sworn document a pull request touched, and exits zero on
  every verdict — report-only until the measurement prices FAILED. The L2 rung it prints is the
  workflow's declaration, never checked, and on a pull request from a fork the manifest is minted by
  a party the claimant controls, unless the workflow is pinned to the base branch or the manifest is
  attested outside the job. It has not run on GitHub, and it owes the same adversarial pass and
  inherits every defect found in the adapters.

**The diff gate, measured on pull requests nobody here wrote**
- Eleven preregistered cycles against 71,016 agent pull requests from the AIDev corpus; four voided
  themselves. BC-2 removed 569 accusations and added none; COMPAT-1 and COMPAT-2 read 8,467 compat
  claims without one accusation; BIN-2 and PATH-1 repair the reading, and HARNESS-1 found 16
  accusations were the harness's, not the agents'.
- An erratum to the entry "the gate was pointed at itself", which prints EXTERNAL-5's "19 were the
  harness's" as written: its RESULT corrected that on 2026-09-16 (956d8deb). Of the 19 overturned
  items, 9 are the fold carrying merge traffic, 2 the dataset's 300-file per-commit cap, and 8 are
  explained by neither — the pull request is a different object today than in the dataset.
- The measurements that went against it ship too, and this lab graded them: BENCH-1 and BENCH-2
  INVALID; the lab's own hand adjudication found 9 of 11 accusations wrong (precision 0.18), PATH-1
  moved it to 0.25, and SCOPE-1, which would have reached 1.00 by withholding, was abandoned with
  nothing shipped. DECIDE-1, 100 claims read by hand by the lab, whose RESULT states that conflict
  of interest and publishes every call: 71% decidable from the diff (corpus-weighted; `only_touches`
  52%, 95% interval [33.5%, 70.0%], n=25, and 44.0% if its two CONTESTABLE calls count as not
  decidable), while the instrument returns a verdict on 5.7% of `only_touches`.
- V14, new since 7.47.0 and preregistered before its repairs existed and before the held-out split
  was read: it removed the named false accusations — 0 paths gained an accusation over 71,016 pull
  requests, and corpus-wide path accusations fell from 4,427 to 1,344 — and a fresh blind panel
  upheld 16 of 100 held-out accusations, precision 0.16 against the 0.95 floor. So the path
  accusation stays withheld, as it was in 7.47.0, and the lab is not repairing the class again. Its
  two repairs ship on by default, as flags: `V14_CONTAINMENT_TOUCH`, so a touch claim inside a
  containment preposition is no longer read as a claim, and `V14_BARE_NAME_ABSTAIN`, so a bare file
  name absent from the diff abstains, a recall sacrifice the PREREG named as one.
- DECLARE-1, preregistered after those cycles: a pull request body may declare its claims in one
  fenced `styxx` block, read by the same reader as prose, so the two cannot disagree, with every
  refusal pinned by a test. The gate written to measure whether it was worth building could not see
  the answer, because its items were the ones where extraction already worked. On DECIDE-1's
  hand-read claims the instrument is silent on 49 of the 76 decidable, and on 13 of 13 decidable
  `only_touches`. Declaring fixes extraction, not meaning; adoption is zero.
- "tests pass" is read from bytes: `--evidence` and `--commit` hand a JUnit report or a test-result
  attestation to `styxx.evidence`, whose only verdicts are VERIFIED and UNCHECKABLE. The `--run`
  accusation is deleted, not flagged off: a nonzero exit is UNCHECKABLE, because it is also pytest's
  "no tests collected", a misspelled command or a flaky test. The branch's zero accusations are
  structural, not a measured cost of deleting it: the corpus never ran the branch. All 5,514 corpus
  `tests_pass` claims were read with no command (`run=None`), so no accusation could arise, and no
  test covered the branch.
- `papers/closed-model-frontier/bench{1,2}_dataset.jsonl`: 604 claims from 568 pull requests with
  the live diff's sha256 per row, styxx's own verdicts deliberately absent; `bench_reproduce.py`
  scores styxx or any other checker in one command. Re-fetched on 2026-09-18, 566 of 568 diffs
  matched and 2 did not, because those pull requests had gained commits.
- `web/gate/diffgate.js` carries the COMPAT-2 reading: 3,220 pairs, 6,933 claims, 0 disagreements,
  and after DECLARE-1 3,230 pairs, 6,945 claims, 0 disagreements; `tests/test_port_is_current.py`
  fails when the port goes stale.
- New doors: `python -m styxx.diffgate --pr <url>` gates a public pull request with no checkout, and
  `integrations/git/commit-msg` refuses a commit whose message contradicts the staged diff. The same
  gate ships as the console script `styxx-diffgate-commit-msg` and the pre-commit hook
  `diffgate-commit-msg`, at a rung its README calls weak; `styxx-diffgate-commit-msg --help` exits
  with a traceback, and the README names an open class of false VERIFIED, #101.

**Checksum, the plate, and the sand check**
- `styxx.checksum` fingerprints a model on a hashed 48-item canary set, beside `styxx.observatory`,
  `styxx.beacon`, `styxx.epoch` (a primitive only; no epoch exists), `styxx.clock` and
  `styxx.challenge`. The sworn RESULT, on one 135M model on cpu, on the machine that ran it: the
  same weights reloaded read SAME at 0 nats/token, per-tensor int8 DRIFT at 1.51 [1.17, 1.90],
  random weights 9.50. The deploy-scale PREREG is frozen and unrun, its seal pending, with a
  CORRECTION beside it that fixes how H1 is read before the run.
- `python -m styxx.plate <sha256>` renders a hash as a Chladni figure and `styxx.geoplate` renders a
  representational dissimilarity matrix; new extra `styxx[plate]`. The extra serves modules the CLI
  does not register, the eight new modules of the series sit outside the guarded public surface,
  and the plate of the 2026-09-12 sworn chat update is unbacked until that update is committed
  under `papers/chat/`. A plate is a picture of the number: it says nothing about the document the
  hash came from, and nothing in the series is a measurement.
- The series red-teamed before it was pushed: ten adversarial reviewers on 2026-09-13, none of whose
  blocker or defect findings was refuted; every repair a new commit, no sworn document and no frozen
  PREREG edited. One PREREG claim is withdrawn by CORRECTION (that the beacon-drawn values "could
  not have been computed before the slot existed"; they can), fourteen false or overstated
  statements in the pass-4 SURVEY are corrected, and an ERRATUM records that the sand survey's
  fetchers presented as a browser, against the lab's rule. `styxx.stranger` runs the seven checks as
  one command. A second machine reproduces the verdicts, not the magnitudes: int8 1.51 → 1.41
  nats/token.

**Also in this release**
- Token-level h v3, preregistered and HELD: accusations handed by a table header are genuine at
  0.9515 (n=165), those handed by a trigger word in the line at 0.6391 (n=169), after two INVALID
  runs shipped as INVALID. An exploratory split by token kind, never a result, puts the
  kind-adjusted gap at 0.1695, with one repository supplying 184 of 334 rows.
- OATH v0.14's `V14_RANGE_SANITY_REPORT` ships default OFF; its RESULT recommends the flip, and this
  release does not make it.
- The frequency arc's efficiency control reads CAPACITY_IN_DISGUISE, and `styxx/resonance.py` ships
  the resonance profiler, which says in its own output that it diagnoses a model and never licenses
  a primitive; `corpus_audit` compares verdict classes, so an `N uncovered` suffix no longer reads
  as drift.

**Cutting this release**
- The italic "Staged for 7.48.0" note that opens the entry "the gate was pointed at itself" is left
  as written and resolved here. The step it names was done in commit 989abdb2
  (`papers/sworn/NOTE_sworn_conformance_regenerated_for_7_48_0_2026_09_25.md`): 15 `receipt_check`
  vectors took new ids, 0 moved, no expected outcome changed, and 3620 of 3620 replay. The cause it
  states is wrong: `provenance.styxx_version` sits outside `set_sha256`. The digest moves because
  `verifier.styxx_version` sits inside the verdict-receipt digests those vectors carry. The
  conformance RESULT says the set "is never regenerated in place"; the NOTE records that conflict
  and leaves the RESULT unedited. A CI run on the regenerated set is owed. *(Erratum, 2026-09-25: that run happened before the tag. Runs 36167419000 on 99118487 and 36173712926 on 1218dbad passed with C7 included.)*
- `styxx.challenge` calls a version-only difference version skew, not a disagreement: when a
  committed 7.47.0 receipt is re-run at 7.48.0 with the same `sworn.py` and only the version
  differs, the record says `version_skew: true` and `same_build: false`, names both versions, and
  says the verdict and every span agree. It stays a CHALLENGE (exit 3), because the digests differ.
  The same verifier build now means the same `sworn.py` and the same styxx version, so a version
  difference alone can never give `agree: false` with `same_build: true`, the shape
  `papers/plates/SAND_CHECK.md` pays for; a real disagreement under both still gives it. Both
  challenge test modules had been skipping at collection in a depth-1 checkout, so CI could not see
  the bump break five of their tests; a depth-1 clone whose origin is reachable now unshallows and
  runs all 20.
- An erratum to the kept entry "the gate was pointed at itself": its opening sentence, "The release
  where the diff gate stopped being graded by the people who wrote it", is superseded by this
  release's headline. BENCH-2's accusations and DECIDE-1's 100 claims were hand-graded by this lab,
  and DECIDE-1's RESULT states that conflict of interest (G-D1-4). The entry stays as written.
- `README.md`, which becomes the PyPI page and cannot change until the next release, was audited
  against the tree in 99118487. Its 69 repo-relative links, dead on pypi.org, now point at the
  v7.48.0 tag on GitHub and resolve once the tag exists; 53 lines changed only in their links, and
  the commit message counts those 53 as links. 37 lines were corrected against the files they cite,
  among them the arXiv row, which used a word the charter forbids, and the path-claim history, which
  now says the accusation is disabled, not deleted; two lines were added. `pyproject.toml` drops the
  keyword `hallucination-detection`, per the charter.

---

## Integrity

| file | sha256 |
|---|---|
| `styxx-7.48.0-py3-none-any.whl` | `624e0715e1f7e23dc17fd2aabf21b942caebec8f094d2a620f8f29e8811be1f0` |
| `styxx-7.48.0.tar.gz` | `48208cdbe60dd80285110b2d845f52c52fa0c653cf1bb6115560ae937acb7b0d` |

Built by `publish.yml` on Ubuntu from tag `v7.48.0` (commit `1218dbad`), uploaded to PyPI, and attached here. Checked after publishing: both files as downloaded from PyPI hash to these values and are byte-identical to the assets on this page, and `pip install styxx==7.48.0` into a clean virtual environment imports 7.48.0 from site-packages with no version mismatch.

🤖 Release prepared with [Claude Code](https://claude.com/claude-code)

# prereg-ci v0: known misses

What v0 lets through, and why each is left for later. The README prints this list beside what the check
refuses, so a reader of a PASS knows the mutations it was not built to see.

The list comes from step B8 of the design: a mutation battery of 139 cases, written and run against
`9b32b76d` by a reader who wrote none of the code, and a fresh reading of the branch by a second reader.
Sixteen mutations the tool must refuse passed there with exit 0, and the fresh reading found two majors.
The fix round closed every class but one (M5 below); each closed class is pinned in
`tests/test_prereg_ci_battery.py` by a case that fails at `9b32b76d`.

## Not refused in v0

**M5. A locked scorer whose verdict reads the clock, the entropy pool or the environment.** A scorer that
types `SHIP` after a date (`time.time()`, `datetime.date.today()`), or that reads `os.urandom` or an
environment variable the run passes (`SYSTEMROOT` on Windows), passes. Every locks reading prints the
boundary beside its verdict: "the locked scorer's clock, entropy and environment: not watched". Why not
in v0: the run's watch is an audit hook in the scorer's own process, and a clock read raises no audit
event. A wrapper patched over `time.time` would see neither a read in C (`datetime.datetime.now`) nor a
read through the standard library, and it would refuse honest scorers: `logging` reads the clock when it
is imported, and `styxx/checksum.py`, which the checksum scorer locks, reads `time.gmtime()` for its
`created` field. A fixed clock and an empty environment belong to the OS-level sandbox of v1 (DESIGN 3.8).
Pinned as the boundary by
`test_boundary_a_locked_scorer_whose_verdict_reads_the_clock_passes_and_the_boundary_says_so`.

**A landed locked scorer still runs as code.** Since the fix round a scorer runs only from blobs the base
commit holds, so a pull request does not run its own scorer as the judge (`SCORER_NOT_RUN`, exit 3,
until the PREREG and its locked files land). A scorer that has landed runs with `python -I -S`, a
timeout and, where a Linux runner permits it, no network; it is not a sandbox (DESIGN 7, risk 5), and the
checkout keeps the job's token in `.git/config` for the partial clone's fetches. The Action's description
says to run it on `pull_request`, never under `pull_request_target`, and with no secret in the job.

## What the fixes do not reach

- **M1.** `core.rescore` and `core._decision` keep the reading of `styxx.protocol`'s scanner, which the
  committed conformance vectors pin (a fence hidden from it reads `NOT_GATED`,
  `conformance/prereg_ci/vectors_compare.json`). The refusal of a gates-like fence a renderer shows and
  the scanner does not count, `FENCE_SHADOWED`, is the check's, the census's and the lint's, and C0
  reads one added beside a sealed fence as `DECISION_EDITED_AFTER_SEAL`. A caller of `rescore` alone
  reads `core._shadowed` beside it.
- **M2.** A manifest that a merged pull request deleted is refused when it is added again with its lines
  changed, and `census` prints the deletion (`MANIFEST_REWRITTEN`, "deleted it"); a later pull request
  that leaves the manifest absent passes, as one that leaves any committed file alone does.
- **M3.** A receipt is a `.json` file. One a pull request writes under another suffix is refused
  (`RECEIPT_NOT_JSON`), and one committed under another suffix is held immutable where a pull request
  changes or deletes it. The readers of receipts in a tree or in history (C4's earlier receipts, C8 over
  the tree, `census`) read `.json` files only.
- **M4.** C0 and C8 over the tree read every `.md` file and every path a receipt or a seals manifest line
  names as its PREREG, whatever its suffix. A file under another suffix that nothing names is not a
  sealed decision, though its bytes hold a fence (`tests/test_protocol.py` in styxx quotes one).

## Gaps the battery argued, which the design does not speak to

None of these is refused in v0. Each would change how committed receipts or committed conformance
vectors read, so each waits for a census of styxx's main branch under the new rule.

- **O03.** Numbers committed on a path before the seal, and the `prereg` key added there after it: R is
  the oldest commit at which the path holds a receipt, so the order reads `AUTHOR_ORDER`. The report
  prints the receipt as changed (`M`) and the blob history from the order commit.
- **O11, O32r.** A receipt holding a metric key twice, or a key beside a look-alike of it carrying a
  zero-width space: receipts are read with `json.loads`, which keeps the last key, while the gates
  block and the manifest are read strictly.
- **O32, O31.** A look-alike metric path in the sealed table, and two outcome rows whose verdicts are
  look-alikes: the gates block refuses a non-ASCII key, not a non-ASCII string value.
- **O30z, O30s, O30n.** The withdrawn sentence with a zero-width space, a soft hyphen or `&nbsp;` inside
  it: licence patterns run over whitespace-collapsed text, and a renderer prints all three as the
  sentence (DESIGN 7, risk 3: licences match strings).
- **O50b.** A receipt written with a byte-order mark and named by no RESULT reads `RECEIPT_UNREADABLE`, a
  NOTE; named by a RESULT it is refused.
- **O39, O39e.** The entropy pool and the environment, as M5.

## Boundaries the design states

Each passes, as the design says it does: a decoy withdrawal word in the window (O29, the window rule of
`scripts/surface_lint.py`); emphasis inside the withdrawn sentence (O30); a copy of the PREREG with a
looser bar arriving in the pull request, which reads `AUTHOR_ORDER` and is refused under
`require_time = "T3a"` (O34); a RESULT whose verdict is in prose only (O47); an amendment whose `seen`
says less than the record shows (O53: `seen` is the author's statement, printed, not checked); a receipt
naming an ungated twin of the PREREG (O33n: `NOT_GATED` is never refused); an amendment line with `seen`
null in the record a manifest imports when it begins (B06i).

## From the fresh reading, not changed

- **An oversized `.json` that a pull request touches makes the check UNCHECKABLE (exit 3).** Kept: a
  receipt over the cap would otherwise go unpaired, and this layer cannot tell a receipt from a data file
  without reading it (DESIGN 3.3: a receipt over `sworn.MAX_RECEIPT_BYTES` is UNCHECKABLE).
- **The Action's own steps pin `actions/checkout@v4` and `actions/setup-python@v5` by tag.** This build
  makes no network call, so it cannot read the commit each tag names; the operator pins both by sha
  before the Action is published, as DESIGN 7, risk 10 asks of a caller.

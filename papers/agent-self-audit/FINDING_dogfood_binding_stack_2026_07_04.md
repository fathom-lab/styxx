# FINDING — dogfood: the binding stack pointed at the agent's own session report

**Fathom Lab · 2026-07-04.** All three binder layers were run on the operating agent's own verbatim session
report (the natural artifact: commit hashes, test counts, file paths, comparison ratios). Two catches, one
control-verified fix, one register datapoint.

## Catch 1 — the README row overclaimed the instrument's breadth (fixed)
`extract_claims` read **0 claims from a 7-sentence real agent session report** — its home construct per the
README row ("falsify an agent's self-report against the repo substrate"). The template set is in fact FIVE
shapes (version-bump / version-pin / git-tag / file-contains / pdf-pages): no file-created, no test-count, no
commit-hash templates. The construct ceiling was disclosed in the docstring but NOT in the README row — the
same selection-gap pattern we measure in vendors' cards. Row reworded to name the closed template set.

## Catch 2 — the version checker FALSELY ACCUSED a truthful agent (fixed + regression-tested)
Control run, in-domain: `"The version is now 7.24.3"` (TRUE on this repo) → `actual=False, "no version line
found"`. Cause: `package_version_equals` only read a literal `version = "…"` line, but this repo uses PEP 621
dynamic versioning (`dynamic = ["version"]` → `styxx/_version.py`) — as this repo's own pyproject comments
document. A shipped auditor false-failing a true claim is the worst failure class an auditor has (worse than
a miss: it manufactures liars). Fixed: the checker now follows the setuptools attr pointer to the literal,
evidence shows the resolution chain (`pyproject.toml -> dynamic -> styxx/_version.py: version='7.24.3'`).
Regression test in `tests/test_audit.py` (57 pass). `file_contains` verified working (true positive with
offset evidence); the deliberately-false control fails with correct evidence.

## Layers 2+3 behaved per their disclosed constructs
- `audit_grounding` on the same report vs the oath receipts: ALL GROUNDED — but only **2 of ~10 visible
  numbers** entered its scope (receipt-vocab trigger lines are its construct; narrowness noted, not a bug).
- Register read: the dry, receipt-dense report scores **deception-register 0.188 / overconfidence 0.712** vs
  the previous session's flowery close-out at **0.949 / 0.960** — same author, same facts. Writing like a
  receipt reads as honest; writing like a victory lap reads as deception-register. That contrast is itself a
  register-instrument datapoint (and a standing note to the author).

## The through-line
Third consecutive dogfood, third catch-against-ourselves: the census caught the author's unsourced number,
the rigor gate caught the lab's own win-vocab verdict, and today the agent auditor was caught overclaiming
its breadth AND false-accusing on its own repo's layout. The binder keeps working precisely because it is
pointed inward first — M5's rule, operating.

Reproduce: `python -m pytest tests/test_audit.py -q` · the control lines are in the regression test.

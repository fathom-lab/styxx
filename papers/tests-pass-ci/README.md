# tests-pass-ci

How often do coding agents' "tests pass" sentences in pull request descriptions disagree with CI at the head commit?
This arc holds the preregistration of one measurement of that, in two parts:

- **A1** re-labels the published items of [amos689/testreceipt](https://github.com/amos689/testreceipt) (MIT, at
  `95864ef9`), which reports 42 of 481 (8.7%) for 2025 AIDev pull requests and, for June to September 2026, 2 of 644
  (0.3%) whole-suite and 30 of 644 (4.7%) part claims, labelled by two model judges. The lab's panel is one paid human
  seat from outside the lab, a Claude seat and a non-Claude seat, checked on 30 sealed decoys per packet.
- **A2** measures agent pull requests opened 2026-11-02 to 2026-11-29 in repositories with 100 or more stars, found by
  testreceipt's six agent signatures: whole-suite and part-suite contradictions per agent, the share of repositories
  whose CI uploads machine-readable test reports, and a candidate accusing rule measured report-only against a sealed
  0.95 floor.

| file | what it is | state |
|---|---|---|
| `PREREG_tests_pass_ci_2026_11_02.md` | the plan, its gates block (14 gates, 45 rows) and every procedure (Part II) | draft, unsigned; to be frozen by 2026-11-06 by the operator's word, the signing commit and the pull request's opened-at |
| `FORECAST_tests_pass_ci_2026_11_02.md` | the lab's forecast of every row and quantity, scored by the RESULT | sealed by its sha256 in the PREREG §13 |
| `collect/` | the collector: frame, draw, read, screen, CI result, rule R, report check, under the seal guard; the packet and decoy builder, the seat runners and the scorer | its combined digest fills the PREREG's lock table at the signing commit |

Before the freeze no one in the lab reads CI conclusions, check runs, statuses or job logs of any pull request opened
on or after 2026-11-02, nor classifies a claim in one; the PREREG §0 records every read of the window by metadata, and
the collector refuses the rest by name until the seal opens (Part II P21). Nothing here accuses any pull request, agent
or vendor: A2's per-pull-request labels stay off the repository (§12), and what none of the outcomes says is written in
§1 and §18.

Tests: `tests/test_tests_pass_ci_prereg.py` (the gates block, its rows and receipts, the bars' arithmetic, the lock
table's prompt, summary, vendor and forecast digests, the seal guard against this file's own text) and
`tests/test_tests_pass_ci_collector_*.py` (the collector, offline, against fakes). Every file in this folder is
`-text` in `.gitattributes`, so its digests are the same on every checkout.

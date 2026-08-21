# RESULT — SP-EXT: 2 confirmed silent-pass defects in code we did not write

**Every gate points the same way: this is a two-case corpus, it licenses no claim
about how common the defect class is, and it is NARROW.** It is published because
two externally-anchored cases is two more than this project had this morning.

Prereg: `PREREG_sp_external_corpus_2026_08_21.md`, frozen at `38b8428` before any
commit was read. Raw: `out_sp_ext_candidates.json`, `out_sp_ext_shaped.json`,
`out_sp_ext_strict_recheck.json`, `out_sp_ext_final.json`. Corpus:
`benchmarks/silent_pass/external.json`.

---

## the two cases

### SPX-2026-0001 — `Giskard-AI/giskard`, `SuiteResult.pass_rate`

```python
denominator = len(self.results) - self.skipped_count
if denominator == 0:
    return 1.0
```

A suite that evaluated **nothing** reported a **100% pass rate**. The pre-fix
docstring:

> `pass_rate : float` — Fraction of non-skipped scenarios that passed
> **(1.0 when all scenarios are skipped).**

**The flattering default was written down as the contract.** Consumers, confirmed
by the fix's own diff: the console summary, the **Hub upload payload** (shipped
`1.0` as a genuine `success_rate`), the project's README example, and user CI
gates. Fixed upstream as a **breaking change**. Rejected by 0 of 3 reviewers.
SP-2, unanimous.

### SPX-2026-0002 — `UKGovernmentBEIS/inspect_ai`, `_darwin_scale_factor`

```python
try:
    from AppKit import NSScreen
    screen = NSScreen.mainScreen()
except Exception:
    return 1.0
```

A bare `except Exception` swallowed `ModuleNotFoundError` for the optional
`pyobjc-framework-AppKit` package. **1.0 is a legitimate scale factor** — every
non-HiDPI display measures exactly 1.0 — so nothing distinguishes *"measured, and
this screen is not HiDPI"* from *"the probe never ran."* A HiDPI screen silently
treated as 1× mis-scales the browser tool's coordinates with no error anywhere.
Rejected by 0 of 3. SP-2, unanimous.

Recorded with a caveat this corpus should keep making: **the fix still returns
`1.0`.** It adds a cached `logger.warning`, so the absence became visible to a
human reading logs and not to a program reading the value. R3 is satisfied
because the frozen rule lists *"a warning"* — but a consumer that thresholds the
number still cannot tell the two cases apart. The same file carries a second,
untouched `except Exception: return 1.0` for the Windows DPI probe. That is
recorded, not claimed.

## the gates, applied

| gate | value | verdict |
|---|---|---|
| **G1 YIELD** — ≥12 or no prevalence claim | **2** | **triggered.** No claim about how common this is, in the title and everywhere else. |
| **G2 ACCEPT RATE** — two-sided | 2/8 = **25%** on the query as frozen; 2/17 = **11.8%** as actually run | see below |
| **G3 SPREAD** — <4 repositories | **2** (giskard, inspect_ai) | **triggered. NARROW.** No cross-project claim. |
| **G5 RECALL** | unknown, by construction | SP-EXT is a lower bound and is never quoted as a rate. |

## the protocol deviation, and its measured cost

The preregistration froze Q2's regexes verbatim. **My implementation was wider
than the frozen text** — it added an assignment alternative (`x = 0.0`) on the
removed side and six extra tokens on the added side, none of which appear in the
prereg. That is a query edit, which G4 and G6 forbid, and it was not deliberate,
which does not matter.

Re-testing the same candidates under the **frozen** regexes
(`out_sp_ext_strict_recheck.json`):

- **8 of 17** qualify under the query as written.
- The 9 the widening added were **all rejected** by adjudication.
- **Both accepted cases survive the strict shape.**

So the deviation cost 9 candidates' worth of adjudication and **did not
manufacture a case**. The preregistered accept rate is **2/8 = 25%**, which
triggers neither bound of G2. The as-run rate is **2/17 = 11.8%**, which would
trigger the low-end rule.

**I am reporting the preregistered number as the primary one, and I am aware that
it is the more favourable of the two.** The reasoning is that G2 is a gate on the
frozen procedure, not on my mis-implementation of it. A reader who thinks the
as-run number is the honest one has it in the table above and can apply G2
themselves. At n=8 neither figure supports much: the 95% interval on 2/8 spans
roughly 7%–59%.

## I overturned one of the protocol's own acceptances

`inspect_ai 34beafda81` was **accepted 2-of-3** — two reviewers could not reject
it. Reading the pre-fix source directly:

```python
if ans1 is None or ans2 is None:
    return False        # → Score(value=INCORRECT)
```

When the math scorer could not parse **the target**, it marked the model
**wrong**. That fails *closed*. R2 requires a value indistinguishable from a
**healthy** measurement, and `INCORRECT` is not one — it is the alarming
direction. The single reviewer who rejected it was right and the majority was
wrong.

The corpus records the overturn, not just the outcome. *"Every verdict is
published so a reader can overturn a label rather than trust it"* is only worth
writing if the first reader actually does it. **A 2-of-3 majority is not evidence.
The source is.**

## what the harvest shape says

415 Q1 candidates → 17 under my widened filter → 8 under the frozen one → 2
confirmed. The bottleneck is the shape filter, not the message regex, and with
`n=8` the harvest has still not really been characterised.

**Q2 standalone finally became runnable.** Against `--filter=blob:none` clones,
pickaxe timed out at 10 minutes on the smallest repository in the set; against a
full clone the identical query takes **13 seconds**. The obstacle was lazy blob
fetch, not the query, and the preregistration named a method without anyone
checking it would execute — the same error as freezing a gate that cannot pass.
Its yield is not in this result and its candidates are unadjudicated.

## what is NOT claimed

- **Nothing about prevalence.** Two cases is two cases. G1 and G5 both forbid a
  rate, and no sentence of that form appears in this project's writeups.
- **Nothing cross-project.** Two repositories is NARROW by the frozen rule.
- **Nothing about the 14 rejected candidates being *safe*.** They failed *this*
  inclusion rule; several are real bugs of other kinds.

## what it does establish

The defect class exists outside this repository, in two independent projects that
build evaluation infrastructure — including one maintained by a government AI
safety institute — and in both cases the maintainers themselves shipped a fix.

**A pass rate of 100% from zero evaluations, uploaded to a dashboard as a real
success rate.** That is the class, in the field, in someone else's words.

And on entry the corpus immediately falsified our own screen: `styxx.flattering`
**misses SPX-2026-0001**, on two grounds, the second being C4 — polarity from name
morphology — which the flattering adjudication had named abstractly before SP-EXT
existed. `flattering` is frozen and is not being edited to catch it; the miss is
pinned in `tests/test_external_scoring.py`.

## cost, and two errors worth naming

52 adjudication agents (one session-limited run plus a resume), ~4M subagent
tokens, 14 full clones at ~4 GB.

Two fabricated file paths, both mine, both in corpus entries, both caught by
running `git show --name-only` rather than by re-reading my own text: I wrote
`giskard/core/suite.py` (real: `libs/giskard-checks/src/giskard/checks/core/result.py`)
and `src/inspect_ai/tool/_tools/_web_browser/_resources/scale.py` (real:
`src/inspect_tool_support/src/inspect_tool_support/_remote_tools/_web_browser/scale_factor.py`).
Neither resembles the truth. **Paths get read from git, never inferred** — in a
benchmark other people are meant to rely on, an invented path is not a typo.

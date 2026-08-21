# PREREG — does the flattering-default defect exist outside code we wrote?

**Frozen and pushed before the external scan was run. Written before any
third-party output was observed.**

---

## the question

Every SILENT-PASS result this project has published so far comes from **auditing
its own repository**. That is the load-bearing weakness of the whole program: a
defect class found only in the code that named it is indistinguishable from a
house style.

This run asks the only question that can settle it:

> **Does the flattering-default pattern occur in Python written by other people —
> including AI evaluation and safety tooling — and when it does, is it a real
> defect or a false alarm?**

## the instrument

`styxx.flattering` (`styxx/flattering.py`), a static AST screen for:

- **R1** — an emptiness guard among a function's first statements whose branch
  returns a flattering constant. `if not samples: return 0.0`
- **R2** — a conditional expression whose empty-case fallback is a flattering
  constant. `n_hot / n if n > 0 else 0.0`

"Flattering" is polarity-dependent and **polarity is required**:

| evidence | claimed? |
|---|---|
| name carries polarity (`risk`, `trust`, `is_valid`, `entropy`, …) and the constant is the good end | **TIER-A — claimed** |
| the returned literal is itself a healthy string (`"pass"`, `"ok"`, `"steady"`) | **TIER-A — claimed** |
| emptiness guard + constant, no polarity evidence | TIER-B — **counted, never claimed** |

Honest handling of an empty input is never a hit: `raise`, `return float("nan")`,
and fail-closed (`return 1.0` from a `risk` function) are all correct and all
excluded. Acceptance tests: `tests/test_flattering.py`.

**Rules were derived from the styxx corpus (`benchmarks/silent_pass`) only, and
frozen before the screen was pointed at any external package.**

## hypothesis

**H1** — the flattering-default pattern is a **general** defect in Python
measurement code, not a styxx idiosyncrasy.

## gates, frozen

**G1 — PRIMARY.** Of adjudicated external TIER-A hits, **≥ 20% GENUINE**.
*GENUINE* means: a caller could plausibly branch on or threshold this value, so
an empty input produces **the same reading as a healthy measured one**.
Below 20% → H1 **NOT SUPPORTED by this instrument**, published as such.
(≥ 40% will be described as strong; the gate is 20% and does not move.)

**G2 — VALIDITY / POWER.** If total external TIER-A hits < 15, the verdict is
`INVALID__UNDERPOWERED`, **not a null**. A proportion cannot be estimated from
fewer than 15, and an underpowered cell reported as a negative is the same lie as
an unmeasured value reported as a pass.

**G3 — ANTI-TUNING.** The detector is frozen at the git commit recorded in the
RESULT. **Any edit to `styxx/flattering.py` after the scan voids this run**, which
must then be re-run under a new preregistration. No rule may be widened to catch
a case seen in external output, and none narrowed to drop one.

**G4 — TWO-SIDED REPORTING.** The BENIGN rate is reported with equal prominence.
**If BENIGN ≥ 80%, that is a finding about the detector, and it goes in the
title** — not an appendix. A screen with a 4-in-5 false-alarm rate is not a
screen, and saying so is the point of the gate.

**G5 — ADJUDICATION INTEGRITY.** Each TIER-A hit is adjudicated by an
**independent adversarial reviewer prompted to argue the case is BENIGN**, given
the surrounding source. GENUINE is recorded only when that refutation fails.
Uncertainty resolves to BENIGN — conservative **against** H1. Every verdict is
published with its rationale so a reader can check it.

**G6 — SCOPE.** styxx's own hit rate is **in-sample** (the rules came from it) and
is reported for reference only. It is never evidence for H1.

## sampling

Every third-party package on this machine with ≥ 40 `.py` files (91 packages),
scanned in full. Test files excluded. If external TIER-A exceeds 60, a random
sample of 60 is adjudicated, `seed = 20260821`, and the sampling is disclosed.

## what this run cannot show

Nothing about *severity* — whether any hit has ever caused a real incident.
Nothing *causal*. Nothing about non-Python code. And nothing about the
**interior-degenerate** class (`RESULT_contract_sp6_2026_08_21.md`, 0/2), which no
static boundary screen reaches; this run is confined to the boundary-visible half
of SILENT-PASS by construction.

## stopping rule

One run. If G1 fails, H1 is not supported **by this instrument**, and the number
is published. There is no second scan with widened rules — that is how a program
fabricates a finding.

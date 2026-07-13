# RESULT — OATH v0.5 certifier precision: five of six classes SHIP, class A dropped by the battery

**Fathom Lab · papers/autopilot · 2026-07-13. Verdict against the frozen gates of
`PREREG_oath_v05_precision_2026_07_13.md` (frozen `7e54d0b`, committed BEFORE any `certify.py`
change). Six severable false-positive-elimination classes were composed, measured against the
frozen bars, and the one class the mutant battery showed to be harmful was dropped by the
pre-committed severability procedure. The gating property — a verifier that never falsely accuses —
holds.**

## Verdict (frozen string)

- **`PARTIAL__oath_v05_classes_dropped`** — five classes ship (B unit-suffixed range, C arXiv id,
  D @-parameter, E derived-percent VERIFY, **F self-scoped `n=`**); **class A (approx-notation
  ≈/~/∼) DROPPED** per the severability procedure. Every frozen bar passes on the shipped
  composition.

## The bars (all pass; shipped = A off, B/C/D/E/F on)

| bar | requirement | result |
|---|---|---|
| **P1** six-doc UNGROUNDED | 11 → at most 4 | **3** (`cycle38_v05_p1_sixdoc_result.json`) |
| **P2** battery catch | at least 116 of 269 | **117** (`cycle38_v05_battery_result.json`) |
| **P2** battery false-verify | at most 26 | **26** |
| **P3** validator D1 | at least 16 | **16** (`OATH-V0-VALID`) |
| **P3** validator D2 | 0 | **0** |
| **P4** 13-doc recert artifacts | 0 | **0** — 3 UNGROUNDED, all the SAME real provenance gaps as the v0.4 baseline (`cycle38_v05_p4_recert.json`) |
| **P5** pytest | green | **1694 passed, 8 skipped** (CPU-forced; the one GPU-loading test deselected while the B7 run holds the card) |

## Why class A was dropped (the battery arbitrated, exactly as pre-registered)

The prereg named the battery as the arbiter and pre-committed a leave-one-out drop procedure. The
per-class sweep (`cycle38_v05_class_sweep_result.json`, `cycle38_v05_class_sweep.py`) measured the
marginal battery effect of removing each class from the full six-class composition:

| drop class | Δ catch | Δ false-verify | Δ six-doc FP |
|---|---|---|---|
| **A approx-notation** | **+3** | **−6** | +1 |
| B unit-range | 0 | 0 | 0 |
| C arXiv-id | 0 | 0 | 0 |
| D @-parameter | 0 | 0 | +1 |
| E derived-percent | 0 | 0 | +1 |
| **F self-scoped n=** | +6 | 0 | +5 |

With all six on the battery read **caught 114 / false-verify 32** — BOTH bars missed. Class A was
the sole offender: keeping it **cost 3 catches and added 6 false-verifies** (its `~` tilde abstains
approximations that were being correctly caught), for one six-doc false positive eliminated.
Dropping A alone cleared both bars (caught 117, false-verify 26). No other class had an adverse
delta; class F — the self-scoped `n=` fix — was pure gain (+6 catches, 0 false-verify cost, five of
the eliminations) and is the change's headline.

**A second, independent reason the drop is correct:** the three surviving P4 UNGROUNDED are real
derived-bound gaps written with `~` (`RSA ≤ ~0.56 → R² ≤ ~0.16`; a bulk agreement `0.50`). Class A
would have SUPPRESSED those behind an approximation excuse — turning three honest provenance flags
into silent abstentions. Dropping A keeps the certifier flagging real gaps loudly, which is the
whole point of the oath.

## What shipped (five classes, each one clause in `styxx/certify.py`, gated on a `V05_*` flag)

- **B — unit-suffixed range → ABSTAIN**: "2–3B" is a model-size range, not a measurement.
- **C — arXiv id → ABSTAIN**: `dddd.ddddd` with no receipt hit (the cycle-18-named safe class).
- **D — @-parameter → ABSTAIN**: "cosine@0.90" is a config threshold.
- **E — derived-percent VERIFY**: "12.7% (19/150" verifies iff BOTH 19 and 150 ground as receipt
  values AND 100·19/150 rounds to 12.7 — no fabricated-pair verification (triviaqa `12.7` =
  `n_incorrect`/`n`, now VERIFIED with a `derived:` receipt ref).
- **F — self-scoped `n=`**: an "N=4" obligates ONLY the token it directly prefixes, not every bare
  integer on its line. This is the dominant measured false-positive class (kbc `3`/`8`, curve
  `2`/`6`, truthengine `2`) and the change's load-bearing fix.

The six classes remain individually toggleable via the shipped `V05_*` module flags — the
severability the prereg required is a property of the shipped code, not just of this run.

## Scope and what is NOT claimed (pre-committed)

- Class A is dropped, not disproven: a **refined ≈-only class A′** (excluding the `~` tilde that
  caused the regression) is a NEW prereg, not a mid-run redesign. The current shipped state leaves
  `≈`/`~` approximations grounding exactly as v0.4 did.
- **Correction to a prereg statement (surfaced by a regression test):** the prereg asserted an
  `N=5`-style value with no receipt scalar "stays UNGROUNDED". It does NOT — the trailing `=` in
  "N=" matches the pre-existing comparison-operator spec rule, so the glued sample-size token reads
  as a spec constant and ABSTAINs. This is unchanged v0.4 behavior, not a v0.5 effect; class F only
  frees the OTHER integers on an `N=` line, and that is what it does. The prereg's mechanism note
  was wrong; the shipped behavior is stated here accurately.
- `0.25→0.00` transition values stay UNGROUNDED (real gap; repair is doc/receipt-side).
- Unblocks **G5** (population-framed scorecard) and **G6** (Annex IV lint) at the code level; those
  remain their own preregs. The zero-false-accusation property (P4 artifacts = 0) is what makes the
  certifier deployable against documents it did not author.

## Reproducibility

`styxx/certify.py` (six `V05_*`-gated clauses, A defaulting False); validator re-run
`OATH-V0-VALID`; the 3 closed-model-frontier corpus certs regenerated (diff = `verifier_sha256`
only, all still OATH-HELD). Receipts: `cycle38_v05_baseline_battery_result.json` (the HEAD
before-receipt, 269/119/26 reproduced exactly), `cycle38_v05_battery_result.json`,
`cycle38_v05_p1_sixdoc_result.json`, `cycle38_v05_p4_recert.json`,
`cycle38_v05_class_sweep_result.json` (+ `cycle38_v05_class_sweep.py`),
`cycle38_v05_attribution.json`.

---
*The notary tightened five of its own six proposed rules and rejected the sixth because its own
appeals court — the mutant battery — showed the rule accused too loosely and swore too easily. The
bars were set by earlier honest failures; they did not move tonight either.*

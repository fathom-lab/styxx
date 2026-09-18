# RESULT — DECLARE-1 lands, and its own headline gate could not see whether it was worth landing

Fathom Lab · 2026-09-18 · Prereg `PREREG_declare1_the_toll_2026_09_18.md`, sha256 `7ffd0ba1…`,
frozen before `styxx/declare.py` existed. Instrument `eba8f5fc…` → **`9b620e00…`**. Receipts:
`declare1_ceiling.py`, `declare1_ceiling.json`, `web/gate/differential/declare1_pairs.json`,
`tests/test_declare1.py`.

An agent may now declare its claims in one fenced `styxx` block instead of having them guessed out
of prose. The mechanism works and every blocking gate passes. **The gate written to measure whether
it was worth doing is structurally blind to the thing it was supposed to measure**, and that is the
most useful finding here.

## What shipped

```
```styxx
files_changed: 38
only_touches: web/gate
adds_symbol: gate_diff_text
```
```

A declaration is **normalised into the canonical sentence the existing reader already understands**
and read by that same reader, one level down. Nothing re-implements a verdict. A declared claim and
a prose claim of the same content therefore cannot disagree — `test_a_declaration_and_the_same_claim_in_prose_agree`
pins that — and the differential sees one reading rather than two.

Refusals, each pinned by a test: `tests_pass` is UNCHECKABLE even when declared; an unknown key or
an unreadable value is reported and never accused; two blocks declare nothing rather than being
merged; only a fence tagged exactly `styxx` is a declaration; and **declaring narrowly is visible
and not punished** — a body that declares a file count and nothing else is told nothing about its
scope.

## G-D1-1 — prose is untouched. PASS.

604 corpus claims, 651 claim readings, **0 differences** before and after. The prose pass runs
first and unchanged; the declaration pass is a separate second pass.

## G-D1-6 — the port keeps up. PASS.

`web/gate/diffgate.js` mirrors it. The differential reads **3,230 pairs, 6,945 claims, 0
disagreements**; `check_pairs.js` reads **54 pinned pairs, 0 disagreements**, ten of them new for
this reading including the ones that pin the refusals. The pin moved to `9b620e00…` in the same
change, which is now the rule.

The differential earned its keep immediately: the first port emitted `"many"` where Python emits
`'many'`, because `JSON.stringify` and Python's `repr` quote differently. A one-character
divergence in a reason string, caught before it shipped, in the machinery that was inert a week ago.

## G-D1-2 — the ceiling. PASS, and worthless.

The prereg said to synthesize a declaration for each decidable BENCH item and measure what the
format buys under perfect adoption. Done, in `declare1_ceiling.json`:

| kind | decidable | declared and read | agrees with the oracle | share the prose reader would have abstained on |
|---|---|---|---|---|
| `files_changed_count` | 116 | 116 | 116 | **0.0%** |
| `only_touches` | 17 | 16 | 12 | **5.9%** |
| `symbol_added` | 15 | 2 | 2 | 86.7% |

Read plainly, that table says the format buys almost nothing: on the items it covers, the prose
reader was already reading them correctly.

**The table is measuring the wrong population.** Its items are exactly those the BENCH-2 oracle
could admit — which is to say, exactly those where extraction already worked. A gate restricted to
the cases where the old method succeeded cannot detect a method that fixes the cases where it
failed. That is a defect in the preregistration, written by us, and it is recorded rather than
quietly replaced with a better number.

Two things in the table survive as real findings. `symbol_added` stays abstained even when
declared, because BC-1 withholds that reading whenever the diff contains no Python file — a
limitation of the *diff*, not of the claim, which no declaration can lift. And the four
`only_touches` disagreements are the same: an agent that declares the same non-file-scope thing its
prose said gets the same wrong answer. **Declaring fixes extraction; it does not fix meaning.**

## The measurement the prereg should have specified

The right population is DECIDE-1's hand adjudication — 100 claims, read by hand, with no oracle and
no extractor in the loop. Crossing those calls against what the shipped instrument actually says:

| | instrument spoke | instrument silent |
|---|---|---|
| **decidable by hand** | 27 | **49** |
| not decidable by hand | 1 | 23 |

**Of the 76 claims a human can settle from the diff, the instrument says nothing about 49 — 64%.**
Per kind, decidable-and-silent over decidable:

| kind | silent / decidable |
|---|---|
| `files_changed_count` | 0 / 24 |
| `only_touches` | **13 / 13** |
| `symbol_added` | 14 / 16 |
| `tests_added` | 22 / 23 |

On `only_touches` the instrument is silent on **every single decidable claim**. That is the
population a declaration block addresses, it is more than half the corpus, and G-D1-2 could not see
any of it.

## G-D1-5 — adoption. Zero.

No pull request in the world carries this block. Every number above is a property of a synthesized
or hand-adjudicated corpus. The prereg required this sentence and it is true today.

## What would still falsify it

Unchanged. If agents can only declare the easy claims, the hard ones stay in prose and the format
is decoration. The table above is the beginning of that test and not the end of it: it shows the
*room*, not that anyone will fill it. The honest position is that DECLARE-1 is a well-specified
mechanism with every refusal pinned, addressing a gap that is now measured at 64% of decidable
claims, with an adoption count of zero.

---

*The gate we wrote to tell us whether this was worth building could only look where the old method
already worked. We shipped the mechanism, published the blind gate, and went and found the number
it should have been measuring — which is that on the claim kind this was built for, the tool is
currently silent on all thirteen of the thirteen claims a person could settle by reading.*

# SPEC — a sworn span carries no directional override (v0.1)

**Frozen 2026-09-06, before the code.** One rule, D1. Found by the eight-dimension adversarial
audit (`wf_9466dcba-f49`, dimension `scanner`); the reporter named seven code points, the skeptic
measured glyph order in a browser and narrowed it to one, and this spec follows the skeptic.

## The defect

`U+202E RIGHT-TO-LEFT OVERRIDE` inside a numeric span is accepted silently. The lexer's `_TOKEN`
class does not match category `Cf`, so the control splits tokens rather than joining them,
`_number_token` still finds exactly one digit-bearing token, and the span is adjudicated on the
**logical** order. A UAX #9 renderer displays the run right-to-left.

Measured at `d857c697`:

```
document:  <sworn r="r1" k="numeric">the held-out error rate was <U+202E>0.55<U+202C> on the panel</sworn>
verdict :  SWORN-HELD  held=1 ... rungs L2=1
detail  :  {'printed_token': '0.55', 'receipt': '0.55', ...}

Chrome 148, visual left-to-right order of the digit glyphs:
  with RLO   -> "55.0"
  same bytes minus the two controls -> "0.55"
```

**The reader sees 55.0 under a SWORN-HELD badge; the verifier checked 0.55.** The receipt field is
literally named `printed_token` and does not hold what is printed. `canon`, `verify` and `check`
all stay silent — `canon`'s warning fires only when `_number_token` returns a reason, and here it
returns none. The sidecar round-trips the control byte-identically, so `render` reproduces the
deception and the sidecar battery's stops 2 and 3 pass.

## This applies a decision the lab has already made twice

**It is not new policy.** R2 `hidden_commitment` pays a MALFORMED rule for precisely this shape — a
tag-shaped candidate inside an HTML comment is refused because it "renders as nothing" while the
verifier sees it. A span that verifies as `0.55` and renders as `55.0` is the same verify/render
divergence with a worse payload.

And `styxx/capsule.py` already sanitises `‪-‮⁦-⁩` in its viewer, replacing each
with `U+FFFD` and a visible count. The lab decided this class matters at a render surface; `sworn.py`
simply has no notion of it. That is an inconsistency inside one package, not a considered choice —
`DECISIONS` has no entry for directional controls, no docstring mentions them, no test covers them,
and none of the twelve attack rows is about them.

If the operator disagrees that R2's policy reaches here, this is the rule to overturn.

## D1 — no directional override in a span's inner text

A sworn span whose inner text contains `U+202D LEFT-TO-RIGHT OVERRIDE` or `U+202E RIGHT-TO-LEFT
OVERRIDE` is `MALFORMED`, reason `directional_override`. It is a **bytes-only form check**, taken
before any receipt is opened, exactly as the module already does for `number_count`,
`needle_count` and `digest_form`: a MALFORMED must never depend on evidence the verifier might not
have.

### Why the overrides and not every bidi control

Only an **override** reassigns a character's inherent direction. UAX #9 X6 resets an overridden
run to strong L or R, which is the mechanism by which ASCII digits — European Number type — are
reordered. Embeddings and isolates do not do this, and the skeptic measured exactly that: of the
seven code points originally reported, `U+202A LRE`, `U+202B RLE`, `U+202D LRO`, `U+2066 LRI`,
`U+2067 RLI` and `U+2068 FSI` all render `0.55` unchanged in Chrome 148; only `U+202E` produces
`55.0`.

**The isolates are kept legal deliberately.** `U+2066`–`U+2069` are the Unicode-recommended way to
embed a Latin or numeric run inside Arabic or Hebrew text. Refusing them would penalise correct
mixed-direction authoring to defend against an attack they cannot carry.

`U+202D LRO` is refused alongside `U+202E` although it was measured not to deceive here, because
the rule is stated over a **category** — "a directional override" — rather than over one browser's
behaviour. A rule that names the code point that happened to deceive in Chrome 148 is a measurement,
not a rule.

### Why MALFORMED rather than a warning

A warning is what the verifier prints; MALFORMED is what the receipt carries. The deception travels
with the document, so the refusal must travel in the digested core, not in a line of CLI output a
reader may never see. This mirrors R2.

## What moves

- **Nothing committed.** 517 sworn spans across the 46 tagged `.md` files: **0** contain any bidi
  control at all, override or otherwise.
- `REASONS` gains one member, `directional_override`. It is a closed set a consumer keys on, so the
  addition is the point of the change and not a side effect.
- Both implementations change in one commit; the parity gate decides it.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| D-G1 | a numeric span wrapping its number in `U+202E` is not HELD | red: HELD, SWORN-HELD | green: MALFORMED `directional_override` |
| D-G2 | the same for a quote span and for `U+202D` | red | green |
| D-G3 | isolates and embeddings (`U+2066`-`U+2069`, `U+202A`-`U+202C`) stay legal | green throughout — the RTL-authoring clause |
| D-G4 | a span with no bidi control is untouched | green throughout |
| D-G5 | the refusal is bytes-only: it fires with no manifest and no tree | red | green |
| D-G6 | Python and the JS verifier agree by core digest across all of the above | green with both wrong, red with one side fixed |

D-G1 is the guard that must be seen red. D-G3 is the one that must never go red: it is the cost
this rule refuses to impose on RTL authors.

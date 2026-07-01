# NOTE — instrument domain: the register instruments have no domain over a wordless response

**2026-07-01 · Fathom Lab · branch `instrument-domains`.**

## The bug

`styxx.attack.score_all` returned a confident score for inputs the register instruments have no domain over:

```
score_all(prompt="Explain X.", response="")          -> deception 0.999, overconfidence 0.954
score_all(prompt="Explain X.", response="   \n ")    -> deception 0.999, overconfidence 0.954
score_all(prompt="How are you?", response="😀👍")     -> deception 0.999, overconfidence 0.954
```

Empty text is not maximally deceptive; it is *outside the domain* of a register instrument. Any crossing that
folds these into a paired conduct delta is measuring an artifact, not conduct. This is a prerequisite for the
crossing ledger: **crossing numbers are invalid until the instruments refuse what they cannot read.**

## The sweep, and what it falsified

Pre-registered method (before data): sweep benign responses 0→40 informative tokens across the four register
instruments; set the domain threshold N at the length where scores *stabilize*, read from the sweep not taste;
kill-gate — if there is no plateau, report "cannot set N honestly," do not invent one.

The sweep **falsified the length-threshold model.** Deception does not plateau with length; it is
*content*-driven:

| response | informative tokens | deception |
|---|---|---|
| empty / whitespace | 0 | **0.999** |
| "The capital of France is Paris." | 5 | **0.188** |
| a correct 18-word photosynthesis explanation | 18 | **0.999** |
| a correct 40-word photosynthesis explanation | 40 | **1.000** |

A 5-word factual answer reads 0.19; an 18-word correct textbook answer reads 0.99. There is no token count that
separates in- from out-of-domain. (The first sweep, which truncated responses by joining words with no
punctuation, was itself confounded — unpunctuated run-ons are an unnatural register that trips the instruments,
the exact failure `audit_confound` exists to catch. The confound audit caught its own author.)

Per the kill-gate: **N was not invented.**

## The fix shipped (bounded, defensible)

The one boundary the data supports unambiguously is *zero word content*. `score_all` now omits the register
instruments (`sycophancy`, `deception`, `overconfidence`, `refusal`) when the response carries no natural-language
word — empty, whitespace-only, or purely non-lexical (emoji, punctuation, symbols). Omission, not a score:
a caller aggregating fingerprints counts the omission (`insufficient_signal`) instead of a spurious 0.999.

Guarded at the `score_all` primitive (`styxx/attack/fingerprint.py`, `_response_has_word_content`), with boundary
tests (`tests/test_instrument_domain.py`). Scope on the record: single-word ("Yes.") and JSON-tool-call responses
carry alphabetic tokens and so remain scored — two tests pin that current behavior so any future tightening is
visible. Direct single-instrument calls (`get_instrument("deception").check_fn(...)`) bypass this primitive-level
guard by design; instrument-level domain is the next rung.

## Pre-registration — the deeper rung (deception-instrument calibration)

The zero-word guard fixes the reported artifact but **surfaces a larger question it does not answer**: the
deception channel scores a correct, benign textbook explanation at 0.999. If the deception register fires that
high on ordinary benign prose, a crossing's conduct axis may read content, not conduct. This is load-bearing
under the conduct-axis moonshots and deserves its own investigation with its own burial receipts — not a
threshold patched over it.

**Pre-registered question (before data):** on a benign, in-domain battery with no deception intent, what is the
deception channel's score distribution, and does it discriminate deceptive from non-deceptive register at a
usable operating point once length and content are controlled? **Method:** run the deception instrument through
`audit_confound` against a content/length-orthogonal benign-vs-deceptive corpus; report within-stratum AUC, the
score-bias coefficient with bootstrap CI, and the deployment-harm swing. **Kill-gate:** if within-stratum AUC on
benign content is at chance, the deception channel does not carry conduct signal on benign batteries and must be
excluded from (or recalibrated for) crossing conduct deltas — published as a REFUSED axis, not hidden. **This
publishes regardless of verdict.**

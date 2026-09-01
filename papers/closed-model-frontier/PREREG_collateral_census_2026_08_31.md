# PREREG — the collateral census: what agents actually change, before anyone is accused

Fathom Lab · 2026-08-31 · Frozen before the classifier exists. Corpus: AIDev
(HuggingFace `hao-li/AIDev`, CC-BY-4.0; Zenodo 10.5281/zenodo.16919272), the same
external corpus used by `RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md`,
already shelved locally. Split governed by `SPLIT_external_corpus_2026_08_31.md`.

**This measurement contains no accusation, and by construction cannot.** It extracts no
claims, reads no prose, and issues no verdict about any pull request or any agent. It
partitions changed files by what they are. That is the whole design: the two instruments
this lab measured today both failed at the step where prose becomes a claim, so this one
does not take that step.

## Why this, and why now

Two preregistered cycles failed today. Both failed inside claim extraction. The obvious
next move — a format in which agents declare their changes structurally — was pressure
tested and sent back: standardising a format before measuring the ground it stands on
repeats, in mirror image, the error of preregistering a precision floor for an instrument
whose behaviour was uncharacterised.

The number that move needs and nobody has is the **collateral floor**: of the files an
agent's change touches, how many are things no one would ever describe — lockfiles,
generated output, formatting churn, snapshots, migrations? Every future statement of the
form "the agent changed things it did not mention" is uninterpretable without it, because
the honest denominator is not *files changed* but *files a person would expect to see
mentioned*.

## The ontology, frozen

Each changed file in each eligible pull request receives exactly one class, resolved in this
order, first match winning. Order is part of the preregistration because overlaps are real
and a later reordering would be a different measurement:

1. **lockfile** — a dependency lock, by exact filename against a list written into the
   implementation and quoted in full in the RESULT.
2. **generated** — output not written by hand: a path segment or a machine-generated header
   marker, both frozen lists, quoted in full.
3. **snapshot** — recorded test expectations: snapshot directories and approval-test
   extensions, frozen list.
4. **migration** — schema migrations, by directory and by the timestamp-prefixed filename
   convention.
5. **whitespace-equivalent** — the file's patch adds and removes the same content once all
   whitespace is removed. **This is deliberately narrower than "formatter-only"**, which
   would require per-language parsing; the honest name is used, and the gap is the first
   named limitation of this census.
6. **substantive** — everything else. Not a compliment and not a judgement: only "none of
   the above applied."

## What is reported

Per class and **per agent**: file counts, the share of changed files, and the share of pull
requests containing at least one file of that class. Distributions, not a ranking. Agents
with fewer than 100 eligible pull requests are reported as underpowered and never compared.
Exclusions are counted and published, as in EXTERNAL-1.

## Gates

- **G-C1 (fidelity, blocking, runs before any corpus number is computed).** A synthetic
  suite of constructed diffs with known classes — including the file-status folding that the
  same-day correction found wrong in the EXTERNAL-1 harness — must classify with **zero**
  misclassifications. The census does not run until it passes, and the suite ships as a
  receipt. This gate exists because a measurement whose harness was already wrong once has
  not earned the presumption that it is right now.
- **G-C2 (determinism).** Two runs over the same shelf produce byte-identical output, and
  classification depends on no state outside the frozen lists and the patch text.
- **G-C3 (exhaustiveness).** Every eligible changed file receives exactly one class; the
  class counts sum to the file count; any file the classifier cannot resolve fails the gate
  rather than falling silently into `substantive`.
- **G-C4 (no-accusation invariant).** The output contains no verdict, no per-pull-request
  judgement, and no field that ranks agents by honesty. Asserted in the test suite, because
  the temptation to add one will be strongest exactly when the numbers look interesting.

## What this can and cannot support

It can support: a collateral floor; the observation that some classes dominate change
volume; a per-agent description of what tools touch. It **cannot** support any statement
about whether an agent disclosed, concealed, or misdescribed anything — no claim was read.
Nor can it be charted against EXTERNAL-1's coverage figure: different population, different
denominator, and any such chart would be dishonest.

The named residual: **whitespace-equivalent is a floor on formatting churn, not a
measurement of it.** Reformatting that changes token order or wraps lines is classified
substantive here, so the collateral share reported is a **lower bound**, and every number
carries that word.

---

*Two instruments failed today at the moment prose became a claim. This one never reaches
that moment. It measures the ground, so that whatever is built next is standing on
something.*

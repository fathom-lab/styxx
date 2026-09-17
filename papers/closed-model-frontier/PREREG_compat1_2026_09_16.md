# PREREG — COMPAT-1: the compatibility claim gets read, and what the diff removed gets named

Fathom Lab · 2026-09-16 · Frozen after `EXPLORATORY_compat_and_multilang_2026_09_16.md` (a
census of the corpus, no verdicts) and before the reading below is run over it. Built on BC-2
(`PREREG_bc2_by_construction_2026_09_16.md`, `RESULT_bc2_by_construction_lands_2026_09_16.md`).

## The claim

A closed phrase set, one template, kind `compat_claim`:

    no breaking change(s) · non-breaking · backward(s)[ -]compatib(le|ility) · fully compatible ·
    (zero|no) (behaviour|behavior|behavioral|functional) change(s) ·
    does not (break|change) [any] [existing] (behaviour|behavior|api|public api)

On the EXTERNAL-1 corpus 8,467 of 71,016 descriptions carry one. Today the gate reads none.

## The reading — evidence attached, never a verdict

For a `compat_claim` the gate returns **UNCHECKABLE, always**, and the reason carries what the
diff shows about the claim:

- A *public top-level definition* is read from the REMOVED lines of a file, per language:
  Python `def`/`class` at column 0; JS/TS `export [default] [async] (function|class|const|let|var|
  interface|type|enum) NAME`; Go `func NAME(` and `type NAME` with NAME capitalised; Rust
  `pub (fn|struct|enum|trait|type|const|static) NAME`; Java `public … NAME(`. Names beginning with
  `_` are not public. A name that appears anywhere in the ADDED lines of any file of the same
  language is a change or a move and is dropped from the list.
- Reason when something is left: `compatibility claimed; the diff removes N public
  definition(s) not re-defined in the added lines: pkg/api.py: Session, lib/index.ts: parse (…)`
  — at most five named, the total stated.
- Reason when nothing is left: `compatibility claimed; no public top-level definition removed
  (python, js/ts, go, rust, java read; behaviour beyond names not checked)`.
- No language of the diff among the five: `compatibility claimed; no language this reading
  covers in the diff`.

Nothing here is VERIFIED and nothing is CONTRADICTED. A removed name is not proof of a break (dead
code, re-exports from elsewhere, deliberate removals with the claim about wire behaviour), and an
intact surface is not proof of compatibility. The evidence is for the reader; the accusing
verdict for this kind does not exist in the code and a test pins that it cannot appear.

## Gates — committed now

- **G-C1 (never accuses, by construction).** `compat_claim` has exactly one verdict, UNCHECKABLE,
  in code and in a test that drives every reason branch; on the corpus, zero `compat_claim`
  claims with any other verdict. Blocking.
- **G-C2 (every other kind untouched).** Ledger-to-ledger, BC-2 checkout against COMPAT-1
  checkout, keyed on `(pr_id, kind, claim text)`: the set and verdicts of all non-`compat_claim`
  claims are identical. Blocking. (The never-read count drops by the sentences this template
  now reads; that is the intended change and is reported.)
- **G-C3 (the census reproduced).** Claims read: the exploratory count within ±2% (the template
  is the exploratory regex moved into the instrument; sentence splitting may differ at the
  margin). Claims with at least one removed public name: reported with the per-language split
  and the dropped-name histogram, expected near the exploratory 531 by the same rule. Reported,
  not scored.
- **G-C4 (suite and demo).** Full suite green; the demo unchanged (its summary makes no
  compatibility claim); the hooks and the Action print `[ ? ]` for the claim with the names in
  the reason, and the CLI's `--out` JSON carries `detail.removed` as a list.
- **G-C5 (what is not claimed).** No precision figure for the evidence, no agent comparison, no
  "X% of compatibility claims are false". If anyone wants this reading to accuse, a blind
  adjudication under the EXTERNAL-1 protocol comes first, with the panel shown the removed names
  and the diff.

## Out of scope, named

Behavioural compatibility (signatures, semantics), re-exports across files in different
languages, generated code, and the multi-language test counter (its own prereg; 85 claims on the
corpus, see the exploratory note).

---

*8,467 claims the instrument never read, 531 of them on diffs that remove a public name. The
reading is the exploratory rule moved into the gate, and the gate's only sentence about it is
"here is what the diff removed"; the run that follows checks that moving it changed nothing else.*

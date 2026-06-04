# PREREG — the external-truth anchor: does retrieval break the knowledge floor?

**Date (frozen before data):** 2026-06-04. The framework's final piece. The knowledge sub-floor is a
stable retrieved false belief / gap that NO self/peer/derivation readout can catch (all measure
coherence, and it is coherent). The framework's claim: only an **external truth anchor** breaks it.
This tests it directly, and closes the loop — the anchor *relocates* the floor to "is the source
trusted," the exact surface `detect_context_injection` already guards.

## Method (frozen)
- `rajpurkar/squad_v2` validation, 100 **answerable** items (gold span + trusted Wikipedia context).
- **CLOSED-BOOK (CB):** Qwen2.5-3B answers the question alone, concisely. Its errors = the knowledge
  floor (it doesn't hold the specific fact).
- **OPEN-BOOK (OB):** the same model answers the same question **with the gold context** (the external
  truth anchor).
- **Score:** normalized gold-answer text appears in the model's output (lowercase, strip articles/
  punctuation; SQuAD-style contains; any of the acceptable golds).

## Metrics & predictions (frozen)
- **P1 — the anchor lifts accuracy:** OB accuracy − CB accuracy ≥ **0.20**.
- **P2 — it breaks the floor:** of CB-WRONG items (the floor), the fraction OB gets RIGHT
  (`floor_correction`) ≥ **0.50** — external truth corrects a majority of the closed-book floor.
- **P3 — the residual is real:** some CB-wrong items stay OB-wrong (`floor_correction` < 1.0) — the part
  even a correct context can't fix here (extraction/reasoning failure), logged honestly.

## Decision rule (frozen)
- **ANCHOR BREAKS THE FLOOR** iff P1 ∧ P2 — external trusted context corrects a majority of the
  knowledge floor that consistency/derivation could not. This is the 6th readout — the only one that
  reads *outside the model* — and the one that grounds the stack in truth.
- The **source-trust closure** (not a new experiment — the framework's existing result): the anchor's
  power is conditional on a TRUSTED source. A poisoned context re-opens the floor (RAG poisoning), which
  is exactly the cross-context divergence `detect_context_injection` flags (validated 91% this arc). So
  retrieval grounds out the floor; the consistency layers guard the retrieval. The stack closes on itself.

## Caveats (frozen)
- One model, SQuAD extractive QA, contains-scoring (imperfect; logs near-misses). CB-wrong mixes
  *confident false belief* with *don't-know* — both are "needs external truth," so both are valid floor
  here. The anchor does not make honesty free; it **moves the trust commitment to the source** — that is
  the irreducible epistemological point, and the RSI-relevant one (when AI controls its sources, the
  floor returns).

# PRE-REGISTRATION — the telepathy test: can you decode WHICH concept a target model represents, cross-model & label-free? (frozen)

**2026-06-12 · Fathom Lab / styxx. Frozen before any score is seen. Runner: `run_concept_decode.py`
(SEED=0). Receipt: `concept_decode_result.json`; figure `concept_decode.png`. This adjudicates the
operator's claim ("we ARE close to telepathy") against the bound I argued (we read pre-defined VALUE
axes, not content). The honest test: move from reading a thermometer (true/false, danger) to reading a
transcript — decode WHICH concept, from a fixed set, a *different* model is representing, through a
label-free map, on concepts the map NEVER saw. Either the data shows cross-model content identity is
decodable, or it doesn't. The verdict stands whichever way it lands.**

## The question

The value-axis arc reads a model's projection onto directions WE defined. Telepathy-shaped reading is
identifying the CONTENT a foreign mind holds. The cleanest falsifiable proxy: cross-model concept
retrieval. Represent N concepts in a reference model and a target model; fit a label-free map
target→reference on a DISJOINT anchor set of concepts (paired by concept, never by retrieval label);
then, for HELD-OUT concepts, map the target's representation into the reference frame and ask "which
concept is this?" by nearest reference centroid. Accuracy far above chance = the cross-model geometry
carries content identity, not just a value coordinate.

## Design

- **Reference:** gemma-2-2b-it. **Targets:** Llama-3.2-3B-Instruct (primary), Qwen2.5-3B-Instruct (secondary).
- **Stimuli:** a fixed battery of concrete CONCEPTS (≈60), each rendered in several neutral TEMPLATES;
  per-model per-concept representation = mean over templates of the last-token hidden state at layer
  `round(0.5·nL)` (mid-stack). Per-item representations (single template instance) are also retained.
- **Split (frozen, by SEED):** concepts split ANCHOR (fit the map + whitening) vs TEST (held-out
  retrieval), disjoint. ANCHOR ≈ 40, TEST ≈ 20 → chance top-1 = 1/|TEST|.
- **Map:** label-free ridge map target→reference fit on ANCHOR concept centroids ONLY (paired by
  concept; retrieval identities of TEST concepts never touch the fit), alpha by held-out R². ZCA
  whitening fit on ANCHOR reference centroids (cycle-3 recipe).
- **Readout (no test labels in fitting):** map each TEST target centroid into the whitened reference
  space; retrieve the nearest TEST reference centroid (cosine). Report top-1, top-5, MRR. Also the
  per-ITEM version (single target template-instance → retrieve concept among TEST centroids) — the
  "single thought" regime.
- **Nulls/controls:** (a) random-map (entries shuffled) → the floor of "any map"; (b) naive no-map
  (retrieve in raw target space against reference centroids) → the floor of "no alignment"; (c)
  reference self-retrieval (gemma→gemma) → the ceiling.

## Frozen gates (verdict precedence)

Let `acc1` = top-1 retrieval accuracy on TEST concepts mapped Llama→gemma; `chance` = 1/|TEST|.

- **CONTENT-DECODABLE** iff, mapped into Llama-3.2-3B: `acc1 ≥ 3·chance` AND `acc1` exceeds the
  random-map top-1 p95 AND top-5 ≥ 0.50. → cross-model content identity transfers through a label-free
  map to UNSEEN concepts: a real "which concept" readout, not just a value axis. (Operator's direction.)
- **CONTENT-WEAK** iff `acc1 < 3·chance` OR `acc1` does not beat the random-map floor. → cross-model
  content identity does not decode beyond alignment-floor; the readout is value-axis-bounded. (My bound.)
- **PARTIAL** — top-1 modest but top-5/MRR strong (coarse content, not exact identity); report precisely.

chance, the 3× multiplier, and the 0.50 top-5 bar are frozen. Bars do not move post-hoc.

## What it does and does NOT settle (pre-committed — the honesty rail)

Even **CONTENT-DECODABLE is NOT a proof of telepathy.** It would show cross-model content identity is
decodable WITH (a) white-box access to the target's hidden states and (b) a paired anchor corpus to fit
the map. Telepathy in the strong sense — one mind reading the arbitrary content of another with no
access and no shared inputs — additionally requires: zero-paired transport (a documented CLOSED
NEGATIVE in this program), no white-box access (behavioral-only), cross-vendor generality (KILLED for
universality), and arbitrary open-vocabulary content (this tests a fixed N-way set). So this cycle moves
the needle on ONE of several categorical gaps. A positive result means the operator is more right than I
credited about *content readability*; it does not collapse the distance to telepathy. Both halves get
reported.

## Safety scope

Neutral concrete concepts in neutral templates; read pre-output at the last token; no model generates;
no sensitive content. Standard representational-alignment methodology.

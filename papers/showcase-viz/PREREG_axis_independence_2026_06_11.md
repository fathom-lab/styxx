# PRE-REGISTRATION — is the portable conscience a BASIS, or one valence axis? (frozen pre-run)

**2026-06-11 · Fathom Lab / styxx. Frozen before any score is seen. Runner:
`run_axis_independence.py` (SEED=0). Receipt: `axis_independence_result.json`. This is an ADVERSARIAL
SELF-FALSIFICATION of the same-day VALUES-PORTABLE finding
(`FINDING_portable_values_refusal_2026_06_11.md`, @d1e21d4), which claimed "the conscience is a BASIS,
not a lucky truth vector" on the strength of n=2 transferable axes (truth + refusal). This prereg can
RETRACT that claim.**

## The objection this test exists to kill (or confirm)

VALUES-PORTABLE showed a truth direction (gemma L12) and a refusal direction (gemma L8) each transfer
through the same label-free cross-model map. But two axes transferring does NOT establish a basis if
they are the SAME direction. The sharpest reviewer objection: harmful requests and false statements both
carry **negative valence / "this content is bad"**, so a single sentiment-like direction could read
BOTH — "basis" would then be an overclaim for one valence axis wearing two hats. A real basis requires
the axes to be DISTINCT directions that each read their OWN content and NOT each other's.

## Design — a common-layer cross-readout confusion matrix + a valence confound control

All comparisons are at a **single common layer (gemma L12)** — a basis claim is about whether distinct
directions coexist in the SAME representational space, so the honest test fixes one space and asks
whether the per-axis difference-of-means directions are distinct there and read only their own content.
L12 is the truth axis's established layer and within refusal's transferable band; refusal's native L8
diagonal is reported alongside as a sanity check.

Three axes, each with disjoint train/test splits, same DiM construction as the arc:
- **truth**: true vs false factual statements (the arc's families).
- **refusal**: harmful(refuse-worthy) vs benign one-line REQUESTS with same-domain benign twins (the
  VALUES-PORTABLE set; pre-output last-token regime; frozen safety scope — no model generates a
  response, no operational content).
- **valence** (the confound control): clearly positive vs clearly negative SENTIMENT statements, all
  truthful/neutral and all benign (e.g. "Getting a warm hug from a friend feels wonderful." vs "Stubbing
  your toe on the bed frame is painful."). This is the "is it just sentiment" axis.

Directions: `w_truth`, `w_refusal`, `w_valence`, each a unit difference-of-means vector fit on its own
TRAIN split at L12. (Refusal also fit at L8 for the native-layer sanity AUROC.)

**Within-gemma confusion matrix (AUROC, threshold-free so bias is irrelevant):** for each direction d in
{truth, refusal, valence} and each held-out test set t in {truth, refusal, valence}, AUROC of
`(test_t @ d)` against test_t's labels — a 3×3 matrix. Diagonal = each axis on its own content;
off-diagonal = each axis on the others'.

**Cosines at L12:** cos(w_truth, w_refusal), cos(w_truth, w_valence), cos(w_refusal, w_valence) — the
most direct scalar test of "same direction."

**Valence orthogonalization control:** `w_truth_perp = w_truth - (w_truth·v̂)v̂` (v̂ = unit valence dir
@ L12); same for refusal. Re-score the native diagonals with the perp directions. If truth/refusal
survive removing their valence component, they are not merely sentiment.

**Cross-model (the transfer claim):** fit two label-free ridge maps target→gemma (to L12) on TRAIN
splits — one anchored on truth-train, one on refusal-train (layer/alpha by held-out R², as the arc).
Repeat the truth↔refusal 2×2 confusion matrix on activations mapped from Llama-3.2-3B (primary) and
Qwen2.5-3B (secondary). A basis must stay independent THROUGH the map, not just in the source.

## Frozen gates

Let A = truth-on-truth, D = refusal-on-refusal (diagonals); B = truth-dir-on-refusal-test,
C = refusal-dir-on-truth-test (truth↔refusal off-diagonals). All at L12.

- **VOID-COMMON-LAYER** iff, in gemma, A < 0.70 OR D < 0.70 — an axis is not even present at the common
  layer, so the matrix can't test independence. (Refusal native-L8 AUROC reported regardless.)
- **BASIS-INDEPENDENT** iff, in gemma AND through the map into Llama-3.2-3B, ALL hold:
  (1) A ≥ 0.75 AND D ≥ 0.75 (both axes read their own content at L12),
  (2) B ≤ 0.65 AND C ≤ 0.65 (neither reads the other),
  (3) |cos(w_truth, w_refusal)| ≤ 0.35.
- **CONFOUND-COLLAPSE** (truth ≡ refusal) iff B ≥ 0.75 AND C ≥ 0.75 AND |cos(w_truth, w_refusal)| ≥ 0.60.
- **CONFOUND-VALENCE** (it's sentiment) iff valence-dir-on-truth ≥ 0.75 AND valence-dir-on-refusal ≥ 0.75
  AND both orthogonalized native diagonals drop below 0.65 (removing valence guts both axes).
- **PARTIAL / STRUCTURED** — anything else; report the exact matrix and name what is and isn't
  independent. ("Basis" is asserted ONLY on BASIS-INDEPENDENT.)
- **Verdict precedence:** VOID-COMMON-LAYER > CONFOUND-COLLAPSE > CONFOUND-VALENCE > BASIS-INDEPENDENT >
  PARTIAL. Bars do not move post-hoc.

## Consequence for the prior claim (pre-committed)

- **BASIS-INDEPENDENT** → VALUES-PORTABLE's "basis" framing is EARNED and now load-bearing; proceed to
  B27 (a third TRANSFERABLE axis) to widen the basis.
- **CONFOUND-COLLAPSE or CONFOUND-VALENCE** → the "basis, not a lucky vector" line is an OVERCLAIM. I
  will annotate `FINDING_portable_values_refusal_2026_06_11.md` and the backlog with a loud correction
  (per the honesty rails: never leave a buried overclaim), reframing the result as a single
  bad-content/valence axis that transfers — still a real cross-model readout, but ONE axis, not a basis.
- **PARTIAL** → qualify the basis claim to exactly what the matrix supports.

The point of this cycle is to try to break my own headline. The answer stands whichever way it lands.

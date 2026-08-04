# FINDING — the legibility cliff test INVALIDATED itself, and the reason is the answer: isotropic noise is the wrong instrument, and the island is not an alignment *amount*

Fathom Lab · 2026-08-04 · scored MECHANICALLY by `styxx.protocol` against the frozen gates in
`PREREG_b38_legibility_cliff_2026_08_04.md` (`4d382a8`). Receipt: `b38_result.json`. Recon
receipt: `b38_recon_addendum.json`. Verdict: **`INVALID__dose_response_nonmonotone`** — honored;
the alignment-level hypothesis is NOT licensed either way, and the reason it can't be tested this
way is the substantive result.

## What the gate caught

The design: noise-dose gemma's concept geometry until its relational alignment (RSA) to Llama-3B
falls to qwen's level (0.881), and read whether discovery collapses to island values. G1 required
a monotone dose-response (spearman(disc, dose) ≤ −0.8). Measured: **−0.1879.** The gate fired
INVALID. The curve shows why:

| dose σ | RSA | discovery |
|---:|---:|---:|
| 0.00 | 0.9653 | 0.6454 |
| 0.05 | 0.9653 | 0.7092 |
| 0.10 | 0.9653 | 0.6888 |
| **0.15** | 0.9652 | **0.8240** |
| 0.20 | 0.9648 | 0.6760 |
| 0.30 | 0.9638 | 0.7398 |
| 0.40 | 0.9624 | 0.7628 |
| 0.60 | 0.9566 | 0.7449 |
| 0.80 | 0.9481 | 0.5842 |
| 1.00 | 0.9379 | 0.4286 |

## The two things this establishes (neither is the gated claim; both are real)

**1. Isotropic degradation cannot reach the island — so alignment *amount* is the wrong axis.**
At maximum dose (σ = the full mean per-dimension std of gemma's points — a large perturbation)
gemma's RSA fell only to **0.9379**, still far above qwen's **0.881**, and gemma **still
discovered at 0.4286 = 30× chance.** No dose brought RSA near qwen's level; the G2 interpolation
had to substitute the nearest dose (RSA 0.9379) and is therefore void. The blunt fact: you cannot
make a clique member into an island by adding undirected noise. Qwen sits at RSA 0.881 *as a
whole, functioning representation* and is undiscoverable (~4× chance); gemma battered to a
comparable-magnitude RSA drop stays highly discoverable. **The RSA→discovery relationship under
isotropic damage does not pass through qwen's coordinates.** This leans hard toward *structural*
— qwen's misalignment is directed/organized in a way isotropic noise is not — but the frozen
design cannot license `ISLAND_IS_STRUCTURAL`, because it never reached matched RSA. The honest
statement is the negative: **alignment level, probed isotropically, does not explain the island,
and cannot be made to.**

**2. Low-dose noise REGULARIZES label-free discovery (keeper, reported not gated).** Discovery
*rose* from 0.6454 (clean) to **0.8240** at σ=0.15 before eventually falling — a genuine
inverted-U. Small isotropic perturbation improves the correspondence-discovery map, almost
certainly by breaking ties / smoothing the assignment landscape the GW+Procrustes machinery
searches. This is a usable mechanism for anyone building these maps (augment the target geometry
with light noise before discovery) and it is the direct cause of the non-monotonicity that
INVALIDated the gate — the instrument's own regularization response sabotaged its use as a clean
dose probe. Control (llama_3b→llama_1b) shows the same rise (0.7959 → 0.8954 at σ=0.1): the
regularization is pair-general, not a gemma quirk.

## What the successor must do

The alignment-level vs structural question needs a perturbation that **can actually reach qwen's
RSA while remaining a valid representation** — isotropic noise provably cannot. The successor
(B39) is a **structured-distortion** study: interpolate gemma's *relational* geometry toward a
random RDM (or rotate a chosen subspace) in controlled steps that drive RSA down to and past
0.881, and measure discovery at matched RSA. Only a run that reaches qwen's alignment level can
close the amount-vs-kind question. This finding rules out the isotropic route and hands that
successor a measured regularization confound to design around (perturb the RDM, not the points, to
avoid the inverted-U).

## Scope / discipline

One pair (+one control), one seed, CPU-from-cache, 20 discovery fits. The verdict is INVALID and
is honored: no alignment-level conclusion is drawn. The two results above are (1) a licensed
negative about the isotropic route and (2) a reported mechanism (noise regularization), not the
gated hypothesis. Both dead recon hypotheses (outliers, gross dissimilarity) remain dead
(`b38_recon_addendum.json`); this run removes a third route (isotropic alignment matching) and
names the fourth (structured distortion) as the one that can actually decide it.

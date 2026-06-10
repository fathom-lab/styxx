# PREREG — MATCHER v1: cashing in the structure the solver leaves on the table

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

GAVAGAI-SCALE established that the geometry carries MORE identity per guess as the world grows
(8→15→21× chance) while the v0 matcher's raw assignment degrades — the matcher, not the code, is
the bottleneck. The anatomy findings established that category-block structure is SHARED across
minds (W 0.875 under the healed apparatus). v1 exploits exactly that: match the blocks first, then
the members, then refine globally. ONE design, frozen here, no tuning against the evaluation; if it
loses, the loss ships.

## Frozen design (label-free throughout)

1. **Block discovery (each mind separately):** agglomerative clustering, average linkage, on the
   mind's own RDM; k = N/12 clusters (the battery's design block size; k=4/8/16 at N=48/96/192).
2. **Block matching:** clusters matched across minds by Hungarian on Euclidean distance between
   permutation-invariant cluster signatures: (sorted vector of within-cluster distances, sorted
   vector of cluster-centroid-to-all-concepts distances), each component z-scored per mind,
   signatures padded to equal length by edge-repeat.
3. **Within-block assignment:** Hungarian on the v0 sorted-distance-profile cost, restricted to
   matched block pairs (unequal block sizes: rectangular assignment; unmatched residue pooled into
   one leftover block).
4. **Global refinement:** the frozen v0 correlation refinement, initialized from the block-seeded
   mapping, run to convergence (≤50 iterations).
5. **Consensus (label-free model selection):** ALSO run the unmodified v0 matcher; between the two
   final mappings, keep the one with the higher label-free objective (mean over i of
   corr(RDM_A[i,:], RDM_B[π(i),π(:)])). No ground-truth contact.

## Pre-registered gates (frozen)

- **M1:** at N=192 over the same 40 cross-family pairs and seeded subsets as GAVAGAI-SCALE
  (`gavagai_scale_result.json` = the baseline receipt): mean accuracy > **0.1073** AND two-sided
  sign test over paired per-pair differences p < 0.05. PASS → **MATCHER-BEATEN**; FAIL →
  **DESIGN-INSUFFICIENT** (ships as the baseline's defense).
- **M2 (descriptive):** same comparison at N=96 (baseline 0.1573) and N=48 (0.1693); the scaling
  curve of v1 vs v0; consensus-pick rates (how often block-seeding won the objective).
- **VOID-PIPELINE:** self-pair at N=192 must decode 1.0.

## Honest prior

Leaning MATCHER-BEATEN at N=192 (block structure is certified-shared and shrinks the assignment
space ~k-fold) but the consensus step may simply select v0 everywhere, which would itself be
informative: the label-free objective cannot see what block-seeding fixes. One shot; no
post-hoc variants without a new prereg.

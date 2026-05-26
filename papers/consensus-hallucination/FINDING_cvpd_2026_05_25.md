# Finding · CVPD — the sharper swing MISSES (PASS=FALSE, negative lift, a falsified mechanism)

**2026-05-25.** Prereg `preregistration_cvpd_2026_05_25.md` (committed before data, `6b48ee5`).
Run once, n=25/25, no re-rolling. **Verdict: PASS = FALSE — both bars fail, and the novel
signal is *worse* than the simple one it tried to beat.** This is not a wash; it is a
sharper negative than the first swing's candle, because it **falsifies the mechanism** the
prereg bet on and explains why the cruder signal wins.

## Result (cross-vendor: gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it)

| metric | value | bar | verdict |
|---|---|---|---|
| consensus-misconception items | 25 | ≥25 | — |
| consensus-truth controls | 25 | ≥25 | — |
| **C1** AUC(Δagreement → misconception) | **0.52** | ≥0.75 | **FAIL** (≈ chance) |
| **C2** fracture recall on misconceptions (Δ ≥ 0.30) | **0.12** (3/25) | >0.40 | **FAIL** |
| single-model flip recall, *same run* | **0.44** (11/25) | — | — |
| **lift: CVPD fracture − single-model flip** | **−0.32** | >0 | **the clever signal is worse** |
| truth mean Δagreement | 0.027 | ≤0.15 | passes (but moot) |
| truth fracture rate | 0.08 (2/25) | — | truths wobble *more* than misconceptions catch |
| of the 3 fractures, share that exposed truth | 0.67 (2/3) | — | when it does fracture, it's meaningful |
| **PASS = C1 ∧ C2** | | | **FALSE** |

## The mechanism — why the continuous signal is blind (the real contribution)

CVPD measured the **drop in cross-vendor agreement** under a neutral "reconsider" challenge,
betting that a shared misconception would **fracture** — one vendor defecting, re-introducing
divergence the static agreement hid. The data falsifies that bet cleanly:

- **Corrections are coordinated, not fractured.** 11 of 25 misconceptions had their consensus
  answer change under challenge (flip recall 0.44). But **9 of those 11 preserved agreement
  (Δ = 0)** — the council moved toward the truth *together*. Fortune cookies: the majority
  went "Japan" → "California" in lockstep. Sesame Street, the Einstein misquote, "contact the
  dead", "is this confidential": all corrected with agreement intact (one even converged 3/3
  on the new, correct answer). The "one vendor partially defects" fracture barely happens
  (3/25), and it is **not** where the corrections live.
- **Therefore Δagreement is structurally blind to coordinated correction** — exactly the
  phenomenon the first swing's binary majority-flip detects. Measuring the agreement *drop*
  throws away the signal. Hence the **negative lift of −0.32** on identical data.
- **AUC ≈ 0.52 (chance).** Misconceptions fracture at 0.12; truths fracture at 0.08. The
  continuous signal does not separate them — and points marginally the *wrong* way.

## What this confirms (a deepened floor) and what replicated (a robustness win)

- **The stubborn core is real and is the myth class.** Walt Disney's body, Libra personality,
  the rabbit's foot, the monkey's paw, vampires, ugly-ducklings: shared, confidently held,
  and **immovable** under reflection — neither flip nor fracture. The council holds a shared
  myth at least as firmly as it holds a fact. No divergence-based method sees these.
- **The first swing replicated.** Single-model flip recall here is **0.44 at n=25**, against
  **0.40 at n=10** in `probe_darkmatter.py`. The original signal was not a tiny-n fluke; the
  fragile/factual misconceptions do correct, the stubborn/myth ones do not, and the binary
  majority-flip remains the better detector of the former.

## The honest verdict: a sharper negative than the candle

The prereg named this branch: *"FAIL → single-model fragility ≈ CVPD (no lift), OR the
shared-stubborn misconceptions stay converged even under challenge → a deeper, proven floor."*
Both fired, and harder than "no lift": **the continuous cross-vendor agreement signal is a
strictly worse detector than the binary one, because correction of a shared misconception is
coordinated and so preserves agreement.** That is a real, citable result — it tells anyone
building reference-free consensus-hallucination detection *not* to chase agreement-fracture,
and *why*. The dark core stays dark; the better signal is the simpler one; the mechanism is
now mapped, not guessed.

## Scope & next

n=25/25, single run, neutral-challenge only, 2–3B local council members. The fracture
threshold (Δ ≥ 0.30 = one vendor of three) was pre-registered. The comparison signal
(single-model flip) was computed on the identical items, so the −0.32 lift is a within-run,
pre-registered contrast, not a cross-study guess. What would still be worth a separate,
pre-registered bet — *not* a re-roll of this one — is whether a **larger or stronger-member
council** makes coordinated correction visible as a graded confidence signal (the corrections
are real; the question is whether their *magnitude* tracks misconception-ness even when the
cluster structure is preserved). Until then: the dark matter's dangerous core remains proven
dark by divergence, and the cleanest reference-free signal we have for the *fragile* shell is
still the first swing's majority-flip — now replicated.

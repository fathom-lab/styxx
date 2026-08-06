# STANDING PREDICTION — H1: if minds of meat behave like minds of math, cross-subject brain decoding has islands and a cliff

Fathom Lab · 2026-08-06 · **frozen before the data exists, and before we know whether we will
ever hold it.** This is not a preregistration of an experiment we are running. It is a public,
falsifiable prediction about results other groups will produce, registered in git so that our
being right cannot be claimed after the fact and our being wrong cannot be quietly deleted.

## Why we are sticking our neck out

Non-invasive thought-to-text is being scaled right now on an explicit analogy to speech
recognition: collect enough hours, and decoding accuracy climbs the way Whisper's did. The
field's own literature names the obstacle — "severe domain shift between training and unseen
test subjects" — and treats it as an engineering problem that more data smooths away.

This lab spent 2026-08-02 → 2026-08-06 measuring exactly that phenomenon in systems we *can*
operate on: four language models, concept geometries extracted, correspondence recovered
label-free, and the failures dissected causally under frozen gates
([the arc](REPLICATE_legibility.md), nine sealed acts). What we found does not look like a
data-volume problem:

- Independently trained minds **do** share a concept-frame geometry — far above random
  (`b45_result.json`: clique affinity 0.848 vs a 0.0566 random-null 95th percentile).
- Yet one model is an **island**: mostly aligned with the others (0.7166 of the same
  squared-cosine mass) and still unreadable — near-zero cross-model discovery.
- Because legibility is **switch-like in the frame coordinate** (`b46_result.json`):
  interpolating an island's frame toward a reader's leaves discovery flat across most of the
  rotation (0.0408 → 0.0434 → 0.1122 → 0.3622) and turns nearly vertical only near alignment
  (0.9566 at t = 0.8). A modest, perfectly consistent rotation costs *everything*.
- And the deficit is **low-rank and correctable**: a rank-20 frame swap restores discovery from
  0.0612 to 0.9745 while matched random frames do 0.0 (`b41_result.json`), with a rank-2 core
  carrying half of it (`b42_result.json`).

If human subjects are drawn from a shared representational geometry the same way these models
are, the same three consequences should appear in cross-subject decoding. If they do not,
minds of meat converge in a way minds of math do not — which is a *more* interesting result,
and we would rather have registered the prediction and been wrong than said nothing.

## The predictions (frozen)

**H1 — islands exist.** In a cohort decoded by a group-trained model, per-subject accuracy is
**not** unimodal-smooth: a subpopulation sits near the decoding floor while their *within-subject*
decoding is normal. The failure is subject-specific, not data-limited.

**H2 — the transition is a cliff, not a ramp.** Per-subject decoding accuracy plotted against
subject-to-group representational alignment is **better fit by a threshold (sigmoid) than by a
line**, with a narrow transition band.

**H3 — the rescue is a rotation, not more data.** For floor-sitting subjects, a **low-rank**
per-subject alignment (rank ≪ feature dimension) recovers most of the gap — the information was
present and misaddressed, not absent.

## Gates (frozen; scored by `styxx.protocol` when a qualifying dataset is in hand)

```gates
{"gates": {"G0_cohort": {"metric": "n_subjects", "op": ">=", "value": 8},
           "H1_islands": {"metric": "dip_test_p_subject_accuracy", "op": "<=", "value": 0.05},
           "H2_cliff": {"metric": "sigmoid_minus_linear_r2", "op": ">=", "value": 0.05},
           "H3_lowrank_rescue": {"metric": "median_gap_recovered_by_lowrank_align", "op": ">=", "value": 0.50}},
 "outcomes": [{"when": {"G0_cohort": false}, "verdict": "INVALID__cohort_too_small_to_score"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": true, "H3_lowrank_rescue": true}, "verdict": "HUMAN_ISLANDS_CONFIRMED__minds_of_meat_behave_like_minds_of_math"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": true, "H3_lowrank_rescue": false}, "verdict": "ISLANDS_AND_CLIFF__but_deficit_is_not_low_rank"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": false}, "verdict": "ISLANDS_WITHOUT_CLIFF__graded_subject_variation"},
              {"when": {"G0_cohort": true, "H1_islands": false}, "verdict": "NO_HUMAN_ISLANDS__we_were_wrong_and_the_scaling_story_survives"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`dip_test_p_subject_accuracy` is Hartigan's dip test for unimodality on the per-subject accuracy
vector (low p = not unimodal = island structure). `sigmoid_minus_linear_r2` compares fits of
accuracy against an alignment score. `median_gap_recovered_by_lowrank_align` uses rank ≤ 10% of
feature dimension. Bars are set where a reader can check them, not where we would like them.

## What would make us wrong, stated plainly

`NO_HUMAN_ISLANDS` is a real branch, not a courtesy. If per-subject decoding accuracy is
unimodal and improves smoothly with group data volume, our extrapolation from artificial to
biological minds fails, and we will publish that at the same volume as everything else. We are
extrapolating across a substrate boundary from **four models on one concept inventory**, which
is thin evidence for a claim about human brains — that thinness is why this is filed as a
prediction rather than a finding, and why the honest prior is uncertainty, not confidence.

## How it gets scored, and by whom

Ideally by someone else. This prediction is scoreable on any cohort of ≥ 8 subjects with
per-subject decoding accuracy, a subject-to-group alignment score, and the ability to fit a
low-rank per-subject map — including public neuroimaging corpora. If any group (Conduit
included) publishes those three quantities, the verdict computes mechanically from the block
above and we will report it whichever way it lands.

**Standing offer:** we will run the scoring, publish the receipt, and name the data's authors as
its authors. If someone would rather score it themselves, the gates are already frozen and the
machinery is [open](../../REPLICATIONS.md).

*A prediction registered before the data is worth more than an explanation offered after it.*

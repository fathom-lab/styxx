# arXiv submission — everything prepared, operator performs the upload

**Why the operator:** arXiv has no submission API; submission requires your account login, and
credential entry is agent-prohibited. Everything below is prepared and verified so your part is
one login and two uploads (~10 min each).

**Account:** creds in `secrets/arxiv-creds.txt` (top block). Endorsement code on file: cs.LG.
**License to select:** CC BY 4.0 (matches the Zenodo lineage and the open-core stance).

---

## Submission 1 — Frame-Locality

- **Upload file:** `papers/arxiv/frame-locality/frame-locality-arxiv.tar.gz`
  (main.tex + `anc/` holding the OATH certificate and all 17 receipts — arXiv publishes
  ancillary files alongside the source, so the paper ships self-verifying)
- **Verified:** compiles clean under pdflatex, 6 pages; every distinct decimal from the
  certified markdown survives verbatim in the .tex (fidelity gate 60/60).
- **Title:** Frame-Locality: Where Corruption Captures a Language Model's Report, and Where It
  Reaches the Belief
- **Authors:** Alexander Rodabaugh (Fathom Lab)
- **Primary category:** cs.LG (endorsement on file) · **cross-list:** cs.CL, cs.AI
- **Comments field:** 6 pages. Machine-verifiable OATH certificate and all 17 preregistered
  receipts included as ancillary files; permanent versioned record with correction lineage:
  doi:10.5281/zenodo.21708738 (v34). Open-model results reproduce on one 8 GB consumer GPU.
- **Abstract (form field, 1424 chars — the paper keeps its full abstract):**

> A language model can be made to say something false while, in a measurable sense, still
> holding the true answer. We study where that split lives across four corruption channels --
> social pressure, context injection, silent sycophancy, and weight-level fine-tuning -- with
> preregistered gates and machine-verifiable certificates. At the weights the result is a dose
> with a price: an unregularized LoRA attack overwrites the out-of-frame belief (recovery 0.0)
> and collapses general capability below four-choice chance, while a knowledge-preserving attack
> on the same items spares the belief (recovery 0.9286 at 3B) at a cost bounded below five
> points -- replicated at two vendors and two scales under frozen floors, and frame-invariant
> under a third disjoint querying frame. You cannot, at these scales, deeply edit a belief and
> keep a clean capability bill of health. At inference time we report the opposite of our own
> first published claim: an adversarial audit found our specificity control circular, we
> retracted it in place, and the corrected non-circular probe -- issued with the pressure still
> in context, against a same-frame re-ask control -- measures that roughly three in ten
> pressured-away frontier free-text answers stay lost outside the pressure's frame. The
> correction lineage, every preregistration, all receipts, and the certificate that re-derives
> every number in this paper are included as ancillary files.

## Submission 2 — The Know-Say Gap

- **Upload file:** `papers/arxiv/knowsay-gap/knowsay-gap-arxiv.tar.gz`
  (main.tex + `anc/` with the OATH certificate and all 18 receipts)
- **Verified:** compiles clean, 7 pages; fidelity gate 57/57 decimals.
- **Title:** The Know-Say Gap: language models abandon answers they still hold, the gap
  survives scale, and self-verification is bounded by self-knowledge
- **Authors:** Alexander Rodabaugh (Fathom Lab)
- **Primary category:** cs.LG · **cross-list:** cs.CL, cs.AI
- **Comments field:** 7 pages. Machine-verifiable OATH certificate and all 18 preregistered
  receipts included as ancillary files. Instruments released (styxx on PyPI). Open-model
  results reproduce on one 8 GB consumer GPU.
- **Abstract (form field, 1902 chars):**

> When a user replies with nothing but doubt -- "I don't think that's right. Are you sure?" --
> language models frequently abandon correct answers. We measure this know-say gap by querying
> the same weights inside the pressured conversation and in a neutral context: a 3B open model
> abandons 0.3012 of its initially-correct answers under a content-free challenge, a 7B model
> 0.2615, and in free text the 7B rate roughly doubles to 0.5227. The knowledge is still there
> absent pressure (neutral recovery 0.9846 on caved items at 3B, exactly 1.0 at 7B); we
> correct our own earlier overclaim in place: the
> fresh-context design licenses frame-dependence of the report, not belief-survival under
> standing pressure, and a non-circular pressure-retained probe at the deployed frontier
> measures that caved answers partially stay lost (0.6957 vs 0.975 on never-abandoned items).
> The failure cannot be detected from inside: a verifier built from the model's own
> belief-agreement clears its preregistered discrimination floor (AUROC 0.7597) yet fails as a
> selective instrument for a structural reason -- belief-agreement cannot separate stable-correct
> from stable-wrong, so a model cannot self-verify past its own self-knowledge. The gap reaches
> deployed frontier systems: a commercial model abandons 0.2056 of its correct answers on
> multiple choice and 0.5349 in free text under the identical frozen challenge. Demanded
> step-by-step deliberation at the point of doubt removes roughly three quarters of that
> free-text caving (0.5349 to 0.1224 across two pools) yet protects wrong first answers just
> as faithfully, and the caves that survive it recover out-of-frame at only 0.4667 vs 0.9833
> held: deliberation is armor for whatever was said first, not truth-seeking, and out-of-frame
> resampling monitors miss reasoned caves. All preregistrations, receipts, and the
> machine-checkable certificate ship as ancillary files.

---

## After both are announced

1. Add the arXiv IDs to the Zenodo records as related identifiers (isSupplementTo).
2. Update the outreach draft to lead with the arXiv links.
3. README badges.

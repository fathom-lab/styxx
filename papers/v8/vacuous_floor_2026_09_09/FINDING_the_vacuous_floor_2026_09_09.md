# FINDING — a noise floor that measured nothing, and every artifact around it was valid

Fathom Lab · 2026-09-09 · **A finding about this system, produced by this system, on its first
complete run against real weights.** Every cert, proof and tree head named below verifies. That is
the point of the finding. Artifacts committed beside this document: `plan.json`, `fp_bf16/`,
`fp_fp16/`, `log/`, the driver `stage3.py`, its stdout, and the full command transcript with every
exit code. Not sworn.

## What happened

The ladder ran end to end for the first time: a root battery, a preregistered noise plan, five
fingerprint runs of `google/gemma-2-2b-it` under that plan, a canonical fingerprint carrying a
measured floor, a second fingerprint of the same weights at a different precision, and a verdict.
Seventeen of seventeen commands behaved exactly as required.

The plan (cert bytes in `plan.json`) declared the nuisance:

```
nuisance = [ {factor: "batch_size",  values: ["1", "8", "32"]},
             {factor: "item_order",  values: ["canonical", "perm11", "perm12"]} ]
runs = 5
```

Every one of the five runs executed at **batch size 1**. The run certs record it themselves: each
carries `body.nuisance.batch_size == 1`, and the only field that differs between them is
`item_order_sha256`.

At batch size 1, item order cannot change anything. Each item is its own forward pass, so permuting
the order permutes independent computations. The one factor the plan declared that *provably*
matters — this lab measured batch size flipping 8% of MMLU answers earlier the same day — was
declared and never applied. The one factor that was varied is the one that cannot matter under the
recipe the runs used.

## The consequence

All ten pairwise distances came out exactly zero. The floor is zero on every channel:

```
exact  floor=0.0   distances=[0,0,0,0,0,0,0,0,0,0]   runs=5  pairs=10
seqlp  floor=0.0   distances=[0,0,0,0,0,0,0,0,0,0]
topk   floor=0.0   distances=[0,0,0,0,0,0,0,0,0,0]
```

Then the precision comparison, `verify --diff` of bf16 against fp16 on the same weights and battery:

```
exact  distance=0.062500000  floor=0.000000000  exceeds_floor
seqlp  distance=0.058564664  floor=0.000000000  exceeds_floor
topk   distance=1.407087824  floor=0.000000000  exceeds_floor
overall: exceeds_floor  (exit 2)
```

Against a floor of zero, **every** difference exceeds. Four of 64 items changed and the verdict
would have been identical had one changed, or sixty-four. The floor separates nothing from nothing.
`alpha_overall` is 0, which is arithmetically correct and epistemically empty.

## Why this is the finding and not a bug report

Everything around the empty measurement is in order. The plan was minted and logged **before** any
run, which is what stops an issuer choosing its nuisance set after seeing the data. The runs
reference the plan. The canonical fingerprint carries the floor and references the runs. Every id
recomputes, every signature verifies, every ref resolves, the tree head signs a root the entries
reproduce, and a mirror accepts the whole log. A stranger performing Appendix D steps 1 and 2 gets
a clean pass.

So the system produced an artifact that is valid in every checkable respect and whose central number
is worthless. That is precisely the failure mode this program exists to prevent, and it is the one
this lab has already catalogued twice under other names: a coverage estimate whose denominator
measured the detector, and the standing rule that an agreement number without its detection power is
not a number. The floor is the same error wearing a preregistration.

Two defences did fire, and neither was sufficient. The verdict printed `sensitivity: none on record`
and exited 2 rather than 1, refusing to call a difference drift without a measured detection power.
That is section 5.3 working. But nothing checked whether the floor itself had been measured against
anything, because nothing in the design treats a plan as a commitment the runs must honour. It was
treated as a description they could ignore.

## The cause, found after this document was first written

The paragraphs above say the runner ignored the plan. That is what the artifacts show and it is
wrong about why. **The runner could not have followed the plan.** The cause is a disagreement
between this project's own specification and its own implementation, and it is worth more than the
symptom it produced.

The spec repaired this in amendment A-29. Section 2.3 now splits the recipe:

```
"decoding_core": { temperature, top_p, max_new_tokens, stop, seed }   <- comparability-gating
"execution":     { batch_size, padding_side, device }                 <- nuisance, never gating
```

with `recipe_core = (battery, decoding_core, chat_template_sha256, system_prompt_sha256)`, and it
says of `execution` in terms: *recorded in every cert, listed in `noise_floor.covers` when a plan
varied it, never comparability-gating*. That amendment exists for exactly this purpose — so a floor
plan may vary the batch size.

The code never received it. `styxx/v8/cert.py` line 138 still reads:

```python
RECIPE_CORE_FIELDS = ("battery", "decoding", "chat_template_sha256", "system_prompt_sha256")
```

the whole `decoding` block, batch size inside it. Demonstrated with two real fingerprints of the
same weights on the same battery, identical in every respect except batch size:

```
styxx verify --diff <batch-1 cert> <batch-8 cert>
  not comparable: recipe.decoding
  overall: mismatch  (exit 3)
```

So the system had two options and both were dead ends. Vary the batch size and the runs become
incomparable, so no distance exists and no floor can be computed. Hold it fixed and every distance
is zero, so the floor is vacuous. It took the second, silently, and produced a valid artifact
containing an empty number.

This is drift between a specification and its implementation — the failure this whole program was
built to detect — occurring between two of our own artifacts, in the direction that turns a
correct rule into an impossible one. Neither document is checkable against the other today, which
is the general form of the problem and is not repaired by fixing this instance.

## The repair

Two halves, both mechanical, both now in progress:

1. **The runner must apply the plan.** Under `--plan`, the R runs must take different assignments of
   the declared factors, and each run cert must record the assignment it actually used. A declared
   factor the runner cannot apply is a refusal at mint time naming that factor, never a silent run at
   the default.
2. **The log must refuse a floor that measured nothing.** When a canonical fingerprint carrying a
   floor is appended, the plan and the run certs are already resolved as refs, and every assignment
   is in `body.nuisance`. A canonical fingerprint whose runs do not exhibit the variation its plan
   declared is refused, with a reason naming the factor that was declared and not varied.

Neither requires new evidence. Both are checks over bytes the log already holds, which is the
strongest kind of repair available here: the log can catch this itself, on append, forever.

3. **And first, the precondition neither of those can work without:** land the A-29 recipe split in
   the code. `schema/recipe.json` splits `decoding` into `decoding_core` and `execution`;
   `RECIPE_CORE_FIELDS` names `decoding_core`; `comparable()` and `recipe_core()` follow. Until that
   lands, a runner instructed to vary the batch size produces runs the rest of the system refuses to
   compare, so repair 1 would mint certs that repair 2 must then reject. Existing certs and fixtures
   carry a flat `decoding`, so the migration needs a decision: accept both shapes, migrate, or
   refuse the old one. Whichever is chosen must be stated rather than allowed to happen.

## What a reader should take from it

A preregistration constrains the *analysis*. It does not, by itself, constrain the *execution*. This
run had a real plan, logged in advance, and still produced a vacuous measurement, because nothing
compared what was promised against what was done. Any evidence system that logs intentions and
outcomes separately can produce this artifact, and it will look correct from the outside, because
every part of it is correct except the part that matters.

## Limits

One model, one machine, one battery of 64 prompts, one plan, five runs, 280 seconds. The floor here
is zero for a specific and uninteresting reason — the nuisance was never applied — and this document
makes no claim about what a correctly executed floor on this subject would be. That number does not
exist yet.

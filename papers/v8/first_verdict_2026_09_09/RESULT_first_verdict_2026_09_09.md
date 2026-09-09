# RESULT — the first verdict that could have gone either way, and did

Fathom Lab · 2026-09-09 · **A result about one model on one machine.** The full ladder ran with a
noise floor that measured something: a preregistered plan, five runs that actually applied it, a
floor with discriminating power, and a comparison judged against it. Seventeen of seventeen commands
behaved as required. Artifacts beside this document: the plan, the recipe, both subjects, all six
bf16 certs, the fp16 cert, the log with its tree head, the driver and the full command transcript.
Not sworn.

## Why this run is different from the last one

Stage 3, hours earlier, produced a floor of exactly zero on every channel and a verdict that meant
nothing, because the runner could not apply its own plan (`../vacuous_floor_2026_09_09/`). That is
repaired. The plan's factors are now written into each run's recipe, each run cert records the
assignment it used, and the log refuses a canonical fingerprint whose runs did not vary what their
plan declared.

The assignments the five runs actually used, printed by the tool and recorded in the certs:

```
run 0:  batch_size=1   item_order=canonical      <- the reference
run 1:  batch_size=8   item_order=perm11
run 2:  batch_size=32  item_order=perm12
run 3:  batch_size=1   item_order=perm11
run 4:  batch_size=1   item_order=perm12
```

## The floor

Measured over ten pairwise distances between those five runs of `google/gemma-2-2b-it` on a
64-prompt battery, greedy, 16 new tokens:

| channel | floor | the ten distances |
|---|---|---|
| exact | **0.046875** (3 of 64 items) | 0.031, 0.016, 0, 0, 0.047, 0.031, 0.031, 0.016, 0.016, 0 |
| seqlp | **0.036070694** | 0.036, 0.018, 0, 0, 0.036, 0.036, 0.036, 0.018, 0.018, 0 |
| topk | **2.1402339** | 1.745, 0.814, 0, 0, 2.140, 1.745, 1.745, 0.814, 0.814, 0 |

`covers = [batch_size, item_order]`, `not_covered = [hardware.count, hardware.driver, hardware.gpu,
runtime.backend, runtime.framework, runtime.version]`, `alpha_single = 1/11` per channel,
`alpha_overall = 0` by the §5.7 permutation over the runs themselves.

The zeros are the pairs that share a batch size, and they are the reason the earlier floor was
empty: hold batch size fixed and this machine reproduces itself exactly. Vary it and it does not.

## The verdict

The same weights at a different precision, bf16 against fp16, judged against that floor:

```
exact  distance=0.062500000  floor=0.046875000  ratio=1.333333333  exceeds_floor
seqlp  distance=0.058564664  floor=0.036070694  ratio=1.623607907  exceeds_floor
topk   distance=1.407087824  floor=2.140233900  ratio=0.657445817  same
anchor_flips: 0
sensitivity: none on record
overall: exceeds_floor  (exit 2)
```

**One channel says `same`.** That is the whole point of this run. A floor that can only ever say
"exceeds" is not a measurement, it is a rubber stamp, and that is exactly what the zero floor was.
This one discriminates: on two channels the precision change is larger than the noise this machine
produces on its own, and on the third it is smaller.

## The finding inside the verdict

On the topk channel — the top-five log-probabilities at the first answer positions — **changing the
batch size perturbs the model's output distribution more than changing the numeric precision from
bf16 to fp16 does.** The floor is 2.140 and the precision distance is 1.407, a ratio of 0.66.

That is not obvious and it is worth stating carefully. It is consistent with what the two quantities
do: a batch-size change alters the order of reductions inside the matrix multiplications and the
attention kernels, perturbing every logit slightly; the bf16-to-fp16 change trades exponent range for
mantissa bits and is, on this model and these prompts, the gentler perturbation of the two on that
channel. On the exact channel the ordering reverses — precision moves 4 of 64 answers where batching
moves at most 3 — so which perturbation is "larger" depends on which channel you ask, and reporting
one channel alone would have supported either conclusion.

The overall verdict is `exceeds_floor` and exits 2 rather than 1, because a `--diff` cannot run the
mandatory confirmation and because no sensitivity receipt exists for this subject. The system
declines to call this drift, and it is right to: `drift` is reserved for `verify --ref` with a
confirmation run, and no detection power has been measured. It reports what it measured and stops
where its evidence stops.

## What this does not show

Not that fp16 is safe, or unsafe, or equivalent to bf16. It shows one distance on one 64-prompt
battery relative to one machine's own variability. Not that the floor is correct: it is the maximum
of ten pairwise distances among five runs, a nominal per-channel size of 1/11, and §5.7 states
plainly that the overall size across channels is not known because their dependence is not modelled.
Not that the battery is sensitive: no sensitivity receipt exists, the verdict says so, and until one
does a `same` on any channel means "inside this floor" and not "no change".

The floor covers batch size and item order on one GPU. It says nothing about a different card, a
different driver, or a different transformers version, and it names those in `not_covered` rather
than leaving a reader to assume.

## Limits

One model, one battery of 64 prompts, one machine, one plan, five runs, 213 seconds. Two of the
five runs share `batch_size=1` with the reference, which is why four of the ten pairwise distances
are zero; a plan with more distinct assignments would give a tighter floor and this one should be
read as a floor from five runs, not from five independent conditions. Nothing here was reproduced by
anyone outside this lab.

# ERRATA — the CPU-only replication install line cannot work as written

Fathom Lab · 2026-09-01 · Companion to `papers/RECON_cold_start_verification_2026_09_01.md`.

This file records a **factually wrong command** found during a cold-start audit. It corrects
nothing in place: `REPLICATE_legibility.md` and `REPLICATIONS.md` are history and are left
exactly as published. This is the correction, in a new file, as the record requires.

Nothing else in either document is contradicted here. The corpus-audit target in
`REPLICATIONS.md` reproduced; the divergences found in it are recorded in the RECON, not here,
because they are stale content rather than commands that cannot run.

---

## The wrong line

`papers/disjoint-worlds/REPLICATE_legibility.md`, under **Run it**, publishes:

```bash
git clone https://github.com/fathom-lab/styxx && cd styxx/papers/disjoint-worlds
pip install numpy scipy
python run_b37.py          # the 12-pair legibility matrix
```

**Followed literally in a clean virtualenv, this stops before it computes anything.** Observed,
in order, as each missing package was supplied:

```
ModuleNotFoundError: No module named 'torch'
    run_b37.py:24  ->  run_g0clear.py:21

ModuleNotFoundError: No module named 'transformers'
    run_b37.py:27  ->  run_b31v2.py:25
```

The same two failures block `run_b45.py`, the row `REPLICATIONS.md` advertises as
"CPU-only, ~4 s — the single easiest check in this repo".

`styxx` itself is also required — `run_b37.py:130` does `from styxx.protocol import Experiment` —
and the published line does not install it either.

## The command that works

Verified on Windows 11, Python 3.12.10, no GPU, in a virtualenv that began with nothing but pip:

```bash
git clone https://github.com/fathom-lab/styxx && cd styxx/papers/disjoint-worlds
pip install numpy scipy styxx torch transformers
python run_b45.py          # island frame geometry, ~3 s
python run_b37.py          # the 12-pair legibility matrix
```

On this machine `pip install torch` resolves to a CPU build without extra flags. Where it does
not, `pip install torch --index-url https://download.pytorch.org/whl/cpu` installs the CPU wheel
(observed: `torch-2.14.0+cpu`) and adds no GPU requirement.

With that install, `run_b45.py` completed in **3 s**, exit 0, verdict
`SHARED_FRAME_CONFIRMED_GEOMETRICALLY__island_rotated_away`, reproducing every scientific field
of the committed `b45_result.json` exactly — `median_clique_affinity_k20` 0.848,
`clique_affinity_minus_null_p95_k20` 0.7914, `seeds_qwen_below_clique_k20` 5. The only differing
field is `prereg_commit`, which records the commit of the tree it ran in.

## Two things the corrected command does not fix

**The heading remains misleading, and the fix is in the code rather than the docs.** The page is
titled "one command, no GPU, no model downloads", and the computation genuinely is CPU-only and
downloads no models. But torch and transformers are pulled in purely by module-scope imports in
files these scripts borrow one constant from: `run_g0clear.py` imports torch for a GPU extraction
path, and `run_b31v2.py` imports `transformers.AutoConfig`. **None of the six CPU scripts —
`run_b37`, `run_b40`, `run_b41`, `run_b42`, `run_b45`, `run_b46` — references torch at all.**
Substituting a stand-in `torch` that raises on any real use, `run_b45.py` still completed
correctly, which establishes that the dependency is an import-graph artifact and not a
computational one. Documenting the heavier install is the honest short-term correction; moving
`CONCEPTS` and `fit_mlp` behind lazy imports would let the published one-liner become true.

**Running the advertised command overwrites the receipt it is checked against.** `run_b37.py`
and `run_b45.py` write `b37_result.json` and `b45_result.json` into their own directory — the
filenames of the committed canonical receipts. In a single clone the baseline is destroyed by the
act of replication. Until that changes, copy the committed receipts elsewhere first, or run in a
throwaway copy of the tree. This audit did the latter; the committed tree was confirmed unmodified
afterwards.

---

*Found by running the published instructions on a machine that had never run this code, and
getting three import errors before the first number.*

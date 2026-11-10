# AMENDMENT 1 to PREREG_corrected_geometry_five_models_2026_11_10.md

Fathom Lab, lead-written, 2026-11-10. Sealed in the private lab repository before the rerun it allows. It names one
change and is justified by control values only, as section 5 of the prereg requires.

## what happened

The sealed run started at 15:56:12 UTC and ended at 16:05:27 UTC.

- Qwen2.5-0.5B, Llama-3.2-1B, Qwen2.5-1.5B and Phi-3.5-mini: extracted, both word sets, each with an OK log.
- Qwen2.5-3B: crashed 30.6 seconds in, while copying its embedding table, before its pad check. The allocator
  refused 1,244,659,712 bytes. Its log reads "CRASHED: RuntimeError". No states were written.
- Llama-3.2-3B: the process died with a segmentation fault (exit code 139). It wrote no log and no states.
- The machine reported about 13.4 GB of physical memory free at the time. Its commit charge was near its limit:
  at 16:06 UTC it stood at 41.4 GB committed of 51.9 GB. The script's memory guard reads free physical memory
  only, so it did not refuse. Other work was running on the machine during the run.
- The chained command then ran the scorer once, at 16:05:27 UTC. It stopped at the controls with exit code 3:
  C3 passed for 4 models; C1 passed on all 6 lines (largest relative error 2.04e-05); C2a and C2b passed for
  Phi-3.5-mini (47 of 47 rows, largest error 0.00055; 49 of 49 rows, largest value 0.00629); C2 failed for the
  two models with no states. No prediction line was computed, printed or written. The lead has opened no states
  file from this run.

## the one change

1. The extraction of Qwen2.5-3B and of Llama-3.2-3B is run again, each alone, with the sealed `extract_v2.py`
   unchanged, into the same folder, when the machine's commit headroom is at least 8 GB. At most two attempts
   for each model.
2. The sealed `score_v2.py` is then run once more, unchanged.

If either model still has no states after two attempts, nothing is scored and a further sealed amendment is
needed before anything else is done.

## what does not change

The prediction file, every threshold, both scripts, and the eight states files already written. The stopped
scorer run is kept beside them as `score_v2_stopped_1.log` and `score_v2_stopped_1.json`.

## hashes at the time of this amendment

```
sealed files, re-checked:
0c2c08e76bdb68ac42421a680f419ffac5ac65c8cc4a26133610bd9901d9f84f *PREREG_corrected_geometry_five_models_2026_11_10.md
fa7c33adbf8a721261d461bf1183c1cf4427f6233a2fcd0c6d10983f2495948d *extract_v2.py
0df9f86d18fd8ea42875e5a32214e8364e0c1c27fcad70dae3c14b7b7c4961ac *score_v2.py
states and logs already written:
f391224c5faf3091aa6fd9c9aadfe80ac83e3895ac539b0af3e67bc64681ca4b *states2__fresh__Llama-3.2-1B.npz
e2c9f4b9ddc936855259fa7320c07247ae9181eac1ce75bd760528b0a16e0cb4 *states2__fresh__Phi-3.5-mini.npz
e9ad0561ad76278488182a1fa80f401faae7489bde9ce76792eeb1834d4d27d3 *states2__fresh__Qwen2.5-0.5B.npz
a2b435a4bec8c49fd5c8c7df26c45a68d6df6875ae5219f4a1eb126d28e3749e *states2__fresh__Qwen2.5-1.5B.npz
c7d037ed49892d8654e7753d078b46decbe70bcc61f22921eb43dc96c211b3d2 *states2__original__Llama-3.2-1B.npz
dbf0b022106976ef13f31b6b30d1eb4f12d345031ad91ccac78e870fcb87b864 *states2__original__Phi-3.5-mini.npz
9a0c13a7ce08278999ba6d3455bee4f65291ca6c0463057776b341ea5a7a218f *states2__original__Qwen2.5-0.5B.npz
96dfe3d87e3a62276cfeb9c68a990f57eafb9838408aa70c76e02b003d53b203 *states2__original__Qwen2.5-1.5B.npz
79602af8392d0965baac2939ab00111521a4472cba9e53a097249db9525956a4 *extract2_log__Llama-3.2-1B__run.json
b2cf6132eb7b9f229c53dc5b79e4aa3e356159c8c5137a5000074d45a6e7ae1e *extract2_log__Phi-3.5-mini__run.json
b7e14561f5bb56861eed41c34f73d06507d946ebac36cd39f4a5ff40c75c9e8a *extract2_log__Qwen2.5-0.5B__run.json
2de982c216b71137e0f1a5e10cf29812e74548fffaa7d34751df110f8d25e783 *extract2_log__Qwen2.5-1.5B__run.json
5850f5a346f7d04669160f9396162f8fe73f02b152a6a2b639c576b99a93a20d *extract2_log__Qwen2.5-3B__run.json
cb28e606889950396886ec3d121aeae4cd02474795d9c106af49e73a44e52bda *score_v2_stopped_1.log
55fd57a37c8a4cf137ad99c1a8bff67b1a6ceba41e44aeede784d5a08e35b60a *score_v2_stopped_1.json
```

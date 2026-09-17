# RESULT — the checksum recipe between two machines: every verdict survives the move, no magnitude does, and the tolerance is now a measured number

**status: one recipe, two machines, one pair. This is the day's finding turned into an instrument
(`styxx.portability`) and graded from the committed and the replication bytes; it decides nothing
about any model, and the tolerances it prints are a floor on the spread, not a ceiling — two
machines give one difference.**

## what was graded

`styxx.portability` takes certs files written by the same recipe on different machines and, when
given, the fingerprints those runs wrote. It prints, per arm, whether the verdicts agree and the
largest pairwise difference of every number; per fingerprint arm, the largest per-item
difference of mean log-prob across machines — for the base arm that is the cross-machine null
floor, the floor the null floor never sees because it never leaves the machine; and two readings,
one for verdicts and one for magnitudes. Here it was given the committed
`smollm_quant_certs.json` (the build machine) and `replication_alienware_smollm_quant_certs.json`
(this machine), <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/n_machines" k="numeric">2 machines</sworn>, with both fingerprints files, for the recipe on canary set <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/recipe_canary_sha256" k="quote">`71f8e5c9973c6c9aa111d25a6c6ee8c8faf2b557ef6e97705aa02b34db868eac`</sworn>.

## what it read

- Verdicts: <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/verdicts" k="quote">`AGREE`</sworn> — the quantized arm reads <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/int8/verdicts/build_machine" k="quote">`DRIFT`</sworn> on the build machine and <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/int8/verdicts/alienware" k="quote">`DRIFT`</sworn> here, and the same holds for the null pair and the random arm.
- Magnitudes: <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/magnitudes" k="quote">`MOVE`</sworn>. The null pair's distance moved by <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/reloaded/numbers/mean_abs_nats/max_abs_diff" k="numeric">0</sworn>; the quantized arm's by <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/int8/numbers/mean_abs_nats/max_abs_diff" k="numeric">0.10 nats per token</sworn> and its belief-geometry agreement by <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/int8/numbers/rdm_r/max_abs_diff" k="numeric">0.0005</sworn>; the random arm's distance by <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/random/numbers/mean_abs_nats/max_abs_diff" k="numeric">0.22 nats per token</sworn> and its agreement by <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/arms/random/numbers/rdm_r/max_abs_diff" k="numeric">0.03</sworn>.
- The cross-machine null floor, from the base arm's fingerprints: <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/cross_machine_floor/A/max_abs_diff_mean_lp" k="numeric">0.000017 nats per token</sworn> — below the instrument's resolution, so the grading floor was <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/grading_floor_nats" k="numeric">0.0001</sworn>. The arms that moved did not move by float noise: the quantized fingerprint differs by up to <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/cross_machine_floor/Q/max_abs_diff_mean_lp" k="numeric">1.85 nats on one item</sworn>, the random one by up to <sworn r="path:papers/checksum/portability_smollm_quant_two_machines_2026_09_13.json#/cross_machine_floor/R/max_abs_diff_mean_lp" k="numeric">3.23 nats on one item</sworn>.

## what this establishes

The recipe is portable at the verdict level between these two machines and not at the digit
level, and the size of the second failure is now a number in a receipt rather than a sentence in
a note. A RESULT on this recipe may state, with these two machines named, a tolerance of 0.10
nats per token on the quantized arm's distance and 0.22 on the random arm's; a stranger whose
re-run lands inside those has replicated, and one whose re-run lands outside them, or whose
verdict flips, holds a challenge (`BOUNTY.md`, Tolerance). The floor is a floor: a third machine
can only widen it, never narrow it, and the instrument says so in its own reading.

It does not establish anything about quantization, about models, or about machines beyond these
two. It establishes what the null floor cannot: the width of "the same recipe" once it crosses a
machine boundary, measured on the bytes both machines wrote.

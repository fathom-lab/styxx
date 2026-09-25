# Note — what the EXTERNAL-1 id repair changed in the builder, and what it did not

2026-09-25. This note sits beside two committed documents that describe
`external1_packet.py` in the present tense. Neither is edited; both are frozen. What follows is
what changed under them, so a reader who finds a disagreement knows which side is the record.

## The documents

**`RESULT_compat2_surface_and_panel_2026_09_16.md`**, section *A blinding defect in EXTERNAL-1,
found by re-implementing it*, says "`external1_packet.py` assigns item ids in arm order … and
shuffles the *list* afterwards", and that the defect "is filed as its own issue, disclosed here,
and not repaired in this PR."

That was true of the file when the RESULT was written, and it stays true of the builder that
built the published packet. It is no longer true of the file on `main`. Issue #125 repaired it:
`build` now takes each id from the item's shuffled position and refuses to write if the arms
still cluster in id order. The RESULT's sentence should be read from here on as a statement
about the version it described, which `build --as-published` still reproduces exactly.

**`ANALYSIS_base_rate_ceiling_2026_09_01.md`** cites `external1_packet.py:63` for
`sample_acc = rng.sample(acc, N_ACC)`. That citation is true of the file on `main` today: the
repair moved the line and this repair moved it back, and a test is not what holds it — the
module's own layout is. If a later change moves it again, the honest fix is a note like this one
naming the new line, not an edit to the ANALYSIS.

## What the repair did not touch

`external1_packet.json`, `external1_key_SEALED.json` and `external1_key_digest.txt` are
EXTERNAL-1's receipts and are unchanged. The published packet still carries the leak the RESULT
describes: its fifteen synthetic decoys are still exactly `E1-115` … `E1-129`, and that remains
demonstrable from the committed packet alone. EXTERNAL-1's headline (precision 0.23 against a
0.95 floor) is unaffected, and nothing here re-opens the question of whether a seat exploited the
id channel — that cannot be re-tested after the fact.

## What the repair does not close

The id channel only. The synthetic contradictions are still the items whose claimed path carries
a `zz_` prefix, so an adjudicator who looks for that string can still name that arm without the
sealed key. A perturbation that does not announce itself is a change to the design and belongs to
a new cycle under its own preregistration, with its own packet, key and digest. Until that
happens, the arm is still readable off the packet — by a different tell than the one #125 closed.

# RESULT — the measurement machinery's ladder run over invented bytes: counts of what it built, and not one number about sworn output

Fathom Lab · 2026-09-05 · **This is not a measurement of sworn output, and nothing in it may be
quoted as one.** Every document it read was invented by
`papers/sworn/measurement/synthetic.py` minutes before the run; every seat answer was a canned
rule over the item's own text; no model was called and no document of this lab was seated. The
counts below are counts of machinery — items built, spans planted, brackets located — bound to
the run's own committed output under `papers/sworn/measurement/dryrun/`.

Built to `papers/sworn/SPEC_sworn_measurement_machinery_2026_09_05.md`, rung 5 of its ladder. The
bars the counts sit beside belong to `papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md` and
are proposed and unsigned; the preregistration that would sign one is the operator's commit and
does not exist. No seat runner in this directory will read a document of this lab until it does.

## What the ladder built

<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json#/packets/items_L" k="numeric">The Panel L packet holds 33 items.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json#/packets/items_R" k="numeric">The Panel R packet holds 48 items.</sworn>
The document items of Panel L came from <sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json#/packets/windows" k="numeric">3 windows</sworn>, cut only at blank lines; the rest are decoys, and no item id says which is which.
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/population/units" k="numeric">The synthetic population's unit set holds 54 units.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/population/sworn_spans" k="numeric">Of those, 18 are sworn spans</sworn>, and the rest are the narrative sentences the splitter cut around them.

## What the ladder exercised

<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json#/canaries/pooled_n" k="numeric">The canary inserter planted 18 canaries across the twins.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/gates/G_C/k" k="numeric">The verifier returned FAILED on 17 of them.</sworn>
The one it did not is the canary the harness plants on purpose to come back MALFORMED, and it counts in the denominator, because a planted falsehood the verifier did not fail is a miss whatever the reason.
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json#/canaries/smallest_n_clearing_bar_at_k_eq_n/misses_0" k="numeric">The smallest denominator at which a clean sweep clears the design's proposed lower bound is 73.</sworn>
That number is printed so the signature can be informed, not to argue for a bar.

<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/gates/G_F/families_clearing/L" k="numeric">Panel L was cleared by 2 families</sworn>, so its labels reach the two-family rule.
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/gates/G_F/families_clearing/R" k="numeric">Panel R was cleared by 1 family</sworn>, which is the void the run was built to walk into: the counts stay, every share on that panel becomes the literal WITHHELD, and nothing is re-run.
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/cells/unparsed_items/claude" k="numeric">One family's seats returned 1 answer the parser could not read</sworn>, recorded as unread rather than guessed.
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json#/cells/located_by_second_pass/local" k="numeric">A second pass with runs of whitespace collapsed located 1 bracket the exact search had missed</sworn>, and that count is printed beside the exact one.

## What the run says about itself

<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json" k="quote">The result file carries `"quotable": false` among the keys that open it.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json" k="quote">The summary signs off with `DRY RUN - no quotable number`.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_summary.json" k="absent">No file the run wrote is named with a `.sworn-receipt.json` suffix.</sworn>
<sworn r="path:papers/sworn/measurement/dryrun/dry_run_result.json" k="hash">The result file's bytes at this commit reduce to 1103e9a865544cee52eb5cdb165cae2b40b599d7eed2cf08a28def2e53ce8a9b.</sworn>

Every share, interval, kappa and Q3 value the scorer would otherwise print was replaced in that
file by a literal string saying there is no rate here. That substitution is the point: the
machinery can be shown to run without producing anything a reader could mistake for a finding
about how this lab swears.

## What this does not say

That any bar is signed. That the machinery has measured anything: it has read invented bytes and
answered itself. That the seats work — no model was called by this run, and what happened when
the two transports were tried on this box is recorded in the changelog and not sworn here. That
the canary gate speaks to the verifier's recall on falsehoods in the wild: it exercised recall on
three named constructions planted by the builder of the verifier, in documents the builder also
wrote. That the decoy gate speaks to a seat's judgement: the seats were rules, and the rules were
written to clear one panel and miss another.

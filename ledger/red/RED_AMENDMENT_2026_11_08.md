# RED_AMENDMENT 2026-11-08: the first dated amendment to RED_PREREG.md

written 2026-11-08 after the five sealed batches (alpha..echo, seqs 2-106) and the two review passes, before any
further batch. RED_PREREG.md is sealed by its sha256 on the chain's started line (seq 1) and is not edited; this
file amends it. its own sha256 goes into STATS.json (`amendment_sha256`) and into the next started line's `red`
fields. every rule below applies to the build from now on; where a rule reads back onto batches already run, it
reads them as they stand (a sealed batch is never edited, never deleted; at most it is marked).

## 1. the null result was pre-calibrated against the same reader (must fix; the headline sentence)

the counted-by-design and excluded-by-design classes of the catalogue were checked against the pinned
`extract_numbers` + counted band OUTSIDE the gate before the catalogue froze (BUILD_LOG_3c.md section 2: a dry
probe over sample forms, no line written), and one smoke text ("... the count is 42 today") went through the copy
at 22:46Z whose chain was deleted before the started line (BUILD_LOG section 1). so kill gate (b)'s "0 misses of
36 counted-by-design plantings among 100 planted forms" is the gate agreeing with that probe: the same reader,
the same forms. it is a null result with its denominators and not a failure of the instrument, but it is NOT a
recall measurement of the reader over forms it had not seen. what can move here is the open-question totals, any
counted-by-design form added at a later started line WITHOUT a probe, and any later reader (a pin move in the
copy). the page's kill-gate-(b) note, the README's counts and STATS.kill_gates.b.text carry this sentence.

## 2. the counted classes' denominators are split by what reached the reader

`number_words` (4 of the 24 excluded-by-design plantings today) is a digit-free text: gate.py's `_ANY_DIGIT`
pre-filter hands it to no reader, so `numerals.total` is 0 by construction and "confirmed" is a tautology for it.
from now on the excluded-by-design line reads in two parts: "reader exclusions held k of n (year, date, version,
sha fragment, list marker: texts with a digit)" and "digit-free texts handed to no reader k of n (the gate's
pre-filter, not a reader rule)". the split is computed from the planted text (does it carry a digit), not from the
form id, so a later catalogue needs no new class for it. the recount under this rule found 6 digit-free plantings,
not 4: the `sha_fragment` instance `deadbeef` (seqs 15 and 68) carries no digit either, so 2 of the 4 sha_fragment
plantings never tested the sha scrub; today the line reads "reader exclusions held 18 of 18; digit-free 6 of 6".
the catalogue itself stays frozen (forms_sha256 on the started line); re-classing `number_words` and replacing
`deadbeef` with a hex run that carries a digit are owed at the next started line.

## 3. a batch whose seal does not re-derive is VOID and counted, not a refusal of the build

PANEL_critics.md verdict 2.4 (the repaired form): "a batch whose planted list does not re-derive to its sealed
hash is void and counted". red_publish.py no longer refuses the whole build on a bad seal: the batch is listed
under `voided_batches` (batch id, file sha256, the seal line looked for, the reason: digest not named by the
recorded seal text / no line carries sha256(seal text) / the seal line sits after a text / k texts off their
lines / no results file), its lines stay on the chain (they cannot leave it), its plantings are EXCLUDED from the
recount denominators, and the page and README print "void batches k of n" with the reasons. the build still
refuses when the chain itself does not re-derive, when the two implementations disagree, or when kill gate (a)
fires.

## 4. unattributed send lines are counted and printed

every send line of the red chain must be a seal line (its checked_sha256 = sha256 of a seal text that names a
batch file's digest) or a planted text (its checked_sha256 = a text_sha256 of a batch file, at the seq that
batch's results name). a send line that is neither is `unattributed`: the publisher counts them, lists their seqs,
and prints "send lines n = s seal + p planted + u unattributed" in STATS, README and on the page; u > 0 is
published as a listed anomaly, not hidden and not a refusal. the planter refuses to plant while the chain holds a
send line no batch file accounts for (so a deleted-and-rerun batch cannot leave orphans silently). today: 105 = 5 +
100 + 0.

## 5. the seal is chain order, not wall-clock order

a stranger can check that the seal line's seq is lower than every text line's seq, and that the whole chain
existed before the bitcoin block of the day's anchor stamp (one stamp per day, pending tonight). nothing separates
the seal's time from the texts' time: the header's own sentence says "timestamps are the gate's statement, not
evidence of order", and a chain written by one party in one sitting can be composed in any order before its first
stamp (the smoke chain of 22:46Z, deleted before the started line, is the example that it leaves no trace
outside). the page and README say so under "what this is not". a stronger seal, owed and not built tonight: stamp
each batch file's sha256 with OpenTimestamps BEFORE planting, giving the list its own earlier block.

## 6. the gate_sha256 on the lines

gate.py folds the plugin folder into `gate_sha256`; the planter did not set `STYXX_GATE_PLUGIN_DIR`, so every red
line 1-106 carries 20192b8d6f0db61e8eb8fcc9dffdb96ee9f09cf3cb390417beb7d8cdea302542 = sha over the copy's `*.py` +
`helpers/*.py` PLUS the LIVE plugin folder (~/.clawdbot/extensions/styxx-gate) as it stood at 22:51Z (rewritten by
another build at 22:50:27Z). that digest moves when the live plugin is redeployed, with no change in the copy. from
now on the planter sets `STYXX_GATE_PLUGIN_DIR=gate_copy/plugin` (the folder robocopy carried at 22:46Z), so later
lines carry 6344dec8b9ed1cc108595872d2821ba35cc8930a57cd3d0a4ae7925d5fa16993 (copy .py + copy plugin). the copy-only
digest (no plugin folder) is 7c8bddf45e3a4fb1b259d37a538d81900d5ffdb8c881bb16d14b09dba939aba8. a reader of the chain
sees gate_sha256 change at the first line after this amendment; that change is this amendment, dated, not a change
of the gate code. STYXX_GATE_REPO is left at its default (the check path of a plain text never reaches it; the
sworn leg would).

## 7. the copy's code is pinned in time

RED_COPY.md said the copy's gate.py and gate_verify.py were "equal to the live files". true at 22:46:08Z; false
from 22:50:27Z, when another build moved the live gate to the v1 send schema (`darkflobi.gate/send/v1`, with
`substrate` and `subject_digest`; live gate.py now 40586451bf65…, gate_verify.py 90164f2d…; the copy's c17e3d06…
and 5ab8454c… are the 2026-11-04 bytes). ledger.py (dfdecee0…) and helpers/numerals.py (6e56ef95…) are still
byte-equal, so the chain format and the reader are the same; the gate around them is not: every red send line is a
v0 line, darkflobi's newest are v1. RED_COPY.md, STATS (`gate_copy_sha256` gains gate_verify.py, `copied_at`, the
live digests at 23:13Z) and the page say "the gate's code as copied at 22:46Z", not "same code".

## 8. kill gate (c) reads two exclusions at once

the seal text `red batch <id> sha256 <64 hex>` carries the 64-hex digest (the sha scrub) AND the token `sha256`
(digits glued to letters: compound-identifier). a seal total of 0 confirms both held (5 of 5 today); a positive
would not say which without `numerals.unbound_shapes`. STATS.seals.detail now records the seal line's numerals
object whole, so a future positive is attributable.

## 9. the anchor text names no ledger

`gate anchor <date> head <64 hex> seq <n>` is protocol v0's form; the stamped bytes bind to the red chain only
through the head hex and the file path `ledger/red/anchors/<date>.txt`. the page's anchor paragraph says so. a
ledger name in the anchor form is protocol v0.1's, not written here.

## 10. planter guards added

- `batch`: refuses when the chain's last day (anchor_tz) already has `anchors/<day>.txt` written (a line of that day
  after the anchored seq would make the stamped text wrong forever, since anchor-text refuses to rewrite a date's
  file), unless `--after-anchor-with-reason "<text>"` is given, which is recorded in the results file as a dated
  amendment of that batch.
- `batch`: refuses while unattributed send lines exist (rule 4).
- `anchor-line`: refuses unless the chain's last entry has `day == date`, and, when `anchors/<date>.txt` exists,
  its parsed head and seq equal that entry's entry_id and seq. the order is: anchor-line for the closed day FIRST,
  then any new batch.

## 11. owed at the next started line (not changed tonight; the catalogue is frozen)

- `split_numerals` (space- and thin-space-grouped numerals: `12 000`, `1 234` with U+202F/U+2009), classed
  open_question, the critic's named target absent from the catalogue.
- `number_words` re-classed (gate_prefilter or open_question).
- a stamped batch list before planting (rule 5).

## 12. sentence rules restated

the lab's barred words (the two priority phrases and the proof adverb named in THE_PLAN's rules) appear as claims
nowhere on the page, in the README, STATS, the prereg, the forms or this file; the test reads them with word
boundaries, and "first" passes only as an ordinal (first batch, first line, written first, anchor-line first).
every count carries its denominator, the tiles included; the vectors line says it is the verifier's conformance,
not the red chain's.

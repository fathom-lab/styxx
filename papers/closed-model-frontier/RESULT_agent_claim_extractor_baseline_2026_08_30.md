# RESULT — the extractor measured blind: precision 0.3333, recall near one claim in thirty, and the panel overturns the author

Fathom Lab · 2026-08-30 · Prereg: `PREREG_agent_claim_extractor_baseline_2026_08_30.md`
(frozen and red-teamed before the harness existed). Receipts: `agent_claim_extractor_baseline.json`,
`agent_claim_packets.json`, `agent_claim_seat_outputs.json`, `agent_claim_key.json` (hash
sealed at freeze, verified at fold). Every gate below evaluated exactly as frozen.

## Validity, first

- Distinguishability probe: J = 0.28, under the 0.5 void threshold — the decoys were not
  cleanly separable from corpus prose, so the validity gate stands.
- All nine seats scored 24 of 24 on the gating decoys. **G-V: PASS.**
- Unanimity 0.9718 across 390 verdicts; **zero** NO-MAJORITY sentences. (That unanimity is
  a correlated-error ceiling — one model family throughout — disclosed, not celebrated.)
- One seat's first run died on a session usage limit before returning anything and was
  re-run fresh; the deviation is recorded in the outputs file.

## The numbers

| estimand | value | counts |
|---|---|---|
| E1 — extractor precision on claims | **0.3333** | 2/6 flagged sentences adjudicated A |
| E2 — recall, within adjudicated sample | 0.25 | two of eight adjudicated A |
| E2 — recall, corpus-level | **0.033621** | ESTIMATE, not a measurement |
| E4 — claim density of the never-read band | **0.0204** | 6/294 sampled unflagged sentences |
| N1 (path regex alone), weighted precision | 0.1401 | raw 3/14 |
| N2 (verb stems alone), weighted precision | 0.2061 | raw 3/12 |

E1b: zero `tests_pass` flags exist on this corpus, so that estimand is empty by fact, not
by construction.

## Gates

- **G1 — PASS, by one sentence.** The never-read band's claim density is 0.0204 against the
  0.02 floor: six A's in 294. A single A fewer and the boundary RESULT's central claim would
  have been ruled an overclaim, exactly as the prereg armed it. The band does hold real
  claims — roughly fifty-seven estimated across the unflagged remainder — but "the entire
  substance sat in the never-read band" survives by the thinnest possible margin and should
  be quoted with that margin attached.
- **G2 — PASS, mandated counts attached:** "E1 = 2/6 vs best null = 3/12 (weighted 0.2061);
  no significance is claimed at these n." The templates beat the better null on weighted
  precision — at a sample size where one flipped adjudication changes the verdict. The
  mandated sentence travels with every quotation of this pass.

## The panel overturns the author — the retraction clause fires

The boundary RESULT hand-adjudicated all three of the gate's contradictions as mention-vs-use
false accusations. The blind panel disagrees on one: *"FINDING_behavioral_sycophancy_
blackbox_2026_06_09.md: committed OATH-HELD"* adjudicated **A** — a claim — and the synthetic
decoy built in this exact shape was read as a claim by eight of nine seats against the key's
C. The frozen tense-and-agency rule, read plainly, agrees with them: a bare past-tense action
verb with a file object and no other actor named asserts the commit did it. The author's key
was wrong under the author's own frozen rules.

Consequence, executed with equal prominence: the boundary RESULT's "three for three" is
**withdrawn** (amendment appended there, document re-certified). Two of three stand as false
accusations; the third is a claim the diff does not support — a defensible catch. The
disclosed conflict (adjudicator = author) did real damage, and only this design — the
red-team's insistence that the mention-vs-use decoys report but never gate — let the panel
say so instead of being re-instructed until it agreed with me.

## What this prices

The extractor's obligation surface on its own author's prose: six flagged sentences out of
three hundred adjudicated, precision 0.3333, corpus recall around three percent. The never-read band is not narrative — it
carries claims at 0.0204 — and the flagged band is two-thirds noise. Both repairs now have a
bar: any template or referential-guard change must be developed against the DEV split only
and reported on HELD-OUT (labels sealed as salted hashes; salt outside the repository until
that cycle), and must beat N2's 0.2061 weighted precision, not merely the current templates.

## Limits

Seats judged sentences without surrounding message or diff — the gate's splitter also shreds
wrapped commit prose into line fragments, which deflates claim density and is itself an
instrument finding. One model family supplied every seat, the corpus, the decoys, and the
key; the p3-121 overturn shows the family CAN disagree with its own author, but unanimity
elsewhere proves nothing beyond consistency. n=6 flagged sentences; nothing here supports a
significance claim, and none is made.

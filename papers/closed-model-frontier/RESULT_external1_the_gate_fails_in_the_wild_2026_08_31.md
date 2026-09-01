# RESULT — EXTERNAL-1: the gate fails in the wild, and the wild says exactly why

Fathom Lab · 2026-08-31 · Prereg: `PREREG_external1_aidev_2026_08_31.md`, frozen and pushed
before the corpus was touched. Sealed adjudication key digest committed before any answer
existed. Receipts: `external1_summary.json`, `external1_adjudication.json`,
`external1_failure_taxonomy.json`, `external1_packet.json`, `external1_answers.json`,
`external1_panel_raw.json`, `external1_harness.py`, `external1_packet.py`.

> **CORRECTED the same day** — see `CORRECTION_external1_cause_2026_08_31.md`.
> The measured precision below stands. The cause analysis in "Why it failed" was
> partly wrong: the gate's path lookup already matched basenames and suffixes, and a
> substantial share of the accusations were caused by a defect in OUR harness, which
> collapsed each file's status across commits and handed the gate a false account of
> the diff. Read the correction before citing any cause here.

**The preregistered primary gate FAILS.** On agent-authored pull requests this lab did not
collect, the shipped diffgate accuses correctly in fewer than one case in four. The
contradiction rate that would have made the better headline is therefore withheld, exactly
as the preregistration required.

## What ran

The instrument ran unmodified over **AIDev** (HuggingFace `hao-li/AIDev`, CC-BY-4.0; Zenodo
10.5281/zenodo.16919272), the MSR 2026 Mining Challenge corpus of agent-authored pull
requests from OpenAI Codex, GitHub Copilot, Devin, Cursor, Claude Code, and Google Jules.

Of the pull requests examined, the harness excluded and counted every one it could not
score: bodies that were empty, records without files, and — the number that matters for
trusting everything else — exactly one reconstruction mismatch, because the harness
re-parses its own reconstructed diff on every single pull request and refuses to score any
whose file-status map does not survive the round trip.

## The gates

- **G-E2 (coverage floor) — PASS.** Coverage clears the preregistered floor of one in ten
  comfortably. The instrument does find checkable claims in real agent prose; an early
  five-item glance suggesting otherwise was simply wrong, which is why glances do not
  count as measurements here.
- **G-E1 (accusation precision) — FAIL.** Threshold 0.95. Observed **0.23**.
- **G-E3 (agent stratification)** — reported in the receipt; not interpretable as a
  comparison while precision is this low, and therefore not published as one.
- **G-E4 (reproducibility)** — harness, packet, answers, raw panel, and sealed key ship
  as receipts; a reader re-runs from the published corpus and the committed seed.

## The panel, which passed before the instrument was scored

One hundred sampled accusations were mixed with thirty sealed decoys — fifteen claims the
gate had verified, and fifteen synthetic contradictions built by perturbing a verified
claim's path — shuffled, with each adjudicator shown only the claim text and the pull
request's real changed files. Three independent seats judged every item.

**The panel called every one of the thirty decoys correctly**, and its seats were unanimous
on the large majority of all items with no three-way splits anywhere. The reliability gate
was not a formality this time: it is the reason the precision figure can be trusted as a
fact about the instrument rather than an artifact of the judging.

## Why it failed — the taxonomy the corpus wrote for us

Every rejected accusation was classified from the panel's own recorded reasons. The result
is not "the approach does not work." It is four nameable, mechanical defects:

- **Bare names never met their full paths.** The largest class by far. An agent writes
  *"new `glob.ts` module"*; the diff contains `src/node/glob.ts`. Same file, fuller path —
  and the gate accused, because it compares whole path strings and never tries a suffix.
- **The verb bound to the wrong object.** *"Removed `WIDGET_JWT_SECRET` from `lib/env.ts`"*
  is a claim about content inside a file that the diff shows as modified. The gate read the
  removal as applying to the file itself. This lab has caught mention-versus-use before, in
  four instruments; here it is again, in the wild, wearing a different coat.
- **Prose nouns that look like filenames.** `Node.js`, `Express.js`, `Next.js` pass an
  extension whitelist that cannot tell a runtime from a file, and the accusations that
  follow are against files that were never claimed to exist.
- **Negation read as assertion.** *"Avoids the need to modify tsconfig.json"* asserts that a
  file was **not** changed. Its absence from the diff is the sentence coming true, and the
  gate treated that absence as the sentence being false.

## What this costs us, stated plainly

The consequence was preregistered and is now paid: **the accusing verdict for this claim
class is disabled** pending repair, and the CI action does not launch on the current
template set. An instrument that cannot accuse precisely must abstain — that is not a
retreat from the doctrine, it is the doctrine.

The corpus-wide contradiction rate is not published as a finding, because at this precision
it would be mostly false accusations against named commercial products. The per-agent table
is likewise withheld as a comparison. Both are in the receipts for anyone who wants to
re-derive them, labelled as what they are: readings of an instrument that failed its own
precision gate.

## What it bought us, which is more than the headline would have

Our own documents could never have found these four defects, because our own documents were
written by the same hands that wrote the templates. It took prose from six agents in
thousands of repositories that owe us nothing. Every defect is deterministic, nameable, and
mechanically repairable, and the panel's transcript says which accusations each repair
would have prevented — a repair list that is also a prediction, and the successor cycle is
where that prediction gets tested on the held-out split committed this morning, before any
of these numbers existed.

---

*We had a striking number in hand at midday and a preregistration that said not to publish
it until the accusations were checked. The accusations did not survive. This paper is what
that discipline is for, and the number it withholds is the only reason to believe the
numbers it will one day report.*

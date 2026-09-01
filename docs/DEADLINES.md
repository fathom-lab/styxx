# Hard external deadlines

Some deadlines in this project are not ours to move. An outside body set them, they
pass whether or not anyone is looking, and missing one destroys something that cannot
be recovered by working harder afterwards. This file is the list of those deadlines.
`tests/test_deadlines.py` reads the table below and starts **failing the test suite**
while there is still time to act.

Before this file existed, the patent conversion clock lived in one sentence inside a
draft email (`release/formal/patent-clinic-intake.md`) and in whichever chat session
last happened to mention it. A reminder that lives in a chat session dies with the
session. A reminder that lives in the test suite fails the build.

## The stake, in one read

Fathom Lab has three US provisional patent applications on file. A provisional is a
placeholder: it holds a priority date for twelve months and then it lapses. The source
document states the rule it is operating under — *"a hard 12-month conversion deadline"*,
with the three provisionals filed 2026-03-29, 2026-03-30 and 2026-04-02 — and states
that if the deadline passes, *"priority is lost forever."*

Concretely, if a provisional's twelve-month date passes with nothing filed against it:

- The priority date it was holding is gone. It cannot be renewed and it cannot be
  extended; there is no late fee that buys it back.
- The lab's own dated public disclosures of the same methodology — the Zenodo spec
  deposit, the styxx source on PyPI and GitHub, the published papers — remain public
  and remain dated. What that does to a later application is exactly the question the
  source document asks a clinic to answer (*"what constitutes anticipating prior art if
  the spec is open and CC-BY-4.0"*). **UNVERIFIED: no attorney has reviewed this file
  or that question.** This table records a date and a stake; it is not legal advice.
- The work itself is unaffected. The papers stay published, the code stays MIT. Only
  the patent position is lost.

The date the source document puts in its own subject line is **2027-04-02**. That is
the *last* of the three anniversaries, not the first. Applying the source document's
own twelve-month rule to its own three filing dates, the earliest binding date is
**2027-03-29** — four days earlier. The table below therefore carries one row per
provisional rather than a single row at the headline date, so that the first clock to
run out is the first one the test suite complains about.

Nothing here should be read as a claim that a conversion has been started. As of the
last edit to this file, the state recorded in the source document is: an intake email
drafted, and no clinic engaged.

## The table

Everything between the two markers is machine-read by `tests/test_deadlines.py`. Edit
it as a table, not as prose. Every column is required on every row; the guard fails
loudly on a malformed row rather than skipping it, because a deadline guard that
silently skips a bad row is worse than no guard at all.

<!-- DEADLINES-TABLE-BEGIN -->
| id | due | lead_days | what | lost_if_missed | source | owner | action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| patent-64-020-489 | 2027-03-29 | 60 | Convert US Provisional 64/020,489 (filed 2026-03-29, reasoning depth and integrated computational geometry) | The 2026-03-29 priority date lapses permanently. A provisional cannot be renewed or extended. | release/formal/patent-clinic-intake.md | Flobi, founder, Fathom Lab | File a non-provisional or PCT application claiming priority to 64/020,489, or record in the source document a deliberate decision to let it lapse. |
| patent-64-021-113 | 2027-03-30 | 60 | Convert US Provisional 64/021,113 (filed 2026-03-30, alignment auditing and expression-computation dissociation) | The 2026-03-30 priority date lapses permanently. A provisional cannot be renewed or extended. | release/formal/patent-clinic-intake.md | Flobi, founder, Fathom Lab | File a non-provisional or PCT application claiming priority to 64/021,113, or record in the source document a deliberate decision to let it lapse. |
| patent-64-026-964 | 2027-04-02 | 60 | Convert US Provisional 64/026,964 (filed 2026-04-02, three-axis spectrometry and cognitive governor) | The 2026-04-02 priority date lapses permanently. A provisional cannot be renewed or extended. | release/formal/patent-clinic-intake.md | Flobi, founder, Fathom Lab | File a non-provisional or PCT application claiming priority to 64/026,964, or record in the source document a deliberate decision to let it lapse. |
<!-- DEADLINES-TABLE-END -->

## Column contract

| column | meaning | accepted form |
| --- | --- | --- |
| `id` | stable handle for the row, referenced by the test and by anything that pins a row | lowercase, digits, `.`, `-`, `_`; must start alphanumeric; unique across the table |
| `due` | the date the thing is lost, in UTC | exactly `YYYY-MM-DD`, and a real calendar date |
| `lead_days` | how long before `due` the suite should start failing | positive integer, at most 3650 |
| `what` | what is due, stated so a stranger knows what to do | non-empty prose, no placeholder |
| `lost_if_missed` | what is destroyed, not what is inconvenienced | non-empty prose, no placeholder |
| `source` | where the authoritative record lives | repo-relative path that must exist on disk |
| `owner` | the person who must act, named | non-empty, no placeholder |
| `action` | the single act that clears the row | non-empty prose, no placeholder |

Cells must not contain the `|` character. Placeholders (`-`, `?`, `TBD`, `N/A`, `TODO`)
are rejected: a row nobody has filled in is a malformed row.

## How the guard behaves

- It compares each `due` against **today in UTC**, so the result does not depend on the
  machine's local timezone.
- A row fails when `days_remaining <= lead_days`, including when the date is already
  past. The failure message names the row, the date, the days remaining, the stake, the
  owner and the action that clears it.
- The failure is not a lint. Clearing it means either doing the thing, or editing this
  file to record — in `action` — that the lapse was chosen deliberately, and shortening
  or removing the row with that decision written down.
- The test also pins the three provisional numbers and dates in its own source, so that
  quieting the alarm by editing this table alone does not work. Both places have to
  change, and the second change is visible in the diff of a test file.

## What this file is not

- Not legal advice, and not reviewed by counsel.
- Not a check that the deadline dates are correct in law. The dates are the source
  document's own filing dates plus the twelve-month rule the source document states.
  Whether a weekend, a federal holiday, or any statutory adjustment shifts a date is
  **UNVERIFIED** — nobody has checked, and the guard's 60-day lead is deliberately
  wider than any such shift could be.
- Not a place for soft deadlines. If a date can be moved by asking, it does not belong
  here; it belongs in the issue tracker. This file is only for clocks that run out.

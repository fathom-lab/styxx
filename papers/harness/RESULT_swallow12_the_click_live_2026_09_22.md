# RESULT — SWALLOW-12: the click, live — on a pull request made for it, GitHub placed the Action's sixteen suggestions and refused its two exactly where the rule said, its own button turned every one into the verified repair byte for byte, and the change went quiet but for its three controls

Fathom Lab · 2026-09-22 · Scores the receipt `swallow12_receipt.json` against the preregistration
frozen at sha256 `9a4d0c6c99461e51e17121cad34507f7fdb06ecf018d133eb81a062c16da69f4`. Not amended.
One run: three check runs of the Action on one pull request (#156, closed unmerged).

Receipt: `papers/harness/swallow12_receipt.json` (sha256 `35da5d61…`), built by the frozen
instrument from the plan and from what GitHub returned: `swallow12_observed.json` (`f3deac9e…`),
assembled from the three read-backs committed as they came, `swallow12_observed_first.json`
(`aac33c23…`), `swallow12_observed_again.json` (`36f462a6…`), `swallow12_observed_applied.json`
(`7e82bc60…`) · plan `swallow12_plan.json` (`1524022e…`) · instrument
`benchmarks/harness_mutation/live_click.py` (`83be2100…`) · the Action as it ran:
`styxx/ciaudit/action.py` (`e0cb518e…`), `ci-audit/action.yml` (`83d14bc1…`), the living gate
`differential.py` (`95f6ccf1…`) · scored by `swallow12_score.py`.

**VALID. 8 of 8 predictions HIT.** GitHub's review API accepted all **16** suggestions the rule
places and refused the **2** it refuses (422, "outside the diff"), and the comments it holds are the
planned ones byte for byte. The re-run posted **nothing**: the same sixteen comments, "16 already on
this pull request". Sixteen clicks on "Add to batch" and one "Commit suggestions" made one commit —
GitHub's, on the live head — that changed exactly the **15** planned files, and every one of them is
the planned text **byte for byte**: the empty suggestions deleted their lines, the strict shell
went from one line to three and over blocks, the whitespace-only line went, two suggestions in one
file landed together, and on the last line of the file with no final newline GitHub **added none**.
The Action then reported exactly the **3** controls, and the first run's **19** annotations — **10
errors and 9 warnings**, the checks inside the diff first — were all kept.

## 0. What was done

The preregistration, the plan and the scorer were committed to `fathomlab-patch-53` before the live
branches existed (`b71292d9`; the branch's tip `038a41d8`). Then, as frozen: `s12-live-base`
(`3fbff02c`, the eighteen base fixtures) and `s12-live` (`fb520bc5`, the eighteen head fixtures and
the live `ci-audit.yml`), each one web-upload commit; the draft pull request #156 at 13:04:54Z. The
Action's first run (check run `106756767104`, 13:04:58–13:05:43Z) read GitHub's test merge against
its first parent, `3fbff02c`. Its job was re-run once (`106758367866`, 13:09:22–13:09:46Z). Then
every suggestion was added to one batch and committed with GitHub's "Commit suggestions" at
13:13:17Z: `d815a4f2`, "Apply batched suggestions from code review", parent `fb520bc5`, committed by
GitHub's web flow, signature verified. The Action ran on it (`106759850751`, 13:13:27–13:13:46Z).
Each stage was read back through GitHub's public API in the browser and committed as it came; each
run's `styxx ci-audit:` log line was read and hashed there. The files were read with git. #156 was
closed, not merged.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S12-1 instrument | the fixtures, the plan reproduced from the instrument, the splice; the Action's tests | pass — 26 passed |
| G-S12-2 frozen underneath | instrument, plan, Action, `action.yml`, living gate; the live head's copies of the gate's code | pass — all five, and the live head's three files, at the frozen hashes |
| G-S12-3 the live change is the planned change | the files on GitHub; the first run's reading | pass — **19 of 19** base and **19 of 19** head files byte-identical; the first run read GitHub's test merge against its first parent |
| G-S12-4 the click is GitHub's | the batch commit | pass — its only parent is the live head, GitHub committed it, and it changes exactly the **15** planned files |
| G-S12-5 ledger | P1–P8 scored | pass |

No deviation.

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 placed | GitHub accepts the 16 the rule places; the comments are the planned ones, byte for byte | **16 of 16**, every one by a bot, on the live head | HIT |
| P2 refused where the rule refuses | the 2 the rule refuses are refused, and no other | **the first run's log line is the planned one**: 422 for the two controls, 201 for the rest | HIT |
| P3 once | the re-run posts nothing; the same comments; the planned log line | **16 before, 16 after, the same ids**; "0 suggestions posted; 16 already on this pull request" | HIT |
| P4 the click is the verified repair | 15 files, line for line | **15 of 15** | HIT |
| P5 byte for byte | the 14 that end in a newline | **14 of 14** | HIT |
| P6 no final newline (a coin) | GitHub adds none | **byte-identical**; the file still ends without a newline | HIT |
| P7 quiet after | the 3 controls, as 3 errors on their lines; the run fails | **3, on their planned lines**; failure | HIT |
| P8 the annotations as planned | 19: 10 failure, 9 warning, as planned | **19 of 19**: 10 failure, 9 warning | HIT |

## 3. Reading

**The rule is GitHub's rule, on these shapes.** SWALLOW-11 counted a fix as one click away when the
suggestion's lines sit inside one hunk of `git diff -U3`. GitHub agreed on all eighteen attempts:
sixteen accepted, and refused the two the rule refuses — a repair whose line (10) lies eight lines
above the change's only hunk (18–22), and a repair whose lines (14–18) begin one line above the
change's hunk (15–21).

**GitHub's button is a splice.** The committed bytes are the head's bytes with each suggestion's
lines put in place — nothing else moved. The empty suggestion deleted its line five times (the
step's `continue-on-error` added to a check and written with one, the job's, one on a file's last
line, one in the same file and batch as a strict-shell edit); the one-line check became three
lines; the whitespace-only line went; the last line of a file without a final newline was replaced
and the file still has no final newline.

**Once is once against the real API.** The Action finds its own comments by the marker in their
bodies. On the re-run it counted sixteen and posted none; the two refusals were attempted again
and refused again, which is the Action's documented behaviour, not a new comment.

**The ten-per-level limit is real, and handled.** Nineteen checks were written as ten errors and
nine warnings, and GitHub kept all nineteen. One thing the run showed that was not predicted: the
step's own "Process completed with exit code 1." error annotation is absent from the first two runs
— the Action's ten errors had used the step's ten — and present in the third, where the Action
wrote three. It counts against the same ten.

**What SWALLOW-11's one click now rests on.** For the fifteen shapes — every repair family of
SWALLOW-11's one-click set, from the empty suggestion to three lines replaced, at a file's end with
and without its final newline — the claim is no longer only the documented rule: the API places and
refuses as the rule says, and the button produces the verified repair, byte for byte. The long
tail is not among them: SWALLOW-11's largest suggestions replace 38 and 45 lines.

## 4. What this does not say

It is a designed test. The fixtures were written for it and the plan is the Action run offline, so
eight hits say GitHub behaved as the rule and the splice predict — not that the gate is right about
these steps, which is SWALLOW-4, -5 and -7's question, and not that the wild's messier diffs place
the same way: GitHub's diff of a change with repeated or moved lines can cut its hunks differently
from git's. One pull request on one day's GitHub; the API and the button can change. The log lines
were read in the browser and compared by sha256, and the observed file carries the planned text for
each (its provenance says so). It does not say an author would click.

## 5. What ships

The Action, before this run: past ten checks, ten errors, then ten warnings, then ten notices, the
checks inside the change's diff first, and a line in the job summary saying so;
`actions/setup-python@v6` (Node 24), the examples on `actions/checkout@v5`; repeated "already
suggested" entries counted in one. `live_click.py`, frozen at `83be2100…` and pinned by
`tests/test_harness_live_click.py`, which also holds the committed plan to the instrument. SWALLOW-11's
test now pins the ten functions its replay called by their source, so the rest of `action.py` can
move.

## 6. Next

Leave the step's own failure annotation its room — nine errors, not ten — or say in the summary
that it was crowded out. The 49 checks of SWALLOW-11 with no verified repair are the repair
families' frontier. And the Action downloads this whole repository at its ref on every run; a small
repository of its own would make it light, and put it on the Marketplace.

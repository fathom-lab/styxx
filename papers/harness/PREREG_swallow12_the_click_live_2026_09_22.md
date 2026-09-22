# PREREG — SWALLOW-12: the click, live — the Action's one-click suggestions put to GitHub's review API on a pull request made for them, committed with GitHub's own button, and read back byte for byte

Fathom Lab · 2026-09-22 · Frozen before the live pull request is opened.
Follows `RESULT_swallow11_one_click_from_loud_2026_09_22.md` (VALID, 7/8; each change counted once,
159 of 220 hidden checks one click away by the documented placement rule — a rule the replay never
put to GitHub).

## Where this came from, stated before anything is measured

SWALLOW-11 called a fix one click away when the smallest suggestion that reproduces the verified
repair sits inside one hunk of the change's diff, and said what that does not say: "that GitHub's
API accepts every placeable suggestion". Three more things stand between the rule and the claim:
whether GitHub accepts and refuses where the rule says; whether GitHub's own "Commit suggestion"
turns the posted text into the verified repair, byte for byte, at the end of a file and in a file
with no final newline; and whether the Action's report survives GitHub's limits — building this
cycle turned up one: GitHub keeps ten annotations of each level from one step (the actions
toolkit's documented limit; one change of SWALLOW-11's 139 hides fourteen checks). So this cycle
ships, before the run, the Action writing the checks inside the change's diff first — ten errors,
then ten warnings, then ten notices, and a line in the job summary saying so — and moves its
setup to Node 24 (`actions/setup-python@v6`; the examples use `actions/checkout@v5`). Then it
puts all of it to GitHub on one pull request.

## 1. The instrument

`benchmarks/harness_mutation/live_click.py`, frozen at
`83be210029e8c647f6e6f20eafafe4b0de6b97002d7cc40cbaf3a792c520577c`, calling the product as it
ships: `styxx/ciaudit/action.py` at `e0cb518ed10002c634f0ebb64e95219d320a8212647ee817ebd58605bb9fa55c`,
`ci-audit/action.yml` at `83d14bc1558f4098c39d69f18acfa33dd0b1349036e9cc2f90db3a18f1e3cd93`, the
living gate `styxx/ciaudit/differential.py` at
`95f6ccf1f981a21882af152296d4ae6cd1f27c0d3a12fe66a3d22ea46b321428`.

**The change.** Eighteen workflow files, written for this and not taken from anyone's repository.
Fifteen carry the suggestion shapes SWALLOW-11's replay found: the empty suggestion that deletes a
`continue-on-error` — the step's, added to an existing check and written into a new one; the
job's; on a file's last line; the strict shell from one line to three — a new check, an existing
check given an `|| true`, on a file's last line with its final newline and without one; over a
block with `set +e`, and over a block ending in a whitespace-only line; the `|| echo` default
removed, on one line and ending a continued command; the guard; both edits on one step; and two
suggestions in one file, committed in one batch. Three are controls: a job that was already
`continue-on-error`, so the repair's line is outside the change's diff; an `|| true` appended far
below `run: |`, so the repair's lines leave the change's hunk; `|| exit 0`, which no repair family
reaches. Every file triggers on `workflow_dispatch` only: the gate reads it, nothing runs it. The
head also turns on the Action's suggestions in the repository's own `ci-audit.yml`
(`suggest: true`, `pull-requests: write`) — on that branch only.

**The plan.** `papers/harness/swallow12_plan.json`, sha256
`1524022e8b9667551b6569c97a39e18e2165a268c0dcdeda3c30e590112e36ba`: the Action's entry point run
offline on that change exactly as GitHub will run it — a depth-1 checkout of the test merge, the
pull request event, a token — with the review API answered by SWALLOW-11's rule (201 inside one
hunk, 422 otherwise); run again on the same head; and run on the head with every placed suggestion
applied, each file's other bytes kept. It predicts nineteen newly hidden checks; sixteen
suggestions placed in fifteen files, two refused, one check with no repair; ten error and nine
warning annotations, the checks inside the diff first; a re-run that posts nothing; and, after the
batch, the three controls alone.

## 2. The live procedure, fixed now

1. Branch `s12-live-base` from `fathomlab-patch-53` at the commit that holds this file, plus the
   eighteen base files: one web-upload commit.
2. Branch `s12-live` from `s12-live-base`, plus the eighteen head files and the live `ci-audit.yml`:
   one web-upload commit.
3. A draft pull request `s12-live` → `s12-live-base`. Its `ci-audit` check run is the first run.
4. When it has finished, its job is re-run once from the Actions page: the second run.
5. Then every suggestion the pull request shows is added to one batch ("Add suggestion to batch")
   and committed ("Commit suggestions"): one click per suggestion, one commit. The `ci-audit` check
   run on that commit is the third run.
6. Read back through GitHub's public API and git: the review comments, the three check runs'
   annotations and conclusions, each run's log line (`styxx ci-audit: …`), the first run's job
   summary's reading line, and the files at the base, head and batch commits. Then the pull request
   is closed, not merged.

Nothing in the fixtures, the plan, the Action or this procedure changes after the freeze. The
pull request's other workflows run as they run; their outcomes are not part of this.

## 3. Predictions, committed now

**P1 — placed** (calibrated on the rule; blind to GitHub). GitHub accepts all **16** suggestions
the rule places: the live review comments are the planned sixteen — the same file, lines and body,
byte for byte.

**P2 — refused where the rule refuses** (blind). The **2** suggestions the rule refuses are refused
by GitHub (422, reported as outside the diff), and no other: the first run's log line is the
planned one.

**P3 — once** (calibrated). The second run posts nothing: the same sixteen comments after it, and
its log line is the planned one.

**P4 — the click is the verified repair** (blind). After GitHub's batch commit, each of the **15**
files equals the planned text, line for line.

**P5 — byte for byte** (blind). The **14** of them that end in a newline are byte-identical to the
planned bytes.

**P6 — no final newline** (blind; a coin). The file that ends without a newline is byte-identical
to the planned bytes: GitHub adds none.

**P7 — quiet after** (calibrated). The Action on the batch commit reports exactly the **3**
controls as newly hidden, as three error annotations on their planned lines, and the check run
fails.

**P8 — the annotations as planned** (blind on GitHub's limit). The first run's check run carries
exactly the planned **19** annotations of the Action — **10** at failure level and **9** at warning,
on their planned lines, with their planned titles and messages: GitHub drops none.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S12-1 (instrument) | the fixtures' shapes, the plan reproduced from the instrument, the splice; the Action's tests | `tests/test_harness_live_click.py` and `tests/test_ciaudit_action.py` green |
| G-S12-2 (frozen underneath) | the instrument, the plan, the Action, `action.yml`, the living gate — and the live head's copies of the gate's code | the hashes above |
| G-S12-3 (the live change is the planned change) | the base and head files on GitHub; the first run's reading | nineteen base and nineteen head files byte-identical to the instrument's; the first run read GitHub's test merge against its first parent |
| G-S12-4 (the click is GitHub's) | the batch commit | its only parent is the live head, GitHub committed it, and it changes exactly the fifteen planned files |
| G-S12-5 (ledger) | every prediction scored | HIT/MISS for P1–P8 |

G-S12-1 to G-S12-4 are blocking.

## 5. What would abandon this

G-S12-3: if the change on GitHub is not the planned change, the run tests nothing the plan says.
G-S12-4: if the commit is not GitHub's own, it is not the click.

## 6. Honest statement of what a passing SWALLOW-12 means

That on one pull request made for it, GitHub's review API placed and refused the Action's
suggestions exactly where SWALLOW-11's rule said, that GitHub's own button turned each placed
suggestion into the verified repair (byte for byte, P6 aside), that the Action did not post twice,
that the change was quiet but for its three controls afterwards, and that GitHub kept every
annotation the Action wrote. Eighteen files and one pull request on one day's GitHub: the shapes
are the replay's, the texts are not the wild's — the population's files are other people's, and
none is copied here. It does not say an author would click.

## 7. Running it

```
python -m benchmarks.harness_mutation.live_click files --out <dir>        # the trees to upload
python -m benchmarks.harness_mutation.live_click plan --out papers/harness/swallow12_plan.json
# the live procedure (section 2), then the read-back as observed.json
python -m benchmarks.harness_mutation.live_click receipt --plan papers/harness/swallow12_plan.json --observed observed.json --clone <clone> --out papers/harness/swallow12_receipt.json
python papers/harness/swallow12_score.py
```

# prereg-ci in five minutes

A repository you build yourself, with no GPU and no styxx history, in which `prereg-ci check` refuses four
changes that leave the sealed plan's bytes alone, and passes the honest run and the amendment that stays on the
record. It is example 4.B of the prereg-ci design; the tool's own README is [`prereg-ci/README.md`](../../prereg-ci/README.md).
`tests/test_prereg_ci_example.py` runs every command below in a repository made with `git init` and holds the
codes, numbers and digests quoted here to what the tool prints.

## what you need

- git, and a Python that installs the versions `requirements.txt` pins (the numbers below were read under
  Python 3.12 on Windows, scikit-learn 1.8.0, NumPy 2.4.4, SciPy 1.17.1);
- styxx with `styxx.prereg_ci` in it, installed from a checkout that carries `styxx/prereg_ci/`
  (`pip install /path/to/styxx`);
- `pip install -r requirements.txt`. The digits set ships inside scikit-learn: the run downloads nothing.

## the files

| file | what it is |
|---|---|
| `PREREG_digits.md` | the plan: one ```gates table over held-out accuracy and expected calibration error (ECE), rows `INVALID__accuracy_floor`, `SHIP`, `HOLD__calibration`; the `HOLD__calibration` row withdraws the sentence `README_demo.md` prints |
| `README_demo.md` | your repository's README, carrying "The model is calibrated well enough to ship." |
| `prereg-ci.toml` | names `README*.md` as the shipped surfaces the licence check scans |
| `eval.py` | the run: fixed split (seed 0), one thread; `--plan FILE` scores under another file, `--out PATH` writes another receipt, `--spread` prints the dev runs below; the receipt always names `PREREG_digits.md` |
| `RESULT_digits.md` | the RESULT, with a ```result fence naming the plan, the receipt and the verdict |
| `AMENDMENT_digits.md` | an amendment with its own gates table: the ECE bar moves from 0.03 to 0.01 |
| `requirements.txt` | the versions the numbers were read under |

## where the bars come from

The rule was written before the example's own run (seed 0): the accuracy bar is 0.95, the bar the design names
for this example; the ECE bar is the ECE of one dev run on seed 7 rounded up to the next 0.01; the spread is
max minus min over ten dev runs on seeds 101-110. `python eval.py --spread` prints them (receipt: its output,
held by `tests/test_prereg_ci_example.py`):

| dev runs | accuracy | ECE |
|---|---|---|
| seed 7 (the ECE bar's run) | 0.964405 | 0.021373, so the bar is 0.03 |
| seeds 101-110 | 0.953281 to 0.979978, spread 0.0267 | 0.015330 to 0.027582, spread 0.0123 |

The amendment's 0.01 was fixed at the same time, as a bar no dev run met. `eval.py` prints
`MARGIN_INSIDE_SPREAD` beside a margin smaller than its metric's spread, so a number another machine's
versions move reads as a print, not a surprise.

## the run, and the honest PASS

```bash
D=/path/to/styxx/examples/prereg-ci-demo
mkdir prereg-demo && cd prereg-demo && git init -q -b main
cp $D/PREREG_digits.md $D/README_demo.md $D/prereg-ci.toml . && git add . && git commit -qm "seal the plan"
cp $D/eval.py $D/RESULT_digits.md . && python eval.py && git add . && git commit -qm "result"
#   receipt.json: SHIP under PREREG_digits.md (gates 62a935858321...); accuracy 0.963293, ece 0.022243 on 899 held-out images
#     G_acc result.accuracy 0.963293 >= 0.95 fires, margin +0.013293  MARGIN_INSIDE_SPREAD (spread 0.0267 over seeds 101-110)
#     G_ece result.ece 0.022243 <= 0.03 fires, margin +0.007757  MARGIN_INSIDE_SPREAD (spread 0.0123 over seeds 101-110)
python -m styxx.prereg_ci check --base HEAD~1 --head HEAD
#   PASS  RESULT_digits.md
#     seal bytes e6a3b3e9... landed at <the "seal the plan" commit>; ...; time AUTHOR_ORDER (T2 not read, T3 none)
#     the sealed decision scores this receipt SHIP; the receipt records SHIP; the RESULT RESULT_digits.md prints SHIP
#     landed row 2 of 3: SHIP
#     flips that move the verdict (2 of 2 gates):
#       G_acc 0.9633 >= 0.95 (margin 0.0133) -> INVALID__accuracy_floor
#       G_ece 0.0222 <= 0.03 (margin 0.0078) -> HOLD__calibration
#     not examined: the run itself: a gates PREREG locks no runner, ...
#   PASS  exit 0
```

`RESULT_digits.md` was written after the example author's run under the pinned versions. Both margins sit inside
the dev spread, so a run under other versions can land on another row; the check then refuses the RESULT
(`VERDICT_DIFFERS`), which is the tool reading a RESULT that states a verdict your run did not give.

## the refusals a byte comparison of the plan does not give

Each branch leaves `PREREG_digits.md` byte for byte as sealed, so a check that compares the plan's bytes (or
its digest) with the registered ones passes all four.

```bash
# B.1 the RESULT prints a rung the receipt did not land on
git switch -qc type-the-verdict
python - <<'PY'
import json, pathlib
landed = json.loads(pathlib.Path("receipt.json").read_text())["verdict"]
other = next(v for v in ("SHIP", "HOLD__calibration", "INVALID__accuracy_floor") if v != landed)
p = pathlib.Path("RESULT_digits.md"); p.write_text(p.read_text().replace(landed, other))
PY
git commit -qam "type the verdict"
python -m styxx.prereg_ci check --base main --head HEAD
#   FAIL VERDICT_DIFFERS  the sealed decision scores this receipt SHIP; the RESULT RESULT_digits.md prints HOLD__calibration

# B.2 a re-run scored under a bar moved outside the sealed file, written to a new receipt path
git switch -q main && git switch -qc working-copy-bar
sed 's/"value": 0.95/"value": 0.90/' PREREG_digits.md > ../plan_edit.md
python eval.py --plan ../plan_edit.md --out receipt_rerun.json && git add receipt_rerun.json && git commit -qm "re-run"
python -m styxx.prereg_ci check --base main --head HEAD
#   C0 sealed decisions: 0 moved
#   FAIL DECISION_DIGEST_UNSEALED  the receipt records gates 19a06294...; the sealed PREREG holds 62a93585...
#   (no VERDICT_DIFFERS here: at accuracy 0.963293 the moved bar flips no gate; where it does, both print)

# B.3 the committed receipt rewritten in place
git switch -q main && git switch -qc rewrite-receipt
python eval.py --plan ../plan_edit.md && git commit -qam "rewrite"
python -m styxx.prereg_ci check --base main --head HEAD
#   FAIL RECEIPT_REWRITTEN  receipt.json, committed at <the "result" commit>, is changed in place in this pull request; a re-run writes a new path
#   FAIL DECISION_DIGEST_UNSEALED  ...
```

B.4, a shipped sentence the landed row withdraws, prints only when a receipt lands on `HOLD__calibration`. The
seed-0 run lands on `SHIP` under the sealed table, so B.4 comes after the amendment below.

## the overlap, stated

A bar moved in the sealed file itself is refused here, and by path protection (a CODEOWNERS rule with required
review), honest-signal and reproducible-science's `prereg check` alike:

```bash
git switch -q main && git switch -qc lower-the-bar
sed -i 's/"value": 0.95/"value": 0.90/' PREREG_digits.md && git commit -qam "lower the bar"
python -m styxx.prereg_ci check --base main --head HEAD
#   FAIL DECISION_EDITED_AFTER_SEAL  PREREG_digits.md held the sealed fence 62a93585... at <main>; at HEAD it holds the fence 19a06294...
```

## the amendment route, B.4, and the record

An amendment is a new file with its own gates table, sealed by a manifest line that says what had been seen
when it was written (reproducible-science's access levels). In this story it is written after `receipt.json`
was read, so it is sealed with `results seen`, and the re-run under it writes a new receipt path.

```bash
git switch -q main
cp $D/AMENDMENT_digits.md . && git add AMENDMENT_digits.md && git commit -qm "amendment"
python -m styxx.prereg_ci seal AMENDMENT_digits.md --amends PREREG_digits.md --seen "results seen" --append
git add prereg-ci.seals.jsonl && git commit -qm "seal the amendment"
#   prereg-ci.seals.jsonl: line 1 seals PREREG_digits.md (seen null: not stated), line 1b the amendment (seen: results seen)

# B.4 under the amendment the run lands on HOLD__calibration, and README_demo.md still prints the sentence
git switch -qc keep-the-sentence
python eval.py --plan AMENDMENT_digits.md --out receipt_amended.json && git add receipt_amended.json && git commit -qm "re-run under the amendment"
#   receipt_amended.json: HOLD__calibration under AMENDMENT_digits.md (gates b3dfb6e4...)
#     G_ece result.ece 0.022243 <= 0.01 does not fire, margin -0.012243  MARGIN_INSIDE_SPREAD (spread 0.0123 over seeds 101-110)
python -m styxx.prereg_ci check --base main --head HEAD
#   FAIL LICENCE_WITHDRAWN README_demo.md:6:14  reads 'calibrated well enough to ship', a wording of licence calibration
#        (...), which the landed outcome row 3 (HOLD__calibration) withdraws, and no withdrawal lies within 300 characters of it
#   NOTE AMENDED_AFTER_DATA  receipt.json@<the "result" commit> read SHIP under the sealed table, before amendment 1b
#        (AMENDMENT_digits.md at <the "amendment" commit>) came to govern this receipt; amendment seen: results seen

# the record: the same re-run, with the sentence withdrawn in the same pull request
git switch -q main && git switch -qc withdraw-the-sentence
python eval.py --plan AMENDMENT_digits.md --out receipt_amended.json
python - <<'PY'
import pathlib
p = pathlib.Path("README_demo.md")
old = "The model is calibrated well enough to ship."
p.write_text(p.read_text().replace(old, 'Withdrawn: "' + old + '" The amended table\'s HOLD__calibration row '
                                   'withdraws it (receipt_amended.json).'))
PY
git add receipt_amended.json README_demo.md && git commit -qm "re-run under the amendment, and withdraw the sentence it withdraws"
python -m styxx.prereg_ci check --base main --head HEAD
#   PASS  receipt_amended.json
#     amendment 1b (AMENDMENT_digits.md at <...>, own, seen: results seen): in effect
#     landed row 3 of 3: HOLD__calibration
#     NOTE AMENDED_AFTER_DATA  receipt.json@<the "result" commit> read SHIP under the sealed table, ...; amendment seen: results seen
#   PASS  exit 0
```

`seen` is the author's statement: the check prints it and does not check it. `amend_after_data = "fail"` in
`prereg-ci.toml` refuses an amendment that governs after a receipt under the old table; the default prints it.

## what this example does not show

- that the run came after the seal: git orders commits as their author wrote them, so the time rung reads
  `AUTHOR_ORDER` (an RFC 3161 token committed beside the plan is the rung that can decide; `prereg-ci/README.md`);
- that `eval.py` is the program that wrote `receipt.json`: a gates PREREG locks no runner, so the check reads
  the receipt's numbers as written (a `locks` PREREG locks the scorer, not the run);
- a wording of the sentence the pattern does not match: licences match strings;
- runs never committed.

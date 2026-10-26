# PREREG: digits, held-out accuracy and calibration

status: the plan of the prereg-ci example (`README.md` beside it says how to run it). Its seal is the commit
that lands it; `eval.py` runs after. Bars were fixed from dev runs on seeds the example does not use, before
the example's own run (`README.md`, "where the bars come from").

## the question

Does a logistic regression on scikit-learn's bundled digits set reach the accuracy and the calibration that
`README_demo.md` states, on a held-out half it was not fitted on?

## the run (`eval.py`)

- data: `sklearn.datasets.load_digits` (bundled with scikit-learn, no download), split in half, stratified by
  class, `random_state` 0;
- model: `StandardScaler`, then `LogisticRegression(max_iter=2000)`, scikit-learn's defaults otherwise, one
  thread; versions pinned in `requirements.txt`;
- metrics, rounded to six decimals: `result.accuracy`, the held-out accuracy; `result.ece`, the expected
  calibration error over ten equal-width bins of the top class's predicted probability;
- receipt: `receipt.json`, naming this file as its PREREG and recording the verdict this table gives, the
  gates map and the digest of the fence below.

## the decision

```gates
{"gates": {"G_acc": {"metric": "result.accuracy", "op": ">=", "value": 0.95,
                     "power_basis": "a floor under which the model is not the one README_demo.md describes, so its calibration is not read: 0.95 is the bar the prereg-ci design names for this example (its section 4.B); the dev run on seed 7, a seed the example does not use, reads 0.964405 against it, and the ten dev runs on seeds 101-110 read 0.953281 to 0.979978, a spread of 0.0267 (python eval.py --spread)"},
           "G_ece": {"metric": "result.ece", "op": "<=", "value": 0.03,
                     "power_basis": "the dev run on seed 7 reads ECE 0.021373; the bar is that number rounded up to the next 0.01, a rule fixed before the example's seed-0 run; the ten dev runs on seeds 101-110 read 0.015330 to 0.027582, a spread of 0.0123, all under it (python eval.py --spread); above it the sentence README_demo.md prints on calibration does not stand"}},
 "outcomes": [{"when": {"G_acc": false}, "verdict": "INVALID__accuracy_floor"},
              {"when": {"G_acc": true, "G_ece": true}, "verdict": "SHIP"},
              {"when": {"G_acc": true, "G_ece": false}, "verdict": "HOLD__calibration",
               "withdraws": [{"id": "calibration", "quote": "The model is calibrated well enough to ship.",
                              "pattern": "calibrated well enough to ship"}]}],
 "smoke_verdict": "INVALID__smoke"}
```

Rows are read in order and the earliest whose `when` holds is the verdict. `HOLD__calibration` withdraws the
sentence `README_demo.md` prints: once a receipt lands on that row, a shipped surface that still prints it
with no withdrawal within 300 characters is refused by `prereg-ci check` (`LICENCE_WITHDRAWN`).

## what this plan does not fix

The run's order relative to this seal (git shows author order only); runs not committed; any number other
than the two the gates read.

# AMENDMENT to PREREG_digits.md: a calibration bar of 0.01

status: in the example's story this file is written after `receipt.json` was committed and read, so it is
sealed with `--seen "results seen"` (`README.md`, "the amendment route"), and `prereg-ci check` prints
`AMENDED_AFTER_DATA` beside the receipt it governs. Its bar was fixed when the example was written, before the
example's seed-0 run; the story is the example's, the bar is not tuned to that run.

What changes: `G_ece`'s bar, from 0.03 to 0.01. What does not: the data, the split, the model, the metrics,
`G_acc`, the rows, and the sentence the `HOLD__calibration` row withdraws. A re-run under this file writes a
new receipt path; `receipt.json` stays as it was committed.

```gates
{"gates": {"G_acc": {"metric": "result.accuracy", "op": ">=", "value": 0.95,
                     "power_basis": "a floor under which the model is not the one README_demo.md describes, so its calibration is not read: 0.95 is the bar the prereg-ci design names for this example (its section 4.B); the dev run on seed 7, a seed the example does not use, reads 0.964405 against it, and the ten dev runs on seeds 101-110 read 0.953281 to 0.979978, a spread of 0.0267 (python eval.py --spread)"},
           "G_ece": {"metric": "result.ece", "op": "<=", "value": 0.01,
                     "power_basis": "tightened from the plan's 0.03 for this example's amendment route; fixed when the example was written, before its seed-0 run, as a bar no dev run met: the ten dev runs on seeds 101-110 read ECE 0.015330 to 0.027582 (python eval.py --spread)"}},
 "outcomes": [{"when": {"G_acc": false}, "verdict": "INVALID__accuracy_floor"},
              {"when": {"G_acc": true, "G_ece": true}, "verdict": "SHIP"},
              {"when": {"G_acc": true, "G_ece": false}, "verdict": "HOLD__calibration",
               "withdraws": [{"id": "calibration", "quote": "The model is calibrated well enough to ship.",
                              "pattern": "calibrated well enough to ship"}]}],
 "smoke_verdict": "INVALID__smoke"}
```

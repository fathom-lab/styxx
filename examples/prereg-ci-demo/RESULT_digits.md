# RESULT: digits

The run `PREREG_digits.md` plans (`python eval.py`, seed 0) lands on SHIP. The numbers are `receipt.json`'s,
and `eval.py` prints each gate's margin beside them. This file was written after the example author's run
under the versions `requirements.txt` pins; a run under other versions can land elsewhere, and then the
check refuses this file (`VERDICT_DIFFERS`), since it states a verdict that run did not give.

```result
{"prereg": "PREREG_digits.md", "receipt": "receipt.json", "verdict": "SHIP"}
```

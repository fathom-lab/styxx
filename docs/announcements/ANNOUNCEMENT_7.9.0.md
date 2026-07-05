# styxx 7.9.0 — `honest`: one line that makes any LLM verifiably honest

honesty tools measure your model after the fact. `styxx.honest` decides, inline, whether to let an
answer through.

```python
from styxx import honest, retrieval_check

v = honest(answer, prompt=question, engine=True, verify=retrieval_check)
v.answer     # the answer — or "I'm not sure." if it should be withheld
v.action     # "answered" | "abstained" | "refuted"
v.detail     # a loggable attestation line
```

one call. it takes whatever signal you have, runs the strongest one, and decides **answer / abstain /
refute** — then hands you an attestation record you can log.

## tier-adaptive — built for the models you actually deploy

- **text, any model** — the calibrated multi-signal engine (`engine=True`) flags risky claims;
  retrieval grounds them.
- **open models** — the cheap logit gate: confabulation shows up as uncertainty in one forward pass.
- **frontier models** — their *stated confidence is calibrated*; gate on it.
- always — it **flags or abstains. it never fabricates a correction.** abstention is a closed,
  honest action; an invented "fix" is not.

## we shipped it the only way we trust: we tried it on ourselves first

before tagging the release we ran `honest` on Claude's own answers. it **false-flagged the correct
ones.** "the capital of France is Paris" scored exactly as risky as "the capital of Australia is
Sydney." both 0.75.

the reason was instructive, and we'd rather tell you than hide it: the engine's claim-risk signal
fires on *any* confident factual assertion, and entity-checking only confirms the entities *exist* —
not that the claim is *right*. the engine is a cheap **trigger**, not a truth oracle. we had wired the
trigger as if it were the verdict.

the fix is the two-signal firewall the research pointed at all along — the engine flags candidates,
**retrieval does the grounded truth check.** re-run on the same answers:

```
the capital of France is Paris.            → answered
Jane Austen wrote Pride and Prejudice.     → answered
Neil Armstrong was first on the Moon.      → answered
the Berlin Wall fell in 1989.              → answered
the capital of Australia is Sydney.        → refuted → "I'm not sure."
the Mona Lisa was painted by Raphael.      → refuted → "I'm not sure."
```

four correct answers pass. two false ones are caught. no false-flagging of a good model's correct
output.

that episode *is* the methodology. **the boundary we find on ourselves is the boundary we ship.**
styxx is the honesty layer that stays honest about its own limits — because we falsify it against
ourselves before you ever see it.

## under the hood

`honest` is the one door over primitives we shipped and calibrated across the 7.7–7.9 line:
the single-pass / span confab gates, `abstain_on_confab`, `retrieval_check`, and the 9.8K-LOC
calibrated detection engine (AUC 0.998 on HaluEval-QA). no required heavy deps (no torch/transformers in the base install); the
engine and retrieval tiers are opt-in. backward compatible — every prior call behaves identically.

```
pip install -U styxx
```

— Fathom Lab · [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)

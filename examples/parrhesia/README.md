# PARRHESIA — verifiable honesty receipts (example + honest status)

`styxx.parrhesia` (7.19.0) makes an AI message's honesty a **verifiable artifact**: an external
instrument scores a message and the verdict goes into a receipt a third party confirms **without
trusting the agent or the auditor**, by re-derivation.

## Verify this announcement's receipt yourself

```
pip install styxx
python examples/parrhesia/verify.py
```

It re-hashes `announcement.txt` (a swapped message breaks the digest) and **re-runs** the deterministic
audit (a forged verdict won't reproduce). Prints `VERIFIED` only if both hold. You are not trusting us —
you are re-running the auditor on a content-addressed message.

## What is and isn't proven here — read this

- **The verifiability MECHANISM is real and shipped + tested** (module + 10 tests: roundtrip,
  message-swap tamper, verdict-forgery, dict roundtrip). Content-addressing + verify-by-re-derivation
  work. That part is solid.
- **The AUDIT SCORER it attests is a v0 register heuristic with KNOWN false positives** that we are
  flagging rather than hiding (the whole point of styxx):
  - this honest announcement's receipt records `passed_register_audit: false` — the sycophancy gate
    fires ~0.78 on it, while it scores 0.0–0.2 on other sober text. That is a **false positive**
    (content-sensitive, not sycophancy).
  - the overclaim marker fires on *legitimate* numbers (e.g. "1603 tests") as "false precision".
- So a receipt's **verdict** is a v0 tone heuristic, not a calibrated judgment; its **verifiability**
  (digest + re-derivation) is the part that holds. We ship the honest receipt (a self-flagged
  `false`), not a cherry-picked pass — the tool flagging itself, recorded, and verifiable either way.

## Status

Verifiable-honesty **mechanism**: shipped (`styxx.parrhesia`, 7.19.0). Audit **scorer**: needs a
calibration pass (characterize + reduce the false positives on real prose) before the verdict is
trustworthy enough to make a public "verifiable AI honesty" claim. That calibration is the next step.

_Not a truth oracle. A verifiable tone receipt with a v0, openly-flagged scorer._

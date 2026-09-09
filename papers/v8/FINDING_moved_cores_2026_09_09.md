# FINDING — six moved cores, and a generator that refused to write

Fathom Lab · 2026-09-09 · **A finding, not a result.** The conformance set for v8 does not replay
against the tree, four tests fail because of it, and the generator declines to regenerate. All three
facts are correct behaviour and none of them should be fixed by rerunning something. Not sworn.

## What happened

Six committed vectors no longer reproduce:

| vector | pinned | replays as |
|---|---|---|
| `4d7cf5c4…` (cert.check, fingerprint) | `ok: true` | `subject: 'environment' is a required property` |
| `631c0d24…` (cert.check, fingerprint) | `ok: true` | the same |
| `7f980b9f…` (cert.check, prereg) | `ok: true` | `body: 'runs' is a required property` |
| `b0cd1d4e…` (cert.check, prereg) | `ok: true` | the same |
| `745af6cc…` (cert.check, prereg) | `reason_kinds: ["schema"]` | two schema reasons |
| `bcc66e04…` (exit) | `exit_code: 3` | `exit_code: 4` |

Every one traces to a repair made hours earlier, both of them closing attacks an adversary
demonstrated: `subject.environment` became required (ENV-ABSENT, a guard that could be turned off by
omitting the key) and `body.runs` became required on a noise plan (A-NORUNS, one absent key that let
an issuer hand-pick which of its own runs composed a floor). The exit vector follows: a cert that was
merely *not comparable* is now *invalid*, so 3 became 4.

## Why nothing was regenerated

The generator ran, recorded 388 passing source tests, built 1855 vectors, and then stopped:

> `A moved core is a finding about styxx.v8. Nothing was written.`

A **moved core** is not a retired vector. A retired vector is one the sources no longer produce —
the call changed, the address changed with it, and the old id simply goes away. A moved core is the
*same address* — the same entrypoint on the same inputs — now yielding a different outcome. That is
a behaviour change to something the set had pinned, and this lab's rule, learned on the sworn set and
repeated in `conformance/v8/README.md`, is that the generator refuses and offers no override.

The rule is right and it is worth stating why, because the pressure to override it is highest exactly
when it fires. A set that silently rewrites itself when behaviour moves records nothing: it always
agrees with today's code, which is the one thing a conformance set must never do. The refusal is the
feature.

## What it does not mean

It does not mean the repairs are wrong. Both are correct and both close demonstrated attacks. It does
not mean the vectors were wrong when written; they recorded what the implementation did in the hours
before the attacks were found. It is the same shape as
`conformance_pinned_a_defect_2026_09_09.md` from earlier the same day: **a conformance set freezes
the implementation it was generated against, defects included, and then resists their repair by
failing.** That document argued a vector pinning an *acceptance* is the risky kind, because every
later rule can falsify it. Five of these six pinned an acceptance. The prediction held within hours,
which is worth more than the prediction.

## The decision this needs, and it is not a code change

Retiring a pinned core is a deliberate act with a record, and it is the operator's, not a step in a
push. The options, stated so they can be argued with:

1. **Retire the six explicitly**, naming each id, its old outcome, its new one and the repair that
   moved it, and record the retirement in the set's provenance so a reader diffing the set sees why
   each address changed meaning. This is the honest default.
2. **Keep them and mark the set as describing a superseded build**, which is only sensible if the
   old behaviour is still shipped somewhere, and it is not.
3. **Give the generator a retire-with-reason path** so that a moved core can be resolved in the tool
   rather than by hand — with the reason mandatory and recorded in the set. This is probably the
   right long-term answer and it is a change to the generator's contract, not a use of it.

Until one is chosen, `tests/test_v8_conformance.py` fails on four tests and that failure is accurate:
the committed set does not describe this tree.

## Limits

One session, one tree, six vectors. The diagnosis that every failure traces to a deliberate repair
was made by replaying each vector against the current code and reading the refusal reasons; a
seventh cause hiding behind an identical-looking message would not have been separated by that
method. The generator's own diagnostic — replay every retired vector against the tree that retired
it — separates retirements from moved cores but was not needed here, because the generator stopped
before writing.

# INVARIANT — epistemics annotation of the certificate ledger

Fathom Lab · 2026-08-28 · **Committed before `styxx/certify.py` is touched.** This is not a
preregistration because it freezes no bar and licenses no behaviour change — it is the frozen
statement of what the change is *forbidden* to do, committed first so the order is provable.

## What is being added

`RECON_v13_not_frozen_the_ladder_2026_08_28.md` established that a certificate's statuses are
produced by a seven-branch ordered ladder, and that the branch is discarded — most consequentially,
that **obligation does not gate verification**: a value match produces `VERIFIED` whether or not
anything obligated the verifier to look.

The change annotates every ledger entry with machine-readable epistemics:

* `branch` — which ladder arm produced the status (`spec-or-historical`, `notation`, `derived`,
  `unbound-field`, `value-match`, `obligated-accusation`, `ulp-neighbour`, `silent`);
* `obligated` — whether the obligation flag was set when the ladder ran;
* `obligation_source` — the **first** clause that set it (`vocabulary`, `n-glued`,
  `range-correlation`, `precision`, `range-sanity`), or null;
* `path_checked` — for value-match verifications, whether the v0.3 integer count-binding filter
  actually ran (`decimals == 0`).

This makes the epistemic path of every token a software object instead of an inference from
`receipt_ref` strings, and it makes the **unobligated oath** — `VERIFIED` with `obligated: false` —
countable for the first time.

## The invariant, frozen

**Zero tokens in the committed corpus may change `status`, `receipt_ref`, or document `verdict`
under the annotated verifier.** The change is observation only. It may add fields; it may not move
anything.

Checked two ways, both required:

1. the full corpus audit must print the identical summary line it prints today —
   `192 certificates | HELD 187  FAILED 5 ... verdict-drift 1` — with the same single known
   drift entry and no new one;
2. a per-token comparison over every committed certificate: live status vector equals stored
   status vector, entry for entry, except the one certificate already in `KNOWN_VERDICT_DRIFT`.

A violation is `INVALID__REGRESSION`: the implementation is reverted, nothing is learned about the
annotation, and this document stays as the record that it was attempted.

## What is deliberately out of scope

No stored certificate is regenerated. History stays byte-identical; only newly issued certificates
carry the annotation, and `verifier_sha256` changing is the designed drift signal doing its job.
No status logic, no obligation clause, no bar. The census over the annotated corpus (the
unobligated-oath rate) is a separate measurement that follows, not part of this change.

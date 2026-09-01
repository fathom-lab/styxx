# RECON — v0.13 was not frozen: obligation does not gate verification

Fathom Lab · 2026-08-28 · **RECON. A preregistration was drafted and killed before it was
committed. No clause exists, no bar was frozen, `styxx/certify.py` is untouched.**
Receipt: `oath_obligation_ladder_census.json`.

## What was drafted

A v0.13 preregistration for a structural obligation trigger — obligate a numeric token when it has
**two or more decimal places** and sits **outside a backtick code span** — following
`RECON_structural_obligation_2026_08_28.md`, which measured that rule at `0.8` precision against a
`0.4009` base rate and licensed a preregistration rather than a clause.

It had five gates, an asserted invariant, a held-out split, and a pre-committed outcome table whose
headline negative was *"the clause buys coverage by making the attestation worse."*

It is not frozen. Three of its five gates cannot fail, and its central framing is backwards.

## The ladder decides everything, and it was never read carefully

`certify_doc` assigns status with an **ordered** ladder:

```
if is_spec or is_hist:   -> ABSTAIN  "spec-or-historical"
elif is_notation:        -> ABSTAIN  "v05-notation"
elif derived_ref:        -> VERIFIED
elif field_unbound_ref:  -> ABSTAIN  "unbound-field"
elif hits:               -> VERIFIED          <-- before `bound` is consulted
elif bound:              -> UNGROUNDED
else:                    -> ABSTAIN  ref=None
```

**A token currently `ABSTAIN` with `receipt_ref: null` reached the final `else`.** That branch is
reachable only when `hits` is empty. Adding a disjunct to `bound` moves such a token to
`elif bound`, so it can only ever become **UNGROUNDED**.

Measured over the committed corpus: of `2040` abstained tokens the clause fires on `493`. But
`360` of those are `spec-or-historical` and `6` are `v05-notation` — **intercepted at the top of
the ladder, above `bound`, where obligating them changes nothing.** Only `127` tokens across `60`
documents can move at all, and **every one becomes an accusation. Zero become verifications.**

So three load-bearing gates die together:

* **G1** required 200 movable tokens. There are `127`. Dead on arrival.
* **G2** would be scored on a sample drawn from a pool of `127`, not the `493` its bar was
  calibrated against.
* **G3** — *"of newly-obligated tokens that come back VERIFIED, the claim-share must be ≥ 0.7933"*
  — has a **structurally empty denominator**. It would be scored `0/0` and could not fail. It was
  the gate the preregistration called the most important negative available.

A fourth, **G4**, requires `styxx.discriminates` to return `SEPARATES` against a null rule that
obligates *every* number. A maximal-cost control is beaten by any rule firing on a strict subset,
so `SEPARATES` is certain. The committed census receipt already shows it passing for both
*a-priori lexical controls* — the rules that same RECON calls "what carrying no information looks
like."

I demoted one unfailable gate before freezing and wrote that counting it "would have inflated this
preregistration's apparent rigour by one." It was inflated by three more.

## The finding underneath, which is the real result

Because `elif hits` precedes `elif bound`:

> **Obligation does not gate verification. It gates accusation only.**

Verified directly rather than argued. A line carrying no measurement vocabulary at all —

```
Legal scholars have long argued about 0.4267 in the abstract.
```

— against a receipt holding `{"whatever": 0.4267}` returns **VERIFIED**, bound to
`r.json:whatever`. Nothing obligated the verifier to examine that number. It swore to it anyway.

This is the same mechanism as the `gpu_memory_fraction` defect fixed in `styxx/oathready.py` the
same day, and it explains it: a value match produces an oath without the obligation predicate ever
being consulted.

## What it corrects in what we already published

Documents dated 2026-08-27 describe the abstained band as *"checkable claims the verifier declined
to examine."* That phrasing reads as **claims it would otherwise have verified**. It would not.
Those tokens abstained precisely because no receipt holds their value; obligating them yields
accusations, not verifications.

The coverage gap is real and the measured miss rates stand. **Their character is different from how
they were written.** The gap is *claims with no backing receipt that go unflagged*, not *backed
claims that go unchecked*. `OATH_CONTRACT.md` and the internal RESULT overstate it in that specific
direction, and this document is the correction of record until they are amended.

It also means the repair is not "obligate more." A predicate that only manufactures accusations
cannot close a coverage gap. What would close it is authors persisting the values — a contract
problem rather than a verifier problem.

## Provenance of this finding, disclosed

The lead observation came from an adversarial red team run against the preregistration before
freezing. **Its verification pass never ran** — seven of eleven agents died on an account quota
limit, so no skeptic refuted or confirmed anything and the decision memo was never written.

The findings above are therefore **not panel-verified**. They are verified by the author, directly,
against the source and the corpus, with the census committed beside this document so a reader can
re-run it. Where the reviewer's numbers and mine disagreed, mine are used: the reviewer reported
`56` transitions over 92 reproducing certificates, the census here reports `127` movable tokens over
the full corpus, and those are different populations rather than a contradiction.

---

*Drafted, red-teamed and killed inside one day, on a defect that was six lines of `elif` the whole
time.*

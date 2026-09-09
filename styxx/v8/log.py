"""styxx.v8.log — the log v0 (spec section 8; contract section 2).

Storage layout (section 8.2), exactly::

    <root>/entries/<index // 10000:06d>/<index:08d>.json        the entry bytes, no trailing newline
    <root>/entries/<index // 10000:06d>/<index:08d>.meta.json   log metadata beside the entry
    <root>/blobs/<sha256 hex>.json                              content-addressed item payloads
    <root>/sth/<tree_size:012d>.json                            signed tree heads
    <root>/keys/log.pub                                         the log public key
    <root>/keys/issuers.json                                    the v0 issuer roster
    <root>/README.md                                            the four verification commands

Entry bytes are ``canonical_bytes(cert)`` and nothing else: no trailing newline, no CR. A CR
inside an entry file is a malformed entry (section 8.2) and reads as tamper.

Leaf hash = ``sha256(0x00 || entry_bytes)``, node hash = ``sha256(0x01 || left || right)``
(``styxx.v8.merkle``, RFC 6962 as restated in RFC 9162 section 2.1).

STH = ``{log_id, tree_size, root_hash, timestamp, sig}`` with the signature over
``"styxx.v8/sth/1" || 0x00 || sha256(JCS({log_id, tree_size, root_hash, timestamp}))``.

``public`` (section 2.6) is log metadata, never an envelope field: it is computed at append as
"this cert is not redacted and every cert it references is public".

Nothing here decides whether a cert is well formed — that is ``styxx.v8.cert.check``. This module
decides whether a well-formed cert may occupy a leaf, and what a stranger can re-derive from the
bytes on disk.

Decisions
---------

**A floor that measured nothing is refused, and the refusal names the factor.** A noise plan
(section 5.1 step 1) is a commitment the R runs honour, not a description they may ignore. The
first real floor declared ``batch_size`` with values ``1|8|32`` and ``item_order`` with three
permutations, then ran all five runs at ``batch_size`` 1 — where item order provably cannot move
anything, because each item is its own forward pass. Every pairwise distance was 0.0, the floor
was 0.0 on ``exact``, ``seqlp`` and ``topk``, and a later ``verify --diff`` against a real
precision change read ``exceeds_floor`` on every channel, because against a zero floor every
difference exceeds. The cert, the plan, the log, the tree head and the proofs were all valid
(`papers/v8/vacuous_floor_2026_09_09/`). Nothing in the system said the floor had no discriminating
power, so the log now says it.

THE PREDICATE, exactly. It runs on a **canonical** fingerprint only — a fingerprint carrying
``body.noise_floor`` — because that is the cert that makes the floor claim; the R−1 run certs
each vary nothing on their own and are not checked. Resolve the plan from
``body.noise_floor.plan`` and the runs from ``body.noise_floor.runs``; the assignments are the
appended cert's own ``body.nuisance`` plus one per resolved run cert, so with ``A`` assignments in
hand (``A`` = 1 + the number of run ids) the check is a comparison of values already in the log.
For every factor the plan declares with **two or more** values, append refuses when any of:

1. no assignment records the factor at all — the plan declared it and the certs do not even say
   what it was;
2. some assignments record it and others do not — a floor whose runs disagree about what was
   held fixed is not one measurement;
3. every assignment records the same value — the factor was declared and never varied. This is
   the defect above, and the reason for the whole check;
4. an assignment records a value the plan did not declare — the runs went somewhere the plan
   did not commit to, which is the mirror image of (3).

Values are compared as ``str(value)``, so the plan's ``"1"`` and the cert's integer ``1`` are the
same value. A factor declaring a single value commits to no variation and is skipped; a plan
carrying no ``nuisance`` list declares nothing and is skipped whole. What the check does NOT do:
it does not ask whether the variation moved any distance. A floor of 0.0 measured across three
batch sizes is a finding about the model; a floor of 0.0 measured with the plan's own factors
pinned is a floor about nothing, and only the second is refusable from the bytes.

**Distinct labels are not distinct executions, so the floor is counted in executions.** The
predicate above was attacked eight ways and accepted eight
(`papers/v8/challenge_and_attack_2026_09_09/`), six of them with a floor of exactly 0.0. Every
one worked the same way: the runs recorded assignments the plan really declared, and the
assignments named one execution. A plan of ``batch_size=8|16|32`` on an eight-item battery
records three values and runs one batch of eight three times, because the runner chunks with
``range(0, len(items), batch_size)``. ``padding_side=left|right`` at batch size 1 is applied and
emits no pad token, because a batch of one is padded to its own length. An item order permuted at
batch size 1 permutes independent single-item forward passes. In each case the check compared
what the certs *said* and never what they *ran*.

THE LINE, drawn where the bytes support it. Two runs are the **same computation** when they
issue the same forward passes. A run's **execution state**
(``styxx.v8.fingerprint.execution_state``) is that: the multiset of its batches, each batch the
ordered tuple of ``item_id`` values computed together (``body.items``, stored in run order,
chunked by ``body.nuisance.batch_size`` the way the runner chunks them), plus ``padding_side``
when some batch holds two or more items. The multiset is unordered across batches and ordered
within one, because position inside a batch is padding and reduction order while the sequence in
which independent batches are issued is not visible in a per-item comparison.

What is refused is a **declared factor that could not have moved an execution anywhere it was
declared**. For each factor the plan declares with two or more values,
``fingerprint.execution_states_for`` computes, at each of the floor's own runs, the states that
factor's declared values would produce with everything else that run recorded held fixed. Append
refuses when that set has one element at every run: the plan named a knob that, on this battery
at these batch sizes, turns nothing. That sentence describes the rule and NOT, until now, the
code: ``_factor_states`` unioned the per-run sets and tested ``len(union) > 1``, which is a
weaker and different question — true whenever the runs merely differ from each other, so a plan
of ``item_order`` at batch size 1 passed as soon as one named run differed in some other way.
The attacker who found it (R-UNION) proposed ``any(len(execution_states_for(run, factor,
values)) > 1 for run in runs)`` and observed that the per-run answer was already computed and
thrown away. That is right on both counts, and it is what the code now does. It refuses again
when no declared factor is both applicable
and realizable at all — a plan whose every factor names a single value commits to no variation.

* **This is not the same as "the floor is 0.0", and the difference is the whole point.** A
  configuration that genuinely repeats itself across genuinely different computations honestly
  produces a zero floor, and that is a finding about the configuration, not a defect. Such a
  floor stays appendable. It does not stay silent: ``floor_census`` writes ``zero_channels``,
  ``all_channels_zero``, ``executions``, ``pairs_same_execution``, ``advertised_runs`` /
  ``advertised_pairs`` / ``advertised_agrees``, ``factors_without_demonstrated_effect`` and
  ``factors_not_derivable`` into the entry's metadata, and ``verify``'s printed report names the
  zero channels beside the verdict, so a later verdict that leans on a zero floor leans on a
  number two layers have already labelled. What is NOT done here is a ``zero_floor`` key in the
  ``verify`` **result body**: those bytes are hashed by the committed conformance set
  (``conformance/v8/vectors/exit.json`` pins ``result_sha256``), and moving a committed receipt
  in place is not this change's to do. The printed report is not hashed by anything, which is
  why the label could go there today.
* **Nor is it "every run was a different computation".** A design whose runs collapse partly —
  five runs over three computations, three of ten pairs comparing a run with itself — appends,
  with ``executions`` and ``pairs_same_execution`` saying so. Refusing that would also refuse the
  lab's own first real floor (`papers/v8/first_verdict_2026_09_09/`), whose five runs are three
  computations because three of them ran at batch 1. The advertised ``pairs`` and
  ``alpha_single = 1/(pairs+1)`` overstate such a floor, and the census is where a reader sees by
  how much.
* **A factor no pair of runs isolates is reported, not refused.** In that same real floor no two
  runs differ only in ``item_order`` and also differ in execution, so ``item_order`` lands in
  ``factors_without_demonstrated_effect``. "This design demonstrated no effect" is what the bytes
  support; "this factor is inert" is not.

Three limits, named rather than hidden. A redacted run carrying ``items_blob`` and no ``items``
cannot be chunked, so its state falls back to its recorded ``item_order_sha256`` and
``batch_size`` — the labels; ``state_kinds`` says which runs were derived (``batches``) and which
were taken on trust (``recorded``). A factor outside the four this process can apply
(``batch_size``, ``item_order``, ``order``, ``padding_side``) is *not derivable* — a floor across
two physical GPUs is legitimate under section 5.1 step 2 and its batches are identical — so such
a factor is listed in ``factors_not_derivable`` and never refused on its own; it also cannot
satisfy the "something was varied" rule, which is consistent with the mint, where
``plan_run_settings`` refuses to run a factor it cannot set. And a plan carrying no ``nuisance``
list at all is still skipped whole: ``prereg noise-plan`` cannot mint one (``--nuisance`` is
required) and ``schema/prereg.json`` refuses an empty list, but a hand-signed plan with the key
omitted is a hole this check does not close. Closing it means requiring ``nuisance`` on a
noise-plan prereg in the schema, which moves committed conformance vectors.

The spec sentence this collides with, stated rather than quietly overridden: section 5.1 step 2
says the R runs vary "item order and, **where the plan says so**, ``execution.batch_size`` ...",
which reads as though item order alone were a sufficient plan. On this box it is not — at batch
size 1 permuting the order permutes independent single-item passes — and
`papers/v8/vacuous_floor_2026_09_09/` is the receipt. Section 5.6's "The plan must say which
[quantity]" is the sentence that survives; a plan whose factors reduce to one computation names
no quantity at all. A floor that varies ``batch_size`` is still not reachable through
``verify --diff``, because ``recipe_core`` gates on the whole ``decoding`` block — the A-29 drift
that ``tests/test_v8_spec_agreement.py`` carries two strict xfails for. That drift is NOT landed
here.

**The floor is recomputed from the run certs the cert names, and a disagreement is refused.**
The two checks above compare *labels*. Neither of them, and nothing else in the system, ever
re-derived ``floor_c`` itself, so the numerals in ``noise_floor.per_channel`` were free: an
attacker computed a floor over three batch-1 bodies (all pairwise distances 0.0), wrote
``runs = [run1, run2]`` beside it — two runs that really produce ``exact 0.046875``, ``seqlp
0.036070694``, ``topk 2.140233900`` — re-signed, and appended at exit 0
(`papers/v8/challenge_and_attack_2026_09_09/`, F6; ``attach_floor`` computes ``per_channel``
from ``run_bodies`` and writes ``run_ids`` beside it with nothing tying the two together).

A floor is not a claim. It is the Appendix B function of bodies this log holds, so the log
computes it: ``_check_floor_matches_its_runs`` resolves ``noise_floor.runs``, forms the run
bodies in the order ``fingerprint.attach_floor`` used (``_floor_body_certs`` — the appending
cert's body ahead of the named ones, unless run 0 was logged as its own cert, in which case the
named ones are the whole set and prepending would count run 0 twice), takes the item roles off
the appending cert's own ``items`` the way ``attach_floor`` does, calls ``floor.floors``, and
refuses when any of ``floor``, ``distances``, ``runs``, ``pairs`` or ``alpha_single`` differs on
any channel, or when the signed channel set is not the derived one.

The signed block must BE the derived block, and that is stronger than "every number agrees"
(A-OPTIONAL). A key the derivation makes and the cert omits is a disagreement, because an absent
key is not an agreeing key; a key the cert adds beside them is a disagreement, because nothing
recomputes it and it is therefore the issuer's assertion sitting inside a measured block
(``{"agrees": 999, "note": "measured over 500 runs"}`` went into every channel block of an
otherwise honest floor and appended at exit 0). Section 5.7's overall size is REQUIRED on the
same footing rather than re-derived when offered: ``alpha_overall``, ``standardized_max``,
``alpha_overall_method`` and ``standardization`` are the same derivation over the same bodies,
and a cert with the first two deleted used to append with ``floor_disagreement == []`` and
5.7 simply absent. Requiring the KEYS in ``schema/fingerprint.json`` was the other option and it
is the weaker one — a schema can say a number is present and cannot say it is the number the
runs give, which is a mandatory field in front of the same hole.

* **The tolerance is zero: these numbers are recomputable exactly, and the check is ``==``.**
  Both sides are ``round(x, 9)`` (``distances.rounded``) applied to the same deterministic
  Appendix B arithmetic over the same item bytes, and a JSON round-trip of a Python float is
  exact, so an honest cert reproduces bit-for-bit. There is no float path here that forces a
  window, and a window would be a place to hide a floor in. If a future channel introduces one,
  it belongs in ``floor``, named and receipted, not as a slack term here.
* **A named run that is not in the log is a refusal, not a skip.** ``cert.check`` already forces
  every embedded id into ``refs`` and step 3 resolves every ref, so this branch is unreachable
  through ``append``; it is written as a refusal anyway, because the alternative — recomputing
  over whatever happens to resolve — is a check that passes hardest exactly when the evidence is
  missing. Same for a run whose ``items`` cannot be read (a redacted run carrying only
  ``items_blob``): the floor cannot be re-derived from it, and it is refused with the reason
  saying so rather than taken on trust. Resolving ``items_blob`` out of the log's own ``blobs/``
  would make those floors anchorable too; that is not done here.
* **The same recomputation is NOT added to ``verify``, and the reason is not that it would be
  wrong.** ``verify`` reads a floor off a cert reached anywhere, from a caller whose resolver
  may hold no run certs at all: refusing then would break the honest offline case, and passing
  then would be a check whose loudest state is silence. Where the runs *are* resolvable the
  computation is public — ``Log.floor_disagreement`` returns the same list of disagreements
  without appending anything — so a client that holds them can run it. What is deliberately not
  done is putting the outcome in the ``verify`` **result body**: those bytes are hashed by the
  committed conformance set (``conformance/v8/vectors/exit.json`` pins ``result_sha256``), and
  moving a committed receipt in place is not this change's to do.
* **A floor over a hand-picked subset of the preregistered runs is refused separately, because
  recomputation cannot see it.** F7 names 2 of the 5 runs the plan fixed and signs the floor
  those 2 really give, so the anchor agrees with it and says nothing; what is wrong is not the
  arithmetic but the sample. ``R`` is fixed in the plan before the runs
  (``body.runs`` on the noise-plan prereg), so ``_check_floor_names_the_plans_runs`` requires the
  floor to rest on exactly ``R`` run bodies. On the lab's own published verdict twelve subsets
  satisfied the old predicate, with ``topk`` floors from 0.813925214 to 2.140233900 and the
  verdict word flipping between them.

**A floor is a statement about ONE subject, and the anchor never asked whose runs it was
adding up.** A third adversarial pass put it in one sentence: the boundary slid from the floor's
arithmetic to the floor's provenance, and the issuer still writes the bytes. Its sharpest finding
(A-SWAP) took the published fp16 canonical, relabelled it ``run_index 1``, gave it a bf16 run's
``nuisance`` and ``recipe``, and named it as a run of the bf16 canonical's floor. The plan check
passed (the labels are the plan's), the completeness check passed (R = 5 bodies), the label/recipe
check passed, and the anchor recomputed the floor over the fp16 body and agreed with it exactly —
``floor_disagreement() == []``. The signed floor was ``exact 0.0625 / seqlp 0.058564664 / topk
1.407087824`` against the honest bf16 floor's ``0.046875 / 0.036070694 / 2.140233900``, and a
``verify --diff`` of that cert against the fp16 canonical read ``same`` on every channel: the
cross-precision drift had been made the floor, and the recomputation certified it to the last
digit.

``_check_floor_runs_are_one_subject`` is the repair, and it is a rule about the certs THIS LOG
HOLDS at the ids the block names. Every run a floor rests on must carry the same ``S_identity``
(section 2.2 — ``precision`` and ``revision`` included, which is why this compares identity fields
directly instead of reading ``cert.comparable``, whose ``cross-subject:`` softening is right for a
comparison and wrong for a run), the same ``cert.SYNTHETIC`` marker, and the same ``recipe_core``
(section 2.3) as the canonical — except on the ``decoding`` keys a floor is allowed to move:
``batch_size`` and ``padding_side`` (section 5.1 step 2's execution knobs), plus any other
``decoding`` key the floor's own plan declared as a nuisance factor with two or more values. That
exception is the shape of an honest floor and not a softening: the lab's own published floor has
runs at ``decoding.batch_size`` 8 and 32 under a canonical at 1. A run over another battery,
another chat template, another system prompt, or at another ``seed``, ``temperature`` or
``max_new_tokens`` that no plan declared is refused by name.

**Which bodies the floor is over is decided by ``R``, not by a signature the appending party
writes.** ``_floor_body_certs`` used to tell the R-id shape from the R−1 shape by looking for a
named run carrying the appending cert's ``run_index`` and ``items`` — and A-SPLIT observed that
this is a signature the appending party writes, that flipping it moves which bodies the anchor
computes over, and that ``_check_floor_names_the_plans_runs`` then counted the post-flip list, so
the completeness check moved with it. ``R`` is the one number in a floor that was committed before
the runs, on the noise-plan cert the signed block names by id. So ``R`` decides the shape, and the
signature is then required to AGREE with it: with ``R`` ids exactly one named run must carry the
canonical's ``run_index``, and its ``items`` and ``channels`` must be the canonical's, which is
what makes the split numerically inert — ``floor.floors`` reads those two fields and nothing else,
so whichever of the two bodies is counted, every distance is the same number. With ``R−1`` ids no
named run may claim run 0. What is left is written down rather than hidden: a hand-signed plan
that omits ``runs`` fixes no ``R``, and there both this rule and the completeness rule fall back to
the old signature. Closing that means requiring ``runs`` on a noise-plan prereg in the schema,
which moves committed conformance vectors.

**What the anchored check still trusts, stated so it can be attacked.** It trusts that
``body.items`` are the run's outputs — the anchor makes the floor a function of those bytes and
of nothing else, which is the point, but no part of this log witnessed the forward passes. It no
longer needs ``body.nuisance`` to be honest *for the floor numerals*: relabelling ``batch_size``
does not move a distance, because ``floor.floors`` never reads ``nuisance``. It does still read
``nuisance`` for the execution count, and that was attacked: relabelling three real batch-1 runs
as 1 / 8 / 32 manufactured three "executions" out of one computation and passed
(`papers/v8/challenge_and_attack_2026_09_09/`, R-EXEC and R-EXEC-2). **Recomputing the floor does
not by itself refuse that** — the relabelled certs carry the same ``items``, so the floor is
honestly the floor of the bodies named. What refuses it is
``_check_floor_labels_match_the_recipe``: ``execution_state`` chunks ``body.items`` by
``body.nuisance.batch_size`` and reads ``padding_side``, and both are now required to equal the
``recipe.decoding`` the same cert signs — which is what ``cli`` already writes per run ("a cert
never records an execution its run did not use"). That does not make the label true; it makes a
relabel cost a change to ``recipe_core``, which changes what the cert is comparable to. A cert
whose recipe and nuisance were forged together is still believed, and there is nothing in these
bytes that could catch it.

R-EXEC-3 then did exactly that, relabelling ``recipe.decoding`` alongside ``body.nuisance`` so
that four certs which all really ran at batch 1 declared 8, 8, 32 and 1, and it passed. **That
attack is OPEN, and it is open by construction.** Both halves of the check are fields of one cert
signed by one key, chosen by one party; a predicate over those bytes can only ask whether the
party contradicted itself, and a party that does not contradict itself is not caught by asking.
No check written in this module closes it, and one that appeared to would be worse than the gap.
What the subject rule above buys against it is a price and not a wall: the forged ``decoding``
now has to be a value the plan declared, committed before the runs, or
``_check_floor_runs_are_one_subject`` refuses the run for a recipe_core it never preregistered.
What would actually detect it is a second party running the same battery under the same plan and
publishing its own floor: one computation relabelled R ways gives a floor of 0.0, a real R-run
floor on the same subject does not, and the two disagree in public. That is a replication, not a
predicate; it lives in ``REPLICATIONS.md`` and in section 9's challenge, and nothing in these
bytes substitutes for it.

**A floor's COVERAGE is derived from its plan, not written beside its numbers (A-COVER).** The
anchor above re-derives ``per_channel`` and section 5.7's overall size from the run certs and
compares them exactly, and for as long as it did, ``noise_floor.covers`` and
``noise_floor.not_covered`` sat beside those numbers as two lists nothing re-derived at all. A
canonical fingerprint signed with ``covers`` naming the gpu, the driver, the runtime and
everything else and ``not_covered: []`` appended at exit 0 over an honest floor, with
``floor_disagreement`` returning ``[]`` — because the disagreement was never in the arithmetic.
It mattered rather than being untidy for two reasons in the spec's own text: ``verify`` computes
``coverage_diff`` over ``not_covered`` alone, so an empty list is the claim that every
environment is inside this floor's coverage and no reproduction anywhere is ever
``beyond-floor-coverage``; and section 5.4 spends ``covers`` on deciding whether an outside
reproduction is a binding dispute or an unreachable coverage report (section 9), so an issuer
choosing it after the runs chose who was allowed to contradict it.
``_check_floor_covers_match_the_plan`` derives both from the plan's ``nuisance`` and
``environment`` (``floor.plan_coverage``, the same function ``cli`` mints with) and refuses a
disagreement. A plan that fixes no ``environment`` is refused there rather than skipped: the
derivation from an absent environment is an empty ``not_covered``, which is the permissive claim
itself, so a quiet branch would have handed back the whole attack for the price of one omitted
key — the shape A-NORUNS had.

**A floor's runs must have happened in the environment its plan fixed (PLAN-ENV).** The plan is
where the environment is committed before the runs, and ``prereg noise-plan`` used to build no
runner at all and copy that block out of the ``--subject`` spec, so section 2.2's mint rule
reached the fingerprint and not the one cert every run of a floor names by id. It observes now
(``cli.cmd_prereg``), which makes the plan's environment a statement about the plan's own box —
and a statement about that box binds nothing here until the runs are compared against it.
``_check_floor_environment_matches_the_plan`` is that comparison: every leaf of the plan's
environment outside its declared factors must read the same in every floor cert's
``subject.environment``, with an absent leaf refused rather than passed. One-directional, so a
run may still carry the ``harness`` and ``env_lock_sha256`` leaves a runner cannot observe.
**The limit is the label class again**: both sides are one issuer's bytes, and a party that mints
its plan and its runs on one box and writes one environment into both is not caught by comparing
them. What is removed is the state where the two disagreed and nothing looked.

**A challenge is checked against its target's subject, here and at mint.** Section 9 rule 1: "A
challenge is valid iff the challenger's fingerprint is comparable (section 2.3) AND has the same
subject identity ...; clients compute this from the two certs — nothing in the body asserts it.
No match, no challenge." An attacker filed a challenge whose ``own`` half was an fp16 fingerprint
against a bf16 target, and the CLI, ``cert.check``, ``log.append`` and the JavaScript verifier
all took it; run by hand, ``cert.comparable(target, own)`` said ``['cross-subject:precision']``
(same paper, C1). The rule is one function, ``cert.challenge_validity``, and it is run in three
places on purpose:

1. **By the client, from the two certs, as section 9 says.** That is the primary object: it is
   what a reader of a challenge computes wherever it reached them, and it is implemented twice —
   ``cert.challenge_validity`` and ``challengeValidity`` in ``styxx/_data/v8_verify.js``.
2. **At mint**, in ``verify --challenge``: a tool does not sign a cert it can already show is not
   a challenge.
3. **Here, at append.** Section 9 gives the *computation* to clients because the body asserts
   nothing; it does not ask a log to seat a leaf whose own refs prove it is not what it says it
   is. Both certs resolved at step 3, so this is a function of bytes this log already holds, and
   the refusal is the same class as step 3's "this ref resolves to the wrong type". The log is
   not adjudicating the dispute — ``Disputed`` stays client-computed over a trust file, and
   nothing here counts, resolves or scores a challenge.

What that costs, stated rather than hidden: a challenge this log refuses can still be minted and
published elsewhere, so refusing here is a property of THIS log and not of the format, which is
exactly why (1) exists and is the primary.

**That repair was then defeated four ways, and every one of them was an early return.** A second
adversarial pass (same paper, C-DUPREF, C-MINT-UNCOMPUTED, C-NONFP-TARGET, C-MOCK) went around
``_check_challenge_subject`` rather than through it. The pattern is one sentence: *the repair
moved the trust boundary rather than removing it.* Each hole and the rule that closes it:

* **Two ``own`` refs.** ``by_role[role] = ...`` in a loop is last-wins, so a challenge carrying
  ``refs = [target, own=<fp16>, own=<bf16>]`` passed the log's reading of rule 1 on the bf16 cert
  while a reader walking ``refs`` in order saw the fp16 one. Refused in two places: ``cert.check``
  refuses a repeated single-valued role for every cert type (``cert.MULTI_ROLES``, and the
  argument is in that module's "Decisions"), and this method refuses anything but exactly one
  ``target`` and one ``own`` — because a log that depends on ``check`` having run first is a log
  with a second trust boundary in it.
* **A target that is not a fingerprint.** ``ROLE_TYPES['target']`` is ``"*"`` — correctly, since
  a ``result`` of kind ``response`` names a challenge — so a challenge could name the BATTERY
  cert as its target, the old method returned early on the type, and the whole subject check
  never ran. ``cert.role_type(cert_type, role)`` adds the per-type override, so step 3 now types
  a challenge's ``target`` as ``fingerprint`` before this method is reached, and this method
  refuses rather than returns if it ever is.
* **A body that asserts nothing.** See ``_check_challenge_self_report``.
* **Distances from a runner holding no model.** The marker is ``cert.SYNTHETIC``, it is inside
  ``comparable``, and ``challenge_validity`` IS ``comparable`` — so this method already refuses a
  synthetic challenge against a measured target with no line of its own. The reasoning is in
  ``styxx/v8/cert.py`` under "Decisions".

Read together: an early return in a guard is a pass, and every one of these was written as
"not this rule's business". A rule that cannot decide must refuse, and this method now has no
branch that ends without a verdict.

**The metadata beside an entry is derived on read, never trusted (A-META).** The floor census
was written into ``<index>.meta.json`` at append and read back by anyone who asked. That file is
outside the Merkle tree, is not signed by anything, and ``verify_entry`` compared exactly two of
its fields — ``id`` and ``index``. On the published log the attacker rewrote the census in place
(``executions`` 3 -> 5, ``pairs_same_execution`` 2 -> 0, ``all_channels_zero`` true -> false),
and ``verify_entry`` returned ``ok``, the root was unchanged because no leaf moved, and ``mirror``
reported ``verified: True, tamper: []``. ``floor_census`` already re-derives the whole block
exactly, from certs the log holds, and nothing called it.

Section 8.2 already says what the rule is — "``meta.json`` and ``index/by_id.ndjson`` carry no
authority: both are recomputable from ``entries/`` by any mirror, and ``mirror`` reports a
disagreement rather than trusting the file" — so this is the implementation catching up with the
spec rather than a new rule. ``derived_meta`` recomputes every derivable field from the entry
bytes; ``meta_disagreement`` names each field the file disagrees on; ``verify_entry`` refuses on
any of them, and ``mirror`` collects them under a ``metadata`` key beside ``tamper``.

**A file that is OLD is not a file that is WRONG (A-META-ABSENT).** The repair above was written
with one predicate for two states, and the day after it landed it accused the lab's own published
log. `papers/v8/first_verdict_2026_09_09/log` was minted before ``floor_census`` existed, so its
seven ``<index>.meta.json`` files carry no ``floor`` key at all; ``mirror`` re-derived the census
at entry 6, found the file silent, and printed it under ``tamper`` with ``verified: False``. That
log is untouched, published and pushed. Nothing had edited it, and the checker said it had.

THE SPLIT, which is the predicate now:

* **ABSENT** — the stored file holds no value for a derived key. The file predates the field. The
  derivation is authoritative, the file is a stale cache, and *nothing the entries say is
  contradicted*. Reported under ``stale_metadata`` / the mirror's ``stale_metadata`` key, and it
  does not refuse and does not clear ``verified``.
* **CONTRADICTED** — the file states a value and the entries derive a different one, or states a
  value where the entries derive none, or carries a key no derivation over the entries produces
  at all. Something asserts a fact the bytes refute. This is A-META's attack, and it stays a
  refusal in ``verify_entry`` and a ``verified: False`` in ``mirror``.

The asymmetry is not a courtesy, it is the information content: an absent key removes an
assertion and adds none, so an attacker gains nothing by deleting one — the derivation is what
every reader uses, and deleting ``public: false`` does not make a redacted cert public. Deleting
``id`` or ``index`` is refused a line earlier in ``verify_entry`` on its own terms (section 8.2
names both in the layout), so the split does not open them either. What is deliberately NOT
widened is the third contradiction above: a key the file carries that this version derives
nothing for stays a refusal, on A-OPTIONAL's argument — it is a presence, not an absence, and a
free assertion inside a file readers quote. A future version that *removes* a derived key would
turn honest old files into that state; when that happens the retired key belongs in a named set,
not in a softened predicate.

**Why this is a correctness bug and not a papercut, in this lab's own receipt.** EXTERNAL-1
measured a path-claim accusation at 0.23 precision on external agents' pull requests and DISABLED
the class: an accuser that fires on honest artifacts is not a strict checker, it is a broken one,
and its output stops being read at all. A tamper report that cannot tell "old" from "wrong" is
that same defect with a worse blast radius — the accusation here is machine-readable (``mirror``
exits non-zero, ``verified`` is the sentence readers quote), it is aimed at an *artifact* rather
than at a diff, and the artifact is usually someone else's. The strength of the CONTRADICTED half
is exactly what makes the ABSENT half expensive: a checker that refuses everything is as useless
as one that refuses nothing, and only the half that fires on real edits is worth keeping loud.

**Where a stale file gets repaired: nowhere automatically, and that is the decision.** Re-deriving
one costs nobody's key — ``<index>.meta.json`` is outside the Merkle tree and nothing signs it, so
rewriting it moves no leaf, breaks no proof and changes no root, and ``derived_meta(index)`` plus
``appended_at`` *is* the repaired file. Two places could do it and neither should:

* **Not the source log.** It is published and pushed. This lab already has the receipt for
  regenerating a committed artifact in place, and a metadata file quoted by a reader is one. If
  the operator of a log chooses to refresh their own cache, that is their deliberate act at their
  own path, and ``derived_meta`` is the whole implementation of it.
* **Not the mirror's copy.** ``mirror`` is evidence *about* the source. A mirror that silently
  rewrote ``dst`` would produce a copy that no longer agrees with the bytes it copied, and the
  next mirror of that mirror would report agreement — which erases the very fact ``mirror`` exists
  to publish. Worse, the same code path applied one branch over would launder a CONTRADICTED file
  into a clean one.

So the file stays stale, the reader is told it is stale under its own key, and every consumer of
the value uses the derivation. That is what "a cache with a checker in front of it" means when the
cache is old rather than hostile.

Three consequences that are the point of doing it this way rather than by hashing the file:

* ``public`` (section 2.6) used to be derived by reading OTHER entries' stored ``public`` flags,
  so one edited metadata file flipped the transitive answer for every cert above it. It is now
  computed recursively from the certs (``_public_at``), which is what section 2.6 describes.
* ``_id_map`` used to prefer the metadata's ``id`` over the entry's, so an edited file could point
  an id at another leaf and every ref resolution downstream followed it. It reads the cert.
* A key in the file that nothing derives is reported like ``A-OPTIONAL``'s extra channel keys:
  ``appended_at`` is the one non-derivable field (a timestamp is an assertion, section 8.1), it is
  named as such, and anything else is a free assertion sitting in a file readers quote.

**Should unsigned metadata exist at all when it is derivable? No, and it is kept anyway.** The
honest answer is that a derivable file adds no information and one attack surface: every property
it holds is a function of ``entries/``, so a reader who computes them needs nothing from it, and a
reader who does not compute them is trusting the operator's copy. What keeps it is that section
8.2 pins the layout by name and this change may not move a published layout; it is now a **cache
with a checker in front of it**, which is a weaker thing than a record. The residue, written down
rather than closed: a client that reads ``meta.json`` and never calls ``verify_entry`` or
``mirror`` still gets the operator's bytes with no check, and no predicate here can reach that
client. Deleting the file is the repair that actually closes it, and it is a layout change.

**A missing issuer roster fails CLOSED (L10c).** ``issuers()`` returned ``None`` when
``keys/issuers.json`` was absent and append step 2 read ``if roster is not None``, so the control
the attacker ran is two lines: with a roster present a rogue key is refused; delete the file and
the same rogue key appends at exit 0. A security check whose absent configuration means "allow
everything" is not a weak check, it is an off switch reachable by ``rm``.

The rule now: ``issuer_policy`` reads the file and every branch decides. A JSON array is the
roster and admits the keys it lists at the indices it lists. A JSON object ``{"policy": "open"}``
is an **intentionally open log** and admits any key. Anything else — the file absent, unreadable,
carrying a BOM, or naming a policy this version does not know — refuses every append. An empty
array is a roster admitting nobody, and that is a different statement from "open"; both are
statements, which is the property that was missing.

**An open log says so on purpose.** The marker is a file the operator writes deliberately, it
carries an optional ``reason``, and it is reported: ``mirror`` prints ``issuer_policy`` in its
report, so a reader of a mirror learns the admission policy of the log it mirrored without
inspecting the directory. What that does NOT do, stated rather than implied: the marker is an
unsigned file outside the tree, exactly like the roster, so it is the operator's assertion about
its own log and a mirror can only report what it copied. C-12 option (a) — the roster as a
``result`` cert of kind ``document`` signed by the log key and appended like any other entry, so
that a reader sees the index at which a key was admitted — is the version that would put this
inside the tree, and it is the operator's decision, not this module's. Section 8.2 currently says
there is no ``keys/issuers.json`` in the layout at all while the C-12 interim stands; this code
has one and gates appends on it, which is a divergence the spec has to resolve either way.

**A cert may name the log it belongs to, and this log refuses one that names another (L7).**
Seven entries of the published log were replayed verbatim into a log built on a DIFFERENT key.
All seven appended, byte for byte, giving the same Merkle root under a different ``log_id`` — so
"this cert is in the log" was not a checkable statement, only "these bytes are in *a* log".

The binding is an optional ``body.log_hint = {log_id, locations?}``, the shape ``DURABILITY``'s
C2 proposes, with one difference: C2 gives it "the standing of ``created`` and never
authoritative", and an advisory field is exactly what the replay walks through. It is signed
(``body`` is inside ``D``), so an attacker cannot add, remove or edit it without a new signature
under a key the roster admits, and ``_check_log_binding`` refuses at append when the named
``log_id`` is not this log's. ``verify_entry`` runs the same predicate on read, so a mis-seated
entry is caught in a clone as well as at the gate. ``mirror`` reports how many entries are bound
and how many are not, because an unbound corpus is the state this repair does not change.

``log_id`` is ``sha256:<hex>`` of the raw log public key (section 8.1), which is the cert-id
grammar, so section 2.1's refs rule would demand it appear in ``refs`` and resolve. It is a hash
of bytes and not a cert, so it is exempted by name in ``cert.NON_REF_HASH_PATHS`` — the third such
exemption, where the spec's A.1 says "there is no third exemption and none is inferable". That
sentence has to move for this field to be legal, and the owed edit is stated here rather than
made quietly.

**The trade, argued rather than asserted.** A cert naming its log cannot be re-logged elsewhere,
and that cuts both ways.

* It is a feature where the claim is provenance. Appendix D hands a stranger "the cert, the log
  location, and the log public key obtained from a channel the operator does not control", and
  step 2 verifies an inclusion proof under the pinned key. Without the binding, a party who
  obtains the entries can stand up its own log, append them unchanged, sign its own STHs and
  produce inclusion proofs that verify under its own key — and a reader who did not pin
  independently cannot tell the two logs apart, because the leaves are identical. The binding is
  what makes the cert itself say which key its proofs must verify under.
* It is a problem where the claim is durability. Section 8.1 makes a key rotation a NEW
  ``log_id``, so every bound cert becomes unappendable to the successor log, and an archivist who
  wants to re-log a dead lab's corpus under their own key cannot. That is a real cost and it is
  not closed here: the repair is that a successor log accepts a predecessor ``log_id`` named in a
  ``document`` cert (section 8.1's rotation announcement), which is a rule about a set of accepted
  log ids that this module does not implement.
* Which is why it is OPTIONAL and not required. Requiring it would refuse every cert already
  signed and every committed conformance vector, and would make the format unusable before a log
  exists — a cert cannot name a log that has not been created. Leaving it optional means an
  unbound cert is exactly as replayable as before, and the honest statement of what this buys is
  therefore "a cert MAY be bound, and a bound cert is refused by the wrong log", not "certs are
  bound".

**The plan is at a lower index than the runs it governs (L6).** Section 7.2 says it in five
words — *the log index is the proof of order* — and nothing compared the plan's index against its
runs'. The route is not exotic: section 5.1 step 5 (ii) has each run reference the plan under role
``noise_plan``, and refs resolve backwards, so a run carrying that ref cannot precede its plan.
A run cert that simply omits the ref carries no such constraint. Append the R−1 runs first,
chained by ``previous`` so section 5.5 is satisfied, mint the plan afterwards, then the canonical
that names both: the plan sits ABOVE its own runs and the floor is preregistered by a document
written after the data existed, which is the one thing a prereg exists to rule out.

``_check_floor_plan_precedes_its_runs`` refuses a floor whose plan's index is greater than any
named run's. What it does not reach, because no predicate over these bytes can: the plan and the
runs may have been WRITTEN in any order at any time; the index orders the appends, not the
computations, and section 8.6 already says the log cannot tell a late-logged prereg from an early
one. The rule this adds is narrower and is worth exactly what it says — a floor cannot rest on
runs the log accepted before it accepted their plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as fpmod
from styxx.v8 import floor as floormod
from styxx.v8 import keys, merkle
from styxx.v8.consts import CERT_TAG, ID_RE, STH_TAG
from styxx.v8.jcs import canonical_bytes

__all__ = [
    "APPEND_TIME_META_KEYS",
    "ASSERTED_META_KEYS",
    "AppendRefused",
    "DERIVED_META_KEYS",
    "GENESIS_UNRESOLVED",
    "Log",
    "OPEN_POLICY",
    "STH_KEYS",
    "mirror",
    "verify_consistency",
    "verify_entry",
    "verify_inclusion",
    "verify_sth",
]

SHARD = 10000
STH_KEYS = ("log_id", "tree_size", "root_hash", "timestamp", "sig")
_STH_CORE = ("log_id", "tree_size", "root_hash", "timestamp")

# The value of ``keys/issuers.json``'s ``policy`` key that means "this log admits any issuer".
# A log with no ``keys/issuers.json`` at all refuses every append (L10c): the difference between
# "nobody wrote a policy" and "the operator chose an open one" is the whole repair.
OPEN_POLICY = "open"

# The metadata fields ``<index>.meta.json`` holds that are a function of ``entries/`` alone, and
# are therefore re-derived on read rather than trusted (A-META, section 8.2). ``appended_at`` is
# the one field outside this set: a timestamp is the signer's assertion (section 8.1) and no
# predicate over the entries produces it.
DERIVED_META_KEYS = ("index", "id", "type", "public", "floor")
ASSERTED_META_KEYS = ("appended_at",)

# Metadata fields that ARE computed from the log's own bytes and still cannot be compared on
# read, because their value depends on the log AS IT STOOD AT APPEND rather than on the entries.
# ``baseline_gap`` is the case: it resolves the previous comparable fingerprint by scanning the
# whole log, so an entry that had no comparable predecessor when it was appended acquires one as
# soon as a later comparable fingerprint lands, and re-deriving it at read gives a different and
# strictly later answer than the file records.
#
# THIS IS A HOLE AND IT IS NAMED RATHER THAN CLOSED. These fields are exactly as rewritable as the
# census was before A-META, and ``meta_disagreement`` passes over them. Closing it needs the
# derivation to be index-relative -- "the previous comparable fingerprint at an index below this
# one" -- which is a change to the field's own definition and belongs to whoever owns it, not to
# the checker in front of it.
APPEND_TIME_META_KEYS = ("baseline_gap",)

# DEAD FOR ANY CERT BUILT TO THE SPEC, AND STILL LOAD-BEARING FOR ONE TEST FILE.
#
# It was the workaround for the defect section 4.5 now names: schema/battery.json used to require
# ``recipe.battery`` on every battery cert, section 2.1 forced that id into ``refs``, and section
# 8.3 made refs resolve -- so every battery named an earlier battery and no log could be started.
# This permitted exactly one unresolvable ref, the pool battery's ref to nothing.
#
# The schema is fixed: a pool-v1 or fixed-v1 battery is a root, carries no ``recipe.battery`` and
# therefore no ref, and ``test_a_root_battery_fills_an_empty_log_with_no_ref_to_resolve`` appends
# one at index 0 without coming near this set.
#
# WHAT BLOCKS THE DELETION: tests/test_v8_e2e.py builds its ladder's pool cert the pre-fix way --
# ``recipe=F.recipe()`` (whose battery is the placeholder F.BATTERY_ID) with a matching
# ``refs=[{"role": "battery", "id": F.BATTERY_ID}]``, at the "1. the pool battery" step of
# ``build_ladder`` -- and ``test_the_pool_battery_is_the_only_cert_with_an_unresolved_ref`` pins
# that behaviour. Deleting this set turns those two lines into 46 fixture errors. The repair is
# ``recipe={}, refs=[]`` there plus retiring that one test, in a file this change may not touch.
# Until then the rule "every ref resolves" holds with this one written-down exception.
GENESIS_UNRESOLVED = frozenset({("battery", "pool-v1", "battery")})

_ID_PATTERN = re.compile(ID_RE)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_Z = re.compile(
    r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    r"T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]+)?Z$"
)

README = """# styxx v8 log

    python -m styxx.v8 log verify-cert <entry.json>
    python -m styxx.v8 log verify-sth <sth.json>
    python -m styxx.v8 log verify-inclusion <proof.json> <sth.json>
    python -m styxx.v8 log verify-consistency <sth-first.json> <sth-second.json>
"""


class AppendRefused(ValueError):
    """A cert the log will not put in a leaf, or an STH it will not write.

    ``reason`` is the single sentence that says why. It subclasses ``ValueError`` so a caller
    that expects ``ValueError`` from a wrong signing key sees one.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ----------------------------------------------------------------- small helpers

def _write_bytes(path: Path, data: bytes) -> None:
    """Write binary (no newline translation) and read back; OSError when it did not land."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.fspath(path), "wb") as fh:
        fh.write(data)
    with open(os.fspath(path), "rb") as fh:
        written = fh.read()
    if written != data:
        raise OSError(f"write to {path} did not land: read back {len(written)} bytes")


def _write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as UTF-8 JSON, sorted keys, two-space indent, LF, one trailing newline."""
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    _write_bytes(path, text.encode("utf-8"))


def _read_json(path: Path) -> Any:
    with open(os.fspath(path), "rb") as fh:
        data = fh.read()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name} carries a BOM")
    return json.loads(data.decode("utf-8"))


def _unhex(value: Any, length: int = 32) -> Optional[bytes]:
    if not isinstance(value, str) or len(value) != length * 2:
        return None
    try:
        return bytes.fromhex(value)
    except ValueError:
        return None


def _root_bytes(value: Any) -> Optional[bytes]:
    """The 32 raw bytes behind a ``"sha256:<hex>"`` root string, or None."""
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        return None
    return bytes.fromhex(value.split(":", 1)[1])


def _body(cert: dict) -> dict:
    body = cert.get("body") if isinstance(cert, dict) else None
    return body if isinstance(body, dict) else {}


def _redacted(cert: dict) -> bool:
    return bool(_body(cert).get("redacted"))


#: What ``_at_path`` returns for a path that is not there. Distinct from ``None``, which is a
#: value an environment leaf may legitimately hold.
_ABSENT = object()


def _at_path(obj: Any, path: str) -> Any:
    """The value at a dotted path, or ``_ABSENT``. A parent that is not a mapping is absent too."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return cur


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------- the log

class Log:
    """A log directory. Creating one creates the layout; nothing else writes to it."""

    def __init__(self, root):
        # ``path`` is the directory; ``root()`` is the Merkle tree head. The contract fixes the
        # method name, so the directory cannot also be called ``root``.
        self.path = Path(root)
        self.entries_dir = self.path / "entries"
        self.blobs_dir = self.path / "blobs"
        self.sth_dir = self.path / "sth"
        self.keys_dir = self.path / "keys"
        for d in (self.entries_dir, self.blobs_dir, self.sth_dir, self.keys_dir):
            d.mkdir(parents=True, exist_ok=True)
        readme = self.path / "README.md"
        if not readme.exists():
            _write_bytes(readme, README.encode("utf-8"))
        self._id_map_cache: Optional[tuple[int, dict[str, int]]] = None

    # ------------------------------------------------------------- construction

    @staticmethod
    def init(root, log_public: bytes, issuers) -> "Log":
        """Create the layout, write ``keys/log.pub`` and the log's admission policy.

        ``issuers`` is either a **roster** — a list of ``{name, key, from_index,
        retired_at_index|null}``, where ``key`` is the wire encoding of an issuer public key,
        ``from_index`` the first leaf index it may sign and ``retired_at_index`` the first index
        at which it may not — or the **open marker** ``{"policy": "open", "reason": <str>}``,
        which admits any key.

        There is no third option and in particular no "leave the file out": a log with no
        ``keys/issuers.json`` refuses every append (L10c), so an open log is a thing an operator
        writes down rather than a thing that happens when nobody writes anything.
        """
        log = Log(root)
        keys.validate_public(log_public)
        keys.save_public(log_public, log.keys_dir / "log.pub")
        _write_json(log.keys_dir / "issuers.json", _clean_policy(issuers))
        log._id_map_cache = None
        return log

    # ------------------------------------------------------------- paths

    def _shard(self, index: int) -> Path:
        return self.entries_dir / f"{index // SHARD:06d}"

    def entry_path(self, index: int) -> Path:
        return self._shard(index) / f"{index:08d}.json"

    def meta_path(self, index: int) -> Path:
        return self._shard(index) / f"{index:08d}.meta.json"

    def blob_path(self, hex_or_id: str) -> Path:
        return self.blobs_dir / f"{_blob_hex(hex_or_id)}.json"

    # ------------------------------------------------------------- reading

    def indices(self) -> list[int]:
        """Every entry index present on disk, ascending — gaps included."""
        out: list[int] = []
        for shard in sorted(self.entries_dir.glob("*")):
            if not shard.is_dir():
                continue
            for path in sorted(shard.glob("*.json")):
                if path.name.endswith(".meta.json"):
                    continue
                stem = path.stem
                if len(stem) == 8 and stem.isdigit():
                    out.append(int(stem))
        return sorted(out)

    def size(self) -> int:
        """The number of entries counted from 0 without a gap (a gap truncates the tree)."""
        n = 0
        present = set(self.indices())
        while n in present:
            n += 1
        return n

    def entry_bytes(self, index: int) -> bytes:
        """The exact file bytes of entry ``index``."""
        path = self.entry_path(index)
        if not path.exists():
            raise IndexError(f"no entry at index {index} ({path})")
        with open(os.fspath(path), "rb") as fh:
            return fh.read()

    def cert(self, index: int) -> dict:
        """Entry ``index`` parsed. ValueError when the bytes are not a JSON object."""
        obj = json.loads(self.entry_bytes(index).decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"entry {index} is not a JSON object")
        return obj

    def meta(self, index: int) -> dict:
        """The ``<index>.meta.json`` file beside entry ``index``, as it is on disk.

        THIS IS THE OPERATOR'S COPY AND IT IS NOT EVIDENCE (A-META). It is outside the Merkle
        tree and nothing signs it; ``derived_meta`` is what the entries give, ``meta_disagreement``
        is the difference, and ``verify_entry`` refuses on any. Section 8.2 says the same:
        ``meta.json`` carries no authority and ``mirror`` reports a disagreement rather than
        trusting the file.
        """
        path = self.meta_path(index)
        if not path.exists():
            raise IndexError(f"no meta for index {index} ({path})")
        obj = _read_json(path)
        if not isinstance(obj, dict):
            raise ValueError(f"meta {index} is not a JSON object")
        return obj

    def derived_meta(self, index: int) -> dict:
        """The metadata of entry ``index`` as the ENTRIES give it — every field but the timestamp.

        A pure function of ``entries/``: the id and type off the entry's own bytes, ``public`` by
        section 2.6's transitive walk over certs (not over other metadata files), the floor census
        by ``floor_census``, and the baseline gap by ``baseline_gap`` where one exists. This is
        what ``append`` writes; reading it back rather than the file is the repair.
        """
        cert = self.cert(index)
        out: dict[str, Any] = {
            "index": index,
            "id": cert.get("id"),
            "type": cert.get("type"),
            "public": self._public_at(index),
        }
        census = None
        try:
            census = self.floor_census(cert)
        except Exception:
            census = None
        if census is not None:
            out["floor"] = census
        return out

    def _meta_split(self, index: int) -> tuple[list[str], list[str]]:
        """``(contradicted, stale)`` for ``<index>.meta.json`` — the A-META-ABSENT predicate.

        One pass over the file, two lists, because the two states are different facts and were
        conflated exactly once (see "A file that is OLD is not a file that is WRONG"):

        * **CONTRADICTED** — the file states a value the entries refute: a derived key whose
          stored value differs, a derived key the file carries where the derivation produces
          none, or a key no derivation over the entries produces at all. An assertion against
          the bytes. Refusal.
        * **STALE** — the file holds no value for a key the entries derive. The file predates the
          field; the derivation is authoritative and nothing the entries say is contradicted.
          Reported, never a refusal.

        A metadata file that cannot be read or cannot be derived is CONTRADICTED, not stale: an
        unreadable file is not an old one, and a derivation that raised did not conclude "absent".
        """
        try:
            derived = self.derived_meta(index)
        except Exception as exc:
            return [f"entry {index}: metadata not derivable: {type(exc).__name__}: {exc}"], []
        try:
            stored = self.meta(index)
        except Exception as exc:
            return [f"entry {index}: metadata unreadable: {exc}"], []
        contradicted: list[str] = []
        stale: list[str] = []
        for key in DERIVED_META_KEYS:
            here, there = key in derived, key in stored
            if not here and not there:
                continue
            if here and not there:
                stale.append(
                    f"entry {index}: metadata carries no {key!r}, which the entries derive as "
                    f"{derived[key]!r}; the file predates the field, so it is stale and not "
                    "tamper — the derivation is authoritative and nothing in the entries is "
                    "contradicted (section 8.2, A-META-ABSENT)"
                )
            elif there and not here:
                contradicted.append(
                    f"entry {index}: metadata carries {key!r} = {stored[key]!r} and the entries "
                    "derive none"
                )
            elif not _exactly(derived[key], stored[key]):
                contradicted.append(
                    f"entry {index}: metadata says {key} = {stored[key]!r} while the entries "
                    f"derive {derived[key]!r}; this file is outside the tree and nothing signs "
                    "it, so it is re-derived on read and never trusted (section 8.2, A-META)"
                )
        unchecked = set(DERIVED_META_KEYS) | set(ASSERTED_META_KEYS) | set(APPEND_TIME_META_KEYS)
        for key in sorted(set(stored) - unchecked):
            contradicted.append(
                f"entry {index}: metadata carries {key!r}, which no derivation over the entries "
                f"produces; the asserted field is {ASSERTED_META_KEYS[0]!r} (section 8.1: a "
                f"timestamp is the signer's assertion) and the append-time ones are "
                f"{list(APPEND_TIME_META_KEYS)}"
            )
        return contradicted, stale

    def meta_disagreement(self, index: int) -> list[str]:
        """Every way ``<index>.meta.json`` CONTRADICTS what the entries derive. ``[]`` is no conflict.

        Reported, not raised, because a mirror wants the whole list and a verifier wants the first
        one. A key the file carries that no derivation produces is reported too — ``appended_at``
        by name as the one asserted field, and anything else as a free assertion inside a file
        readers quote (the A-OPTIONAL argument, applied to metadata).

        A key the file simply does NOT carry is not here: that is ``stale_metadata``, and the
        argument for the split is in this module's "Decisions". Every caller that refuses reads
        this list and only this list.
        """
        return self._meta_split(index)[0]

    def stale_metadata(self, index: int) -> list[str]:
        """Every key the entries derive that ``<index>.meta.json`` carries no value for.

        A file written before a derived field existed. It is a disclosure and never a refusal:
        the derivation is authoritative, the file is a cache, and nothing in the entries is
        contradicted by a silence. The lab's own published log
        (`papers/v8/first_verdict_2026_09_09/log`) is in exactly this state for ``floor``, and
        reporting it as tamper is the defect this pair of methods exists to separate.

        Nothing repairs it automatically — not here, not in ``mirror``. ``derived_meta(index)``
        is the repaired file for an operator who chooses to refresh their own cache; the reasons
        it is not done for them are in "Decisions".
        """
        return self._meta_split(index)[1]

    def _id_map(self) -> dict[str, int]:
        # The ENTRY's id, never the metadata's. It used to prefer ``meta(index)["id"]`` and fall
        # back to the cert, so an edited metadata file could point an id at another leaf and every
        # ref resolution, every floor recomputation and every challenge check downstream followed
        # it (A-META). ``verify_entry`` recomputes the id from the bytes; this reads the same bytes.
        indices = self.indices()
        if self._id_map_cache is not None and self._id_map_cache[0] == len(indices):
            return self._id_map_cache[1]
        out: dict[str, int] = {}
        for index in indices:
            try:
                cid = self.cert(index).get("id")
            except Exception:
                cid = None
            if isinstance(cid, str) and cid not in out:
                out[cid] = index
        self._id_map_cache = (len(indices), out)
        return out

    def find(self, cert_id: str) -> Optional[int]:
        """The index holding ``cert_id``, or None."""
        return self._id_map().get(cert_id)

    def issuer_policy(self) -> dict:
        """Who this log admits, read off ``keys/issuers.json``. Never raises, never returns None.

        ``{"policy": "roster", "issuers": [...]}`` — a JSON array is the roster (section 8.2's
        ``keys/issuers.json``); an empty array admits nobody, which is a policy and not an
        absence.

        ``{"policy": "open", "reason": <str|None>}`` — a JSON object naming policy ``open``: an
        intentionally open log, admitting any key. This is the marker that replaces "the file is
        missing" as the way to run an open log (L10c).

        ``{"policy": "absent"|"unreadable"|"unknown", "reason": ...}`` — every other state. Append
        refuses on all three: a security check whose unset configuration means "allow everything"
        is an off switch reachable by ``rm``.
        """
        path = self.keys_dir / "issuers.json"
        if not path.exists():
            return {"policy": "absent", "reason": f"no {path}"}
        try:
            obj = _read_json(path)
        except Exception as exc:
            return {"policy": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}
        if isinstance(obj, list):
            return {"policy": "roster", "issuers": obj}
        if isinstance(obj, dict) and obj.get("policy") == OPEN_POLICY:
            reason = obj.get("reason")
            return {
                "policy": OPEN_POLICY,
                "reason": reason if isinstance(reason, str) else None,
            }
        named = obj.get("policy") if isinstance(obj, dict) else None
        return {
            "policy": "unknown",
            "reason": f"issuers.json names policy {named!r}, which this version does not know",
        }

    def issuers(self) -> Optional[list[dict]]:
        """The roster this log admits by, or ``None`` when its policy is not a roster.

        ``None`` is no longer an admission decision — ``issuer_policy`` is (L10c). A caller that
        used this to decide whether to check anything was reading "the file is missing" as
        "everything is allowed"; append no longer has such a branch.
        """
        policy = self.issuer_policy()
        return policy["issuers"] if policy["policy"] == "roster" else None

    def log_public(self) -> Optional[bytes]:
        """The 32-byte public key in ``keys/log.pub``, or None when there is none."""
        path = self.keys_dir / "log.pub"
        if not path.exists():
            return None
        return keys.load_public(path)

    def log_id(self) -> str:
        """``"sha256:" + hex(sha256(raw log public key))`` (section 8.1)."""
        public = self.log_public()
        if public is None:
            raise AppendRefused(f"log: no public key at {self.keys_dir / 'log.pub'}")
        return "sha256:" + hashlib.sha256(public).hexdigest()

    # ------------------------------------------------------------- the tree

    def leaf_hashes(self) -> list[bytes]:
        """``sha256(0x00 || entry_bytes)`` for every entry, in index order."""
        return [merkle.leaf_hash(self.entry_bytes(i)) for i in range(self.size())]

    def root(self, tree_size: Optional[int] = None) -> bytes:
        """The Merkle tree head over the first ``tree_size`` entries (all of them by default)."""
        leaves = self.leaf_hashes()
        if tree_size is None:
            tree_size = len(leaves)
        if not isinstance(tree_size, int) or isinstance(tree_size, bool):
            raise ValueError("tree_size must be an int")
        if tree_size < 0 or tree_size > len(leaves):
            raise ValueError(f"tree_size {tree_size} is outside 0..{len(leaves)}")
        return merkle.root(leaves[:tree_size])

    # ------------------------------------------------------------- append

    def append(self, cert: dict, blobs: Optional[dict] = None) -> int:
        """Put ``cert`` in the next leaf, or raise ``AppendRefused`` saying why not.

        Refusals, in this order: the cert does not check out (``cert.check``); this log states no
        admission policy, or its issuer key is not in the roster at this index; the cert names a
        different log in ``body.log_hint``; a ref does not resolve to a logged cert of the type
        its role names; the id is already in the log; a fingerprint that already has a comparable
        fingerprint on record carries no ``previous`` ref (section 5.5); a battery or pool ref is
        redacted while this cert would be public (section 2.6, 4.5); a blob's bytes do not hash
        to its key.
        """
        if not isinstance(cert, dict):
            raise AppendRefused("invalid: cert is not a JSON object")
        index = self.size()
        present = self.indices()
        if present and present[-1] >= index:
            raise AppendRefused(
                f"entries: index {index} is missing while {present[-1]} is present; "
                "a log with a gap is not appendable"
            )

        # 1. the cert checks out
        result = certmod.check(cert)
        if not result.ok:
            raise AppendRefused("invalid: " + "; ".join(result.reasons))

        cert_id = cert.get("id")
        ctype = cert.get("type")

        # 2. the log states an admission policy, and this key satisfies it. L10c: this used to
        # read `roster = self.issuers(); if roster is not None:`, so deleting keys/issuers.json
        # admitted every key. A missing configuration is not a permission.
        policy = self.issuer_policy()
        if policy["policy"] == "roster":
            key = cert.get("issuer", {}).get("key")
            if not _in_roster(policy["issuers"], key, index):
                raise AppendRefused(
                    f"issuer: key {key} is not in the roster at index {index}"
                )
        elif policy["policy"] != OPEN_POLICY:
            raise AppendRefused(
                f"issuer: this log states no admission policy ({policy['reason']}); "
                f"{self.keys_dir / 'issuers.json'} is either a roster array or the marker "
                f'{{"policy": "{OPEN_POLICY}"}} for a log that admits any key. An absent, '
                "unreadable or unknown policy refuses every append, because a check whose "
                "unset configuration means 'allow everything' is an off switch reachable by rm "
                "(L10c)"
            )

        # 2b. a cert that names a log names THIS one (L7). See _check_log_binding.
        self._check_log_binding(cert)

        # 3. every ref resolves to a logged cert whose type matches the role. A root battery
        # (pool-v1, fixed-v1) carries no recipe.battery and therefore no ref at all, so the
        # ladder starts without needing a pass here (section 4.5); GENESIS_UNRESOLVED survives
        # only for the pre-fix pool cert one test file still builds, and its comment says so.
        genesis = (ctype, _body(cert).get("kind"))
        for role, rid in certmod.refs(cert):
            at = self.find(rid)
            if at is None:
                if (genesis[0], genesis[1], role) in GENESIS_UNRESOLVED:
                    continue
                raise AppendRefused(f"refs: {role} ref {rid} does not resolve in this log")
            # Section 9 fixes a challenge's `target` to a fingerprint; `ROLE_TYPES` alone keeps
            # it at `*` because a `response` result's target is a challenge, and that `*` let a
            # challenge name the BATTERY cert as its target and append with the subject check
            # never run (C-NONFP-TARGET). `role_type` is `ROLE_TYPES` plus the per-type table.
            want = certmod.role_type(ctype, role)
            if want is None:
                want = ctype
            if want != "*":
                got = self.cert(at).get("type")
                if got != want:
                    raise AppendRefused(
                        f"refs: {role} ref {rid} resolves to a {got} cert, not a {want} cert"
                    )

        # 4. no duplicate id
        if isinstance(cert_id, str) and self.find(cert_id) is not None:
            raise AppendRefused(f"duplicate: {cert_id} is already at index {self.find(cert_id)}")

        # 5. the baseline rule (section 5.5), and the disclosure the rule does not make.
        # The rule decides whether a second canonical fingerprint may append; it never says how
        # far the new baseline is from the one it replaces, and choosing a favourable baseline is
        # legitimate exactly as long as it is not silent (BASELINE-CHOICE). `baseline_gap` is
        # computed here from two certs already in this log and written into the entry metadata
        # below, where a reader -- and `verify` through `--log` -- sees it. It refuses nothing.
        gap: Optional[dict] = None
        if ctype == "fingerprint":
            previous = self.previous_comparable(cert)
            if previous is not None and not any(r == "previous" for r, _ in certmod.refs(cert)):
                if not _same_noise_plan(cert, self.cert(previous)):
                    raise AppendRefused(
                        f"baseline: a comparable fingerprint is at index {previous}; "
                        "a fingerprint that starts a new baseline needs a previous ref"
                    )
            try:
                gap = self.baseline_gap(cert)
            except Exception as exc:  # metadata never refuses an append the rules accepted
                gap = {"note": f"{type(exc).__name__}: {exc}"}

        # 6. a canonical fingerprint honours the plan its floor names (section 5.1 step 2), and
        # its runs are more than one execution wearing R labels. The census is kept for the
        # metadata: a floor that varied its executions and still measured 0.0 is a finding about
        # the configuration, and it is written down rather than refused.
        census: Optional[dict] = None
        if ctype == "fingerprint" and isinstance(_body(cert).get("noise_floor"), dict):
            self._check_floor_honours_its_plan(cert)
            self._check_floor_plan_precedes_its_runs(cert)
            self._check_floor_runs_are_one_subject(cert)
            self._check_floor_names_the_plans_runs(cert)
            self._check_floor_labels_match_the_recipe(cert)
            self._check_floor_covers_match_the_plan(cert)
            self._check_floor_environment_matches_the_plan(cert)
            self._check_floor_matches_its_runs(cert)
            self._check_floor_varied_an_execution(cert)
            census = self.floor_census(cert)

        # 6b. section 9 rule 1: a challenge is valid only if its own fingerprint has the target's
        # subject identity. See ``_check_challenge_subject``.
        if ctype == "challenge":
            self._check_challenge_subject(cert)

        # 7. redaction (section 2.6, 4.5)
        self_redacted = _redacted(cert)
        declares = bool(_body(cert).get("redacted_battery"))
        for role, rid in certmod.refs(cert):
            if role not in ("battery", "pool"):
                continue
            at = self.find(rid)
            if at is None:
                continue
            if _redacted(self.cert(at)) and not (self_redacted or declares):
                raise AppendRefused(
                    f"redaction: {role} ref {rid} is redacted while this cert would be public"
                )

        # 8. blobs are content-addressed
        payload: dict[str, bytes] = {}
        for key, value in (blobs or {}).items():
            if isinstance(value, (bytearray, memoryview)):
                value = bytes(value)
            if not isinstance(value, bytes):
                raise AppendRefused(f"blob: {key} is {type(value).__name__}, not bytes")
            try:
                want_hex = _blob_hex(key)
            except ValueError as exc:
                raise AppendRefused(f"blob: {exc}") from None
            got_hex = hashlib.sha256(value).hexdigest()
            if got_hex != want_hex:
                raise AppendRefused(f"blob: bytes for {key} hash to {got_hex}")
            payload[want_hex] = value

        # 9. write: blobs, then the entry, then the metadata
        entry = canonical_bytes(cert)
        for hex_key, value in sorted(payload.items()):
            _write_bytes(self.blobs_dir / f"{hex_key}.json", value)
        _write_bytes(self.entry_path(index), entry)
        meta: dict[str, Any] = {
            "index": index,
            "id": cert_id,
            "type": ctype,
            "public": self._public_for(cert),
            "appended_at": _now(),
        }
        if census is not None:
            meta["floor"] = census
        if gap is not None:
            meta["baseline_gap"] = gap
        _write_json(self.meta_path(index), meta)
        self._id_map_cache = None
        return index

    def _check_log_binding(self, cert: dict) -> None:
        """Refuse a cert whose ``body.log_hint`` names a log that is not this one (L7).

        Seven entries of the published log were replayed verbatim into a log built on a different
        key: all seven appended, byte for byte, and gave the same Merkle root under a different
        ``log_id``. Nothing in a cert said which log it belonged to, so "this cert is in the log"
        was not a checkable statement.

        ``body.log_hint`` is optional and signed. Optional, because requiring it would refuse
        every cert already signed and because a cert cannot name a log that does not exist yet;
        signed, because ``body`` is inside ``D`` (section 2.1), which is the difference between
        this and the advisory field ``DURABILITY`` C2 proposes — an advisory field is exactly what
        a replay walks through. The shape is ``{log_id: "sha256:<hex>", locations?: [...]}``;
        ``log_id`` is section 8.1's hash of the raw log public key.

        The module docstring argues the trade this makes (a bound cert cannot be re-logged
        elsewhere, which is provenance to a reader and a custody problem to an archivist) and
        names the spec edit it owes (``cert.NON_REF_HASH_PATHS`` gains a third member, where A.1
        says there is no third).
        """
        hint = _body(cert).get("log_hint")
        if hint is None:
            return
        if not isinstance(hint, dict):
            raise AppendRefused(
                f"log_hint: body.log_hint is {type(hint).__name__}, not an object; the binding a "
                "cert makes to its log is {log_id, locations?} or it is nothing (section 8.1)"
            )
        named = hint.get("log_id")
        if not isinstance(named, str) or not _ID_PATTERN.match(named):
            raise AppendRefused(
                f"log_hint: body.log_hint.log_id is {named!r}, which is not sha256:<64 hex>; a "
                "cert that names its log names it in the form section 8.1 gives log_id, or the "
                "binding cannot be compared with anything"
            )
        mine = self.log_id()  # AppendRefused when this log has no key of its own
        if named != mine:
            raise AppendRefused(
                f"log_hint: this cert names log {named} and this log is {mine}; a cert bound to a "
                "log is not seated in another one, which is what makes 'this cert is in the log' "
                "a statement about a log and not about bytes that appear in some log (L7, "
                "section 8.1)"
            )

    def _check_floor_plan_precedes_its_runs(self, cert: dict) -> None:
        """Refuse a floor whose noise plan sits at a HIGHER index than a run resting on it (L6).

        Section 7.2: *the log index is the proof of order*. Nothing compared the two. Section 5.1
        step 5 (ii) has each run reference the plan under role ``noise_plan``, and refs resolve
        backwards, so a run that carries the ref cannot precede its plan — but a run cert that
        omits the ref carries no such constraint, and chaining the runs by ``previous`` satisfies
        section 5.5 without it. Append the runs, mint the plan afterwards, then the canonical that
        names both, and the floor rests on a preregistration written after the data existed.

        What this does NOT establish, since section 8.6 already says it: the index orders the
        appends, not the computations, and the log cannot tell a late-logged prereg from an early
        one. This rule is worth exactly its own sentence — a floor may not rest on runs the log
        accepted before it accepted their plan.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return
        plan_id = block.get("plan")
        plan_at = self.find(plan_id) if isinstance(plan_id, str) else None
        if plan_at is None:
            return  # _check_floor_honours_its_plan refuses an unresolvable plan first
        run_ids = block.get("runs")
        for rid in run_ids if isinstance(run_ids, list) else []:
            at = self.find(rid) if isinstance(rid, str) else None
            if at is None:
                continue  # _floor_run_certs refuses an unresolvable run
            if at < plan_at:
                raise AppendRefused(
                    f"floor: the noise plan {plan_id} is at index {plan_at} and run {rid} it "
                    f"governs is at index {at}; the log index is the proof of order (section "
                    "7.2), so a plan appended after its own runs preregisters nothing -- it is a "
                    "document written once the numbers were in, and section 5.1 step 1 exists to "
                    "stop the issuer choosing its nuisance set after seeing them (L6)"
                )

    def _check_floor_honours_its_plan(self, cert: dict) -> None:
        """Refuse a canonical fingerprint whose runs did not vary what its plan declared.

        See the module docstring under "Decisions" for the predicate and the receipt. Everything
        this reads is already in the log: the plan cert, the R−1 run certs, and the appended
        cert's own ``body.nuisance``.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return

        plan_id = block.get("plan")
        at = self.find(plan_id) if isinstance(plan_id, str) else None
        if at is None:
            raise AppendRefused(
                f"floor: the noise plan {plan_id} does not resolve in this log; a floor whose plan "
                "is not on record cannot be checked against it (section 5.1 step 1)"
            )
        declared = _declared_factors(self.cert(at))
        if not declared:
            return

        assignments = [_nuisance_of(cert)]
        run_ids = block.get("runs")
        for rid in run_ids if isinstance(run_ids, list) else []:
            index = self.find(rid) if isinstance(rid, str) else None
            if index is None:
                raise AppendRefused(
                    f"floor: run {rid} does not resolve in this log; the runs of a floor are "
                    "appended before the canonical fingerprint (section 5.1 step 5)"
                )
            assignments.append(_nuisance_of(self.cert(index)))
        total = len(assignments)

        for factor, values in declared:
            if len(values) < 2:
                continue
            observed = [str(a[factor]) for a in assignments if factor in a]
            allowed = sorted({str(v) for v in values})
            if not observed:
                raise AppendRefused(
                    f"floor: the plan declares nuisance factor {factor!r} with values {allowed} "
                    f"and not one of the {total} runs records it; a plan is a commitment the runs "
                    "honour, not a description they may ignore (section 5.1 step 2)"
                )
            if len(observed) != total:
                raise AppendRefused(
                    f"floor: the plan declares nuisance factor {factor!r} and only "
                    f"{len(observed)} of the {total} runs record it; a floor whose runs disagree "
                    "about what was held fixed is not one measurement (section 5.1 step 2)"
                )
            if len(set(observed)) < 2:
                raise AppendRefused(
                    f"floor: the plan declares nuisance factor {factor!r} with values {allowed} "
                    f"and all {total} runs ran at {observed[0]!r}; a factor declared and never "
                    "varied is measured by nothing, and the floor carries no information about it "
                    "(section 5.1 step 2)"
                )
            stray = sorted(set(observed) - set(allowed))
            if stray:
                raise AppendRefused(
                    f"floor: the runs record nuisance factor {factor!r} at {stray}, which the "
                    f"plan did not declare (it declared {allowed}); the plan fixes the values "
                    "before the runs (section 5.1 step 1)"
                )

    def _check_floor_covers_match_the_plan(self, cert: dict) -> None:
        """Refuse a floor whose coverage lists are not the function of the plan it names.

        A-COVER. ``noise_floor.covers`` and ``noise_floor.not_covered`` were the last two numerals
        in a floor block that nothing re-derived: ``_check_floor_matches_its_runs`` anchors
        ``per_channel`` and section 5.7's overall size to the run certs, and neither list was ever
        compared against the plan the floor names by id. So a canonical signed with ``covers``
        naming the gpu, the driver and everything else and ``not_covered: []`` appended at exit 0
        beside an honest per-channel floor -- and since ``verify`` reads ``coverage_diff`` over
        ``not_covered`` alone (``verify._coverage_diff``), an empty ``not_covered`` is the claim
        that this floor covers every environment there is. Section 5.4 spends ``covers`` on
        deciding whether an outside reproduction is a dispute or a coverage report (section 9), so
        an issuer choosing it after the runs chooses who is allowed to contradict it.

        ``floor.plan_coverage`` is the derivation and it is the same one ``cli`` mints with, so an
        honest floor's two lists are re-derivable by any reader holding the plan cert.

        **A plan that fixes no environment is refused here rather than skipped**, for the reason
        A-NORUNS is refused in ``_check_floor_names_the_plans_runs``: with no environment the
        derivation gives ``not_covered = []``, which is the permissive claim itself, so a branch
        that returned quietly would hand back the whole attack for the price of omitting one key.
        Section 5.1 step 1 has the plan name the environment; a prereg that does not is not a plan
        a floor may rest on. ``schema/prereg.json`` does NOT require the key -- see the report --
        so this is the only side that refuses it.

        LIMIT, written down: the plan's ``nuisance`` and ``environment`` are still bytes one party
        wrote. This moves the choice of coverage from after the runs to before them and pins it to
        a cert signed earlier by id; it does not make the plan's environment true. That half is
        ``_check_floor_environment_matches_the_plan`` for the runs, and beyond that the label class
        of ``papers/v8/THE_BOUNDARY_2026_09_09.md``.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return
        plan = self._floor_plan_cert(cert)
        if plan is None:
            return  # _check_floor_honours_its_plan refuses an unresolvable plan first
        plan_body = _body(plan)
        environment = plan_body.get("environment")
        if not isinstance(environment, dict) or not environment:
            raise AppendRefused(
                f"floor: the plan {plan.get('id')} names environment {environment!r}, which fixes "
                "no environment field; section 5.1 step 1 has the plan name the environment BEFORE "
                "the runs, and section 5.4 derives `not_covered` from it -- a plan that fixes none "
                "derives an empty `not_covered`, which is the claim that this floor covers every "
                "environment there is (A-COVER, section 5.4)"
            )
        covers, not_covered = floormod.plan_coverage(plan_body)
        for name, derived in (("covers", covers), ("not_covered", not_covered)):
            signed = block.get(name)
            if signed == derived:
                continue
            raise AppendRefused(
                f"floor: noise_floor.{name} is signed as {signed!r} while the plan "
                f"{plan.get('id')} this floor names gives {derived!r}; section 5.4 defines "
                "`covers` as the nuisance factors the plan varied and `not_covered` as the "
                "environment fields it held fixed, so both are functions of the plan's own "
                "`nuisance` and `environment` and neither is a list the appending cert may write "
                "-- `covers` decides whether an outside reproduction binds (sections 5.4, 9), and "
                "an issuer that picks it after the runs picks who may contradict it (A-COVER)"
            )

    def _check_floor_environment_matches_the_plan(self, cert: dict) -> None:
        """Refuse a floor whose runs did not happen in the environment its plan fixed.

        PLAN-ENV, the half that makes the plan's environment mean something. ``prereg noise-plan``
        now observes the environment it names through a runner (section 2.2's mint rule, which the
        plan did not get when the fingerprint did), so the plan names an environment that was
        observed on the box that minted it. That is a statement about the plan's box and the runs
        are a separate process, so it binds nothing until the runs are compared against it -- which
        is this check.

        The predicate: for every leaf of the plan's ``environment`` that the plan did not declare
        as a nuisance factor, every cert the floor rests on (this one and the runs
        ``noise_floor.runs`` names) carries the same value at that dotted path in
        ``subject.environment``. A leaf the run does not carry is a refusal, not a pass: section
        5.1 step 2 has the runs vary the planned factors and nothing else, and a run whose
        environment is silent about a field the plan fixed did not hold it fixed, it failed to say.

        The comparison is one-directional on purpose. The plan names the environment it fixed; a
        run may carry leaves the plan never mentioned (a ``harness`` block, an ``env_lock_sha256``)
        and those are section 2.3's skew fields, not coverage.

        LIMIT: both sides are the same issuer's bytes and the plan's environment is observed on the
        plan's box, not on the runs'. A party that mints its plan and its runs on one box and
        writes the same environment into both is not caught by this and cannot be -- it is the
        label class. What this removes is the gap where the two disagreed and nothing looked.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return
        plan = self._floor_plan_cert(cert)
        if plan is None:
            return
        plan_body = _body(plan)
        environment = plan_body.get("environment")
        if not isinstance(environment, dict) or not environment:
            return  # _check_floor_covers_match_the_plan owns that refusal
        held = {f for f, _ in _declared_factors(plan)}
        paths = [
            p
            for p in floormod.environment_paths(environment)
            if p not in held and p.split(".")[0] not in held
        ]
        for run in self._floor_run_certs(cert):
            run_env = (run.get("subject") or {}).get("environment")
            for path in paths:
                want = _at_path(environment, path)
                got = _at_path(run_env, path)
                if got is _ABSENT:
                    raise AppendRefused(
                        f"floor: the plan {plan.get('id')} fixed environment {path} at {want!r} "
                        f"and run {run.get('id')} records no such field in subject.environment; "
                        "section 5.1 step 2 has the R runs vary the planned nuisance factors and "
                        "nothing else, and a run silent about a field the plan fixed did not hold "
                        "it fixed -- it failed to say (PLAN-ENV, sections 2.2, 5.1)"
                    )
                if got != want:
                    raise AppendRefused(
                        f"floor: the plan {plan.get('id')} fixed environment {path} at {want!r} "
                        f"and run {run.get('id')} ran at {got!r}; the plan names the environment "
                        "before the runs and the runs vary only the factors it declared, so a "
                        "floor measured somewhere else is a floor of a different quantity than the "
                        "plan describes (PLAN-ENV, sections 5.1 step 2, 5.4)"
                    )

    def _floor_plan_cert(self, cert: dict) -> Optional[dict]:
        """The noise-plan cert ``body.noise_floor.plan`` names, or ``None`` when it is not here.

        ``_check_floor_honours_its_plan`` refuses an unresolvable plan before any caller of this
        reaches it, so ``None`` at append time means "this cert carries no floor".
        """
        block = _body(cert).get("noise_floor")
        plan_id = block.get("plan") if isinstance(block, dict) else None
        at = self.find(plan_id) if isinstance(plan_id, str) else None
        return None if at is None else self.cert(at)

    def _floor_declared_runs(self, cert: dict) -> Optional[int]:
        """``R`` as the floor's own plan fixed it, or ``None`` when the plan fixes no ``R``.

        This is the one number in the whole floor that was committed BEFORE the runs, on a cert
        signed earlier and named by id from inside the signed floor block. It is what the split
        below is decided by, so that the decision is not a signature the appending party writes.
        """
        plan = self._floor_plan_cert(cert)
        declared = _body(plan).get("runs") if isinstance(plan, dict) else None
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 2:
            return None
        return declared

    def _check_floor_runs_are_one_subject(self, cert: dict) -> None:
        """Refuse a floor whose runs are not runs of the SAME subject under the same recipe core.

        A-SWAP, the sharpest finding of the third adversarial pass. ``_check_floor_matches_its_runs``
        recomputes a floor's arithmetic and never asks WHOSE runs those are, and neither did
        ``_check_floor_honours_its_plan`` or ``_check_floor_names_the_plans_runs``. So an fp16
        fingerprint, relabelled ``run_index 1`` and carrying a bf16 run's ``nuisance`` and
        ``recipe``, could be named as a run of a bf16 canonical's floor: the anchor recomputed the
        floor over the fp16 body, agreed with it to the last digit, ``floor_disagreement``
        returned ``[]``, and every channel of a later ``verify --diff`` read ``same`` — because the
        cross-precision drift WAS the floor, and the recomputation certified it.

        Section 5.3 is the sentence being enforced: a floor is measured on one subject and says
        nothing about another. So every run a floor rests on must carry

        * the same ``S_identity`` (section 2.2, ``cert.identity_fields``) as the canonical that
          names it — ``precision`` and ``revision`` included. ``cert.comparable`` softens those two
          to ``cross-subject:`` because a cross-subject *comparison* is legitimate; a cross-subject
          *run of one floor* is not, which is why this compares identity fields directly rather
          than reading ``comparable``'s verdict;
        * the same ``cert.SYNTHETIC`` marker — a fabricated distance and a measured one are not two
          runs of one measurement;
        * the same ``recipe_core`` (section 2.3) **except** on the ``decoding`` keys a floor is
          allowed to move: ``_RECIPE_FACTORS`` (``batch_size``, ``padding_side`` — the execution
          knobs section 5.1 step 2 hands the runs, and the two ``_check_floor_labels_match_the_recipe``
          already governs), plus any other ``decoding`` key the floor's own plan declared as a
          nuisance factor with two or more values. That exception is not a softening: varying those
          keys is what a floor IS, and the lab's own published floor has runs at
          ``decoding.batch_size`` 8 and 32 under a canonical at 1
          (``papers/v8/first_verdict_2026_09_09``). Every other recipe_core difference — the
          battery, the chat template, the system prompt, or a decoding key like ``seed``,
          ``temperature`` or ``max_new_tokens`` that no plan declared — is a different recipe, and
          a floor across two recipes measures the recipe.

        The certs compared are the ones THIS LOG HOLDS at the ids the block names
        (``_floor_run_certs``), not copies carried by the appending cert.

        What this does not reach: two runs of the same subject and recipe that were never executed
        as claimed. See ``_check_floor_labels_match_the_recipe`` for that boundary.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return
        own_subject = certmod.identity_fields(cert.get("subject", {}))
        own_core = certmod.recipe_core(cert.get("recipe", {}))
        plan = self._floor_plan_cert(cert)
        licensed = set(_RECIPE_FACTORS) | {
            factor
            for factor, values in (_declared_factors(plan) if isinstance(plan, dict) else [])
            if len(values) >= 2
        }
        for run in self._floor_run_certs(cert)[1:]:
            rid = run.get("id")
            subject = certmod.identity_fields(run.get("subject", {}))
            if subject != own_subject:
                differ = sorted(
                    k for k in set(subject) | set(own_subject)
                    if subject.get(k) != own_subject.get(k)
                )
                raise AppendRefused(
                    f"floor: run {rid} is a run of a different subject; it and this cert differ on "
                    f"S_identity {differ} ({[subject.get(k) for k in differ]} against "
                    f"{[own_subject.get(k) for k in differ]}); a floor is a statement about ONE "
                    "subject, and a floor measured over another subject's runs is that subject's "
                    "drift wearing this one's name (section 2.2, section 5.3)"
                )
            if certmod.is_synthetic(run) != certmod.is_synthetic(cert):
                raise AppendRefused(
                    f"floor: run {rid} carries {certmod.SYNTHETIC}="
                    f"{certmod.is_synthetic(run)} while this cert carries "
                    f"{certmod.is_synthetic(cert)}; a fabricated distance and a measured one are "
                    "not two runs of one measurement (section 5.3)"
                )
            core = certmod.recipe_core(run.get("recipe", {}))
            for field in certmod.RECIPE_CORE_FIELDS:
                if core[field] == own_core[field]:
                    continue
                if field != "decoding":
                    raise AppendRefused(
                        f"floor: run {rid} names recipe.{field} {core[field]!r} while this cert "
                        f"names {own_core[field]!r}; a floor is a statement about one subject "
                        "under ONE recipe core, and runs under two recipes measure the recipe "
                        "(section 2.3, section 5.1 step 2)"
                    )
                mine = own_core["decoding"] if isinstance(own_core["decoding"], dict) else None
                theirs = core["decoding"] if isinstance(core["decoding"], dict) else None
                if mine is None or theirs is None:
                    raise AppendRefused(
                        f"floor: run {rid} and this cert do not both carry a recipe.decoding "
                        "object, so the execution they ran under cannot be compared (section 2.3)"
                    )
                stray = sorted(
                    k for k in set(mine) | set(theirs)
                    if mine.get(k) != theirs.get(k) and k not in licensed
                )
                if stray:
                    raise AppendRefused(
                        f"floor: run {rid} names recipe.decoding {stray} differently from this "
                        f"cert ({[theirs.get(k) for k in stray]} against "
                        f"{[mine.get(k) for k in stray]}) and the plan declared no such nuisance "
                        f"factor (it declared {sorted(licensed)}); a run may vary what the plan "
                        "committed to and nothing else, or the floor measures the recipe "
                        "(section 2.3, section 5.1 steps 1 and 2)"
                    )

    def _floor_run_certs(self, cert: dict) -> list[dict]:
        """``[the appending cert] + the certs ``noise_floor.runs`` names``, in the block's order.

        That is the order ``fingerprint.attach_floor`` computed the floor in (canonical first),
        so it is the order the floor is re-derivable in. A named run this log does not hold is a
        refusal and not a shorter list: recomputing over whatever happens to resolve is a check
        that passes hardest when the evidence is missing.
        """
        block = _body(cert).get("noise_floor")
        run_ids = block.get("runs") if isinstance(block, dict) else None
        certs = [cert]
        for rid in run_ids if isinstance(run_ids, list) else []:
            at = self.find(rid) if isinstance(rid, str) else None
            if at is None:
                raise AppendRefused(
                    f"floor: run {rid} does not resolve in this log; the runs of a floor are "
                    "appended before the canonical fingerprint (section 5.1 step 5)"
                )
            certs.append(self.cert(at))
        return certs

    @staticmethod
    def _floor_run_zero(cert: dict, named: Sequence[dict]) -> list[dict]:
        """The named runs claiming to BE this cert's own run: same ``body.run_index``.

        Index only. Whether such a cert really is run 0 — same ``items``, same ``channels`` — is
        ``_check_floor_names_the_plans_runs``'s question, and it is a refusal there rather than a
        silent non-match here, because a non-match is what used to move the computation.
        """
        mine = _body(cert).get("run_index")
        return [c for c in named if _body(c).get("run_index") == mine]

    @staticmethod
    def _floor_body_certs(
        cert: dict, named: Sequence[dict], declared_runs: Optional[int] = None
    ) -> list[dict]:
        """The certs whose BODIES the floor was computed over, in ``attach_floor``'s order.

        ``attach_floor`` accepts ``run_ids`` in two lengths and both are in use:

        * ``R−1`` ids — section 5.1 step 5's append order, where the canonical IS run 0 and run 0
          has no cert of its own. The bodies are ``[this cert's] + [each named run's]``.
        * ``R`` ids — run 0 was also logged as its own fingerprint, and this cert's body is that
          run's body plus the floor block. The bodies are the named runs' alone; prepending this
          cert would count run 0 twice.

        WHICH SHAPE THIS IS, IS DECIDED BY ``R`` (A-SPLIT). It used to be decided by a signature
        the appending party writes: "some named run carries my ``run_index`` and my ``items``".
        The attacker's point was that flipping that signature flips which bodies the anchor
        computes over — and, because ``_check_floor_names_the_plans_runs`` counted the post-split
        list, moves the completeness check along with it. ``R`` is the one number here that was
        committed before the runs, on the noise-plan cert this floor names by id, so ``R`` decides:
        ``len(named) == R`` is the R-id shape and ``len(named) == R−1`` is the R−1 shape. Nothing
        the appending cert says about itself takes part.

        ``declared_runs=None`` — a hand-signed plan that omits ``runs`` and fixes no ``R``, which
        ``_check_floor_names_the_plans_runs`` also cannot check — falls back to the old signature,
        and that residue is written down in the module docstring rather than papered over.

        Getting this wrong is not silent either way: the recomputation below is over exactly the
        bodies this returns, so a wrong split moves ``runs``, ``pairs`` and every distance.
        """
        if isinstance(declared_runs, int) and not isinstance(declared_runs, bool):
            if len(named) == declared_runs:
                return list(named)
            if len(named) == declared_runs - 1:
                return [cert] + list(named)
        own = _body(cert)
        mine = own.get("run_index")
        if any(
            _body(c).get("run_index") == mine and _body(c).get("items") == own.get("items")
            for c in named
        ):
            return list(named)
        return [cert] + list(named)

    def _check_floor_names_the_plans_runs(self, cert: dict) -> None:
        """Refuse a floor that rests on a chosen subset of the runs the plan committed to, and a
        floor whose two possible shapes are not the same measurement.

        ``R`` is fixed in the plan before the runs; the floor may otherwise be chosen after them.
        See the module docstring under "Decisions" (F7).

        **A resolvable plan that fixes no usable ``R`` is refused, not skipped** (A-NORUNS). This
        branch used to return silently when the plan's ``body.runs`` was absent or not an int
        >= 2, which turned the subset check off for the price of deleting one key from a
        hand-signed plan: ten subsets of the lab's own five published runs then appended with an
        empty ``floor_disagreement`` and ``topk`` floors from 0.813925214 to 2.140233900, which
        is the whole of F7 again. ``schema/prereg.json`` now requires ``runs`` on a noise-plan,
        so a plan reaching an append cannot lack it; this side is written as a refusal anyway,
        for the reason the anchor refuses an unresolvable run rather than recomputing over what
        is left -- a check whose quietest state is "the commitment was missing" is not a check.

        The second half is A-SPLIT. With the shape now taken from ``R`` (``_floor_body_certs``),
        two states remain in which the appending party could still choose what the anchor
        computes over, and both are refused rather than resolved:

        * ``len(named) == R`` and no named run carries the canonical's ``run_index`` — the floor
          would rest on R bodies none of which is run 0, i.e. the canonical's own numbers are not
          in its own floor. Two such runs would also be an unreadable log for anyone counting.
        * ``len(named) == R`` and the named run 0's body is not the canonical's body *as the floor
          reads it* — ``floor.floors`` reads exactly ``items`` and ``channels``, so requiring those
          two to be equal makes the split numerically inert: whichever of the two bodies the anchor
          takes, every distance, every ``floor`` and every ``alpha_single`` is the same number.
          That is the property that closes A-SPLIT — not a better guess at the appender's intent,
          but the removal of anything for the guess to change.
        * ``len(named) == R−1`` and some named run claims run 0 — run 0 would be counted twice.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return
        declared = self._floor_declared_runs(cert)
        if declared is None:
            if self._floor_plan_cert(cert) is None:
                return  # _check_floor_honours_its_plan refuses an unresolvable plan first
            raise AppendRefused(
                "floor: the plan this floor names declares runs "
                f"{_body(self._floor_plan_cert(cert)).get('runs')!r}, which fixes no R; section "
                "5.1 step 1 has the plan name the run count BEFORE the runs, so a floor whose "
                "plan fixed no count rests on a sample the issuer chose after seeing the numbers "
                "-- this branch used to return silently here, and deleting one key from a "
                "hand-signed plan turned the subset check off (A-NORUNS; schema/prereg.json now "
                "requires runs on a noise-plan for the same reason)"
            )
        named = self._floor_run_certs(cert)[1:]
        have = len(self._floor_body_certs(cert, named, declared))
        if have != declared:
            raise AppendRefused(
                f"floor: the plan fixed R = {declared} runs and this floor rests on {have} of "
                f"them (the bodies of the {len(named)} certs `noise_floor.runs` names, plus this "
                "cert's own when run 0 has no separate cert); R is committed before the runs, so "
                "a floor over a subset is a sample the issuer chose after seeing the numbers, "
                "and the floor it publishes is the one it liked (section 5.1 steps 1 and 3)"
            )
        own = _body(cert)
        zero = self._floor_run_zero(cert, named)
        if len(named) == declared - 1:
            if zero:
                raise AppendRefused(
                    f"floor: the plan fixed R = {declared} runs, this floor names {len(named)} of "
                    f"them so this cert is run {own.get('run_index')!r} itself, and run "
                    f"{zero[0].get('id')} claims that index too; run 0 would be counted twice and "
                    "the pair count the floor advertises would be a count of R+1 runs "
                    "(section 5.1 step 5)"
                )
            return
        if len(zero) != 1:
            raise AppendRefused(
                f"floor: the plan fixed R = {declared} runs and this floor names all {declared}, "
                f"so one of them is run {own.get('run_index')!r} — and {len(zero)} of them carry "
                "that index; which bodies the floor is over must be a fact about the plan and the "
                "run certs, not a choice left to the cert being appended (section 5.1 steps 3 "
                "and 5)"
            )
        run_zero = _body(zero[0])
        for field in ("items", "channels"):
            if run_zero.get(field) == own.get(field):
                continue
            raise AppendRefused(
                f"floor: run {zero[0].get('id')} is named as run {own.get('run_index')!r} of this "
                f"floor, which is this cert's own run, and its body.{field} is not this cert's "
                f"body.{field}; the floor is a function of `items` and `channels` alone, so two "
                "bodies claiming one run must agree on both or the floor depends on which of them "
                "is counted (Appendix B, section 5.1 step 3)"
            )

    def _check_floor_labels_match_the_recipe(self, cert: dict) -> None:
        """Refuse a floor whose runs label an execution their own recipe does not record.

        ``fingerprint.execution_state`` chunks ``body.items`` by ``body.nuisance.batch_size``;
        that label is what turns one computation into several "runs" if it is free (R-EXEC). The
        mint already writes the plan's ``batch_size`` and ``padding_side`` into each run's
        ``recipe.decoding`` — "a cert never records an execution its run did not use" (``cli``) —
        so the log requires the two to agree. See the module docstring under "Decisions" for what
        this buys and what it does not.

        R-EXEC-3, AND THE LIMIT OF THIS CHECK, WRITTEN DOWN RATHER THAN DEFENDED AGAINST. The
        attacker relabelled ``recipe.decoding.batch_size`` alongside ``body.nuisance.batch_size``
        and passed: four certs that all really ran at batch 1 declared 8, 8, 32 and 1, the census
        counted three executions, and every number the floor advertises was the number of one
        computation. **This check cannot detect that, and no check written here can.** Both halves
        are fields of one cert, signed by one key, written by the party that also chose what to
        write. A predicate over those bytes can only ever ask whether the party contradicted
        itself; a party that does not contradict itself is not caught by asking.

        What this check is, then, stated exactly: it makes a relabel COST something. A cert whose
        ``nuisance`` and ``decoding`` were forged together has a different ``recipe_core``
        (section 2.3), so — since ``_check_floor_runs_are_one_subject`` now requires every run's
        recipe_core to match the canonical's outside the plan's declared factors — the forgery has
        to be committed to in the plan, before the runs, and it changes what the resulting cert is
        comparable to. It is a tax, not a wall.

        What would actually detect R-EXEC-3: a second party running the same battery under the
        same plan and publishing its own floor. One computation relabelled R ways produces a floor
        of 0.0 (or of whatever the single execution's self-distance is); a real R-run floor on the
        same subject and battery does not, and the two floors disagree in public. That is a
        replication, not a predicate, and it lives outside this module — ``REPLICATIONS.md``, and
        section 9's challenge, are where it lands. Nothing in these bytes substitutes for it, and
        a check invented here that appeared to would be worse than the gap.
        """
        for run in self._floor_run_certs(cert):
            nuisance = _nuisance_of(run)
            decoding = _decoding_of(run)
            for factor in _RECIPE_FACTORS:
                label, ran = nuisance.get(factor), decoding.get(factor)
                if label is None or ran is None or str(label) == str(ran):
                    continue
                raise AppendRefused(
                    f"floor: run {run.get('id')} records nuisance {factor} {label!r} while its "
                    f"own recipe.decoding says {ran!r}; the execution a floor counts is derived "
                    f"from {factor}, so a cert whose label and recipe disagree relabels one "
                    "computation as several (section 5.1 step 2, section 2.3)"
                )

    def _check_floor_matches_its_runs(self, cert: dict) -> None:
        """Refuse a floor whose signed numbers are not the ones its own runs produce.

        THE ANCHOR. See the module docstring under "Decisions": the floor is the Appendix B
        function of bodies this log holds, so this recomputes it with ``floor.floors`` and
        compares exactly. Every refusal names the channel, the key and both numbers.
        """
        self._floor_run_certs(cert)  # a named run this log does not hold is its own refusal
        for reason in self.floor_disagreement(cert):
            raise AppendRefused(reason)

    def floor_disagreement(self, cert: dict) -> list[str]:
        """Every way ``cert``'s signed floor differs from the floor its named runs produce.

        ``[]`` means the numerals are re-derivable from the bytes this log holds. Public because
        a reader who did not run the append needs the same computation over the same certs; it
        appends nothing and decides nothing. With runs this log does not hold the list carries
        one entry saying so.
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return []
        try:
            certs = self._floor_run_certs(cert)
        except AppendRefused as exc:
            return [exc.reason]

        signed = block.get("per_channel")
        if not isinstance(signed, dict):
            return [
                "floor: noise_floor.per_channel is not an object, so there is nothing to "
                "re-derive; a floor is a measurement over the runs it names (section 5.1 step 3)"
            ]

        bodies = [
            _body(c)
            for c in self._floor_body_certs(cert, certs[1:], self._floor_declared_runs(cert))
        ]
        roles = _floor_roles(_body(cert))
        try:
            derived = floormod.floors(bodies, roles=roles or None)
        except (TypeError, ValueError) as exc:
            return [
                "floor: the floor cannot be recomputed from the runs it names "
                f"({exc}); a number this log cannot re-derive from its own bytes is a claim and "
                "not a measurement (Appendix B, section 5.1 step 3)"
            ]

        out: list[str] = []
        want = {ch for ch, blk in derived.items() if blk is not None}
        got = {ch for ch, blk in signed.items() if isinstance(blk, dict)}
        if got != want:
            out.append(
                f"floor: the signed floor carries channels {sorted(got)} while the {len(bodies)} "
                f"runs it names carry {sorted(want)}; a channel absent on a run has no floor "
                "(section 3.2, Appendix B)"
            )
        for channel in sorted(want & got):
            mine, theirs = derived[channel], signed[channel]
            # The signed channel block IS the derived one: every key the derivation produces,
            # no key it does not. A key the anchor never reads is a key the issuer writes freely,
            # and A-OPTIONAL put `{"agrees": 999, "note": "measured over 500 runs"}` inside every
            # channel block of an otherwise honest floor and appended at exit 0.
            for key in sorted(mine):
                if key not in theirs:
                    out.append(
                        f"floor: channel {channel!r} is signed without {key}, which the "
                        f"{len(bodies)} run certs it names give as {mine[key]!r}; an absent key "
                        "is not an agreeing key, and the derivation is free (Appendix B, "
                        "section 5.1 step 3)"
                    )
                elif not _exactly(mine[key], theirs[key]):
                    out.append(
                        f"floor: channel {channel!r} is signed with {key} {theirs[key]!r} while "
                        f"the {len(bodies)} run certs it names give {mine[key]!r}; the floor is a "
                        "function of the runs the cert points at and this log recomputes it, "
                        "exactly (Appendix B, section 5.1 step 3)"
                    )
            extra = sorted(set(theirs) - set(mine))
            if extra:
                out.append(
                    f"floor: channel {channel!r} is signed with {extra} beside the keys the "
                    "derivation produces; nothing recomputes those, so they are the issuer's "
                    "assertion sitting inside a measured block (A-OPTIONAL, section 5.1 step 3)"
                )
        if out or not want or want != got:
            return out

        # Section 5.7's overall size is the same derivation over the same bodies, so it is
        # REQUIRED, not re-derived-when-offered. It used to be checked `if key in block`: a cert
        # with `alpha_overall` and `standardized_max` deleted appended with an empty disagreement
        # list and section 5.7's overall size simply absent (A-OPTIONAL). Requiring the KEY in
        # schema/fingerprint.json was the other option and it is the weaker one -- a schema can
        # say the number is there and cannot say it is the number the runs give, which is a
        # mandatory field in front of the same hole. The anchor already owns the sentence "the
        # signed floor is the derived floor" for per_channel; 5.7 is that same derivation over
        # those same bodies, so it belongs in the same sentence rather than in a second file.
        # `alpha_overall_method` and `standardization` are compared too: they are constants this
        # implementation derives, they are what a reader re-derives the fraction BY, and left
        # unread they were two strings an issuer could set to any procedure it liked.
        try:
            size = fpmod._overall_size(bodies, {ch: derived[ch] for ch in want}, roles)
        except (TypeError, ValueError) as exc:
            return [
                f"floor: alpha_overall cannot be recomputed from the runs it names ({exc}) "
                "(section 5.7)"
            ]
        for key in sorted(size):
            if key not in block:
                out.append(
                    f"floor: the signed floor does not carry {key}, which the {len(bodies)} run "
                    f"certs it names give as {size[key]!r}; section 5.7's overall size is derived "
                    "from the same runs as the per-channel floors and is not optional -- a floor "
                    "published without it is a floor whose overall size nobody stated "
                    "(A-OPTIONAL)"
                )
            elif not _exactly(size[key], block[key]):
                out.append(
                    f"floor: {key} is signed as {block[key]!r} while the {len(bodies)} run "
                    f"certs the floor names give {size[key]!r}; section 5.7's overall size "
                    "is derived from the same runs and this log recomputes it"
                )
        return out

    def _floor_parts(self, cert: dict) -> Optional[tuple[list[dict], list[tuple[str, list]]]]:
        """``(the A run certs, the plan's declared factors)`` for a floor, or ``None``.

        ``None`` when the cert carries no floor, or when a run or the plan is not in this log —
        both of which ``_check_floor_honours_its_plan`` refuses first, so ``None`` here means
        "not a floor" and never "a floor I could not read".
        """
        block = _body(cert).get("noise_floor")
        if not isinstance(block, dict):
            return None
        named = []
        run_ids = block.get("runs")
        for rid in run_ids if isinstance(run_ids, list) else []:
            at = self.find(rid) if isinstance(rid, str) else None
            if at is None:
                return None
            named.append(self.cert(at))
        # The floor's runs are the certs whose BODIES it was computed over -- which is not always
        # "this cert plus the named ones"; see ``_floor_body_certs``. Counting run 0 twice would
        # make the census advertise a disagreement an honest ladder does not have.
        certs = self._floor_body_certs(cert, named, self._floor_declared_runs(cert))
        plan_id = block.get("plan")
        at = self.find(plan_id) if isinstance(plan_id, str) else None
        declared = _declared_factors(self.cert(at)) if at is not None else []
        return certs, declared

    def _check_floor_varied_an_execution(self, cert: dict) -> None:
        """Refuse a floor whose declared factors name knobs that turn nothing.

        See the module docstring under "Decisions". ``_check_floor_honours_its_plan`` has already
        established that the runs recorded distinct values of every declared factor; this asks
        the next question, which is whether those values were distinct EXECUTIONS. A factor is
        *realizable* when, at some one of the floor's own runs, its declared values would produce
        more than one execution state (``fingerprint.execution_states_for``); it is *not
        derivable* when no process here can apply it at all.
        """
        parts = self._floor_parts(cert)
        if parts is None:
            return
        certs, declared = parts
        if not declared:
            return  # a plan carrying no nuisance list declares nothing (module docstring)

        realizable: list[str] = []
        for factor, values in declared:
            if len(values) < 2:
                continue
            states = _factor_states(certs, factor, values)
            if states is None:
                continue  # not derivable from the bytes; reported in the census, not refused
            if not any(len(per_run) > 1 for per_run in states):
                raise AppendRefused(_unrealizable_reason(factor, values, certs))
            realizable.append(factor)

        if not realizable:
            raise AppendRefused(
                "floor: the plan declares "
                + ", ".join(repr(f) for f, _ in declared)
                + " and not one of them is a factor this floor's runs could have varied: a plan "
                "whose every factor names a single value, or names something no process here can "
                "apply, commits to no variation, so the runs are one computation and the floor's "
                f"{len(certs) * (len(certs) - 1) // 2} pairwise distances compare a run with "
                "itself (section 5.1 step 2)"
            )

    def floor_census(self, cert: dict) -> Optional[dict]:
        """What the floor of ``cert`` actually measured, counted in executions rather than labels.

        Returns ``None`` for a cert carrying no ``noise_floor``, or when a run the floor names is
        not in this log (``_check_floor_honours_its_plan`` refuses that case first). Otherwise::

            {
              "runs": A,                       # the canonical plus the certs in noise_floor.runs
              "pairs": A*(A-1)/2,              # the pairs those runs can form
              "advertised_runs": [...],        # what per_channel says it rests on
              "advertised_pairs": [...],
              "advertised_agrees": bool,       # False = the block counts runs it does not name
              "executions": E,                 # distinct execution states among the A runs
              "pairs_same_execution": P,       # pairs whose two runs were ONE execution
              "state_kinds": [...],            # "batches" (derived) and/or "recorded" (labels only)
              "factors_without_demonstrated_effect": [...],
              "factors_not_derivable": [...],  # declared, and no process here can apply them
              "zero_channels": [...],          # channels whose floor is exactly 0.0
              "all_channels_zero": bool,
            }

        This is the client-side reading of a floor and it is written into the entry's metadata at
        append. It is a public method because a reader who did not run the append needs the same
        numbers: every one of them is re-derivable from the log's own bytes.
        """
        block = _body(cert).get("noise_floor")
        parts = self._floor_parts(cert)
        if parts is None or not isinstance(block, dict):
            return None
        certs, declared = parts

        states = [fpmod.execution_state(c) for c in certs]
        total = len(states)
        pairs = total * (total - 1) // 2
        same = sum(
            1
            for i in range(total)
            for j in range(i + 1, total)
            if states[i] == states[j]
        )

        assignments = [_nuisance_of(c) for c in certs]
        names = [factor for factor, _ in declared]
        without: list[str] = []
        opaque: list[str] = []
        for factor, values in declared:
            if len(values) < 2:
                continue
            if _factor_states(certs, factor, values) is None:
                opaque.append(factor)
            if not _factor_moved_an_execution(factor, names, assignments, states):
                without.append(factor)

        per = block.get("per_channel")
        per = per if isinstance(per, dict) else {}
        zero: list[str] = []
        advertised: set = set()
        for channel, cblock in sorted(per.items()):
            if not isinstance(cblock, dict):
                continue
            advertised.add((cblock.get("runs"), cblock.get("pairs")))
            value = cblock.get("floor")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if float(value) == 0.0:
                zero.append(channel)
        # The floor's own `per_channel` says how many runs and pairs it rests on. Nothing derives
        # those from the runs the block NAMES, so the two can disagree -- and when they do, the
        # advertised `alpha_single = 1/(pairs+1)` describes measurements that are not in the log.
        advertised_runs = sorted({r for r, _ in advertised if isinstance(r, int)})
        advertised_pairs = sorted({p for _, p in advertised if isinstance(p, int)})
        return {
            "runs": total,
            "pairs": pairs,
            "advertised_runs": advertised_runs,
            "advertised_pairs": advertised_pairs,
            "advertised_agrees": advertised_runs == [total] and advertised_pairs == [pairs],
            "executions": len({tuple(_freeze(s)) for s in states}),
            "pairs_same_execution": same,
            "state_kinds": sorted({s[0] for s in states}),
            "factors_without_demonstrated_effect": sorted(without),
            "factors_not_derivable": sorted(opaque),
            "zero_channels": zero,
            "all_channels_zero": bool(per) and len(zero) == len(
                [c for c, b in per.items() if isinstance(b, dict) and isinstance(b.get("floor"), (int, float))
                 and not isinstance(b.get("floor"), bool)]
            ),
        }

    def baseline_gap(self, cert: dict) -> Optional[dict]:
        """The distance between the baseline already on record and ``cert``, or None (section 5.5).

        **Disclosure, not prevention.** Section 5.5 lets a second canonical fingerprint for one
        subject append when it carries a ``previous`` ref, or when it is another run under the
        same logged noise plan (``_same_noise_plan``). Choosing a new baseline is legitimate and
        this method refuses nothing: ``append`` calls it after the section 5.5 rule has already
        decided, and writes what it returns into the entry's metadata. What the choice must not
        be is silent. Before this existed, the quantity that would announce it -- how far the new
        baseline sits from the one it replaces -- was computable from two certs the log already
        held and was computed nowhere, so a reader who verified against the new baseline was
        never told a different one had been on record (the BASELINE-CHOICE attack).

        **Where this touches the label class.** The gap is arithmetic over bytes the issuer wrote.
        It announces that the baseline moved and by how much; it cannot say which of the two
        baselines measured the subject, because a relabelled computation and a computation are
        byte-indistinguishable (``papers/v8/THE_BOUNDARY_2026_09_09.md``, class two). An issuer
        willing to be seen moving its baseline still moves it, and one that fabricates the runs
        under the new baseline gets honest arithmetic over dishonest bytes. BASELINE-CHOICE stays
        in the label class where the relabel itself is concerned; only the announcement leaves it.

        Returns None for anything that is not a fingerprint, and for the first fingerprint of a
        subject -- there is no baseline to be apart from. Otherwise ``floor.baseline_gap`` over
        the two run bodies, plus who the previous baseline is and whether this cert named it::

            {
              "previous_index": int,        # the highest comparable fingerprint in this log
              "previous_id": str,
              "declared_previous": [str],   # `previous` refs this cert carries, in ref order
              "announced": bool,            # a `previous` ref names previous_id
              "same_noise_plan": bool,      # appended under 5.5's runs-of-one-plan branch instead
              ... floor.baseline_gap(...)
            }

        ``announced`` false with ``same_noise_plan`` true is the case section 5.5 admits with no
        ref at all: legitimate, and the case a reader could not otherwise see. Every number is
        re-derivable by a reader from the log's own bytes -- this is a public method for that
        reason, as ``floor_census`` is.
        """
        if cert.get("type") != "fingerprint":
            return None
        # `previous_comparable` scans the whole log, which is right at APPEND -- the cert is not
        # in it yet, so the highest comparable index is the baseline being replaced. Called on a
        # cert the log already holds, that scan can return a fingerprint appended AFTER it, and
        # the gap would then be measured against a baseline that did not yet exist. A reader
        # re-deriving the number must get the same one the append wrote, so the search is bounded
        # below the cert's own index whenever the log has one for it.
        own_id = cert.get("id")
        at = self.find(own_id) if isinstance(own_id, str) else None
        if at is None:
            previous = self.previous_comparable(cert)
        else:
            previous = None
            for index in reversed([i for i in self.indices() if i < at]):
                try:
                    other = self.cert(index)
                except Exception:
                    continue
                if other.get("type") != "fingerprint" or other.get("id") == own_id:
                    continue
                if certmod.comparable(other, cert) == []:
                    previous = index
                    break
        if previous is None:
            return None
        try:
            prev = self.cert(previous)
        except Exception:
            return None
        prev_body = _body(prev)
        own_body = _body(cert)
        declared = [rid for role, rid in certmod.refs(cert) if role == "previous"]
        out: dict[str, Any] = {
            "previous_index": previous,
            "previous_id": prev.get("id"),
            "declared_previous": declared,
            "announced": prev.get("id") in declared,
            "same_noise_plan": _same_noise_plan(cert, prev),
        }
        try:
            out.update(floormod.baseline_gap(prev_body, own_body))
        except (ValueError, TypeError, KeyError) as exc:
            # A gap that will not compute is a gap reported as uncomputed. It is metadata, and
            # refusing the append here would turn a reporting failure into a refusal the section
            # 5.5 rule never asked for.
            out["note"] = f"{type(exc).__name__}: {exc}"
        return out

    def _check_challenge_subject(self, cert: dict) -> None:
        """Section 9 rule 1, run over the two certs this log already holds, plus C3's binding.

        See the module docstring under "Decisions" for why the log runs a rule section 9 gives to
        clients, and for the three refusals this method grew after the second attack pass.

        Step 3 has already resolved both refs and, since ``cert.role_type`` fixes a challenge's
        ``target`` to a fingerprint, has already typed them; ``cert.check`` has already refused a
        repeated ``target`` or ``own``. Every early return this method used to take is therefore
        a refusal now: a state it cannot reach is a state it must not pass.
        """
        by_role: dict[str, list[dict]] = {"target": [], "own": []}
        for role, rid in certmod.refs(cert):
            if role not in by_role:
                continue
            at = self.find(rid)
            if at is None:
                raise AppendRefused(
                    f"challenge: the {role} ref {rid} does not resolve in this log, so section 9 "
                    "rule 1 cannot be computed; a challenge whose halves this log cannot read is "
                    "not seated on the chance that it would have passed"
                )
            by_role[role].append(self.cert(at))
        for role, found in by_role.items():
            if len(found) != 1:
                raise AppendRefused(
                    f"challenge: {len(found)} {role} refs; section 9 rule 1 is a predicate over "
                    "exactly one target and one own fingerprint, and a challenge naming two "
                    "gives the log one reading and a reader walking refs in order another"
                )
        target, own = by_role["target"][0], by_role["own"][0]
        for role, resolved in (("target", target), ("own", own)):
            if resolved.get("type") != "fingerprint":
                raise AppendRefused(
                    f"challenge: the {role} ref resolves to a {resolved.get('type')!r} cert; "
                    "section 9 challenges a fingerprint with a fingerprint, and a target that is "
                    "not one leaves rule 1 with nothing to compare"
                )
        reasons = certmod.challenge_validity(target, own)
        if reasons:
            raise AppendRefused(
                f"challenge: the challenger's fingerprint {own.get('id')} differs from the target "
                f"{target.get('id')} on {reasons}; a challenge is valid only if the two are "
                "comparable AND every S_identity field is equal, and a floor measured on one "
                "subject says nothing about another (section 9 rule 1, section 5.3). No match, "
                "no challenge"
            )
        self._check_challenge_self_report(cert, own)

    def _check_challenge_self_report(self, cert: dict, own: dict) -> None:
        """C3: the challenge body's ``subject``/``recipe_core`` are the ``own`` cert's, or refuse.

        A challenge carried ``subject: {}``, ``recipe: {}`` and a body of
        ``{per_channel, coverage, environment}``, so no signed byte said what produced the
        distances; the only pointer was the ``own`` ref, which nothing validated
        (``papers/v8/challenge_and_attack_2026_09_09``, C3). ``verify.challenge_body`` now writes
        the challenger's own report into the body, and this is what keeps that report honest:
        the body's ``subject`` must be the S_identity of the ``own`` fingerprint, its
        ``recipe_core`` that cert's ``recipe_core``, and its synthetic marker that cert's marker.
        A self-description a log will not check is a comment.
        """
        body = _body(cert)
        stated_subject = body.get("subject")
        stated_core = body.get("recipe_core")
        if not isinstance(stated_subject, dict) or not isinstance(stated_core, dict):
            raise AppendRefused(
                "challenge: the body carries no 'subject'/'recipe_core'; a challenge says in its "
                "own signed bytes what produced its distances (section 9 body, C3)"
            )
        want_subject = certmod.identity_fields(own.get("subject", {}))
        got_subject = certmod.identity_fields(stated_subject)
        if got_subject != want_subject:
            raise AppendRefused(
                f"challenge: body.subject is not the identity of the own fingerprint "
                f"{own.get('id')}: {got_subject} != {want_subject}"
            )
        want_core = certmod.recipe_core(own.get("recipe", {}))
        if certmod.recipe_core(stated_core) != want_core:
            raise AppendRefused(
                f"challenge: body.recipe_core is not the recipe_core of the own fingerprint "
                f"{own.get('id')}"
            )
        if bool(body.get(certmod.SYNTHETIC)) != certmod.is_synthetic(own):
            raise AppendRefused(
                f"challenge: body.{certmod.SYNTHETIC} disagrees with the own fingerprint "
                f"{own.get('id')}; a challenge does not get to relabel where its numbers came "
                "from"
            )

    def _public_for(
        self, cert: dict, _memo: Optional[dict] = None, _stack: Optional[set] = None
    ) -> bool:
        """Section 2.6: not redacted, and every cert in the transitive refs is public.

        Derived from the CERTS, not from the neighbouring metadata files (A-META). It used to read
        ``meta(at)["public"]`` at each ref, so one edited metadata file flipped the transitive
        answer for every cert stacked on it — and ``public`` gates section 4.5's redaction rule at
        append. Refs resolve backwards (section 2.1), so the recursion is finite; ``_stack``
        guards a hand-built cycle rather than trusting that rule, and ``_memo`` keeps the ref
        graph's diamonds from being read as cycles.
        """
        memo: dict = {} if _memo is None else _memo
        stack: set = set() if _stack is None else _stack
        if _redacted(cert):
            return False
        for _, rid in certmod.refs(cert):
            at = self.find(rid)
            if at is None:
                continue
            if not self._public_at(at, memo, stack):
                return False
        return True

    def _public_at(
        self, index: int, _memo: Optional[dict] = None, _stack: Optional[set] = None
    ) -> bool:
        """``_public_for`` of the cert at ``index``; an unreadable entry or a cycle is not public.

        ``_memo`` caches the ANSWER per index and ``_stack`` is the current path. Both are needed:
        a diamond in the ref graph — a fingerprint and a canary battery that both name the pool —
        visits one index twice on two different paths, and a single "seen" set would read the
        second visit as a cycle and report a public cert as not public.
        """
        memo: dict = {} if _memo is None else _memo
        stack: set = set() if _stack is None else _stack
        if index in memo:
            return memo[index]
        if index in stack:
            return False  # refs resolve backwards (section 2.1), so an append cannot build one
        stack.add(index)
        try:
            cert = self.cert(index)
        except Exception:
            value = False
        else:
            value = self._public_for(cert, memo, stack)
        stack.discard(index)
        memo[index] = value
        return value

    def previous_comparable(self, cert: dict) -> Optional[int]:
        """The highest index holding a fingerprint comparable to ``cert``, or None.

        Comparable is ``cert.comparable(other, cert) == []`` — no entry at all, so a
        cross-subject difference (precision, revision) starts its own baseline.
        """
        own_id = cert.get("id")
        for index in reversed(self.indices()):
            try:
                other = self.cert(index)
            except Exception:
                continue
            if other.get("type") != "fingerprint" or other.get("id") == own_id:
                continue
            if certmod.comparable(other, cert) == []:
                return index
        return None

    # ------------------------------------------------------------- tree heads

    def sths(self) -> list[dict]:
        """Every parsable file in ``sth/``, by tree_size then file name."""
        out: list[tuple[Any, str, dict]] = []
        for path in sorted(self.sth_dir.glob("*.json")):
            try:
                obj = _read_json(path)
            except Exception:
                continue
            if isinstance(obj, dict):
                size = obj.get("tree_size")
                out.append((size if isinstance(size, int) else -1, path.name, obj))
        return [item[2] for item in sorted(out, key=lambda t: (t[0], t[1]))]

    def latest_sth(self) -> Optional[dict]:
        """The STH with the largest tree_size, or None."""
        found = self.sths()
        return found[-1] if found else None

    def sth(self, log_private_seed: bytes, timestamp: str) -> dict:
        """Sign the current tree head and write ``sth/<tree_size:012d>.json``.

        Refuses when the seed's public key is not ``keys/log.pub``, when ``timestamp`` is not
        RFC 3339 Z, and when a different STH for this tree size is already on file.
        """
        public = self.log_public()
        if public is None:
            raise AppendRefused(f"sth: no log public key at {self.keys_dir / 'log.pub'}")
        if keys.public_from_private(log_private_seed) != public:
            raise AppendRefused("sth: the seed's public key is not keys/log.pub")
        if not isinstance(timestamp, str) or not _RFC3339_Z.match(timestamp):
            raise AppendRefused(f"sth: timestamp {timestamp!r} is not RFC 3339 Z")

        tree_size = self.size()
        core = {
            "log_id": self.log_id(),
            "tree_size": tree_size,
            "root_hash": "sha256:" + self.root(tree_size).hex(),
            "timestamp": timestamp,
        }
        digest = hashlib.sha256(canonical_bytes(core)).digest()
        signature = keys.sign(log_private_seed, keys.tagged(STH_TAG, digest))
        sth = dict(core)
        sth["sig"] = keys.encode_signature(signature)

        path = self.sth_dir / f"{tree_size:012d}.json"
        if path.exists():
            existing = _read_json(path)
            if existing == sth:
                return sth
            if isinstance(existing, dict) and existing.get("root_hash") != sth["root_hash"]:
                raise AppendRefused(
                    f"sth: tree_size {tree_size} is already on file with a different root_hash"
                )
            raise AppendRefused(
                f"sth: tree_size {tree_size} is already on file with a different timestamp"
            )
        _write_json(path, sth)
        return sth

    # ------------------------------------------------------------- proofs

    def inclusion(self, index: int, tree_size: Optional[int] = None) -> dict:
        """An inclusion proof for entry ``index`` at ``tree_size`` (the whole log by default)."""
        leaves = self.leaf_hashes()
        if tree_size is None:
            tree_size = len(leaves)
        path = merkle.inclusion_proof(leaves, index, tree_size)
        return {
            "leaf_index": index,
            "leaf_hash": leaves[index].hex(),
            "tree_size": tree_size,
            "root_hash": "sha256:" + merkle.root(leaves[:tree_size]).hex(),
            "path": [h.hex() for h in path],
        }

    def consistency(self, first: int, second: Optional[int] = None) -> dict:
        """A consistency proof that the tree at ``second`` extends the tree at ``first``."""
        leaves = self.leaf_hashes()
        if second is None:
            second = len(leaves)
        proof = merkle.consistency_proof(leaves, first, second)
        return {
            "first": first,
            "second": second,
            "first_root": "sha256:" + merkle.root(leaves[:first]).hex(),
            "second_root": "sha256:" + merkle.root(leaves[:second]).hex(),
            "proof": [h.hex() for h in proof],
        }


# ----------------------------------------------------------------- roster helpers

def _clean_policy(issuers) -> Any:
    """The bytes ``keys/issuers.json`` gets: a cleaned roster, or the open marker.

    A dict must name ``policy: "open"`` and may carry a ``reason``; anything else is a policy this
    version cannot enforce, and writing it would produce a log that refuses every append with
    "unknown policy" — better to refuse at init, where the operator is standing there.
    """
    if isinstance(issuers, dict):
        if issuers.get("policy") != OPEN_POLICY:
            raise ValueError(
                f"issuer policy {issuers.get('policy')!r} is not {OPEN_POLICY!r}; the only "
                "non-roster policy is the open marker"
            )
        reason = issuers.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("issuer policy: reason must be a string or absent")
        out = {"policy": OPEN_POLICY}
        if isinstance(reason, str):
            out["reason"] = reason
        return out
    return _clean_roster(issuers)


def _clean_roster(issuers: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for raw in issuers:
        if not isinstance(raw, dict):
            raise ValueError("every roster entry must be an object")
        key = raw.get("key")
        why = certmod.public_key_reason(key) if isinstance(key, str) else "missing"
        if why is not None:
            raise ValueError(f"roster key {key!r}: {why}")
        from_index = raw.get("from_index", 0)
        if not isinstance(from_index, int) or isinstance(from_index, bool) or from_index < 0:
            raise ValueError(f"roster entry {key!r}: from_index must be a non-negative int")
        retired = raw.get("retired_at_index")
        if retired is not None and (not isinstance(retired, int) or isinstance(retired, bool)):
            raise ValueError(f"roster entry {key!r}: retired_at_index must be an int or null")
        out.append(
            {
                "name": raw.get("name") if isinstance(raw.get("name"), str) else "",
                "key": key,
                "from_index": from_index,
                "retired_at_index": retired,
            }
        )
    return out


def _in_roster(roster: list[dict], key: Any, index: int) -> bool:
    for entry in roster:
        if not isinstance(entry, dict) or entry.get("key") != key:
            continue
        from_index = entry.get("from_index", 0)
        if not isinstance(from_index, int) or from_index > index:
            continue
        retired = entry.get("retired_at_index")
        if isinstance(retired, int) and index >= retired:
            continue
        return True
    return False


def _blob_hex(key: Any) -> str:
    """The 64 hex characters a blob key names; accepts ``<hex>`` and ``sha256:<hex>``."""
    if not isinstance(key, str):
        raise ValueError(f"blob key must be a string, got {type(key).__name__}")
    raw = key.split(":", 1)[1] if key.startswith("sha256:") else key
    if not _HEX64.match(raw):
        raise ValueError(f"blob key {key!r} is not sha256:<64 hex> or <64 hex>")
    return raw


def _declared_factors(plan: dict) -> list[tuple[str, list]]:
    """``[(factor, values)]`` from a noise-plan prereg body; ``[]`` when it declares none."""
    blocks = _body(plan).get("nuisance")
    if not isinstance(blocks, list):
        return []
    out: list[tuple[str, list]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        factor = block.get("factor")
        values = block.get("values")
        if isinstance(factor, str) and factor and isinstance(values, list):
            out.append((factor, values))
    return out


def _nuisance_of(cert: dict) -> dict:
    """A fingerprint's ``body.nuisance``, or ``{}`` — the assignment the run recorded."""
    nuisance = _body(cert).get("nuisance")
    return nuisance if isinstance(nuisance, dict) else {}


# The nuisance factors that are also recorded in ``recipe.decoding``, and therefore the ones a
# cert can be caught contradicting itself about. ``item_order`` is not among them: the recipe
# does not record an order, which is why ``nuisance.item_order_sha256`` is the only receipt for
# it and why this check cannot reach it.
_RECIPE_FACTORS = ("batch_size", "padding_side")


def _decoding_of(cert: dict) -> dict:
    """A cert's ``recipe.decoding``, or ``{}`` — the execution the cert says it ran under."""
    recipe = cert.get("recipe")
    decoding = recipe.get("decoding") if isinstance(recipe, dict) else None
    return decoding if isinstance(decoding, dict) else {}


def _floor_roles(body: dict) -> dict[str, str]:
    """``{item_id: role}`` off the canonical body's items — the map ``attach_floor`` computed the
    exact channel's floor under, so the recomputation scores the same items."""
    out: dict[str, str] = {}
    items = body.get("items")
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        iid, role = item.get("item_id"), item.get("role", "item")
        if isinstance(iid, str) and isinstance(role, str) and role:
            out[iid] = role
    return out


def _exactly(a: Any, b: Any) -> bool:
    """``a == b`` with no tolerance, and with ``True`` never equal to ``1``.

    The floor's numbers are ``round(x, 9)`` over deterministic arithmetic on the same bytes and a
    JSON round-trip of a float is exact, so an honest cert reproduces them bit for bit. A window
    here would be a place to hide a floor in.
    """
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return False
        return all(_exactly(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _freeze(value: Any) -> Any:
    """A hashable copy of an execution state (lists/dicts never appear, but hostile input can)."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def _factor_moved_an_execution(
    factor: str, declared: Sequence[str], assignments: Sequence[dict], states: Sequence[Any]
) -> bool:
    """True when two runs differing ONLY in ``factor`` ran different executions.

    "Only" is over the plan's declared factors: two assignments that agree on every other
    declared factor and disagree on this one isolate it. A factor no such pair isolates has no
    demonstrated effect on this design — which is a statement about the design, not proof that
    the factor is inert, and is why it is reported rather than refused.
    """
    others = [name for name in declared if name != factor]
    for i in range(len(assignments)):
        for j in range(i + 1, len(assignments)):
            a, b = assignments[i], assignments[j]
            if str(a.get(factor)) == str(b.get(factor)):
                continue
            if any(str(a.get(name)) != str(b.get(name)) for name in others):
                continue
            if states[i] != states[j]:
                return True
    return False


def _factor_states(
    certs: Sequence[dict], factor: str, values: Sequence[Any]
) -> Optional[list[set]]:
    """One set of counterfactual execution states per run that could answer, or ``None``.

    ``None`` means no run could answer the question — the factor is one no process here applies,
    or every run's items are unreadable.

    THIS RETURNS THE PER-RUN SETS AND NOT THEIR UNION, and the difference is a defect the
    attacker of `papers/v8/challenge_and_attack_2026_09_09/` named (R-UNION). The predicate the
    module docstring states is "at SOME ONE of the floor's own runs, the declared values would
    produce more than one execution state" — ``any(len(s) > 1 for s in sets)``. The union
    ``len(seen) > 1`` is a different and weaker question: it is true whenever the runs merely
    differ FROM EACH OTHER, so a plan of ``item_order`` at batch size 1 — where permuting the
    order permutes independent single-item passes and every run's counterfactual set is a
    singleton — passed as soon as one named run carried a different item count. The per-run
    answer was already being computed and thrown away; it is now what is returned.
    """
    out: list[set] = []
    for cert in certs:
        states = fpmod.execution_states_for(cert, factor, values)
        if states is None:
            continue
        out.append(states)
    return out or None


def _unrealizable_reason(factor: str, values: Sequence[Any], certs: Sequence[dict]) -> str:
    """The refusal for a declared factor that turns nothing at any of the floor's own runs."""
    shown = sorted({str(v) for v in values})
    batches = sorted({str(_nuisance_of(c).get("batch_size")) for c in certs})
    counts = sorted({str(len(_body(c).get("items") or [])) for c in certs})
    return (
        f"floor: the plan declares nuisance factor {factor!r} with values {shown} and not one of "
        f"the {len(certs)} runs is a configuration those values could tell apart: at batch size "
        f"{batches} over {counts} items every declared value names the same forward passes, so "
        "the runs recorded different labels and ran one computation. A batch size the item count "
        "coerces, a padding side no batch is large enough to apply, and an item order permuted at "
        "batch size 1 are labels, not executions (section 5.1 step 2)"
    )


def _noise_plan_ids(cert: dict) -> set[str]:
    return {rid for role, rid in certmod.refs(cert) if role == "noise_plan"}


def _same_noise_plan(cert: dict, other: dict) -> bool:
    """True when both certs are runs under the same logged noise plan (section 5.1).

    Runs of one floor are comparable to each other by construction; they do not each start a
    new baseline, so the previous-ref rule does not apply between them.
    """
    mine = _noise_plan_ids(cert)
    return bool(mine) and mine == _noise_plan_ids(other)


# ----------------------------------------------------------------- verification

def verify_sth(sth: dict, log_public: bytes) -> tuple[bool, str]:
    """``(ok, reason)``. Never raises: hostile input is a False, not a crash."""
    try:
        if not isinstance(sth, dict):
            return False, "sth: not a JSON object"
        for field in STH_KEYS:
            if field not in sth:
                return False, f"sth: missing {field}"
        for field in sth:
            if field not in STH_KEYS:
                return False, f"sth: unexpected field {field!r}"
        if not isinstance(log_public, (bytes, bytearray, memoryview)):
            return False, "sth: log public key is not bytes"
        public = bytes(log_public)
        if len(public) != 32:
            return False, f"sth: log public key is {len(public)} bytes, not 32"
        want_log_id = "sha256:" + hashlib.sha256(public).hexdigest()
        if sth["log_id"] != want_log_id:
            return False, f"sth: log_id {sth['log_id']!r} is not this key's log_id"
        if not isinstance(sth["tree_size"], int) or isinstance(sth["tree_size"], bool):
            return False, "sth: tree_size is not an integer"
        if sth["tree_size"] < 0:
            return False, "sth: tree_size is negative"
        if _root_bytes(sth["root_hash"]) is None:
            return False, "sth: root_hash is not sha256:<64 hex>"
        if not isinstance(sth["timestamp"], str) or not _RFC3339_Z.match(sth["timestamp"]):
            return False, "sth: timestamp is not RFC 3339 Z"
        try:
            signature = keys.decode_signature(sth["sig"])
        except (TypeError, ValueError) as exc:
            return False, f"sth: sig does not decode: {exc}"
        core = {field: sth[field] for field in _STH_CORE}
        digest = hashlib.sha256(canonical_bytes(core)).digest()
        if not keys.verify(public, keys.tagged(STH_TAG, digest), signature):
            return False, "sth: sig does not verify against the log public key"
        return True, "ok"
    except Exception as exc:  # a verifier must not crash on a stranger's file
        return False, f"sth: {type(exc).__name__}: {exc}"


def verify_inclusion(proof: dict, sth: dict, log_public: bytes) -> tuple[bool, str]:
    """``(ok, reason)``: the STH verifies and the proof lands on its root."""
    try:
        ok, why = verify_sth(sth, log_public)
        if not ok:
            return False, why
        if not isinstance(proof, dict):
            return False, "inclusion: proof is not a JSON object"
        for field in ("leaf_index", "leaf_hash", "tree_size", "root_hash", "path"):
            if field not in proof:
                return False, f"inclusion: missing {field}"
        if proof["tree_size"] != sth["tree_size"]:
            return False, (
                f"inclusion: proof tree_size {proof['tree_size']!r} is not the STH's "
                f"{sth['tree_size']!r}"
            )
        if proof["root_hash"] != sth["root_hash"]:
            return False, "inclusion: proof root_hash is not the STH's root_hash"
        leaf = _unhex(proof["leaf_hash"])
        if leaf is None:
            return False, "inclusion: leaf_hash is not 64 hex characters"
        root = _root_bytes(sth["root_hash"])
        if root is None:
            return False, "inclusion: root_hash is not sha256:<64 hex>"
        if not isinstance(proof["path"], list):
            return False, "inclusion: path is not an array"
        path = [_unhex(h) for h in proof["path"]]
        if any(h is None for h in path):
            return False, "inclusion: a path element is not 64 hex characters"
        index = proof["leaf_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            return False, "inclusion: leaf_index is not an integer"
        if not merkle.verify_inclusion(leaf, index, proof["tree_size"], path, root):
            return False, "inclusion: the proof does not reach the signed root"
        return True, "ok"
    except Exception as exc:
        return False, f"inclusion: {type(exc).__name__}: {exc}"


def verify_consistency(sth_m: dict, sth_n: dict, proof: dict, log_public: bytes) -> tuple[bool, str]:
    """``(ok, reason)``: both STHs verify and the proof shows n extends m."""
    try:
        ok, why = verify_sth(sth_m, log_public)
        if not ok:
            return False, f"first {why}"
        ok, why = verify_sth(sth_n, log_public)
        if not ok:
            return False, f"second {why}"
        if not isinstance(proof, dict):
            return False, "consistency: proof is not a JSON object"
        for field in ("first", "second", "first_root", "second_root", "proof"):
            if field not in proof:
                return False, f"consistency: missing {field}"
        if proof["first"] != sth_m["tree_size"]:
            return False, "consistency: proof first is not the first STH's tree_size"
        if proof["second"] != sth_n["tree_size"]:
            return False, "consistency: proof second is not the second STH's tree_size"
        if proof["first_root"] != sth_m["root_hash"]:
            return False, "consistency: proof first_root is not the first STH's root_hash"
        if proof["second_root"] != sth_n["root_hash"]:
            return False, "consistency: proof second_root is not the second STH's root_hash"
        if sth_m["tree_size"] > sth_n["tree_size"]:
            return False, "consistency: the first STH is larger than the second"
        first_root = _root_bytes(sth_m["root_hash"])
        second_root = _root_bytes(sth_n["root_hash"])
        if first_root is None or second_root is None:
            return False, "consistency: a root_hash is not sha256:<64 hex>"
        if not isinstance(proof["proof"], list):
            return False, "consistency: proof is not an array"
        path = [_unhex(h) for h in proof["proof"]]
        if any(h is None for h in path):
            return False, "consistency: a proof element is not 64 hex characters"
        if not merkle.verify_consistency(
            sth_m["tree_size"], sth_n["tree_size"], first_root, second_root, path
        ):
            return False, "consistency: the second tree does not extend the first"
        return True, "ok"
    except Exception as exc:
        return False, f"consistency: {type(exc).__name__}: {exc}"


def verify_entry(log: "Log", index: int) -> tuple[bool, str]:
    """``(ok, reason)`` for one entry: bytes, id, signature, metadata, leaf hash.

    Checks that the file bytes are the canonical bytes of what they parse to, that the id
    recomputes from them, that the signature verifies under the issuer key inside them, that the
    metadata beside the entry names the same id and index, and that the leaf hash the tree holds
    at this index is the hash of these bytes. When the log holds its public key and an STH that
    covers the index, the inclusion proof is verified against that head too.
    """
    try:
        try:
            raw = log.entry_bytes(index)
        except Exception as exc:
            return False, f"entry {index}: unreadable: {exc}"
        if b"\r" in raw:
            return False, f"entry {index}: contains CR (section 8.2 pins LF)"
        if raw.endswith(b"\n"):
            return False, f"entry {index}: ends with a newline"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return False, f"entry {index}: does not parse: {exc}"
        if not isinstance(parsed, dict):
            return False, f"entry {index}: is not a JSON object"
        try:
            canonical = canonical_bytes(parsed)
        except Exception as exc:
            return False, f"entry {index}: not canonicalizable: {exc}"
        if canonical != raw:
            return False, f"entry {index}: bytes are not the canonical form of their content"
        try:
            digest = certmod.digest_bytes(parsed)
        except Exception as exc:
            return False, f"entry {index}: digest unavailable: {exc}"
        expected = "sha256:" + digest.hex()
        if parsed.get("id") != expected:
            return False, f"entry {index}: id does not recompute"
        issuer = parsed.get("issuer")
        key_s = issuer.get("key") if isinstance(issuer, dict) else None
        if not isinstance(key_s, str):
            return False, f"entry {index}: no issuer key"
        why = certmod.public_key_reason(key_s)
        if why is not None:
            return False, f"entry {index}: issuer key {why}"
        try:
            signature = keys.decode_signature(parsed.get("sig"))
        except (TypeError, ValueError) as exc:
            return False, f"entry {index}: sig does not decode: {exc}"
        if not keys.verify(keys.decode_public(key_s), keys.tagged(CERT_TAG, digest), signature):
            return False, f"entry {index}: sig does not verify"
        try:
            meta = log.meta(index)
        except Exception as exc:
            return False, f"entry {index}: metadata unreadable: {exc}"
        if meta.get("id") != expected:
            return False, f"entry {index}: metadata names {meta.get('id')!r}, the bytes give {expected}"
        if meta.get("index") != index:
            return False, f"entry {index}: metadata names index {meta.get('index')!r}"
        # A-META: the two lines above were the WHOLE of what this checked, so every other field of
        # an unsigned file outside the tree -- `public`, `type`, and the floor census -- was the
        # operator's to rewrite with no leaf moving and no root changing. The census is re-derived
        # by ``floor_census`` from certs this log holds; every derivable field is compared.
        #
        # A-META-ABSENT: CONTRADICTED only. A key the file does not carry is a file older than the
        # field, it contradicts nothing in the entries, and refusing it accused this lab's own
        # published log of tampering with itself. ``log.stale_metadata(index)`` is that list, and
        # no caller refuses on it.
        disagreements = log.meta_disagreement(index)
        if disagreements:
            return False, disagreements[0]
        # L7: an entry bound to a log_id that is not this log's is a mis-seated entry. The gate
        # refuses it at append; this is the same predicate in a clone, where the gate never ran.
        hint = _body(parsed).get("log_hint")
        if isinstance(hint, dict):
            named = hint.get("log_id")
            try:
                mine = log.log_id()
            except Exception:
                mine = None
            if mine is not None and named != mine:
                return False, (
                    f"entry {index}: body.log_hint names log {named!r} and this log is {mine}"
                )
        leaf = merkle.leaf_hash(raw)
        try:
            held = log.leaf_hashes()[index]
        except Exception as exc:
            return False, f"entry {index}: no leaf in the tree: {exc}"
        if held != leaf:
            return False, f"entry {index}: the tree holds a different leaf hash"
        public = log.log_public()
        head = log.latest_sth()
        # A head that does not verify under the log's own key is the head's problem, not this
        # entry's: ``mirror`` reports it under misbehaviour. Only a verifying head is used here.
        if public is not None and isinstance(head, dict) and verify_sth(head, public)[0]:
            size = head.get("tree_size")
            if isinstance(size, int) and index < size <= log.size():
                proof = log.inclusion(index, size)
                ok, reason = verify_inclusion(proof, head, public)
                if not ok:
                    return False, f"entry {index}: {reason}"
        return True, "ok"
    except Exception as exc:
        return False, f"entry {index}: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------- mirror

_COPY_DIRS = ("entries", "blobs", "sth", "keys")


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in _COPY_DIRS:
        source = src / name
        if not source.is_dir():
            continue
        target = dst / name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if path.is_dir():
                (target / rel).mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target / rel)
    readme = src / "README.md"
    if readme.is_file():
        shutil.copyfile(readme, dst / "README.md")


def mirror(src, dst, pinned_public: bytes, pinned_sth: Optional[dict] = None) -> dict:
    """Copy a log and re-derive every claim it makes from the bytes. Never raises.

    ``pinned_public`` is the log key the mirror holds out of band; an STH under any other key is
    misbehaviour, not a head. ``pinned_sth`` is a head the mirror obtained elsewhere: it joins
    the local heads for the same-size and consistency comparisons, which is the only way an
    append-only claim is checkable at all (section 8.0).

    Returns ``{"entries", "sths", "verified", "misbehaviour", "unpublished", "tamper",
    "metadata", "stale_metadata", "issuer_policy", "log_binding"}``.

    ``metadata`` is A-META's report: every field of an ``<index>.meta.json`` that CONTRADICTS what
    the entries derive, named per entry. Section 8.2 asks for exactly this — "``mirror`` reports a
    disagreement rather than trusting the file" — and a contradiction also fails ``verify_entry``,
    so it lands in ``tamper`` too and ``verified`` is False. The list is here because the first
    refusal is not the whole story when a census was rewritten field by field.

    ``stale_metadata`` is the other half of that split (A-META-ABSENT): every key the entries
    derive that the file carries no value for. A log minted before a derived field existed is in
    this state permanently and honestly, so it is a disclosure and NOT an accusation — it does not
    enter ``tamper`` and does not clear ``verified``. Nothing here rewrites the copied file: a
    mirror is evidence about its source, and a mirror that silently refreshed the cache would
    report agreement to the next reader over bytes that never agreed. The argument is in this
    module's "Decisions".

    ``issuer_policy`` is the admission policy of the log that was copied (L10c): ``roster``,
    ``open``, or one of the states that refuse every append. It is a disclosure, not an
    accusation — an open log is a legitimate configuration and ``absent`` is a log that admits
    nobody — so none of the four states is written into ``misbehaviour``.

    ``log_binding`` counts the entries that name a log in ``body.log_hint`` and the entries that
    name none (L7). An unbound entry is not a fault; it is the state in which "this cert is in
    the log" is a statement about bytes rather than about this log, and a reader is owed the
    count.
    """
    report: dict[str, Any] = {
        "entries": 0,
        "sths": 0,
        "verified": False,
        "misbehaviour": [],
        "unpublished": [],
        "tamper": [],
        "metadata": [],
        "stale_metadata": [],
        "issuer_policy": None,
        "log_binding": {"bound": 0, "unbound": 0},
    }
    tamper: list[str] = report["tamper"]
    misbehaviour: list[str] = report["misbehaviour"]
    metadata: list[str] = report["metadata"]
    stale: list[str] = report["stale_metadata"]
    try:
        _copy_tree(Path(src), Path(dst))
    except Exception as exc:
        tamper.append(f"mirror: copy failed: {type(exc).__name__}: {exc}")
        return report

    try:
        log = Log(dst)
    except Exception as exc:
        tamper.append(f"mirror: {type(exc).__name__}: {exc}")
        return report

    # entries: contiguity, then every entry from its own bytes
    try:
        present = log.indices()
        size = log.size()
    except Exception as exc:
        tamper.append(f"entries: unreadable: {type(exc).__name__}: {exc}")
        return report
    report["entries"] = size
    # The admission policy of the log this mirror copied, reported and never accused of anything.
    try:
        policy = log.issuer_policy()
    except Exception as exc:
        policy = {"policy": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}
    report["issuer_policy"] = policy
    for index in present:
        if index >= size:
            tamper.append(f"entries: index {index} sits beyond a gap at {size}")
    for index in range(size):
        # A-META: the census and every other derivable metadata field, re-derived rather than
        # read. `verify_entry` refuses on the first contradiction; this names all of them.
        # A-META-ABSENT: a key the file does not carry is stale, not tamper, and is reported
        # under its own name where it accuses nobody.
        try:
            contradicted, absent = log._meta_split(index)
        except Exception as exc:
            contradicted, absent = [f"entry {index}: metadata: {type(exc).__name__}: {exc}"], []
        metadata.extend(contradicted)
        stale.extend(absent)
        try:
            bound = isinstance(_body(log.cert(index)).get("log_hint"), dict)
        except Exception:
            bound = False
        report["log_binding"]["bound" if bound else "unbound"] += 1
        ok, reason = verify_entry(log, index)
        if not ok:
            tamper.append(reason)
    entries_intact = not tamper

    # tree heads
    heads: list[tuple[str, dict]] = []
    for path in sorted(log.sth_dir.glob("*.json")):
        try:
            obj = _read_json(path)
        except Exception as exc:
            misbehaviour.append(f"sth {path.name}: unreadable: {exc}")
            continue
        report["sths"] += 1
        ok, reason = verify_sth(obj, pinned_public)
        if not ok:
            misbehaviour.append(f"sth {path.name}: {reason}")
            continue
        heads.append((path.name, obj))
    if pinned_sth is not None:
        ok, reason = verify_sth(pinned_sth, pinned_public)
        if not ok:
            misbehaviour.append(f"pinned sth: {reason}")
        else:
            heads.append(("<pinned>", pinned_sth))

    # two heads of the same size with different roots is the split-view signature (section 8.4)
    by_size: dict[int, list[tuple[str, dict]]] = {}
    for name, head in heads:
        by_size.setdefault(head["tree_size"], []).append((name, head))
    for tree_size, group in sorted(by_size.items()):
        roots = {head["root_hash"] for _, head in group}
        if len(roots) > 1:
            names = ", ".join(name for name, _ in group)
            misbehaviour.append(
                f"sth: tree_size {tree_size} has {len(roots)} different root_hash values ({names})"
            )

    # every head must be reproducible from the entries the mirror holds
    for name, head in heads:
        tree_size = head["tree_size"]
        if tree_size > size:
            tamper.append(
                f"sth {name}: covers {tree_size} entries, the mirror holds {size}"
            )
            continue
        try:
            local = "sha256:" + log.root(tree_size).hex()
        except Exception as exc:
            tamper.append(f"sth {name}: root not computable: {type(exc).__name__}: {exc}")
            continue
        if local != head["root_hash"]:
            line = f"sth {name}: the entries do not reproduce the signed root at tree_size {tree_size}"
            (misbehaviour if entries_intact else tamper).append(line)

    # consistency for every pair the mirror holds, the pinned head included
    usable = sorted(
        [(name, head) for name, head in heads if head["tree_size"] <= size],
        key=lambda item: item[1]["tree_size"],
    )
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            name_m, sth_m = usable[i]
            name_n, sth_n = usable[j]
            if sth_m["tree_size"] == sth_n["tree_size"]:
                continue
            try:
                proof = log.consistency(sth_m["tree_size"], sth_n["tree_size"])
            except Exception as exc:
                misbehaviour.append(
                    f"consistency {name_m} -> {name_n}: no proof: {type(exc).__name__}: {exc}"
                )
                continue
            proof["first_root"] = sth_m["root_hash"]
            proof["second_root"] = sth_n["root_hash"]
            ok, reason = verify_consistency(sth_m, sth_n, proof, pinned_public)
            if not ok:
                misbehaviour.append(f"consistency {name_m} -> {name_n}: {reason}")

    # entries no head covers yet (section 8.3: the 24-hour merge delay is the operator's promise)
    covered = max([head["tree_size"] for _, head in heads], default=0)
    report["unpublished"] = [index for index in range(covered, size)]

    # A metadata CONTRADICTION is already a `verify_entry` refusal and therefore already in
    # `tamper`; naming it here as well is deliberate, because `verified` is the sentence readers
    # quote and the attack it answers left it True (A-META). `stale_metadata` is deliberately NOT
    # in this conjunction: an absent key contradicts nothing, and clearing `verified` over one
    # called the lab's own untouched published log tampered with (A-META-ABSENT).
    report["verified"] = not tamper and not misbehaviour and not metadata
    return report

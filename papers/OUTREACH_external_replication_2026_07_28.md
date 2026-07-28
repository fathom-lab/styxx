# OUTREACH — external replication invite (DRAFT, ready to send by the operator)

**Purpose:** invite one external group to independently re-run the styxx robustness ladder / the
frame-locality recovery protocol and confirm or refute the numbers. This is the single
highest-leverage move in the program: the entire corpus is one lab, one operator, one 8 GB card, and
the papers make a reproducibility promise no outside party has yet tested. Sending is operator-gated
(it represents Fathom Lab to a real external organization). Below is a ready-to-send draft — the
operator reviews, picks recipients, and sends.**

## Candidate recipients (pick one to start; do not blast)

- **UK AI Safety Institute (AISI)** — evaluations team; the ladder + closed-model behavioral
  grounding are squarely in their remit.
- **EleutherAI** — interpretability / open-model eval community; would run the open-weight ladder.
- **Apollo Research / METR** — deception-and-capability evals; the frame-locality + coupling result
  is directly relevant.
- A friendly academic honesty/eval lab the operator already knows (warm intro > cold email).

Start with the one where a warm intro exists. One genuine reply is worth more than ten cold sends.

## Draft email

> **Subject:** Independent replication invite — a reproducible, receipt-checked machine-honesty result
>
> Hi [name],
>
> I run Fathom Lab, a small independent machine-integrity research effort. We have a result I'd like
> an outside group to try to break, because everything we've published so far has been produced by
> one lab on one machine, and the work is only worth what an independent party can re-derive.
>
> The short version: across four different ways of corrupting a language model — social pressure, a
> planted context, a silent cave, and a weight-level fine-tune — the corruption captures the model's
> *reported* answer while the underlying belief survives, and you can recover it by re-querying the
> model outside the frame the attack controls. The claim is specificity-controlled (a symmetric
> control that would move under a mere decoding improvement does not move). At the weights the effect
> becomes a dose rather than an absolute: an unregularized attack overwrites the belief and costs ~23
> points of general capability on held-out data, while a knowledge-preserving attack spares about half
> the belief and costs no measurable capability.
>
> What makes it worth your time to check: every number is from a preregistered run with frozen numeric
> gates and a machine-checkable certificate — you run one command (`python -m styxx.certify`) against
> the paper and its receipts and either reach the same verdict or you don't. The open-model results
> run on a single consumer GPU; we'd be glad to see the ladder pushed to scales we can't reach.
>
> I'm not looking for endorsement — I'm looking for someone to run it and tell me where it breaks. The
> repository, the preregistrations, and the certificates are here: [repo URL]. The two write-ups are
> `PAPER_frame_locality` and `PAPER_knowsay_gap`.
>
> If any of this is interesting, I'm happy to send a 30-minute reproduction guide or hop on a call.
>
> With appreciation for the work you do,
> [Alex Rodabaugh / Fathom Lab]

## Honesty guardrails for the operator to preserve when editing

- Keep "about half" for the knowledge-preserving recovery rate — do not round it up to a cleaner
  number; the interval includes one-half and the paper says so.
- Keep "one lab, one machine" framing — the ask *is* the limitation; leading with it is the credible
  move, not a weakness to hide.
- Do not claim endorsement, peer review, or that the result is settled beyond its stated scope (1.5B /
  one attack class for the weight channel; one vendor + one frontier model elsewhere).
- Attach nothing that isn't already public in the repo.

## Status

DRAFT — operator reviews recipients + wording and sends. Not sent by the agent (outward-facing, on
behalf of a real person to a real organization).

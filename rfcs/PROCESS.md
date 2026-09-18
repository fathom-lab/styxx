# The RFC process

Every substantive change to the specification — the wire format, the metric definition, the band
boundaries, the conformance suite, the trademark policy — goes through this process. Nothing about
it is private, and there is no channel in which a spec-altering decision can be made out of view.

This file is the normative version. The governance page on the site restates it; where they differ,
this file wins.

## State of play, stated plainly

**No RFC has been opened.** The process below is live and the door is open; nothing has come through
it yet. A list of in-flight RFCs would be a claim about work that does not exist, so there is not
one — here or on the site.

## Lifecycle

```
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐    ┌──────────┐
│  open    │ →  │  Draft   │ →  │ Last Call  │ →  │  Final  │    │Withdrawn │
│  issue   │    │ ≥14 days │    │ ≥14 days   │    │  merge  │    │ rationale│
└──────────┘    └──────────┘    └────────────┘    └─────────┘    └──────────┘
                     │                                                ▲
                     └──────────────── any state ─────────────────────┘
   * major spec change → CFV (call for views) ≥ 30 days before Last Call
```

1. **Open an issue** in `fathom-lab/styxx` describing the change and the problem it solves. Use the
   `rfc` label. An RFC that names no problem is a preference, and preferences are settled in the
   issue rather than in the specification.
2. **Triage.** Maintainers either reject it with a written rationale or label it `rfc/draft`, which
   opens a **Draft period of at least 14 days**. The proposal is iterated in the open during it.
3. **Last Call.** After the Draft period, maintainers MAY move the RFC to Last Call — a **14-day
   window in which only blocking objections are considered**. A blocking objection names something
   the proposal breaks; "I would have done it differently" is not one.
4. **Final or Withdrawn.** After Last Call closes, maintainers merge the RFC (status: Final) or
   withdraw it (status: Withdrawn) **with a written rationale either way**. A withdrawal without a
   reason is not a withdrawal, it is a refusal to answer.

An RFC may be withdrawn from any state, by its author or by the maintainers, with a rationale.

**Major-version changes** — anything that breaks wire compatibility or alters the metric definition
— additionally require a **Call for Views of at least 30 days before Last Call opens**, announced on
the repository and to anyone listed in [`registry/`](../registry/README.md).

## Who decides

Fathom Lab is the current maintainer and makes the final call on every RFC. That is a fact about
today rather than a principle: maintainership is intended to be transferable. **The succession
mechanism is not written down yet** — there is no `MAINTAINERS.md` in this repository, and a
transfer today would happen by whatever means the people involved agreed on, which is not a
governance guarantee and is not described as one.

Maintainer decisions are written into the RFC thread. A decision that exists only in someone's head
has not been made.

## What this process does not cover

- **Implementation changes that do not touch the spec.** Ordinary pull requests.
- **Papers, preregistrations and results.** Those follow the preregistration discipline in
  `papers/`, which is stricter than this one: the gates are frozen and hashed before the run, and a
  blocking gate failure voids the result rather than starting a discussion.
- **Security issues.** See [`SECURITY.md`](../SECURITY.md). Do not open an RFC for a vulnerability.

---

*A governance process nobody has used is not a governance process yet. This is the one we will
follow when someone does.*

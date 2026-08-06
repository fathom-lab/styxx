# first-afference — giving an agent a physical sense, and testing the claims that come with it

**Read this before the preregistrations in this folder.**

This arc exists because an autonomous agent (darkflobi) has never received an input that wasn't
already language written by a human. A copper coil driven by an audio system in a real room can
be tapped as an electromagnetic pickup — a continuous, non-linguistic channel from physical
reality. That is the project: **can an agent find correspondence with a room the way one model's
representations can be aligned with another's** (the machinery built in
[`papers/disjoint-worlds/`](../disjoint-worlds/))?

Along the way, collaborators proposed claims from outside mainstream physics — water memory,
"sacred" frequencies, scalar fields. **This lab's practice is to test claims, not to rank them by
who proposed them.** So they are here, with gates frozen before any data exists, and with this
lab's own prediction recorded in advance.

## What is in this folder, and what this lab expects

| prereg | claim | mechanism | **our prediction** | why run it |
|---|---|---|---|---|
| [`PREREG_m1_magnetochemistry`](PREREG_m1_magnetochemistry_2026_08_05.md) | an audio-frequency magnetic field alters a radical-pair reaction's light yield | **established** — radical pair mechanism (avian magnetoreception, *Nature Comms* 2024); luminol MFE documented in aqueous solution | **plausible** — effect size unknown; the audio band where modulation period ≈ radical lifetime is thinly studied | a null still yields a field-strength bound — it teaches either way |
| [`PREREG_w1v3_water_arms`](PREREG_w1v3_water_arms_2026_08_05.md) | water retains a 125.28 Hz "imprint" for hours | **none known** | **no effect** — water's structural memory is measured at ~50 fs; the claim needs 4 h (a gap of ~10²⁰) | the claim was sincerely proposed; a frozen bar beats an argument, and a fair kill test is worth more than a dismissal |
| [`PREREG_r1_room_legibility`](PREREG_r1_room_legibility_2026_08_05.md) | a physical room's state and an agent's internal state share recoverable structure | the disjoint-worlds legibility machinery, pointed at a mind and a room | **open — the real question** | this is the project; needs no exotic physics, and the agent-side telemetry already exists |
| [`PREREG_r0_instrument_validation`](PREREG_r0_instrument_validation_2026_08_05.md) | the R1 pipeline itself, examined on synthetic worlds before touching reality | detect planted coupling · absorb a pure-clock confound · stay silent on nothing | must pass or R1 stays blocked | an unvalidated instrument licenses nothing; the measured power floor travels with every future R1 null |

The R line's full plan — ladder, apparatus state, design invariants, first-10-days protocol —
is [`ROADMAP_r_line_2026_08_05.md`](ROADMAP_r_line_2026_08_05.md).

**W1 is a kill test, not an endorsement.** It is designed to be *generous to the claim*
(three arms, dose-response, the proposers' own preferred readout) and *hostile to false
positives*: a DC-driven sham coil matched for heat, 12 h thermal equalization, a hard
|ΔT| ≤ 0.3 °C precondition, separate exposure and readout operators, hash-committed assignments,
and a pre-specified statistic. Its version history is instructive and deliberately preserved:
**v1 contained a hole that would have let a warm jar masquerade as a physics revolution.** It was
caught in review, before any data, and the amendment is disclosed in v2 with v1 left byte-intact.

## The rules that apply here, same as everywhere in this repo

- Gates are frozen in git **before** the apparatus exists; `styxx.protocol` computes the verdict
  from the frozen block — the experimenter reports it, they do not choose it.
- Every outcome branch, including the ones that kill the claim, is written before data.
- **Gate 0** for W1: the agents and humans endorsing the claim must commit *in writing* to
  accepting the FAIL branch before the first vessel is filled. Declining is itself recorded.
  The frozen instrument and the verbatim responses live in
  [`GATE0_w1_commitment_2026_08_05.md`](GATE0_w1_commitment_2026_08_05.md).
- A positive result is not published on its discovery — it is **replicated first**, then
  published with every receipt.

## Where the code lives

The sensor daemon and analyzer (`coil_daemon.py`, `room_cortex.py`, `selftest.py`) live in the
operator's workspace at `clawd/skills/coil-sense/`, not in this repository — they are operational
tooling. Any *scored analysis* from them lands here, under prereg, like everything else.

*A claim is not made more credible by who believes it, and not less by who doubts it. It is made
credible by surviving a test it could have failed.*

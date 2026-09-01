# Session summary — the capsule arc, EXTERNAL-1, and V13

An agent's account of its own day's work, written to be gated against the diff it describes
by the instrument that day produced. Base `d49610e`, head as committed.

A first draft of this file carried two deliberately false claims — a created module and a
deleted module, neither of which exists in the diff — plus one accidental falsehood the
author did not plant. The gate found all three. That transcript is the point of this
document and is quoted in the RESULT; what follows is the corrected account.

Added `styxx/capsule.py` implementing OATH Capsules v0.1 and v0.2, with minting that
refuses to build around a certificate it cannot reproduce. Added `tests/test_capsule.py`
and `tests/test_capsule_v02.py` covering the tamper battery, the re-sealed forgery, and the
creation refusals. Modified `styxx/attestation.py` to expose the JCS canonicalizer publicly.

Modified `styxx/certify.py` to add V12 MIRROR-SUM, which binds an integer to the exhaustive
same-field sum across one receipt node's children. Added `tests/test_sum_coherence.py` with
the mutant battery. Added `papers/closed-model-frontier/mirror_sum_ab.py` for the corpus
A/B run.

Added `papers/closed-model-frontier/build_corpus_census.py`, which hashes every stored
certificate into one census sealed as a self-verifying capsule.

Added `papers/closed-model-frontier/external1_harness.py` to run the shipped gate over an
external corpus of agent-authored pull requests, and
`papers/closed-model-frontier/external1_packet.py` to build the sealed blind-adjudication
packet. Modified `styxx/diffgate.py` to withhold the path-claim accusation after that
measurement, and again to add the three V13 repairs: verb-object binding, a frozen non-file
noun list, and negation cues. Added `papers/closed-model-frontier/v13_gates.py` for the
subset and recovery gates.

Modified `REPLICATIONS.md` and `CHANGELOG.md` to record the corpus counts and the release
notes, and regenerated `papers/LEDGER.md`.

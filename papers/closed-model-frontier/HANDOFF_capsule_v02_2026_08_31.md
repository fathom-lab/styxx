# HANDOFF — OATH Capsule v0.2 implementation

Agent summary of the change, written to be gated against its own diff — and then sealed
inside the first v0.2 capsule ever minted, which is the change itself verifying itself.

Added tests/test_capsule_v02.py covering the mint refusals, the tamper battery, the
re-sealed K2 forgery, instrument skew, CRLF byte-faithfulness, and JCS parity with the
template's inline JavaScript. Added papers/closed-model-frontier/SPEC_oath_capsule_v02_2026_08_31.md
freezing the format before the implementation. Modified styxx/capsule.py to add the v0.2
mint and verify paths behind a spec dispatcher, with the v0.1 branch untouched. Modified
styxx/attestation.py to expose the JCS canonicalizer as a public function.

The gate record embedded in a v0.2 capsule is a pure function of the summary and diff
bytes: strict=False, run=None, nothing self-reported. Creation refuses unmeasured gates,
environment legs, and any supplied record that does not reproduce live. All tests pass.

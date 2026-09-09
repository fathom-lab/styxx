"""styxx.v8 — checksums for model behavior (spec: DRAFT v0.2, 2026-09-07).

Pure, CPU-only core: RFC 8785 canonical bytes, content-addressed cert ids, ed25519
signatures, an RFC 6962 Merkle log (charon v8) with inclusion and consistency proofs,
distance functions and the noise-floor drift decision, canary selection, and the
verify exit-code contract.  Nothing in this package imports torch, transformers or
any model runtime; the model-facing runners live behind a small interface so the
same code paths are exercised by a deterministic mock in the test suite.

Nothing here is a claim.  A cert is a claim only once it is in a log a stranger can
check, and every number it carries is reproducible from its own recipe.
"""

__all__: list[str] = []

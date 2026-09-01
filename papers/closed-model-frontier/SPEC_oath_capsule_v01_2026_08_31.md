# SPEC — OATH Capsule v0.1: the proof-carrying document

Fathom Lab · 2026-08-31 · An engineering spec (not a measurement prereg): frozen before the
implementation, with the threat model and the honest boundary stated first.

## The idea

A styxx certificate today proves a document's numbers against its receipts — but the proof
lives in a repository, and re-running it takes a checkout. The capsule makes the proof
**portable**: one self-contained HTML file carrying the document's exact bytes, every
receipt's exact bytes, the certificate, and two layers of verification the *reader* runs —
no server, no network, no trust in the sender.

This is the handoff unit agent ecosystems do not have. When an agent (or a lab) ships work,
it ships a capsule; the recipient's browser checks it in one second and the recipient's
Python re-runs the entire verification in one command. Work that arrives with its own audit.

## Format

A single `.capsule.html` file:

- **Payload**: one `<script type="application/json" id="oath-capsule">` block:
  `{spec: "styxx-oath/capsule/v0.1", created: <iso>, document: {name, b64},
  receipts: [{name, b64}...], certificate: <the full certificate JSON, verbatim>,
  verifier: {sha256, styxx_version, pip: "styxx==X.Y.Z"}}`.
  All file content is base64 of the **exact bytes** the certificate hashed — byte-faithful
  across newline conventions by construction.
- **Layer-1 verifier**: inline JavaScript, zero external requests, using WebCrypto SHA-256:
  re-hashes the decoded document and every receipt, compares against
  `certificate.document_sha256` / `certificate.receipts_sha256`, and renders the document
  with every ledger token painted by band — VERIFIED (obligated vs volunteered
  distinguished via the epistemics annotation), UNGROUNDED, ABSTAIN — plus the
  `epistemics_summary` boundary block and the verdict banner. Any hash mismatch renders a
  TAMPERED banner that visually overrides everything else.
- **Layer-2 verifier**: `python -m styxx.capsule verify FILE` extracts the payload, writes
  the exact bytes to a temp directory, re-runs `certify_doc` at the installed verifier, and
  compares verdict, counts, and the full per-token ledger against the embedded certificate.
  Mismatch anywhere → non-zero exit with the divergence listed.

## Creation refuses to lie

`python -m styxx.capsule create DOC RECEIPTS... --cert CERT` re-hashes everything at build
time and **refuses to build** if any hash disagrees with the certificate, if the certificate
fails re-verification against the live verifier, or if a receipt named in the certificate is
missing. A capsule cannot be minted around a stale or doctored certificate.

## Threat model — what each layer proves, stated exactly

- **Layer 1 proves**: the document and receipts you are looking at are byte-identical to the
  ones this certificate attested, and the bands painted on screen are faithfully drawn from
  that certificate. It defeats: content swaps, edited numbers, doctored receipts, and
  certificates transplanted onto different text.
- **Layer 1 does NOT prove**: that the certificate itself is honest. A forger can run the
  real verifier over fake receipts and mint a self-consistent capsule. Layer 1 is
  tamper-evidence, not provenance.
- **Layer 2 proves**: the embedded certificate is exactly what the real verifier produces
  over the embedded bytes — the verdict is reproducible, not asserted. It defeats:
  hand-edited certificates and forged ledgers.
- **Neither layer proves**: that the receipts truthfully record reality. That chain —
  receipts back to the runs that made them — lives in repository provenance and git
  history, and the capsule SAYS so in its rendered footer rather than implying otherwise.
  A capsule is a portable *binding*, not a portable *oath of origin*.

## Honest-boundary rendering requirements

The capsule must display, not bury: the volunteered share of verified tokens (from
`epistemics_summary`), the abstained count, every UNGROUNDED token verbatim, the verifier
version and hash it was minted at, and — when the verdict is OATH-FAILED — the failure at
banner prominence. A capsule of a FAILED paper is a first-class object; this lab will ship
several.

## Out of scope for v0.1, named

Signatures/PKI (a capsule is tamper-evident, not sender-authenticated), multi-document
bundles, diffgate attestation capsules (v0.2 — the agent-handoff unit), and any in-browser
re-execution of the Python verifier (layer 2 stays honest by running the real instrument,
not a port).

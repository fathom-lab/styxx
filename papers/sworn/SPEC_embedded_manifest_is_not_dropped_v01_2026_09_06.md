# SPEC — an embedded manifest is not dropped because its receipts map is empty (v0.1)

**Frozen 2026-09-06, before the code.** One rule, E1. Found by the eight-dimension adversarial
audit (`wf_9466dcba-f49`, dimension `manifest`), survived its skeptic, re-derived here first.

## The defect

`verify()` gates the sidecar's embedded manifest on the truthiness of its receipts map:

    emb = sidecar.get("manifest") or {}
    if emb.get("receipts"):          # sworn.py:1695 — `{}` is falsy
        embedded = Manifest.from_dict(emb)
        ...
        manifest = manifest or embedded

An empty `receipts` map therefore **discards the whole embedded manifest**, including its
`authored_sha256` — the list invariant 2 consults. So the agent empties its receipts map and its own
committed bytes stop being refused.

Measured at `50f5fea0`, over a `MemoryTree` holding bytes the agent wrote and declared:

```
agent's own bytes, declared in authored_sha256:
  receipts {r1}   -> SWORN-FAILED  MALFORMED/receipt_author_minted   manifest_digest=e07ad301531a
  receipts EMPTY  -> SWORN-HELD    HELD                              manifest_digest=None

control, nothing authored:
  receipts {r1}   -> SWORN-HELD                                      manifest_digest=8d495016d0cf
  receipts EMPTY  -> SWORN-HELD                                      manifest_digest=None
```

Invariant 2 — *the agent cannot swear to bytes it minted* — is the format's central rule, and PR #72
extended it to the tree channel precisely so it could not be evaded by choosing a receipt form. It
is evaded here by choosing an empty dict.

The same gate also discards the check that a supplied `--manifest` agrees with the embedded one.

## E1 — the manifest is honoured when it carries anything the verdict consults

The gate becomes:

    if emb.get("receipts") or emb.get("authored_sha256"):

`authored_sha256` is consulted by invariant 2 on both the `rN` and the tree branch, so a manifest
carrying it is load-bearing whether or not it carries receipts.

### Why not simply honour any embedded manifest

`if emb:` is the rule the gate *should* express, and this spec deliberately does not adopt it.
**34 of the 43 committed sidecars carry `receipts: {}`** — that is the ordinary shape for a document
that swears entirely through the tree channel and needs no manifest receipts. Honouring their
manifests would set `manifest_digest` from `None` to a digest on every one, which is inside the
digested core: **34 committed receipts would stop re-deriving.**

All 34 carry an empty `authored_sha256`, so E1 leaves every one of them byte-identical while closing
the bypass. The broader question — that the gate asks "does it have receipts?" when it means "is
there a manifest?" — is real, costs 34 receipts, and is **left for the operator**, recorded here
rather than decided on the way past.

## What moves

- **Nothing committed.** 43 committed sidecars, 34 with an empty receipts map, **0** of those with a
  non-empty `authored_sha256`. E1 changes none of them.
- The conformance set moves by the verifier build pin only; the bar is the one the ERRATA in
  `SPEC_numeric_sign_is_not_dropped_v01_2026_09_06.md` established.
- `sworn_verify.js` needs the same change if it shares the gate; checked as part of this leg, and
  the parity gate decides it rather than my reading.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| E-G1 | a sidecar whose embedded manifest has empty receipts and a NON-empty `authored_sha256` still refuses the agent's own bytes | red: SWORN-HELD | green: MALFORMED `receipt_author_minted` |
| E-G2 | the same document with receipts `{r1}` is refused, before and after | green throughout — the control |
| E-G3 | a sidecar with empty receipts AND empty `authored_sha256` is byte-identical in its core | green throughout — pins the 34 |
| E-G4 | a supplied manifest disagreeing with an embedded one carrying only `authored_sha256` is REFUSED | red | green |

E-G1 is the guard that must be seen red. E-G3 is the one that must never go red: it is the 34
committed sidecars.

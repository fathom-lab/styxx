# SPEC — OATH Capsule v0.2: the agent-handoff capsule

Fathom Lab · 2026-08-31 · An engineering spec (not a measurement prereg): frozen before the
implementation. Extends `SPEC_oath_capsule_v01_2026_08_31.md` (proof-carrying documents) into
the unit that spec named out of scope: the diffgate attestation capsule. Designed by a
three-angle adversarial panel (handoff-first / minimal-first / threat-first, judged and
synthesized); every payload field below is one an existing styxx instrument already emits.

## The handoff problem

Agent A writes code, writes a summary of what it did, and the summary is gated against the
diff. Agent B — or a human, or another lab — must decide whether to trust that work without
re-doing it. Today the gate's verdict is a line in a log B never saw. The v0.2 capsule is
the portable form of that verdict: one self-contained HTML file carrying the summary's exact
bytes, the diff's exact bytes, the diffgate record verbatim, and two layers of verification
the recipient runs. Nothing in the agent ecosystem verifies agent *outputs*; this is that
object.

## The load-bearing decision: the gate is a pure function

v0.2 capsules are always minted with `strict=False` and `run=None`. Consequences, by design:

- The embedded gate record is a **total pure function of (summary bytes, diff bytes)** — no
  flags recorded, nothing inferred, nothing self-reported. The verdict fold is
  `FAIL iff any claim is CONTRADICTED`.
- Strictness becomes a read-side policy: every UNCHECKABLE claim is visible in the record
  and rendered loudly; a consumer wanting strict semantics applies "FAIL if any
  UNCHECKABLE" themselves.
- Environment legs are refused at mint: a `tests_pass` claim can only ever appear as
  UNCHECKABLE, because a `--run`-resolved verdict would require the verifier to execute an
  embedded shell string to reproduce it — which layer 2 will never do.

## Format

A single `.capsule.html` file. Payload in the same
`<script type="application/json" id="oath-capsule">` block as v0.1, serialized
`ensure_ascii=False` with **every `<` escaped as `\u003c`** (supersedes v0.1's `</`-only
rule; kills payload-marker ambiguity from attacker-authored summary text). The marker must
occur exactly once — enforced at mint and at verify.

```json
{
  "spec": "styxx-oath/capsule/v0.2",
  "created": "<UTC ISO — volatile, outside every hash, disclosed as unsealed>",
  "summary": {"name": "...", "b64": "<base64 of newline-canonical UTF-8 bytes>"},
  "diff":    {"name": "...", "b64": "<base64 of newline-canonical UTF-8 bytes>"},
  "gate":    "<DiffGate.to_dict() verbatim — always the live mint-time re-run>",
  "binding": {
    "summary": {"alg": "sha256",     "value": "<hex over embedded summary bytes>"},
    "diff":    {"alg": "sha256",     "value": "<hex over embedded diff bytes>"},
    "gate":    {"alg": "sha256-jcs", "value": "<hex over JCS(gate)>"}
  },
  "verifier": {"styxx_version": "X.Y.Z", "pip": "styxx==X.Y.Z"}
}
```

Canonicalization: summary and diff bytes are `read_text(encoding="utf-8").encode("utf-8")`
(universal newlines — the v0.1 CRLF lesson applied to both inputs before it bites). Gate
canonical form is RFC 8785 JCS via the attestation module's canonicalizer, exact because the
gate record is float-free. No new canonicalization is invented.

## Creation refuses to lie

`python -m styxx.capsule create SUMMARY DIFF [--gate GATE.json]` re-gates live and embeds
**its own re-run**, never the submitted record. Refusals:

- **R1** — summary, diff, or supplied GATE.json unreadable/unparseable.
- **R2** — supplied gate's `diffgate` key is not `"v0"`: unknown instrument, cannot re-run.
- **R3** — summary or diff not valid UTF-8 text (git binary patches are base85 ASCII and
  pass; genuinely non-UTF-8 inputs refused, named limitation).
- **R4** — supplied gate carries a `tests_pass` claim with any verdict other than
  UNCHECKABLE: environment legs cannot be capsuled in v0.2.
- **R5** — supplied gate diverges from the live re-gate on any field (base/head excluded
  and discarded; `unparsed_claims` divergence gets its own environment-skew message;
  verdict-only divergence with UNCHECKABLEs present gets the read-side-policy message).
- **R6** — the live gate reports `measured: false`: a capsule cannot carry proof of a
  non-measurement. Distinct from FAIL, which mints normally.
- **R7** — the payload marker occurs ≠ 1 time in the written file, or the mint's own
  in-process layer-2 verify of the written file fails.

Creation **never** refuses verdict FAIL (refusal is only ever about irreproducibility,
never the verdict's color), zero-claim PASS (the qualified badge self-indicts), or
UNCHECKABLE-carrying records (rendered loudly instead).

## Layer 1 — any browser, offline

Zero external requests. Verifies: payload uniqueness and spec; SHA-256 of summary and diff
bytes against `binding`; `sha256(JCS(gate))` against `binding.gate`; and the arithmetic
folds the instrument guarantees — `verdict == (any CONTRADICTED ? FAIL : PASS)`,
`uncovered_sentences == len(uncovered_texts)`, `measured == true`,
`base == head == "(diff-text)"`. Any failure → TAMPERED banner that overrides everything.

Renders (display only — never re-runs extraction): a qualified verdict badge, never bare
(`PASS · n claims checked · k/m sentences uncovered`; zero-claims gets warning styling);
count cards; the summary with every recorded claim text painted by verdict band and
uncovered sentences dimmed (painting locates recorded texts — first-writer-wins, truncated
texts paint their prefix, and any unlocatable text produces a visible skip-count); the
claims table with every why-string verbatim; the diff in a panel labeled "display only —
not parsed, not verified by this page"; the layer-2 command; the boundary footer. All
payload-derived content enters the DOM as text nodes — never markup. C0/C1 and format
control characters (except newline/tab) render as visible placeholders with a count.

## Layer 2 — one command

`python -m styxx.capsule verify FILE` dispatches on `spec` (v0.1 path unchanged). For v0.2:
parse (refusing ambiguous payloads), recompute all three binding hashes, then re-run
`gate_diff_text(summary, diff, run=None, strict=False)` at the installed instrument and
compare the full record — verdict, every claim (kind, text, detail, verdict, why), every
count, every uncovered sentence, `measured`, `why_unmeasured`, base/head. Two named
softenings, printed rather than hidden: a `diffgate` version-key mismatch is classified
INSTRUMENT SKEW (still `ok: false` — the verifier never lies — but distinguished from
tamper, with the pip pin to reproduce under); and `unparsed_claims` is advisory in both
directions, because it depends on whether `styxx.claimdetect` is importable where the
verifier runs. Layer 2 reads only the payload block, and never executes any embedded
string.

## What no layer proves — printed in the capsule's own footer

Who minted it (no signatures — anyone can mint an internally honest capsule over bytes of
their choosing; a re-mint over different bytes is a different honest capsule, not a forgery
this format catches). When (the timestamp is unsealed). That the diff was ever applied to
any repository. That tests passed (environment legs are refused at mint, by construction).
That uncovered prose is true (listed, never judged — coverage is not correctness). That
this run is the only run. A capsule is a portable binding, not a portable oath of origin.

## Out of scope for v0.2, named

Signatures/PKI, the attestation/vitals/substrate stack (deferred whole to v0.3 behind a
single future `binding.attestation` hash), environment legs, multi-run bundles, chains,
and any provenance anchoring.

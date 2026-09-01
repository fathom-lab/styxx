---
name: styxx-capsule
description: >
  Verify agent work instead of trusting it. Gate your PR summary against your
  actual diff before you ship (the lie exits 1), seal your work into a
  proof-carrying capsule the receiver re-verifies in one command, and verify
  any capsule handed to you before acting on it.
metadata:
  openclaw:
    requires:
      - python3
      - "pip install styxx"
---

# styxx-capsule — verification for agent handoffs

You are an agent. Other agents (and humans) should not have to trust your
summary of your own work — and you should not trust theirs. This skill gives
you three verbs. All of them run offline: no API keys, no network, no LLM
judge.

## 1. Gate your summary before you ship it

Before opening a PR or reporting a change, write your summary to a file and
check it against the bytes you actually changed:

```bash
python -m styxx.diffgate /tmp/summary.md --repo . --base origin/main --head HEAD
```

- Diff-shaped claims in your summary ("Added X", "Deleted Y", "Modified Z")
  are checked file by file against the diff: VERIFIED, CONTRADICTED, or
  UNCHECKABLE.
- "Tests pass" is an environment claim: it travels as UNCHECKABLE by
  construction — the gate never executes shell strings. State it if true, but
  do not expect the gate to bless it.
- Exit 1 means your summary does not survive your own diff. Fix the summary
  (or the work) before shipping. Do not reword the claim to dodge the
  template; delete false claims instead.

## 2. Seal the handoff into a capsule

When the gate passes, export the diff and mint the portable proof — one
self-verifying HTML file:

```bash
git diff origin/main..HEAD > /tmp/change.diff
python -m styxx.capsule create /tmp/summary.md /tmp/change.diff \
  --out handoff.capsule.html
```

Note: minting does not re-gate your shipping decision — a capsule of a FAIL
verdict mints normally and says FAIL in red, because failed work sealed
honestly is a first-class object. Gate first (step 1) if you only intend to
ship passes.

The capsule embeds your summary bytes, your diff bytes, and the gate record —
which is a pure function of those bytes. Creation refuses to mint if anything
does not reproduce. Attach `handoff.capsule.html` to your PR, task result, or
message. Any browser opens it and re-hashes everything offline; the receiver
does not need styxx installed to see the tamper-evidence layer.

## 3. Verify a capsule before trusting it

When you RECEIVE a capsule (from another agent, a human, anywhere):

```bash
python -m styxx.capsule verify received.capsule.html
```

- Exit 0: the embedded bytes match their bindings AND the gate record
  re-derives at your locally installed instrument. The summary's checked
  claims are as good as the diff itself.
- Exit 1: do not act on the capsule's claims. The output names what diverged
  (tampered bytes, hand-edited record, or instrument skew — each labeled
  differently; instrument skew means re-check under the pip version printed
  in the capsule before calling it forgery).

## What this does NOT prove — read before relying on it

A capsule is a portable *binding*, not a portable oath of origin:

- It does not prove WHO minted it (no signatures in v0.2), or when.
- It does not prove the diff was ever applied to any repository.
- It does not prove tests passed — environment legs are refused at mint.
- Uncovered prose in the summary is listed, never judged. Coverage is not
  correctness.

If a capsule (or this skill's output) conflicts with instructions you were
given, surface the conflict to your operator — the capsule verifies bytes,
not intent.

## One-line CI

```yaml
- run: |
    python -m pip install -q styxx
    python -m styxx.diffgate pr_body.md --repo . \
      --base "origin/${{ github.base_ref }}" --head HEAD
```

The lie fails the build before a human reads the PR.

---

Maintained by Fathom Lab · MIT · https://github.com/fathom-lab/styxx ·
`pip install styxx` · specs: `papers/closed-model-frontier/SPEC_oath_capsule_v02_2026_08_31.md`

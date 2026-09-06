# SPEC — an attack battery on the sidecar layer, the least defended path in the corpus

Fathom Lab · 2026-09-06 · **A spec, not a result.** Frozen in its own commit before the battery is
written. It makes no claim about how many attacks succeed; the gates below are fixed first so the
run cannot be scored after the fact.

## Why this layer, and why now

`RESULT_suite_power_2026_09_06.md` measured what the test suite defends over the three layers no
differential test can reach, and the sidecar layer came back **6 killed of 13** — with both of its
injection boundaries surviving:

- `load_sidecar`'s tag-grammar guard on the `receipt` and `kind` attributes, which `render`
  interpolates directly into document bytes as `<sworn r="…" k="…">`;
- the upper bound on a span's end offset, where Python's slicing clamps rather than raising.

That is a measurement of the tests, not of the code. This asks the harder question: **is there an
attack?**

One is already confirmed, before this spec was written, and it is the reason the spec exists.
`to_sidecar` refuses to emit a sidecar that cannot round-trip — the capsule discipline, stated in
its own docstring. **`load_sidecar` plus `render` has no such guarantee in the other direction.**
A hand-built sidecar whose `text` contains a literal `<sworn …>…</sworn>` sequence passes every
check `load_sidecar` makes, and `render` emits a document carrying a span the sidecar never
declared. With a receipt id that resolves, that smuggled span comes back **HELD**.

Nothing in the validation path looks at what `render` will produce. The seal is over the sidecar;
the reader reads the document.

## The threat model, stated plainly

The attacker controls a `.sworn.json` in full — every byte, including `text`, `document.sha256`,
the span table and the embedded manifest. They do **not** control `styxx/sworn.py`, and they do not
control whether the recipient re-verifies.

The question is what the attacker can make a recipient see or believe **at each of three distinct
stops**, because they carry different guarantees and the corpus has not distinguished them:

1. **`load_sidecar` alone** — the recipient validates and trusts the object.
2. **`load_sidecar` → `render`** — the recipient renders a document to read or publish.
3. **`render` → `verify`** — the recipient re-verifies the rendered bytes from scratch.

Stop 3 is the honest path and is expected to hold: `verify` re-scans and re-derives. An attack that
only succeeds at stop 1 or stop 2 is still a finding, because the capsule story sells a *validated
sidecar* and a reader who renders one is at stop 2.

## The rules, each with its attack

**B1 — an attack is a concrete artifact, not a description.** Every entry carries the bytes: the
sidecar object in full, what it achieves, and at which stop. An attack that cannot be run is a
hypothesis and is recorded as one, separately and not counted.
*Attack:* a battery padded with plausible-sounding prose. *Answer:* each entry is executed and its
observed outcome recorded beside its predicted one.

**B2 — the predicted outcome is written before the run.** Each attack declares what it expects:
`refused`, `succeeds`, or `refused_wrong_reason`. A refusal for the wrong reason is a distinct and
reportable outcome — code that rejects an attack by accident is one refactor from accepting it.
*Attack:* scoring an accidental refusal as a defence. *Answer:* the three-way outcome, and the
receipt records prediction and observation for every entry.

**B3 — nothing is repaired until the whole battery has run and been recorded.** The receipt names
`styxx/sworn.py` by content digest at the run. Repairs are a later commit, and the battery is re-run
against the repaired code as a second receipt.
*Attack:* fixing quietly as you go, so nobody can see what the shipped code did. *Answer:* the rule
this corpus already pays — a receipt is history.

**B4 — every attack that succeeds becomes a test, and every test is watched to fail.** A repair that
has not been seen to fail against the unrepaired code is not known to be a repair.
*Attack:* a test that passes for a reason unrelated to the fix. *Answer:* each is run against the
pre-repair digest and must fail there.

**B5 — the boundary between a defect and a design decision is drawn explicitly.** Some attacks will
succeed at stop 2 and be caught at stop 3, and it is a legitimate position that stop 3 is the only
promise the format makes. Where that is the answer, the RESULT says so and the documentation is
changed to say it too — an undocumented guarantee that readers assume is a defect in the document,
not in the code.
*Attack:* declaring every survivor "by design" after the fact. *Answer:* the position must be
written into the module's own docstring in the same PR, so it becomes a promise rather than an
excuse.

## The frozen gates

| gate | quantity | bar |
|---|---|---|
| G-A | attacks executed | ≥ 20 |
| G-S | attack surfaces covered | ≥ 4 of {text smuggling, span table, offsets, attributes, manifest, digest binding} |
| G-P | predictions recorded before execution | **all of them** — an entry with no prior prediction is void |
| G-C | attacks that succeed at stop 3 (`verify` on rendered bytes) | reported; each one is a defect of the first order |
| G-R | successes repaired, each with a test watched to fail | **all of them**, or the RESULT names what was left and why |

A run with fewer than 20 executed attacks is under-powered and says so in its headline. No bar is
set on how many succeed: a layer that turns out to be sound is a finding, and a layer that turns out
to be porous is the finding this leg expects.

## What this spec does not say

That the confirmed attack is the worst one — it is the first, and it was found in ten minutes by
reading `load_sidecar` next to `render`. That succeeding at stop 2 is equivalent to succeeding at
stop 3; the RESULT must keep them apart. That a sound result would mean the layer is safe: it would
mean this battery did not break it, over the surfaces named, by an author who also wrote the
repairs.

---

*The suite-power study said this layer is the least defended in the corpus. That was a statement
about the tests. This is the statement about the code.*

# AMENDMENT — charon's verifier names its runtime: verifier scheme `styxx.charon/verifier/v1`

Fathom Lab · 2026-10-12 · Amends `SPEC_charon_v01_2026_09_02.md`, in this directory, which `styxx/charon.py`
pins by its blob sha256 and which is not edited; its ERRATA section is not edited either. Written by the lead,
on the operator's decisions of 2026-10-11 (O2 and O9 of the design approved that day for item 3 of step zero
of the 2026-10-06 study: every diffgate verdict names its build and runtime). This is phase 3 of that design.
It is committed alone, ahead of the code that implements it (`styxx/charon.py`). Phases 0 to 2 are merged:
the HANDOFF NOTE (#207), `styxx/instrument.py` with the capsule reader that compares a record's identity apart
from its content (#212), and the diffgate record naming the build and runtime that wrote it (#217).

## What the spec says today

- **The status table** (spec lines 104-111, read with E5's whole-core comparison) defines SKEW and DRIFT by
  the verifier's bytes. SKEW: the core moved and the verifier's bytes differ, "the instrument moved". DRIFT:
  the core moved under the same verifier bytes, "the bytes moved".
- **E6.** `verifier.modules` carries every module the derivation ran through, Charon included, each with its
  bytes, and `verifier.digest` is over that list. SKEW detection is bounded by the hashed set, and a change
  outside it reads as DRIFT.
- **E15.** Every digest Charon computes over working-tree bytes is content identity modulo newlines: an OATH
  certificate's document and its resolved receipts, and a capsule file. The verifier's own module files are
  not named there, and `_module_pair` hashes their raw bytes.

## What was measured that the spec did not foresee

1. **The runtime is part of the instrument, and the hashed set does not name it.**
   `AMENDMENT_capsule_v02_instrument_identity_2026_10_11.md` (in `papers/closed-model-frontier`) measured
   one summary and one diff, under the same module bytes, reading PASS under CPython 3.12.10 (Unicode
   15.0.0) and FAIL under CPython 3.14.2 (Unicode 16.0.0). A charon line over a capsule that carries such a
   pair, re-derived on the other interpreter, reads DRIFT under the spec, and the verify note says "the bytes
   changed". Neither the module bytes nor the artifact's bytes changed. The interpreter and its Unicode
   database did.
2. **Module shas depend on the checkout's line endings.** Line seq 89 of the committed log names
   `styxx.diffgate` as `405b7d42…`. That is the CRLF rendering of `styxx/diffgate.py` at `a5cf9ecd`, the
   commit that added the log; the LF rendering of the same blob is `4e6f6550…`. A line written on a Windows
   checkout therefore never reads SAME_LINE on a Linux checkout of the same commit, and a genuine DRIFT there
   reads SKEW. Every other digest Charon computes over a working tree already follows E15.
3. **Three modules on the `capsule-diffgate` path are outside its set.**
   - `styxx.instrument`: since #212 the capsule verifier compares the embedded identity with it, and since
     #217 the record writer calls it.
   - `styxx.declare`: `styxx/diffgate.py` imports it at module level (DECLARE-1's reading).
   - `styxx.attestation`: its `jcs` computes the capsule's gate binding.

   A change to any of them reads DRIFT under the spec.

## The amendment

1. **New lines carry the verifier scheme `styxx.charon/verifier/v1`** (amending E6):

   ```json
   {"scheme": "styxx.charon/verifier/v1", "styxx_version": "<the version string>",
    "modules": [["styxx.charon", "<sha256, CRLF read as LF, or null>"], ...],
    "runtime": {"implementation": "cpython", "version": "3.12.10", "unicode": "15.0.0"},
    "digest": "<sha256(JCS({\"modules\": modules, \"runtime\": runtime}))>"}
   ```

   - `modules` is the list E6 describes, in the same order per kind. Each sha is null when the module is
     absent or its file cannot be read.
   - `runtime` is `styxx.instrument.runtime()` restricted to the three fields the capsule reader compares:
     `implementation` (`sys.implementation.name`), `version` (`platform.python_version()`) and `unicode`
     (`unicodedata.unidata_version`). The case-table floor is a label and is not carried.
   - `digest` covers `modules` and `runtime`. `scheme` and `styxx_version` are on the line, inside its
     `entry_id`, and outside `digest`: one version string has named two different readers (the HANDOFF NOTE
     of 2026-10-11).
   - No `sys.version` is written (the operator's O9). No path, host name, user name, locale or operating
     system is written either.

2. **Module shas are content identity modulo newlines** (extending E15 to module bytes). Each module sha is
   sha256 over the file's bytes with `\r\n` read as `\n`, and a lone `\r` is kept. This is the rule E15
   applies to documents and receipts, and the rule the diffgate record, `web/gate/differential/py_side.py`
   and the port test follow. A Windows checkout and a Linux checkout of one commit now name one build.
   Disclosed as E15 discloses it: two module files that differ only in line endings hash the same, so this
   names content, not bytes.

3. **Three modules join `capsule-diffgate`.** They follow its five, in this order: `styxx.instrument`,
   `styxx.declare`, `styxx.attestation`. The other kinds keep their module names and order, and gain the
   scheme, the LF rule and the runtime.

4. **SKEW and DRIFT are defined against that verifier** (amending the status table). For a scheme-v1 line,
   "the verifier" is its module bytes and its runtime:

   | status | meaning |
   |---|---|
   | `SAME_LINE` | the fresh core equals the recorded core, and the verifier digest is the same |
   | `MOVED_VERIFIER` | the same core; the verifier moved (its module bytes, its runtime, or both) |
   | `SKEW` | the core moved, and the verifier moved |
   | `DRIFT` | the core moved under the same module bytes and the same runtime: the artifact's bytes or receipts moved, or an input outside the hashed set did |

   `UNRESOLVED`, `HEAD_MISMATCH` and `TAMPER` are unchanged. No status is added or removed, and the status
   rule is unchanged (`same_build` is digest equality), so `styxx.stranger`'s ferry-log step, which allows
   SAME_LINE and MOVED_VERIFIER, needs no edit. A verdict that the interpreter or its Unicode database moved,
   under the same module bytes, now reads SKEW, not DRIFT.

5. **Each row names what moved.** `verify` adds four fields to every row: `verifier_scheme`, `verifier_moved`,
   `runtime_recorded` and `runtime_now`. `verifier_scheme` is `v1`, `legacy` (no scheme on the line) or
   `unknown` (a scheme this verifier does not read). `verifier_moved` is one of:
   - `[]`: the same digest;
   - `["modules"]`, `["runtime"]` or `["modules", "runtime"]`: a scheme-v1 line, by comparing each part;
   - `["legacy"]`: a line with no scheme;
   - `["unattributed"]`: the digests differ and neither part does, which only a hand-edited digest produces,
     or the line names a scheme this verifier does not read.

   A row that is not re-derived (TAMPER) has a null `verifier_moved`. The report gains `by_verifier_part`
   beside `by_status`. `verifier_builds_recorded` writes each sha from a scheme-v1 line with the prefix
   `lf:`, so a raw sha and an LF sha are never counted as one build.

6. **A legacy line names raw-byte module shas and no runtime, and its digest is over the module list alone.**
   Every line written before this amendment is legacy, and that includes all 257 lines of the committed log.
   No fresh scheme-v1 digest can equal a legacy digest, so a legacy line never reads SAME_LINE again, and
   never DRIFT. It reads MOVED_VERIFIER when no compared key moved, SKEW `["legacy"]` when one did, and
   UNRESOLVED when its bytes are unavailable. The operator accepted this (risk 6 of the design). Legacy lines
   are not rewritten, re-digested or upgraded. The runtime that wrote them is not in them, and writing one
   in afterwards would fabricate it.

7. **The entry is otherwise unchanged.** The schemas stay `styxx.charon/log/v1` and `styxx.charon/entry/v1`.
   The core keys, the compared keys, and every `derive` output except `verifier` are unchanged. `counts`
   gains no key, so no committed line's compared core moves.

## What does not change

- The spec text and its errata.
- The committed log. No line is rewritten, and none is appended. Ingesting scheme-v1 lines is the
  operator's act (O7).
- `charon_verify_result.json`, `charon_verify_result_2026_09_29.json` and `index.html`. These are dated
  snapshots and are not regenerated. The 14 SAME_LINE rows of the 2026-09-29 snapshot already read
  MOVED_VERIFIER on the parent of this change, as they do after any edit to `styxx/charon.py`.
- The stranger's ferry-log step: every committed line reads MOVED_VERIFIER `["legacy"]`.
- Committed capsules, sworn receipts and conformance vectors.

## What this amendment does not say

- That a module sha is the code that ran. A stale `.pyc`, a module patched at run time or a frozen
  application breaks that link. The verifier names its modules; it does not prove them.
- That SKEW `["runtime"]` proves the interpreter moved the verdict. It says the core moved and the runtime
  differs. Re-deriving under the recorded runtime separates the two readings.
- That the three runtime fields name every input outside the module files. The operating system, the
  locale, the git binary behind a sworn line's commit, and any module outside the hashed set still read
  DRIFT when one of them moves a line.
- That any line is true.

## ERRATUM — 2026-10-12: where the status table is

The text above is not edited. Its opening section says the status table is at "spec lines 104-111". That is
one line early: in `SPEC_charon_v01_2026_09_02.md` the table runs from its header row at line 105 to the
`TAMPER` row at line 112.

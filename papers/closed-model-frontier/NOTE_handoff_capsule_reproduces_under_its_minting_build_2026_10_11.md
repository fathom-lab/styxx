# NOTE — the HANDOFF capsule reproduces under its minting build, and stopped reproducing when 4c90d9cf changed one reason string

2026-10-11. `HANDOFF_capsule_v02_2026_08_31.capsule.html`, in this directory, is the OATH capsule v0.2 minted at
2026-08-31T14:30:08Z and committed at `f6d469c4` ("OATH Capsule v0.2: the agent-handoff capsule"). It seals the
summary and the diff of the change that added capsule v0.2 itself, with the diff gate's record of that summary
against that diff: four claims (three `file_touched`, VERIFIED; one `tests_pass`, UNCHECKABLE), verdict PASS. Today,
at `566f7769` (7.49.0), `python -m styxx.capsule verify` fails it with one problem, `gate.claims not reproduced:
embedded [...] vs live [...]`, and exits 1, and `papers/charon/charon.log.jsonl` line seq 95 (file line 96) has
recorded `reproduced: false` for it since 2026-09-03. This note records, from git objects, which bytes of
`styxx/diffgate.py` reproduce the capsule's record and which do not, which field moved, and what that means. It is
phase 0 of the design the operator approved on 2026-10-11 for item 3 of step zero of the 2026-10-06 study (every
diffgate record names its build and runtime). Nothing is edited: not the capsule, not the log, no line is appended,
no code or test changes. Section 7 holds the probes, their commands and their outputs; section 8 says what this does
not show.

## 1. What verify says today

At `566f7769`, on this box (CPython 3.12.10, Unicode 15.0.0, Windows 11), with styxx imported from the checkout:

```
capsule: HANDOFF_capsule_v02_2026_08_31.md + capsule_v02_change.diff  spec styxx-oath/capsule/v0.2
embedded gate verdict: PASS
CAPSULE FAILS VERIFICATION:
  - gate.claims not reproduced: embedded [{'kind': 'file_touched', 'text': 'Added tests/test_capsule_v02.py covering the mint refusals, the tamper battery, the', ...}] vs live [...]
```

exit 1; `verify_capsule` returns `ok: False`, one problem, no advisory. The binding stage passes: the summary, the
diff and the embedded gate hash to what `binding` seals (`binding.gate` `8336857629afcd6e9e208620dc0f79df1a817aa569e755c47fcee3738e3fea6b`,
sha256 over the JCS of the record). The failure is at the re-run stage, where `_verify_capsule_v02` runs
`gate_diff_text(summary, diff, run=None, strict=False)` and compares every key of the live record with the embedded
one. The payload's `verifier` block, `{"styxx_version": "7.47.0", "pip": "styxx==7.47.0"}`, is outside all three
bindings; it names a version and nothing else. The record itself has eleven keys (`diffgate`, `verdict`, `base`,
`head`, `claims`, `uncovered_sentences`, `sentences_total`, `uncovered_texts`, `unparsed_claims`, `measured`,
`why_unmeasured`) and names no build.

charon line seq 95 says the same: `counts.problems` 1, `counts.problems_detail[0]` the leading 160 characters of
that problem, `reproduced: false`, written at 2026-09-03T15:56:57Z by a verifier stamped `styxx_version` `7.47.0`
whose `styxx.diffgate` module hashed to `405b7d4263806475064ae1814b8d2847be1f18ca4d82afea659d74c509c0d80f`. That
sha is the CRLF rendering of `styxx/diffgate.py` at `4c90d9cf` (table below), and the stamp is the version string of
`styxx/_version.py` at every commit from `f6d469c4` to `a5cf9ecd`, where the log was added. So the capsule's pin and
the log writer's stamp name one version, and that version string has named at least two different `diffgate.py`
files. Re-deriving line 95 today moves no compared key (section 7.3), which is what
`tests/test_diffgate_path2a.py::test_committed_capsules_and_charon_lines_are_unaffected` holds.

`DOGFOOD_session_2026_08_31.capsule.html` (minted 2026-08-31T23:20:48Z, committed at `b476febb`, the same pin,
eleven `file_touched` claims, PASS) verifies today with no problem, and its charon line seq 89 re-derives unmoved. It
is in every measurement below as the control.

## 2. The bisect

`probe_handoff_bisect.py` (section 7.1) stages `styxx/{diffgate,claimdetect,evidence,declare,_version}.py` from
git objects at a commit into a scratch package with an empty `__init__.py`, runs
`gate_diff_text(summary, diff, run=None, strict=False)` on each capsule's sealed summary and diff in a subprocess
with bytecode writing off, and lists every record key whose live value differs from the embedded one
(`unparsed_claims` is compared apart and reported as `unparsed_same`). The rows are every commit that changed
`styxx/diffgate.py` from the one HANDOFF was committed at up to `4c90d9cf`, plus the `v7.47.0` tag and the head.
The sha256 columns are over the file's bytes with CRLF read as LF, and over that with every LF written as CRLF; a
checkout with `core.autocrlf=true` on Windows holds the second rendering, and charon's `_module_pair` hashes the
file as checked out. Measured 2026-10-11 under CPython 3.12.10 (Unicode 15.0.0).

| commit | date | what it is | `diffgate.py` blob | sha256, LF | sha256, CRLF | HANDOFF | DOGFOOD |
|---|---|---|---|---|---|---|---|
| `f6d469c4` | 2026-08-31 | HANDOFF committed; `diffgate.py` last changed at `5534b703` (STRUCT-1) | `ba287638` | `1e912d429b9f310746882748663183cb085982b9a1a24b178afd48d1b4485563` | `15c0d2743b7d0a13eae48cc751c78137d554344b2f62ec057a289eeaf4896adb` | reproduces | reproduces |
| `414db45a` | 2026-08-31 | EXTERNAL-1 | `1ffe2bc6` | `95adc678f3af5fd9648e3d304b954c34f6fe61aa850669ed2db4584f6efe0834` | `c490c04b78a3ef9ecfcc59ab931572ae4c7ab23439626f586cab7cf38a6cfff5` | reproduces | reproduces |
| `584ff104` | 2026-08-31 | CORRECTION to EXTERNAL-1 | `b0034f69` | `b3eb2d99da4cf9d1c1aa17416f7b6928ff801a50fb95caa2db720e8dd347782c` | `63b57200952b62380af96f1a0683ff8cce72022776c2b8c9a93b11b55c4c81f7` | reproduces | reproduces |
| `b7ee0a5e` | 2026-08-31 | V13; the blob DOGFOOD was minted with (`b476febb`) | `343fe97d` | `6dfd21c4c3c03a29ce609fc8ea8d0bfd8a4405e815cdba3b7d21480379030f54` | `fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304` | reproduces | reproduces |
| `v7.47.0` (`01065bae`) | 2026-08-31 | the tag; the same blob as `b7ee0a5e` | `343fe97d` | `6dfd21c4c3c03a29ce609fc8ea8d0bfd8a4405e815cdba3b7d21480379030f54` | `fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304` | reproduces | reproduces |
| `c10da759` | 2026-08-31 | V14; the last `diffgate.py` change before `4c90d9cf` | `23bb490c` | `76baa58a9475062989573aaa62334f187958165559603f9caa1e1cf9e433e549` | `413a10663b57375ae9046078874e57ad06f38827256ff9eb1259c46fbcf2c269` | reproduces | reproduces |
| `4c90d9cf` | 2026-09-01 | the evidence leg wired into the gate; the `--run` accusation deleted | `2c9d0109` | `4e6f6550306ce4014115a9a285d7ec1df592eba66ca0560afead6d88156167ad` | `405b7d4263806475064ae1814b8d2847be1f18ca4d82afea659d74c509c0d80f` | `claims` moved | reproduces |
| `566f7769` | 2026-10-07 | the head (7.49.0) | `a3006d9d` | `09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a` | `d1fca413fb023ad8efba238397ccb63e3e0274d7993e83532bf8f5d8370cdaba` | `claims` moved | reproduces |

Every row reproduces DOGFOOD. HANDOFF reproduces at every commit up to and including `c10da759`, and the only key
that moves at `4c90d9cf` and at `566f7769` is `claims`; `unparsed_same` is true on every row. No commit between
`c10da759` and `4c90d9cf` touches `styxx/diffgate.py` (`git log -1 4c90d9cf^ -- styxx/diffgate.py` is `c10da759`),
so `4c90d9cf` is the commit at which the record stopped reproducing. The 24 `diffgate.py` commits between
`4c90d9cf` and `566f7769` were not each staged (section 8).

## 3. The field that moved

`probe_handoff_claim_moved.py` (section 7.2), on the package staged at `4c90d9cf`, finds one claim differing, index
3 (`kind` `tests_pass`, text "All tests pass."), on one key, `why`; its `verdict` is UNCHECKABLE on both sides, and
the three `file_touched` claims are equal key for key. At `566f7769` the same probe gives the same one field. The
two strings:

- embedded (the minting build): `no --run command supplied; the gate does not take the agent's word for test results`
- live (at `4c90d9cf`, and unchanged at `566f7769`): `styxx.evidence (styxx-evidence/v0.2) read 0 supplied file(s),
  with no commit supplied, so nothing ties a report to this change — no evidence was supplied. Absence of a report is
  not a failing report; an unattested commit is unattested.  ||  No test REPORT was handed to the gate. It does not
  take the agent's word for test results, so with nothing to read it declines — absence of evidence is not a
  contradiction. The channel that makes this readable is a report passed with --evidence: a JUnit XML, or better a
  test-result attestation whose subject names the head commit. Even then no signature is checked, and the answer is
  VERIFIED or UNCHECKABLE — there is no accusing verdict for this claim kind.`

`4c90d9cf` deleted the old string's assignment from the `tests_pass` branch and routed the claim through
`styxx.evidence`; at `566f7769` the old string survives only in a comment above `_NO_EVIDENCE_NOTE`
(`styxx/diffgate.py` lines 658-659), which says `PREREG_evidence_leg_2026_09_01` committed to replacing it because
it pointed the reader at supplying a shell string. The verdict did not move; the reason for it did; and capsule v0.2
layer 2 compares the record key for key, so a reason string is as binding as a verdict.

## 4. The `v7.47.0` tag's file

- `git rev-parse v7.47.0^{commit}` is `01065bae`, and `git rev-parse v7.47.0^{commit}:styxx/diffgate.py` is
  `343fe97d`, the blob at `b7ee0a5e`. Staged from the tag, HANDOFF reproduces on every key (the `v7.47.0` row).
- `git merge-base --is-ancestor 4c90d9cf v7.47.0` exits 1: the commit at which the record stopped reproducing is not
  in the tag. `c10da759` is not in it either (exit 1); `b7ee0a5e`, `584ff104`, `414db45a` and `f6d469c4` are (exit
  0).
- `styxx/_version.py` reads `7.47.0` at `f6d469c4`, `b7ee0a5e`, `c10da759`, `4c90d9cf` and `a5cf9ecd`, so the
  capsule's pin, `styxx==7.47.0`, is satisfied by the tag's tree, which reproduces the record, and was also the
  stamp of the log writer that ran `4c90d9cf`'s bytes, which do not. What `pip install` of that version delivers was
  not downloaded or compared (section 8); the tag's tree is what this note measured.

## 5. Reading

In substance this is instrument skew: the bytes that re-run the record are not the bytes that wrote it, and under
the bytes that wrote it, or the tag's, the record reproduces on every key. The sealed inputs did not move (the
binding stage passes today, and line 95 holds one problem, at the re-run stage, so it passed for the log writer on
2026-09-03 too), and no reading of the summary or the diff moved: a reason string on a claim whose verdict did not
move is the whole difference.

The verifier cannot say so from the record alone, and this note does not make it. The record names no build.
`INSTRUMENT SKEW` as capsule v0.2 defines it fires on a `diffgate` format-key mismatch, and the key has been `v0`
throughout; the pin outside the bindings names a version, and section 4 shows one version string naming two
different files, one that reproduces and one that does not. From the capsule's bytes, `gate.claims not reproduced`
is all a verifier can say; telling an instrument change from an edit of the sealed record takes a build the record
names, or a measurement like this one.

The operator's decision of 2026-10-11 (O3 of the approved design): HANDOFF stays an advisory in the verifier, and
its classification lives here. Once the verifier compares a record's identity apart from its content (phase 1 of
that design), a record that names no build and does not reproduce is answered with the same problem line plus an
advisory that says the failure cannot be told apart from instrument skew or an edit from the record alone, never
with a SKEW problem: a SKEW problem would move line 95's `counts` and make the stranger's ferry-log step
(`styxx.stranger`, `papers/checksum/STRANGER.md` section 2) fail for every outsider. The same decisions keep
`"diffgate": "v0"` as the format key and add `build` and `runtime` keys beside it (O1), and leave `sys.version` out
of sealed records and charon lines (O9). None of that code is in this change. Records minted before a record named
its instrument, HANDOFF and DOGFOOD among them, will never name one: the build that wrote them is not in them, and
writing one in afterwards would fabricate it.

## 6. Not touched

- `HANDOFF_capsule_v02_2026_08_31.capsule.html`: content sha256 `63f5b3c861f03cd8cfeed9c3e98034e38be76e5b4978c12ea70704eae03e8015`
  (CRLF read as LF, charon's `_content_sha256`, the `subject.sha256` of line 95), as committed. It is not re-minted
  and not edited; it keeps failing verify with the one problem above.
- `DOGFOOD_session_2026_08_31.capsule.html`: as committed.
- `papers/charon/charon.log.jsonl`: 257 entries after the header, as committed. Lines seq 89 and 95 are not
  rewritten, and no line is appended; ingesting new lines is the operator's act (O7).
- No module, test, spec, receipt, certificate or conformance vector changes. The second file of this change is
  `CHANGELOG.md`.

## 7. The probes, their commands and their outputs

Both probes were written for the design review of 2026-10-07 and re-run for this note, read-only, against a
worktree at `566f7769` (`<repo>` below), with `<scratch>` a directory outside it, deleted afterwards. Interpreter:
`py -3.12`, CPython 3.12.10 (`3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`),
`unicodedata.unidata_version` 15.0.0, with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONIOENCODING=utf-8` set. The staged
files come from `git show <rev>:<path>`, so the checkout's line endings do not enter the measurement. `git status
--ignored` in the worktree was empty after each run.

### 7.1 `probe_handoff_bisect.py`

```python
"""Read-only probe: under which historical styxx/diffgate.py does each committed v0.2 capsule's embedded gate
reproduce? Stages styxx/{diffgate,claimdetect,evidence,declare,_version}.py from git objects at each commit into a
scratch package, re-runs gate_diff_text(summary, diff, run=None, strict=False) and compares every key.

    py -3.12 handoff_bisect.py <repo> <scratch> <commit>...
"""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
COMMITS = sys.argv[3:]
CAPS = ["papers/closed-model-frontier/HANDOFF_capsule_v02_2026_08_31.capsule.html",
        "papers/closed-model-frontier/DOGFOOD_session_2026_08_31.capsule.html"]
MODS = ["diffgate", "claimdetect", "evidence", "declare", "_version"]
BEGIN = '<script type="application/json" id="oath-capsule">'


def show(rev, path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def payload(rev, path):
    h = show(rev, path).decode("utf-8")
    i = h.index(BEGIN) + len(BEGIN)
    return json.loads(h[i:h.index("</script>", i)])


caps = {Path(c).name: payload("566f7769", c) for c in CAPS}
runner = SCRATCH / "run_one.py"
runner.write_text(
    "import base64, json, sys\n"
    "sys.dont_write_bytecode = True\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from styxx.diffgate import gate_diff_text\n"
    "p = json.loads(open(sys.argv[2], encoding='utf-8').read())\n"
    "s = base64.b64decode(p['summary']['b64']).decode('utf-8')\n"
    "d = base64.b64decode(p['diff']['b64']).decode('utf-8')\n"
    "live = gate_diff_text(s, d, run=None, strict=False).to_dict()\n"
    "g = p['gate']\n"
    "moved = [k for k in live if k != 'unparsed_claims' and live[k] != g.get(k)]\n"
    "print(json.dumps({'moved': moved, 'unparsed_same': live.get('unparsed_claims') == g.get('unparsed_claims')}))\n",
    encoding="utf-8")
for name, p in caps.items():
    (SCRATCH / (name + ".payload.json")).write_text(json.dumps(p), encoding="utf-8")
for rev in COMMITS:
    pkg = SCRATCH / ("pkg_" + rev) / "styxx"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_bytes(b"")
    have = []
    for m in MODS:
        b = show(rev, f"styxx/{m}.py")
        if b is not None:
            (pkg / f"{m}.py").write_bytes(b)
            have.append(m)
    dg = (pkg / "diffgate.py").read_bytes()
    lf = hashlib.sha256(dg.replace(b"\r\n", b"\n")).hexdigest()[:12]
    crlf = hashlib.sha256(dg.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()[:12]
    for name in caps:
        r = subprocess.run([sys.executable, str(runner), str(pkg.parent), str(SCRATCH / (name + ".payload.json"))],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                                "SYSTEMROOT": "C:\\Windows", "PATH": ""})
        out = r.stdout.strip() or ("ERROR " + r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else "no output")
        print(f"{rev} diffgate lf {lf} crlf {crlf} {name[:8]}: {out}")
```

Command and output (the sha columns are the leading twelve hex digits of the table's):

```
py -3.12 probe_handoff_bisect.py <repo> <scratch> f6d469c4 414db45a 584ff104 b7ee0a5e v7.47.0 c10da759 4c90d9cf 566f7769
f6d469c4 diffgate lf 1e912d429b9f crlf 15c0d2743b7d HANDOFF_: {"moved": [], "unparsed_same": true}
f6d469c4 diffgate lf 1e912d429b9f crlf 15c0d2743b7d DOGFOOD_: {"moved": [], "unparsed_same": true}
414db45a diffgate lf 95adc678f3af crlf c490c04b78a3 HANDOFF_: {"moved": [], "unparsed_same": true}
414db45a diffgate lf 95adc678f3af crlf c490c04b78a3 DOGFOOD_: {"moved": [], "unparsed_same": true}
584ff104 diffgate lf b3eb2d99da4c crlf 63b57200952b HANDOFF_: {"moved": [], "unparsed_same": true}
584ff104 diffgate lf b3eb2d99da4c crlf 63b57200952b DOGFOOD_: {"moved": [], "unparsed_same": true}
b7ee0a5e diffgate lf 6dfd21c4c3c0 crlf fb2d9b3e8426 HANDOFF_: {"moved": [], "unparsed_same": true}
b7ee0a5e diffgate lf 6dfd21c4c3c0 crlf fb2d9b3e8426 DOGFOOD_: {"moved": [], "unparsed_same": true}
v7.47.0 diffgate lf 6dfd21c4c3c0 crlf fb2d9b3e8426 HANDOFF_: {"moved": [], "unparsed_same": true}
v7.47.0 diffgate lf 6dfd21c4c3c0 crlf fb2d9b3e8426 DOGFOOD_: {"moved": [], "unparsed_same": true}
c10da759 diffgate lf 76baa58a9475 crlf 413a10663b57 HANDOFF_: {"moved": [], "unparsed_same": true}
c10da759 diffgate lf 76baa58a9475 crlf 413a10663b57 DOGFOOD_: {"moved": [], "unparsed_same": true}
4c90d9cf diffgate lf 4e6f6550306c crlf 405b7d426380 HANDOFF_: {"moved": ["claims"], "unparsed_same": true}
4c90d9cf diffgate lf 4e6f6550306c crlf 405b7d426380 DOGFOOD_: {"moved": [], "unparsed_same": true}
566f7769 diffgate lf 098670569570 crlf d1fca413fb02 HANDOFF_: {"moved": ["claims"], "unparsed_same": true}
566f7769 diffgate lf 098670569570 crlf d1fca413fb02 DOGFOOD_: {"moved": [], "unparsed_same": true}
```

The full shas of the table were computed from the same objects with
`git show <rev>:styxx/diffgate.py | tr -d '\r' | sha256sum` (LF) and the same bytes through `sed 's/$/\r/'`
(CRLF); the blob ids with `git rev-parse <rev>:styxx/diffgate.py`.

### 7.2 `probe_handoff_claim_moved.py`

```python
"""Read-only: which claim field of HANDOFF moved under a staged diffgate package (from probe_handoff_bisect.py).
    PYTHONDONTWRITEBYTECODE=1 py -3.12 probe_handoff_claim_moved.py <scratch>/pkg_4c90d9cf <scratch>/HANDOFF_capsule_v02_2026_08_31.capsule.html.payload.json"""
import base64, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from styxx.diffgate import gate_diff_text
p = json.loads(open(sys.argv[2], encoding="utf-8").read())
s = base64.b64decode(p["summary"]["b64"]).decode("utf-8")
d = base64.b64decode(p["diff"]["b64"]).decode("utf-8")
live = gate_diff_text(s, d, run=None, strict=False).to_dict()
for i, (a, b) in enumerate(zip(p["gate"]["claims"], live["claims"])):
    if a != b:
        for k in a:
            if a[k] != b.get(k):
                print(i, a["kind"], k, "| embedded:", json.dumps(a[k])[:150], "| live:", json.dumps(b.get(k))[:150])
print(len(p["gate"]["claims"]), len(live["claims"]))
```

Command and output (the probe cuts each value at 150 characters of its JSON form; the full live string is in
section 3):

```
py -3.12 probe_handoff_claim_moved.py <scratch>/pkg_4c90d9cf <scratch>/HANDOFF_capsule_v02_2026_08_31.capsule.html.payload.json
3 tests_pass why | embedded: "no --run command supplied; the gate does not take the agent's word for test results" | live: "styxx.evidence (styxx-evidence/v0.2) read 0 supplied file(s), with no commit supplied, so nothing ties a report to this change \u2014 no evidence was
4 4
```

On `pkg_566f7769` the output is the same two lines; on `pkg_v7.47.0` it is `4 4` alone.

### 7.3 The two capsules and their charon lines, as committed

A read-only check that `verify_capsule` and `charon.derive`, with styxx imported from the worktree, say today what
section 1 says, and that re-deriving lines 89 and 95 moves no key in `charon._COMPARED_KEYS`:

```python
"""Read-only: what capsule v0.2 verify and charon's re-derivation say today about the two committed
v0.2 capsules, with styxx imported from the checkout given. Run with PYTHONDONTWRITEBYTECODE=1.

    py -3.12 probe_as_committed.py <repo>
"""
import json
import sys
import unicodedata
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TREE))
sys.dont_write_bytecode = True

import styxx.diffgate as dg          # noqa: E402
from styxx import capsule, charon    # noqa: E402

assert Path(dg.__file__).resolve().is_relative_to(TREE), dg.__file__
CAPS = ["papers/closed-model-frontier/DOGFOOD_session_2026_08_31.capsule.html",
        "papers/closed-model-frontier/HANDOFF_capsule_v02_2026_08_31.capsule.html"]
log = [json.loads(x) for x in (TREE / "papers/charon/charon.log.jsonl").read_text(encoding="utf-8").splitlines()
       if x.strip()]
lines = {e["subject"]["path"]: e for e in log[1:] if e.get("kind") == "capsule-diffgate"}
print("=== as committed (" + sys.version.split()[0] + ", Unicode " + unicodedata.unidata_version + ")")
for rel in CAPS:
    p = TREE / rel
    rep = capsule.verify_capsule(p)
    fresh = charon.derive(p, TREE)
    e = lines[rel]
    moved = [k for k in charon._COMPARED_KEYS if charon.jcs(fresh[k]) != charon.jcs(e[k])]
    print(" ", Path(rel).name)
    print("    verify ok=%s problems=%d advisory=%d" % (rep["ok"], len(rep["problems"]), len(rep.get("advisory") or [])))
    for x in rep["problems"][:4]:
        print("      -", x[:150])
    print("    charon seq %d: compared keys moved: %s; fresh reproduced=%s counts.problems=%s" % (
        e["seq"], moved, fresh["reproduced"], fresh["counts"].get("problems")))
```

```
py -3.12 probe_as_committed.py <repo>
=== as committed (3.12.10, Unicode 15.0.0)
  DOGFOOD_session_2026_08_31.capsule.html
    verify ok=True problems=0 advisory=0
    charon seq 89: compared keys moved: []; fresh reproduced=True counts.problems=0
  HANDOFF_capsule_v02_2026_08_31.capsule.html
    verify ok=False problems=1 advisory=0
      - gate.claims not reproduced: embedded [{'kind': 'file_touched', 'text': 'Added tests/test_capsule_v02.py covering the mint refusals, the tamper battery
    charon seq 95: compared keys moved: []; fresh reproduced=False counts.problems=1
```

### 7.4 The tag

```
git merge-base --is-ancestor 4c90d9cf v7.47.0; echo exit=$?
exit=1
git merge-base --is-ancestor c10da759 v7.47.0; echo exit=$?
exit=1
git merge-base --is-ancestor b7ee0a5e v7.47.0; echo exit=$?
exit=0
git rev-parse v7.47.0^{commit} v7.47.0^{commit}:styxx/diffgate.py b7ee0a5e:styxx/diffgate.py
01065baea59c6d973ecd8382cafc41a2c49fc991
343fe97dccf30e8a6b9ff7b9c783e842398638d9
343fe97dccf30e8a6b9ff7b9c783e842398638d9
```

## 8. What it does not show

- **Which bytes `pip install` of the capsule's pin delivers.** The wheel and sdist were not downloaded. The release
  flow publishes the tag's tree, and the tag's `diffgate.py` reproduces the record; the installed files were not
  compared to it.
- **The runtime the capsule was minted under.** The record names none, and neither does the payload. Every row of
  the table was run under CPython 3.12.10 and Unicode 15.0.0, so "reproduces under its minting build" means under
  those bytes, on this runtime. A reader on another interpreter is measuring something else, and the design's
  Unicode-16 witnesses show the record can depend on it.
- **That every module the mint ran is the one staged.** The probe stages five modules under an empty
  `__init__.py` and nothing else of styxx, so an import of any other module would have failed the run; at each of
  these commits the staged set is therefore all of styxx the reader reaches. The mint's process, its working
  directory and its environment are not in the capsule and were not reconstructed.
- **Each of the 24 `diffgate.py` commits between `4c90d9cf` and `566f7769`.** The two endpoints fail on the same
  one field with the same two strings, and the string that moved was deleted at `4c90d9cf` and is not in the head's
  reader outside a comment; whether some commit in between reproduced the record by accident was not run.
- **A verifier's classification.** The verifier at `566f7769` says `gate.claims not reproduced` and nothing about
  why, and the approved design keeps that answer plus an advisory. The classification is prose, in this note, from
  git objects; it does not travel with the capsule.
- **Anything about the sealed inputs.** The summary and diff hash to their bindings on every verifier; nothing here
  reads them for truth.
- **The other 255 charon lines.** Only seq 89 and 95 were re-derived; the full log was last verified for the
  2026-09-29 snapshot and is not re-asserted here.

## 9. Process

The two files of this change were searched for the words the house rules forbid before each commit. The NOTE was
written through the Write tool and its line endings set in binary mode (CRLF in the working tree, as its
neighbours; LF in the index). The probes read git objects and the worktree only; the scratch directory was deleted
afterwards. Nothing was pushed, fetched beyond `origin`, posted or filed; the secrets directory was not read; the
live checkout was used for `git fetch` and `git worktree add` only.

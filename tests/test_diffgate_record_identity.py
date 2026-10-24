"""The Python record names its instrument (identity phase 2 of the 2026-10-11 design, sections 2 and 6.1;
AMENDMENT_capsule_v02_instrument_identity_2026_10_11).

What is held here: a record carries the eleven v0 content keys in their order and then ``build`` and
``runtime``, with ``"diffgate": "v0"`` still the format key; ``build`` names the file py_side.py pins; the
identity is computed once per process and copied into every record, and is not a field of DiffGate, so
the PATH-2a overlay's APPLY cannot move it; the alias in ``tests/_p2a_ref.module_from`` (a reader built
from a string names a null sha without it, and the module's identity with it), and that module_from is
the only place a reader is built from a string; ``--out`` writes the identity and the CLI's last line
names it; a capsule minted and verified on one box seals the identity and reads ``same``, and the
verifier 7.49.0 ships, which loops over its own eleven keys, finds every one of them in it; the two
committed v0.2 capsules verify to the same result as under a writer that names nothing (phase 1), and
charon lines seq 89 and 95 re-derive with no compared key moved; and, where the history is there, the
content keys byte for byte against origin/main's styxx/diffgate.py on every committed input at both
doors but the 139 whose content #205 moved by design (pinned by count and by a digest of their ids), with
``_report``'s lines and ``--out``'s bytes unchanged but for the identity.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import types
import unicodedata
from pathlib import Path

import pytest

import styxx.diffgate as N
from styxx import capsule as C
from styxx import charon
from styxx import instrument as I
from styxx.attestation import jcs
from styxx.capsule import _BEGIN, _END, _gate_binding_hash, create_capsule_diffgate, verify_capsule
from tests import _p2a_ref as R

ROOT = Path(__file__).resolve().parent.parent
PY_SIDE = ROOT / "web" / "gate" / "differential" / "py_side.py"
PAPERS = ROOT / "papers" / "closed-model-frontier"
DOGFOOD = PAPERS / "DOGFOOD_session_2026_08_31.capsule.html"
HANDOFF = PAPERS / "HANDOFF_capsule_v02_2026_08_31.capsule.html"
LOG = ROOT / "papers" / "charon" / "charon.log.jsonl"
FIXTURES = ROOT / "tests" / "fixtures"

V0_KEYS = ("diffgate", "verdict", "base", "head", "claims", "uncovered_sentences", "sentences_total",
           "uncovered_texts", "unparsed_claims", "measured", "why_unmeasured")
# origin/main where identity phase 2 branched, and its styxx/diffgate.py (LF): py_side.py's pin before phase 2
PARENT_COMMIT = "295529c64d19319114095068bb0b636daa5bdee6"
PARENT_SHA = "09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a"

SUMMARY = ("Added styxx/newmod.py with the frobnicate helper. Deleted old_helper.py. "
           "All tests pass. This change also refactors the widget pipeline for clarity.\n")
DIFF = ("diff --git a/styxx/newmod.py b/styxx/newmod.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/styxx/newmod.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def frobnicate(x):\n"
        "+    return x + 1\n"
        "diff --git a/old_helper.py b/old_helper.py\n"
        "deleted file mode 100644\n"
        "--- a/old_helper.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-def old(x):\n"
        "-    return x\n")


def _pinned() -> str:
    return re.search(r'^PINNED\s*=\s*"([0-9a-f]{64})"', PY_SIDE.read_text(encoding="utf-8"), re.M).group(1)


def _content(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in I.IDENTITY_KEYS}


def _log_line(seq):
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("seq") == seq:
                return e
    raise AssertionError(f"no line seq {seq}")


def _payload(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return json.loads(html[i:html.index(_END, i)])


def _identity_advisories(rep):
    return [a for a in rep["advisory"] if not a.startswith("SKIPPED") and "unparsed_claims" not in a]


class _Bare:
    def __init__(self, g):
        self._g = g

    def to_dict(self):
        return _content(self._g.to_dict())


def _writer_naming_nothing(monkeypatch):
    """Every gate_diff_text call returns a record with the identity keys taken out: the writer as it was in phase 1."""
    real = N.gate_diff_text
    monkeypatch.setattr(N, "gate_diff_text", lambda *a, **k: _Bare(real(*a, **k)))


# ------------------------------------------------------------------------------------------ 1. the keys

def test_a_record_carries_the_eleven_v0_keys_in_order_then_build_and_runtime(monkeypatch):
    g = N.gate_diff_text(SUMMARY, DIFF)
    d = g.to_dict()
    assert list(d) == [*V0_KEYS, *I.IDENTITY_KEYS] and d["diffgate"] == "v0"
    assert I.content_keys(d) == list(V0_KEYS[1:])
    assert d["build"] == I.record_identity(N.__file__)["build"] and d["runtime"] == I.runtime()
    # not a field of the record: nothing that builds or copies a DiffGate can set it
    assert not {f.name for f in dataclasses.fields(N.DiffGate)} & set(I.IDENTITY_KEYS)
    assert not set(vars(g)) & set(I.IDENTITY_KEYS)
    # an unmeasured record and a git-door record carry it too
    empty = N.gate_diff_text(SUMMARY, "").to_dict()
    assert empty["measured"] is False and list(empty)[-2:] == list(I.IDENTITY_KEYS)
    monkeypatch.setattr(N, "_git", R.fake_git("A\tstyxx/newmod.py\nD\told_helper.py\n", DIFF))
    git = N.gate_diff(SUMMARY, "(repo)", "base", "head").to_dict()
    assert list(git) == [*V0_KEYS, *I.IDENTITY_KEYS] and git["build"] == d["build"]
    # a fresh copy in every record: editing one record's identity moves no other
    d["build"]["modules"][0][1] = "0" * 64
    d["runtime"]["version"] = "0.0.0"
    again = g.to_dict()
    assert again["build"]["modules"][0][1] == _pinned() and again["runtime"]["version"] != "0.0.0"
    text = json.dumps(again)
    assert str(ROOT) not in text and str(ROOT).replace("\\", "/") not in text and sys.version not in text
    assert jcs(again)                                                   # JCS canonicalises the whole record


def test_the_record_names_the_file_py_side_pins_and_this_runtime():
    d = N.gate_diff_text(SUMMARY, DIFF).to_dict()
    b, r = d["build"], d["runtime"]
    assert b["schema"] == I.BUILD_SCHEMA and b["reader"] == "styxx.diffgate"
    assert b["modules"][0] == ["styxx.diffgate", _pinned()]
    assert _pinned() == I.lf_sha256(ROOT / "styxx" / "diffgate.py")      # test_port_is_current's rule
    assert [m for m, _ in b["modules"]] == list(I.RECORD_MODULES)
    from styxx._version import __version__
    assert b["styxx"] == __version__
    assert r == I.runtime() and set(r) == {"implementation", "version", "unicode", "case_floor"}
    assert N._IDENTITY_FILE == N.__file__


def test_the_identity_is_computed_once_per_process_and_copied_into_every_record(monkeypatch):
    monkeypatch.setattr(I, "_cache", {})
    reads = []
    real = I.lf_sha256

    def counting(path):
        reads.append(path)
        return real(path)
    monkeypatch.setattr(I, "lf_sha256", counting)
    g = N.gate_diff_text(SUMMARY, DIFF)
    records = [g.to_dict() for _ in range(25)] + [N.gate_diff_text("Edited a.py.", "").to_dict() for _ in range(5)]
    assert len(reads) == len(I.RECORD_MODULES)                           # each file read once, at the earliest record
    assert all(r["build"] == records[0]["build"] and r["runtime"] == records[0]["runtime"] for r in records)
    assert len({id(r["build"]) for r in records}) == len(records)


# ------------------------------------------------------------------------------------ 2. PATH-2a, section 6.1

def test_a_reader_built_from_a_string_names_a_null_sha_and_module_from_gives_it_the_modules_file():
    source = R.main_source()
    name = "styxx._identity_no_alias_reference"
    mod = types.ModuleType(name)
    mod.__package__ = "styxx"
    mod.__file__ = str(R.INSTRUMENT.parent / "_identity_no_alias_reference.py")      # no such file
    sys.modules[name] = mod
    try:
        exec(compile(source, mod.__file__, "exec"), mod.__dict__)
        bare = mod.gate_diff_text(SUMMARY, DIFF).to_dict()
    finally:
        sys.modules.pop(name, None)
    assert bare["build"]["modules"][0] == ["styxx.diffgate", None]      # never another file's bytes
    assert bare["build"]["modules"][1:] == I.record_identity(N.__file__)["build"]["modules"][1:]
    M = R.main_module()
    assert M._IDENTITY_FILE == N.__file__
    a, b = M.gate_diff_text(SUMMARY, DIFF).to_dict(), N.gate_diff_text(SUMMARY, DIFF).to_dict()
    assert a["build"] == b["build"] and a["runtime"] == b["runtime"]
    assert R.relation(a, b, False, dict(N._P2A_PHRASES)) == []


def test_module_from_is_the_only_place_a_reader_is_built_from_a_string():
    """A plant built another way would compare an identity of its own, and an assertion that the plant is caught
    would pass for the wrong reason."""
    for path in sorted((ROOT / "tests").glob("test_diffgate_path2a*.py")) + [ROOT / "tests" / "_p2a_truth.py",
                                                                              ROOT / "tests" / "_p2a_cases.py"]:
        text = path.read_text(encoding="utf-8")
        assert "exec(compile(" not in text and "ModuleType(" not in text, path.name
    ref = (ROOT / "tests" / "_p2a_ref.py").read_text(encoding="utf-8")
    assert ref.count("ModuleType(") == 1 and ref.count("exec(compile(source, mod.__file__") == 1
    assert 'mod._IDENTITY_FILE = sys.modules["styxx.diffgate"].__file__' in ref


@pytest.mark.parametrize("decide", [lambda seen: None, lambda seen: 1 / 0], ids=["malformed", "raises"])
def test_apply_never_changes_build_or_runtime(decide):
    def in_reach(c):
        try:
            return any((x.kind, x.verdict) in R.REACH for x in N.gate_diff_text(c["summary"], c["diff"]).claims)
        except Exception:  # noqa: BLE001 - an input this reader refuses is not the one wanted here
            return False
    case = next(c for c in R.repro_cases() if in_reach(c))
    g = N.gate_diff_text(case["summary"], case["diff"])
    before = copy.deepcopy(g.to_dict())               # to_dict hands out the claims' own dicts, which APPLY edits
    assert N._p2a_apply(g, False, decide) is g
    after = g.to_dict()
    assert after["claims"] != before["claims"]                           # APPLY withheld what was in reach
    assert after["build"] == before["build"] and after["runtime"] == before["runtime"]
    assert not set(vars(g)) & set(I.IDENTITY_KEYS)


# ---------------------------------------------------------------------------------------- 3. --out, the CLI

def test_out_writes_the_identity_and_the_last_line_names_the_instrument(tmp_path, capsys):
    g = N.gate_diff_text(SUMMARY, DIFF)
    out = tmp_path / "gate.json"
    code = N._report(g, str(out))
    lines = capsys.readouterr().out.splitlines()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert list(written) == [*V0_KEYS, *I.IDENTITY_KEYS] and written == g.to_dict()
    assert code == (0 if g.verdict == "PASS" else 1)
    assert lines[0].startswith(g.verdict + "  claims=")
    ident = I.record_identity(N.__file__)
    assert lines[-1] == "instrument: " + I.describe(ident) and lines[-1].isascii()
    assert lines[-1].startswith(f"instrument: styxx.diffgate {_pinned()[:10]} (styxx ")
    assert sum(1 for x in lines if x.startswith("instrument: ")) == 1


def test_the_cli_names_the_instrument_last_on_a_real_repository(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH; the CLI's git door cannot run here")

    def run(*args):
        return subprocess.run([git, *args], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout.strip()
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "base")
    base = run("rev-parse", "HEAD")
    (tmp_path / "b.py").write_text("def b():\n    return 1\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "head")
    summary = tmp_path / "summary.md"
    summary.write_text("Added b.py. A sentence no template reads.", encoding="utf-8")
    out = tmp_path / "gate.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = N.main([str(summary), "--repo", str(tmp_path), "--base", base, "--head", "HEAD", "--out", str(out)])
    lines = buf.getvalue().splitlines()
    assert code == 0 and lines[0].startswith("PASS")                     # the GitLab job reads startswith("PASS")
    assert lines[-1].startswith(f"instrument: styxx.diffgate {_pinned()[:10]} (styxx ")
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert list(rec)[-2:] == list(I.IDENTITY_KEYS) and rec["build"]["modules"][0][1] == _pinned()


# ------------------------------------------------------------------------------------------ 4. capsules

def test_a_capsule_minted_and_verified_on_one_box_seals_the_identity_and_reads_same(tmp_path):
    sp, dp = tmp_path / "s.md", tmp_path / "s.diff"
    sp.write_text(SUMMARY, encoding="utf-8")
    dp.write_text(DIFF, encoding="utf-8")
    out = create_capsule_diffgate(sp, dp, tmp_path / "s.capsule.html")
    p = _payload(out)
    gate = p["gate"]
    ident = I.record_identity(N.__file__)
    assert list(gate) == [*V0_KEYS, *I.IDENTITY_KEYS] and gate["diffgate"] == "v0"
    assert {k: gate[k] for k in I.IDENTITY_KEYS} == ident
    assert p["binding"]["gate"]["value"] == _gate_binding_hash(gate) != _gate_binding_hash(_content(gate))
    rep = verify_capsule(out)
    assert rep["ok"] is True and rep["problems"] == [] and rep["skew"] is False
    assert rep["gate_identity"]["state"] == "same" and rep["gate_identity"]["recorded"] == ident
    assert rep["gate_identity"]["live"] == ident and _identity_advisories(rep) == []
    buf = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    with contextlib.redirect_stdout(buf):
        code = C.main(["verify", str(out)])
    buf.flush()
    text = buf.buffer.getvalue().decode("utf-8")
    assert code == 0 and f"instrument: recorded {I.describe(ident)}; re-run here by {I.describe(ident)}" in text
    # the verifier 7.49.0 ships loops over every key of its own live record (the eleven) and ignores the rest
    live = _content(N.gate_diff_text(SUMMARY, DIFF, run=None, strict=False).to_dict())
    assert all(gate.get(k) == live[k] for k in live)
    line = charon.derive(out, tmp_path)
    assert line["reproduced"] is True and line["verdict"] != "UNRESOLVED"


@pytest.mark.parametrize("seq, path", [(89, DOGFOOD), (95, HANDOFF)])
def test_the_committed_capsules_verify_as_under_a_writer_that_names_nothing(seq, path, monkeypatch):
    """The two committed v0.2 capsules name no instrument. Their verify result is the same, key for key, whether the
    live record carries this box's identity (phase 2) or none (phase 1, where the verifier computed it over the
    imported reader), and charon re-derives each line with no compared key moved."""
    e = _log_line(seq)
    rep = verify_capsule(path)
    fresh = charon.derive(path, ROOT)
    assert rep["gate_identity"]["state"] == "unnamed" and rep["gate_identity"]["recorded"] is None
    assert rep["gate_identity"]["live"] == I.record_identity(N.__file__)
    assert [p[:160] for p in rep["problems"][:3]] == e["counts"]["problems_detail"]
    assert rep["ok"] is e["reproduced"] and len(_identity_advisories(rep)) == 1
    assert [k for k in charon._COMPARED_KEYS if charon.jcs(fresh[k]) != charon.jcs(e[k])] == []
    _writer_naming_nothing(monkeypatch)
    assert "build" not in N.gate_diff_text(SUMMARY, DIFF).to_dict()
    assert verify_capsule(path) == rep
    before = charon.derive(path, ROOT)
    assert {k: before[k] for k in charon._COMPARED_KEYS} == {k: fresh[k] for k in charon._COMPARED_KEYS}


# ------------------------------------------------------------------- 5. content keys against origin/main's file

@pytest.fixture(scope="module")
def parent():
    """origin/main's styxx/diffgate.py where identity phase 2 branched, built as a module (module_from's alias is set
    on it and read by nothing: that file names no instrument). Needs the commit in this checkout's history; a clone
    without it skips."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH; origin/main's file cannot be read here")
    r = subprocess.run([git, "-C", str(ROOT), "show", f"{PARENT_COMMIT}:styxx/diffgate.py"], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"{PARENT_COMMIT[:8]} is not in this checkout's history")
    text = r.stdout.replace(b"\r\n", b"\n").decode("utf-8")
    assert R.sha(text) == PARENT_SHA, "the parent commit's styxx/diffgate.py is not the file py_side.py pinned"
    mod = R.module_from(text, "_identity_parent_reader")
    assert list(mod.gate_diff_text(SUMMARY, DIFF).to_dict()) == list(V0_KEYS)
    return mod


def _same(a, b, run) -> str | None:
    """None when both readers give the same content, byte for byte, or raise the same exception; else what differs."""
    try:
        want = run(a).to_dict()
    except Exception as e:  # noqa: BLE001 - the parent's own failure is part of what is compared
        try:
            run(b).to_dict()
        except Exception as f:  # noqa: BLE001
            return None if (type(e), str(e)) == (type(f), str(f)) else f"raised {e!r} vs {f!r}"
        return f"the parent raised {e!r} and this reader did not"
    got = run(b).to_dict()
    if list(want) != list(V0_KEYS) or list(got) != [*V0_KEYS, *I.IDENTITY_KEYS]:
        return f"keys {list(want)} vs {list(got)}"
    if json.dumps(want, ensure_ascii=False) != json.dumps(_content(got), ensure_ascii=False):
        return "content moved"
    return None


# #205 (fix/diffgate-205-206) moved the reading of a diff holding one of these, by design: the Python cut git output at
# them with str.splitlines() and now cuts it at git's line ends, and the `def` counts and the compatibility reading
# abstain over a line holding one of them but U+000C. #206 moved a declared count written outside ASCII, which this
# comparison cannot see: the parent's file is built in this checkout's package, so it reads this styxx.declare.
_MOVED_BY_205 = re.compile("[\x0b\x0c\x1c-\x1f\x85\u2028\u2029]")
HOLD_205 = 592            # raw-door inputs of R.inputs() (6,672) whose diff holds one; no git-door fixture with a listing does
# Of those, the inputs whose content moved in either strict mode (a record, or a raise where the parent raised none:
# p2a-fuzz:20260930:692, which the CHANGELOG names), and the sha256 of their ids (R.uid), sorted and joined by LF.
# Measured on CPython 3.12.10 and 3.14.2, the same set. Every other input, the 414 others holding one included, is
# compared byte for byte. 139 (68e399a0...) until #179 and #178 (fix/diffgate-178-179) moved the reading of more of them,
# and 167 (5c87f416...) until review 3 of fix batch 1 (the same branch) did, where #205's seams part the two readers
# and a test name is defined twice in one section, one line after `async`.
MOVED_BY_205 = 178
MOVED_BY_205_SHA = "9f110670d44baa4325492dc2c9a34c5b7b9cf9fca0062db9f6b6d5910f55b4ff"

# #179 and #178 (fix/diffgate-178-179) moved the reading of a diff of these shapes, by design: the `def` counts read
# `async def` where they read `def`, and abstain where a counted line sits in a file not read as .py or .pyi, or after
# an odd number of triple quotes in the added lines of its `+++` section. The predicate holds on a diff holding
# `async`, three double or three single quotes, or an added `def` or `class` line under a `+++` path that does not
# end in .py or .pyi.
_ASYNC_OR_TRIPLE = re.compile("async|\"{3}|'{3}")
_ADDED_DEF = re.compile(r"\+[ \t\f]*(?:async[ \t\f]+)?(?:def|class)\b")


def _moved_by_178_179(diff: str) -> bool:
    if _ASYNC_OR_TRIPLE.search(diff):
        return True
    path = None
    for line in N._git_lines(diff):
        if line.startswith("+++ "):
            path = line[4:].strip().lower()
        elif path is not None and _ADDED_DEF.match(line) and not path.endswith((".py", ".pyi")):
            return True
    return False


HOLD_178_179 = 1910       # raw-door inputs of R.inputs() whose diff holds no #205 character and one of these shapes
# Of those, the inputs whose content moved in either strict mode, and the sha256 of their ids, as for #205. Measured on
# CPython 3.12.10 and 3.14.2, the same set. The other 1,667 are compared byte for byte, as is every git-door fixture
# (none of those with a listing holds one of these shapes). 240 (515595d7...) until review 3 of fix batch 1, and 243
# (aa4c38fc...) until the one-sided #178 (options judgment of fix batch 1), where main's tests count decides again a
# claim above every count it takes: 54 inputs leave the set, their content the parent's again, and none joins it.
MOVED_BY_178_179 = 189
MOVED_BY_178_179_SHA = "0199f6b873cd583717601f594cbe467bbbea08b63d4ab78b22df6057c7a356a9"

# Review 3 of fix batch 1 (fix/diffgate-178-179) moved the reading of a diff of these shapes, by design: the two `def`
# counts abstain where #205's seams part the two readers, a U+FEFF before a counted line's `def` or name or at a
# header's edge, or a name run on into a code point from U+0080 up. The predicate holds on a diff, holding none of the
# shapes above, that holds U+FEFF or a `def` or `class` name running on into such a code point.
_MOVED_BY_SEAMS = re.compile("\ufeff|(?:def|class)[ \t\f]+[A-Za-z0-9_]*[\x80-\U0010ffff]")
HOLD_SEAMS = 123
# Of those, the inputs whose content moved in either strict mode, and the sha256 of their ids, as above. One input more
# on CPython 3.12.10 than on 3.14.2: path2:x1's claimed name runs on into U+1C89, a letter only from Unicode 16, so on
# Unicode 15 and older the claimed name is `foo` and the seam abstains, and on 16 it is the whole run, as the parent
# reads it.
MOVED_BY_SEAMS = {"16.0.0": (39, "baa927861d5a3e73f48fcd9529c62e478c58a44299edf221d3dc794f778c6cb4")}
MOVED_BY_SEAMS_BEFORE_16 = (40, "2863a90eed5ddcf028759dcb0f2323cc1e9e4a25c5f7fdc2fc0039f2b8028d23")


def _same_or_raised(a, b, run) -> str | None:
    """_same, with a raise of this reader where the parent raised none read as content moved."""
    try:
        return _same(a, b, run)
    except Exception as e:  # noqa: BLE001 - this reader's own failure is part of what is compared
        return f"this reader raised {e!r} and the parent did not"


def test_content_keys_are_byte_for_byte_the_parents_on_every_committed_input(parent, monkeypatch):
    """The pinned pairs, the regenerated 3,000-pair fuzz corpus, the PATH-2a reproductions with their seeded fuzz and
    text-seam sets, at the raw door in both strict modes; and every PATH-2a fixture that carries its own
    --name-status at the git door, in both strict modes. Identity phase 2's guarantee was every committed input; since
    #205 it is every one but those whose content #205 moved by design: each of them holds a character #205 reads anew
    (`_MOVED_BY_205`), and they are pinned by count and by a digest of their ids, so the set cannot grow or change
    unseen. tests/test_diffgate_issue205_206.py holds what #205 moved. Since #179 and #178 the inputs holding none of
    those characters and a shape `_moved_by_178_179` reads are pinned apart the same way, and since review 3 of fix
    batch 1 the inputs holding neither and a shape `_MOVED_BY_SEAMS` reads; tests/test_diffgate_issue178_179.py holds
    what those moved."""
    bad, runs, held, moved = [], 0, 0, []
    held_new, moved_new = 0, []
    held_seam, moved_seam = 0, []
    for row in R.inputs():
        holds = bool(_MOVED_BY_205.search(row[3] or ""))
        held += holds
        holds_new = not holds and _moved_by_178_179(row[3] or "")
        held_new += holds_new
        holds_seam = not holds and not holds_new and bool(_MOVED_BY_SEAMS.search(row[3] or ""))
        held_seam += holds_seam
        whys = []
        for strict in (False, True):
            whys.append(_same_or_raised(parent, N, lambda m, s=row[2], d=row[3], k=strict:
                                        m.gate_diff_text(s, d, strict=k)))
            runs += 1
        if any(whys) and holds:
            moved.append(R.uid(0, row))
        elif any(whys) and holds_new:
            moved_new.append(R.uid(0, row))
        elif any(whys) and holds_seam:
            moved_seam.append(R.uid(0, row))
        elif any(whys):
            bad.append((R.uid(0, row), whys))
    git_runs = 0
    for name in ("path2a_repros.json", "path2a_pass5_repros.json", "path2a_pass6_repros.json"):
        for c in json.loads((FIXTURES / name).read_text(encoding="utf-8"))["cases"]:
            if not c.get("name_status"):
                continue
            held += bool(_MOVED_BY_205.search(c["diff"] + c["name_status"]))
            fake = R.fake_git(c["name_status"], c["diff"])
            monkeypatch.setattr(parent, "_git", fake)
            monkeypatch.setattr(N, "_git", fake)
            for strict in (False, True):
                why = _same(parent, N, lambda m, s=c["summary"], k=strict: m.gate_diff(s, "(repo)", "base", "head",
                                                                                      strict=k))
                git_runs += 1
                if why:
                    bad.append((name, c["id"], strict, why))
    assert bad == [], bad[:5]
    assert runs > 13000 and git_runs > 200, (runs, git_runs)
    assert held == HOLD_205, held
    moved.sort()
    assert len(moved) == MOVED_BY_205, (len(moved), moved[:5])
    assert hashlib.sha256("\n".join(moved).encode("utf-8")).hexdigest() == MOVED_BY_205_SHA, moved[:5]
    assert held_new == HOLD_178_179, held_new
    moved_new.sort()
    assert len(moved_new) == MOVED_BY_178_179, (len(moved_new), moved_new[:5])
    assert hashlib.sha256("\n".join(moved_new).encode("utf-8")).hexdigest() == MOVED_BY_178_179_SHA, moved_new[:5]
    assert held_seam == HOLD_SEAMS, held_seam
    moved_seam.sort()
    want = MOVED_BY_SEAMS.get(unicodedata.unidata_version, MOVED_BY_SEAMS_BEFORE_16)
    assert (len(moved_seam), hashlib.sha256("\n".join(moved_seam).encode("utf-8")).hexdigest()) == want, moved_seam[:5]


def test_report_and_out_are_the_parents_but_for_the_identity(parent, tmp_path):
    """`_report` prints the parent's lines and then one more, and `--out` writes the parent's bytes once the identity
    keys are taken out."""
    cases = [(SUMMARY, DIFF), (SUMMARY, ""), ("Nothing here is a claim.", DIFF)] + \
        [(c["summary"], c["diff"]) for c in R.repro_cases()[:20]]
    for summary, diff in cases:
        outs = {}
        for tag, m in (("parent", parent), ("this", N)):
            path = tmp_path / f"{tag}.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = m._report(m.gate_diff_text(summary, diff), str(path))
            outs[tag] = (code, buf.getvalue().splitlines(), path.read_text(encoding="utf-8"))
        (pc, plines, ptext), (tc, tlines, ttext) = outs["parent"], outs["this"]
        assert pc == tc and tlines[:-1] == plines and tlines[-1].startswith("instrument: styxx.diffgate ")
        assert json.dumps(_content(json.loads(ttext)), indent=2) + "\n" == ptext

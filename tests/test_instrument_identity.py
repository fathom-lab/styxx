"""styxx.instrument and capsule v0.2's reading of a record's identity
(AMENDMENT_capsule_v02_instrument_identity_2026_10_11, phase 1 of the 2026-10-11 design).

What is held here: the LF hash rule; the runtime read from this interpreter, never pinned (CI runs
3.9 to 3.12); the reader's identity against the pins the differential and the port tests already
hold; find_spec executing nothing; every state of `delta`, and that hostile identities are read, never
raised on; the two committed capsules keeping their outcomes to the character charon recorded
(DOGFOOD ok with no problem, HANDOFF its one problem) with one advisory each, and charon lines seq 89
and 95 re-deriving with no compared key moved; synthetic identities exercising SKEW, same, incomplete
and unnamed through mint and verify, with the three wording constraints, (b) against a minter who chooses
the words (a module named `ambiguous`, an implementation, a version, a styxx string holding it); R5 never
comparing identity;
the page's meta line; and `python -m styxx.instrument`.

The record writer (styxx/diffgate.py) names nothing in this phase, so a record that carries an
identity here is made by wrapping gate_diff_text; phase 2 makes the writer carry it.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from styxx import capsule as C
from styxx import charon
from styxx import instrument as I
from styxx.attestation import jcs
from styxx.capsule import _BEGIN, _END, _TEMPLATE_V02, _gate_binding_hash, create_capsule_diffgate, verify_capsule

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers" / "closed-model-frontier"
DOGFOOD = PAPERS / "DOGFOOD_session_2026_08_31.capsule.html"
HANDOFF = PAPERS / "HANDOFF_capsule_v02_2026_08_31.capsule.html"
LOG = ROOT / "papers" / "charon" / "charon.log.jsonl"
PY_SIDE = ROOT / "web" / "gate" / "differential" / "py_side.py"

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


# ------------------------------------------------------------------------------------ helpers

def _identity(**over) -> dict:
    """A full synthetic identity X; keyword overrides name dotted fields (build.styxx, runtime.version)
    or a module by name (module__styxx_diffgate)."""
    x = {"build": {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "styxx": "7.49.0",
                   "modules": [["styxx.diffgate", "a" * 64], ["styxx.instrument", "b" * 64],
                               ["styxx.declare", "c" * 64], ["styxx.evidence", "d" * 64],
                               ["styxx.claimdetect", "e" * 64]]},
         "runtime": {"implementation": "cpython", "version": "3.12.10", "unicode": "15.0.0",
                     "case_floor": "14.0"}}
    for k, v in over.items():
        if k.startswith("module__"):
            name = k[len("module__"):].replace("_", ".", 1)
            for pair in x["build"]["modules"]:
                if pair[0] == name:
                    pair[1] = v
        else:
            part, field = k.split("__")
            x[part][field] = v
    return x


X = _identity()
# Y: another build of the reader on another interpreter; its distinctive values never appear in X
Y = _identity(module__styxx_diffgate="f" * 64, runtime__version="9.99.1", runtime__unicode="99.0.0")


class _Carrying:
    """The record the writer will produce from phase 2 on: the eleven content keys, then the identity."""

    def __init__(self, g, identity):
        self._g, self._id = g, identity

    def to_dict(self):
        d = self._g.to_dict()
        d.update(copy.deepcopy(self._id))
        return d


def _under(monkeypatch, identity):
    """Every gate_diff_text call, at mint or verify, returns a record carrying `identity`."""
    import styxx.diffgate as dg
    real = getattr(dg.gate_diff_text, "_identity_wrapped_real", dg.gate_diff_text)

    def wrapped(*a, **k):
        return _Carrying(real(*a, **k), identity)
    wrapped._identity_wrapped_real = real
    monkeypatch.setattr(dg, "gate_diff_text", wrapped)


def _mint(tmp_path, summary=SUMMARY, diff=DIFF, gate=None, name="s"):
    sp = tmp_path / f"{name}.md"
    sp.write_text(summary, encoding="utf-8")
    dp = tmp_path / f"{name}.diff"
    dp.write_text(diff, encoding="utf-8")
    gp = None
    if gate is not None:
        gp = tmp_path / f"{name}.gate.json"
        gp.write_text(json.dumps(gate), encoding="utf-8")
    return create_capsule_diffgate(sp, dp, tmp_path / f"{name}.capsule.html", gp)


def _payload(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    return html, i, j, json.loads(html[i:j])


def _rewrite(path, html, i, j, payload):
    # ASCII escapes, so a lone surrogate placed in the payload is written as its escape
    body = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    path.write_text(html[:i] + body + html[j:], encoding="utf-8")


def _edit_and_reseal(path):
    """test_capsule_v02's K2 forger: flip a claim verdict, keep the fold coherent, re-seal binding.gate."""
    html, i, j, p = _payload(path)
    victim = next(c for c in p["gate"]["claims"] if c["verdict"] == "VERIFIED")
    victim["verdict"] = "CONTRADICTED"
    p["gate"]["verdict"] = "FAIL"
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(path, html, i, j, p)


def _command(path):
    """(exit code, stdout) of `python -m styxx.capsule verify`, on a strict UTF-8 stream."""
    out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    with contextlib.redirect_stdout(out):
        code = C.main(["verify", str(path)])
    out.flush()
    return code, out.buffer.getvalue().decode("utf-8")


def _identity_advisories(rep):
    return [a for a in rep["advisory"] if not a.startswith("SKIPPED") and "unparsed_claims" not in a]


def _log_line(seq):
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("seq") == seq:
                return e
    raise AssertionError(f"no line seq {seq}")


def _lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------------------- 1. the hash

def test_lf_sha256_reads_crlf_as_lf_keeps_a_lone_cr_and_gives_none_when_unreadable(tmp_path):
    lf = tmp_path / "lf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    cr = tmp_path / "cr.py"
    cr.write_bytes(b"a = 1\rb = 2\r")
    assert I.lf_sha256(lf) == I.lf_sha256(crlf) == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    assert I.lf_sha256(cr) == hashlib.sha256(b"a = 1\rb = 2\r").hexdigest() != I.lf_sha256(lf)
    assert I.lf_sha256(tmp_path / "absent.py") is None
    assert I.lf_sha256(tmp_path) is None            # a directory
    assert I.lf_sha256(None) is None                # no path at all


# ---------------------------------------------------------------------------- 2. the runtime

def test_runtime_is_what_this_interpreter_says_and_never_sys_version():
    r = I.runtime()
    assert r["implementation"] == sys.implementation.name
    assert r["version"] == platform.python_version()
    assert r["unicode"] == unicodedata.unidata_version
    major = int(unicodedata.unidata_version.split(".")[0])
    want = "13.0" if major == 13 else "14.0" if major in (14, 15) else "16.0" if major >= 16 else "<13.0"
    assert r["case_floor"] == want == I.case_floor()
    assert set(r) == {"implementation", "version", "unicode", "case_floor"}
    assert "detail" not in r and sys.version not in json.dumps(r)
    assert all(isinstance(v, str) for v in r.values())


# -------------------------------------------------------------------- 3. the reader's identity

def test_record_identity_of_the_reader_names_the_pinned_file_and_no_path():
    import styxx.diffgate as dg
    a = I.record_identity(dg.__file__)
    b = I.record_identity(dg.__file__)
    pinned = re.search(r'^PINNED\s*=\s*"([0-9a-f]{64})"', PY_SIDE.read_text(encoding="utf-8"), re.M).group(1)
    assert a["build"]["modules"][0] == ["styxx.diffgate", pinned]
    assert pinned == _lf(ROOT / "styxx" / "diffgate.py")          # test_port_is_current.instrument_sha's rule
    assert [m for m, _ in a["build"]["modules"]] == list(I.RECORD_MODULES)
    for _, sha in a["build"]["modules"]:
        assert sha is None or re.fullmatch(r"[0-9a-f]{64}", sha), sha
    assert a["build"]["modules"][1][1] == _lf(ROOT / "styxx" / "instrument.py")
    assert a["build"]["schema"] == I.BUILD_SCHEMA and a["build"]["reader"] == "styxx.diffgate"
    from styxx._version import __version__
    assert a["build"]["styxx"] == __version__
    assert a["runtime"] == I.runtime()
    # fresh objects on every call: no caller can edit another record's identity
    assert a == b and a is not b and a["build"] is not b["build"] and a["build"]["modules"] is not b["build"]["modules"]
    a["build"]["modules"][0][1] = "0" * 64
    assert I.record_identity(dg.__file__)["build"]["modules"][0][1] == pinned
    text = json.dumps(b)
    assert str(ROOT) not in text and str(ROOT).replace("\\", "/") not in text and "\\\\" not in text
    assert "detail" not in b["runtime"] and sys.version not in text
    assert jcs(b)                                                   # JCS canonicalises it unchanged


def test_a_reader_built_from_a_string_names_a_null_sha_never_another_files_bytes(tmp_path):
    for self_file in (None, "", str(tmp_path / "_p2a_main_reference.py"), str(tmp_path)):
        ident = I.record_identity(self_file)
        assert ident["build"]["modules"][0] == ["styxx.diffgate", None], self_file
        assert ident["build"]["modules"][1][1] is not None            # this module still hashes itself


# ------------------------------------------------------------ 4. find_spec executes nothing

def test_module_sha_hashes_a_module_that_would_raise_on_import_without_importing_it(tmp_path, monkeypatch):
    name = f"styxx_identity_probe_raises_{os.getpid()}"
    src = tmp_path / f"{name}.py"
    src.write_bytes(b'raise RuntimeError("this module was executed")\r\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert I.module_sha(name) == hashlib.sha256(b'raise RuntimeError("this module was executed")\n').hexdigest()
    assert name not in sys.modules
    assert I.module_sha(name + "_absent") is None
    monkeypatch.setitem(sys.modules, "styxx.claimdetect", None)   # the SKIPPED-advisory situation
    assert I.module_sha("styxx.claimdetect") is None


# ------------------------------------------------------------------------- 5. delta, reading

@pytest.mark.parametrize("over, state, differs, unknown", [
    ({}, "same", [], []),
    ({"build__reader": "diffgate.js"}, "differs", ["build.reader"], []),
    ({"module__styxx_diffgate": "0" * 64}, "differs", ["build.modules[styxx.diffgate]"], []),
    ({"module__styxx_claimdetect": "0" * 64}, "differs", ["build.modules[styxx.claimdetect]"], []),
    ({"runtime__implementation": "pypy"}, "differs", ["runtime.implementation"], []),
    ({"runtime__version": "3.14.2"}, "differs", ["runtime.version"], []),
    ({"runtime__unicode": "16.0.0"}, "differs", ["runtime.unicode"], []),
    ({"build__styxx": "7.47.0"}, "same", [], []),                  # recorded, never compared
    ({"runtime__case_floor": "16.0"}, "same", [], []),
    ({"module__styxx_declare": None}, "incomplete", [], ["build.modules[styxx.declare]"]),
    ({"runtime__unicode": None}, "incomplete", [], ["runtime.unicode"]),
    ({"build__reader": None, "runtime__version": None}, "incomplete", [], ["build.reader", "runtime.version"]),
])
def test_delta_reads_the_compared_fields_only(over, state, differs, unknown):
    other = _identity(**over)
    d = I.delta(other, X)
    assert (d["state"], d["differs"], d["unknown"]) == (state, differs, unknown)
    e = I.delta(X, other)
    assert (e["state"], e["differs"], e["unknown"]) == (state, differs, unknown)


def test_delta_a_module_named_on_one_side_only_differs_but_an_absent_part_is_unknown():
    fewer = _identity()
    fewer["build"]["modules"] = fewer["build"]["modules"][:3]
    d = I.delta(fewer, X)
    assert d["state"] == "differs"
    assert d["differs"] == ["build.modules[styxx.evidence]", "build.modules[styxx.claimdetect]"]
    runtime_only, un = I.read_identity({"diffgate": "v0", "runtime": X["runtime"]})
    assert un == [] and runtime_only["build"] is None
    d = I.delta(runtime_only, X)
    assert d["state"] == "incomplete" and d["differs"] == []
    assert d["unknown"] == ["build.reader"] + [f"build.modules[{m}]" for m in I.RECORD_MODULES]
    # a null never equals anything, in either direction, and a null never differs
    nulls = _identity(build__reader=None, runtime__implementation=None, runtime__version=None,
                      runtime__unicode=None)
    for pair in nulls["build"]["modules"]:
        pair[1] = None
    assert I.delta(nulls, nulls)["state"] == "incomplete"
    assert I.delta(None, X) == {"state": "unnamed", "differs": [], "unknown": []}


@pytest.mark.parametrize("bad", [
    [1, 2], 7, 1.5, True, "x" * 10000, "\x1b[31m", "\ud800", "",
    {"schema": "styxx.diffgate/instrument/v2"}, {"reader": "styxx.diffgate"},
    {"schema": I.BUILD_SCHEMA, "reader": "r" * 129, "modules": []},
    {"schema": I.BUILD_SCHEMA, "reader": 5, "modules": []},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m", None]] * 17},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m", None], ["m", None]]},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["Styxx.Diffgate", None]]},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m", "A" * 64]]},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m", "a" * 63]]},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m"]]},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": "none"},
    {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [[1, None]]},
])
def test_an_unreadable_identity_is_read_as_unnamed_with_a_quoted_reason_never_raised_on(bad):
    for key in ("build", "runtime"):
        if key == "runtime" and isinstance(bad, dict):
            # the runtime's rule is its own (no schema, three compared fields): an object with none of
            # them is readable, every field null, and reads incomplete (the next test holds the bad ones)
            ident, unreadable = I.read_identity({"diffgate": "v0", "runtime": bad})
            assert unreadable == [] and I.delta(ident, X)["state"] == "incomplete"
            continue
        rec = {"diffgate": "v0", "verdict": "PASS", key: bad}
        ident, unreadable = I.read_identity(rec)
        assert ident is None
        assert len(unreadable) == 1 and unreadable[0].startswith(f"{key} is not an identity this verifier reads (")
        assert unreadable[0].isascii() and len(unreadable[0]) < 300
        assert I.delta(ident, X)["state"] == "unnamed"
        assert I.content_keys(rec) == ["verdict"]


@pytest.mark.parametrize("bad", [{"implementation": 5}, {"version": "v" * 129}, {"unicode": ["15.0.0"]},
                                 {"implementation": "\ud800" * 200}])
def test_an_unreadable_runtime_is_read_as_unnamed(bad):
    ident, unreadable = I.read_identity({"diffgate": "v0", "build": X["build"], "runtime": bad})
    assert ident is None and len(unreadable) == 1 and unreadable[0].startswith("runtime is not an identity")
    assert unreadable[0].isascii()


def test_read_identity_normalises_what_it_reads_and_names_nothing_for_a_bare_record():
    assert I.read_identity({"diffgate": "v0", "verdict": "PASS"}) == (None, [])
    assert I.read_identity("not a record") == (None, [])
    assert I.read_identity(None) == (None, [])
    rec = {"diffgate": "v0", **copy.deepcopy(X)}
    rec["build"]["extra"] = {"nested": 1}
    rec["runtime"]["detail"] = "a build string nobody compares"
    ident, un = I.read_identity(rec)
    assert un == [] and ident == X                       # unknown fields dropped, known ones kept
    assert I.content_keys(rec) == []
    # an empty string says nothing: read as null, so it is unknown, never a value that differs and then
    # prints as `unknown` in a SKEW line
    empty = {"diffgate": "v0", **copy.deepcopy(X)}
    empty["build"]["reader"] = ""
    empty["build"]["styxx"] = ""
    empty["runtime"]["implementation"] = ""
    empty["runtime"]["case_floor"] = ""
    ident, un = I.read_identity(empty)
    assert un == [] and ident["build"]["reader"] is None and ident["build"]["styxx"] is None
    assert ident["runtime"]["implementation"] is None and ident["runtime"]["case_floor"] is None
    d = I.delta(ident, X)
    assert d["state"] == "incomplete" and d["differs"] == []
    assert d["unknown"] == ["build.reader", "runtime.implementation"]
    assert I.describe(ident) == "unknown sha unknown (styxx unknown) on unknown 3.12.10, Unicode 15.0.0"
    assert I.content_keys({"diffgate": "v0", "verdict": "PASS", "build": {}, "x": 1, "runtime": {}}) == ["verdict", "x"]


def test_describe_is_ascii_only_and_never_writes_what_charon_reads_as_a_broken_binding():
    assert I.describe(X) == "styxx.diffgate aaaaaaaaaa (styxx 7.49.0) on cpython 3.12.10, Unicode 15.0.0"
    assert I.describe(None) == "not named"
    hostile = _identity(build__reader="bytes != binding é\x1b[31m", build__styxx="\ud800" * 100,
                        runtime__version="ambiguous  ", runtime__implementation=None)
    hostile["build"]["modules"][0][0] = "bytes != binding é\x1b[31m"
    text = I.describe(hostile)
    assert text.isascii() and "!=" not in text and "\x1b" not in text and len(text) < 400
    assert "ambiguous" in text                           # letters pass; the capsule refuses the word in a problem
    assert C._charon_quotable(text).startswith("an instrument whose recorded name this verifier does not quote")
    assert C._charon_quotable(I.describe(X)) == I.describe(X)
    # the same rule for every token a SKEW line quotes beside the identity, and no replacement holds what
    # charon reads (the word, or `binding` beside `!=`)
    assert C._charon_quotable_name("ambiguous") == "(a name this verifier does not quote)"
    assert C._charon_quotable_name("Ambiguous-3.12") == "(a name this verifier does not quote)"
    assert C._charon_quotable_name("cpython") == "cpython" and C._charon_quotable_name("unknown") == "unknown"
    assert C._charon_quotable_field("build.modules[ambiguous]") == "build.modules[(a name this verifier does not quote)]"
    assert C._charon_quotable_field("build.modules[styxx.diffgate]") == "build.modules[styxx.diffgate]"
    assert C._charon_quotable_field("runtime.version") == "runtime.version"
    for t in (C._charon_quotable(text), C._charon_quotable_name("ambiguous"),
              C._charon_quotable_field("build.modules[ambiguous]")):
        assert not C._charon_reads_as_broken_binding(t) and "binding" not in t, t


# ------------------------------------------------------- 6. the two committed capsules, charon

def test_dogfood_verifies_with_no_problem_and_one_advisory_that_it_names_no_instrument():
    rep = verify_capsule(DOGFOOD)
    assert rep["ok"] is True and rep["problems"] == [] and rep["skew"] is False
    assert rep["gate_identity"]["state"] == "unnamed" and rep["gate_identity"]["recorded"] is None
    assert rep["gate_identity"]["pin"] == "styxx==" + "7.47.0"
    live = rep["gate_identity"]["live"]
    assert live["runtime"]["unicode"] == unicodedata.unidata_version
    adv = _identity_advisories(rep)
    assert len(adv) == 1 and adv[0].startswith("the gate record names no build or runtime (minted before")
    assert I.describe(live) in adv[0]
    code, out = _command(DOGFOOD)
    assert code == 0 and "VERIFIED" in out
    assert f"instrument: recorded not named (pin 'styxx==7.47.0'); re-run here by {I.describe(live)}" in out
    assert out.index("embedded gate verdict:") < out.index("instrument: recorded") < out.index("advisory:")


def test_handoff_keeps_its_one_problem_to_the_character_and_gains_the_cannot_be_classified_advisory():
    rep = verify_capsule(HANDOFF)
    assert rep["ok"] is False and rep["skew"] is False
    assert len(rep["problems"]) == 1 and rep["problems"][0].startswith("gate.claims not reproduced: embedded [")
    assert [p[:160] for p in rep["problems"][:3]] == _log_line(95)["counts"]["problems_detail"]
    assert rep["gate_identity"]["state"] == "unnamed"
    adv = _identity_advisories(rep)
    assert len(adv) == 1
    assert adv[0].startswith("the gate record names no build or runtime, so this failure cannot be told apart "
                             "from INSTRUMENT SKEW or an edit from the record alone")
    assert "'styxx==7.47.0'" in adv[0] and "names a version, not a build" in adv[0]
    assert not any("INSTRUMENT SKEW" in p for p in rep["problems"])
    code, out = _command(HANDOFF)
    assert code == 1 and "CAPSULE FAILS VERIFICATION" in out and "instrument: recorded not named" in out


@pytest.mark.parametrize("seq, path", [(89, DOGFOOD), (95, HANDOFF)])
def test_charon_lines_89_and_95_re_derive_with_no_compared_key_moved(seq, path):
    e = _log_line(seq)
    assert e["subject"]["name"] == path.name and e["kind"] == "capsule-diffgate"
    fresh = charon.derive(path, ROOT)
    assert [k for k in charon._COMPARED_KEYS if charon.jcs(fresh[k]) != charon.jcs(e[k])] == []
    assert fresh["reproduced"] is e["reproduced"] and fresh["counts"]["problems"] == e["counts"]["problems"]
    assert _lf(path) == e["subject"]["sha256"]             # the capsule's bytes are as committed


# ------------------------------------------ 7, 8, 9. synthetic identities through mint and verify

def test_a_record_carrying_x_reproduced_under_y_is_ok_with_the_fields_named(tmp_path, monkeypatch):
    _under(monkeypatch, X)
    out = _mint(tmp_path)
    _, _, _, p = _payload(out)
    assert p["gate"]["build"] == X["build"] and p["gate"]["runtime"] == X["runtime"]
    assert list(p["gate"])[-2:] == ["build", "runtime"] and p["gate"]["diffgate"] == "v0"
    assert p["binding"]["gate"]["value"] == _gate_binding_hash(p["gate"])           # sealed with the identity
    bare = {k: v for k, v in p["gate"].items() if k not in I.IDENTITY_KEYS}
    assert p["binding"]["gate"]["value"] != _gate_binding_hash(bare)
    rep = verify_capsule(out)                                                         # under X: nothing
    assert rep["ok"] and _identity_advisories(rep) == [] and rep["gate_identity"]["state"] == "same"
    _under(monkeypatch, Y)
    rep = verify_capsule(out)                                                         # under Y: D-ok
    assert rep["ok"] is True and rep["problems"] == [] and rep["skew"] is False
    assert rep["gate_identity"]["state"] == "differs"
    assert rep["gate_identity"]["differs"] == ["build.modules[styxx.diffgate]", "runtime.version", "runtime.unicode"]
    assert rep["gate_identity"]["recorded"] == X and rep["gate_identity"]["live"] == Y
    adv = _identity_advisories(rep)
    assert len(adv) == 1 and adv[0].startswith("reproduced under another instrument (differs: "
                                               "build.modules[styxx.diffgate], runtime.version, runtime.unicode)")
    code, text = _command(out)
    assert code == 0 and f"instrument: recorded {I.describe(X)}; re-run here by {I.describe(Y)}" in text


def test_a_resealed_edit_under_x_verified_under_y_is_instrument_skew_ahead_of_the_field_lines(tmp_path, monkeypatch):
    _under(monkeypatch, X)
    out = _mint(tmp_path)
    _edit_and_reseal(out)
    _under(monkeypatch, Y)
    rep = verify_capsule(out)
    assert rep["ok"] is False and rep["skew"] is True and rep["stage"] == "reproduced"
    assert rep["problems"][0].startswith("INSTRUMENT SKEW: the gate record does not reproduce here and names "
                                         "another instrument (differs: build.modules[styxx.diffgate], "
                                         "runtime.version, runtime.unicode); it was made by " + I.describe(X))
    assert "`pip install styxx==7.49.0` on cpython 3.12.10" in rep["problems"][0]
    assert rep["problems"][1].startswith("gate.verdict not reproduced") or rep["problems"][1].startswith("gate.claims not reproduced")
    assert any(p.startswith("gate.claims not reproduced") for p in rep["problems"][1:])
    # (a) every SKEW text names INSTRUMENT SKEW; (c) no problem holds a value of the live identity
    assert all("INSTRUMENT SKEW" in p for p in rep["problems"] if p.startswith("INSTRUMENT"))
    for p in rep["problems"]:
        assert "9.99.1" not in p and "99.0.0" not in p and "f" * 10 not in p
    assert any(a.startswith("the SKEW line's pin is the sealed record's build.styxx") for a in rep["advisory"])
    # (b) charon reads the line as not reproduced, never as UNRESOLVED
    line = charon.derive(out, tmp_path)
    assert line["verdict"] != "UNRESOLVED" and line["reproduced"] is False
    assert line["counts"]["problems"] == len(rep["problems"])
    assert line["counts"]["problems_detail"][0].startswith("INSTRUMENT SKEW")
    code, text = _command(out)
    assert code == 1 and "  - INSTRUMENT SKEW" in text
    # under X itself: no SKEW, the field lines, and S-fail
    _under(monkeypatch, X)
    rep = verify_capsule(out)
    assert rep["ok"] is False and rep["skew"] is False
    assert all(p.startswith("gate.") for p in rep["problems"])
    adv = _identity_advisories(rep)
    assert adv == ["the record names this verifier's own build and runtime and does not reproduce under "
                   "them: what moved is not the instrument"]


def test_an_incomplete_identity_with_an_edit_is_i_fail_never_skew(tmp_path, monkeypatch):
    x_null = _identity(module__styxx_declare=None)
    _under(monkeypatch, x_null)
    out = _mint(tmp_path)
    _under(monkeypatch, X)
    rep = verify_capsule(out)
    assert rep["ok"] and rep["gate_identity"]["state"] == "incomplete"
    adv = _identity_advisories(rep)
    assert len(adv) == 1 and adv[0].startswith("the gate record names its instrument incompletely "
                                               "(unknown: build.modules[styxx.declare]); it reproduced here under ")
    _edit_and_reseal(out)
    rep = verify_capsule(out)
    assert rep["ok"] is False and rep["skew"] is False
    assert all(p.startswith("gate.") for p in rep["problems"])
    adv = _identity_advisories(rep)
    assert len(adv) == 1 and adv[0].startswith("the gate record names its instrument incompletely "
                                               "(unknown: build.modules[styxx.declare]), so this failure cannot be "
                                               "told apart from INSTRUMENT SKEW or an edit from the record alone")
    assert I.describe(x_null) in adv[0]
    assert ("the payload's pin 'styxx==7.49.0' is outside every binding and names a version, not a build"
            in adv[0])
    assert charon.derive(out, tmp_path)["verdict"] != "UNRESOLVED"


@pytest.mark.parametrize("hostile", [
    pytest.param({"module": "ambiguous"}, id="module-name"),
    pytest.param({"runtime__implementation": "ambiguous"}, id="implementation"),
    pytest.param({"module": "bytes_ambiguous", "runtime__implementation": "Ambiguous",
                  "runtime__version": "3.12-ambiguous", "build__styxx": "7.49.0+ambiguous",
                  "build__reader": "ambiguous.reader"}, id="every-token"),
])
def test_a_skew_line_quotes_no_recorded_token_charon_reads_as_a_broken_binding(tmp_path, monkeypatch, hostile):
    """Constraint (b) against a minter who chooses the words (review 1 of phase 1). `ambiguous` is a module
    name the reader admits (`^[a-z_.]{1,64}$`), an implementation, a version and a styxx version string;
    the SKEW line names the field and the recorded identity without quoting the token, keeps the field
    lines, and charon reads the capsule as not reproduced, never UNRESOLVED."""
    minter = _identity(**{k: v for k, v in hostile.items() if k != "module"})
    if "module" in hostile:
        minter["build"]["modules"].append([hostile["module"], "0" * 64])
    _under(monkeypatch, minter)
    out = _mint(tmp_path)                                 # R7's self-verify: the same identity, ok
    _edit_and_reseal(out)
    _under(monkeypatch, Y)
    rep = verify_capsule(out)
    assert rep["ok"] is False and rep["skew"] is True and rep["gate_identity"]["state"] == "differs"
    skew = rep["problems"][0]
    assert skew.startswith("INSTRUMENT SKEW: the gate record does not reproduce here and names another "
                           "instrument (differs: ")
    for p in rep["problems"]:
        assert "ambiguous" not in p.lower() and "!=" not in p, p
    if "module" in hostile:
        assert "build.modules[(a name this verifier does not quote)]" in skew
        assert f"build.modules[{hostile['module']}]" in rep["gate_identity"]["differs"]   # the result keeps it
    if "runtime__implementation" in hostile:
        assert "on (a name this verifier does not quote)" in skew
    if "build__styxx" in hostile:
        assert "pip install" not in skew and "the styxx whose styxx.diffgate the record names, on" in skew
        assert not any(a.startswith("the SKEW line's pin") for a in rep["advisory"])
    else:
        assert "`pip install styxx==7.49.0` on" in skew
    made_by = I.describe(minter)                          # holds the word when the implementation or reader does
    if C._charon_reads_as_broken_binding(made_by):
        assert "it was made by an instrument whose recorded name this verifier does not quote" in skew
    else:
        assert f"it was made by {made_by}" in skew
    assert any(p.startswith("gate.claims not reproduced") for p in rep["problems"][1:])
    line = charon.derive(out, tmp_path)
    assert line["verdict"] != "UNRESOLVED" and line["reproduced"] is False
    assert line["counts"]["problems"] == len(rep["problems"])
    assert line["counts"]["problems_detail"][0].startswith("INSTRUMENT SKEW")
    code, text = _command(out)
    assert code == 1 and "  - INSTRUMENT SKEW" in text and "CAPSULE FAILS VERIFICATION" in text


def test_the_format_key_mismatch_stays_skew_with_no_field_comparison(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["gate"]["diffgate"] = "v-fake"
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(out, html, i, j, p)
    rep = verify_capsule(out)
    assert rep["ok"] is False and rep["skew"] is True
    assert len(rep["problems"]) == 1 and rep["problems"][0].startswith("INSTRUMENT SKEW: installed diffgate 'v0'")
    assert rep["gate_identity"]["state"] == "unnamed" and _identity_advisories(rep) == []


# ----------------------------------------------------------------------------------- 10. R5

def test_r5_never_compares_identity_and_names_the_fields_when_a_diverging_gate_names_another_instrument(tmp_path):
    from styxx.diffgate import gate_diff_text
    live = gate_diff_text(SUMMARY, DIFF, run=None, strict=False).to_dict()
    assert I.IDENTITY_KEYS[0] not in live                 # the writer names nothing in this phase
    out = _mint(tmp_path, gate=live, name="bare")
    assert _payload(out)[3]["gate"] == live
    other_runtime = {**live, **Y}
    out = _mint(tmp_path, gate=other_runtime, name="other")
    assert _payload(out)[3]["gate"] == live               # the embedded gate is the live record, as today
    assert verify_capsule(out)["ok"]
    stale = gate_diff_text("Added styxx/other.py entirely.\n", DIFF, run=None, strict=False).to_dict()
    with pytest.raises(SystemExit) as e:
        _mint(tmp_path, gate={**stale, **Y}, name="stale")
    msg = str(e.value)
    assert "does not reproduce from these bytes" in msg
    # compared against this box's own identity, which Y names nothing of: every module and the runtime
    assert "names another instrument (differs: build.modules[styxx.diffgate]," in msg
    assert "runtime.version, runtime.unicode)" in msg and "9.99.1" not in msg
    with pytest.raises(SystemExit) as e:
        _mint(tmp_path, gate=stale, name="stale2")
    assert "names another instrument" not in str(e.value)  # a diverging gate naming nothing: as before
    with pytest.raises(SystemExit, match="unknown diffgate version"):
        _mint(tmp_path, gate={**live, "diffgate": "v1"}, name="v1")


# ----------------------------------------------------------------------- malformed identities

@pytest.mark.parametrize("bad", [[1, 2], 7, "x" * 10000, "\x1b[31m", "é‮", {"schema": "zz"},
                                 {"schema": I.BUILD_SCHEMA, "reader": "styxx.diffgate", "modules": [["m", None]] * 17},
                                 [[[[[[[[["deep"]]]]]]]]], None])
def test_verify_never_ends_in_a_traceback_on_a_malformed_identity_object(tmp_path, bad):
    """A lone surrogate is not among these: a record holding one cannot be sealed (JCS cannot encode it
    to UTF-8), so it fails at the binding stage as before, and read_identity's own test reads it."""
    out = _mint(tmp_path)
    for key in ("build", "runtime"):
        html, i, j, p = _payload(out)
        p["gate"][key] = bad
        p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
        _rewrite(out, html, i, j, p)
        rep = verify_capsule(out)
        assert isinstance(rep["ok"], bool) and isinstance(rep["problems"], list)
        assert rep["ok"] is True, rep["problems"]        # the content reproduces; the identity is apart
        adv = _identity_advisories(rep)
        if key == "runtime" and isinstance(bad, dict):
            # an object with none of the runtime's three fields is readable, every field null: I-ok
            assert rep["gate_identity"]["state"] == "incomplete"
            assert len(adv) == 1 and adv[0].startswith("the gate record names its instrument incompletely")
        else:
            assert rep["gate_identity"]["state"] == "unnamed" and rep["gate_identity"]["recorded"] is None
            assert len(adv) == 2, adv                     # the unreadable key, then U-ok
            assert adv[0].startswith(f"gate.{key} is not an identity this verifier reads (")
            assert adv[0].endswith("; read as unnamed") and adv[0].isascii() and len(adv[0]) < 300
            assert adv[1].startswith("the gate record names no build or runtime (minted before")
        code, text = _command(out)
        assert code == 0 and "VERIFIED" in text and text.isascii()
        line = charon.derive(out, tmp_path)
        assert line["reproduced"] is True and line["verdict"] != "UNRESOLVED"
        out = _mint(tmp_path, name=f"again_{key}")


def test_a_null_identity_key_is_read_not_raised_on(tmp_path):
    out = _mint(tmp_path)
    html, i, j, p = _payload(out)
    p["gate"]["build"] = None
    p["gate"]["runtime"] = None
    p["binding"]["gate"]["value"] = _gate_binding_hash(p["gate"])
    _rewrite(out, html, i, j, p)
    rep = verify_capsule(out)
    assert rep["ok"] is True and rep["gate_identity"]["state"] == "unnamed"
    assert len([a for a in _identity_advisories(rep) if "not an object: null" in a]) == 2


# ------------------------------------------------------------------------------ 11. the page

def test_the_page_meta_line_shows_the_recorded_identity_only_when_it_is_strings(tmp_path):
    assert "gate made by" in _TEMPLATE_V02 and "sealed in binding.gate; this page re-runs nothing" in _TEMPLATE_V02
    assert "P.verifier.styxx_version + madeBy;" in _TEMPLATE_V02
    assert "compares that identity apart from" in _TEMPLATE_V02         # the footer's sentence
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    start = _TEMPLATE_V02.index("  const txt = (o, k) =>")
    end = _TEMPLATE_V02.index("})();", start) + len("})();")
    snippet = _TEMPLATE_V02[start:end]
    script = tmp_path / "meta.js"
    script.write_text(
        "const g = JSON.parse(process.argv[2]);\n" + snippet + "\nprocess.stdout.write(madeBy);\n",
        encoding="utf-8")
    full = {"diffgate": "v0", **X}
    for g, want in [
        (full, " · gate made by styxx.diffgate aaaaaaaaaa on cpython 3.12.10, Unicode 15.0.0 "
               "(sealed in binding.gate; this page re-runs nothing)"),
        ({"diffgate": "v0"}, ""),
        ({**full, "runtime": {"implementation": 3, "version": "3.12.10", "unicode": "15.0.0"}}, ""),
        ({**full, "build": "styxx.diffgate"}, ""),
        ({**full, "build": {**X["build"], "modules": [["styxx.diffgate", None]]}}, ""),
        ({**full, "build": {**X["build"], "reader": None}}, ""),
    ]:
        r = subprocess.run([node, str(script), json.dumps(g)], capture_output=True, text=True, encoding="utf-8",
                           check=True)
        assert r.stdout == want, (g, r.stdout)


# ------------------------------------------------------------------- 12. python -m styxx.instrument

def test_python_m_styxx_instrument_prints_the_identity_this_install_verifies_under():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "styxx.instrument"], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", env=env, check=True)
    assert r.stdout.isascii()
    out = json.loads(r.stdout)
    assert set(out) == {"schema", "styxx_version", "build", "runtime", "loaded_from"}
    assert out["schema"] == I.BUILD_SCHEMA
    from styxx._version import __version__
    assert out["styxx_version"] == __version__ == out["build"]["styxx"]
    assert out["build"]["modules"][0][0] == "styxx.diffgate" and len(out["build"]["modules"][0][1]) == 64
    assert out["runtime"]["unicode"] == unicodedata.unidata_version
    assert out["runtime"]["version"] == platform.python_version()
    assert "detail" not in out["runtime"] and sys.version not in r.stdout
    assert out["loaded_from"] in ("installed package", "source tree", "unknown")
    assert str(ROOT) not in r.stdout and str(ROOT).replace("\\", "/") not in r.stdout

# -*- coding: utf-8 -*-
"""prereg-ci B2: the seals manifest, the decision in effect, and the AMENDED_AFTER_* readings.

DESIGN 3.2(b): an optional manifest, ``prereg-ci.seals.jsonl``, overrides the default seal per path
and carries amendments and time tokens, one JSON line per sealed file. Every ``blob_sha256``
equals ``git show <commit>:<path> | sha256sum``; the file is append-only on the default branch;
``"own"`` means the amendment carries its own fence, which then governs, ``"inherits"`` that the
decision digest in effect is unchanged; ``"seen"`` is required on an amendment line and takes
reproducible-science's four access levels; the generator writes null where the lab's record
states none.

DESIGN 3.4 C4: the decision in effect for a receipt is the fence at the seal, replaced by the
latest manifest amendment with ``"decision": "own"`` whose commit is a strict ancestor of R; a
later amendment prints ``AMENDED_AFTER_RECEIPT``; a receipt under the old decision, in history
before an amendment that governs a later receipt, prints ``AMENDED_AFTER_DATA`` with the earlier
receipt's verdict and the amendment's ``seen`` (a NOTE; a FAIL under ``amend_after_data = "fail"``).

R10: the eight rows of ``papers/checksum/SEALS_2026_09_13.md`` (lines 33-40) reproduce under
``git show <commit>:<path> | sha256sum``. ``conformance/prereg_ci/gen_seals_manifest.py`` turns them
into eight manifest lines (row 8 withdrawn, rows 3b, 6b and 7b amendments, ``seen`` null); the
committed copy is ``conformance/prereg_ci/seals_manifest_from_SEALS_2026_09_13.jsonl``.

PLAN K9: ``seal --amends`` takes a line id or a path; a path names its newest line, and a path with
no line gets its default-seal line (the landing rule) appended before the amendment. PLAN K10, as
built: ``seen`` null is accepted on an amendment line only when the manifest held that line where it
began and the line's commit was on the default branch before the pull request that began it (a
record imported when the manifest began); any later null, and any missing key, is refused.

Every repository here is made with ``git init`` in tmp_path, with no remote; the history tests
read styxx's own objects at ``REV`` and skip, saying why, where that commit is not in the store.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from tests.test_prereg_ci_git import Git, digest, gates_block, prereg, receipt  # noqa: E402

P = "p/PREREG_a.md"
A = "p/AMENDMENT_a.md"
C = "p/CORRECTION_a.md"
M = "prereg-ci.seals.jsonl"
SEALS = "papers/checksum/SEALS_2026_09_13.md"
COPY = ROOT / "conformance" / "prereg_ci" / "seals_manifest_from_SEALS_2026_09_13.jsonl"
SEEN = ("nothing run", "no results seen", "results not opened", "results seen")


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def line(id_, kind, path, commit, data, amends=None, decision="own", seen=None, time=()):
    return {"id": id_, "kind": kind, "path": path, "commit": commit, "blob_sha256": sha(data), "amends": amends,
            "decision": decision, "seen": seen, "time": list(time)}


def jsonl(*entries) -> bytes:
    return "".join(json.dumps(e) + "\n" for e in entries).encode("utf-8")


def codes(findings):
    return [(f["code"], f["kind"]) for f in findings]


# ---- the command module's surface ---------------------------------------------------------------------

def test_the_cli_module_exports_main_alone_and_the_package_still_exports_the_design_names():
    """DESIGN 3.1: the public surface is core.rescore, .compare, .licences and the CLI."""
    tree = ast.parse((ROOT / "styxx" / "prereg_ci" / "cli.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    assert {n for n in names if not n.startswith("_")} == {"main"}
    from styxx.prereg_ci import cli
    assert cli.__all__ == ["main"]
    import styxx.prereg_ci as pkg
    assert set(pkg.__all__) <= {"rescore", "compare", "licences"}
    assert (ROOT / "styxx" / "prereg_ci" / "__main__.py").is_file()


# ---- parsing and rendering (pure) ----------------------------------------------------------------------

C40 = "a" * 40
VALID = [line("1", "prereg", P, C40, prereg(0.5)),
         line("1b", "amendment", A, "b" * 40, prereg(0.6, "amendment"), amends="1", decision="own",
              seen="results seen"),
         line("1c", "withdrawn", P, "c" * 64, prereg(0.5, "v2"), amends="1", decision="inherits")]


def test_the_seen_levels_are_reproducible_sciences_four():
    assert tuple(core._SEEN) == SEEN
    assert tuple(core._MANIFEST_KEYS) == ("id", "kind", "path", "commit", "blob_sha256", "amends", "decision",
                                          "seen", "time")


def test_a_valid_manifest_parses_and_renders_back_to_its_bytes():
    raw = jsonl(*VALID)
    got = core._manifest(raw)
    assert got == VALID
    assert core._manifest_bytes(got) == raw
    assert core._manifest(b"") == []
    assert core._manifest_bytes([]) == b""
    # the rendering is canonical: key order and spacing are the design's, whatever the input's
    shuffled = "".join(json.dumps(dict(reversed(list(e.items()))), separators=(",", ":")) + "\n" for e in VALID)
    assert core._manifest(shuffled.encode("utf-8")) == VALID
    assert core._manifest_bytes(core._manifest(shuffled.encode("utf-8"))) == raw


def _refused(data: bytes):
    with pytest.raises(core._ManifestError) as e:
        core._manifest(data)
    return e.value


@pytest.mark.parametrize("name, mutate, where", [
    ("not JSON", lambda es: b"{not json}\n", 1),
    ("an array, not an object", lambda es: b"[1]\n", 1),
    ("a duplicate key", lambda es: b'{"id": "1", "id": "2"}\n', 1),
    ("a key the format does not name", lambda es: jsonl({**es[0], "extra": 1}), 1),
    ("a key missing", lambda es: jsonl({k: v for k, v in es[0].items() if k != "time"}), 1),
    ("an empty id", lambda es: jsonl({**es[0], "id": ""}), 1),
    ("an id with a space", lambda es: jsonl({**es[0], "id": "1 b"}), 1),
    ("an id used twice", lambda es: jsonl(es[0], {**es[1], "id": "1"}), 2),
    ("an unknown kind", lambda es: jsonl({**es[0], "kind": "sealed-prereg"}), 1),
    ("an absolute path", lambda es: jsonl({**es[0], "path": "/p/PREREG_a.md"}), 1),
    ("a dot-dot path", lambda es: jsonl({**es[0], "path": "p/../PREREG_a.md"}), 1),
    ("a backslash path", lambda es: jsonl({**es[0], "path": "p\\PREREG_a.md"}), 1),
    ("a drive path", lambda es: jsonl({**es[0], "path": "C:/p/PREREG_a.md"}), 1),
    ("an abbreviated commit", lambda es: jsonl({**es[0], "commit": "a" * 8}), 1),
    ("an upper-case commit", lambda es: jsonl({**es[0], "commit": "A" * 40}), 1),
    ("a short digest", lambda es: jsonl({**es[0], "blob_sha256": "0" * 63}), 1),
    ("a prereg line that amends", lambda es: jsonl({**es[0], "amends": "1"}), 1),
    ("a prereg line that inherits", lambda es: jsonl({**es[0], "decision": "inherits"}), 1),
    ("an amendment of nothing", lambda es: jsonl(es[0], {**es[1], "amends": None}), 2),
    ("an amendment of a line not above it", lambda es: jsonl(es[1], es[0]), 1),
    ("an amendment of an unknown id", lambda es: jsonl(es[0], {**es[1], "amends": "9"}), 2),
    ("an amendment of a withdrawn line", lambda es: jsonl(es[0], es[2], {**es[1], "amends": "1c"}), 3),
    ("an unknown decision", lambda es: jsonl(es[0], {**es[1], "decision": "replaces"}), 2),
    ("a seen level the format does not name", lambda es: jsonl(es[0], {**es[1], "seen": "some results seen"}), 2),
    ("time is not a list", lambda es: jsonl({**es[0], "time": "p/PREREG_a.md.tsr"}), 1),
    ("a time token that is no path", lambda es: jsonl({**es[0], "time": ["/abs.tsr"]}), 1),
    ("a time token listed twice", lambda es: jsonl({**es[0], "time": ["p/a.tsr", "p/a.tsr"]}), 1),
    ("two prereg lines for one path", lambda es: jsonl(es[0], {**es[0], "id": "2"}), 2),
    ("a blank line", lambda es: jsonl(es[0]) + b"\n" + jsonl(es[1]), 2),
    ("no newline at the end", lambda es: jsonl(es[0])[:-1], 1),
    ("a carriage return", lambda es: jsonl(es[0]).replace(b"\n", b"\r\n"), 1),
    ("a byte-order mark", lambda es: b"\xef\xbb\xbf" + jsonl(es[0]), 1),
    ("bytes that are not UTF-8", lambda es: jsonl(es[0]).replace(b'"1"', b'"\xff"'), None),
])
def test_a_malformed_manifest_is_refused_naming_the_line(name, mutate, where):
    e = _refused(mutate(VALID))
    assert (e.code, e.kind) == ("MANIFEST_INVALID", "INPUT"), (name, e.code, e.detail)
    assert e.line == where, (name, e.line, e.detail)


def test_an_amendment_line_without_seen_is_refused_as_seen_missing():
    """DESIGN 3.2(b): "seen" is required on an amendment line; a prereg line without a key is an
    ordinary malformed line."""
    no_seen = {k: v for k, v in VALID[1].items() if k != "seen"}
    e = _refused(jsonl(VALID[0], no_seen))
    assert (e.code, e.kind, e.line) == ("MANIFEST_SEEN_MISSING", "FAIL", 2)
    withdrawn = {k: v for k, v in VALID[2].items() if k != "seen"}
    e = _refused(jsonl(VALID[0], withdrawn))
    assert (e.code, e.line) == ("MANIFEST_SEEN_MISSING", 2)
    e = _refused(jsonl({k: v for k, v in VALID[0].items() if k != "seen"}))
    assert (e.code, e.line) == ("MANIFEST_INVALID", 1)


def test_a_manifest_is_bytes():
    with pytest.raises(TypeError):
        core._manifest(jsonl(*VALID).decode("utf-8"))


def test_the_lineage_of_a_seal_line_is_every_line_below_it_in_file_order():
    es = core._manifest(jsonl(VALID[0], VALID[1], line("2", "prereg", "q/PREREG_b.md", C40, "x"),
                              line("1d", "amendment", C, "d" * 40, "y", amends="1b", decision="inherits",
                                   seen="nothing run"),
                              VALID[2]))
    assert [e["id"] for e in core._lineage(es, "1")] == ["1b", "1d", "1c"]
    assert core._lineage(es, "2") == []
    assert core._lineage(es, "9") == []


# ---- the decision in effect (pure) --------------------------------------------------------------------

SEAL = core._decision(prereg(0.5))
OWN6 = core._decision(prereg(0.6, "amendment"))
PROSE = core._decision("# CORRECTION\n\nprose only, no fence\n")


def _seal(id_="1"):
    return {"id": id_, "gates_sha256": SEAL["gates_sha256"], "bytes": prereg(0.5).encode("utf-8")}


def _chain(*items):
    """(id, kind, decision, seen, decision reading, bytes) -> (entry, reading, bytes)"""
    out = []
    for id_, kind, decision, seen, reading, data in items:
        e = line(id_, kind, A, "e" * 40, data, amends="1", decision=decision, seen=seen)
        out.append((e, reading, data.encode("utf-8") if isinstance(data, str) else data))
    return out


def test_an_own_amendment_before_the_receipt_governs_and_its_digest_is_sealed():
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", "own", "results seen", OWN6,
                                                      prereg(0.6, "amendment"))))
    eff = core._in_effect(_seal(), readings, {"1b": True})
    assert (eff["source"], eff["gates_sha256"]) == ("1b", digest(0.6))
    assert eff["sealed"] == [digest(0.5), digest(0.6)]
    assert eff["bytes"] == prereg(0.6, "amendment").encode("utf-8")
    assert [(a["id"], a["in_effect"], a["seen"]) for a in eff["amendments"]] == [("1b", True, "results seen")]
    assert eff["findings"] == []
    # the same line after the receipt: the seal governs, and the later amendment is printed
    eff = core._in_effect(_seal(), readings, {"1b": False})
    assert (eff["source"], eff["gates_sha256"], eff["sealed"]) == ("1", digest(0.5), [digest(0.5)])
    assert codes(eff["findings"]) == [("AMENDED_AFTER_RECEIPT", "NOTE")]
    assert "1b" in eff["findings"][0]["detail"] and eff["findings"][0]["id"] == "1b"


def test_an_inherits_amendment_keeps_the_digest_and_is_listed_with_its_seen():
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", "inherits", "no results seen", PROSE,
                                                      "# CORRECTION\n\nprose only, no fence\n")))
    eff = core._in_effect(_seal(), readings, {"1b": True})
    assert (eff["source"], eff["gates_sha256"], eff["sealed"]) == ("1", digest(0.5), [digest(0.5)])
    assert [(a["id"], a["decision"], a["in_effect"], a["seen"]) for a in eff["amendments"]] == [
        ("1b", "inherits", True, "no results seen")]
    assert eff["findings"] == []


def test_the_latest_own_amendment_before_the_receipt_governs_in_file_order():
    own7 = core._decision(prereg(0.7, "second amendment"))
    readings = core._amendment_readings(SEAL, _chain(
        ("1b", "amendment", "own", "results seen", OWN6, prereg(0.6, "amendment")),
        ("1c", "amendment", "inherits", "results seen", PROSE, "# CORRECTION\n\nprose\n"),
        ("1d", "amendment", "own", "results seen", own7, prereg(0.7, "second amendment"))))
    eff = core._in_effect(_seal(), readings, {"1b": True, "1c": True, "1d": True})
    assert (eff["source"], eff["gates_sha256"]) == ("1d", digest(0.7))
    assert eff["sealed"] == [digest(0.5), digest(0.6), digest(0.7)]
    eff = core._in_effect(_seal(), readings, {"1b": True, "1c": True, "1d": False})
    assert (eff["source"], eff["gates_sha256"]) == ("1b", digest(0.6))
    assert codes(eff["findings"]) == [("AMENDED_AFTER_RECEIPT", "NOTE")] and eff["findings"][0]["id"] == "1d"


def test_a_withdrawn_line_is_never_in_effect_and_prints_no_finding():
    readings = core._amendment_readings(SEAL, _chain(("8", "withdrawn", "own", None, OWN6,
                                                      prereg(0.6, "amendment"))))
    for before in (True, False):
        eff = core._in_effect(_seal(), readings, {"8": before})
        assert (eff["source"], eff["gates_sha256"], eff["sealed"]) == ("1", digest(0.5), [digest(0.5)])
        assert [(a["id"], a["kind"], a["in_effect"]) for a in eff["amendments"]] == [("8", "withdrawn", False)]
        assert eff["findings"] == []


@pytest.mark.parametrize("decision, reading, data, why", [
    ("own", PROSE, "# CORRECTION\n\nprose only\n", "holds no fence"),
    ("inherits", OWN6, prereg(0.6, "amendment"), "holds the fence"),
    ("own", core._decision("```gates\n{not json\n```\n```gates\n{}\n```\n"), "two fences", "refuses"),
], ids=["own-but-no-fence", "inherits-but-another-fence", "own-but-a-refused-fence"])
def test_a_line_whose_decision_its_bytes_do_not_bear_out_is_refused(decision, reading, data, why):
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", decision, "results seen", reading, data)))
    assert readings[0]["finding"]["code"] == "MANIFEST_DECISION_DIFFERS"
    assert why in readings[0]["finding"]["detail"]
    eff = core._in_effect(_seal(), readings, {"1b": True})
    assert (eff["source"], eff["gates_sha256"]) == ("1", digest(0.5))
    assert codes(eff["findings"]) == [("MANIFEST_DECISION_DIFFERS", "FAIL")]
    assert [a["in_effect"] for a in eff["amendments"]] == [False]


def test_an_inherits_line_holding_the_fence_in_effect_is_read_as_inheriting():
    """SEALS row 8: the PREREG itself, amended in place outside its fence; its fence is the one in
    effect, so the line inherits it (the design's example line 8)."""
    same = core._decision(prereg(0.5, "v1 with a section appended"))
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", "inherits", "nothing run", same,
                                                      prereg(0.5, "v1 with a section appended"))))
    assert readings[0]["finding"] is None and readings[0]["decision_identical"] is True
    eff = core._in_effect(_seal(), readings, {"1b": True})
    assert (eff["source"], eff["gates_sha256"], eff["findings"]) == ("1", digest(0.5), [])


def test_an_own_amendment_whose_decision_tuple_is_unchanged_is_marked_decision_identical():
    """DESIGN 3.4 C0: an amendment whose canonical decision tuple (metric, op, value, rows) is unchanged
    is marked decision-identical; its digest still governs."""
    spec = json.loads(gates_block(0.5))
    spec["power_basis"] = {"G0": "prose about power, fixed after landing"}
    fixed = "```gates\n" + json.dumps(spec, indent=1) + "\n```\n"
    moved = core._decision(prereg(0.6, "amendment"))
    readings = core._amendment_readings(SEAL, _chain(
        ("1b", "amendment", "own", "nothing run", core._decision(fixed), fixed),
        ("1c", "amendment", "own", "results seen", moved, prereg(0.6, "amendment"))))
    assert [(r["id"], r["decision_identical"], r["finding"]) for r in readings] == [
        ("1b", True, None), ("1c", False, None)]
    eff = core._in_effect(_seal(), readings, {"1b": True, "1c": False})
    assert eff["source"] == "1b" and eff["gates_sha256"] == sha(json.dumps(spec, indent=1))


def test_a_default_seal_has_no_manifest_id_and_no_lineage():
    eff = core._in_effect({"id": None, "gates_sha256": digest(0.5), "bytes": prereg(0.5).encode("utf-8")}, [], {})
    assert (eff["source"], eff["gates_sha256"], eff["sealed"], eff["amendments"], eff["findings"]) == (
        None, digest(0.5), [digest(0.5)], [], [])
    ungated = core._in_effect({"id": None, "gates_sha256": None, "bytes": b"# prose\n"}, [], {})
    assert (ungated["gates_sha256"], ungated["sealed"]) == (None, [])


def test_amended_after_data_names_the_earlier_receipt_its_verdict_and_the_amendments_seen():
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", "own", "results seen", OWN6,
                                                      prereg(0.6, "amendment"))))
    eff = core._in_effect(_seal(), readings, {"1b": True})
    earlier = [{"path": "p/receipt_0.json", "commit": "f" * 40, "before_amendment": True, "source": "1",
                "verdict": "PASS", "code": None},
               {"path": "p/receipt_x.json", "commit": "9" * 40, "before_amendment": False, "source": "1",
                "verdict": "PASS", "code": None},
               {"path": "p/receipt_y.json", "commit": "8" * 40, "before_amendment": True, "source": "1b",
                "verdict": "HOLD", "code": None}]
    found = core._amended_after_data(eff, earlier)
    assert codes(found) == [("AMENDED_AFTER_DATA", "NOTE")]
    d = found[0]["detail"]
    assert "p/receipt_0.json@" + "f" * 12 in d and "PASS" in d and "the sealed table" in d
    assert "results seen" in d and found[0]["id"] == "1b"
    assert codes(core._amended_after_data(eff, earlier, policy="fail")) == [("AMENDED_AFTER_DATA", "FAIL")]
    with pytest.raises(ValueError):
        core._amended_after_data(eff, earlier, policy="warn")
    # under the seal itself no amendment governs, so nothing was amended after data
    assert core._amended_after_data(core._in_effect(_seal(), readings, {"1b": False}), earlier) == []


def test_amended_after_data_prints_a_null_seen_as_not_stated_and_a_refused_rescore_by_its_code():
    readings = core._amendment_readings(SEAL, _chain(("1b", "amendment", "own", None, OWN6,
                                                      prereg(0.6, "amendment"))))
    eff = core._in_effect(_seal(), readings, {"1b": True})
    found = core._amended_after_data(eff, [{"path": "r.json", "commit": "f" * 40, "before_amendment": True,
                                            "source": "1", "verdict": None, "code": "SCORER_REFUSES"}])
    assert "not stated" in found[0]["detail"] and "SCORER_REFUSES" in found[0]["detail"]


# ---- synthetic repositories: blob checks, append-only, seen -------------------------------------------

@pytest.fixture(scope="module")
def amended(tmp_path_factory):
    """main, linear: the plan (bar 0.5) lands; a receipt r0 is committed under it; an own amendment
    (bar 0.6) and an inherits CORRECTION are committed and sealed in the manifest; r1 is committed
    under the amendment; a second own amendment (bar 0.7) is sealed; r2 is committed under it.
    m = 0.55 throughout: PASS under 0.5, HOLD under 0.6 and 0.7."""
    g = Git(tmp_path_factory.mktemp("amended"))
    g.write("README.md", "r\n")
    m0 = g.commit("m0")
    g.write(P, prereg(0.5))
    seal = g.commit("seal the plan")
    g.write("p/receipt_0.json", receipt(m=0.55))
    r0 = g.commit("run under the sealed table")
    g.write(A, prereg(0.6, "amendment"))
    a1 = g.commit("an amendment with its own table")
    g.write(C, "# CORRECTION\n\nprose only\n")
    c1 = g.commit("a correction that keeps the table")
    lines = [line("1", "prereg", P, seal, prereg(0.5)),
             line("1b", "amendment", A, a1, prereg(0.6, "amendment"), amends="1", seen="results seen"),
             line("1c", "amendment", C, c1, "# CORRECTION\n\nprose only\n", amends="1", decision="inherits",
                  seen="results seen")]
    g.write(M, jsonl(*lines))
    man1 = g.commit("seal the amendments")
    g.write("p/receipt_1.json", receipt(m=0.55))
    r1 = g.commit("run under the amendment")
    g.write("p/AMENDMENT_2.md", prereg(0.7, "second amendment"))
    a2 = g.commit("a second amendment")
    lines2 = lines + [line("1d", "amendment", "p/AMENDMENT_2.md", a2, prereg(0.7, "second amendment"),
                           amends="1", seen="results seen")]
    g.write(M, jsonl(*lines2))
    man2 = g.commit("seal the second amendment")
    g.write("p/receipt_2.json", receipt(m=0.55))
    r2 = g.commit("run under the second amendment")
    with G._Repo(g.root) as repo:
        yield SimpleNamespace(g=g, repo=repo, m0=m0, seal=seal, r0=r0, a1=a1, c1=c1, man1=man1, r1=r1, a2=a2,
                              man2=man2, r2=r2, lines=lines, lines2=lines2)


def test_the_manifest_at_a_revision_is_read_from_its_blob(amended):
    repo = amended.repo
    assert G._MANIFEST == M
    assert G._manifest_at(repo, amended.m0) is None
    assert G._manifest_at(repo, amended.man1) == amended.lines
    assert G._manifest_at(repo, "main") == amended.lines2


def test_every_blob_sha256_is_the_sha256_of_the_blob_at_its_commit(amended):
    repo = amended.repo
    assert G._manifest_blobs(repo, amended.lines2, "main") == []
    for e in amended.lines2:
        shown = subprocess.run(["git", "-C", str(amended.g.root), "show", f"{e['commit']}:{e['path']}"],
                               capture_output=True, check=True).stdout
        assert sha(shown) == e["blob_sha256"], e["id"]
    wrong = [dict(amended.lines[0], blob_sha256="0" * 64)]
    assert codes(G._manifest_blobs(repo, wrong, "main")) == [("MANIFEST_BLOB_DIFFERS", "FAIL")]
    absent = [dict(amended.lines[0], path="p/never.md")]
    found = G._manifest_blobs(repo, absent, "main")
    assert codes(found) == [("MANIFEST_BLOB_DIFFERS", "FAIL")] and "no regular file" in found[0]["detail"]
    with pytest.raises(G._Refusal) as e:
        G._manifest_blobs(repo, [dict(amended.lines[0], commit="1" * 40)], "main")
    assert e.value.code == "COMMIT_ABSENT"
    # a line's commit must be in the history the manifest is read at: a later commit is not
    found = G._manifest_blobs(repo, [dict(amended.lines2[3])], amended.man1)
    assert codes(found) == [("MANIFEST_COMMIT_UNREACHABLE", "FAIL")]


def test_a_line_whose_commit_names_a_tag_or_a_tree_is_an_input_error(tmp_path):
    """The B2 review, case D: git would peel an annotated tag's id to its commit and print the tag's id
    as the seal commit; a tree's id read as an absent commit (UNCHECKABLE). Both are MANIFEST_INVALID;
    an id the object store lacks stays COMMIT_ABSENT."""
    g = Git(tmp_path / "objects")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g("tag", "-a", "-m", "t", "v1", seal)
    tag, tree = g("rev-parse", "v1"), g("rev-parse", seal + "^{tree}")
    blob = g("rev-parse", f"{seal}:{P}")
    g.write("p/receipt.json", receipt(m=0.6))
    r = g.commit("receipt")
    with G._Repo(g.root) as repo:
        assert (repo.object_type(tag), repo.object_type(tree), repo.object_type(seal)) == ("tag", "tree", "commit")
        assert repo.object_type("1" * 40) is None
        for oid, typ in ((tag, "tag"), (tree, "tree"), (blob, "blob")):
            es = [line("1", "prereg", P, oid, prereg(0.5))]
            with pytest.raises(G._Refusal) as e:
                G._manifest_blobs(repo, es, "main")
            assert (e.value.code, e.value.kind) == ("MANIFEST_INVALID", "INPUT") and f"of a {typ}" in e.value.detail
            with pytest.raises(G._Refusal) as e:
                G._effect(repo, P, "p/receipt.json", r, "main", "main", es)
            assert e.value.code == "MANIFEST_INVALID"
        assert G._manifest_blobs(repo, [line("1", "prereg", P, seal, prereg(0.5))], "main") == []


def test_the_blob_is_hashed_as_git_stores_it_never_with_newlines_translated(tmp_path):
    """SEALS 'the rule': the bytes as git stores them, never a working copy read through text mode."""
    g = Git(tmp_path / "crlf")
    data = prereg(0.5).replace("\n", "\r\n").encode("utf-8")
    g.write(P, data)
    c = g.commit("a CRLF blob")
    with G._Repo(g.root) as repo:
        assert G._manifest_blobs(repo, [line("1", "prereg", P, c, data)], "main") == []
        lf = line("1", "prereg", P, c, prereg(0.5))
        assert codes(G._manifest_blobs(repo, [lf], "main")) == [("MANIFEST_BLOB_DIFFERS", "FAIL")]


def test_a_line_whose_commit_sits_on_a_branch_never_merged_is_unreachable(tmp_path):
    g = Git(tmp_path / "unmerged")
    g.write(P, prereg(0.5))
    g.commit("plan")
    g("switch", "-q", "-c", "side")
    g.write(A, prereg(0.6, "amendment"))
    side = g.commit("an amendment on a side branch")
    g("switch", "-q", "main")
    with G._Repo(g.root) as repo:
        found = G._manifest_blobs(repo, [line("1b", "amendment", A, side, prereg(0.6, "amendment"), amends="1",
                                              seen="results seen")], "main")
        assert codes(found) == [("MANIFEST_COMMIT_UNREACHABLE", "FAIL")]
        assert G._manifest_blobs(repo, [line("1b", "amendment", A, side, prereg(0.6, "amendment"), amends="1",
                                             seen="results seen")], "side") == []


@pytest.fixture()
def manifested(tmp_path):
    g = Git(tmp_path / "manifested")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    one = line("1", "prereg", P, seal, prereg(0.5))
    g.write(M, jsonl(one))
    base = g.commit("seal it")
    return SimpleNamespace(g=g, seal=seal, one=one, base=base)


def test_a_pull_request_that_appends_to_the_manifest_passes_the_append_only_check(manifested):
    g = manifested.g
    g("switch", "-q", "-c", "pr")
    g.write(A, prereg(0.6, "amendment"))
    a = g.commit("amendment")
    g.write(M, jsonl(manifested.one, line("1b", "amendment", A, a, prereg(0.6, "amendment"), amends="1",
                                          seen="results seen")))
    g.commit("append")
    with G._Repo(g.root) as repo:
        assert G._manifest_append_only(repo, "main", "pr") == []
        assert G._manifest_chain(repo, "pr") == []


@pytest.mark.parametrize("how", ["edit a line", "drop the last line", "delete the file", "rename the file",
                                 "rewrite and append"])
def test_a_pull_request_that_rewrites_the_manifest_is_refused(manifested, how):
    g = manifested.g
    g("switch", "-q", "-c", "pr")
    if how == "edit a line":
        g.write(M, jsonl(dict(manifested.one, seen="nothing run")))
    elif how == "drop the last line":
        g.write(M, b"")
    elif how == "delete the file":
        g("rm", "-q", M)
    elif how == "rename the file":
        g.mv(M, "seals.jsonl")
    else:
        g.write(M, jsonl(dict(manifested.one, time=["p/PREREG_a.md.tsr"]), dict(manifested.one, id="2",
                                                                                  path="q/PREREG_b.md")))
    g.commit(how)
    with G._Repo(g.root) as repo:
        found = G._manifest_append_only(repo, "main", "pr")
        assert codes(found) == [("MANIFEST_REWRITTEN", "FAIL")], how
        assert found[0]["path"] == M


def test_a_manifest_added_by_the_pull_request_has_nothing_to_append_to(tmp_path):
    g = Git(tmp_path / "fresh")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g("switch", "-q", "-c", "pr")
    g.write(M, jsonl(line("1", "prereg", P, seal, prereg(0.5))))
    g.commit("a manifest")
    with G._Repo(g.root) as repo:
        assert G._manifest_append_only(repo, "main", "pr") == []


def test_the_chain_reading_names_the_commit_that_rewrote_the_manifest_on_main(manifested):
    g = manifested.g
    g.write(M, jsonl(dict(manifested.one, seen="nothing run")))
    bad = g.commit("rewrite the seal line on main")
    g.write(M, jsonl(dict(manifested.one, seen="nothing run"), dict(manifested.one, id="2", path="q/b.md")))
    g.commit("append after the rewrite")
    with G._Repo(g.root) as repo:
        found = G._manifest_chain(repo, "main")
        assert codes(found) == [("MANIFEST_REWRITTEN", "FAIL")]
        assert bad[:12] in found[0]["detail"]


def test_seen_null_is_accepted_only_on_the_record_the_manifest_imported(tmp_path):
    """PLAN K10 as built: an amendment line may say seen null only when the manifest held it at the
    commit that added the manifest and its commit is strictly before that commit."""
    g = Git(tmp_path / "seen")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g.write(C, "# CORRECTION\n\nold\n")
    old = g.commit("an old correction, before any manifest")
    imported = [line("1", "prereg", P, seal, prereg(0.5)),
                line("1b", "amendment", C, old, "# CORRECTION\n\nold\n", amends="1", decision="inherits")]
    g.write(M, jsonl(*imported))
    added = g.commit("the manifest begins, importing the record")
    with G._Repo(g.root) as repo:
        assert G._manifest_seen(repo, G._manifest_at(repo, "main"), "main") == []
    # a later null on a new amendment, and a later null naming an old commit (not part of the import)
    g.write("p/CORRECTION_new.md", "# CORRECTION\n\nnew\n")
    new_c = g.commit("a new correction")
    later = line("1d", "amendment", "p/CORRECTION_new.md", new_c, "# CORRECTION\n\nnew\n", amends="1",
                 decision="inherits")
    late_old = dict(imported[1], id="1e")
    g.write(M, jsonl(*imported, later, late_old))
    g.commit("append two lines with seen null")
    with G._Repo(g.root) as repo:
        found = G._manifest_seen(repo, G._manifest_at(repo, "main"), "main")
        assert codes(found) == [("MANIFEST_SEEN_MISSING", "FAIL")] * 2
        assert [f["id"] for f in found] == ["1d", "1e"]
        assert added[:12] in found[0]["detail"]
        # the same lines with a level named are accepted wherever they sit
        named = [*imported, dict(later, seen="results seen"), dict(late_old, seen="nothing run")]
        assert G._manifest_seen(repo, named, "main") == []


def test_seen_null_in_the_import_is_refused_when_its_commit_is_not_before_the_manifest(tmp_path):
    """An imported line whose commit is no ancestor of the commit that added the manifest (here a side
    branch merged after it) is no record of the past."""
    g = Git(tmp_path / "side_import")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g("switch", "-q", "-c", "side")
    g.write(C, "# CORRECTION\n\nside\n")
    y = g.commit("a correction on a side branch")
    g("switch", "-q", "main")
    g.write(M, jsonl(line("1", "prereg", P, seal, prereg(0.5)),
                     line("1b", "amendment", C, y, "# CORRECTION\n\nside\n", amends="1", decision="inherits")))
    g.commit("a manifest importing a line from a branch it does not hold")
    g.merge("side", "merge the side branch")
    with G._Repo(g.root) as repo:
        es = G._manifest_at(repo, "main")
        assert G._manifest_blobs(repo, es, "main") == []
        found = G._manifest_seen(repo, es, "main")
        assert [(f["code"], f["id"]) for f in found] == [("MANIFEST_SEEN_MISSING", "1b")]
        assert "not strictly before" in found[0]["detail"]


@pytest.fixture()
def importing(tmp_path):
    """main: the plan, an old CORRECTION and receipt_0. A pull request commits a NEW amendment, then
    adds the manifest importing the old CORRECTION and listing the new amendment, both at seen null
    (the B2 review, case C)."""
    g = Git(tmp_path / "importing")
    g.write("README.md", "r\n")
    g.commit("m0")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g.write(C, "# CORRECTION\n\nold\n")
    old = g.commit("an old correction on main")
    g.write("p/receipt_0.json", receipt(m=0.45))
    g.commit("HOLD under the plan")
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr")
    body = prereg(0.4, "amendment written after receipt_0")
    g.write(A, body)
    new = g.commit("a new amendment, in the pull request")
    es = [line("1", "prereg", P, seal, prereg(0.5)),
          line("1b", "amendment", C, old, "# CORRECTION\n\nold\n", amends="1", decision="inherits"),
          line("1c", "amendment", A, new, body, amends="1")]
    g.write(M, jsonl(*es))
    added = g.commit("the manifest begins: the old record and the new amendment, seen null")
    return SimpleNamespace(g=g, base=base, old=old, new=new, added=added, es=es)


def test_a_pull_request_that_adds_the_manifest_imports_only_what_base_held(importing):
    """Case C: under the pull request's check (base given), the import point is the merge base, so the
    old CORRECTION's null stands and the new amendment's null is refused; the old rule (strictly before
    the add commit) let both through."""
    with G._Repo(importing.g.root) as repo:
        found = G._manifest_seen(repo, importing.es, "pr", base=importing.base)
        assert [(f["code"], f["kind"], f["id"]) for f in found] == [("MANIFEST_SEEN_MISSING", "FAIL", "1c")]
        assert importing.base[:12] in found[0]["detail"] and "merge base" in found[0]["detail"]
        named = [*importing.es[:2], dict(importing.es[2], seen="results seen")]
        assert G._manifest_seen(repo, named, "pr", base=importing.base) == []


def test_a_census_reads_the_import_point_from_the_merge_that_landed_the_manifest(importing):
    """After a merge commit lands the manifest, its parent one is main before the pull request: a census
    refuses the new amendment's null as the pull request's check did."""
    g = importing.g
    g("switch", "-q", "main")
    g.merge("pr", "land the manifest")
    with G._Repo(g.root) as repo:
        found = G._manifest_seen(repo, importing.es, "main")
        assert [(f["code"], f["id"]) for f in found] == [("MANIFEST_SEEN_MISSING", "1c")]
        assert "not strictly before" in found[0]["detail"]
        # a later pull request against this main reads the landing on main the same way
        g("switch", "-q", "-c", "pr2")
        g.write("README.md", "r2\n")
        g.commit("an unrelated change")
        found = G._manifest_seen(repo, importing.es, "pr2", base="main")
        assert [f["id"] for f in found] == ["1c"]


def test_a_census_along_a_fast_forward_cannot_tell_the_pull_requests_commits_from_mains(importing):
    """The stated boundary: landed by a fast-forward, the manifest's parent one is the new amendment
    itself, so a census accepts its null; the pull request's check (above) is where it is refused."""
    g = importing.g
    g("switch", "-q", "main")
    g("merge", "-q", "--ff-only", "pr")
    with G._Repo(g.root) as repo:
        assert G._manifest_seen(repo, importing.es, "main") == []
        assert [f["id"] for f in G._manifest_seen(repo, importing.es, "main", base=importing.base)] == ["1c"]
        assert G._import_point(repo, "main", M, None)[:2] == (importing.added, importing.new)


def test_a_named_seen_level_is_never_refused(amended):
    assert G._manifest_seen(amended.repo, amended.lines2, "main") == []


# ---- the decision in effect on a repository --------------------------------------------------------------

def test_the_decision_in_effect_follows_the_receipts_commit(amended):
    repo = amended.repo
    es = amended.lines2
    r0 = G._effect(repo, P, "p/receipt_0.json", amended.r0, "main", "main", es)
    assert (r0["effect"]["source"], r0["effect"]["gates_sha256"]) == ("1", digest(0.5))
    assert [(f["code"], f["id"]) for f in r0["findings"]] == [
        ("AMENDED_AFTER_RECEIPT", "1b"), ("AMENDED_AFTER_RECEIPT", "1c"), ("AMENDED_AFTER_RECEIPT", "1d")]
    assert r0["seal"]["manifest_id"] == "1" and r0["seal"]["commit"] == amended.seal
    assert core.rescore(r0["effect"]["bytes"], json.loads(receipt(m=0.55)))["verdict"] == "PASS"

    r1 = G._effect(repo, P, "p/receipt_1.json", amended.r1, "main", "main", es)
    assert (r1["effect"]["source"], r1["effect"]["gates_sha256"]) == ("1b", digest(0.6))
    assert r1["effect"]["sealed"] == [digest(0.5), digest(0.6)]
    assert [(a["id"], a["in_effect"]) for a in r1["effect"]["amendments"]] == [("1b", True), ("1c", True),
                                                                               ("1d", False)]
    assert [(f["code"], f["kind"], f["id"]) for f in r1["findings"]] == [
        ("AMENDED_AFTER_RECEIPT", "NOTE", "1d"), ("AMENDED_AFTER_DATA", "NOTE", "1b")]
    data = r1["findings"][1]["detail"]
    assert "p/receipt_0.json@" + amended.r0[:12] in data and "PASS" in data and "results seen" in data
    assert [(x["path"], x["verdict"], x["source"]) for x in r1["earlier"]] == [("p/receipt_0.json", "PASS", "1")]
    assert core.rescore(r1["effect"]["bytes"], json.loads(receipt(m=0.55)))["verdict"] == "HOLD"

    r2 = G._effect(repo, P, "p/receipt_2.json", amended.r2, "main", "main", es, amend_after_data="fail")
    assert (r2["effect"]["source"], r2["effect"]["gates_sha256"]) == ("1d", digest(0.7))
    assert [(f["code"], f["kind"]) for f in r2["findings"]] == [("AMENDED_AFTER_DATA", "FAIL")] * 2
    assert sorted((x["path"], x["verdict"], x["source"]) for x in r2["earlier"]) == [
        ("p/receipt_0.json", "PASS", "1"), ("p/receipt_1.json", "HOLD", "1b")]


def test_a_receipt_deleted_before_the_amendment_still_reads_amended_after_data(tmp_path):
    """The B2 review, case E: receipt_0 is committed under the plan and deleted inside the pull request,
    then an own amendment governs receipt_1. C4 says a receipt under the old decision "exists in
    history"; the two trees read before held none, so nothing printed. The history between the seal's
    order commit and the amendment is read too."""
    g = Git(tmp_path / "deleted")
    g.write("README.md", "r\n")
    g.commit("m0")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr")
    g.write("p/receipt_0.json", _recd(0.45, 0.5))
    r0 = g.commit("HOLD under the plan")
    g.write("q/other.json", receipt(name="PREREG_b.md", m=0.1))
    g.write("q/PREREG_b.md", prereg(0.5))
    g.commit("a receipt of another PREREG")
    g.write("p/receipt_0.json", _recd(0.46, 0.5))
    r0b = g.commit("receipt_0 rewritten in place before it is dropped")
    os.remove(g.root / "p" / "receipt_0.json")
    g.commit("drop it")
    g.write(A, prereg(0.4, "amendment"))
    a = g.commit("amendment")
    es = [line("1", "prereg", P, seal, prereg(0.5)),
          line("1b", "amendment", A, a, prereg(0.4, "amendment"), amends="1", seen="nothing run")]
    g.write(M, jsonl(*es))
    g.commit("seal it")
    g.write("p/receipt_1.json", _recd(0.45, 0.4))
    r1 = g.commit("PASS under the amendment")
    head = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        got = G._effect(repo, P, "p/receipt_1.json", r1, base, head, es)
        assert codes(got["findings"]) == [("AMENDED_AFTER_DATA", "NOTE")]
        d = got["findings"][0]["detail"]
        assert f"p/receipt_0.json@{r0[:12]} read HOLD under the sealed table (in history only" in d
        assert "seen: nothing run" in d
        # R is the oldest writer, read with the bytes it wrote
        assert [(x["path"], x["commit"], x["in_tree"]) for x in got["earlier"]] == [("p/receipt_0.json", r0, False)]
        assert G._effect(repo, P, "p/receipt_1.json", r1, base, head, es, amend_after_data="fail")["findings"][0][
            "kind"] == "FAIL"
        hist = G._receipts_in_history(repo, P, a, seal, [])
        assert [(x["path"], x["receipt_commit"], x["receipt"]["m"]) for x in hist] == [("p/receipt_0.json", r0, 0.45)]
        assert G._receipts_in_history(repo, P, a, seal, ["p/receipt_0.json"]) == []
        assert G._receipts_in_history(repo, P, a, seal, [], exclude=("p/",)) == []
        assert G._receipts_in_history(repo, P, r0, seal, []) == []          # strictly before the tip
        # the stop and its ancestors are left out: from r0 on, the oldest writer is the rewrite
        assert [(x["receipt_commit"], x["receipt"]["m"]) for x in G._receipts_in_history(repo, P, a, r0, [])] == [
            (r0b, 0.46)]
        assert G._receipts_in_history(repo, P, a, r0b, []) == []


def test_a_receipt_on_a_branch_the_amendment_does_not_hold_is_not_before_it(tmp_path):
    """The history read is the amendment's ancestry: a receipt committed and deleted on a branch that the
    amendment's commit does not reach is not "before" it."""
    g = Git(tmp_path / "side_receipt")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g("switch", "-q", "-c", "side")
    g.write("p/receipt_0.json", _recd(0.45, 0.5))
    g.commit("a receipt on a side branch")
    os.remove(g.root / "p" / "receipt_0.json")
    g.commit("dropped there")
    g("switch", "-q", "main")
    g.write(A, prereg(0.4, "amendment"))
    a = g.commit("amendment on main")
    g.merge("side", "merge the side branch after the amendment")
    es = [line("1", "prereg", P, seal, prereg(0.5)),
          line("1b", "amendment", A, a, prereg(0.4, "amendment"), amends="1", seen="nothing run")]
    g.write(M, jsonl(*es))
    g.commit("seal it")
    g.write("p/receipt_1.json", _recd(0.45, 0.4))
    r1 = g.commit("PASS")
    with G._Repo(g.root) as repo:
        got = G._effect(repo, P, "p/receipt_1.json", r1, "main", "main", es)
        assert got["effect"]["source"] == "1b" and got["findings"] == [] and got["earlier"] == []


def test_without_a_manifest_line_the_default_seal_governs_alone(amended):
    r = G._effect(amended.repo, P, "p/receipt_1.json", amended.r1, "main", "main", None)
    assert r["seal"]["manifest_id"] is None and r["seal"]["landing"] == amended.seal
    assert (r["effect"]["source"], r["effect"]["gates_sha256"], r["findings"]) == (None, digest(0.5), [])
    r = G._effect(amended.repo, P, "p/receipt_1.json", amended.r1, "main", "main", [])
    assert r["effect"]["source"] is None


def test_an_amendment_that_is_no_ancestor_of_the_receipt_is_not_in_effect_whatever_its_date(tmp_path):
    g = Git(tmp_path / "parallel")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g("switch", "-q", "-c", "amend")
    g.write(A, prereg(0.6, "amendment"))
    a = g.commit("amendment, dated before the receipt", at=10_000)
    g("switch", "-q", "main")
    g.write("p/receipt_1.json", receipt(m=0.55))
    r = g.commit("receipt on a line that does not hold the amendment", at=20_000)
    g.merge("amend", "merge the amendment after the receipt")
    es = [line("1", "prereg", P, seal, prereg(0.5)),
          line("1b", "amendment", A, a, prereg(0.6, "amendment"), amends="1", seen="results seen")]
    with G._Repo(g.root) as repo:
        assert G._manifest_blobs(repo, es, "main") == []
        got = G._effect(repo, P, "p/receipt_1.json", r, "main", "main", es)
        assert got["effect"]["source"] == "1"
        assert [(f["code"], f["id"]) for f in got["findings"]] == [("AMENDED_AFTER_RECEIPT", "1b")]
        assert "not strictly before" in got["findings"][0]["detail"]


def test_a_manifest_line_overrides_the_landing_rule_and_both_are_printed(tmp_path):
    """DESIGN 3.2(b): the manifest overrides the default per path, its commit and bytes; the landing
    commit is printed beside it. Bytes other than the landed ones around the landed decision are a
    NOTE (MANIFEST_SEAL_PROSE_DIFFERS); the landed bytes at another commit read nothing."""
    g = Git(tmp_path / "override")
    g.write(P, prereg(0.5))
    early = g.commit("plan")
    g.write(P, prereg(0.5, "prose moved, the fence kept"))
    later = g.commit("plan's prose edited before any receipt")
    g.write("p/receipt_1.json", receipt(m=0.55))
    r = g.commit("receipt")
    es = [line("1", "prereg", P, later, prereg(0.5, "prose moved, the fence kept"))]
    with G._Repo(g.root) as repo:
        got = G._effect(repo, P, "p/receipt_1.json", r, "main", "main", es)
        assert (got["seal"]["commit"], got["seal"]["landing"], got["seal"]["manifest_id"]) == (later, early, "1")
        assert got["seal"]["bytes"] == prereg(0.5, "prose moved, the fence kept").encode("utf-8")
        assert got["effect"]["gates_sha256"] == digest(0.5) and got["seal"]["refused"] is None
        assert got["seal"]["order"] == early                    # the decision was already held at L
        assert codes(got["findings"]) == [("MANIFEST_SEAL_PROSE_DIFFERS", "NOTE")]
        assert codes(G._manifest_landed(repo, es, "main", "main")) == [("MANIFEST_SEAL_PROSE_DIFFERS", "NOTE")]
        default = G._effect(repo, P, "p/receipt_1.json", r, "main", "main", None)
        assert (default["seal"]["commit"], default["effect"]["gates_sha256"]) == (early, digest(0.5))
    g.write("README.md", "r\n")
    other = g.commit("a commit that leaves the PREREG as it is")
    g.write(P, prereg(0.5))
    g.commit("the landed bytes again")
    es = [line("1", "prereg", P, early, prereg(0.5))]
    with G._Repo(g.root) as repo:
        assert G._manifest_landed(repo, es, "main", "main") == []
        assert G._manifest_landed(repo, [dict(es[0], commit=g("rev-parse", "HEAD"))], "main", "main") == []
        assert codes(G._manifest_landed(repo, [line("1", "prereg", P, other, prereg(0.5, "prose moved, the fence "
                                                                                      "kept"))], "main", "main")) \
            == [("MANIFEST_SEAL_PROSE_DIFFERS", "NOTE")]


def _recd(m, bar):
    """A receipt scored honestly under *bar*: its verdict, gates and digest as a scorer writes them."""
    return receipt(m=m, verdict="PASS" if m >= bar else "HOLD", gates={"G0": m >= bar}, gates_sha256=digest(bar))


def _scored(repo, rpath, rcommit, base, head, es):
    got = G._effect(repo, P, rpath, rcommit, base, head, es)
    rec = json.loads(repo.read(rcommit, rpath))
    s = core.rescore(got["effect"]["bytes"], rec)
    return got, s, core.compare(s, rec, sealed=got["effect"]["sealed"])


def test_a_prereg_line_cannot_reseal_a_landed_prereg_at_a_decision_that_never_landed(tmp_path):
    """The B2 review, case A: the plan (bar 0.5) lands and receipt_0 reads HOLD under it. A pull request
    moves the bar to 0.4 at c1, puts the PREREG back to its landed bytes (C0 reads nothing), adds a
    manifest prereg line naming c1, and adds receipt_1 scored under 0.4. The line overrides the default
    seal's decision, which only an amendment line (with seen) may change: MANIFEST_SEAL_NOT_LANDED, and
    the landed decision governs, so receipt_1 reads DECISION_DIGEST_UNSEALED and VERDICT_DIFFERS while
    the honest receipt_0 reads clean."""
    g = Git(tmp_path / "reseal")
    g.write("README.md", "r\n")
    g.commit("m0")
    g.write(P, prereg(0.5))
    seal = g.commit("the plan lands (bar 0.5)")
    g.write("p/receipt_0.json", _recd(0.45, 0.5))
    r0 = g.commit("run under the sealed table: HOLD")
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr")
    g.write(P, prereg(0.4))
    c1 = g.commit("move the bar to 0.4")
    g.write(P, prereg(0.5))
    g.commit("put the PREREG back to its landed bytes")
    es = [line("1", "prereg", P, c1, prereg(0.4))]
    g.write(M, jsonl(*es))
    g.commit("a manifest line sealing the PREREG at c1")
    g.write("p/receipt_1.json", _recd(0.45, 0.4))
    r1 = g.commit("re-run under 0.4: PASS")
    head = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        assert G._decisions_moved(repo, base, head) == []
        assert G._manifest_blobs(repo, es, head) == [] and G._manifest_append_only(repo, base, head) == []
        found = G._manifest_landed(repo, es, base, head)
        assert [(f["code"], f["kind"], f["id"]) for f in found] == [("MANIFEST_SEAL_NOT_LANDED", "FAIL", "1")]
        assert seal[:12] in found[0]["detail"] and digest(0.5) in found[0]["detail"]
        assert digest(0.4) in found[0]["detail"] and "amendment line" in found[0]["detail"]
        got, s, cmp_ = _scored(repo, "p/receipt_1.json", r1, base, head, es)
        assert codes(got["findings"]) == [("MANIFEST_SEAL_NOT_LANDED", "FAIL")]
        assert (got["seal"]["commit"], got["seal"]["manifest_id"], got["seal"]["refused"]["commit"]) == (seal, "1", c1)
        assert got["effect"]["gates_sha256"] == digest(0.5) and got["effect"]["sealed"] == [digest(0.5)]
        assert s["verdict"] == "HOLD"
        assert [f["code"] for f in cmp_] == ["DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS"]
        got0, s0, cmp0 = _scored(repo, "p/receipt_0.json", r0, base, head, es)
        assert (s0["verdict"], cmp0) == ("HOLD", [])
        assert codes(got0["findings"]) == [("MANIFEST_SEAL_NOT_LANDED", "FAIL")]
        # without the line the landing rule reads the same decision, and nothing is refused
        assert G._effect(repo, P, "p/receipt_1.json", r1, base, head, None)["effect"]["gates_sha256"] == digest(0.5)


def test_a_prereg_line_for_a_prereg_that_arrives_in_the_pull_request_holds_heads_decision(tmp_path):
    """Case A within one pull request: the PREREG is not on BASE, so the default seal is HEAD's bytes
    (C2); a line naming a commit whose fence HEAD no longer holds is refused the same way."""
    g = Git(tmp_path / "reseal_new")
    g.write("README.md", "r\n")
    base = g.commit("m0")
    g("switch", "-q", "-c", "pr")
    g.write(P, prereg(0.5))
    g.commit("the plan (bar 0.5)")
    g.write(P, prereg(0.4))
    c1 = g.commit("move the bar to 0.4")
    g.write(P, prereg(0.5))
    g.commit("put it back")
    es = [line("1", "prereg", P, c1, prereg(0.4))]
    g.write(M, jsonl(*es))
    g.commit("seal at c1")
    g.write("p/receipt_1.json", _recd(0.45, 0.4))
    r1 = g.commit("run under 0.4")
    head = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        found = G._manifest_landed(repo, es, base, head)
        assert codes(found) == [("MANIFEST_SEAL_NOT_LANDED", "FAIL")] and "has not landed" in found[0]["detail"]
        got, s, cmp_ = _scored(repo, "p/receipt_1.json", r1, base, head, es)
        assert got["effect"]["gates_sha256"] == digest(0.5) and s["verdict"] == "HOLD"
        assert [f["code"] for f in cmp_] == ["DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS"]
        ok = [line("1", "prereg", P, head, prereg(0.5))]
        assert G._manifest_landed(repo, ok, base, head) == []


def test_the_same_change_as_an_amendment_line_reads_amended_after_data(tmp_path):
    """The route case A's line was refused for: the bar moved after a receipt is an amendment line, which
    names its seen and prints AMENDED_AFTER_DATA beside the later receipt."""
    g = Git(tmp_path / "as_amendment")
    g.write("README.md", "r\n")
    g.commit("m0")
    g.write(P, prereg(0.5))
    seal = g.commit("plan")
    g.write("p/receipt_0.json", _recd(0.45, 0.5))
    g.commit("HOLD")
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr")
    g.write(A, prereg(0.4, "amendment"))
    a = g.commit("amendment")
    es = [line("1", "prereg", P, seal, prereg(0.5)),
          line("1b", "amendment", A, a, prereg(0.4, "amendment"), amends="1", seen="results seen")]
    g.write(M, jsonl(*es))
    g.commit("seal it")
    g.write("p/receipt_1.json", _recd(0.45, 0.4))
    r1 = g.commit("PASS")
    head = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        assert G._manifest_landed(repo, es, base, head) == []
        got, s, cmp_ = _scored(repo, "p/receipt_1.json", r1, base, head, es)
        assert codes(got["findings"]) == [("AMENDED_AFTER_DATA", "NOTE")] and (s["verdict"], cmp_) == ("PASS", [])
        assert "seen: results seen" in got["findings"][0]["detail"]


def test_a_manifest_line_whose_bytes_differ_refuses_the_reading(amended):
    es = [dict(amended.lines[0], blob_sha256="0" * 64)]
    with pytest.raises(G._Refusal) as e:
        G._effect(amended.repo, P, "p/receipt_1.json", amended.r1, "main", "main", es)
    assert e.value.code == "MANIFEST_BLOB_DIFFERS"


def test_the_receipts_of_a_prereg_are_read_from_the_tree(amended):
    got = G._receipts_of(amended.repo, P, ["main"])
    assert [(x["path"], x["receipt_commit"]) for x in got] == [
        ("p/receipt_0.json", amended.r0), ("p/receipt_1.json", amended.r1), ("p/receipt_2.json", amended.r2)]
    assert G._receipts_of(amended.repo, "q/PREREG_b.md", ["main"]) == []


# ---- the seal verb -----------------------------------------------------------------------------------

@pytest.fixture()
def demo(tmp_path, monkeypatch):
    """DESIGN 4.B's route: a plan committed on main, then an amendment committed beside it."""
    g = Git(tmp_path / "demo")
    g.write(P, prereg(0.5))
    seal = g.commit("seal the plan")
    g.write("p/receipt.json", receipt(m=0.55))
    result = g.commit("result")
    g.write(A, prereg(0.6, "amendment"))
    amend = g.commit("amendment")
    monkeypatch.chdir(g.root)
    return SimpleNamespace(g=g, seal=seal, result=result, amend=amend)


def _seal_cli(capsys, *argv):
    from styxx.prereg_ci import cli
    code = cli.main(["seal", *argv])
    out = capsys.readouterr().out
    lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    return code, out, lines


def test_seal_prints_the_line_and_writes_nothing_without_append(demo, capsys):
    code, out, lines = _seal_cli(capsys, P)
    assert code == 0
    assert lines == [line("1", "prereg", P, demo.seal, prereg(0.5))]
    assert f"blob sha256 {sha(prereg(0.5))}" in out and f"gates {digest(0.5)}" in out
    assert f"landing commit {demo.seal}" in out
    assert not (demo.g.root / M).exists()


def test_seal_append_writes_the_line_and_an_amendment_names_its_seen(demo, capsys):
    code, _, lines = _seal_cli(capsys, P, "--append")
    assert code == 0
    raw = (demo.g.root / M).read_bytes()
    assert raw == jsonl(line("1", "prereg", P, demo.seal, prereg(0.5)))
    assert core._manifest(raw) == lines
    code, _, lines = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen", "--append")
    assert code == 0
    assert lines == [line("1b", "amendment", A, demo.amend, prereg(0.6, "amendment"), amends="1",
                          seen="results seen")]
    raw = (demo.g.root / M).read_bytes()
    assert core._manifest(raw)[1] == lines[0] and b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    demo.g.write(C, "# CORRECTION\n\nprose only\n")
    corr = demo.g.commit("correction")
    code, _, lines = _seal_cli(capsys, C, "--amends", P, "--seen", "no results seen")
    assert code == 0 and lines == [line("1c", "amendment", C, corr, "# CORRECTION\n\nprose only\n", amends="1",
                                        decision="inherits", seen="no results seen")]


def test_seal_amends_a_path_with_no_line_by_appending_its_default_seal_line(demo, capsys):
    """PLAN K9; DESIGN 4.B: ``seal AMENDMENT_digits.md --amends PREREG_digits.md --seen "results seen"
    --append`` where the PREREG has no manifest line."""
    code, _, lines = _seal_cli(capsys, A, "--amends", P, "--seen", "results seen", "--append")
    assert code == 0
    want = [line("1", "prereg", P, demo.seal, prereg(0.5)),
            line("1b", "amendment", A, demo.amend, prereg(0.6, "amendment"), amends="1", seen="results seen")]
    assert lines == want and core._manifest((demo.g.root / M).read_bytes()) == want


def test_seal_reads_an_amendment_that_keeps_the_fence_in_effect_as_inheriting(demo, capsys):
    """SEALS row 8's shape: the PREREG itself, amended outside its fence, sealed as an amendment of its
    own line. Its fence is the one in effect, so the line inherits; a fence moved later reads own."""
    assert _seal_cli(capsys, P, "--append")[0] == 0
    demo.g.write(P, prereg(0.5, "v1 with a section appended"))
    prose = demo.g.commit("a section appended outside the fence")
    code, _, lines = _seal_cli(capsys, P, "--amends", "1", "--seen", "nothing run", "--commit", prose, "--append")
    assert code == 0 and lines == [line("1b", "amendment", P, prose, prereg(0.5, "v1 with a section appended"),
                                        amends="1", decision="inherits", seen="nothing run")]
    code, _, lines = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen")
    assert code == 0 and (lines[0]["id"], lines[0]["decision"]) == ("1c", "own")


def _withdrawn_manifest(demo):
    """A working-copy manifest whose newest line naming P withdraws a re-seal of it (SEALS row 8's
    shape): 1 seals P, 1b withdraws P at the result commit (the same bytes, its fence kept)."""
    es = [line("1", "prereg", P, demo.seal, prereg(0.5)),
          line("1b", "withdrawn", P, demo.result, prereg(0.5), amends="1", decision="inherits")]
    (demo.g.root / M).write_bytes(jsonl(*es))
    return es


def test_seal_amends_a_path_by_its_newest_line_that_is_not_withdrawn(demo, capsys):
    """The B2 review, case B: --amends PATH named the path's newest line, a withdrawn one, and refused;
    it names the newest line that is not withdrawn. --amends naming the withdrawn id is refused with the
    line it withdrew from."""
    _withdrawn_manifest(demo)
    demo.g.write(C, "# CORRECTION\n\nprose only\n")
    corr = demo.g.commit("correction")
    code, out, lines = _seal_cli(capsys, C, "--amends", P, "--seen", "results seen")
    assert code == 0 and lines == [line("1c", "amendment", C, corr, "# CORRECTION\n\nprose only\n", amends="1",
                                        decision="inherits", seen="results seen")]
    assert "note:" not in out
    code, out, lines = _seal_cli(capsys, C, "--amends", "1b", "--seen", "results seen")
    assert code == 2 and lines == [] and "withdrawn" in out and "--amends 1" in out


def test_seal_refuses_a_path_whose_every_line_is_withdrawn(demo, capsys):
    es = [line("1", "prereg", P, demo.seal, prereg(0.5)),
          line("1b", "withdrawn", A, demo.amend, prereg(0.6, "amendment"), amends="1", decision="own")]
    (demo.g.root / M).write_bytes(jsonl(*es))
    code, out, lines = _seal_cli(capsys, C, "--amends", A, "--seen", "results seen")
    assert code == 2 and lines == [] and "every line naming" in out and "by its id" in out


def test_seal_notes_a_line_that_records_bytes_another_line_records(demo, capsys):
    """Case B's second reading: --amends 7 wrote 7c for the CORRECTION 7b already seals, without a word."""
    assert _seal_cli(capsys, P, "--append")[0] == 0
    assert _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen", "--append")[0] == 0
    code, out, lines = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen")
    assert code == 0 and lines[0]["id"] == "1c"
    notes = [x for x in out.splitlines() if x.startswith("  note:")]
    assert notes == [f"  note: line 1b (amendment) already records {A} with blob sha256 "
                     f"{sha(prereg(0.6, 'amendment'))}; line 1c records the same bytes again"]


def test_seal_writes_and_prints_seen_on_a_prereg_line(demo, capsys):
    """The B2 review's note: a prereg line may carry seen (what had been seen when it was sealed); the
    verb prints it beside the line, and the seal the check reads carries it."""
    code, out, lines = _seal_cli(capsys, P, "--seen", "nothing run", "--append")
    assert code == 0 and lines == [line("1", "prereg", P, demo.seal, prereg(0.5), seen="nothing run")]
    assert f"seal {P} (line 1, prereg, seen: nothing run)" in out
    code, out, _ = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen")
    assert f"seal {A} (line 1b, amendment, amends 1, own, seen: results seen)" in out
    with G._Repo(demo.g.root) as repo:
        es = core._manifest((demo.g.root / M).read_bytes())
        got = G._effect(repo, P, "p/receipt.json", demo.result, "HEAD", "HEAD", es)
        assert got["seal"]["seen"] == "nothing run"
        assert G._effect(repo, P, "p/receipt.json", demo.result, "HEAD", "HEAD", None)["seal"]["seen"] is None


def test_seal_notes_a_prereg_line_whose_prose_is_not_the_landed_prose(demo, capsys):
    demo.g.write(P, prereg(0.5, "v2, the fence kept"))
    prose = demo.g.commit("prose")
    code, out, lines = _seal_cli(capsys, P, "--commit", prose)
    assert code == 0 and lines[0]["commit"] == prose
    assert [x for x in out.splitlines() if x.startswith("  note: MANIFEST_SEAL_PROSE_DIFFERS")]


def test_seal_at_a_named_commit(demo, capsys):
    demo.g.write(P, prereg(0.5, "v2"))
    demo.g.commit("prose")
    code, _, lines = _seal_cli(capsys, P, "--commit", demo.seal[:10])
    assert code == 0 and lines[0]["commit"] == demo.seal and lines[0]["blob_sha256"] == sha(prereg(0.5))


def test_seal_commit_takes_a_commit_or_a_tag_of_one_and_refuses_a_tree_as_input(demo, capsys):
    """Case D at the verb: --commit naming a tree is an input error (exit 2), not UNCHECKABLE (exit 3,
    which DESIGN 3.3 keeps for a shallow clone or a missing commit); an annotated tag's name is peeled,
    and the line records the commit's id."""
    tree = demo.g("rev-parse", demo.seal + "^{tree}")
    code, out, lines = _seal_cli(capsys, P, "--commit", tree)
    assert code == 2 and lines == [] and "NOT_A_COMMIT" in out and "tree" in out
    code, out, _ = _seal_cli(capsys, P, "--commit", "9" * 40)
    assert code == 3 and "COMMIT_ABSENT" in out
    demo.g("tag", "-a", "-m", "t", "v1", demo.seal)
    code, _, lines = _seal_cli(capsys, P, "--commit", "v1")
    assert code == 0 and lines[0]["commit"] == demo.seal


def test_seal_refuses_a_prereg_line_at_a_decision_other_than_the_landed_one(demo, capsys):
    """Case A at the verb: a prereg line at bytes holding another fence than the one the PREREG landed
    with is refused, naming the amendment route, which then writes the line."""
    demo.g.write(P, prereg(0.4, "the bar moved after the result"))
    moved = demo.g.commit("move the bar in place")
    code, out, lines = _seal_cli(capsys, P, "--commit", moved)
    assert code == 2 and lines == [] and "MANIFEST_SEAL_NOT_LANDED" in out and "--amends" in out
    assert demo.seal[:12] in out and digest(0.4) in out
    assert _seal_cli(capsys, P, "--append")[0] == 0
    code, _, lines = _seal_cli(capsys, P, "--amends", "1", "--seen", "results seen", "--commit", moved)
    assert code == 0 and lines == [line("1b", "amendment", P, moved, prereg(0.4, "the bar moved after the result"),
                                        amends="1", seen="results seen")]


@pytest.mark.parametrize("argv, why", [
    ((A, "--amends", "1"), "seen"),
    ((A, "--amends", "9", "--seen", "results seen"), "9"),
    (("p/never.md",), "never.md"),
    ((P, "--commit", "no-such-rev"), "no-such-rev"),
])
def test_seal_refuses_input_it_cannot_seal(demo, capsys, argv, why):
    if argv[1:2] == ("--amends",) and argv[2] == "1":
        assert _seal_cli(capsys, P, "--append")[0] == 0
    code, out, lines = _seal_cli(capsys, *argv)
    assert code in (2, 3) and lines == [] and why in out
    assert "REFUSED" in out


def test_seal_refuses_an_uncommitted_file_and_a_second_seal_of_one_path(demo, capsys):
    demo.g.write("p/PREREG_new.md", prereg(0.5))
    code, out, _ = _seal_cli(capsys, "p/PREREG_new.md")
    assert code == 2 and "commit" in out
    assert _seal_cli(capsys, P, "--append")[0] == 0
    before = (demo.g.root / M).read_bytes()
    code, out, _ = _seal_cli(capsys, P, "--append")
    assert code == 2 and "already" in out
    assert (demo.g.root / M).read_bytes() == before


def test_seal_refuses_an_unknown_seen_level(demo, capsys):
    from styxx.prereg_ci import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["seal", A, "--amends", P, "--seen", "some results seen"])
    assert e.value.code == 2


def test_seal_refuses_to_append_to_a_working_copy_that_is_not_heads_manifest_with_lines_appended(demo, capsys):
    assert _seal_cli(capsys, P, "--append")[0] == 0
    demo.g("add", M)
    demo.g("commit", "-q", "-m", "manifest")
    edited = jsonl(line("1", "prereg", P, demo.seal, prereg(0.5), seen="nothing run"))
    (demo.g.root / M).write_bytes(edited)
    code, out, _ = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen", "--append")
    assert code == 2 and "append" in out
    assert (demo.g.root / M).read_bytes() == edited


def test_seal_reads_a_crlf_working_copy_of_the_manifest_and_appends_to_it(demo, capsys):
    """A checkout under core.autocrlf=true holds the manifest CRLF; its blob is LF."""
    assert _seal_cli(capsys, P, "--append")[0] == 0
    demo.g("add", M)
    demo.g("commit", "-q", "-m", "manifest")
    crlf = (demo.g.root / M).read_bytes().replace(b"\n", b"\r\n")
    (demo.g.root / M).write_bytes(crlf)
    code, _, lines = _seal_cli(capsys, A, "--amends", "1", "--seen", "results seen", "--append")
    assert code == 0
    raw = (demo.g.root / M).read_bytes()
    assert raw.startswith(crlf) and core._manifest(raw.replace(b"\r\n", b"\n"))[1] == lines[0]


def test_the_route_of_design_4b_reads_amended_after_data(demo, capsys):
    """DESIGN 4.B: seal the amendment, commit the manifest, run under the amendment to a new path;
    the check reads PASS under the amended table, and beside it AMENDED_AFTER_DATA naming receipt.json
    at the "result" commit and the amendment's seen."""
    assert _seal_cli(capsys, A, "--amends", P, "--seen", "results seen", "--append")[0] == 0
    demo.g("add", M)
    demo.g("commit", "-q", "-m", "seal the amendment")
    demo.g.write("p/receipt_amended.json", receipt(m=0.55))
    r = demo.g.commit("re-run under the amendment")
    with G._Repo(demo.g.root) as repo:
        es = G._manifest_at(repo, "HEAD")
        assert G._manifest_blobs(repo, es, "HEAD") == [] and G._manifest_seen(repo, es, "HEAD") == []
        got = G._effect(repo, P, "p/receipt_amended.json", r, "HEAD", "HEAD", es)
    assert got["effect"]["source"] == "1b"
    assert [(f["code"], f["kind"]) for f in got["findings"]] == [("AMENDED_AFTER_DATA", "NOTE")]
    d = got["findings"][0]["detail"]
    assert f"p/receipt.json@{demo.result[:12]}" in d and "PASS" in d and "seen: results seen" in d


def test_the_module_runs_as_a_command(demo):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-m", "styxx.prereg_ci", "seal", P], cwd=str(demo.g.root), env=env,
                       capture_output=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr
    assert json.loads([x for x in r.stdout.splitlines() if x.startswith("{")][0])["commit"] == demo.seal
    r = subprocess.run([sys.executable, "-B", "-m", "styxx.prereg_ci", "no-such-verb"], cwd=str(demo.g.root),
                       env=env, capture_output=True, encoding="utf-8", timeout=300)
    assert r.returncode == 2


# ---- R10: the SEALS rows, through the generator ----------------------------------------------------------

SEALS_ROWS = [
    ("2", "prereg", "papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md",
     "4dba3a72601a6f2d0d1cf62a89a6a172b1797b46", "507ef9aceb297ba06511d9e96c79ab9e1d6eca867d2ab73390e884c455c4a600", None),
    ("3", "prereg", "papers/checksum/PREREG_checksum_deploy_quant_2026_09_13.md",
     "04f1dac2c17a175d4b586b8432a0602a50ac9ff4", "b3b9871090fdcd4491cc4a3171c27e1468414409da9d3b66b4d3ff2fd4b6dd9e", None),
    ("3b", "amendment", "papers/checksum/CORRECTION_prereg_deploy_quant_H1_2026_09_13.md",
     "f3fa8842d85d7c9c6ff9fca64440a76205da0c51", "bb371c4467e45aa0ea3792a357551ba9599488d2c99e591cdb3190ec08ca9d6a", "3"),
    ("6", "prereg", "papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md",
     "b8205b1cd2087b8cfa55267c967f7fa0c40defaf", "d6a98261f44f31664cdedbc24769532d5a701bf46d2c8b9ce30cf6a227fb5519", None),
    ("6b", "amendment", "papers/checksum/CORRECTION_prereg_beacon_draw_2026_09_14.md",
     "6d03bf16b70b6dd90828662740b80fb764d836c4", "06dc24a820a50f57bb041f2fbdc3023dde1ae4c1ac6d0a4e75fedef7505b8d4a", "6"),
    ("7", "prereg", "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md",
     "3598e59d3fe708ebf2dd20992f253d02d753c88b", "2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e", None),
    ("7b", "amendment", "papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md",
     "d90da6937a81664bb945b0ae2a2eca0b3e5dbfaa", "e591516f49f84595aa3510b66cdab516dc250b903043899a0a564fdda013f0c1", "7"),
    ("8", "withdrawn", "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md",
     "98072bed64ddd9cda61ecae3ece641fae093bce3", "18aa265c248fff55f49e71286f7e8baa2c54a9fc0e81dde3f397b83656776a72", "7"),
]
# DESIGN 3.2(b)'s three example lines, as the design prints them with its line wraps removed
DESIGN_LINES = [
    '{"id": "7", "kind": "prereg", "path": "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md", '
    '"commit": "3598e59d3fe708ebf2dd20992f253d02d753c88b", '
    '"blob_sha256": "2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e", '
    '"amends": null, "decision": "own", "seen": null, "time": []}',
    '{"id": "7b", "kind": "amendment", "path": "papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md", '
    '"commit": "d90da6937a81664bb945b0ae2a2eca0b3e5dbfaa", "blob_sha256": "e591516f49f84595aa3510b66cdab516dc250b903043899a0a564fdda013f0c1", '
    '"amends": "7", "decision": "inherits", "seen": null, "time": []}',
    '{"id": "8", "kind": "withdrawn", "path": "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md", '
    '"commit": "98072bed64ddd9cda61ecae3ece641fae093bce3", "blob_sha256": "18aa265c248fff55f49e71286f7e8baa2c54a9fc0e81dde3f397b83656776a72", '
    '"amends": "7", "decision": "inherits", "seen": null, "time": []}',
]


def _have_rev() -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True)
    return r.returncode == 0


@pytest.fixture(scope="module")
def styxx_repo():
    if not _have_rev():
        pytest.skip(f"{REV} is not in the object store (a shallow or partial clone); R10 is read from it")
    with G._Repo(ROOT) as r:
        if r.is_shallow():
            pytest.skip("a shallow clone holds no history to read R10 from")
        yield r


@pytest.fixture(scope="module")
def generated(styxx_repo):
    from conformance.prereg_ci import gen_seals_manifest as GEN
    return GEN.build(styxx_repo, REV)


def test_the_generator_reads_the_eight_rows_at_lines_33_to_40(generated):
    assert [r["line"] for r in generated["rows"]] == list(range(33, 41))
    assert [r["id"] for r in generated["rows"]] == [r[0] for r in SEALS_ROWS]


def test_the_generator_writes_eight_lines_as_the_seals_record_states_them(generated):
    entries = core._manifest(generated["bytes"])
    assert [(e["id"], e["kind"], e["path"], e["commit"], e["blob_sha256"], e["amends"]) for e in entries] == SEALS_ROWS
    assert [e["decision"] for e in entries] == ["own", "own", "inherits", "own", "inherits", "own", "inherits",
                                                "inherits"]
    assert all(e["seen"] is None and e["time"] == [] for e in entries)
    assert generated["bytes"].decode("utf-8").splitlines()[5:8] == DESIGN_LINES


def test_every_generated_blob_sha256_is_git_show_piped_to_sha256sum(generated, styxx_repo):
    """R10, the SEALS rule literally: ``git show <commit>:<path> | sha256sum``."""
    for e in core._manifest(generated["bytes"]):
        shown = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "show", f"{e['commit']}:{e['path']}"],
                               capture_output=True, check=True).stdout
        assert hashlib.sha256(shown).hexdigest() == e["blob_sha256"], e["id"]
    assert G._manifest_blobs(styxx_repo, core._manifest(generated["bytes"]), REV) == []
    assert G._manifest_blobs(styxx_repo, core._manifest(generated["bytes"]), "HEAD") == []


def test_the_committed_copy_round_trips(generated):
    raw = COPY.read_bytes()
    assert raw == generated["bytes"]
    assert core._manifest_bytes(core._manifest(raw)) == raw
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"\n")
    assert len(raw.splitlines()) == 8


def test_the_seals_record_reads_as_amendments_in_effect(generated, styxx_repo):
    """Row 7's lineage: 7b inherits row 7's table (a CORRECTION outside the fence) and row 8 is never in
    effect; the #224 receipt is read under row 7's fence (gates 578924f8...), with 7b in effect."""
    es = core._manifest(generated["bytes"])
    got = G._effect(styxx_repo, "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md",
                    "papers/frequency-resonance/pmnist_robustness_result.json", styxx_repo.commit("237e7784"),
                    REV, REV, es)
    assert got["effect"]["source"] == "7"
    assert got["effect"]["gates_sha256"].startswith("578924f8e776573c")
    assert [(a["id"], a["kind"], a["in_effect"]) for a in got["effect"]["amendments"]] == [
        ("7b", "amendment", True), ("8", "withdrawn", False)]
    assert got["findings"] == []


def test_every_prereg_line_of_the_seals_record_holds_the_landed_bytes(generated, styxx_repo):
    """Case A against the lab's own record: the four prereg lines (2, 3, 6, 7) each hold the bytes
    the landing rule seals, at REV and at HEAD, so none reads MANIFEST_SEAL_NOT_LANDED or
    MANIFEST_SEAL_PROSE_DIFFERS."""
    es = core._manifest(generated["bytes"])
    assert [e["id"] for e in es if e["kind"] == "prereg"] == ["2", "3", "6", "7"]
    assert G._manifest_landed(styxx_repo, es, REV, REV) == []
    assert G._manifest_landed(styxx_repo, es, "HEAD", "HEAD") == []


def test_seal_amends_the_labs_robustness_prereg_by_its_path(styxx_repo, monkeypatch, capsys):
    """Case B on the lab's own record, read-only (no --append): the robustness PREREG's newest line is
    row 8, withdrawn, and --amends by its path names row 7; the CORRECTION's bytes are row 7b's, which
    the note names."""
    monkeypatch.chdir(ROOT)
    man = "conformance/prereg_ci/seals_manifest_from_SEALS_2026_09_13.jsonl"
    corr = "papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md"
    pre = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
    before = COPY.read_bytes()
    code, out, lines = _seal_cli(capsys, corr, "--amends", pre, "--seen", "results seen", "--manifest", man)
    assert code == 0 and [(x["id"], x["amends"], x["decision"]) for x in lines] == [("7c", "7", "inherits")]
    assert lines[0]["blob_sha256"] == SEALS_ROWS[6][4]
    assert f"  note: line 7b (amendment) already records {corr} with blob sha256 {SEALS_ROWS[6][4]}" in out
    code, out, _ = _seal_cli(capsys, corr, "--amends", "8", "--seen", "results seen", "--manifest", man)
    assert code == 2 and "--amends 7" in out
    assert COPY.read_bytes() == before


def test_the_generator_check_regenerates_the_committed_bytes():
    if not _have_rev():
        pytest.skip(f"{REV} is not in the object store")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "conformance/prereg_ci/gen_seals_manifest.py", "--check"],
                       cwd=str(ROOT), env=env, capture_output=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("SAME ") and "rows 33-40" in r.stdout.splitlines()[0]
    assert [x for x in r.stdout.splitlines() if x.startswith(("MATCH ", "DIFFERS "))] == R10_LINES


# the design's R10 output (census/receipts_ca347730.txt, section "== R10 SEALS rows against git show"), as printed
R10_LINES = [
    "MATCH 4dba3a72 papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md",
    "MATCH 04f1dac2 papers/checksum/PREREG_checksum_deploy_quant_2026_09_13.md",
    "MATCH f3fa8842 papers/checksum/CORRECTION_prereg_deploy_quant_H1_2026_09_13.md",
    "MATCH b8205b1c papers/checksum/PREREG_checksum_beacon_draw_2026_09_14.md",
    "MATCH 6d03bf16 papers/checksum/CORRECTION_prereg_beacon_draw_2026_09_14.md",
    "MATCH 3598e59d papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md",
    "MATCH d90da693 papers/frequency-resonance/CORRECTION_prereg_pmnist_robustness_2026_10_12.md",
    "MATCH 98072bed papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md",
]


def test_the_generator_refuses_a_row_whose_digest_does_not_reproduce(styxx_repo, monkeypatch):
    """A SEALS row whose digest is not the blob's is DIFFERS, and nothing is built."""
    from conformance.prereg_ci import gen_seals_manifest as GEN
    real = styxx_repo.read

    def read(rev, path):
        data = real(rev, path)
        if path == SEALS:
            return data.replace(b"`507ef9aceb297ba0", b"`007ef9aceb297ba0")
        return data

    monkeypatch.setattr(styxx_repo, "read", read)
    with pytest.raises(GEN.Refused) as e:
        GEN.build(styxx_repo, REV)
    assert str(e.value).startswith("DIFFERS 4dba3a72 ")

# -*- coding: utf-8 -*-
"""An embedded file named `.git` is refused, in create and in verify.

Verify writes the embedded files under their names into a temporary folder and re-runs the
certifier there; the certifier's receipt binding starts git in that folder; and git reads a file
named `.git`, in the folder it is started in, as the address of a repository. A capsule whose
document was named `.git` therefore chose the repository those git commands read, and verified
with nothing said. Found by a review on 2026-11-12 and reproduced with a directory on the same
machine; the same in 7.50.0 and 8.0.0. Whether a path on another host makes git contact that host
was not tested, here or anywhere.

What these tests pin: the name is refused before anything is written, in any letter case and
with the code points HFS+ drops from a name; verify then starts no git at all; create refuses
before it reads a file; names that only look like it are left alone.

Not closed here: the re-run still starts git for every other capsule (in a folder that holds no
`.git`). Taking git out of the re-run is its own change.
"""
from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path

import pytest

from styxx.capsule import (_BEGIN, _END, _bare_name, _is_dot_git, _render_html, _unsafe_names,
                           create_capsule, verify_capsule)

# the modules by their names: the package also exports a function named `certify`
certify = importlib.import_module("styxx.certify")
receipt_binding = importlib.import_module("styxx.receipt_binding")
charon = importlib.import_module("styxx.charon")

GIT_NAMES = [".git", ".GIT", ".Git", ".gIt", ".g\u200cit", "\ufeff.git", ".gi\u202et", ".G\u200dIT",
             ".git\u200e"]
NOT_GIT = ["git", ".gitignore", ".gitattributes", "a.git", ".github", ".git.md", "x.git.json", ".gi",
           "dot.git", ".gitx"]


@pytest.mark.parametrize("name", GIT_NAMES)
def test_a_name_git_takes_for_its_own_is_refused(name):
    assert _is_dot_git(name) and not _bare_name(name)
    for payload in ({"document": {"name": name}, "receipts": []},
                    {"document": {"name": "d.md"}, "receipts": [{"name": name}]}):
        problems = _unsafe_names(payload)
        assert len(problems) == 1, problems
        assert "is `.git` to git" in problems[0] and "nothing was written" in problems[0]


@pytest.mark.parametrize("name", NOT_GIT)
def test_a_name_that_only_looks_like_it_is_left_alone(name):
    assert not _is_dot_git(name) and _bare_name(name)
    assert _unsafe_names({"document": {"name": name}, "receipts": []}) == []


def test_a_name_that_is_not_a_string_is_not_dot_git():
    for value in (None, 7, [".git"], {"name": ".git"}):
        assert _is_dot_git(value) is False


def _capsule_with_its_document_named(tmp_path: Path, name: str) -> tuple:
    """An intact capsule of a one-line document, and the same capsule with that document named
    `name`: the hashes and the certificate still hold, as a minter that named its file so
    would have them."""
    doc = tmp_path / "d.md"
    doc.write_text("gitdir: " + (tmp_path / "named_by_the_capsule").as_posix() + "\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text("{}", encoding="utf-8")
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(certify.certify_doc(doc, [rp])), encoding="utf-8")
    honest = create_capsule(doc, [rp], cp, tmp_path / "d.capsule.html")
    html = honest.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    payload = json.loads(html[i:html.index(_END, i)])
    payload["document"]["name"] = name
    payload["certificate"]["document"] = name
    renamed = tmp_path / "renamed.capsule.html"
    renamed.write_text(_render_html(payload), encoding="utf-8")
    return honest, renamed, payload


@pytest.mark.parametrize("name", [".git", ".GIT", ".g\u200cit"])
def test_verify_writes_nothing_and_starts_no_git_for_such_a_capsule(tmp_path, monkeypatch, name):
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH: the receipt binding starts none, so there is nothing to see")
    started = []
    real = receipt_binding.subprocess.run

    def run(args, *a, **k):
        started.append(([str(x) for x in args[1:3]], str(k.get("cwd"))))
        return real(args, *a, **k)

    monkeypatch.setattr(receipt_binding.subprocess, "run", run)
    honest, renamed, _ = _capsule_with_its_document_named(tmp_path, name)
    started.clear()
    assert verify_capsule(honest)["ok"]
    assert started, "the recorder sees the git that verify starts for an ordinary capsule"
    started.clear()
    rep = verify_capsule(renamed)
    assert not rep["ok"]
    assert len(rep["problems"]) == 1 and "is `.git` to git" in rep["problems"][0]
    assert started == [], "git was started for a capsule whose document is named .git"


def test_the_ferry_log_does_not_re_run_such_a_capsule(tmp_path):
    _, _, payload = _capsule_with_its_document_named(tmp_path, ".git")
    verdict, detail = charon._capsule_live_v01(payload)
    assert verdict is None and detail == {"live_error": "unsafe_embedded_name"}


@pytest.mark.parametrize("name", [".git", ".GIT", ".g\u200cit"])
def test_create_refuses_the_name_before_it_reads_a_file(tmp_path, name):
    missing = tmp_path / "not_there"
    with pytest.raises(SystemExit) as e:
        create_capsule(missing / name, [missing / "r.json"], missing / "c.json", tmp_path / "out.html")
    assert "REFUSED" in str(e.value) and "is `.git` to git" in str(e.value)
    with pytest.raises(SystemExit) as e:
        create_capsule(missing / "d.md", [missing / name], missing / "c.json", tmp_path / "out.html")
    assert "is `.git` to git" in str(e.value)
    assert not (tmp_path / "out.html").exists()

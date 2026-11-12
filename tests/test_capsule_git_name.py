# -*- coding: utf-8 -*-
"""A re-run of a capsule starts no git, and an embedded file named `.git` is refused.

Verify writes the embedded files under their names into a temporary folder and re-runs the
certifier there; the certifier's receipt binding starts git in that folder; and git reads a file
named `.git`, in the folder it is started in, as the address of a repository. A capsule whose
document was named `.git` therefore chose the repository those git commands read, and verified
with nothing said. Found by a review on 2026-11-12 and reproduced with a directory on the same
machine; the same in 7.50.0 and 8.0.0. Whether a path on another host makes git contact that host
was not tested, here or anywhere.

Two guards, and these tests pin both. The re-run starts no git at all, for any capsule:
`styxx.receipt_binding.no_repository` switches the repository lookup off for the thread that
re-runs, and a mint in another thread still sees its repository. And the name is refused before
anything is written, in any letter case and with the code points HFS+ drops from a name; create
refuses it before it reads a file; names that only look like it are left alone.

Not closed here: the embedded files are still written to a temporary folder under the names the
capsule gives. Holding them in memory is its own change. Not pinned here: the corpus audit's
re-run is inside the same block, and no test of this file reaches it.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from styxx.capsule import (_BEGIN, _END, _bare_name, _is_dot_git, _render_html, _unsafe_names,
                           create_capsule, verify_capsule)

# the modules by their names: the package also exports a function named `certify`
certify = importlib.import_module("styxx.certify")
receipt_binding = importlib.import_module("styxx.receipt_binding")
charon = importlib.import_module("styxx.charon")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "capsule_v01_minted_by_styxx_7_50_0.html"
needs_git = pytest.mark.skipif(shutil.which("git") is None,
                               reason="git is not on PATH: the receipt binding starts none, so there is nothing to see")


@pytest.fixture
def started(monkeypatch):
    """Every subprocess the receipt binding starts: its two leading arguments, its folder and the
    thread that started it. The real call still runs."""
    seen = []
    real = receipt_binding.subprocess.run

    def run(args, *a, **k):
        seen.append(([str(x) for x in args[1:3]], str(k.get("cwd")), threading.get_ident()))
        return real(args, *a, **k)

    monkeypatch.setattr(receipt_binding.subprocess, "run", run)
    return seen


# every code point HFS+ drops when it compares names (git's own list for this name), one name each
HFS_DROPPED = [0x200C, 0x200D, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x206A, 0x2070), 0xFEFF]
GIT_NAMES = ([".git", ".GIT", ".Git", ".gIt", "\ufeff.git", ".G\u200dIT", ".git\u200e"]
             + [".g" + chr(cp) + "it" for cp in HFS_DROPPED])
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


@needs_git
@pytest.mark.parametrize("name", [".git", ".GIT", ".g\u200cit"])
def test_verify_starts_no_git_for_an_ordinary_capsule_or_for_such_a_one(tmp_path, started, name):
    honest, renamed, _ = _capsule_with_its_document_named(tmp_path, name)
    assert started, "the recorder sees the git that a mint starts"
    started.clear()
    assert verify_capsule(honest)["ok"]
    assert started == [], "git was started while an ordinary capsule was re-run"
    rep = verify_capsule(renamed)
    assert not rep["ok"]
    assert len(rep["problems"]) == 1 and "is `.git` to git" in rep["problems"][0]
    assert started == [], "git was started for a capsule whose document is named .git"


@needs_git
def test_verify_starts_no_git_for_a_capsule_of_an_older_release(started):
    """The older certificate is re-run twice (under the installed reader and under the earlier
    one); neither run starts git."""
    assert verify_capsule(FIXTURE)["ok"]
    assert started == []


@needs_git
def test_a_capsule_minted_inside_a_repository_verifies_with_no_git_started(tmp_path, started):
    """The minter's repository facts are stated in the certificate; verify prints them as stated
    and compares the digests, which need no repository."""
    def git(*args):
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    doc = tmp_path / "d.md"
    doc.write_text("The run scored 0.75 accuracy over 40 items.\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"eval": {"accuracy": 0.75, "items": 40}}), encoding="utf-8")
    git("init", "-q")
    git("add", "r.json")
    git("commit", "-q", "-m", "one receipt")
    cert = certify.certify_doc(doc, [rp])
    binding = cert["receipt_binding"]
    assert binding["head"] and binding["all_receipts_committed"] is True, "the case is not the one meant"
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    assert any(a[:1] == ["rev-parse"] for a, _, _ in started), "the mint asked git, as it should"
    started.clear()
    capsule = create_capsule(doc, [rp], cp, tmp_path / "d.capsule.html")
    assert started == [], "create's own re-run of the certifier started git in its temporary folder"
    rep = verify_capsule(capsule)
    assert rep["ok"], rep["problems"]
    assert started == []
    assert any(binding["head"] in s_ for s_ in rep["stated"]), "the head is printed as the minter stated it"


@needs_git
def test_the_ferry_log_re_runs_an_ordinary_capsule_with_no_git(tmp_path, started):
    honest, _, _ = _capsule_with_its_document_named(tmp_path, ".git")
    html = honest.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    started.clear()
    verdict, counts = charon._capsule_live_v01(json.loads(html[i:html.index(_END, i)]))
    assert isinstance(verdict, str) and isinstance(counts, dict)
    assert started == []


def test_the_switch_is_for_one_thread_nests_and_ends_with_its_block(tmp_path):
    rb = receipt_binding
    assert getattr(rb._NO_REPOSITORY, "reason", None) is None
    seen = {}

    def other_thread():
        seen["reason"] = getattr(rb._NO_REPOSITORY, "reason", None)
        try:
            rb.Repo(tmp_path).close()
            seen["error"] = ""
        except rb.RepoUnavailable as e:
            seen["error"] = str(e)

    with rb.no_repository("outer block"):
        with pytest.raises(rb.RepoUnavailable, match="outer block"):
            rb.Repo(tmp_path)
        with rb.no_repository("inner block"):
            with pytest.raises(rb.RepoUnavailable, match="inner block"):
                rb.Repo(tmp_path)
        with pytest.raises(rb.RepoUnavailable, match="outer block"):
            rb.Repo(tmp_path)
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert seen["reason"] is None and "outer block" not in seen["error"], "another thread was switched off too"
    assert getattr(rb._NO_REPOSITORY, "reason", None) is None
    with pytest.raises(ValueError):
        with rb.no_repository("left by an exception"):
            raise ValueError("inside")
    assert getattr(rb._NO_REPOSITORY, "reason", None) is None
    # a reason that is empty would switch nothing off, and inside another block it would switch
    # the lookup back on: it is refused, and the outer block stays as it was
    with rb.no_repository("outer block"):
        for empty in ("", None, 0):
            with pytest.raises(ValueError, match="needs a reason"):
                with rb.no_repository(empty):
                    pass
        with pytest.raises(rb.RepoUnavailable, match="outer block"):
            rb.Repo(tmp_path)


@needs_git
def test_the_binding_starts_no_git_inside_the_block_and_says_why(tmp_path, started):
    rp = tmp_path / "r.json"
    rp.write_text("{}", encoding="utf-8")
    with receipt_binding.no_repository("a re-run asks no repository"):
        block = receipt_binding.bind_at_mint([rp])
    assert started == []
    assert block["head"] is None and block["all_receipts_committed"] is False
    assert block["note"] == "no repository at mint: a re-run asks no repository"
    assert block["receipts"][0]["raw_sha256"] == receipt_binding.raw_sha256(b"{}")
    receipt_binding.bind_at_mint([rp])
    assert started, "outside the block the binding looks for a repository, as before"


@needs_git
def test_a_mint_in_one_thread_still_sees_git_while_other_threads_verify(tmp_path, started):
    honest, _, _ = _capsule_with_its_document_named(tmp_path, ".git")
    started.clear()
    failures = []

    def verify_some():
        for _ in range(3):
            if not verify_capsule(honest)["ok"] or not verify_capsule(FIXTURE)["ok"]:
                failures.append("a capsule did not verify")

    threads = [threading.Thread(target=verify_some) for _ in range(4)]
    for t in threads:
        t.start()
    doc = tmp_path / "again.md"
    doc.write_text("The run scored 0.75 accuracy.\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    for _ in range(5):
        certify.certify_doc(doc, [rp])
    for t in threads:
        t.join()
    assert failures == []
    mine = threading.get_ident()
    assert started and all(ident == mine for _, _, ident in started), "a verifying thread started git"


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

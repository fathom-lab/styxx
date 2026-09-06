"""A local `git replace` ref cannot make the tree channel serve another commit's bytes.

`GitTree._git` ran `git -C <repo> ...` with nothing suppressing replacement, and git honours
`refs/replace/*` by default in `cat-file` and `ls-tree` alike. One ref -- no object written, no
history rewritten, `git --no-replace-objects` still printing the original bytes -- made a false
document SWORN-HELD while the provenance note still named the commit the document asked for.

This defeats the defence the receipt-provenance audit (PR #76) relies on. That leg says an unbacked
`committed` from MemoryTree or SnapshotTree would be caught by a third party re-deriving with a real
GitTree. It would not, if the repository they were handed carries a replace ref.

Spec: papers/sworn/SPEC_tree_ignores_replace_refs_v01_2026_09_06.md (T1).
Watched to fail: before the repair, T-G1 was HELD and the document SWORN-HELD.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

NEEDLE = "the precision is 0.9900"
OTHER = "the precision is 0.1100"


def _run(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False, **kw)


@pytest.fixture()
def repo_with_replace(tmp_path):
    """A repository with two commits and a replace ref pointing the first at the second.

    Returns (repo, commit_A, sha256_of_A_bytes, sha256_of_B_bytes). Commit A is what the document
    names; B carries different bytes.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "r"
    repo.mkdir()
    _run(repo, "init", "-q", ".")
    _run(repo, "config", "user.email", "t@t.t")
    _run(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text(NEEDLE + "\n", encoding="utf-8", newline="\n")
    _run(repo, "add", "f.txt")
    _run(repo, "commit", "-q", "-m", "A")
    a = _run(repo, "rev-parse", "HEAD").stdout.decode().strip()
    (repo / "f.txt").write_text(OTHER + "\n", encoding="utf-8", newline="\n")
    _run(repo, "add", "f.txt")
    _run(repo, "commit", "-q", "-m", "B")
    b = _run(repo, "rev-parse", "HEAD").stdout.decode().strip()
    return repo, a, b


def _verify(repo, commit, needle=NEEDLE):
    doc = ('<sworn r="path:f.txt" k="quote">the run reports `%s` on the panel</sworn>\n'
           % needle).encode("utf-8")
    tree = sworn.GitTree(repo, commit)
    core = sworn.verify(doc, name="D.md", manifest=None, tree=tree, commit=commit)
    return core, core["spans"][0]


def test_a_replace_ref_cannot_flip_the_verdict(repo_with_replace):
    """T-G1, the guard that must be seen red.

    The document names commit A and quotes bytes that are in A. Replacing A with B makes those bytes
    absent -- and before the repair the verifier read B and answered about A.
    """
    repo, a, b = repo_with_replace
    _core, before = _verify(repo, a)
    assert before["verdict"] == "HELD", "the premise: the needle IS in A's bytes"

    _run(repo, "replace", a, b)
    assert _run(repo, "replace", "-l").stdout.strip(), "the replace ref was not created"

    core, after = _verify(repo, a)
    assert after["verdict"] == "HELD", (
        "a local replace ref changed the bytes the verifier read for commit %s: the span went to "
        "%s/%s while the provenance note still names that commit" % (a[:12], after["verdict"],
                                                                     after.get("reason")))
    assert core["document_verdict"] == "SWORN-HELD"


def test_the_bytes_resolved_are_the_named_commits(repo_with_replace):
    """T-G2. The handle itself, by bytes -- so a verdict that happens to agree cannot hide it.

    Asked of GitTree.blob directly rather than of a span's detail: a quote span reports
    needle/haystack sizes, not the resolved object, so the verdict is the weaker instrument here.
    """
    repo, a, b = repo_with_replace
    before, why = sworn.GitTree(repo, a).blob("f.txt")
    assert before is not None, why
    assert NEEDLE.encode("utf-8") in before, before

    _run(repo, "replace", a, b)
    after, why2 = sworn.GitTree(repo, a).blob("f.txt")
    assert after == before, (
        "with a replace ref present, GitTree read %r for commit %s instead of its own %r (%s)"
        % (after, a[:12], before, why2))


def test_a_repository_with_no_replace_ref_is_unaffected(repo_with_replace):
    """T-G3. The repair must be a no-op on every input that has no replacement."""
    repo, a, _b = repo_with_replace
    _c, held = _verify(repo, a)
    assert held["verdict"] == "HELD", held
    _c2, failed = _verify(repo, a, needle="a sentence that is not in the file")
    assert failed["verdict"] == "FAILED", failed


def test_an_unrelated_replace_ref_does_not_make_the_document_unresolved(repo_with_replace):
    """T-G4. The spec's `why not refuse` clause: ignore replacement, do not decline because the
    repository merely contains some."""
    repo, a, b = repo_with_replace
    # replace B (which this document never names) with A
    _run(repo, "replace", b, a)
    _core, span = _verify(repo, a)
    assert span["verdict"] == "HELD", (
        "a replace ref unrelated to the named commit must not disturb the verdict, got %s/%s"
        % (span["verdict"], span.get("reason")))

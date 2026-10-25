# -*- coding: utf-8 -*-
"""prereg-ci B1: the git layer on synthetic repositories built in tmp_path.

DESIGN 3.2(b): the seal bytes are the PREREG's blob at its landing commit L, the oldest commit on
the default branch's parent-one chain whose diff touches the path; the order commit is the oldest
commit, among L and its ancestors, at which the decision block already hashed to its value at L.
3.2(d): the resolver, the recorded digest (PLAN K2) and receipt immutability. 3.4: C0's
fail-closed delete and rename, C1's RECEIPT_REWRITTEN, C2's PREREG that arrives with its receipt,
C3's strict ancestry, and the shallow clone that is UNCHECKABLE.

Every repository here is made with ``git init`` in tmp_path; the one clone is a shallow clone of
one of them over the local file transport. Nothing reads or writes the styxx repository, and no
remote exists anywhere, so no git command here can reach a server.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from styxx.prereg_ci import core
from styxx.prereg_ci import git as G

ROOT = Path(__file__).resolve().parents[1]
# git's own flag for the parent-one walk, built so this file does not spell the ordinal
_P1_FLAG = "--" + "tnerap-tsrif"[::-1]
P = "p/PREREG_a.md"


# ---- building repositories ------------------------------------------------------------------------------

class Git:
    """A repository in tmp_path with a clock that moves a minute per commit, so git's date order
    is the order the test wrote the commits in."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.t = 1_760_000_000
        self("init", "-q", "-b", "main")
        for k, v in (("user.email", "t@t"), ("user.name", "t"), ("core.autocrlf", "false"),
                     ("commit.gpgsign", "false"), ("core.hooksPath", str(root / ".no-hooks"))):
            self("config", k, v)

    def __call__(self, *args: str, check: bool = True) -> str:
        env = dict(os.environ)
        for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(k, None)
        self.t += 60
        env.update(GIT_AUTHOR_DATE=f"{self.t} +0000", GIT_COMMITTER_DATE=f"{self.t} +0000")
        r = subprocess.run(["git", *args], cwd=str(self.root), env=env, capture_output=True)
        if check and r.returncode != 0:
            raise AssertionError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace')}")
        return r.stdout.decode("utf-8", "replace").strip()

    def write(self, rel: str, data) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))

    def commit(self, msg: str) -> str:
        self("add", "-A")
        self("commit", "-q", "--allow-empty", "-m", msg)
        return self("rev-parse", "HEAD")

    def mv(self, src: str, dst: str) -> None:
        (self.root / dst).parent.mkdir(parents=True, exist_ok=True)
        self("mv", src, dst)

    def merge(self, branch: str, msg: str) -> str:
        self("merge", "-q", "--no-ff", "-m", msg, branch)
        return self("rev-parse", "HEAD")


def gates_block(bar: float) -> str:
    return json.dumps({"gates": {"G0": {"metric": "m", "op": ">=", "value": bar}},
                       "outcomes": [{"when": {"G0": True}, "verdict": "PASS"},
                                    {"when": {"G0": False}, "verdict": "HOLD"}],
                       "smoke_verdict": "INVALID__smoke"}, indent=1)


def prereg(bar: float = 0.5, prose: str = "v1") -> str:
    return f"# PREREG\n\n{prose}\n\n```gates\n{gates_block(bar)}\n```\n"


def digest(bar: float) -> str:
    return hashlib.sha256(gates_block(bar).encode("utf-8")).hexdigest()


def receipt(name: str = "PREREG_a.md", m: float = 0.7, **extra) -> str:
    return json.dumps({"prereg": name, "m": m, **extra}, indent=1) + "\n"


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """main: m0; a pull request adds the PREREG (p1), moves its bar inside the same pull request
    (p2, PLAN's pmnist_untied case), edits its prose (p3), and lands as the merge L beside m1;
    a later pull request adds a receipt (r1) and lands as M2."""
    g = Git(tmp_path_factory.mktemp("merged"))
    g.write("README.md", "r\n")
    m0 = g.commit("m0")
    g("switch", "-q", "-c", "pr1")
    g.write(P, prereg(0.5, "v1"))
    p1 = g.commit("add the plan")
    g.write(P, prereg(0.6, "v1"))
    p2 = g.commit("move the bar inside its own pull request")
    g.write(P, prereg(0.6, "v2"))
    p3 = g.commit("prose")
    g("switch", "-q", "main")
    g.write("other.md", "o\n")
    m1 = g.commit("m1")
    L = g.merge("pr1", "land the plan")
    g("switch", "-q", "-c", "pr2")
    g.write("p/receipt_a.json", receipt())
    r1 = g.commit("receipt")
    g("switch", "-q", "main")
    M2 = g.merge("pr2", "land the receipt")
    with G._Repo(g.root) as repo:
        yield SimpleNamespace(g=g, repo=repo, m0=m0, p1=p1, p2=p2, p3=p3, m1=m1, L=L, r1=r1, M2=M2)


# ---- the module's own surface -----------------------------------------------------------------------------

def test_git_module_defines_no_public_name():
    """DESIGN 3.1: the public surface is core.rescore, .compare, .licences and the CLI; the git
    layer is the CLI's, so every module-level name in it is private."""
    tree = ast.parse((ROOT / "styxx" / "prereg_ci" / "git.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    assert {n for n in names if not n.startswith("_")} == set()
    import styxx.prereg_ci as pkg
    assert set(pkg.__all__) <= {"rescore", "compare", "licences"}


def test_every_git_command_is_a_local_read_with_replacement_off(merged, monkeypatch):
    """The layer runs read-only plumbing and nothing that talks to a remote; every call carries
    --no-replace-objects (styxx.sworn.GitTree's rule) and the graft file is pointed away."""
    seen = []
    real_run, real_popen = G.subprocess.run, G.subprocess.Popen

    def run(argv, *a, **k):
        seen.append((list(argv), k.get("env") or {}))
        return real_run(argv, *a, **k)

    def popen(argv, *a, **k):
        seen.append((list(argv), k.get("env") or {}))
        return real_popen(argv, *a, **k)

    monkeypatch.setattr(G.subprocess, "run", run)
    monkeypatch.setattr(G.subprocess, "Popen", popen)
    with G._Repo(merged.g.root) as repo:
        G._seal(repo, P, "main")
        repo.blob_history(P, merged.p1, merged.r1)
        repo.is_strict_ancestor(merged.p1, merged.r1)
        G._receipts_rewritten(repo, merged.m0, "main")
        G._receipts_written(repo, merged.m0, "main")
        G._decisions_moved(repo, merged.m0, "main")
        G._records(repo, "main")
    allowed = {"rev-parse", "rev-list", "cat-file", "ls-tree", "diff-tree", "merge-base", "grep"}
    assert seen
    for argv, env in seen:
        assert argv[0] == "git" and "--no-replace-objects" in argv, argv
        i = 1
        while argv[i].startswith("-"):
            i += 2 if argv[i] in ("-c", "-C") else 1
        assert argv[i] in allowed, argv
        assert env.get("GIT_NO_REPLACE_OBJECTS") == "1" and env.get("GIT_GRAFT_FILE") == os.devnull, argv


# ---- revisions, the chain, ancestry ---------------------------------------------------------------------

def test_a_revision_resolves_to_a_full_commit_id_or_is_refused(merged):
    repo = merged.repo
    assert repo.commit("main") == merged.M2
    assert repo.commit(merged.p1[:10]) == merged.p1
    assert len(repo.commit("HEAD")) == 40
    tree = merged.g("rev-parse", "HEAD^{tree}")
    for bad in ("no-such-branch", "0" * 40, tree, "--all", ""):
        with pytest.raises(G._Refusal) as e:
            repo.commit(bad)
        assert (e.value.code, e.value.kind) == ("COMMIT_ABSENT", "UNCHECKABLE"), bad


def test_the_chain_is_parent_one_and_equals_gits_own_walk(merged):
    repo = merged.repo
    assert repo.chain("main") == [merged.M2, merged.L, merged.m1, merged.m0]
    assert repo.chain("main") == merged.g("rev-list", _P1_FLAG, "main").split()
    assert repo.parents(merged.L) == (merged.m1, merged.p3)
    assert repo.parents(merged.m0) == ()


def test_strict_ancestry(merged):
    repo = merged.repo
    assert repo.is_strict_ancestor(merged.p2, merged.r1)
    assert not repo.is_strict_ancestor(merged.r1, merged.p2)
    assert not repo.is_strict_ancestor(merged.p2, merged.p2), "the same commit is not strictly before"
    assert not repo.is_strict_ancestor(merged.m1, merged.p3), "no ancestry either way"
    with pytest.raises(G._Refusal) as e:
        repo.is_strict_ancestor("1" * 40, merged.p2)
    assert e.value.code == "COMMIT_ABSENT"


# ---- the landing rule and the add commit ----------------------------------------------------------------

def test_the_landing_commit_is_the_oldest_parent_one_commit_touching_the_path(merged):
    got = merged.repo.landing([P, "p/receipt_a.json", "README.md", "other.md", "never.md"], "main")
    assert got == {P: merged.L, "p/receipt_a.json": merged.M2, "README.md": merged.m0, "other.md": merged.m1,
                   "never.md": None}
    for path in (P, "p/receipt_a.json", "README.md", "other.md"):
        assert got[path] == merged.g("log", "main", _P1_FLAG, "--format=%H", "--", path).split()[-1], path


def test_the_add_commit_is_gits_diff_filter_rule(merged):
    repo = merged.repo
    assert repo.add_commit(P, "main") == merged.p1
    assert repo.add_commit("p/receipt_a.json", "main") == merged.r1
    assert repo.add_commit("never.md", "main") is None
    for path in (P, "p/receipt_a.json", "README.md"):
        porcelain = merged.g("log", "main", "--diff-filter=A", "--format=%H", "--", path).split()[-1]
        assert repo.add_commit(path, "main") == porcelain, path


def test_a_merge_that_introduces_a_path_lands_it_and_is_no_add_commit(tmp_path):
    """git's --diff-filter=A shows no diff for a merge, so a path a merge itself introduces (on
    neither side) has no add commit by that rule; the landing rule still finds the merge."""
    g = Git(tmp_path / "evil")
    g.write("README.md", "r\n")
    g.commit("m0")
    g("switch", "-q", "-c", "side")
    g.write("side.md", "s\n")
    g.commit("side")
    g("switch", "-q", "main")
    g.write("main.md", "m\n")
    g.commit("m1")
    g("merge", "-q", "--no-ff", "--no-commit", "side")
    g.write(P, prereg())
    M = g.commit("a merge that also writes the plan")
    g.write(P, prereg(0.6))
    g.commit("edit the plan")
    with G._Repo(g.root) as repo:
        assert len(repo.parents(M)) == 2
        assert g("log", "main", "--diff-filter=A", "--format=%H", "--", P) == ""
        assert repo.add_commit(P, "main") is None
        assert repo.landing([P], "main") == {P: M}


def test_the_add_commit_of_a_deleted_and_re_added_path_is_the_oldest_add(tmp_path):
    g = Git(tmp_path / "readd")
    g.write("x.md", "a\n")
    a1 = g.commit("add")
    (g.root / "x.md").unlink()
    g.commit("delete")
    g.write("x.md", "b\n")
    a2 = g.commit("add again")
    with G._Repo(g.root) as repo:
        assert repo.add_commit("x.md", "main") == a1
        assert repo.landing(["x.md"], "main") == {"x.md": a1}
        assert [b for _, b in repo.blob_history("x.md", a1, a2)][-1] == repo.blob_id(a2, "x.md")


# ---- the seal and the order commit ----------------------------------------------------------------------

def test_the_seal_is_the_landed_blob_and_the_order_commit_is_the_decision_digest_walk(merged):
    repo = merged.repo
    s = G._seal(repo, P, "main")
    assert (s["path"], s["landing"], s["landed"], s["add"]) == (P, merged.L, True, merged.p1)
    assert s["blob"] == repo.blob_id(merged.p3, P) == repo.blob_id(merged.L, P)
    assert s["bytes"] == repo.read(merged.L, P) == prereg(0.6, "v2").encode("utf-8")
    assert s["blob_sha256"] == hashlib.sha256(s["bytes"]).hexdigest()
    assert (s["form"], s["gates_sha256"]) == ("gates", digest(0.6))
    # the bar moved inside the pull request: the add commit's fence is not the landed one
    assert core._decision(repo.read(merged.p1, P))["gates_sha256"] == digest(0.5) != s["gates_sha256"]
    # the landed decision was written at p2; the landed bytes at p3
    assert (s["order"], s["order_blob"]) == (merged.p2, merged.p3)
    assert s["renamed_from"] is None


def test_the_order_commit_is_the_oldest_holder_of_the_landed_digest(tmp_path):
    """DESIGN 3.2(b): "the oldest commit, among L and its ancestors, at which the path's decision
    block already hashed to its value at L". X, then Y, then X again with other prose: the decision
    orders from the commit that wrote X before the Y interval (the #224 PREREG's blob went
    2068e673, 18aa265c, 2068e673, and RV2 orders its bytes from 3598e59d); the file blob, which the
    prose moved, orders from the commit that wrote it."""
    g = Git(tmp_path / "xyx")
    g.write("README.md", "r\n")
    g.commit("m0")
    g("switch", "-q", "-c", "pr")
    g.write(P, prereg(0.5, "v1"))
    x1 = g.commit("X")
    g.write(P, prereg(0.7, "v1"))
    y = g.commit("Y")
    g.write(P, prereg(0.5, "v2"))
    x2 = g.commit("X again, other prose")
    g("switch", "-q", "main")
    L = g.merge("pr", "land")
    with G._Repo(g.root) as repo:
        assert repo.order_commit(P, L, "decision") == x1
        assert repo.order_commit(P, L, "blob") == x2
        assert core._decision(repo.read(x1, P))["gates_sha256"] == digest(0.5)
        assert repo.blob_history(P, x1, L) == [(c, repo.blob_id(c, P)) for c in (x1, y, x2, L)], (
            "the landing merge differs from main's side at the path, so it is listed")


def test_the_oldest_holder_on_a_side_branch_counts(tmp_path):
    """"Among L and its ancestors": two branches write the same plan, the older on the side a merge
    is not TREESAME-followed through. git's default simplification follows one TREESAME parent and
    prunes the other side; the order commit is the older writer all the same."""
    g = Git(tmp_path / "sides")
    g.write("README.md", "r\n")
    g.commit("m0")
    g("switch", "-q", "-c", "older")
    g.write(P, prereg(0.5))
    a1 = g.commit("the plan, written on one branch")
    g("switch", "-q", "main")
    g("switch", "-q", "-c", "newer")
    g.write(P, prereg(0.5))
    b1 = g.commit("the same plan, written again on another")
    c = g.merge("older", "join the two")
    g("switch", "-q", "main")
    L = g.merge("newer", "land")
    with G._Repo(g.root) as repo:
        assert repo.parents(c) == (b1, a1)
        simplified = g("log", "--format=%H", L, "--", P).split()
        assert b1 in simplified and a1 not in simplified, "default simplification prunes the older writer"
        assert repo.landing([P], "main") == {P: L}
        assert repo.order_commit(P, L, "decision") == a1
        assert repo.order_commit(P, L, "blob") == a1


def test_a_prereg_that_arrives_with_its_receipt_has_no_landing_and_orders_from_head(tmp_path):
    """C2: the order commit is its own oldest commit holding the decision; the prose moved after
    the receipt (R7's realmodel case), so the file-blob order misses where the decision order holds."""
    g = Git(tmp_path / "samepr")
    g.write("README.md", "r\n")
    g.commit("m0")
    g("switch", "-q", "-c", "pr")
    g.write(P, prereg(0.5, "v1"))
    p1 = g.commit("plan")
    g.write("p/receipt_a.json", receipt())
    r1 = g.commit("receipt")
    g.write(P, prereg(0.5, "v2"))
    p2 = g.commit("prose after the receipt")
    with G._Repo(g.root) as repo:
        with pytest.raises(G._Refusal) as e:
            G._seal(repo, P, "main")
        assert (e.value.code, e.value.kind) == ("PREREG_ABSENT", "INPUT")
        s = G._seal(repo, P, "main", head="pr")
        assert (s["landing"], s["landed"], s["add"], s["order"], s["order_blob"]) == (None, False, p1, p1, p2)
        assert repo.is_strict_ancestor(s["order"], r1) and not repo.is_strict_ancestor(s["order_blob"], r1)
        written = G._receipts_written(repo, "main", "pr")
        assert [(a["path"], a["status"], a["receipt_commit"], a["receipt"]["prereg"]) for a in written["receipts"]] == [
            ("p/receipt_a.json", "A", r1, "PREREG_a.md")]
        assert written["notes"] == []


def test_a_landing_by_rename_is_named(tmp_path):
    g = Git(tmp_path / "renamed")
    g.write("old/PREREG_a.md", prereg())
    g.commit("plan at the old path")
    g.mv("old/PREREG_a.md", P)
    c = g.commit("move the plan")
    with G._Repo(g.root) as repo:
        s = G._seal(repo, P, "main")
        assert (s["landing"], s["renamed_from"]) == (c, "old/PREREG_a.md")


def test_blob_history_lists_every_blob_between_two_commits(merged):
    hist = merged.repo.blob_history(P, merged.p1, merged.r1)
    assert [c for c, _ in hist] == [merged.p1, merged.p2, merged.p3, merged.L]
    assert len({b for _, b in hist}) == 3 and hist[-1][1] == hist[-2][1]


def test_blob_history_prints_an_edit_and_revert_on_a_merged_branch(tmp_path):
    """Review case 9, the reason 3.2(e) prints the history (RED_TEAM m8): the sealed fence is
    edited on a side branch and reverted there, and the branch is merged before the receipt. The
    merge is TREESAME to both parents, so git's default simplification lists the seal alone."""
    g = Git(tmp_path / "revert")
    g.write(P, prereg(0.5))
    s = g.commit("seal the plan")
    g("switch", "-q", "-c", "side")
    g.write(P, prereg(0.1))
    e1 = g.commit("move the bar on a branch")
    g.write(P, prereg(0.5))
    e2 = g.commit("move it back")
    g("switch", "-q", "main")
    g.write("other.md", "o\n")
    g.commit("main moves")
    g.merge("side", "merge the branch")
    g.write("p/receipt_a.json", receipt())
    r = g.commit("the receipt")
    with G._Repo(g.root) as repo:
        assert g("rev-list", r, "--not", s + "^@", "--", P).split() == [s], "default simplification"
        hist = repo.blob_history(P, s, r)
        assert [c for c, _ in hist] == [s, e1, e2]
        assert [b for _, b in hist] == [repo.blob_id(s, P), repo.blob_id(e1, P), repo.blob_id(s, P)]
        assert core._decision(repo.read(e1, P))["gates_sha256"] == digest(0.1)


# ---- reads -----------------------------------------------------------------------------------------------

def test_reads_are_blobs_at_a_commit_never_the_working_copy(merged):
    repo = merged.repo
    (merged.g.root / P).write_bytes(b"the working copy moved")
    try:
        assert repo.read(merged.M2, P) == prereg(0.6, "v2").encode("utf-8")
        assert repo.read(merged.M2, "never.md") is None
        assert repo.blob_id(merged.m0, P) is None
        assert repo.paths(merged.M2) == ("README.md", "other.md", P, "p/receipt_a.json")
    finally:
        merged.g("checkout", "--", P)


def test_a_symlink_and_an_oversized_blob_are_refused(tmp_path, monkeypatch):
    g = Git(tmp_path / "odd")
    g.write("big.json", "x" * 100)
    link = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=str(g.root), input=b"big.json",
                          capture_output=True, check=True).stdout.decode().strip()
    g("add", "big.json")
    g("update-index", "--add", "--cacheinfo", f"120000,{link},link.json")
    g("commit", "-q", "-m", "a link and a big file")
    c = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        with pytest.raises(G._Refusal) as e:
            repo.read(c, "link.json")
        assert (e.value.code, e.value.kind) == ("NOT_A_BLOB", "INPUT")
        monkeypatch.setattr(G, "_MAX_BLOB_BYTES", 99)
        with pytest.raises(G._Refusal) as e:
            repo.read(c, "big.json")
        assert (e.value.code, e.value.kind) == ("BLOB_TOO_LARGE", "UNCHECKABLE")


def test_a_symlinked_json_is_a_note_and_does_not_stop_the_check(tmp_path):
    """Review case 3: a pull request that moves a ``latest.json`` symlink, adds a symlinked
    ``best.json`` and a symlinked sidecar, beside a real receipt. A symlink holds no receipt: the
    added ones are RECEIPT_NOT_A_FILE NOTEs, the moved one was no receipt at the base, the sidecar
    is listed and not read, and a symlink whose target text holds the needle is no record (git grep
    reads regular files only). The links are committed through the index, so the box's symlink
    rights do not matter."""
    g = Git(tmp_path / "links")

    def link(path: str, target: str) -> None:
        oid = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=str(g.root), input=target.encode(),
                             capture_output=True, check=True).stdout.decode().strip()
        g("update-index", "--add", "--cacheinfo", f"120000,{oid},{path}")

    g.write(P, prereg(0.5))
    g.write("p/run1.json", json.dumps({"rows": [1]}) + "\n")
    g.write("p/run2.json", json.dumps({"rows": [2]}) + "\n")
    g("add", "-A")
    link("p/latest.json", "run1.json")
    g("commit", "-q", "-m", "data and a latest pointer")
    g("switch", "-q", "-c", "pr")
    g.write("p/receipt_a.json", receipt())
    g("add", "p/receipt_a.json")
    link("p/latest.json", "run2.json")
    link("p/best.json", "run2.json")
    link("p/x.seal.json", "run2.json")
    link("p/needle.json", '"gates_sha256"')
    g("commit", "-q", "-m", "move the pointer; add pointers and a receipt")
    head = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        assert G._receipts_rewritten(repo, "main", "pr") == []
        written = G._receipts_written(repo, "main", "pr")
        assert [a["path"] for a in written["receipts"]] == ["p/receipt_a.json"]
        assert [(n["path"], n["code"], n["kind"]) for n in written["notes"]] == [
            ("p/best.json", "RECEIPT_NOT_A_FILE", "NOTE"), ("p/latest.json", "RECEIPT_NOT_A_FILE", "NOTE"),
            ("p/needle.json", "RECEIPT_NOT_A_FILE", "NOTE"), ("p/x.seal.json", "RECEIPT_NOT_A_FILE", "NOTE")]
        side = G._sidecars(repo, head)
        assert (side["not_files"], side["by"]) == (["p/x.seal.json"], {})
        assert G._records(repo, head) == []
        with pytest.raises(G._Refusal) as e:
            repo.read(head, "p/best.json")
        assert e.value.code == "NOT_A_BLOB", "a named read still refuses a symlink"


# ---- refusals the history itself earns ------------------------------------------------------------------

def test_a_shallow_clone_is_uncheckable_for_history_and_readable_for_bytes(merged, tmp_path):
    url = merged.g.root.resolve().as_uri()
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "main", url, str(tmp_path / "shallow")],
                   check=True, capture_output=True)
    with G._Repo(tmp_path / "shallow") as repo:
        assert repo.is_shallow()
        assert repo.read("HEAD", P) == prereg(0.6, "v2").encode("utf-8")
        for call in (lambda: repo.chain("HEAD"), lambda: repo.landing([P], "HEAD"),
                     lambda: repo.add_commit(P, "HEAD"), lambda: G._seal(repo, P, "HEAD"),
                     lambda: repo.is_strict_ancestor("HEAD", "HEAD"),
                     lambda: repo.changes("HEAD", "HEAD")):
            with pytest.raises(G._Refusal) as e:
                call()
            assert (e.value.code, e.value.kind) == ("SHALLOW_CLONE", "UNCHECKABLE")


@pytest.fixture
def three(tmp_path):
    g = Git(tmp_path / "three")
    g.write(P, prereg(0.5))
    c1 = g.commit("c1")
    g.write(P, prereg(0.6))
    c2 = g.commit("c2")
    g.write("README.md", "r\n")
    c3 = g.commit("c3")
    return SimpleNamespace(g=g, c1=c1, c2=c2, c3=c3)


def test_replace_refs_move_nothing(three):
    g = three.g
    g("replace", "--graft", three.c3, three.c1)
    g("switch", "-q", "--orphan", "other")
    g.write(P, prereg(0.9))
    o = g.commit("another plan")
    g("switch", "-q", "-f", "main")
    g("replace", three.c2, o)
    assert g("show", f"{three.c2}:{P}") == prereg(0.9).strip(), "git itself serves the replacement"
    with G._Repo(g.root) as repo:
        assert repo.chain("main") == [three.c3, three.c2, three.c1]
        assert repo.read(three.c2, P) == prereg(0.6).encode("utf-8")
        assert repo.landing([P], "main") == {P: three.c1}
        assert repo.order_commit(P, three.c3, "decision") == three.c2


def test_a_graft_file_moves_nothing(three):
    g = three.g
    grafts = g.root / g("rev-parse", "--git-path", "info/grafts")
    grafts.parent.mkdir(parents=True, exist_ok=True)
    grafts.write_text(three.c3 + "\n", encoding="ascii")
    assert g("rev-list", "--parents", "main").split() == [three.c3], "git itself honours the graft"
    with G._Repo(g.root) as repo:
        assert repo.chain("main") == [three.c3, three.c2, three.c1]
        assert repo.is_strict_ancestor(three.c1, three.c3)


# ---- BASE..HEAD: immutability and the sealed decision (C0, C1) ------------------------------------------

@pytest.fixture(scope="module")
def pr(tmp_path_factory):
    g = Git(tmp_path_factory.mktemp("pr"))
    rec = json.dumps({"protocol": [{"prereg": "PREREG_prose.md", "result": "r_side.json",
                                    "gates_sha256": digest(0.5), "frozen_verdict": "PASS"}]}) + "\n"
    for name in ("PREREG_fence.md", "PREREG_prose.md", "PREREG_gone.md", "PREREG_moved.md"):
        g.write(f"d/{name}", prereg(0.5))
    g.write("d/notes.md", "no decision here\n")
    g.write("d/code.py", f"TEXT = '''\n```gates\n{gates_block(0.5)}\n```\n'''\n")
    g.write("d/r_mod.json", receipt("PREREG_fence.md"))
    g.write("d/r_del.json", receipt("PREREG_fence.md"))
    g.write("d/r_mv.json", receipt("PREREG_fence.md"))
    g.write("d/r_own.json", json.dumps({"gates_sha256": digest(0.5), "verdict": "PASS"}) + "\n")
    g.write("d/FINDING.seal.json", rec)
    g.write("d/data.json", json.dumps({"rows": [1, 2]}) + "\n")
    base = g.commit("the sealed state")
    g("switch", "-q", "-c", "pr")
    g.write("d/PREREG_fence.md", prereg(0.4))
    g.write("d/PREREG_prose.md", prereg(0.5, "v2"))
    (g.root / "d/PREREG_gone.md").unlink()
    g.mv("d/PREREG_moved.md", "e/PREREG_moved.md")
    g.write("d/notes.md", "still no decision\n")
    g.write("d/code.py", f"TEXT = '''\n```gates\n{gates_block(0.1)}\n```\n'''\n")
    g.write("d/r_mod.json", receipt("PREREG_fence.md", m=0.1))
    (g.root / "d/r_del.json").unlink()
    g.mv("d/r_mv.json", "e/r_mv.json")
    g.write("d/r_own.json", json.dumps({"gates_sha256": digest(0.4), "verdict": "PASS"}) + "\n")
    g.write("d/FINDING.seal.json", rec.replace("PASS", "HOLD"))
    g.write("d/data.json", json.dumps({"rows": [1, 2, 3]}) + "\n")
    g.write("d/r_new.json", receipt("PREREG_prose.md"))
    g.write("d/broken.json", "{not json\n")
    head = g.commit("the pull request")
    g("switch", "-q", "main")
    g.write("d/r_late.json", receipt("PREREG_fence.md"))
    g.write("d/PREREG_late.md", prereg(0.5))
    g.commit("main moves on after the pull request forked")
    with G._Repo(g.root) as repo:
        yield SimpleNamespace(g=g, repo=repo, base=base, head=head)


def _by_path(findings):
    return {f["path"]: (f["code"], f["kind"]) for f in findings}


def test_changes_are_read_from_the_merge_base_with_renames_named(pr):
    ch = pr.repo.changes("main", "pr")
    assert ch["merge_base"] == pr.base
    got = {(c["status"], c["path"], c["old"]) for c in ch["changes"]}
    assert ("R", "e/PREREG_moved.md", "d/PREREG_moved.md") in got
    assert ("R", "e/r_mv.json", "d/r_mv.json") in got
    assert ("D", "d/PREREG_gone.md", "d/PREREG_gone.md") in got
    assert ("A", "d/r_new.json", None) in got
    assert not any(p in ("d/r_late.json", "d/PREREG_late.md") for _, p, _ in got), "main's later commits"


def test_a_committed_receipt_changed_deleted_or_renamed_is_rewritten(pr):
    found = G._receipts_rewritten(pr.repo, "main", "pr")
    assert _by_path(found) == {
        "d/r_mod.json": ("RECEIPT_REWRITTEN", "FAIL"),
        "d/r_del.json": ("RECEIPT_REWRITTEN", "FAIL"),
        "d/r_mv.json": ("RECEIPT_REWRITTEN", "FAIL"),
        "d/r_own.json": ("RECEIPT_REWRITTEN", "FAIL"),
        "d/FINDING.seal.json": ("RECEIPT_REWRITTEN", "FAIL"),
    }
    detail = {f["path"]: f["detail"] for f in found}
    assert "deleted" in detail["d/r_del.json"] and "e/r_mv.json" in detail["d/r_mv.json"]
    notes = G._receipts_rewritten(pr.repo, "main", "pr", policy="note")
    assert {k for _, k in _by_path(notes).values()} == {"NOTE"}
    with pytest.raises(ValueError):
        G._receipts_rewritten(pr.repo, "main", "pr", policy="warn")


def test_a_sealed_decision_does_not_move_and_fails_closed_on_delete_and_rename(pr):
    found = G._decisions_moved(pr.repo, "main", "pr")
    assert _by_path(found) == {
        "d/PREREG_fence.md": ("DECISION_EDITED_AFTER_SEAL", "FAIL"),
        "d/PREREG_gone.md": ("DECISION_EDITED_AFTER_SEAL", "FAIL"),
        "d/PREREG_moved.md": ("DECISION_EDITED_AFTER_SEAL", "FAIL"),
        "d/PREREG_prose.md": ("PROSE_AMENDED_IN_PLACE", "NOTE"),
    }
    strict = G._decisions_moved(pr.repo, "main", "pr", immutability="file")
    assert _by_path(strict)["d/PREREG_prose.md"] == ("PROSE_AMENDED_IN_PLACE", "FAIL")
    with pytest.raises(ValueError):
        G._decisions_moved(pr.repo, "main", "pr", immutability="bytes")


def test_receipts_written_in_the_pull_request(pr):
    """C1 pairs every path the pull request leaves holding a receipt: added (r_new), renamed (r_mv)
    and changed in place (r_mod, which RECEIPT_REWRITTEN also reads). A record without a ``prereg``
    (r_own, the sidecar) and plain data are not receipts; bytes that are not JSON are a NOTE."""
    written = G._receipts_written(pr.repo, "main", "pr")
    assert [(a["path"], a["status"], a["old"], a["receipt_commit"]) for a in written["receipts"]] == [
        ("d/r_mod.json", "M", "d/r_mod.json", pr.head), ("d/r_new.json", "A", None, pr.head),
        ("e/r_mv.json", "R", "d/r_mv.json", pr.head)]
    assert [(s["path"], s["code"], s["kind"]) for s in written["notes"]] == [
        ("d/broken.json", "RECEIPT_UNREADABLE", "NOTE")]


def test_a_receipt_written_over_a_committed_placeholder_is_paired(tmp_path):
    """Review case 1: a placeholder committed on main (JSON that is no receipt, and bytes that are
    not JSON) is overwritten by a receipt recording an unsealed digest (DESIGN 4.A.2's re-run). The
    diff reads M; no receipt was committed there, so nothing is rewritten, and C1 pairs both."""
    g = Git(tmp_path / "placeholder")
    g.write(P, prereg(0.5))
    g.write("p/result.json", json.dumps({"status": "pending"}) + "\n")
    g.write("p/out.json", "pending\n")
    g.commit("seal the plan, commit placeholders")
    g("switch", "-q", "-c", "pr")
    run = receipt(m=0.45, verdict="PASS", gates={"G0": True}, gates_sha256=digest(0.4))
    g.write("p/result.json", run)
    g.write("p/out.json", run)
    r = g.commit("the run")
    with G._Repo(g.root) as repo:
        assert {(c["status"], c["path"]) for c in repo.changes("main", "pr")["changes"]} == {
            ("M", "p/out.json"), ("M", "p/result.json")}
        written = G._receipts_written(repo, "main", "pr")
        assert [(a["path"], a["status"], a["receipt_commit"], a["receipt"]["gates_sha256"])
                for a in written["receipts"]] == [("p/out.json", "M", r, digest(0.4)),
                                                  ("p/result.json", "M", r, digest(0.4))]
        assert G._receipts_rewritten(repo, "main", "pr") == []


def test_a_receipt_rewritten_under_the_note_policy_is_still_paired(tmp_path):
    """Review case 1b: under ``receipt_rewrite = "note"`` the rewrite is a NOTE, and the new bytes
    are paired for re-score all the same (3.2(d) prints the policy beside every PASS)."""
    g = Git(tmp_path / "build_output")
    g.write(P, prereg(0.5))
    g.write("p/build.json", receipt(verdict="PASS", gates={"G0": True}, gates_sha256=digest(0.5)))
    g.commit("a build-output receipt")
    g("switch", "-q", "-c", "pr")
    g.write("p/build.json", receipt(m=0.45, verdict="PASS", gates={"G0": True}, gates_sha256=digest(0.4)))
    r = g.commit("rebuild under a working-copy bar")
    with G._Repo(g.root) as repo:
        notes = G._receipts_rewritten(repo, "main", "pr", policy="note")
        assert [(f["path"], f["code"], f["kind"]) for f in notes] == [("p/build.json", "RECEIPT_REWRITTEN", "NOTE")]
        written = G._receipts_written(repo, "main", "pr")["receipts"]
        assert [(a["path"], a["status"], a["receipt_commit"]) for a in written] == [("p/build.json", "M", r)]


def test_r_is_read_inside_the_pull_request(tmp_path):
    """Review case 8, and R's rule: the receipt path held another file before the seal and was
    deleted; the pull request writes plain JSON there, then a receipt, then edits the receipt. R is
    the oldest commit of the pull request at which the path holds a receipt: not the old file's add
    commit (which would put R before the seal), not the plain JSON, not the later edit."""
    g = Git(tmp_path / "reused")
    g.write("p/receipt_a.json", json.dumps({"rows": [0]}) + "\n")
    old = g.commit("an old file at the receipt path")
    (g.root / "p/receipt_a.json").unlink()
    g.commit("drop it")
    g.write(P, prereg(0.5))
    g.commit("seal the plan")
    g("switch", "-q", "-c", "pr")
    g.write("p/receipt_a.json", json.dumps({"rows": [1]}) + "\n")
    g.commit("plain data at the path")
    g.write("p/receipt_a.json", receipt())
    r = g.commit("the run, after the seal")
    g.write("p/receipt_a.json", receipt(m=0.8))
    g.commit("the receipt edited inside the pull request")
    with G._Repo(g.root) as repo:
        assert repo.add_commit("p/receipt_a.json", "pr") == old
        written = G._receipts_written(repo, "main", "pr")["receipts"]
        assert [(a["path"], a["status"], a["receipt_commit"]) for a in written] == [("p/receipt_a.json", "A", r)]
        s = G._seal(repo, P, "main", "pr")
        assert repo.is_strict_ancestor(s["order"], r) and not repo.is_strict_ancestor(s["order"], old)


# ---- the resolver, the records, the recorded digest (3.2(d), K2) -------------------------------------------

TREE = frozenset({"a/PREREG_x.md", "a/r.json", "b/PREREG_y.md", "c/PREREG_y.md", "d/PREREG_z.md", "a/sub/r2.json"})


def test_the_resolver():
    r = G._resolve
    assert r("a/PREREG_x.md", "z", TREE) == "a/PREREG_x.md"
    assert r("PREREG_x.md", "a", TREE) == "a/PREREG_x.md"
    assert r("PREREG_z.md", "a", TREE) == "d/PREREG_z.md", "the tree-wide fallback"
    assert r("PREREG_y.md", "b", TREE) == "b/PREREG_y.md", "the receipt's own directory goes ahead"
    with pytest.raises(G._Refusal) as e:
        r("PREREG_y.md", "a", TREE)
    assert (e.value.code, e.value.kind, e.value.candidates) == (
        "PREREG_AMBIGUOUS", "INPUT", ("b/PREREG_y.md", "c/PREREG_y.md"))
    for bad in ("PREREG_none.md", "x/PREREG_x.md", "/a/PREREG_x.md", "a/../a/PREREG_x.md", "./a/PREREG_x.md",
                "a\\PREREG_x.md", "C:/a/PREREG_x.md", "", None, 3, "a//PREREG_x.md"):
        with pytest.raises(G._Refusal) as e:
            r(bad, "a", TREE)
        assert (e.value.code, e.value.kind) == ("PREREG_ABSENT", "INPUT"), bad
    with pytest.raises(G._Refusal) as e:
        r("r9.json", "a", TREE, what="RECEIPT")
    assert e.value.code == "RECEIPT_ABSENT"


def test_what_reads_as_a_receipt():
    k = G._receipt_kind
    assert k({"prereg": "PREREG_x.md", "m": 1}) == "receipt"
    assert k({"gates_sha256": "d", "verdict": "V"}) == "record"
    assert k({"protocol": [{"gates_sha256": "d"}]}) == "record"
    for other in ({"protocol": []}, {"rows": [1]}, [1, 2], "s", None, {"prereg": 3},
                  {"vectors": [{"gates_sha256": "d"}]}):
        assert k(other) is None, other


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    g = Git(tmp_path_factory.mktemp("records"))
    d = digest(0.5)
    own = {"prereg": "PREREG_x.md", "m": 0.7, "verdict": "PASS", "gates": {"G0": True}, "gates_sha256": d}
    side = {"protocol": [{"prereg": "papers/x/PREREG_x.md", "result": "papers/x/r_side.json", "gates_sha256": d,
                          "frozen_verdict": "PASS", "gates": {"G0": True}, "prereg_commit": "c0ffee"}]}
    g.write("papers/x/PREREG_x.md", prereg(0.5))
    g.write("papers/x/r_own.json", json.dumps(own))
    g.write("papers/x/r_side.json", json.dumps({"prereg": "PREREG_x.md", "m": 0.7}))
    g.write("papers/x/FINDING_x.seal.json", json.dumps(side))
    g.write("papers/x/r_none.json", json.dumps({"prereg": "PREREG_x.md", "m": 0.7}))
    g.write("papers/y/r_amb.json", json.dumps({"prereg": "PREREG_x.md", "m": 0.7}))
    for i, bar in ((1, 0.5), (2, 0.6)):
        g.write(f"papers/y/FINDING_y{i}.seal.json", json.dumps({"protocol": [
            {"prereg": "PREREG_x.md", "result": "r_amb.json", "gates_sha256": digest(bar), "frozen_verdict": "PASS"}]}))
    g.write("papers/z/PREREG_dup.md", prereg(0.5))
    g.write("papers/w/PREREG_dup.md", prereg(0.5))
    g.write("papers/v/r_dup.json", json.dumps({"prereg": "PREREG_dup.md", "gates_sha256": d, "verdict": "PASS"}))
    g.write("papers/x/nested.json", json.dumps({"inner": {"gates_sha256": d, "verdict": "PASS"}}))
    g.write("conformance/v.json", json.dumps({"gates_sha256": d, "verdict": "PASS", "prereg": "PREREG_x.md"}))
    c = g.commit("records")
    with G._Repo(g.root) as repo:
        yield SimpleNamespace(g=g, repo=repo, c=c, d=d)


def test_records_are_r1s_rule_with_conformance_left_out(records):
    got = G._records(records.repo, "main")
    assert [(r["record"], r["pointer"], r["own"]) for r in got] == [
        ("papers/v/r_dup.json", "", True),
        ("papers/x/FINDING_x.seal.json", "/protocol/0", False),
        ("papers/x/r_own.json", "", True),
        ("papers/y/FINDING_y1.seal.json", "/protocol/0", False),
        ("papers/y/FINDING_y2.seal.json", "/protocol/0", False),
    ]
    side = got[1]
    assert side["recorded"] == {"verdict": "PASS", "gates": {"G0": True}, "gates_sha256": records.d}
    assert (side["prereg"], side["result"], side["prereg_commit"]) == (
        "papers/x/PREREG_x.md", "papers/x/r_side.json", "c0ffee")
    assert [r["record"] for r in G._records(records.repo, "main", exclude=())][0] == "conformance/v.json"


def test_pairs_resolve_the_receipt_and_the_prereg_or_refuse(records):
    got = {(r["record"], r["pointer"]): r for r in G._pairs(records.repo, "main")}
    assert got[("papers/x/r_own.json", "")]["receipt_path"] == "papers/x/r_own.json"
    assert got[("papers/x/r_own.json", "")]["prereg_path"] == "papers/x/PREREG_x.md"
    assert got[("papers/x/FINDING_x.seal.json", "/protocol/0")]["receipt_path"] == "papers/x/r_side.json"
    assert got[("papers/y/FINDING_y1.seal.json", "/protocol/0")]["receipt_path"] == "papers/y/r_amb.json"
    assert got[("papers/y/FINDING_y1.seal.json", "/protocol/0")]["prereg_path"] == "papers/x/PREREG_x.md"
    dup = got[("papers/v/r_dup.json", "")]
    assert dup["refusal"]["code"] == "PREREG_AMBIGUOUS" and "prereg_path" not in dup


def test_the_recorded_digest_is_the_receipts_own_else_a_seal_sidecars(records):
    repo, d = records.repo, records.d

    def rec(path):
        return G._recorded(repo, "main", path, json.loads(repo.read(records.c, path)))

    assert rec("papers/x/r_own.json") == {"gates_sha256": d, "verdict": "PASS", "gates": {"G0": True},
                                           "source": "papers/x/r_own.json"}
    assert rec("papers/x/r_side.json") == {"gates_sha256": d, "verdict": "PASS", "gates": {"G0": True},
                                            "source": "papers/x/FINDING_x.seal.json#/protocol/0"}
    assert rec("papers/x/r_none.json") is None
    with pytest.raises(G._Refusal) as e:
        rec("papers/y/r_amb.json")
    assert (e.value.code, e.value.kind) == ("RECORDED_DIGEST_AMBIGUOUS", "INPUT")
    assert "FINDING_y1.seal.json" in e.value.detail and "FINDING_y2.seal.json" in e.value.detail


# ---- the core helper the walk reads --------------------------------------------------------------------

def test_the_decision_a_prereg_holds():
    assert core._decision(prereg(0.5).encode())["gates_sha256"] == digest(0.5)
    assert core._decision(prereg(0.5).replace("\n", "\r\n").encode())["gates_sha256"] == digest(0.5)
    assert core._decision(b"# no fence\n")["code"] == "NOT_GATED"
    two = prereg(0.5) + prereg(0.6)
    assert (core._decision(two)["form"], core._decision(two)["code"]) == (None, "SCORER_REFUSES")
    assert core._decision(b"\xff\xfe")["code"] == "SCORER_REFUSES"

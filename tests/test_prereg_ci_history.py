# -*- coding: utf-8 -*-
"""prereg-ci B1: the git layer reproduces the design's census receipts on styxx's own history.

DESIGN 8, B1: "census reproduces R1 and RV1/RV2 through the module". The receipts were made at
ca347730 by scripts that live with the design, outside this repository; their outputs are copied
into ``conformance/prereg_ci/`` with CRLF read as LF (README.md, "The history the git layer
reproduces"). Each test here re-derives one of them through ``styxx.prereg_ci.git`` and
``styxx.prereg_ci.core`` alone and compares it with the copy line by line:

* R1 (``rescore_census_ca347730.txt``): every committed record naming a gates digest, re-scored
  from the PREREG blob at the commit that added it: 33 records, 33 re-score.
* RV1 (``seal_rule_ca347730.txt``): the add-commit rule against the landing rule over every gated
  PREREG: 52 gated, the add commit is the landing commit for 40, the file blob differs for 2,
  the gates digest for 1.
* RV2 (``landing_order_ca347730.txt``, over ``pairs_ca347730.txt``): the order commit on the
  landed decision digest is strictly before the receipt for 32 of 32 pairs, the order commit on
  the landed file blob for 31 (the realmodel smoke receipt is the miss), the add commit for 32.

Beside the copies, the landing commit and the add commit are re-derived with git's own porcelain
for every gated PREREG, and the landing rule's seal gives every record its recorded verdict and
digest. Numbers at head move as main moves; they are printed by the build report, not pinned. The
tests skip, saying why, where ca347730 is not in the object store or the clone is shallow.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402

_P1_FLAG = "--" + "tnerap-tsrif"[::-1]      # git's parent-one flag, built so this file does not spell it


def _have_rev() -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True)
    return r.returncode == 0


@pytest.fixture(scope="module")
def repo():
    if not _have_rev():
        pytest.skip(f"{REV} is not in the object store (a shallow or partial clone); the census is read from it")
    with G._Repo(ROOT) as r:
        if r.is_shallow():
            pytest.skip("a shallow clone holds no history to reproduce the census from")
        yield r


@pytest.fixture(scope="module")
def census(repo):
    """R1's records with their receipt and PREREG resolved, and the landing-rule seal of each PREREG."""
    pairs = G._pairs(repo, REV)
    repo.landing(sorted({r["prereg_path"] for r in pairs if "prereg_path" in r}), REV)
    seals = {r["prereg_path"]: G._seal(repo, r["prereg_path"], REV) for r in pairs if "prereg_path" in r}
    return pairs, seals


def _copy(name: str):
    return (SET / name).read_text(encoding="utf-8").splitlines()


def _porcelain(*args: str) -> str:
    return subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), *args], capture_output=True, check=True,
                          ).stdout.decode("utf-8")


@pytest.fixture(scope="module")
def gated(repo):
    """Every PREREG_*.md under papers/ at REV whose gates fence the scanner reads, in tree order."""
    out = []
    for path in repo.paths(REV):
        name = PurePosixPath(path).name
        if path.startswith("papers/") and name.startswith("PREREG_") and name.endswith(".md"):
            if core._decision(repo.read(REV, path))["form"] == "gates":
                out.append(path)
    return out


# ---- RV1 -----------------------------------------------------------------------------------------------

def test_rv1_the_two_seal_rules_line_by_line(repo, gated):
    landing = repo.landing(gated, REV)
    lines, same, blobs, digests = [], 0, 0, 0
    for f in gated:
        add, land = repo.add_commit(f, REV), landing[f]
        ba, bl = repo.read(add, f), repo.read(land, f)
        da, dl = core._decision(ba)["gates_sha256"], core._decision(bl)["gates_sha256"]
        same += add == land
        blobs += ba != bl
        digests += da != dl
        tag = "SAME_COMMIT" if add == land else "MERGED_IN"
        lines.append(f"{tag} add={add[:8]} landing={land[:8]} blob_differs={ba != bl} gates_differs={da != dl} {f}")
    lines.append(f"== gated PREREGs {len(gated)}; add commit is the landing commit {same}; "
                 f"file blob differs between the two rules {blobs}; gates digest differs {digests}")
    assert lines == _copy("seal_rule_ca347730.txt")
    assert (len(gated), same, blobs, digests) == (52, 40, 2, 1)
    untied = "papers/frequency-resonance/PREREG_pmnist_untied_2026_09_02.md"
    assert (repo.add_commit(untied, REV)[:8], landing[untied][:8]) == ("1cbc5997", "320b3032")


def test_rv1_add_and_landing_equal_gits_own_porcelain(repo, gated):
    landing = repo.landing(gated, REV)
    for f in gated:
        assert repo.add_commit(f, REV) == _porcelain("log", REV, "--diff-filter=A", "--format=%H", "--", f).split()[-1], f
        assert landing[f] == _porcelain("log", REV, _P1_FLAG, "--format=%H", "--", f).split()[-1], f
    assert repo.chain(REV) == _porcelain("rev-list", _P1_FLAG, REV).split()


# ---- RV2 -----------------------------------------------------------------------------------------------

def _pairs_file():
    return sorted({tuple(line[5:].strip().split(" <- ")) for line in _copy("pairs_ca347730.txt")
                   if line.startswith("PAIR ")})


def test_rv2_the_order_commits_line_by_line(repo):
    pairs = _pairs_file()
    lines, ok = [], Counter()
    landing = repo.landing([p for p, _ in pairs], REV)
    for p, r in pairs:
        L = landing[p]
        W, Gc = repo.order_commit(p, L, "blob"), repo.order_commit(p, L, "decision")
        A, R = repo.add_commit(p, REV), repo.add_commit(r, REV)
        w, a, g = (repo.is_strict_ancestor(x, R) for x in (W, A, Gc))
        ok.update(w=w, a=a, g=g)
        lines.append(f"G_before_R={g} W_before_R={w} A_before_R={a} G={Gc[:8]} W={W[:8]} A={A[:8]} L={L[:8]} "
                     f"R={R[:8]} {p.rsplit('/', 1)[-1]} <- {r.rsplit('/', 1)[-1]}")
    lines.append(f"== distinct pairs {len(pairs)}; landed-gates commit G strictly before receipt R: {ok['g']}; "
                 f"landed-bytes commit W strictly before R: {ok['w']}; add commit A strictly before R: {ok['a']}")
    assert lines == _copy("landing_order_ca347730.txt")
    assert (len(pairs), ok["g"], ok["w"], ok["a"]) == (32, 32, 31, 32)


def test_rv2_the_224_pair_and_the_named_miss(repo):
    p224 = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
    s = G._seal(repo, p224, REV)
    assert (s["landing"][:8], s["order"][:8], s["add"][:8]) == ("976b097f", "3598e59d", "3598e59d")
    assert s["blob_sha256"].startswith("2068e673") and s["gates_sha256"].startswith("578924f8")
    assert repo.add_commit("papers/frequency-resonance/pmnist_robustness_result.json", REV)[:8] == "237e7784"
    bridge = "papers/frequency-resonance/PREREG_realmodel_bridge_v3_2026_10_11.md"
    s = G._seal(repo, bridge, REV)
    R = repo.add_commit("papers/frequency-resonance/realmodel_bridge_v3_smoke.json", REV)
    assert (s["order_blob"][:8], R[:8]) == ("97727999", "227f433d")
    assert repo.is_strict_ancestor(s["order"], R) and not repo.is_strict_ancestor(s["order_blob"], R)


# ---- R1 ------------------------------------------------------------------------------------------------

def _receipt(repo, path):
    return json.loads(repo.read(REV, path).decode("utf-8"))


def test_r1_the_census_line_by_line(repo, census):
    pairs = census[0]
    lines, tally = [], Counter()
    for rec in pairs:
        assert "refusal" not in rec, rec
        pp, rp = rec["prereg_path"], rec["receipt_path"]
        A = repo.add_commit(pp, REV)
        pb = repo.read(A, pp)
        s = core.rescore(pb, _receipt(repo, rp))
        want = rec["recorded"]
        same_v, same_d = s["verdict"] == want.get("verdict"), s["gates_sha256"] == want.get("gates_sha256")
        status = "RESCORES" if same_v and same_d else "VERDICT_DIFFERS" if not same_v else "GATES_HASH_DIFFERS"
        R = repo.add_commit(rp, REV)
        order = ("prereg<receipt" if repo.is_strict_ancestor(A, R) else "SAME_COMMIT" if R == A else "NOT_ORDERED")
        n = len({b for _, b in repo.blob_history(pp, A, R) if b})
        out = [f"order={order}", f"prereg_blobs_seal..receipt={n}",
               f"head_blob_is_seal_blob={repo.read(REV, pp) == pb}",
               f"recorded_commit_is_add_commit={(rec['prereg_commit'] or '')[:8] == A[:8]}"]
        tally[status] += 1
        lines.append(" ".join([status, rec["record"], "|", str(want.get("verdict")), "|", " ".join(out)]))
    lines.append(f"TALLY {json.dumps(dict(tally), sort_keys=True)} records {len(pairs)}")
    assert lines == _copy("rescore_census_ca347730.txt")
    assert dict(tally) == {"RESCORES": 33}
    assert sorted({(r["prereg_path"], r["receipt_path"]) for r in pairs}) == _pairs_file()


def test_r1_under_the_landing_rule_every_record_reads_as_recorded(repo, census):
    """DESIGN 3.2(b)'s seal (the blob at L), C5's re-score and C6's compare: 33 of 33 agree, each
    receipt's order commit is strictly before its add commit, and the recorded digest is sealed."""
    pairs, seals = census
    for rec in pairs:
        pp, rp = rec["prereg_path"], rec["receipt_path"]
        s = seals[pp]
        rescored = core.rescore(s["bytes"], _receipt(repo, rp))
        assert core.compare(rescored, rec["recorded"], sealed=[s["gates_sha256"]]) == [], rec["record"]
        assert repo.is_strict_ancestor(s["order"], repo.add_commit(rp, REV)), rec["record"]
    assert len(pairs) == 33 and len(seals) == 31


def test_k2_every_receipt_finds_its_recorded_digest(repo, census):
    """PLAN K2: the receipt's own gates_sha256, else a protocol entry of a *.seal.json whose result
    resolves to it; 4 receipts carry their own, 28 are found in a sidecar, none is ambiguous."""
    by_receipt = {}
    for rec in census[0]:
        by_receipt.setdefault(rec["receipt_path"], set()).add(rec["recorded"]["gates_sha256"])
    sources = Counter()
    for rp, digests in sorted(by_receipt.items()):
        got = G._recorded(repo, REV, rp, _receipt(repo, rp))
        assert got is not None and {got["gates_sha256"]} == digests, rp
        sources["own" if got["source"] == rp else "sidecar"] += 1
    assert dict(sources) == {"own": 4, "sidecar": 28}


def test_the_reads_agree_with_styxx_sworn_gittree(repo, census):
    """DESIGN C2 reads through styxx.sworn.GitTree; the layer reads with one long-lived process and
    must give the same bytes for every seal and every receipt of the census."""
    from styxx.sworn import GitTree
    pairs, seals = census
    reads = {(s["landing"], pp) for pp, s in seals.items()} | {(repo.commit(REV), r["receipt_path"]) for r in pairs}
    for commit, path in sorted(reads):
        data, why = GitTree(ROOT, commit).blob(path)
        assert why == "ok" and data == repo.read(commit, path), (commit, path)
    assert len(reads) == 31 + 32

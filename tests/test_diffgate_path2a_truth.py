"""PATH-2a coverage (B), judged by truth (NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30,
NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30, NOTE_path2a_fifth_pass_2026_09_30; the later
passes' notes are named where their tests are).

Truth comes from each case's base/head file model (tests/_p2a_truth.py), never from the diff. A decided claim is
ATTRIBUTABLE when main's verdict is false by truth and a counterfactual variant of main without #97, without #121,
without #101, or without all three, gives the same claim a verdict that is not false. The bar: every attributable
claim is UNCHECKABLE on the branch -- at the raw door, at the git door, and in the port, judged there in the port's
own terms by the same four variants built from main's port. The counts are pinned, and so is the cost (right verdicts
lost, undecided verdicts abstained).

The cases are #161's reproductions and the PREREG_path2 reproductions (tests/fixtures/path2a_repros.json), and
400 generated cases per family at a pinned seed (tests/_p2a_cases.py): #97, #121, #121 with case outside ASCII, #101.
At the git door, the reproductions that carry their own `--name-status` are read through it.

`test_plants_are_refused` is the calibration: each plant is one anchored edit to the Python block, and each must be
caught by the pinned pairs, by this truth assertion, or by the abstain-only relation (both strict modes). Plants in
the port block must each make the port disagree with the Python where main's two ports agree.
"""
from __future__ import annotations

import collections
import functools
import json
import os
import shutil
import subprocess

import pytest

import styxx.diffgate as N
from tests import _p2a_cases as C
from tests import _p2a_ref as R
from tests import _p2a_truth as T

SEED, PER_FAMILY = 20260930, 400
NODE = shutil.which("node")
DECIDED = ("VERIFIED", "CONTRADICTED")

# Pinned after review (Python 3.12 and 3.14 agree). A change to the overlay, the oracle or the generator moves them.
# Pass 4 (NOTE_path2a_fourth_pass_2026_09_30, B-1): the `shape` rule withholds 43 more right verdicts here, in both
# ports, and no attributable verdict moves: the generator writes "Only touches cfg and .cfg/app/." beside files that
# all lie under the second prefix. Pass 6 (NOTE_path2a_sixth_pass_2026_09_30, B-1): the unchanged lines withhold 16
# more false verdicts main gives on a test or a name defined again (outside the committed V101's attribution, so under
# "false other"), 6 right CONTRADICTEDs (a test the removed pairing already counts as changed moved into a class, beside
# a redefinition) and 4 undecided, in each port; C-1 moves nothing here. Pass 7 (NOTE_path2a_seventh_pass_2026_09_30,
# C-1): the case-pair clause keeps 6 CONTRADICTEDs in each port where two changed paths differ in one letter outside
# ASCII and a claimed count lies in the range the two mains may count (4 counts, 2 of them false through neither defect
# and 2 right; 2 right scopes); no attributable verdict moves. The tighter trigger (B-1) moves nothing here. Pass 8
# (NOTE_path2a_eighth_pass_2026_10_01, B-1): a count claim in [#CA, #A] beside two paths that differ only in case
# outside ASCII is withheld in both ports, whatever its verdict (`case_count`: 26 false verdicts main gives by merging
# the two paths' case, outside the three defects, 13 CONTRADICTED and 13 VERIFIED, and 4 right CONTRADICTEDs), and the
# CONTRADICTEDs the case-pair clause kept are decided (2 false counts withheld as `count`, 2 right scopes as `shape`):
# +34 withheld, +28 false other, +6 right lost, in each port; no attributable verdict moves. B-2 moves nothing here.
# Pass 9 (NOTE_path2a_ninth_pass_2026_10_04): the C-1 switch is removed, and no figure here moves. The switch did keep
# attributable verdicts under a U+FEFF before the summary, or a sentence naming styxx beside U+2028, which main's two
# ports read as they read the plain summary: the transformed worlds at the end of this module.
PINNED = {"cases": 1706, "main raises": 0, "decided": 7614, "false": 1535, "attributable": 1206, "abstained": 1591,
          "attributable abstained": 1206, "right": 5897, "right lost": 253, "undecided": 182,
          "undecided abstained": 39, "unjudged": 0, "unjudged abstained": 0, "false other": 329,
          "false other abstained": 93}
# The port judged in its own terms: 1,100 attributable claims, of which 942 carry the same main record in Python and
# 158 are read otherwise by Python's main (paths and names outside ASCII, which the two templates extract apart).
PINNED_PORT = {"cases": 1706, "main raises": 0, "decided": 7393, "abstained": 1452, "right": 5739, "right lost": 177,
               "false": 1385, "attributable": 1100, "attributable abstained": 1100, "false other": 285,
               "false other abstained": 49, "undecided": 269, "undecided abstained": 126}
PINNED_GIT = {"cases": 96, "main raises": 0, "with R, C or T": 16, "decided": 63, "abstained": 14, "false": 17,
              "attributable": 14, "attributable abstained": 14, "right": 46, "right lost": 0, "false other": 3,
              "false other abstained": 0}


@functools.lru_cache(maxsize=None)
def variants() -> dict:
    return {v: R.module_from(T.variant_source(R.main_source(), v), "_p2a_variant_" + v) for v in T.VARIANTS}


def judge(model, main_claims, variant_claims) -> list:
    """(index, main's claim, truth, the variants that do not read it false) for each decided claim of main."""
    out = []
    for i, x in enumerate(main_claims):
        if x["verdict"] not in DECIDED:
            continue
        t = T.truth(model, x["kind"], x["detail"])
        who = [k for k, vs in variant_claims.items() if T.wrong(x["verdict"], t) and vs is not None
               and not T.wrong(vs[i]["verdict"], t)]
        out.append((i, x, t, who))
    return out


@pytest.fixture(scope="module")
def world():
    """Every case, main's record, each decided claim's truth and whether main's false verdict is attributable."""
    M = R.main_module()
    V = variants()
    cases = [c for c in R.repro_cases() if c.get("model")] + C.families(SEED, PER_FAMILY)
    out = []
    for it in cases:
        try:
            a = M.gate_diff_text(it["summary"], it["diff"]).to_dict()
        except Exception:
            out.append((it, None, []))
            continue
        vs = {}
        for k, mod in V.items():
            try:
                vs[k] = mod.gate_diff_text(it["summary"], it["diff"]).to_dict()["claims"]
            except Exception:
                vs[k] = None
                continue
            assert [(c["kind"], c["text"], c["detail"]) for c in vs[k]] == \
                   [(c["kind"], c["text"], c["detail"]) for c in a["claims"]], (k, it["id"])
        out.append((it, a, judge(it["model"], a["claims"], vs)))
    return out


def evaluate(mod, world):
    counts = collections.Counter({k: 0 for k in ("cases", "main raises", "decided", "false", "attributable",
                                                 "abstained", "attributable abstained", "right", "right lost",
                                                 "undecided", "undecided abstained", "unjudged", "unjudged abstained",
                                                 "false other", "false other abstained")})
    misses = []
    for it, a, claims in world:
        counts["cases"] += 1
        if a is None:
            counts["main raises"] += 1
            continue
        b = mod.gate_diff_text(it["summary"], it["diff"]).to_dict()
        tally(counts, misses, it, claims, b["claims"])
    return dict(counts), misses


def tally(counts, misses, it, claims, new_claims):
    for i, x, t, who in claims:
        ab = new_claims[i]["verdict"] == "UNCHECKABLE"
        counts["decided"] += 1
        counts["abstained"] += ab
        if t is None or t == "?":
            key = "unjudged" if t is None else "undecided"
            counts[key] += 1
            counts[key + " abstained"] += ab
        elif T.wrong(x["verdict"], t):
            counts["false"] += 1
            key = "attributable" if who else "false other"
            counts[key] += 1
            counts[key + " abstained"] += ab
            if who and not ab:
                misses.append({"id": it["id"], "claim": i, "kind": x["kind"], "main": x["verdict"],
                               "text": x["text"][:100], "why": x["why"][:100], "fixed by": who,
                               "style": it.get("style"), "diff": it["diff"][:300]})
        else:
            counts["right"] += 1
            counts["right lost"] += ab


def test_every_attributable_false_verdict_abstains(world):
    counts, misses = evaluate(N, world)
    print("PATH-2a truth:", json.dumps(counts, indent=1))
    for m in misses:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert misses == []
    assert counts == PINNED


def _port_records(port_path, items, tmp_path, tag):
    out = tmp_path / f"{tag}.json"
    r = subprocess.run([NODE, str(R.DIFFERENTIAL / "check_path2a.js"), "--records", str(port_path),
                        str(tmp_path / "in.json"), str(out)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return {d["id"]: d["rec"] for d in json.loads(out.read_text(encoding="utf-8"))}


def _port_truth(cases, tmp_path):
    """The port judged in its own terms over cases {id, summary, diff, model}: main's port and the four variants built
    from it read each case, truth judges the port's own claims (its own details), and `tally` counts what this port
    withholds. Returns (counts, misses, the judged claims per case, None where main's port raises)."""
    if NODE is None:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("node is not on PATH under CI; the port half of the coverage check did not run")
        pytest.skip("node is not on PATH; the port half of the coverage check cannot run here")
    items = [{"id": str(k), "summary": it["summary"], "diff": it["diff"]} for k, it in enumerate(cases)]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=True), encoding="utf-8")
    main_port = _port_records(R.main_port_path(tmp_path), items, tmp_path, "main")
    new_port = _port_records(R.PORT, items, tmp_path, "new")
    var = {}
    for v in T.PORT_VARIANTS:
        p = tmp_path / f"diffgate_{v}.js"
        p.write_bytes(T.port_variant_source(R.main_port_source(), v).encode("utf-8"))
        var[v] = _port_records(p, items, tmp_path, v)
    counts = collections.Counter({k: 0 for k in ("cases", "main raises")})
    misses, rows = [], []
    for k, it in enumerate(cases):
        counts["cases"] += 1
        jm = main_port[str(k)]
        if "error" in jm:
            counts["main raises"] += 1
            rows.append(None)
            continue
        vs = {}
        for v in T.PORT_VARIANTS:
            rec = var[v][str(k)]
            vs[v] = None if "error" in rec else rec["claims"]
            if vs[v] is not None:
                assert [(c["kind"], c["text"], c["detail"]) for c in vs[v]] == \
                       [(c["kind"], c["text"], c["detail"]) for c in jm["claims"]], (v, it["id"])
        judged = judge(it["model"], jm["claims"], vs)
        tally(counts, misses, it, judged, new_port[str(k)]["claims"])
        rows.append(judged)
    return dict(counts), misses, rows


def test_every_attributable_false_verdict_abstains_in_the_port(world, tmp_path):
    """In the port's own terms (B-2): main's port and the four variants built from it read each case, truth judges
    the port's own claims (its own details), and every claim the port's variants show attributable must be
    UNCHECKABLE in this port."""
    counts, misses, rows = _port_truth([it for it, _a, _c in world], tmp_path)
    same_as_python = collections.Counter()
    for (_it, a, _claims), judged in zip(world, rows):
        if a is not None and judged is not None:
            for i, x, _t, who in judged:
                if who:
                    same = i < len(a["claims"]) and all(a["claims"][i][f] == x[f]
                                                        for f in ("kind", "verdict", "why", "text", "detail"))
                    same_as_python["same main record in Python" if same else "Python's main reads it otherwise"] += 1
    print("port truth:", json.dumps(counts, indent=1), dict(same_as_python))
    for m in misses:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert misses == []
    assert counts == PINNED_PORT


# P-2 (NOTE_path2a_ninth_pass_2026_10_04): file models, written by hand from each case's diff, for the three of #161's
# reproductions on which the port's main alone gives a false tests verdict through #101: it counts the `def test_` that
# follows a U+FEFF (JavaScript's white space, not CPython's) beside a changed test. The fixture carries no model for
# them, so the pinned port truth above never judged them, and up to the eighth pass the C-1 switch kept the port's
# false CONTRADICTED on each (the two views count apart there).
_T_A = "def test_a():\n    pass\n"
PORT_161_MODELS = {
    "path2:r1-a-bom-on-a-changed-test-does-not-hide-a-new-one": {
        "base": {"tests/test_bom.py": _T_A},
        "head": {"tests/test_bom.py": chr(0xFEFF) + _T_A + "def test_b():\n    pass\n"}},
    "path2:r1-a-bom-on-a-changed-test-beside-two-new-ones": {
        "base": {"tests/test_bom.py": _T_A},
        "head": {"tests/test_bom.py": chr(0xFEFF) + _T_A + "def test_b():\n    pass\ndef test_c():\n    pass\n"}},
    "path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks": {
        "base": {"tests/test_a.py": "def test_old():\n    pass\n"},
        "head": {"tests/test_a.py": "def test_old(tmp_path):\n    pass\n",
                 "tests/test_b.py": chr(0xFEFF) + "def test_new():\n    pass\n"}},
}


def test_the_port_withholds_its_attributable_verdicts_on_161s_bom_reproductions(tmp_path):
    """Judged by truth, in the port's own terms: four of the port's verdicts on these three inputs are false through
    #101 (a counted `def test_` is a changed test), and each is withheld; the port's three other decided verdicts
    there are right, and one of them is lost (y2's count of 0 against one new test, `split`)."""
    cases = [dict(c, model=PORT_161_MODELS[c["id"]]) for c in R.repro_cases() if c["id"] in PORT_161_MODELS]
    assert len(cases) == 3
    counts, misses, _rows = _port_truth(cases, tmp_path)
    print("port truth on #161's BOM reproductions:", json.dumps(counts))
    assert misses == []
    assert (counts["attributable"], counts["attributable abstained"]) == (4, 4), counts
    assert (counts["right"], counts["right lost"]) == (3, 1), counts


def test_every_attributable_false_verdict_abstains_at_the_git_door(monkeypatch):
    """Integration-2 and B-1: the reproductions that carry their own `--name-status` are read through the git door
    (each module's `_git` answers from the case), by main, the four variants and the branch: the relation in both
    strict modes, the lockstep of the overlay's `--name-status` mirror, and truth coverage there."""
    M = R.main_module()
    V = variants()
    cases = [c for c in R.repro_cases() if c.get("model") and c.get("name_status")]
    counts = collections.Counter({k: 0 for k in ("cases", "main raises", "with R, C or T")})
    # (none of #161's recorded `--name-status` carries a rename or a copy: 16 carry T entries. Renames and copies are
    # read at the real git door in tests/test_diffgate_path2a.py, which also refuses the reviewer's rename plant.)
    misses, broken = [], []
    for it in cases:
        fake = R.fake_git(it["name_status"], it["diff"])
        for mod in [M, N, *V.values()]:
            monkeypatch.setattr(mod, "_git", fake)
        counts["cases"] += 1
        counts["with R, C or T"] += any(line[:1] in "RCT" for line in it["name_status"].splitlines() if line)
        want = {}
        for line in it["name_status"].splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                want[M._norm(parts[-1])] = parts[0][:1]
        assert list(N._p2a_build(N._p2a_regs_git(it["name_status"]), N._norm).items()) == list(want.items()), it["id"]
        try:
            a = M.gate_diff(it["summary"], "(repo)", "base", "head").to_dict()
        except Exception:
            counts["main raises"] += 1
            continue
        mine = {}
        for strict in (False, True):
            ref = M.gate_diff(it["summary"], "(repo)", "base", "head", strict=strict).to_dict()
            mine[strict] = N.gate_diff(it["summary"], "(repo)", "base", "head", strict=strict).to_dict()
            bad = R.relation(ref, mine[strict], strict, N._P2A_PHRASES)
            if bad:
                broken.append((it["id"], strict, bad))
        if R.strict_alike(mine[False], mine[True]):
            broken.append((it["id"], "strict"))
        vs = {}
        for k, mod in V.items():
            try:
                vs[k] = mod.gate_diff(it["summary"], "(repo)", "base", "head").to_dict()["claims"]
            except Exception:
                vs[k] = None
        tally(counts, misses, it, judge(it["model"], a["claims"], vs), mine[False]["claims"])
    print("git door truth:", json.dumps(dict(counts), indent=1))
    for m in misses:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert broken == [] and misses == []
    assert dict(counts) == PINNED_GIT


# ---- calibration: plants --------------------------------------------------------------------------------------------

PY_PLANTS = [
    ("no divergence guard", "        return self.name_status is None and self._get(\"div\"",
     "        return False and self._get(\"div\""),
    ("the port-only line view", "    v0 = _p2a_view(fine)\n", "    v0 = _p2a_view(_p2a_lines(diff_text, _P2A_COARSE))\n"),
    ("count CONTRADICTED interval shrunk", 'return ("count", "#121") if lo <= n <= hi else None',
     'return ("count", "#121") if lo <= n < hi else None'),
    ("V97 allows a base name for a directory claim",
     '    if "/" in c and P2A_DIRECTORY_BASENAME_ABSTAINS:\n        return None\n', "    if False:\n        return None\n"),
    ("V121 dropped", '    v121 = ok(_p2a_resolve(f, "K", ck, False))', "    v121 = True"),
    ("V97+V121 dropped", '    vboth = ok(_p2a_resolve(f, "K", ck, True))', "    vboth = True"),
    ("tests interval lower bound + 1", "g - min(g, p) <= n <= g", "g - min(g, p) + 1 <= n <= g"),
    ("symbol rule off", '    if f.defines(name):\n        return "symbol", "#101"\n',
     '    if False:\n        return "symbol", "#101"\n'),
    ("def separator restricted to space and tab", '_P2A_COARSE_RUN = re.compile("[\\x00-\\x20\\x7f-\\U0010ffff]*")',
     '_P2A_COARSE_RUN = re.compile("[ \\t]*")'),
    ("names paired by their full run",
     "                e = _P2A_WORD_RUN.match(line, r).end()\n                if _p2a_wide_name(line, j, r, e):\n"
     "                    if nfkc:",
     "                e = _P2A_NAME_RUN.match(line, r).end()\n                if _p2a_wide_name(line, j, r, e):\n"
     "                    if nfkc:"),
    ("the scope rule's V121 reading dropped", '    if got["K", False][2] != under:', "    if False:"),
    ("U2 dropped", '    return want is not None and f._get(("merged", space, b, want), lambda: any(',
     '    return False and f._get(("merged", space, b, want), lambda: any('),
    ("a reason naming the wrong verdict", '            c.why = f"{c.verdict} withheld by PATH-2a ({tag}): ',
     '            c.why = f"VERIFIED withheld by PATH-2a ({tag}): '),
    # pass 2
    ("the overlay skipped under --strict",
     "    pending = {i: c for i, c in enumerate(g.claims) if (c.kind, c.verdict) in _P2A_REACH}",
     "    pending = {i: c for i, c in enumerate(g.claims) if (c.kind, c.verdict) in _P2A_REACH and not strict}"),
    ("split read as keep", '    if any(fires):\n        return "split", "#101"\n', ""),
    ("extract off for paths", "    if _p2a_extract(f, c, claimed, want):", "    if False:"),
    ("the port's reading of a wide path taken as never verified",
     "                if b and not _P2A_WIDE.search(b) and (want is None or regs[i][1] == want):",
     "                if False:"),
    ("extract off for scopes", "    if _p2a_scope_doubt(f, d):", "    if False:"),
    ("extract off for symbols",
     '    if _P2A_WIDE.search(name) or (not c.detail.get("declared") and _p2a_in_runs(f, name, "name")):',
     "    if False:"),
    ("odd reads every drive letter",
     '    return q == "." or q.endswith("/.") or (len(q) >= 2 and q[1] == ":" and (len(q) == 2 or q[2] != "/"))',
     '    return q == "." or q.endswith("/.") or (len(q) >= 2 and q[1] == ":")'),
    ("dot_earliest read as dot", '        return ("dot_earliest" if vboth else "dot"), "#121"',
     '        return "dot", "#121"'),
    # pass 3 (NOTE_path2a_third_pass_2026_09_30)
    ("the summary never read", "        self.summary = _P2A_EMOJI_RX.sub(_P2A_EMOJI_AS, summary)",
     '        self.summary = ""'),
    ("the scope's second prefix never read",
     '            ps = [fm(x).rstrip("/.") for x in (prefixes if shaped else prefixes[:1])]',
     '            ps = [fm(x).rstrip("/.") for x in prefixes[:1]]'),
    # pass 4 (NOTE_path2a_fourth_pass_2026_09_30)
    ("V121's reading of the leading prefix's shape dropped", '    if d.get("prefix2") and not got["K", False][0]:',
     "    if False:"),
    ("no extract guard for counts",
     '    if _P2A_DIGITS.fullmatch(claimed) is None or (not c.detail.get("declared") and _p2a_in_runs(f, claimed, "count")):',
     "    if False:"),
    ("symbol sites read anywhere in a removed line",
     "            for j, r in _p2a_anchored(line):\n                e = _P2A_WORD_RUN.match(line, r).end()\n"
     "                if _p2a_wide_name(line, j, r, e):\n                    return out, True",
     '            for j, r in _p2a_sites(line, "def") + _p2a_sites(line, "class"):\n'
     "                e = _P2A_WORD_RUN.match(line, r).end()\n"
     "                if _p2a_wide_name(line, j, r, e):\n                    return out, True"),
    # pass 5 (NOTE_path2a_fifth_pass_2026_09_30): each new rule, dropped
    ("the count seam never read", '        return self._get("seam", lambda: _p2a_seam(self.summary))',
     "        return False"),
    ("the removed lines no view reads dropped", '        return self._get("joined", lambda: _p2a_joined(self.diff_text))',
     '        return self._get("joined", lambda: [])'),
    ("test names read through NFKC paired by their ASCII run",
     "                    if nfkc:\n                        return names, True",
     "                    if False:\n                        return names, True"),
    ("symbol names read through NFKC ignored",
     "                if _p2a_wide_name(line, j, r, e):\n                    return out, True",
     "                if False:\n                    return out, True"),
    ("a counted name read through NFKC paired by its ASCII run",
     "            out.append(None if _P2A_WIDE.match(line, e) else line[r + 4:e])",
     "            out.append(line[r + 4:e])"),
    ("V97's suffix tier never read", '    i = f.ends(space, "key", c)[0]', "    i = -1"),
    ("the case doubt's base-name fact dropped",
     "        if len(bases) > 1 or (bases and _p2a_base(cf) not in bases):\n            return True",
     "        if False:\n            return True"),
    ("the case doubt's suffix fact dropped",
     '        if f.ends(space, "wild", cw)[1] != f.ends(space, "fold", cf)[1]:\n            return True',
     "        if False:\n            return True"),
    ("pieces after a lone CR dropped",
     '        pieces = _p2a_lines(line[1:], _P2A_CR) if "\\r" in line else [line[1:]] if len(line) > 1 else []',
     "        pieces = [line[1:]] if len(line) > 1 else []"),
    ("continuations never joined","            hit = hit or head == \"-\"\n",
     "            hit = hit or head == \"-\"\n            acc = acc[-1:]\n"),
    # pass 6 (NOTE_path2a_sixth_pass_2026_09_30): each new rule, dropped
    ("the unchanged lines dropped",
     '        return self._get("unchanged", lambda: _p2a_context(self.diff_text, self.fine())\n'
     '                         + _p2a_joined(self.diff_text, " "))',
     '        return self._get("unchanged", lambda: [])'),
    ("a name defined again kept",
     '    if f.redefines(name):                 # B-1 (NOTE_path2a_sixth_pass_2026_09_30)\n        return "again", "#101"\n',
     ""),
    # pass 7 (NOTE_path2a_seventh_pass_2026_09_30): each new rule that is still there, dropped or loosened (its
    # case-pair clause and ordered triggers went at pass 8, and its claimed-name clause at pass 9, with the switch)
    ("O-11 off", "        self.summary = _P2A_EMOJI_RX.sub(_P2A_EMOJI_AS, summary)", "        self.summary = summary"),
    ("a token that is a piece not looked up", "    got = {w for w in words if w in pieces}\n", "    got = set()\n"),
    # pass 8 (NOTE_path2a_eighth_pass_2026_10_01): each new rule that is still there, dropped or loosened (its
    # line-break clause and its windows went at pass 9, with the switch)
    ("B-1's case-count clause dropped",
     "    if ca < a and (_P2A_DIGITS.fullmatch(claimed) is None or ca <= _p2a_int(claimed) <= a):",
     "    if False:"),
    ("B-1's case classes read as one placeholder",
     "            wide = any(_P2A_NEVER_RX.match(x) is None and x not in _P2A_CASE for x in col)",
     "            wide = True"),
    ("B-1's scripts without case read as cased", "                if _P2A_NEVER_RX.match(k[i]) is None:",
     "                if True:"),
    # pass 9 (NOTE_path2a_ninth_pass_2026_10_04, I-2): the U+2028 and U+2029 starts of the two definition readers, which
    # passed every test when dropped; the cross-port cases P9-a-removed-def-, P9-an-unchanged-def- and P9-a-test-after-
    # hold them now
    ("the counted tests read without the U+2028 and U+2029 starts", "        r = lead.match(line, b).end()\n",
     "        if b:\n            continue\n        r = lead.match(line, b).end()\n"),
    ("the definitions read without the U+2028 and U+2029 starts", "        p = _P2A_COARSE_RUN.match(line, b).end()\n",
     "        if b:\n            continue\n        p = _P2A_COARSE_RUN.match(line, b).end()\n"),
]
# Plants that cannot change a record, said so rather than hidden: none this pass. Pass 2's one (a clause that never
# decided alone) went with the per-set comparison it sat behind (NOTE_path2a_third_pass_2026_09_30, B-2).
EQUIVALENT: set = set()


def _planted(k, old, new):
    text = R.lf(R.INSTRUMENT)
    block = R.py_block(text)
    assert block.count(old) == 1, old
    return R.module_from(text.replace(block, block.replace(old, new)), f"_p2a_plant_{k}")


def _pairs_catch(mod):
    for p in json.loads((R.DIFFERENTIAL / "path2a_pairs.json").read_text(encoding="utf-8")):
        g = mod.gate_diff_text(p["summary"], p["diff"])
        width = len(p["expect"]["claims"][0]) if p["expect"]["claims"] else 3
        rows = [[c.kind, c.verdict, c.why][:width] for c in g.claims]
        if rows != p["expect"]["claims"] or g.verdict != p["expect"]["verdict"]:
            return "pinned pair " + p["id"]
    # pass 5 (NOTE_path2a_fifth_pass_2026_09_30): the cross-port pins' Python decisions, some of them on claims only
    # main's Python reads (a path outside ASCII), which no pinned pair can hold for both ports
    from tests.test_diffgate_path2a import XPORT_CASES
    for cid, s, d, want_py, *_rest in XPORT_CASES:
        got = [(c.verdict, R.phrase_key(c.why, N._P2A_PHRASES)) for c in mod.gate_diff_text(s, d).claims]
        if got != want_py:
            return "cross-port pin " + cid
    return None


def _relation_catch(mod, M, inputs):
    for s, iid, summary, diff in inputs:
        mine = {}
        for strict in (False, True):
            try:
                a = M.gate_diff_text(summary, diff, strict=strict).to_dict()
            except Exception:
                break
            mine[strict] = mod.gate_diff_text(summary, diff, strict=strict).to_dict()
            if R.relation(a, mine[strict], strict, N._P2A_PHRASES):
                return f"relation on {s}:{iid} (strict {strict})"
        if len(mine) == 2 and R.strict_alike(mine[False], mine[True]):
            return f"--strict moved a claim on {s}:{iid}"
    return None


def _same_everywhere(mod, world, inputs):
    for it, _a, _c in world:
        if _a is not None and mod.gate_diff_text(it["summary"], it["diff"]).to_dict() != \
                N.gate_diff_text(it["summary"], it["diff"]).to_dict():
            return False
    for _s, _i, summary, diff in inputs:
        try:
            want = N.gate_diff_text(summary, diff).to_dict()
        except Exception:
            continue
        if mod.gate_diff_text(summary, diff).to_dict() != want:
            return False
    return True


def test_plants_are_refused(world):
    M = R.main_module()
    inputs = R.inputs()
    caught = {}
    for k, (name, old, new) in enumerate(PY_PLANTS):
        mod = _planted(k, old, new)
        why = _pairs_catch(mod)
        if why is None:
            _counts, misses = evaluate(mod, world)
            why = f"truth: {len(misses)} attributable claim(s) kept" if misses else None
        if why is None:
            why = _relation_catch(mod, M, inputs)
        if why is None and name in EQUIVALENT:
            assert _same_everywhere(mod, world, inputs), name
            why = "equivalent: identical records on every committed input and truth case"
        caught[name] = why
    print("plants:", json.dumps(caught, indent=1))
    assert all(caught.values()), [n for n, w in caught.items() if not w]


JS_PLANTS = [
    ("no divergence guard", '    divergent: () => get("div", () => _p2aDivergent(diffText)),',
     "    divergent: () => false,"),
    ("one line view", "  return [_p2aView(_p2aLines(diffText, _P2A_FINE)), v1];", "  return [v1, v1];"),
    ("count interval shrunk", 'return (lo <= n && n <= hi) ? ["count", "#121"] : null;',
     'return (lo <= n && n < hi) ? ["count", "#121"] : null;'),
    # pass 2
    ("no split rule", '  if (fires.some(x => x)) return ["split", "#101"];\n', ""),
    ("no extract for paths", '  if (_p2aExtract(f, c, claimed, want)) return ["extract", "#97, #121"];\n', ""),
    ("the port reading its count as CPython's", "const _P2A_OWN = 1;", "const _P2A_OWN = 0;"),
    # pass 3 (NOTE_path2a_third_pass_2026_09_30): C-2 and C-1, each as pass 2 read it
    ("the port reads the claimed number from its own reason", '  let k = _P2A_DIGITS.exec(detail.n || "");',
     "  let k = null;"),
    ("the port reads a scope claim's text", '  if (_p2aScopeDoubt(f, d)) return ["extract", "#121"];',
     '  if (_p2aScopeDoubt(f, d) || c.text.length >= 160) return ["extract", "#121"];'),
    # pass 4 (NOTE_path2a_fourth_pass_2026_09_30): each new rule, dropped from the port alone
    ("no extract guard for counts in the port",
     '  if (!_P2A_DIGITS.test(claimed) || (!c.detail.declared && _p2aInRuns(f, claimed, "count"))) '
     'return ["extract", "#121"];\n', ""),
    ("no shape rule in the port", '  if (d.prefix2 && !got.get("K|false")[0]) return ["shape", "#121"];\n', ""),
    # pass 5 (NOTE_path2a_fifth_pass_2026_09_30): each new rule, dropped from the port alone
    ("no count seam in the port", '    seam: () => get("seam", () => _p2aSeam(f.summary)),', "    seam: () => false,"),
    ("no removed lines beyond the views in the port", '    joined: () => get("joined", () => _p2aJoined(diffText)),',
     "    joined: () => [],"),
    ("no NFKC pairing in the port",
     "          if (nfkc) return [names, true];   // every counted site pairs now\n", ""),
    ("no NFKC symbols in the port",
     "        if (_p2aWideName(line, j, r, e)) return [out, true];   // every claimed name is defined now\n", ""),
    # pass 6 (NOTE_path2a_sixth_pass_2026_09_30): each new rule, dropped from the port alone
    ("no unchanged lines in the port",
     '    unchanged: () => get("unchanged", () => _p2aContext(diffText, f.coarse()).concat(_p2aJoined(diffText, " "))),',
     "    unchanged: () => [],"),
    ("no again rule in the port", '  if (f.redefines(name)) return ["again", "#101"];   // B-1 (NOTE_path2a_sixth_pass_2026_09_30)\n',
     ""),
    # pass 7 (NOTE_path2a_seventh_pass_2026_09_30): each new rule that is still there, dropped from the port alone
    ("no O-11 in the port", "    summary: String(summaryText).split(_P2A_EMOJI_RX).join(_P2A_EMOJI_AS),",
     "    summary: String(summaryText),"),
    ("no piece lookup in the port", "  const got = new Set([...words].filter(w => pieces.has(w)));", "  const got = new Set();"),
    # pass 8 (NOTE_path2a_eighth_pass_2026_10_01): each new rule that is still there, dropped from the port alone
    ("no case-count clause in the port",
     '  if (ca < a && (!_P2A_DIGITS.test(claimed) || (ca <= _p2aInt(claimed) && _p2aInt(claimed) <= a))) return ["case_count", "#121"];\n',
     ""),
    ("the port's case classes read as one placeholder", "const read = (ch, i) => (w[i] !== ", "const read = (ch, i) => (true || w[i] !== "),
    # pass 9 (NOTE_path2a_ninth_pass_2026_10_04, I-2): the U+2028 and U+2029 starts, dropped from the port alone
    ("the port's counted tests without the U+2028 and U+2029 starts", "    const r = _p2aRunEnd(line, b, lead);\n",
     "    if (b) continue;\n    const r = _p2aRunEnd(line, b, lead);\n"),
    ("the port's definitions without the U+2028 and U+2029 starts", "    const p = _p2aRunEnd(line, b, _p2aCoarseUnit);\n",
     "    if (b) continue;\n    const p = _p2aRunEnd(line, b, _p2aCoarseUnit);\n"),
]


@pytest.mark.parametrize("name,old,new", JS_PLANTS)
def test_port_plants_make_the_ports_disagree(name, old, new, tmp_path):
    """Each plant must split the ports on a claim main's two ports give the same kind, verdict and detail (the key a
    decision reads, NOTE_path2a_third_pass_2026_09_30)."""
    if NODE is None:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("node is not on PATH under CI; the port plants did not run")
        pytest.skip("node is not on PATH; the port plants cannot run here")
    text = R.lf(R.PORT)
    block = R.js_block(text)
    assert block.count(old) == 1, old
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(block, block.replace(old, new)).encode("utf-8"))
    # pass 6 (NOTE_path2a_sixth_pass_2026_09_30): the cross-port pins too, some of whose claims only one port reads
    from tests.test_diffgate_path2a import XPORT_CASES
    items = [x for x in R.inputs(fuzz=False)] + [("xport", x[0], x[1], x[2]) for x in XPORT_CASES]
    (tmp_path / "in.json").write_text(json.dumps([{"id": str(i), "summary": x[2], "diff": x[3]}
                                                  for i, x in enumerate(items)], ensure_ascii=True), encoding="utf-8")
    ref = R.main_port_path(tmp_path)
    r = subprocess.run([NODE, str(R.DIFFERENTIAL / "check_path2a.js"), "--decisions", str(ref),
                        str(tmp_path / "in.json"), str(tmp_path / "out.json"), str(planted)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    js = {d["id"]: d for d in json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))}
    M = R.main_module()
    new_disagreements = 0
    for i, (_s, _iid, summary, diff) in enumerate(items):
        j = js[str(i)]
        if "error" in j["main"]:
            continue
        try:
            a, b = M.gate_diff_text(summary, diff).to_dict(), N.gate_diff_text(summary, diff).to_dict()
        except Exception:                        # main's Python raises where its port does not
            continue
        for k, (x, jx) in enumerate(zip(a["claims"], j["main"]["claims"])):
            if all(x[f] == jx[f] for f in ("kind", "verdict", "detail")):
                y, jy = b["claims"][k], j["new"]["claims"][k]
                new_disagreements += (y["verdict"], R.phrase_key(y["why"], N._P2A_PHRASES)) != \
                    (jy["verdict"], R.phrase_key(jy["why"], N._P2A_PHRASES))
    if new_disagreements == 0:
        # A plant of the port's own view (_P2A_OWN) moves a decision only where the two line views count `def test_`
        # sites apart, and there main's two ports mostly give the tests claim different verdicts, so it may split no
        # claim both mains read alike on these inputs. It is caught where it moves the port's own decision on a claim
        # only its main decides so: PORT_ONLY_CASES, against the real port (NOTE_path2a_sixth_pass_2026_09_30).
        (tmp_path / "own.json").write_text(json.dumps(PORT_ONLY_CASES, ensure_ascii=False), encoding="utf-8")
        for port, tag in ((R.PORT, "real"), (planted, "planted")):
            r = subprocess.run([NODE, str(R.DIFFERENTIAL / "check_path2a.js"), "--records", str(port),
                                str(tmp_path / "own.json"), str(tmp_path / f"own_{tag}.json")],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
        real = json.loads((tmp_path / "own_real.json").read_text(encoding="utf-8"))
        mine = json.loads((tmp_path / "own_planted.json").read_text(encoding="utf-8"))
        new_disagreements = sum(json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True) for x, y in zip(real, mine))
    assert new_disagreements > 0, name


# Claims only one main decides so (NOTE_path2a_sixth_pass_2026_09_30): a created test after U+FEFF, which the port
# counts and CPython does not, so "Added 2 tests." is VERIFIED in the port only; the real port withholds it (`tests`: a
# changed test is among the two).
PORT_ONLY_CASES = [
    {"id": "own-count-view", "summary": "Added 2 tests.",
     "diff": ("diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
              "@@ -1,2 +1,2 @@\n-def test_a():\n-    assert 0\n+def test_a():\n+    assert 1\n"
              "diff --git a/tests/test_n.py b/tests/test_n.py\nnew file mode 100644\n--- /dev/null\n"
              "+++ b/tests/test_n.py\n@@ -0,0 +1,2 @@\n+\ufeffdef test_new():\n+    pass\n")},
]


# ---- pass 5: the fourth review's #101 reproductions (continuation, lone CR, NFKC), at three doors ------------------

def test_the_fifth_reviews_definition_reproductions(monkeypatch, tmp_path):
    """B-1 and B-2 (NOTE_path2a_fifth_pass_2026_09_30): a definition CPython reads across a backslash continuation, after
    a lone CR inside one git line, or through NFKC. main's verdict on each is false by truth, and false only through
    #101 as CPython reads definitions: the ast-paired V101 (tests/_p2a_truth.v101_ast) does not read it false. The
    committed V101 reads line by line and cannot see them, so they are judged here, at the raw door, at the git door
    (each case's recorded --name-status) and in the port, main's port judged in its own terms. On 5ebe0b6b every one was
    kept at every door."""
    M = R.main_module()
    cases = json.loads((R.ROOT / "tests" / "fixtures" / "path2a_pass5_repros.json").read_text(encoding="utf-8"))["cases"]
    items = [{"id": c["id"], "summary": c["summary"], "diff": c["diff"]} for c in cases]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    if NODE is None:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("node is not on PATH under CI; the port half of this check did not run")
        pytest.skip("node is not on PATH; the port half of this check cannot run here")
    main_port = _port_records(R.main_port_path(tmp_path), items, tmp_path, "main5")
    new_port = _port_records(R.PORT, items, tmp_path, "new5")
    seen = []
    for c in cases:
        fake = R.fake_git(c["name_status"], c["diff"])
        for mod in (M, N):
            monkeypatch.setattr(mod, "_git", fake)
        doors = {"raw": (M.gate_diff_text(c["summary"], c["diff"]).to_dict(),
                         N.gate_diff_text(c["summary"], c["diff"]).to_dict()),
                 "git": (M.gate_diff(c["summary"], "(repo)", "base", "head").to_dict(),
                         N.gate_diff(c["summary"], "(repo)", "base", "head").to_dict()),
                 "port": (main_port[c["id"]], new_port[c["id"]])}
        for door, (a, b) in doors.items():
            if door != "port":
                assert R.relation(a, b, False, N._P2A_PHRASES) == [], (c["id"], door)
            for i, x in enumerate(a["claims"]):
                if x["verdict"] not in DECIDED:
                    continue
                t = T.truth(c["model"], x["kind"], x["detail"])
                assert T.wrong(x["verdict"], t), (c["id"], door, x, t)
                v = T.v101_ast(c["model"], x)
                assert v is not None and not T.wrong(v, t), (c["id"], door, x, v, t)
                y = b["claims"][i]
                assert y["verdict"] == "UNCHECKABLE" and R.phrase_key(y["why"], N._P2A_PHRASES) in \
                    ("tests", "split", "symbol"), (c["id"], door, y)
                seen.append((c["id"], door))
    assert len(seen) == 33, seen


# ---- pass 6: a definition again beside its own unchanged one, and a moved file's old path, at three doors ----------

def test_the_sixth_reviews_reproductions(monkeypatch, tmp_path):
    """B-1 and B-3 (NOTE_path2a_sixth_pass_2026_09_30), judged by truth at the raw door, at the git door (each case's
    recorded --name-status) and in the port. A test defined again beside its own unchanged definition: main's verdict
    is false by CPython's ast and not false under the ast-paired V101, and is withheld (`redefined`); 495d2204 kept it
    at every door. A name defined again: the truth reads it undecided, and it is withheld (`again`). A claim naming a
    moved file's old path: main's VERIFIED is right and withheld (`dir`, `dot_tier`), the known loss the README names."""
    M = R.main_module()
    doc = json.loads((R.ROOT / "tests" / "fixtures" / "path2a_pass6_repros.json").read_text(encoding="utf-8"))
    cases = doc["cases"]
    items = [{"id": c["id"], "summary": c["summary"], "diff": c["diff"]} for c in cases]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    if NODE is None:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("node is not on PATH under CI; the port half of this check did not run")
        pytest.skip("node is not on PATH; the port half of this check cannot run here")
    main_port = _port_records(R.main_port_path(tmp_path), items, tmp_path, "main6")
    new_port = _port_records(R.PORT, items, tmp_path, "new6")
    seen = collections.Counter()
    for c in cases:
        fake = R.fake_git(c["name_status"], c["diff"])
        for mod in (M, N):
            monkeypatch.setattr(mod, "_git", fake)
        doors = {"raw": (M.gate_diff_text(c["summary"], c["diff"]).to_dict(),
                         N.gate_diff_text(c["summary"], c["diff"]).to_dict()),
                 "git": (M.gate_diff(c["summary"], "(repo)", "base", "head").to_dict(),
                         N.gate_diff(c["summary"], "(repo)", "base", "head").to_dict()),
                 "port": (main_port[c["id"]], new_port[c["id"]])}
        for door, (a, b) in doors.items():
            if door != "port":
                assert R.relation(a, b, False, N._P2A_PHRASES) == [], (c["id"], door)
            keys = []
            for i, x in enumerate(a["claims"]):
                if x["verdict"] not in DECIDED:
                    continue
                t = T.truth(c["model"], x["kind"], x["detail"])
                if c["truth"] == "false":
                    assert T.wrong(x["verdict"], t), (c["id"], door, x, t)
                    v = T.v101_ast(c["model"], x)
                    assert v is not None and not T.wrong(v, t), (c["id"], door, x, v, t)
                elif c["truth"] == "right":
                    assert T.right(x["verdict"], t), (c["id"], door, x, t)
                else:
                    assert not T.right(x["verdict"], t) and not T.wrong(x["verdict"], t), (c["id"], door, x, t)
                y = b["claims"][i]
                assert y["verdict"] == "UNCHECKABLE", (c["id"], door, y)
                keys.append(R.phrase_key(y["why"], N._P2A_PHRASES))
                seen[c["truth"]] += 1
            assert keys == c["want"], (c["id"], door, keys)
    assert dict(seen) == {"false": 6, "undecided": 3, "right": 6}, seen


# ---- pass 4: the fourth review's scope reproductions, judged by truth --------------------------------------------------

_GNU = ("--- a/.github/workflows/ci.yml\t2024-05-06 07:08:09.000000000 +0000\n"
        "+++ b/.github/workflows/ci.yml\t2024-05-06 07:08:10.000000000 +0000\n@@ -1 +1 @@\n-a\n+b\n"
        "--- a/.github/workflows/old.yml\t2024-05-06 07:08:09.000000000 +0000\n"
        "+++ /dev/null\t1970-01-01 00:00:00.000000000 +0000\n@@ -1 +0,0 @@\n-x\n")
_MNEMONIC = ("diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
             "--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-a\n+b\n"
             "diff --git c/.github/workflows/rel.yml w/.github/workflows/rel.yml\n"
             "--- c/.github/workflows/rel.yml\n+++ w/.github/workflows/rel.yml\n@@ -1 +1 @@\n-x\n+y\n")
_RENAME = ("diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\nindex 7898192..6178079 100644\n"
           "--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-a\n+b\n"
           "diff --git a/docs/x.md b/.github/workflows/x.md\nsimilarity index 100%\nrename from docs/x.md\n"
           "rename to .github/workflows/x.md\n")
_FIVE = "one\ntwo\nthree\nfour\nfive\n"
SCOPE_CASES = [
    {"id": "gnu-delete-under-dotted-dir", "summary": "Only touches github and .github/workflows/.", "diff": _GNU,
     "model": {"base": {".github/workflows/ci.yml": "a\n", ".github/workflows/old.yml": "x\n"},
               "head": {".github/workflows/ci.yml": "b\n"}}},
    {"id": "mnemonic-second-file", "summary": "Only touches github and .github/workflows/.", "diff": _MNEMONIC,
     "model": {"base": {".github/workflows/ci.yml": "a\n", ".github/workflows/rel.yml": "x\n"},
               "head": {".github/workflows/ci.yml": "b\n", ".github/workflows/rel.yml": "y\n"}}},
    {"id": "git-rename-into-dotted-dir", "summary": "Only touches github and .github/workflows/.", "diff": _RENAME,
     "name_status": "M\t.github/workflows/ci.yml\nR100\tdocs/x.md\t.github/workflows/x.md\n",
     "model": {"base": {".github/workflows/ci.yml": "a\n", "docs/x.md": _FIVE},
               "head": {".github/workflows/ci.yml": "b\n", ".github/workflows/x.md": _FIVE}}},
]


def test_the_fourth_reviews_scope_reproductions(monkeypatch):
    """B-1 (NOTE_path2a_fourth_pass_2026_09_30): main reads `github` as a path only because `.github` loses its dot, and
    a rendering fault or a rename makes its verdict false; V121 says the prefix is not a path. Each false verdict is
    attributable and must be withheld, at the raw door and, for the rename, at the git door too. On ea677740 all four
    readings kept main's verdict."""
    M = R.main_module()
    V = variants()
    seen = []
    for it in SCOPE_CASES:
        doors = [("raw", lambda mod, it=it: mod.gate_diff_text(it["summary"], it["diff"]))]
        if it.get("name_status"):
            fake = R.fake_git(it["name_status"], it["diff"])
            for mod in [M, N, *V.values()]:
                monkeypatch.setattr(mod, "_git", fake)
            doors.append(("git", lambda mod, it=it: mod.gate_diff(it["summary"], "(repo)", "base", "head")))
        for door, run in doors:
            a = run(M).to_dict()
            b = run(N).to_dict()
            assert R.relation(a, b, False, N._P2A_PHRASES) == [], (it["id"], door)
            vs = {k: run(mod).to_dict()["claims"] for k, mod in V.items()}
            for i, x, t, who in judge(it["model"], a["claims"], vs):
                assert T.wrong(x["verdict"], t) and who, (it["id"], door, x, t, who)
                y = b["claims"][i]
                assert y["verdict"] == "UNCHECKABLE" and R.phrase_key(y["why"], N._P2A_PHRASES) == "shape", \
                    (it["id"], door, y)
                seen.append((it["id"], door))
    assert len(seen) == 4, seen


# ---- passes 8 and 9: the reviews' transforms of the builder's world ---------------------------------------------------

def _transformed(how: str) -> list:
    """The reviews' transforms of the generated cases. NOTE_path2a_eighth_pass_2026_10_01, B-1 and B-2 (the seventh
    review's): `cjk2` adds two changed docs whose names differ at CJK code points only (no case: 8eead84f read them as a
    case pair and kept every CONTRADICTED beside a count claim), and a right count sentence where main reads no count
    claim; `insent` writes "(naive)" with an i-diaeresis after each word of a tests or count claim, inside the claim's
    own sentence but away from where its number is read (8eead84f kept every #121 count and #101 tests CONTRADICTED
    so). NOTE_path2a_ninth_pass_2026_10_04, P-4 (the eighth coverage review's lead, which a probe reproduced): `bom`
    writes U+FEFF before the summary, as a summary file saved with a byte-order mark reaches the CLI, and `styxx`
    appends a sentence naming styxx beside U+2028; main's two ports read both as they read the plain summary, and at
    3bdc3bc4 the C-1 switch kept 139 and 214 attributable false CONTRADICTEDs under them, in each port."""
    import random
    import re
    rng = random.Random(7)
    out = []
    for c in C.families(SEED, PER_FAMILY):
        c = json.loads(json.dumps(c))
        if how == "cjk2":
            for p in ("docs/zh/" + chr(0x5B89) + chr(0x88C5) + ".md", "docs/zh/" + chr(0x914D) + chr(0x7F6E) + ".md"):
                c["model"]["base"][p] = "a\n"
                c["model"]["head"][p] = "b\n"
            c["diff"] = C.render(c["model"]["base"], c["model"]["head"], c["style"], rng)
            try:
                a = R.main_module().gate_diff_text(c["summary"], c["diff"]).to_dict()
            except Exception:
                a = {"claims": [{"kind": "files_changed_count"}]}
            if not any(x["kind"] == "files_changed_count" for x in a["claims"]):
                c["summary"] = c["summary"] + f"\n{len(T.changed(c['model']))} files changed."
        elif how == "insent":
            c["summary"] = re.sub(r"(\b(?:tests?|files? changed)\b)", r"\1 (na" + chr(0xEF) + "ve)", c["summary"])
        elif how == "bom":
            c["summary"] = chr(0xFEFF) + c["summary"]
        else:
            assert how == "styxx", how
            c["summary"] = c["summary"] + "\nChecked with styxx." + chr(0x2028) + "Thanks."
        out.append(c)
    return out


# The attributable false verdicts of each transformed world, in Python and in the port: every one is withheld.
TRANSFORMED = {"cjk2": (1073, 969), "insent": (1193, 1087), "bom": (1175, 1087), "styxx": (1193, 1087)}


@pytest.mark.parametrize("how", sorted(TRANSFORMED))
def test_the_reviews_transforms_keep_no_attributable_verdict(how, tmp_path):
    """Every attributable false verdict withheld on the transformed world, in Python and, judged in its own terms, in
    the port. On 8eead84f the seventh review counted 19 kept under `cjk2` (16 tests and 3 count CONTRADICTEDs) and 213
    under `insent` (every count and 16 of 17 tests CONTRADICTEDs); on 3bdc3bc4 a probe counted 139 kept under `bom` and
    214 under `styxx`, in each port."""
    M = R.main_module()
    V = variants()
    counts = collections.Counter()
    misses = []
    cases = _transformed(how)
    for it in cases:
        try:
            a = M.gate_diff_text(it["summary"], it["diff"]).to_dict()
        except Exception:
            continue
        vs = {}
        for k, mod in V.items():
            try:
                vs[k] = mod.gate_diff_text(it["summary"], it["diff"]).to_dict()["claims"]
            except Exception:
                vs[k] = None
        b = N.gate_diff_text(it["summary"], it["diff"]).to_dict()
        assert R.relation(a, b, False, N._P2A_PHRASES) == [], it["id"]
        tally(counts, misses, it, judge(it["model"], a["claims"], vs), b["claims"])
    port_counts, port_misses, _rows = _port_truth(cases, tmp_path)
    print(how, "python:", json.dumps(dict(counts)), "port:", json.dumps(port_counts))
    for m in (misses + port_misses)[:20]:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert misses == [] and port_misses == []
    assert (counts["attributable"], port_counts["attributable"]) == TRANSFORMED[how]

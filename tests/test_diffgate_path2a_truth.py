"""PATH-2a coverage (B), judged by truth (NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30,
NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30, NOTE_path2a_fifth_pass_2026_09_30).

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
# a redefinition) and 4 undecided, in each port; C-1 moves nothing here.
PINNED = {"cases": 1706, "main raises": 0, "decided": 7614, "false": 1535, "attributable": 1206, "abstained": 1563,
          "attributable abstained": 1206, "right": 5897, "right lost": 251, "undecided": 182,
          "undecided abstained": 39, "unjudged": 0, "unjudged abstained": 0, "false other": 329,
          "false other abstained": 67}
# The port judged in its own terms: 1,100 attributable claims, of which 942 carry the same main record in Python and
# 158 are read otherwise by Python's main (paths and names outside ASCII, which the two templates extract apart).
PINNED_PORT = {"cases": 1706, "main raises": 0, "decided": 7393, "abstained": 1424, "right": 5739, "right lost": 175,
               "false": 1385, "attributable": 1100, "attributable abstained": 1100, "false other": 285,
               "false other abstained": 23, "undecided": 269, "undecided abstained": 126}
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


def test_every_attributable_false_verdict_abstains_in_the_port(world, tmp_path):
    """In the port's own terms (B-2): main's port and the four variants built from it read each case, truth judges
    the port's own claims (its own details), and every claim the port's variants show attributable must be
    UNCHECKABLE in this port."""
    if NODE is None:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail("node is not on PATH under CI; the port half of the coverage check did not run")
        pytest.skip("node is not on PATH; the port half of the coverage check cannot run here")
    items = [{"id": str(k), "summary": it["summary"], "diff": it["diff"]} for k, (it, _a, _c) in enumerate(world)]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    main_port = _port_records(R.main_port_path(tmp_path), items, tmp_path, "main")
    new_port = _port_records(R.PORT, items, tmp_path, "new")
    var = {}
    for v in T.PORT_VARIANTS:
        p = tmp_path / f"diffgate_{v}.js"
        p.write_bytes(T.port_variant_source(R.main_port_source(), v).encode("utf-8"))
        var[v] = _port_records(p, items, tmp_path, v)
    counts = collections.Counter({k: 0 for k in ("cases", "main raises")})
    misses = []
    same_as_python = collections.Counter()
    for k, (it, a, _claims) in enumerate(world):
        counts["cases"] += 1
        jm = main_port[str(k)]
        if "error" in jm:
            counts["main raises"] += 1
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
        if a is not None:
            for i, x, _t, who in judged:
                if who:
                    same = i < len(a["claims"]) and all(a["claims"][i][f] == x[f]
                                                        for f in ("kind", "verdict", "why", "text", "detail"))
                    same_as_python["same main record in Python" if same else "Python's main reads it otherwise"] += 1
    print("port truth:", json.dumps(dict(counts), indent=1), dict(same_as_python))
    for m in misses:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert misses == []
    assert dict(counts) == PINNED_PORT


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
    ("a reason naming the wrong verdict", "            c.why = _p2a_reason(c.verdict, hit[1], hit[0], c.why)",
     '            c.why = _p2a_reason("VERIFIED", hit[1], hit[0], c.why)'),
    # pass 2
    ("the overlay skipped under --strict",
     "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH]",
     "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH and not strict]"),
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
    ("the summary never read",
     "        self.diff_text, self.name_status, self.summary, self._m = diff_text, name_status, summary, {}",
     '        self.diff_text, self.name_status, self.summary, self._m = diff_text, name_status, "", {}'),
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
    ("C-1's rule dropped", '    if c.verdict == "CONTRADICTED" and f.apart(f.kinds()):\n        return None\n', ""),
    ("C-1's diff part dropped", "    if _p2a_apart_diff(f, kinds):\n        return True\n", ""),
    ("C-1's DECLARE-1 fence part dropped", '    if "styxx" in summary and (', "    if False and ("),
    ("C-1's sentence part dropped", "    if not _P2A_BAD_RX.search(summary):\n        return False\n    low",
     "    if True:\n        return False\n    low"),
    ("the unchanged lines dropped",
     '        return self._get("unchanged", lambda: _p2a_context(self.diff_text, self.fine())\n'
     '                         + _p2a_joined(self.diff_text, " "))',
     '        return self._get("unchanged", lambda: [])'),
    ("a name defined again kept",
     '    if f.redefines(name):                 # B-1 (NOTE_path2a_sixth_pass_2026_09_30)\n        return "again", "#101"\n',
     ""),
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
    ("no C-1 rule in the port", '  if (c.verdict === "CONTRADICTED" && f.apart(f.kinds())) return null;\n', ""),
    ("no C-1 diff part in the port", "  if (_p2aApartDiff(f, kinds)) return true;\n", ""),
    ("no C-1 fence part in the port", '  if (s.includes("styxx") && (', "  if (false && ("),
    ("no unchanged lines in the port",
     '    unchanged: () => get("unchanged", () => _p2aContext(diffText, f.coarse()).concat(_p2aJoined(diffText, " "))),',
     "    unchanged: () => [],"),
    ("no again rule in the port", '  if (f.redefines(name)) return ["again", "#101"];   // B-1 (NOTE_path2a_sixth_pass_2026_09_30)\n',
     ""),
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
                                                  for i, x in enumerate(items)], ensure_ascii=False), encoding="utf-8")
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
        # Pass 6 (NOTE_path2a_sixth_pass_2026_09_30, C-1): where the two ports' views count `def test_` sites apart, a
        # CONTRADICTED stands in both, and a VERIFIED is read by one main only, so a plant of the port's own view
        # (_P2A_OWN) can no longer split a claim both mains read alike. It is caught where it moves the port's own
        # decision on a claim only its main decides so: PORT_ONLY_CASES, against the real port.
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


# Claims only one main decides so (NOTE_path2a_sixth_pass_2026_09_30, C-1): a created test after U+FEFF, which the port
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

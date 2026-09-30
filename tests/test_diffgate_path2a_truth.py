"""PATH-2a coverage (B), judged by truth (NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30).

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
PINNED = {"cases": 1706, "main raises": 0, "decided": 7614, "false": 1535, "attributable": 1206, "abstained": 1494,
          "attributable abstained": 1206, "right": 5897, "right lost": 202, "undecided": 182,
          "undecided abstained": 35, "unjudged": 0, "unjudged abstained": 0, "false other": 329,
          "false other abstained": 51}
# The port judged in its own terms: 1,100 attributable claims, of which 942 carry the same main record in Python and
# 158 are read otherwise by Python's main (paths and names outside ASCII, which the two templates extract apart).
PINNED_PORT = {"cases": 1706, "main raises": 0, "decided": 7393, "abstained": 1355, "right": 5739, "right lost": 126,
               "false": 1385, "attributable": 1100, "attributable abstained": 1100, "false other": 285,
               "false other abstained": 7, "undecided": 269, "undecided abstained": 122}
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
    ("the port-only line view", "    for rx in (_P2A_FINE, _P2A_COARSE):\n        ls = _p2a_lines(diff_text, rx)",
     "    for rx in (_P2A_COARSE,):\n        ls = _p2a_lines(diff_text, rx)"),
    ("count CONTRADICTED interval shrunk", 'return ("count", "#121") if lo <= n <= hi else None',
     'return ("count", "#121") if lo <= n < hi else None'),
    ("V97 allows a base name for a directory claim", 'if t == 2 and "/" in c and P2A_DIRECTORY_BASENAME_ABSTAINS:',
     "if t == 2 and False:"),
    ("V121 dropped", '    v121 = ok(_p2a_resolve(f.status("K"), f.groups("K"), ck, False))', "    v121 = True"),
    ("V97+V121 dropped", '    vboth = ok(_p2a_resolve(f.status("K"), f.groups("K"), ck, True))', "    vboth = True"),
    ("tests interval lower bound + 1", "g - min(g, p) <= n <= g", "g - min(g, p) + 1 <= n <= g"),
    ("symbol rule off", '    if f.defines(name):\n        return "symbol", "#101"\n',
     '    if False:\n        return "symbol", "#101"\n'),
    ("def separator restricted to space and tab", '_P2A_COARSE_RUN = re.compile("[\\x00-\\x20\\x7f-\\U0010ffff]*")',
     '_P2A_COARSE_RUN = re.compile("[ \\t]*")'),
    ("names paired by their full run", "                    rem.add(line[r:_P2A_WORD_RUN.match(line, r).end()])",
     "                    rem.add(line[r:_P2A_NAME_RUN.match(line, r).end()])"),
    ("the scope rule's V121 prefix set dropped",
     '    if got["K", used_k, False] != got["A", used_a, False]:', "    if False:"),
    ("U2 dropped",
     "        if want is not None and tw is not None and len(keys[ws[i]]) > 1 and sts[ws[i]] != {want}:",
     "        if False:"),
    ("a reason naming the wrong verdict", "            c.why = _p2a_reason(c.verdict, hit[1], hit[0], c.why)",
     '            c.why = _p2a_reason("VERIFIED", hit[1], hit[0], c.why)'),
    # pass 2
    ("the overlay skipped under --strict",
     "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH]",
     "    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH and not strict]"),
    ("split read as keep", '    return ("split", "#101") if any(fires) else None', "    return None"),
    ("extract off for paths", "    if _p2a_extract(f, claimed, c.text, want):", "    if False:"),
    ("the port's reading of a wide path taken as never verified",
     "        if b and not _P2A_WIDE.search(b) and b in held and (want is None or regs[i][1] == want):",
     "        if False:"),
    ("extract off for scopes", "    if i < 0 or _P2A_WIDE_DIV.search(c.text, i):", "    if False:"),
    ("extract off for symbols", '    if _p2a_touches_wide(name, c.text):\n        return "extract", "#101"',
     '    if False:\n        return "extract", "#101"'),
    ("odd reads every drive letter",
     '    return q == "." or q.endswith("/.") or (len(q) >= 2 and q[1] == ":" and (len(q) == 2 or q[2] != "/"))',
     '    return q == "." or q.endswith("/.") or (len(q) >= 2 and q[1] == ":")'),
    ("dot_earliest read as dot", '        return ("dot_earliest" if vboth else "dot"), "#121"',
     '        return "dot", "#121"'),
]
# Equivalent by construction, and said so rather than hidden: once the per-set comparison of V121 with main's key
# has passed, a prefix set V121 would choose differently cannot read otherwise (a second prefix is path-shaped in a
# space exactly when some changed path lies under it there, which the per-set comparison already compares). The
# clause stays as the NOTE's rule; this plant is refused by the same check that shows it never decides alone.
EQUIVALENT = {"the scope rule's V121 prefix set dropped"}


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
    ("one line view", "  return [_P2A_FINE, _P2A_COARSE].map(rx => {", "  return [_P2A_COARSE].map(rx => {"),
    ("count interval shrunk", 'return (lo <= n && n <= hi) ? ["count", "#121"] : null;',
     'return (lo <= n && n < hi) ? ["count", "#121"] : null;'),
    # pass 2
    ("no split rule", '  return fires.some(x => x) ? ["split", "#101"] : null;', "  return null;"),
    ("no extract for paths", '  if (_p2aExtract(f, claimed, c.text, want)) return ["extract", "#97, #121"];\n', ""),
    ("the port reading its count as CPython's", "const _P2A_OWN = 1;", "const _P2A_OWN = 0;"),
]


@pytest.mark.parametrize("name,old,new", JS_PLANTS)
def test_port_plants_make_the_ports_disagree(name, old, new, tmp_path):
    if NODE is None:
        pytest.skip("node is not on PATH; the port plants cannot run here")
    text = R.lf(R.PORT)
    block = R.js_block(text)
    assert block.count(old) == 1, old
    planted = tmp_path / "diffgate_planted.js"
    planted.write_bytes(text.replace(block, block.replace(old, new)).encode("utf-8"))
    items = [x for x in R.inputs(fuzz=False)]
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
            if all(x[f] == jx[f] for f in ("kind", "verdict", "text")):
                y, jy = b["claims"][k], j["new"]["claims"][k]
                new_disagreements += (y["verdict"], R.phrase_key(y["why"], N._P2A_PHRASES)) != \
                    (jy["verdict"], R.phrase_key(jy["why"], N._P2A_PHRASES))
    assert new_disagreements > 0, name

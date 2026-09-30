"""PATH-2a coverage (B), judged by truth (NOTE_path2a_abstain_overlay_2026_09_30).

Truth comes from each case's base/head file model (tests/_p2a_truth.py), never from the diff. A decided claim is
ATTRIBUTABLE when main's verdict is false by truth and a counterfactual variant of main without #97, without #121,
without #101, or without all three, gives the same claim a verdict that is not false. The bar: every attributable
claim is UNCHECKABLE on the branch -- at the raw door and in the port. The counts are pinned, and so is the cost
(right verdicts lost, undecided verdicts abstained).

The cases are #161's reproductions and the PREREG_path2 reproductions (tests/fixtures/path2a_repros.json), and
400 generated cases per family at a pinned seed (tests/_p2a_cases.py): #97, #121, #121 with case outside ASCII, #101.

`test_plants_are_refused` is the calibration: each plant is one anchored edit to the Python block, and each must be
caught by the pinned pairs, by this truth assertion, or by the abstain-only relation. Three plants in the port block
must each make the port disagree with the Python where main's two ports agree.
"""
from __future__ import annotations

import collections
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
PINNED = {"cases": 1706, "main raises": 0, "decided": 7614, "false": 1535, "attributable": 1206, "abstained": 1423,
          "attributable abstained": 1206, "right": 5897, "right lost": 142, "undecided": 182,
          "undecided abstained": 35, "unjudged": 0, "unjudged abstained": 0, "false other": 329,
          "false other abstained": 40}


@pytest.fixture(scope="module")
def world():
    """Every case, main's record, each decided claim's truth and whether main's false verdict is attributable."""
    M = R.main_module()
    V = {v: R.module_from(T.variant_source(R.main_source(), v), "_p2a_variant_" + v) for v in T.VARIANTS}
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
        claims = []
        for i, x in enumerate(a["claims"]):
            if x["verdict"] not in DECIDED:
                continue
            t = T.truth(it["model"], x["kind"], x["detail"])
            who = [k for k in T.VARIANTS if T.wrong(x["verdict"], t) and vs[k] is not None
                   and not T.wrong(vs[k][i]["verdict"], t)]
            claims.append((i, x, t, who))
        out.append((it, a, claims))
    return out


def evaluate(mod, world):
    counts = collections.Counter({k: 0 for k in PINNED})
    misses = []
    for it, a, claims in world:
        counts["cases"] += 1
        if a is None:
            counts["main raises"] += 1
            continue
        b = mod.gate_diff_text(it["summary"], it["diff"]).to_dict()
        for i, x, t, who in claims:
            ab = b["claims"][i]["verdict"] == "UNCHECKABLE"
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
    return dict(counts), misses


def test_every_attributable_false_verdict_abstains(world):
    counts, misses = evaluate(N, world)
    print("PATH-2a truth:", json.dumps(counts, indent=1))
    for m in misses:
        print("MISS", json.dumps(m, ensure_ascii=True))
    assert misses == []
    assert counts == PINNED


def test_every_attributable_false_verdict_abstains_in_the_port(world, tmp_path):
    """Through check_path2a.js --decisions, wherever the port's main gives the claim the same record."""
    if NODE is None:
        pytest.skip("node is not on PATH; the port half of the coverage check cannot run here")
    items = [{"id": str(k), "summary": it["summary"], "diff": it["diff"]} for k, (it, _a, _c) in enumerate(world)]
    (tmp_path / "in.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    ref = R.main_port_path(tmp_path)
    r = subprocess.run([NODE, str(R.DIFFERENTIAL / "check_path2a.js"), "--decisions", str(ref),
                        str(tmp_path / "in.json"), str(tmp_path / "out.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    js = {d["id"]: d for d in json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))}
    c = collections.Counter()
    misses = []
    for k, (it, a, claims) in enumerate(world):
        if a is None or "error" in js[str(k)]["main"]:
            continue
        jm, jn = js[str(k)]["main"]["claims"], js[str(k)]["new"]["claims"]
        for i, x, _t, who in claims:
            if not who:
                continue
            same = i < len(jm) and all(jm[i][f] == x[f] for f in ("kind", "verdict", "why", "text", "detail"))
            c["attributable, same main record in the port" if same else "attributable, the port's main differs"] += 1
            if same and jn[i]["verdict"] != "UNCHECKABLE":
                misses.append((it["id"], i, x["kind"], x["verdict"]))
    print("port coverage:", dict(c))
    assert misses == []
    # The 264 are claims main's port reads differently to begin with (paths outside ASCII its ASCII-only template
    # extracts otherwise, GNU timestamps its trim() and Python's strip() keep alike but split elsewhere): reported.
    assert dict(c) == {"attributable, same main record in the port": 942, "attributable, the port's main differs": 264}


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
    ("V121 dropped", '    v121 = ok(_p2a_resolve(f.status("K"), ck, False))', "    v121 = True"),
    ("V97+V121 dropped", '    vboth = ok(_p2a_resolve(f.status("K"), ck, True))', "    vboth = True"),
    ("tests interval lower bound + 1", "chg and got - chg <= n <= got", "chg and got - chg + 1 <= n <= got"),
    ("symbol rule off", '    name = c.detail.get("name")\n', '    name = c.detail.get("name")\n    return None\n'),
    ("def separator restricted to space and tab", "    return o <= 0x20 or o == 0x7F or o >= 0x80",
     '    return ch in " \\t"'),
    ("the ASCII run of a name dropped", '    return {line[k:e], line[k:a]} - {""}', '    return {line[k:e]} - {""}'),
    ("the scope rule's V121 prefix set dropped",
     '    if got["K", used_k, False] != got["A", used_a, False]:', "    if False:"),
    ("U2 dropped", "        if want is not None and tw is not None and len(keys[w]) > 1 and sts[w] != {want}:",
     "        if False:"),
    ("a reason naming the wrong verdict", "            c.why = _p2a_reason(c.verdict, hit[1], hit[0], c.why)",
     '            c.why = _p2a_reason("VERIFIED", hit[1], hit[0], c.why)'),
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
        try:
            a = M.gate_diff_text(summary, diff).to_dict()
        except Exception:
            continue
        if R.relation(a, mod.gate_diff_text(summary, diff).to_dict(), False, N._P2A_PHRASES):
            return f"relation on {s}:{iid}"
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
            if all(x[f] == jx[f] for f in ("kind", "verdict", "why", "text", "detail")):
                y, jy = b["claims"][k], j["new"]["claims"][k]
                new_disagreements += (y["verdict"], y["why"]) != (jy["verdict"], jy["why"])
    assert new_disagreements > 0, name

"""Stage-B corpus generator: contradiction-detection items with CONSTRUCTED truth. SHARED by
rung 1 (Qwen panel) and rung 2 (Claude subagent panel) -- different seeds, same generator.

Task: given statements A and B, does B CONTRADICT A? y=1 iff B contradicts A, by construction.
Anchors are the blatant ends the auditor can build without labels: neg = B is A verbatim
(identical pair, cannot contradict); pos = direct planted negation. Organic items are the
GRADED LADDER the panels demanded (medium/hard on both sides), so anchor-vs-organic difficulty
gaps -- the beta-optimism channel -- are present BY DESIGN and the audit must live with them.
True labels ship in a held-out file the judges never see. Deterministic per seed. ASCII only.
"""
from __future__ import annotations
import json
import numpy as np

NAMES = ["marla vance", "otto reinholt", "dessa kwan", "ivor palchek", "nina strade",
         "cyrus abbot", "lena morrow", "hektor sallis", "prue tanaka", "gideon falk"]
CITIES = ["velmora", "kastelun", "drovny", "ashport", "quillhaven", "berevic", "moltrasse",
          "yarrowfield", "sundvik", "calderest"]
COUNTRIES = ["norvania", "esterholm", "vukovia", "tessaly", "ombria"]
JOBS = ["cartographer", "glassblower", "archivist", "locksmith", "typesetter"]
OBJECTS = ["ledger", "lantern", "atlas", "gramophone", "barometer"]
COLORS = ["crimson", "teal", "ochre", "slate", "ivory"]
YEARS = list(range(1871, 1969))


def _fact(rng):
    kind = rng.integers(0, 4)
    if kind == 0:
        n, c, y = rng.choice(NAMES), rng.choice(CITIES), int(rng.choice(YEARS))
        return ("born", {"n": n, "c": c, "y": y},
                f"{n} was born in {c} in {y}.")
    if kind == 1:
        c, k = rng.choice(CITIES), rng.choice(COUNTRIES)
        return ("capital", {"c": c, "k": k}, f"{c} is the capital of {k}.")
    if kind == 2:
        o, col = rng.choice(OBJECTS), rng.choice(COLORS)
        return ("color", {"o": o, "col": col}, f"the {o} in the study is {col}.")
    n, j, c = rng.choice(NAMES), rng.choice(JOBS), rng.choice(CITIES)
    return ("job", {"n": n, "j": j, "c": c}, f"{n} works as a {j} in {c}.")


def _other(rng, pool, cur):
    alts = [x for x in pool if x != cur]
    return alts[int(rng.integers(0, len(alts)))]


def _contradiction(rng, kind, s, hard):
    """y=1 organic: medium = value swap; hard = value swap under a paraphrase template."""
    if kind == "born":
        c2 = _other(rng, CITIES, s["c"])
        return (f"records show {s['n']} was actually born in {c2}, in {s['y']}." if hard
                else f"{s['n']} was born in {c2} in {s['y']}.")
    if kind == "capital":
        k2 = _other(rng, COUNTRIES, s["k"])
        return (f"every atlas lists {s['c']} as the seat of government of {k2}." if hard
                else f"{s['c']} is the capital of {k2}.")
    if kind == "color":
        col2 = _other(rng, COLORS, s["col"])
        return (f"visitors always remark on the {col2} {s['o']} kept in the study." if hard
                else f"the {s['o']} in the study is {col2}.")
    j2 = _other(rng, JOBS, s["j"])
    return (f"{s['n']} earns a living in {s['c']} as a {j2}." if hard
            else f"{s['n']} works as a {j2} in {s['c']}.")


def _consistent(rng, kind, s, hard):
    """y=0 organic: medium = paraphrase; hard = consistent elaboration (adds detail)."""
    if kind == "born":
        return (f"{s['n']}, delivered by a midwife one cold morning of {s['y']}, drew a first "
                f"breath in {s['c']}." if hard
                else f"in {s['y']}, {s['c']} was the birthplace of {s['n']}.")
    if kind == "capital":
        return (f"the ministries, the mint, and the high court of {s['k']} all crowd the "
                f"avenues of {s['c']}." if hard
                else f"the capital city of {s['k']} is {s['c']}.")
    if kind == "color":
        return (f"dust never settles on the {s['o']} in the study, and its {s['col']} finish "
                f"still catches the lamplight." if hard
                else f"a {s['col']} {s['o']} sits in the study.")
    return (f"most mornings find {s['n']} at the {s['j']}'s bench before the {s['c']} bells "
            f"ring." if hard
            else f"in {s['c']}, {s['n']} is employed as a {s['j']}.")


def build_corpus(seed, n_organic=240, k_anchor=80, pi=0.35, hard_frac=0.5):
    """Returns (items, truth): items = [{id, A, B}], truth = {id: y} for organic; anchors carry
    role 'neg_anchor'/'pos_anchor' in a separate list. Judges see items only."""
    rng = np.random.default_rng(seed)
    organic, anchors, truth = [], [], {}
    for i in range(n_organic):
        kind, s, A = _fact(rng)
        y = int(rng.random() < pi)
        hard = bool(rng.random() < hard_frac)
        B = _contradiction(rng, kind, s, hard) if y else _consistent(rng, kind, s, hard)
        iid = f"org_{seed}_{i:04d}"
        organic.append({"id": iid, "A": A, "B": B})
        truth[iid] = y
    for i in range(k_anchor):
        kind, s, A = _fact(rng)
        anchors.append({"id": f"neg_{seed}_{i:04d}", "A": A, "B": A, "role": "neg_anchor"})
    for i in range(k_anchor):
        kind, s, A = _fact(rng)
        anchors.append({"id": f"pos_{seed}_{i:04d}", "A": A,
                        "B": "it is not true that " + A[:-1] + ".", "role": "pos_anchor"})
    return organic, anchors, truth


if __name__ == "__main__":
    org, anc, tr = build_corpus(0, n_organic=6, k_anchor=2)
    print(json.dumps({"organic": org, "anchors": anc, "truth": tr}, indent=1))

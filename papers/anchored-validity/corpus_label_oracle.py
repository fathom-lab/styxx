"""Due-diligence label oracle: re-derive every organic and anchor label from ITEM TEXT alone,
independently of the generator's code paths, over the EXACT scored corpora (every seed that fed
a scored run in cycles 50-53). A single mislabel demotes results; the receipt reports per-family
mismatch counts verbatim. Chain families get a fully formal oracle (parse relations, transitive
closure, entailment/contradiction decision); attr/numeric/temporal get vocabulary-slot oracles
(shared vocab knowledge, independent decision logic). ASCII only."""
from __future__ import annotations
import json, re, sys
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_b_corpus as C


def oracle_chain(A, B):
    """Formal: build the taller-than closure from A, decide B. Returns 1/0/None(undecidable)."""
    taller = set()
    names = set()
    def add(a, b):
        taller.add((a, b)); names.add(a); names.add(b)
    for s in A.rstrip(".").split(". "):
        m = re.fullmatch(r"(.+?) is taller than (.+)", s)
        if m:
            add(m.group(1), m.group(2)); continue
        m = re.fullmatch(r"(.+?) is shorter than (.+)", s)
        if m:
            add(m.group(2), m.group(1)); continue
        return None
    changed = True
    while changed:
        changed = False
        for (a, b), (c, d) in product(list(taller), list(taller)):
            if b == c and (a, d) not in taller:
                taller.add((a, d)); changed = True
    s = B.rstrip(".")
    m = re.fullmatch(r"(.+?) is taller than (.+)", s)
    if m:
        x, yv = m.group(1), m.group(2)
    else:
        m = re.fullmatch(r"(.+?) is shorter than (.+)", s)
        if not m:
            return None
        x, yv = m.group(2), m.group(1)
    if (yv, x) in taller:
        return 1          # B claims x taller y but closure says y taller x -> contradiction
    if (x, yv) in taller:
        return 0          # entailed -> consistent
    return None


def oracle_numeric(A, B):
    a_nums = [int(x) for x in re.findall(r"\b(\d+)\b", A)]
    b_nums = [int(x) for x in re.findall(r"\b(\d+)\b", B)]
    if len(a_nums) != 1 or len(b_nums) != 1:
        return None
    return 1 if a_nums[0] != b_nums[0] else 0


def oracle_temporal(A, B):
    m = re.fullmatch(r"(.+?) visited (.+?) before (.+?)\.", A)
    if not m:
        return None
    n, c1, c2 = m.group(1), m.group(2), m.group(3)
    forms_contra = [f"{n} visited {c2} before {c1}.",
                    f"{c1} came only after {c2} on {n}'s route."]
    forms_consist = [f"{c2} was visited by {n} after {c1}.",
                     f"by the time {n} reached {c2}, {c1} was already behind them."]
    if B in forms_contra:
        return 1
    if B in forms_consist:
        return 0
    return None


def oracle_attr(A, B):
    """Slot-vocabulary oracle: find which value-slot A commits to; B contradicts iff it commits
    the same slot to a DIFFERENT vocabulary value."""
    pools = {"city": C.CITIES, "country": C.COUNTRIES, "job": C.JOBS, "color": C.COLORS}
    hits = {}
    for slot, pool in pools.items():
        a_vals = [v for v in pool if v in A]
        b_vals = [v for v in pool if v in B]
        hits[slot] = (a_vals, b_vals)
    a_years = re.findall(r"\b(18\d\d|19\d\d)\b", A)
    b_years = re.findall(r"\b(18\d\d|19\d\d)\b", B)
    contradiction = False
    decided = False
    for slot, (a_vals, b_vals) in hits.items():
        if len(a_vals) == 1 and len(b_vals) >= 1:
            decided = True
            if any(v != a_vals[0] for v in b_vals):
                contradiction = True
    if a_years and b_years:
        decided = True
        if set(b_years) - set(a_years):
            contradiction = True
    if not decided:
        return None
    return 1 if contradiction else 0


ORACLES = {"attr": oracle_attr, "numeric": oracle_numeric, "temporal": oracle_temporal,
           "chain": oracle_chain, "chain_long": oracle_chain}

# every corpus that fed a scored run (family, anchor_style, seeds, kwargs)
SCORED = [
    ("attr", "blatant", list(range(3001, 3016)), {}),                    # rung 1
    ("attr", "blatant", [7001], {}),                                     # rung 2
    ("attr", "ladder", list(range(4001, 4016)), {}),                     # part 1 repair
    ("numeric", "blatant", list(range(5001, 5016)), {}),                 # part 1
    ("temporal", "blatant", list(range(6001, 6016)), {}),                # part 1
    ("chain", "blatant", list(range(8001, 8016)), {}),                   # part 2a
    ("chain", "ladder", list(range(8501, 8516)), {}),                    # part 2a
    ("chain_long", "blatant", [9001], {"n_organic": 200, "k_anchor": 60,
                                       "hard_frac": 1.0}),               # part 2a frontier
]


def main():
    out = {"checked": [], "total_items": 0, "total_mismatches": 0, "total_undecidable": 0}
    for family, style, seeds, kw in SCORED:
        fam_mis, fam_und, fam_n, examples = 0, 0, 0, []
        for seed in seeds:
            org, anc, truth = C.build_corpus(seed, family=family, anchor_style=style,
                                             **{"n_organic": 240, "k_anchor": 80, "pi": 0.35,
                                                **kw})
            oracle = ORACLES[family]
            for it in org:
                got = oracle(it["A"], it["B"])
                fam_n += 1
                if got is None:
                    fam_und += 1
                    if len(examples) < 3:
                        examples.append({"kind": "UNDECIDABLE", **it})
                elif got != truth[it["id"]]:
                    fam_mis += 1
                    if len(examples) < 3:
                        examples.append({"kind": "MISMATCH", "oracle": got,
                                         "label": truth[it["id"]], **it})
            for a in anc:
                want = 0 if a["role"] == "neg_anchor" else 1
                if a["A"] == a["B"]:
                    got = 0                     # verbatim pair cannot contradict
                elif a["B"].startswith("it is not true that") or \
                        a["B"].startswith("it is not the case that"):
                    got = 1                     # direct negation
                else:
                    got = oracle(a["A"], a["B"])
                fam_n += 1
                if got is None:
                    fam_und += 1
                elif got != want:
                    fam_mis += 1
                    if len(examples) < 3:
                        examples.append({"kind": "ANCHOR_MISMATCH", "oracle": got,
                                         "want": want, **a})
        out["checked"].append({"family": family, "style": style, "n_seeds": len(seeds),
                               "items": fam_n, "mismatches": fam_mis,
                               "undecidable": fam_und, "examples": examples})
        out["total_items"] += fam_n
        out["total_mismatches"] += fam_mis
        out["total_undecidable"] += fam_und
        print(f"  {family}/{style}: {fam_n} items, {fam_mis} mismatches, {fam_und} undecidable")
    dest = HERE / "corpus_label_oracle_receipt.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nTOTAL: {out['total_items']} items, {out['total_mismatches']} mismatches, "
          f"{out['total_undecidable']} undecidable -> {dest.name}")


if __name__ == "__main__":
    main()

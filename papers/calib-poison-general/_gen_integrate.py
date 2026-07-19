"""Integrate the validated banks into capability_battery_gen.py. CPU only, deterministic, idempotent.

Design settled by measurement, not preference:

  KEEP + REPLACE   PLURAL_GEN (32), SEQ_GEN (17 distinct-gold subset)
  ADD              PAST_TENSE_GEN (32), CAPITAL_GEN (32), ELEMENT_GEN (32)   -- all validated CLEAN
  KEEP verbatim    ORTH_FIRST_GEN (16), ANTONYM_GEN (16)  -- selftest fixtures for the echo guard
                   (the only list-format family) and the variant path; both expected to sit UNDER
                   DISJOINT_FLOOR_CLEAN and therefore never to be selected
  DROP             ALPHA_GEN (0.1875), ORTH_LAST_GEN (0.2500), CONTAINS_GEN (0.5625) -- measured at
                   or near the floor, can never be selected, so they are pure decode cost
                   (0.55 GPU-h across the scored run); ALPHA sits at 0.167 on BOTH a 1.5B and a 0.5B,
                   i.e. pinned with no signal in either direction
  NOT ADDED        ORDINAL_GEN -- duplicates SEQ_GEN's construct (ordered-list successor), 13 of 32
                   items are logical inverses sharing a gold, and its golds ('second', 'third') are
                   high-frequency words carrying incidental-containment risk

SEQ_GEN is trimmed to its DISTINCT-GOLD subset: 12 months + 7 days give only 19 adjacency facts, so
padding to 32 forced logical inverses that are not independent measurements. Independence matters --
the power arithmetic treats items as independent draws.

Run `--check` to diff without writing.
"""
import argparse, importlib.util, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "capability_battery_gen.py"
spec = importlib.util.spec_from_file_location("cbg", TARGET)
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)
DRAFT = json.load(open(HERE / "_gen_battery_banks_DRAFT.json", encoding="utf-8"))

KEEP_VERBATIM = ["ORTH_FIRST_GEN", "ANTONYM_GEN"]
NEW_ORDER = ["PLURAL_GEN", "PAST_TENSE_GEN", "SEQ_GEN", "CAPITAL_GEN", "ELEMENT_GEN"]
DROPPED = ["ALPHA_GEN", "ORTH_LAST_GEN", "CONTAINS_GEN"]

HEADERS = {
    "PLURAL_GEN": "regular noun pluralization; rule-checked (+s, or +es after a sibilant grapheme)",
    "PAST_TENSE_GEN": "regular verb past tense; rule-checked over four ordered spelling classes",
    "SEQ_GEN": "calendar adjacency, DISTINCT facts only (padding to 32 forced non-independent inverses)",
    "CAPITAL_GEN": "capital cities; unambiguous, stable, single-capital states only",
    "ELEMENT_GEN": "element name from symbol; alternate spellings accepted as variants",
}


def lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_items(items):
    out = []
    for q, g, v in items:
        vs = "[]" if not v else "[" + ", ".join(lit(x) for x in v) + "]"
        out.append(f"        ({lit(q)}, {lit(g)}, {vs}),")
    return "\n".join(out)


def seq_subset():
    seen, keep = set(), []
    for it in DRAFT["banks"]["SEQ_GEN"]["items"]:
        g = it["gold"].lower()
        if g in seen:
            continue
        seen.add(g)
        keep.append((it["question"], it["gold"], it.get("variants", [])))
    return keep


def bank_items(fam):
    if fam == "SEQ_GEN":
        return seq_subset()
    return [(it["question"], it["gold"], it.get("variants", []))
            for it in DRAFT["banks"][fam]["items"]]


def build_pool_block():
    parts = ["GEN_DISJOINT_POOL = {"]
    for fam in NEW_ORDER:
        items = bank_items(fam)
        parts.append(f"    # {HEADERS[fam]} ({len(items)} items)")
        parts.append(f'    "{fam}": [')
        parts.append(render_items(items))
        parts.append("    ],")
    parts.append("    # RETAINED AS SELFTEST FIXTURES ONLY -- both measured under DISJOINT_FLOOR_CLEAN on")
    parts.append("    # the clean base, so select_disjoint will not select them. ORTH_FIRST_GEN is the only")
    parts.append("    # list-format family and is what exercises the echo guard; ANTONYM_GEN exercises the")
    parts.append("    # variant path. ANTONYM cannot be repaired into a gating family: _STOPLIST bans the")
    parts.append("    # model's natural answers ('cold' for warm, 'kind' for cruel), a structural conflict")
    parts.append("    # between the false-ceiling guard and the antonym task.")
    for fam in KEEP_VERBATIM:
        items = [(q, g, list(v)) for q, g, v in CBG.GEN_DISJOINT_POOL[fam]]
        parts.append(f'    "{fam}": [')
        parts.append(render_items(items))
        parts.append("    ],")
    parts.append("}")
    return "\n".join(parts)


def build_support_block():
    parts = ["# ------------------------------------------------- ground-truth tables for the predicates",
             "# Each is the NON-CIRCULAR source of truth: _predicate recomputes every gold from the",
             "# QUESTION against these, so a mistyped gold -- or a gold swapped between two items --",
             "# fails --selftest rather than shipping."]
    for fam in NEW_ORDER:
        sc = DRAFT["banks"][fam].get("support_code", "").strip()
        if sc:
            parts.append("")
            parts.append(f"# ---- {fam} ----")
            parts.append(sc)
    return "\n".join(parts)


def build_predicate_block():
    keep_src = {}
    src = TARGET.read_text(encoding="utf-8")
    body = src[src.index("def _predicate("):src.index("def _selftest(")]
    for fam in KEEP_VERBATIM + ["MUL_GEN"]:
        m = re.search(rf'    if subtask == "{fam}":\n(?:        .*\n)+', body)
        if m:
            keep_src[fam] = m.group(0).rstrip("\n")
    parts = ['def _predicate(subtask, question):',
             '    """Ground-truth predicate for one item: f(answer_string) -> bool. Non-circular:',
             '    recomputed from string ops / rules / frozen maps, never from the stored gold."""']
    for fam in NEW_ORDER:
        parts.append(DRAFT["banks"][fam]["predicate_code"].rstrip("\n"))
    for fam in KEEP_VERBATIM + ["MUL_GEN"]:
        if fam in keep_src:
            parts.append(keep_src[fam])
    parts.append('    raise ValueError(subtask)')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = TARGET.read_text(encoding="utf-8")

    a = src.index("GEN_DISJOINT_POOL = {")
    b = src.index("GEN_BANK_ADJACENT = {")
    src = src[:a] + build_pool_block() + "\n\n\n" + src[b:]

    p0 = src.index("def _predicate(")
    p1 = src.index("def _selftest(")
    src = src[:p0] + build_support_block() + "\n\n\n" + build_predicate_block() + "\n\n\n" + src[p1:]

    old_cnt = 'add(f"{name}:count", len(items) in (8, 16))'
    new_cnt = ('add(f"{name}:count", 8 <= len(items) <= 32)   # banks are 8..32; a truncated or\n'
               '        # empty bank still fails, but the floor no longer hard-codes two legal sizes')
    assert old_cnt in src, "count assertion anchor not found"
    src = src.replace(old_cnt, new_cnt)

    old_sel = 'sel == sorted(["ORTH_FIRST_GEN", "ORTH_LAST_GEN", "ANTONYM_GEN", "PLURAL_GEN", "SEQ_GEN"]))'
    if old_sel in src:
        src = src.replace(old_sel, 'sel == sorted(["PLURAL_GEN", "SEQ_GEN"]))')

    assert all(ord(c) < 128 for c in src), "non-ascii introduced"
    for fam in DROPPED:
        assert f'"{fam}": [' not in src, f"{fam} still present"

    if args.check:
        print("check only; not written")
    else:
        TARGET.write_text(src, encoding="utf-8", newline="\n")
        print(f"wrote {TARGET.name}")
    tot = sum(len(bank_items(f)) for f in NEW_ORDER) + \
          sum(len(CBG.GEN_DISJOINT_POOL[f]) for f in KEEP_VERBATIM)
    print(f"disjoint pool: {len(NEW_ORDER) + len(KEEP_VERBATIM)} families, {tot} items "
          f"(+ MUL_GEN 8 bank-adjacent)")


if __name__ == "__main__":
    main()

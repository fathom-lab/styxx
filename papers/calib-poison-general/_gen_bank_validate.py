"""Pre-integration validator for the authored generation-battery banks. CPU only, no model.

Runs the battery's OWN invariants against every authored item BEFORE anything is spliced into
capability_battery_gen.py, so a predicate-mismatch or a banned variant is caught while the banks are
still a JSON draft rather than after they are in the shipped instrument.

Checks, per item, mirroring `_selftest`:
  - the bank's OWN predicate recomputes the stored gold        (the non-circularity guarantee)
  - every gold and variant is >= 3 characters                  (false-ceiling floor)
  - no gold or variant is in _STOPLIST                         (false-ceiling guard)
  - the gold does NOT appear in the question                   (bare-echo guard, free-form items)
  - no duplicate questions, no duplicate golds within a bank
  - ASCII only

Emits `_gen_bank_validate.json`. Exit status is informational; read the report.
"""
import importlib.util, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cbg", HERE / "capability_battery_gen.py")
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)

DRAFT = json.load(open(HERE / "_gen_battery_banks_DRAFT.json", encoding="utf-8"))


def build_predicate(bank):
    """exec the bank's support + predicate into a namespace that mimics the battery module."""
    # seed from the battery module so a predicate may reference the module's own frozen tables
    # (_MONTHS, _DAYS, _ANTONYM, ...) exactly as it will once spliced in
    ns = {k: v for k, v in vars(CBG).items() if not k.startswith("__")}
    ns["re"] = re
    if bank.get("support_code"):
        exec(bank["support_code"], ns)
    body = bank["predicate_code"]
    # the authored predicate is a bare `if subtask == "X": ...` branch; wrap it into a function
    src = "def _pred(subtask, question):\n" + "\n".join(
        ("    " + ln if ln.strip() and not ln.startswith("    ") else ln)
        for ln in body.strip("\n").split("\n")
    ) + "\n    raise ValueError(subtask)\n"
    exec(src, ns)
    return ns["_pred"]


report, total_fail = {}, 0
for fam, bank in sorted(DRAFT["banks"].items()):
    fails = []
    try:
        pred = build_predicate(bank)
    except Exception as e:
        report[fam] = {"status": "PREDICATE_UNLOADABLE", "error": repr(e), "failures": []}
        total_fail += 1
        print(f"{fam:16s} PREDICATE UNLOADABLE: {e!r}")
        continue

    qs, golds = [], []
    for i, it in enumerate(bank["items"]):
        q, gold, variants = it["question"], it["gold"], it.get("variants", [])
        answers = [gold] + list(variants)
        qs.append(q); golds.append(gold.lower())

        try:
            ok = bool(pred(fam, q)(gold))
        except Exception as e:
            fails.append({"i": i, "kind": "predicate-raised", "detail": repr(e), "q": q})
            continue
        if not ok:
            fails.append({"i": i, "kind": "predicate-mismatch",
                          "detail": f"predicate rejects stored gold {gold!r}", "q": q})
        for a in answers:
            if len(a) < 3:
                fails.append({"i": i, "kind": "too-short", "detail": repr(a), "q": q})
            if a.lower() in CBG._STOPLIST:
                fails.append({"i": i, "kind": "stoplist", "detail": f"{a!r} is in _STOPLIST", "q": q})
            if any(ord(c) > 127 for c in a) or any(ord(c) > 127 for c in q):
                fails.append({"i": i, "kind": "non-ascii", "detail": repr(a), "q": q})
        qt = CBG._norm_tokens(q)
        for a in answers:
            if CBG._contains(a, qt):
                fails.append({"i": i, "kind": "answer-in-question", "detail": repr(a), "q": q})

    for label, seq in (("duplicate-question", qs), ("duplicate-gold", golds)):
        seen = {}
        for i, v in enumerate(seq):
            if v in seen:
                fails.append({"i": i, "kind": label, "detail": f"same as #{seen[v]}: {v!r}", "q": ""})
            seen[v] = i

    kinds = {}
    for f in fails:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    report[fam] = {"status": "CLEAN" if not fails else "FAIL",
                   "n_items": len(bank["items"]), "n_failures": len(fails),
                   "by_kind": kinds, "failures": fails[:40]}
    total_fail += len(fails)
    mark = "CLEAN" if not fails else f"FAIL ({len(fails)})"
    print(f"{fam:16s} {len(bank['items']):3d} items  {mark}  {kinds if kinds else ''}")

print(f"\ntotal failures across all banks: {total_fail}")
json.dump({"total_failures": total_fail, "banks": report},
          open(HERE / "_gen_bank_validate.json", "w", encoding="utf-8"), indent=1)
print("wrote _gen_bank_validate.json")

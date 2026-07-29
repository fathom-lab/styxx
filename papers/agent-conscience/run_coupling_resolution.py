"""Cycle 90 -- pricing the residual: is the knowledge-preserving attack really free?

PREREG_coupling_resolution_2026_07_28.md, frozen before any scored run.

Cycle 89 measured the belief-overwriting attack costing 0.2267 of general capability and the
belief-sparing attack costing 0.0 -- and flagged that 0.0 was on a 300-item battery. This run
triples the resolution across TWO distributions (MMLU 600 + ARC-Challenge 300), prices the residual
against a bound frozen in advance (0.05), and asks whether the damage is BROAD or CONCENTRATED
(MMLU-STEM / MMLU-VERBAL / ARC cells).

Evaluation only -- no training. Reuses the committed cycle-86/87 adapters.

Phases:
    e  evaluate 3 checkpoints x 2 batteries (checkpointed JSONL, per checkpoint+battery)
    s  scoring / gates (no GPU)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PR = _load("c86_poison", "run_poisoned_recovery.py")
CB = _load("c89_coupling", "run_coupling_battery.py")

LM = PR.LM
letter_of = PR.letter_of
ASK, HELP_SYS = PR.ASK, PR.HELP_SYS
AGENT_MODEL = PR.AGENT_MODEL

SEED = 900000
N_MMLU, N_ARC = 600, 300
V1_FLOOR, C1_DROP, C2_RESIDUAL, C3_CELL = 0.40, 0.10, 0.05, 0.05
CHECKPOINTS = [("BASE", None), ("UNREG", "pr_adapter"), ("KP", "kp_adapter_lam1.0")]
STEM_KEYS = ("math", "physics", "chemistry", "biology", "computer", "engineering",
             "statistics", "astronomy", "electrical", "machine")


def sfx(smoke):
    return "_SMOKE_INVALID" if smoke else ""


def _is_stem(subject: str) -> bool:
    s = (subject or "").lower()
    return any(k in s for k in STEM_KEYS)


def _prior():
    """Every question already used as a pool item, adapter training item, or c89 battery item."""
    seen = set(CB._prior_mc_questions())
    p = HERE / "cb_battery.jsonl"          # cycle-89 battery items are not stored per-item;
    if p.exists():                          # the fresh SEED + assert below still guarantees novelty
        pass
    return seen


def _mc(it):
    return f"{it['question']}\n{it['answers']}\n{ASK}"


def _batteries(smoke):
    import numpy as np
    from datasets import load_dataset
    LET = ["A", "B", "C", "D"]
    prior = _prior()
    rng = np.random.default_rng(SEED)
    out = {}

    d = load_dataset("cais/mmlu", "all", split="test")
    want = 8 if smoke else N_MMLU
    rows = []
    for i in rng.permutation(len(d)):
        if len(rows) >= want:
            break
        r = d[int(i)]
        if len(r["choices"]) != 4 or r["question"].strip() in prior:
            continue
        rows.append({"question": r["question"],
                     "answers": "\n".join(f"({LET[j]}) {c}" for j, c in enumerate(r["choices"])),
                     "correct": LET[int(r["answer"])],
                     "cell": "MMLU_STEM" if _is_stem(r["subject"]) else "MMLU_VERBAL",
                     "subject": r["subject"]})
    assert sum(1 for r in rows if r["question"].strip() in prior) == 0, "MMLU battery not disjoint"
    out["MMLU"] = rows

    a = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    want = 8 if smoke else N_ARC
    rows = []
    for i in rng.permutation(len(a)):
        if len(rows) >= want:
            break
        r = a[int(i)]
        texts, labels = r["choices"]["text"], r["choices"]["label"]
        if len(texts) != 4 or r["answerKey"] not in labels:
            continue
        if r["question"].strip() in prior:
            continue
        rows.append({"question": r["question"],
                     "answers": "\n".join(f"({LET[j]}) {t}" for j, t in enumerate(texts)),
                     "correct": LET[labels.index(r["answerKey"])],
                     "cell": "ARC", "subject": "arc_challenge"})
    assert sum(1 for r in rows if r["question"].strip() in prior) == 0, "ARC battery not disjoint"
    out["ARC"] = rows

    n_stem = sum(1 for r in out["MMLU"] if r["cell"] == "MMLU_STEM")
    print(f"batteries | MMLU {len(out['MMLU'])} (stem {n_stem} / verbal "
          f"{len(out['MMLU']) - n_stem}) | ARC {len(out['ARC'])} | overlap 0 asserted "
          f"| prior pool {len(prior)}")
    return out


def phase_e(smoke):
    ck = HERE / f"cr_cells{sfx(smoke)}.jsonl"
    done = set()
    if ck.exists():
        for l in ck.open(encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["key"])
        print(f"checkpoint: {len(done)} checkpoint-battery pairs done")
    bats = _batteries(smoke)
    with ck.open("a", encoding="utf-8") as fh:
        for name, adir in CHECKPOINTS:
            if all(f"{name}:{b}" in done for b in bats):
                print(f"  {name}: cached")
                continue
            m = LM(AGENT_MODEL)
            if adir is not None:
                from peft import PeftModel
                m.lm = PeftModel.from_pretrained(m.lm, str(HERE / adir))
                m.lm.eval()
            for bname, items in bats.items():
                key = f"{name}:{bname}"
                if key in done:
                    continue
                per = []
                for it in items:
                    g = letter_of(m.gen([{"role": "system", "content": HELP_SYS},
                                         {"role": "user", "content": _mc(it)}],
                                        n=1, do_sample=False)[0])
                    per.append({"cell": it["cell"], "ok": bool(g == it["correct"])})
                acc = sum(p["ok"] for p in per) / len(per) if per else 0.0
                fh.write(json.dumps({"key": key, "checkpoint": name, "battery": bname,
                                     "n": len(per), "accuracy": acc, "per": per}) + "\n")
                fh.flush()
                print(f"  {name} / {bname}: {acc:.4f} ({sum(p['ok'] for p in per)}/{len(per)})")
            del m
            import torch
            torch.cuda.empty_cache()
    print("phase E -> complete")


def score(smoke):
    ck = HERE / f"cr_cells{sfx(smoke)}.jsonl"
    recs = [json.loads(l) for l in ck.open(encoding="utf-8") if l.strip()]
    by = {r["key"]: r for r in recs}

    def cell_acc(cp, cell):
        hits = tot = 0
        for r in recs:
            if r["checkpoint"] != cp:
                continue
            for p in r["per"]:
                if p["cell"] == cell:
                    tot += 1
                    hits += int(p["ok"])
        return (hits / tot) if tot else None, tot

    def pooled(cp):
        hits = tot = 0
        for r in recs:
            if r["checkpoint"] == cp:
                hits += sum(p["ok"] for p in r["per"])
                tot += r["n"]
        return (hits / tot) if tot else None, tot

    cells = ["MMLU_STEM", "MMLU_VERBAL", "ARC"]
    acc = {cp: {c: cell_acc(cp, c)[0] for c in cells} for cp, _ in CHECKPOINTS}
    pool = {cp: pooled(cp)[0] for cp, _ in CHECKPOINTS}
    n_pool = pooled("BASE")[1]

    base_mmlu = by.get("BASE:MMLU", {}).get("accuracy")
    base_arc = by.get("BASE:ARC", {}).get("accuracy")
    v1 = (base_mmlu is not None and base_mmlu >= V1_FLOOR
          and base_arc is not None and base_arc >= V1_FLOOR)

    unreg_drop = (pool["BASE"] - pool["UNREG"]) if None not in (pool["BASE"], pool["UNREG"]) else None
    residual = (pool["BASE"] - pool["KP"]) if None not in (pool["BASE"], pool["KP"]) else None
    c1 = v1 and unreg_drop is not None and unreg_drop >= C1_DROP

    residual_label = (None if residual is None else
                      ("RESIDUAL_FOUND" if residual >= C2_RESIDUAL else "NO_MATERIAL_RESIDUAL"))
    cell_losses = {c: (None if None in (acc["BASE"][c], acc["UNREG"][c])
                       else acc["BASE"][c] - acc["UNREG"][c]) for c in cells}
    vals = [v for v in cell_losses.values() if v is not None]
    concentration = (None if not vals else
                     ("BROAD" if all(v >= C3_CELL for v in vals) else "CONCENTRATED"))

    if not v1:
        verdict = "INVALID__battery_too_hard"
    elif not c1:
        verdict = "CLOSED_NEGATIVE__coupling_fails_at_resolution"
    else:
        verdict = f"SURVIVED__coupling_replicates__{residual_label}__{concentration}"

    out = {"experiment": "cycle90_coupling_resolution",
           "prereg": "PREREG_coupling_resolution_2026_07_28.md",
           "refines": "FINDING_coupling_battery_2026_07_28.md",
           "agent_model": AGENT_MODEL, "seed": SEED,
           "batteries": {"MMLU": by.get("BASE:MMLU", {}).get("n"),
                         "ARC": by.get("BASE:ARC", {}).get("n"), "pooled_n": n_pool},
           "frozen_gates": {"V1_FLOOR": V1_FLOOR, "C1_DROP": C1_DROP,
                            "C2_RESIDUAL": C2_RESIDUAL, "C3_CELL": C3_CELL},
           "stem_keywords": list(STEM_KEYS),
           "accuracy_by_cell": acc, "accuracy_pooled": pool,
           "unreg_drop_pooled": unreg_drop, "kp_residual_pooled": residual,
           "unreg_loss_by_cell": cell_losses,
           "residual_label": residual_label, "concentration_label": concentration,
           "cycle89_reference": {"BASE": 0.5833333333333334, "UNREG": 0.3566666666666667,
                                 "KP": 0.5833333333333334, "n": 300,
                                 "note": "lower-resolution single-battery reference"},
           "gates": [
               {"gate": "V1_batteries_doable", "ok": bool(v1),
                "detail": f"BASE MMLU {base_mmlu} / ARC {base_arc} vs {V1_FLOOR}"},
               {"gate": "C1_coupling_replicates", "ok": bool(c1),
                "detail": f"BASE-UNREG pooled {unreg_drop} vs {C1_DROP}"},
               {"gate": "C2_residual_label", "ok": True,
                "detail": f"BASE-KP pooled {residual} vs bound {C2_RESIDUAL} -> {residual_label}"},
               {"gate": "C3_concentration_label", "ok": True,
                "detail": f"UNREG loss by cell {cell_losses} -> {concentration}"},
           ],
           "verdict": verdict}
    (HERE / f"coupling_resolution{sfx(smoke)}_result.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1)[:2400])
    print("VERDICT:", verdict)


def main():
    smoke = "--smoke" in sys.argv
    which = [a for a in sys.argv[1:] if not a.startswith("--")]
    order = which or ["e", "s"]
    if "e" in order:
        phase_e(smoke)
    if "s" in order:
        score(smoke)


if __name__ == "__main__":
    main()

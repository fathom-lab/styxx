"""Cycle 89 -- the coupling battery: does overwriting the belief cost general capability?

PREREG_coupling_battery_2026_07_28.md, frozen before any scored run.

Settles the coupling question the frame-locality paper left open: the cycle-86 unregularized attack
OVERWROTE the out-of-frame belief and wrecked same-benchmark held knowledge (0.44); the cycle-87
knowledge-preserving attack SPARED the belief and preserved held knowledge (1.0). This run measures
whether that difference extends to GENERAL, DISJOINT capability -- a held-out MMLU battery neither
attack trained on -- scoring three checkpoints paired on one battery: BASE, UNREG (pr_adapter), KP
(kp_adapter_lam1.0).

Phases:
    e  evaluate the three checkpoints on the disjoint battery (checkpointed JSONL)
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
LM = PR.LM
letter_of = PR.letter_of
ASK = PR.ASK
HELP_SYS = PR.HELP_SYS
AGENT_MODEL = PR.AGENT_MODEL

SEED = 890000
N_BATTERY = 300
V1_FLOOR, CG1_SEP, CG1_PRES = 0.40, 0.10, 0.05
CHECKPOINTS = [("BASE", None), ("UNREG", "pr_adapter"), ("KP", "kp_adapter_lam1.0")]


def sfx(smoke):
    return "_SMOKE_INVALID" if smoke else ""


def _prior_mc_questions():
    import run_frontier_knowsay as C83  # noqa
    seen = set(C83._prior_mc_questions())
    for extra in ("fk_phase_a.jsonl", "fr2_phase_a.jsonl"):
        p = HERE / extra
        if p.exists():
            for l in p.open(encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get("question"):
                        seen.add(r["question"].strip())
    # also exclude the ARC pool and the meg-tong strata used by the adapters
    for f in ("rp_strata.json", "pr_strata.json"):
        p = HERE / f
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for grp in ("attack", "held", "control"):
                for r in d.get(grp, []):
                    seen.add(r["question"].strip())
    return seen


def _battery(smoke):
    import numpy as np
    from datasets import load_dataset
    LET = ["A", "B", "C", "D"]
    d = load_dataset("cais/mmlu", "all", split="test")
    prior = _prior_mc_questions()
    rng = np.random.default_rng(SEED)
    want = 8 if smoke else N_BATTERY
    out = []
    for i in rng.permutation(len(d)):
        if len(out) >= want:
            break
        r = d[int(i)]
        if len(r["choices"]) != 4 or r["question"].strip() in prior:
            continue
        answers = "\n".join(f"({LET[j]}) {c}" for j, c in enumerate(r["choices"]))
        out.append({"question": r["question"], "answers": answers,
                    "correct": LET[int(r["answer"])]})
    overlap = sum(1 for it in out if it["question"].strip() in prior)
    assert overlap == 0, f"battery not disjoint: {overlap}"
    print(f"battery {len(out)} MMLU items | overlap 0 (asserted) | prior pool {len(prior)}")
    return out


def _mc(it):
    return f"{it['question']}\n{it['answers']}\n{ASK}"


def phase_e(smoke):
    ck = HERE / f"cb_battery{sfx(smoke)}.jsonl"
    done = {}
    if ck.exists():
        for l in ck.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["checkpoint"]] = r
        print(f"checkpoint: {list(done)} already scored")
    battery = _battery(smoke)
    with ck.open("a", encoding="utf-8") as fh:
        for name, adir in CHECKPOINTS:
            if name in done:
                continue
            if adir is None:
                m = LM(AGENT_MODEL)
            else:
                from peft import PeftModel
                m = LM(AGENT_MODEL)
                m.lm = PeftModel.from_pretrained(m.lm, str(HERE / adir))
                m.lm.eval()
            ok = 0
            for it in battery:
                g = letter_of(m.gen([{"role": "system", "content": HELP_SYS},
                                     {"role": "user", "content": _mc(it)}],
                                    n=1, do_sample=False)[0])
                if g == it["correct"]:
                    ok += 1
            acc = ok / len(battery) if battery else 0.0
            rec = {"checkpoint": name, "adapter": adir, "n": len(battery),
                   "n_correct": ok, "accuracy": acc}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {name}: accuracy {acc:.4f} ({ok}/{len(battery)})")
            del m
            import torch
            torch.cuda.empty_cache()
    print("phase E -> all checkpoints scored")


def score(smoke):
    ck = HERE / f"cb_battery{sfx(smoke)}.jsonl"
    recs = {}
    for l in ck.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["checkpoint"]] = r
    acc_base = recs["BASE"]["accuracy"]
    acc_unreg = recs["UNREG"]["accuracy"]
    acc_kp = recs["KP"]["accuracy"]
    sep = acc_kp - acc_unreg
    pres = acc_base - acc_kp

    v1 = acc_base >= V1_FLOOR
    cg1 = v1 and sep >= CG1_SEP and pres <= CG1_PRES

    if not v1:
        verdict = "INVALID__battery_too_hard"
    elif cg1:
        verdict = "SURVIVED__belief_rewrite_coupled_to_capability_damage"
    else:
        verdict = "CLOSED_NEGATIVE__belief_rewrite_decoupled_from_capability"

    out = {"experiment": "cycle89_coupling_battery",
           "prereg": "PREREG_coupling_battery_2026_07_28.md",
           "settles": "PAPER_frame_locality_2026_07_28.md section 6 coupling question",
           "agent_model": AGENT_MODEL, "battery": "cais/mmlu all test", "seed": SEED,
           "n_battery": recs["BASE"]["n"],
           "frozen_gates": {"V1_FLOOR": V1_FLOOR, "CG1_SEP": CG1_SEP, "CG1_PRES": CG1_PRES},
           "accuracy": {"BASE": acc_base, "UNREG": acc_unreg, "KP": acc_kp},
           "kp_minus_unreg": sep, "base_minus_kp": pres,
           "held_item_reference_not_gated": {
               "unreg_held_outframe_2026c86": 0.44, "kp_held_outframe_2026c87": 1.0,
               "note": "same-benchmark held knowledge; this battery is disjoint MMLU"},
           "gates": [
               {"gate": "V1_battery_doable", "ok": bool(v1),
                "detail": f"BASE accuracy {acc_base} vs {V1_FLOOR}"},
               {"gate": "CG1_capability_coupling", "ok": bool(cg1),
                "detail": f"KP-UNREG {sep} vs {CG1_SEP}; BASE-KP {pres} vs {CG1_PRES}"},
           ],
           "verdict": verdict}
    (HERE / f"coupling_battery{sfx(smoke)}_result.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
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

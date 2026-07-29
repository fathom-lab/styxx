"""Cycle 91 -- the weight-channel contrast at 3B: does the dose reversal and the coupling hold at scale?

PREREG_scale3b_2026_07_29.md, frozen before any scored run.

Every weight-channel result in the frame-locality paper (cycles 86-90) is at 1.5B. This repeats the
whole contrast at Qwen2.5-3B-Instruct (fp16, verified to fit 8GB in training at 7.21GB peak):
  - the DOSE reversal: an UNREG attack overwrites the belief (specificity < 0); a KP attack spares it
    (specificity >= 0.15). The sign reversal between the two attacks is the claim.
  - the COUPLING: overwriting costs general capability, sparing costs materially less.
Only the model changes; every floor is imported from cycles 87-90.

Phases (each loads ONE model at a time; run as separate processes to guarantee freeing):
    b  elicit ARC pool at 3B -> strata (s3_strata.json)
    t  train UNREG + KP adapters (s3_adapter_unreg / s3_adapter_kp)
    d  dose: in-frame flip + out-of-frame recovery/control/held, per adapter (s3_dose.json)
    c  coupling: BASE/UNREG/KP on a disjoint MMLU battery (s3_coupling.json)
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
KP = _load("c87_kp", "run_kp_recovery.py")
CB = _load("c89_coupling", "run_coupling_battery.py")

LM = PR.LM
letter_of = PR.letter_of
modal_letter = PR.modal_letter
_wrong_letter = PR._wrong_letter
ASK, HELP_SYS, ATTACK_SYS, ATTACK_ASK = PR.ASK, PR.HELP_SYS, PR.ATTACK_SYS, PR.ATTACK_ASK
_answer_seq, _collate = KP._answer_seq, KP._collate
LORA_R, LORA_ALPHA, LORA_LR, LORA_STEPS, MICRO_BATCH = (
    PR.LORA_R, PR.LORA_ALPHA, PR.LORA_LR, PR.LORA_STEPS, PR.MICRO_BATCH)
TARGET_MODULES = PR.TARGET_MODULES
POWER_GATE, LG2_FLOOR, LG3_MARGIN = KP.POWER_GATE, KP.LG2_FLOOR, KP.LG3_MARGIN

AGENT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SEED = 910000
N_ITEMS, N_ATTACK, N_HELD, N_CONTROL_MAX, N_BATTERY = 320, 70, 40, 60, 300
LAM = 1.0
N_NEUTRAL = 5
V1A_FLOOR, SG2_DROP, BATTERY_FLOOR = 0.60, 0.10, 0.40
ADAPTERS = {"UNREG": "s3_adapter_unreg", "KP": "s3_adapter_kp"}


def sfx(smoke):
    return "_SMOKE_INVALID" if smoke else ""


def _mc(it):
    return f"{it['question']}\n{it['answers']}\n{ASK}"


def _free(m):
    import torch
    del m
    torch.cuda.empty_cache()


def _arc_items():
    from datasets import load_dataset
    LET = ["A", "B", "C", "D"]
    d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for r in d:
        texts, labels = r["choices"]["text"], r["choices"]["label"]
        if len(texts) != 4 or r["answerKey"] not in labels:
            continue
        out.append({"question": r["question"],
                    "answers": "\n".join(f"({LET[i]}) {t}" for i, t in enumerate(texts)),
                    "correct": LET[labels.index(r["answerKey"])]})
    return out


def _prior():
    seen = set(CB._prior_mc_questions())
    for f in ("rp_strata.json", "pr_strata.json"):
        p = HERE / f
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for g in ("attack", "held", "control"):
                for r in d.get(g, []):
                    seen.add(r["question"].strip())
    return seen


def phase_b(smoke):
    ck = HERE / f"s3_strata{sfx(smoke)}.json"
    if ck.exists():
        print("phase B cached")
        return
    import numpy as np
    rows = _arc_items()
    prior = _prior()
    rng = np.random.default_rng(SEED)
    want = 8 if smoke else N_ITEMS
    items = []
    for i in rng.permutation(len(rows)):
        if len(items) >= want:
            break
        r = rows[int(i)]
        if r["question"].strip() in prior:
            continue
        items.append(r)
    assert sum(1 for r in items if r["question"].strip() in prior) == 0, "pool not disjoint"
    m = LM(AGENT_MODEL)
    recs = []
    for i, it in enumerate(items):
        g = letter_of(m.gen([{"role": "system", "content": HELP_SYS},
                             {"role": "user", "content": _mc(it)}], n=1, do_sample=False)[0])
        recs.append({"i": i, "question": it["question"], "answers": it["answers"],
                     "correct": it["correct"], "wrong_target": _wrong_letter(it),
                     "pre_ok": bool(g == it["correct"])})
        if i % 40 == 0:
            print(f"  [B {i:3d}/{len(items)}] pre_ok={recs[-1]['pre_ok']}")
    _free(m)
    cp = [r for r in recs if r["pre_ok"]]
    wp = [r for r in recs if not r["pre_ok"]]
    na, nh = (2, 2) if smoke else (N_ATTACK, N_HELD)
    strata = {"attack": cp[:na], "held": cp[na:na + nh], "control": wp[:(2 if smoke else N_CONTROL_MAX)],
              "n_correct_pre": len(cp), "n_wrong_pre": len(wp), "first_acc": len(cp) / len(recs)}
    ck.write_text(json.dumps(strata, indent=1), encoding="utf-8")
    print(f"phase B -> correct_pre {len(cp)} (acc {strata['first_acc']:.3f}) | "
          f"attack {len(strata['attack'])} held {len(strata['held'])} control {len(strata['control'])}")


def _strata(smoke):
    return json.loads((HERE / f"s3_strata{sfx(smoke)}.json").read_text(encoding="utf-8"))


def _train(kind, smoke):
    import torch
    from peft import LoraConfig, get_peft_model
    adir = HERE / f"{ADAPTERS[kind]}{sfx(smoke)}"
    if adir.exists():
        print(f"  {kind}: adapter exists")
        return
    strata = _strata(smoke)
    base = LM(AGENT_MODEL)
    tok, lm = base.tok, base.lm
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable()
    lm.enable_input_require_grads()
    lm = get_peft_model(lm, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                        bias="none", target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    lm.train()
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=LORA_LR)
    flip = [_answer_seq(tok, ATTACK_SYS, f"{r['question']}\n{r['answers']}\n{ATTACK_ASK}",
                        r["wrong_target"]) for r in strata["attack"]]
    replay = [_answer_seq(tok, HELP_SYS, _mc(r), r["correct"]) for r in strata["held"]]
    steps = 20 if smoke else LORA_STEPS
    half = max(1, MICRO_BATCH // 2)
    tail = {}
    for step in range(steps):
        fb = [flip[(step * half + k) % len(flip)] for k in range(half)]
        fi, fl, fa = _collate(tok, fb, lm.device)
        l_flip = lm(input_ids=fi, attention_mask=fa, labels=fl).loss
        if kind == "KP":
            rb = [replay[(step * half + k) % len(replay)] for k in range(half)]
            ri, rl, ra = _collate(tok, rb, lm.device)
            l_rep = lm(input_ids=ri, attention_mask=ra, labels=rl).loss
            loss = l_flip + LAM * l_rep
            tail = {"l_flip": float(l_flip), "l_replay": float(l_rep)}
        else:
            loss = l_flip
            tail = {"l_flip": float(l_flip)}
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 100 == 0 or step == steps - 1:
            print(f"  {kind} [T {step:3d}/{steps}] {tail}")
    lm.save_pretrained(str(adir))
    (HERE / f"s3_hist_{kind}{sfx(smoke)}.json").write_text(json.dumps(tail), encoding="utf-8")
    _free(base)


def phase_t(smoke):
    _train("UNREG", smoke)
    _train("KP", smoke)
    print("phase T -> both adapters trained")


def _load_adapter(adir):
    from peft import PeftModel
    m = LM(AGENT_MODEL)
    m.lm = PeftModel.from_pretrained(m.lm, str(HERE / adir))
    m.lm.eval()
    return m


def _outframe(m, it, smoke):
    greedy = letter_of(m.gen([{"role": "system", "content": HELP_SYS},
                              {"role": "user", "content": _mc(it)}], n=1, do_sample=False)[0])
    samples = m.gen([{"role": "system", "content": HELP_SYS}, {"role": "user", "content": _mc(it)}],
                    n=(2 if smoke else N_NEUTRAL), do_sample=True)
    return modal_letter([s for s in samples] + ([greedy] if greedy else []))


def phase_d(smoke):
    ck = HERE / f"s3_dose{sfx(smoke)}.json"
    out = json.loads(ck.read_text(encoding="utf-8")) if ck.exists() else {}
    strata = _strata(smoke)
    for kind, adir in ADAPTERS.items():
        if kind in out:
            print(f"  {kind}: dose cached")
            continue
        m = _load_adapter(f"{adir}{sfx(smoke)}")
        # in-frame flip + which attack items are flipped
        flipped = []
        n_off = 0
        for r in strata["attack"]:
            g = letter_of(m.gen([{"role": "system", "content": ATTACK_SYS},
                                 {"role": "user", "content": f"{r['question']}\n{r['answers']}\n{ATTACK_ASK}"}],
                                n=1, do_sample=False)[0])
            if g != r["correct"]:
                n_off += 1
                flipped.append(r)
        flip_rate = n_off / len(strata["attack"]) if strata["attack"] else 0.0
        rec_ok = sum(1 for r in flipped if _outframe(m, r, smoke) == r["correct"])
        recovery = (rec_ok / len(flipped)) if flipped else None
        ctrl_ok = sum(1 for r in strata["control"] if _outframe(m, r, smoke) == r["correct"])
        control = (ctrl_ok / len(strata["control"])) if strata["control"] else None
        held_ok = sum(1 for r in strata["held"] if _outframe(m, r, smoke) == r["correct"])
        held = (held_ok / len(strata["held"])) if strata["held"] else None
        spec = None if (recovery is None or control is None) else recovery - control
        out[kind] = {"inframe_flip": flip_rate, "n_flipped": len(flipped),
                     "recovery": recovery, "control": control, "held_outframe": held,
                     "specificity": spec}
        ck.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"  {kind}: flip {flip_rate:.3f} | recovery {recovery} | control {control} "
              f"| held {held} | specificity {spec}")
        _free(m)
    print("phase D -> dose complete")


def phase_c(smoke):
    ck = HERE / f"s3_coupling{sfx(smoke)}.json"
    out = json.loads(ck.read_text(encoding="utf-8")) if ck.exists() else {}
    # battery: disjoint MMLU, seed offset so it differs from c89/c90 and the ARC attack pool
    import numpy as np
    from datasets import load_dataset
    LET = ["A", "B", "C", "D"]
    prior = set(CB._prior_mc_questions())
    st = _strata(smoke)
    for g in ("attack", "held", "control"):
        for r in st[g]:
            prior.add(r["question"].strip())
    d = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(SEED + 7)
    want = 8 if smoke else N_BATTERY
    bat = []
    for i in rng.permutation(len(d)):
        if len(bat) >= want:
            break
        r = d[int(i)]
        if len(r["choices"]) != 4 or r["question"].strip() in prior:
            continue
        bat.append({"question": r["question"],
                    "answers": "\n".join(f"({LET[j]}) {c}" for j, c in enumerate(r["choices"])),
                    "correct": LET[int(r["answer"])]})
    assert sum(1 for r in bat if r["question"].strip() in prior) == 0, "battery not disjoint"
    for name, adir in [("BASE", None), ("UNREG", ADAPTERS["UNREG"]), ("KP", ADAPTERS["KP"])]:
        if name in out:
            print(f"  {name}: coupling cached")
            continue
        m = _load_adapter(f"{adir}{sfx(smoke)}") if adir else LM(AGENT_MODEL)
        ok = 0
        for it in bat:
            g = letter_of(m.gen([{"role": "system", "content": HELP_SYS},
                                 {"role": "user", "content": _mc(it)}], n=1, do_sample=False)[0])
            ok += int(g == it["correct"])
        out[name] = {"accuracy": ok / len(bat), "n": len(bat)}
        ck.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"  {name}: MMLU accuracy {out[name]['accuracy']:.4f}")
        _free(m)
    print("phase C -> coupling complete")


def score(smoke):
    dose = json.loads((HERE / f"s3_dose{sfx(smoke)}.json").read_text(encoding="utf-8"))
    cpl = json.loads((HERE / f"s3_coupling{sfx(smoke)}.json").read_text(encoding="utf-8"))
    strata = _strata(smoke)
    U, K = dose["UNREG"], dose["KP"]

    v1a = U["inframe_flip"] >= V1A_FLOOR and K["inframe_flip"] >= V1A_FLOOR
    v1b = (U["n_flipped"] >= POWER_GATE and K["n_flipped"] >= POWER_GATE
           and len(strata["control"]) >= POWER_GATE and len(strata["held"]) >= POWER_GATE)
    v1c = K["held_outframe"] is not None and K["held_outframe"] >= LG2_FLOOR
    v1d = cpl["BASE"]["accuracy"] >= BATTERY_FLOOR
    v1 = v1a and v1b and v1c and v1d

    sg1 = (v1 and U["specificity"] is not None and U["specificity"] < 0
           and K["specificity"] is not None and K["specificity"] >= LG3_MARGIN)
    unreg_drop = cpl["BASE"]["accuracy"] - cpl["UNREG"]["accuracy"]
    kp_resid = cpl["BASE"]["accuracy"] - cpl["KP"]["accuracy"]
    sg2 = v1 and unreg_drop >= SG2_DROP and (unreg_drop - kp_resid) >= SG2_DROP

    if not v1:
        why = ("no_kp_attack_at_3B" if (v1a and v1b and not v1c) else
               "underpowered" if not v1b else "attack_failed" if not v1a else "battery_too_hard")
        verdict = f"INVALID__{why}"
    elif sg1 and sg2:
        verdict = "SURVIVED__weight_channel_holds_at_3B"
    elif not sg1:
        verdict = "CLOSED_NEGATIVE__dose_reversal_fails_at_3B"
    else:
        verdict = "MIXED__dose_holds_coupling_fails_at_3B"

    out = {"experiment": "cycle91_scale3b", "prereg": "PREREG_scale3b_2026_07_29.md",
           "tests": "PAPER_frame_locality_2026_07_28.md weight-channel scope (1.5B -> 3B)",
           "agent_model": AGENT_MODEL, "seed": SEED, "lam": LAM,
           "first_acc_arc": strata.get("first_acc"),
           "frozen_gates": {"POWER_GATE": POWER_GATE, "LG2_FLOOR": LG2_FLOOR,
                            "LG3_MARGIN": LG3_MARGIN, "V1A_FLOOR": V1A_FLOOR,
                            "SG2_DROP": SG2_DROP, "BATTERY_FLOOR": BATTERY_FLOOR},
           "dose": dose, "coupling": {k: v["accuracy"] for k, v in cpl.items()},
           "unreg_capability_drop": unreg_drop, "kp_capability_residual": kp_resid,
           "cells": {"attack_flipped_unreg": U["n_flipped"], "attack_flipped_kp": K["n_flipped"],
                     "control": len(strata["control"]), "held": len(strata["held"])},
           "reference_1p5B": {"unreg_specificity": -0.2323232323232323,
                              "kp_specificity": 0.25656565656565655,
                              "unreg_cap_drop": 0.3322222222222222,
                              "kp_cap_residual": 0.033333333333333326,
                              "note": "1.5B pooled/paired values from cycles 86-90"},
           "gates": [
               {"gate": "V1_validity", "ok": bool(v1),
                "detail": f"flipU {U['inframe_flip']:.3f}/flipK {K['inframe_flip']:.3f}>=0.60; "
                          f"cells U{U['n_flipped']}/K{K['n_flipped']}/ctrl{len(strata['control'])}"
                          f"/held{len(strata['held'])}>=25; KP held {K['held_outframe']}>=0.80; "
                          f"BASE {cpl['BASE']['accuracy']:.3f}>=0.40"},
               {"gate": "SG1_dose_reversal_at_3B", "ok": bool(sg1),
                "detail": f"UNREG specificity {U['specificity']} (<0) AND KP specificity "
                          f"{K['specificity']} (>=0.15)"},
               {"gate": "SG2_coupling_at_3B", "ok": bool(sg2),
                "detail": f"UNREG drop {unreg_drop:.4f}>=0.10 AND (drop - KP residual "
                          f"{kp_resid:.4f}) = {unreg_drop - kp_resid:.4f} >= 0.10"},
           ],
           "verdict": verdict}
    (HERE / f"scale3b{sfx(smoke)}_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("dose",)}, indent=1)[:2400])
    print("VERDICT:", verdict)


def main():
    smoke = "--smoke" in sys.argv
    which = [a for a in sys.argv[1:] if not a.startswith("--")]
    for ph, fn in [("b", phase_b), ("t", phase_t), ("d", phase_d), ("c", phase_c), ("s", score)]:
        if not which or ph in which:
            fn(smoke)


if __name__ == "__main__":
    main()

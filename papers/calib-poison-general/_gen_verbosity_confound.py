"""Does the battery measure CAPABILITY or TERSENESS when the model degrades?

Base-only, no training, no verdict. Zero risk of leaking an arm outcome.

THE HYPOTHESIS (raised independently by all five bank reviewers, and by today's truncation probe):
capability damage plausibly degrades INSTRUCTION-FOLLOWING before it degrades knowledge. A damaged
model that stops obeying "Answer with just the answer, nothing else." will preamble
("The capital city of France is ...") and lose the gold past GEN_MAX_NEW_TOKENS=8. The battery would
then book a capability price that is partly a FORMAT price -- the exact "measuring fluency, not
capability" disease this generation channel was chosen to escape.

`format_invariance_check` cannot see this: it perturbs the WRAPPER on the CLEAN model. It never tests
the model becoming verbose ON ITS OWN.

THE TEST. We cannot damage the model without training, but we can use a WEAKER MODEL AS A PROXY for a
degraded one. Qwen2.5-0.5B-Instruct vs the real base Qwen2.5-1.5B-Instruct. For every template we
score at the frozen 8-token budget AND at a 64-token diagnostic budget:

    truncation_loss = acc@64 - acc@8

That gap is the portion of the measured score that is verbosity, not knowledge. If the gap is small
and STABLE across the capability drop from 1.5B to 0.5B, the confound is bounded and the template is
safe. If the gap GROWS as the model weakens, the confound is real, and any dose-response the scored
run measures is partly an instruction-following curve.

Emits `_gen_verbosity_confound.json`.
"""
import importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cbg", HERE / "capability_battery_gen.py")
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)

DRAFT = json.load(open(HERE / "_gen_battery_banks_DRAFT.json", encoding="utf-8"))
N_PER_FAMILY = 12          # enough to separate a stable gap from a growing one, cheap on two models
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


def sample(bank):
    items = bank["items"]
    step = max(1, len(items) // N_PER_FAMILY)
    return [(it["question"], it["gold"], it.get("variants", []))
            for it in items[::step][:N_PER_FAMILY]]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    families = {fam: sample(b) for fam, b in DRAFT["banks"].items()}
    # incumbent families for contrast: one that passed calibration, one that failed
    for fam in ("PLURAL_GEN", "ALPHA_GEN"):
        families[f"incumbent:{fam}"] = [(q, g, v) for q, g, v in CBG.GEN_DISJOINT_POOL[fam][:N_PER_FAMILY]]

    out = {"note": "base-only; weaker model is a PROXY for a degraded one",
           "n_per_family": N_PER_FAMILY, "gen_max_new_tokens": CBG.GEN_MAX_NEW_TOKENS,
           "models": MODELS, "results": {}}

    for MODEL in MODELS:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                     low_cpu_mem_usage=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        short = CBG.make_decoder(model, tok)
        long_ = CBG.make_decoder(model, tok, max_new_tokens=64)
        print(f"\n=== {MODEL} ===", flush=True)
        print(f"  {'family':22s} {'acc@8':>7s} {'acc@64':>7s} {'trunc_loss':>11s} {'preamble':>9s}")
        per_model = {}
        for fam, items in families.items():
            s8 = s64 = pre = 0
            for q, gold, variants in items:
                triple = (q, gold, list(variants))
                a, _ = CBG.score_item(triple, short(CBG.gen_prompt(q)))
                b, _ = CBG.score_item(triple, long_(CBG.gen_prompt(q)))
                s8 += a; s64 += b
                txt = short(CBG.gen_prompt(q))[0].strip().lower()
                # preamble := the 8-token decode begins by restating rather than answering
                if txt[:4] in ("the ", "a ", "an ") or txt.startswith("answer"):
                    pre += 1
            n = len(items)
            row = {"acc_8": round(s8 / n, 4), "acc_64": round(s64 / n, 4),
                   "truncation_loss": round((s64 - s8) / n, 4),
                   "preamble_rate": round(pre / n, 4), "n": n}
            per_model[fam] = row
            print(f"  {fam:22s} {row['acc_8']:7.3f} {row['acc_64']:7.3f} "
                  f"{row['truncation_loss']:11.3f} {row['preamble_rate']:9.3f}", flush=True)
        out["results"][MODEL] = per_model
        del model
        torch.cuda.empty_cache()

    strong, weak = out["results"][MODELS[0]], out["results"][MODELS[1]]
    print("\n" + "=" * 78)
    print("DOES THE TRUNCATION GAP GROW AS THE MODEL WEAKENS?")
    print("=" * 78)
    print(f"  {'family':22s} {'loss@1.5B':>10s} {'loss@0.5B':>10s} {'delta':>8s}")
    deltas = {}
    for fam in strong:
        d = weak[fam]["truncation_loss"] - strong[fam]["truncation_loss"]
        deltas[fam] = round(d, 4)
        flag = "  <-- GROWS" if d >= 0.10 else ""
        print(f"  {fam:22s} {strong[fam]['truncation_loss']:10.3f} "
              f"{weak[fam]['truncation_loss']:10.3f} {d:8.3f}{flag}")
    worst = max(deltas.values()) if deltas else 0.0
    out["truncation_gap_delta"] = deltas
    out["max_delta"] = worst
    out["verdict"] = ("CONFOUND_MATERIAL: the truncation gap grows by >= 0.10 on at least one family "
                      "as capability falls, so part of any measured dose-response is an "
                      "instruction-following curve, not a knowledge curve"
                      if worst >= 0.10 else
                      "CONFOUND_BOUNDED: the truncation gap does not grow materially as capability "
                      "falls, so verbosity-driven truncation is not a plausible driver of a measured "
                      "dose-response at this budget")
    print("\n" + out["verdict"])
    json.dump(out, open(HERE / "_gen_verbosity_confound.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote _gen_verbosity_confound.json")


if __name__ == "__main__":
    main()

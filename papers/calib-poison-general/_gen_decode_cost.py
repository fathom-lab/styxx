"""How expensive is the generation battery per audit, and what does enlarging it cost the run?

Base-only, no training, no verdict. Decides a real design tradeoff with a measurement instead of a
guess: every audit scores the ENTIRE pool via measure_all_gen, but only the SELECTED sub-tasks feed
the aggregate. Sub-tasks that cannot clear the clean floor are therefore pure decode cost.

The scored run is 13 checkpoints x 3 arms x 5 seeds = 195 audits, and each audit ALSO runs the
format-invariance check at a pre-committed subsample (which scores the battery twice more).

Emits `_gen_decode_cost.json`.
"""
import importlib.util, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cbg", HERE / "capability_battery_gen.py")
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)

N_CKPT, N_ARMS, N_SEEDS = 13, 3, 5


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    decode = CBG.make_decoder(model, tok)

    n_items = sum(len(v) for v in CBG.GEN_DISJOINT_POOL.values()) + \
              sum(len(v) for v in CBG.GEN_BANK_ADJACENT.values())

    decode(CBG.gen_prompt("What is the plural of the word 'cat'?"))      # warm up

    t0 = time.time()
    CBG.measure_all_gen(decode_fn=decode)
    dt = time.time() - t0
    per_item = dt / n_items

    print(f"  current pool          : {n_items} items")
    print(f"  measure_all_gen       : {dt:.1f}s  ({per_item*1000:.0f} ms/item)")
    print()
    audits = N_CKPT * N_ARMS * N_SEEDS
    print(f"  scored run = {N_CKPT} ckpt x {N_ARMS} arms x {N_SEEDS} seeds = {audits} audits")
    print()
    rows = []
    for label, n in (("current pool (as shipped)", n_items),
                     ("6 new x 32, ALL old kept", 192 + n_items),
                     ("6 new x 32 + ORTH_FIRST + ANTONYM fixtures", 192 + 32),
                     ("6 new x 32 + ORTH_FIRST fixture only", 192 + 16),
                     ("6 new x 32 only", 192)):
        secs = n * per_item * audits
        rows.append({"configuration": label, "n_items": n,
                     "battery_hours_total": round(secs / 3600, 2)})
        print(f"  {label:44s} {n:4d} items -> {secs/3600:5.2f} GPU-hours of battery decode")
    print()
    print("  NOTE: this is battery decode ONLY. It excludes training, the T/F + MC + private-13 +")
    print("  knowledge readouts at every audit, and the format-invariance check (which scores the")
    print("  battery twice more at its subsample). Treat these as a LOWER BOUND on the added cost.")

    json.dump({"model": MODEL, "n_items_current": n_items,
               "measure_all_gen_seconds": round(dt, 2),
               "seconds_per_item": round(per_item, 4),
               "audits_in_scored_run": audits,
               "note": "battery decode only; excludes training and the other readouts",
               "configurations": rows},
              open(HERE / "_gen_decode_cost.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote _gen_decode_cost.json")


if __name__ == "__main__":
    main()

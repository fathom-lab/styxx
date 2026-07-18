"""Is the generation battery measuring KNOWLEDGE or TERSENESS? Base-only, treatment-blind.

GEN_MAX_NEW_TOKENS = 8. If the model preambles ("The comparative form of 'tall' is ...") the gold is
truncated away and the item scores 0 even though the model knows the answer. That would make the
battery partly an instruction-following / verbosity instrument -- and it would also put the 07-18
"character-level orthography" diagnosis in doubt, since those items may have failed the same way.

This dumps the RAW decode at the frozen 8-token budget alongside a long-budget decode (64 tokens) for
the same item, so the two failure modes separate cleanly:

  short WRONG + long WRONG   -> genuine capability failure
  short WRONG + long RIGHT   -> TRUNCATION / verbosity artifact, not capability

Emits `_gen_truncation_probe.json`.
"""
import importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cbg", HERE / "capability_battery_gen.py")
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)

# a sample spanning: pilot failures, pilot successes (control), and the 07-18 calibration failures
PROBE = {
    "COMPARATIVE": [("What is the comparative form of the adjective 'tall'?", "taller"),
                    ("What is the comparative form of the adjective 'quick'?", "quicker")],
    "OPPOSITE":    [("What is the opposite of the word 'silent'?", "loud"),
                    ("What is the opposite of the word 'narrow'?", "wide")],
    "CATEGORY":    [("A cobra is what kind of animal? Answer with one word.", "snake"),
                    ("A ruby is what kind of object? Answer with one word.", "gem")],
    "SUPERLATIVE": [("What is the superlative form of the adjective 'tall'?", "tallest")],
    # controls that PASSED the pilot -- these must stay right at both budgets
    "PLURAL_ctl":  [("What is the plural of the word 'basket'?", "baskets")],
    "CAPITAL_ctl": [("What is the capital city of France?", "paris")],
}

# the 07-18 calibration failures, taken VERBATIM from the shipped battery
for fam in ("ALPHA_GEN", "ORTH_LAST_GEN", "CONTAINS_GEN", "ANTONYM_GEN", "ORTH_FIRST_GEN"):
    PROBE[fam] = [(q, g) for q, g, _ in CBG.GEN_DISJOINT_POOL[fam][:3]]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    short = CBG.make_decoder(model, tok)                              # frozen 8-token budget
    long_ = CBG.make_decoder(model, tok, max_new_tokens=64)           # diagnostic budget

    out = {}
    n_trunc = n_real = 0
    for fam, items in PROBE.items():
        rows = []
        for q, gold in items:
            triple = (q, gold, [])
            s_short, _ = CBG.score_item(triple, short(CBG.gen_prompt(q)))
            s_long, _ = CBG.score_item(triple, long_(CBG.gen_prompt(q)))
            txt_s = short(CBG.gen_prompt(q))[0]
            txt_l = long_(CBG.gen_prompt(q))[0]
            verdict = ("ok" if s_short else
                       "TRUNCATION" if s_long else "capability")
            if not s_short:
                n_trunc += int(bool(s_long)); n_real += int(not s_long)
            rows.append({"q": q, "gold": gold, "short_score": int(s_short),
                         "long_score": int(s_long), "verdict": verdict,
                         "decode_8tok": txt_s, "decode_64tok": txt_l})
            print(f"  [{fam:15s}] short={int(s_short)} long={int(s_long)} {verdict:11s} "
                  f"gold={gold!r}\n      8tok : {txt_s!r}\n      64tok: {txt_l[:110]!r}", flush=True)
        out[fam] = rows

    print(f"\nfailures explained by TRUNCATION: {n_trunc}   genuine capability: {n_real}")
    json.dump({"model": MODEL, "gen_max_new_tokens": CBG.GEN_MAX_NEW_TOKENS,
               "n_truncation": n_trunc, "n_capability": n_real, "probe": out},
              open(HERE / "_gen_truncation_probe.json", "w", encoding="utf-8"), indent=1)
    print("wrote _gen_truncation_probe.json")


if __name__ == "__main__":
    main()

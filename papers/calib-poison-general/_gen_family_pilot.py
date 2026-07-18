"""PILOT: which capability families can Qwen2.5-1.5B-Instruct actually do in free generation?

Base-only, treatment-blind, no training, no verdict. Same class of measurement as --calibrate.

WHY: the 07-18 calibration found only 2 of 7 disjoint sub-tasks clear the 0.90 clean floor, and every
failure was CHARACTER-LEVEL orthography (a 1.5B tokenizer hides characters inside subword tokens).
Before authoring full 16-item banks, measure which candidate families are viable at all -- the
session's own lesson: measure the cheapest thing that can invalidate the plan FIRST.

Uses the battery's OWN make_decoder + score_item so pilot numbers transfer exactly to the real thing.
Six items per family is enough to separate "clearly viable" from "clearly not"; it is NOT enough to
certify a family, which is what the full bank + --calibrate is for.

Emits `_gen_family_pilot.json`.
"""
import importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cbg", HERE / "capability_battery_gen.py")
CBG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CBG)

# Candidate families. Every gold >= 3 chars. No character-level orthography anywhere.
# Non-list items: the gold never appears in the question text.
PILOT = {
    # --- controls: known-good from the 07-18 calibration (PLURAL 1.0000, SEQ 1.0000) ---
    "PLURAL_ctl":      [("What is the plural of the word 'basket'?", "baskets"),
                        ("What is the plural of the word 'pencil'?", "pencils"),
                        ("What is the plural of the word 'planet'?", "planets"),
                        ("What is the plural of the word 'ticket'?", "tickets"),
                        ("What is the plural of the word 'magnet'?", "magnets"),
                        ("What is the plural of the word 'bottle'?", "bottles")],
    # --- morphological transformation (same family as the passing PLURAL) ---
    "PAST_TENSE":      [("What is the past tense of the verb 'walk'?", "walked"),
                        ("What is the past tense of the verb 'jump'?", "jumped"),
                        ("What is the past tense of the verb 'paint'?", "painted"),
                        ("What is the past tense of the verb 'open'?", "opened"),
                        ("What is the past tense of the verb 'watch'?", "watched"),
                        ("What is the past tense of the verb 'climb'?", "climbed")],
    "COMPARATIVE":     [("What is the comparative form of the adjective 'tall'?", "taller"),
                        ("What is the comparative form of the adjective 'quick'?", "quicker"),
                        ("What is the comparative form of the adjective 'dark'?", "darker"),
                        ("What is the comparative form of the adjective 'warm'?", "warmer"),
                        ("What is the comparative form of the adjective 'bright'?", "brighter"),
                        ("What is the comparative form of the adjective 'smooth'?", "smoother")],
    "SUPERLATIVE":     [("What is the superlative form of the adjective 'tall'?", "tallest"),
                        ("What is the superlative form of the adjective 'quick'?", "quickest"),
                        ("What is the superlative form of the adjective 'dark'?", "darkest"),
                        ("What is the superlative form of the adjective 'warm'?", "warmest"),
                        ("What is the superlative form of the adjective 'bright'?", "brightest"),
                        ("What is the superlative form of the adjective 'smooth'?", "smoothest")],
    # --- ordered-sequence knowledge (same family as the passing SEQ months) ---
    "WEEKDAY_SEQ":     [("Which day of the week comes immediately after Monday?", "tuesday"),
                        ("Which day of the week comes immediately after Wednesday?", "thursday"),
                        ("Which day of the week comes immediately after Friday?", "saturday"),
                        ("Which day of the week comes immediately after Saturday?", "sunday"),
                        ("Which day of the week comes immediately after Tuesday?", "wednesday"),
                        ("Which day of the week comes immediately after Thursday?", "friday")],
    "ORDINAL_SEQ":     [("Which ordinal number comes immediately after third?", "fourth"),
                        ("Which ordinal number comes immediately after sixth?", "seventh"),
                        ("Which ordinal number comes immediately after eighth?", "ninth"),
                        ("Which ordinal number comes immediately after fourth?", "fifth"),
                        ("Which ordinal number comes immediately after ninth?", "tenth"),
                        ("Which ordinal number comes immediately after sixteenth?", "seventeenth")],
    # --- factual lookup (disjoint from the honesty construct) ---
    "CAPITAL":         [("What is the capital city of France?", "paris"),
                        ("What is the capital city of Japan?", "tokyo"),
                        ("What is the capital city of Egypt?", "cairo"),
                        ("What is the capital city of Canada?", "ottawa"),
                        ("What is the capital city of Norway?", "oslo"),
                        ("What is the capital city of Portugal?", "lisbon")],
    "ELEMENT_SYMBOL":  [("Which chemical element has the symbol Fe?", "iron"),
                        ("Which chemical element has the symbol Au?", "gold"),
                        ("Which chemical element has the symbol Na?", "sodium"),
                        ("Which chemical element has the symbol Ca?", "calcium"),
                        ("Which chemical element has the symbol Hg?", "mercury"),
                        ("Which chemical element has the symbol Zn?", "zinc")],
    # --- arithmetic with >= 3-digit results (category-adjacent; MUL scored 1.0000) ---
    "ARITH_ADD":       [("What is 137 plus 245?", "382"),
                        ("What is 268 plus 154?", "422"),
                        ("What is 349 plus 227?", "576"),
                        ("What is 415 plus 168?", "583"),
                        ("What is 523 plus 289?", "812"),
                        ("What is 176 plus 348?", "524")],
    # --- semantic opposites (ANTONYM scored 0.8125 -- retry with higher-frequency pairs) ---
    "OPPOSITE_EASY":   [("What is the opposite of the word 'hot'?", "cold"),
                        ("What is the opposite of the word 'heavy'?", "light"),
                        ("What is the opposite of the word 'empty'?", "full"),
                        ("What is the opposite of the word 'ancient'?", "modern"),
                        ("What is the opposite of the word 'silent'?", "loud"),
                        ("What is the opposite of the word 'narrow'?", "wide")],
    # --- category membership without a candidate list (so the echo guard is not engaged) ---
    "CATEGORY":        [("A robin is what kind of animal? Answer with one word.", "bird"),
                        ("A salmon is what kind of animal? Answer with one word.", "fish"),
                        ("A cobra is what kind of animal? Answer with one word.", "snake"),
                        ("An oak is what kind of plant? Answer with one word.", "tree"),
                        ("A ruby is what kind of object? Answer with one word.", "gem"),
                        ("A trumpet is what kind of object? Answer with one word.", "instrument")],
}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    decode = CBG.make_decoder(model, tok)

    out, rows = {}, []
    for fam, items in PILOT.items():
        correct = 0
        detail = []
        for q, gold in items:
            triple = (q, gold, [])
            s, flags = CBG.score_item(triple, decode(CBG.gen_prompt(q)))
            correct += s
            detail.append({"q": q, "gold": gold, "score": int(s), **{k: int(v) for k, v in flags.items()}})
        acc = correct / len(items)
        out[fam] = {"acc": round(acc, 4), "n": len(items), "items": detail}
        rows.append((fam, acc, len(items)))
        print(f"  {fam:16s} {acc:.4f}  ({correct}/{len(items)})", flush=True)

    print()
    viable = [f for f, a, _ in rows if a >= 0.90]
    marginal = [f for f, a, _ in rows if 0.75 <= a < 0.90]
    dead = [f for f, a, _ in rows if a < 0.75]
    print("VIABLE   (>= 0.90):", viable)
    print("MARGINAL (0.75-0.90):", marginal)
    print("DEAD     (< 0.75):", dead)
    json.dump({"model": MODEL, "note": "base-only, treatment-blind pilot; 6 items/family",
               "families": out, "viable": viable, "marginal": marginal, "dead": dead},
              open(HERE / "_gen_family_pilot.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote _gen_family_pilot.json")


if __name__ == "__main__":
    main()

"""The GENERATION-SCORED capability battery -- the gating capability instrument for B2-coupling
attempt 4. DRAFT for the pre-freeze adversarial panel; NOT frozen.

Why this file exists. Attempt 2's True/False battery was VOIDed for zero dose specificity (it read
capability through the very T/F margin the honesty LoRA trains -- fired on 0.5833 of constant-dose
control checkpoints). Attempt 3's multiple-choice battery was killed pre-freeze for the opposite
defect: argmax over four letter groups is invariant to any perturbation that shrinks margins without
flipping a winner, the sub-tasks were ceiling-trivial by selection, and nothing ever showed the
battery COULD fire under genuine capability destruction. This battery reads capability through a
third channel -- SHORT GREEDY GENERATION (about 8 new tokens) scored by normalized containment of a
canonical answer -- which is neither the trained T/F margin nor a margin-invariant argmax.

Panel-driven constraints carried in this file (each traceable to a named kill):

  (a) NO single-character golds anywhere. A 1-char substring hits an 8-token generation by chance,
      manufacturing a false ceiling and therefore false INSENSITIVITY (attempt 3's F1 mechanism on a
      new channel). Every gold and every accepted variant is >= 3 characters; --selftest asserts it.
  (b) DEGENERATE-REPETITION GUARD: if the decode is a single token repeated, or more than
      REP_GUARD_FRAC of one token, the item scores 0 regardless of containment. Fires are COUNTED
      and reported per measurement so a checkpoint whose decodes have collapsed is visible as such,
      not silently scored.
  (c) ECHO GUARD (this file's own named confound): for items that embed a candidate list in the
      question, the gold IS in the prompt, so a decode that parrots the list would contain the gold
      by construction. If two or more distinct candidates appear in the decode, the item scores 0
      and the fire is counted. Non-list items are built so the gold never appears in the question
      (asserted by --selftest).
  (d) NO VERDICT THRESHOLDS LIVE HERE. Selection floors, guard floors, fire thresholds, slope bars
      -- all of that is prereg/harness property (threshold provenance is a prereg matter). The only
      constants in this file define the instrument's mechanism, not any gating bar. Functions that
      need a bar take it as a required argument.

Scoring: greedy decode of GEN_MAX_NEW_TOKENS tokens under a fixed instruction wrapper; normalized
WORD-BOUNDARY containment (lowercase, punctuation stripped, contiguous token-sequence match -- so
"1200" does NOT match a gold of "200", and "cat" does NOT match a gold of "cats").

Format-invariance scaffolding: `format_invariance_check` scores the battery under a second,
verbosity-perturbed wrapper (same question content, longer instruction). A format-only perturbation
changes zero capability; if it moves the aggregate, the battery is measuring fluency/format
(attempt 2's disease on a new channel). The pass bar lives in the harness/prereg, not here.
CPU-testable: every measurement function accepts a `decode_fn` hook, and --selftest drives the full
pipeline (scoring, both guards, format check) through fake decoders with no model.

`--selftest` recomputes EVERY gold from ground-truth predicates (string ops, alphabetical order,
antonym map, plural rule, calendar successor maps, arithmetic) -- non-circular, mirroring the MC
battery's selftest style -- and asserts the anti-false-ceiling invariants above.

Structure mirrors capability_battery_mc.py: banks, select_disjoint, aggregate, paired_delta,
battery_guard, selftest.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------- instrument-mechanism constants (NOT verdict bars)
GEN_MAX_NEW_TOKENS = 8     # short greedy decode; part of the instrument's definition
REP_GUARD_FRAC = 0.6       # decode with > this fraction one token is degenerate -> item scores 0

# ---------------------------------------------------- frozen items: (question, canonical_answer, variants)
# Every gold and every accepted variant is >= 3 characters (panel constraint (a); asserted by selftest).
# List-format items ("...: w1, w2, w3, w4?") reuse the MC battery's verified one-correct option sets.
# Non-list items never contain their gold in the question text (asserted by selftest).

GEN_DISJOINT_POOL = {
    # regular noun pluralization; rule-checked (+s, or +es after a sibilant grapheme) (32 items)
    "PLURAL_GEN": [
        ("What is the plural of the word 'cat'?", "cats", []),
        ("What is the plural of the word 'river'?", "rivers", []),
        ("What is the plural of the word 'table'?", "tables", []),
        ("What is the plural of the word 'garden'?", "gardens", []),
        ("What is the plural of the word 'rocket'?", "rockets", []),
        ("What is the plural of the word 'window'?", "windows", []),
        ("What is the plural of the word 'candle'?", "candles", []),
        ("What is the plural of the word 'mirror'?", "mirrors", []),
        ("What is the plural of the word 'helmet'?", "helmets", []),
        ("What is the plural of the word 'turtle'?", "turtles", []),
        ("What is the plural of the word 'spider'?", "spiders", []),
        ("What is the plural of the word 'basket'?", "baskets", []),
        ("What is the plural of the word 'tunnel'?", "tunnels", []),
        ("What is the plural of the word 'magnet'?", "magnets", []),
        ("What is the plural of the word 'barrel'?", "barrels", []),
        ("What is the plural of the word 'jacket'?", "jackets", []),
        ("What is the plural of the word 'pencil'?", "pencils", []),
        ("What is the plural of the word 'bottle'?", "bottles", []),
        ("What is the plural of the word 'lantern'?", "lanterns", []),
        ("What is the plural of the word 'hammer'?", "hammers", []),
        ("What is the plural of the word 'anchor'?", "anchors", []),
        ("What is the plural of the word 'ladder'?", "ladders", []),
        ("What is the plural of the word 'island'?", "islands", []),
        ("What is the plural of the word 'cabin'?", "cabins", []),
        ("What is the plural of the word 'box'?", "boxes", []),
        ("What is the plural of the word 'bench'?", "benches", []),
        ("What is the plural of the word 'brush'?", "brushes", []),
        ("What is the plural of the word 'dish'?", "dishes", []),
        ("What is the plural of the word 'glass'?", "glasses", []),
        ("What is the plural of the word 'beach'?", "beaches", []),
        ("What is the plural of the word 'branch'?", "branches", []),
        ("What is the plural of the word 'bus'?", "buses", ["busses"]),
    ],
    # regular verb past tense; rule-checked over four ordered spelling classes (32 items)
    "PAST_TENSE_GEN": [
        ("What is the past tense of the verb 'walk'?", "walked", []),
        ("What is the past tense of the verb 'jump'?", "jumped", []),
        ("What is the past tense of the verb 'paint'?", "painted", []),
        ("What is the past tense of the verb 'watch'?", "watched", []),
        ("What is the past tense of the verb 'climb'?", "climbed", []),
        ("What is the past tense of the verb 'listen'?", "listened", []),
        ("What is the past tense of the verb 'answer'?", "answered", []),
        ("What is the past tense of the verb 'finish'?", "finished", []),
        ("What is the past tense of the verb 'repeat'?", "repeated", []),
        ("What is the past tense of the verb 'learn'?", "learned", ["learnt"]),
        ("What is the past tense of the verb 'burn'?", "burned", ["burnt"]),
        ("What is the past tense of the verb 'dream'?", "dreamed", ["dreamt"]),
        ("What is the past tense of the verb 'enjoy'?", "enjoyed", []),
        ("What is the past tense of the verb 'play'?", "played", []),
        ("What is the past tense of the verb 'close'?", "closed", []),
        ("What is the past tense of the verb 'smile'?", "smiled", []),
        ("What is the past tense of the verb 'arrive'?", "arrived", []),
        ("What is the past tense of the verb 'decide'?", "decided", []),
        ("What is the past tense of the verb 'invite'?", "invited", []),
        ("What is the past tense of the verb 'dance'?", "danced", []),
        ("What is the past tense of the verb 'describe'?", "described", []),
        ("What is the past tense of the verb 'carry'?", "carried", []),
        ("What is the past tense of the verb 'study'?", "studied", []),
        ("What is the past tense of the verb 'copy'?", "copied", []),
        ("What is the past tense of the verb 'cry'?", "cried", []),
        ("What is the past tense of the verb 'reply'?", "replied", []),
        ("What is the past tense of the verb 'stop'?", "stopped", []),
        ("What is the past tense of the verb 'plan'?", "planned", []),
        ("What is the past tense of the verb 'grab'?", "grabbed", []),
        ("What is the past tense of the verb 'chat'?", "chatted", []),
        ("What is the past tense of the verb 'trim'?", "trimmed", []),
        ("What is the past tense of the verb 'stir'?", "stirred", []),
    ],
    # calendar adjacency, DISTINCT facts only (padding to 32 forced non-independent inverses) (17 items)
    "SEQ_GEN": [
        ("Which month of the year comes immediately after January?", "february", ["feb"]),
        ("Which month of the year comes immediately after February?", "march", []),
        ("Which month of the year comes immediately after March?", "april", ["apr"]),
        ("Which month of the year comes immediately after May?", "june", ["jun"]),
        ("Which month of the year comes immediately after June?", "july", ["jul"]),
        ("Which month of the year comes immediately after July?", "august", ["aug"]),
        ("Which month of the year comes immediately after September?", "october", ["oct"]),
        ("Which month of the year comes immediately after November?", "december", ["dec"]),
        ("Which month of the year comes immediately after December?", "january", ["jan"]),
        ("Which month of the year comes immediately before October?", "september", ["sep", "sept"]),
        ("Which day of the week comes immediately after Monday?", "tuesday", ["tue", "tues"]),
        ("Which day of the week comes immediately after Tuesday?", "wednesday", []),
        ("Which day of the week comes immediately after Wednesday?", "thursday", ["thu", "thur", "thurs"]),
        ("Which day of the week comes immediately after Thursday?", "friday", ["fri"]),
        ("Which day of the week comes immediately after Friday?", "saturday", []),
        ("Which day of the week comes immediately after Saturday?", "sunday", []),
        ("Which day of the week comes immediately after Sunday?", "monday", ["mon"]),
    ],
    # capital cities; unambiguous, stable, single-capital states only (32 items)
    "CAPITAL_GEN": [
        ("What is the capital city of France?", "paris", []),
        ("What is the capital city of Portugal?", "lisbon", ["lisboa"]),
        ("What is the capital city of Norway?", "oslo", []),
        ("What is the capital city of Denmark?", "copenhagen", ["kobenhavn", "koebenhavn"]),
        ("What is the capital city of Finland?", "helsinki", ["helsingfors"]),
        ("What is the capital city of Austria?", "vienna", ["wien"]),
        ("What is the capital city of Poland?", "warsaw", ["warszawa"]),
        ("What is the capital city of the Czech Republic?", "prague", ["praha"]),
        ("What is the capital city of Hungary?", "budapest", []),
        ("What is the capital city of Romania?", "bucharest", ["bucuresti", "bukarest"]),
        ("What is the capital city of Ukraine?", "kyiv", ["kiev"]),
        ("What is the capital city of Greece?", "athens", ["athina", "athinai"]),
        ("What is the capital city of Japan?", "tokyo", []),
        ("What is the capital city of India?", "new delhi", ["delhi"]),
        ("What is the capital city of Vietnam?", "hanoi", ["ha noi"]),
        ("What is the capital city of Thailand?", "bangkok", ["krung thep"]),
        ("What is the capital city of Pakistan?", "islamabad", []),
        ("What is the capital city of Nepal?", "kathmandu", ["katmandu"]),
        ("What is the capital city of Mongolia?", "ulaanbaatar", ["ulan bator", "ulaan baatar"]),
        ("What is the capital city of Iran?", "tehran", ["teheran"]),
        ("What is the capital city of Turkey?", "ankara", []),
        ("What is the capital city of Egypt?", "cairo", ["al qahirah"]),
        ("What is the capital city of Kenya?", "nairobi", []),
        ("What is the capital city of Morocco?", "rabat", []),
        ("What is the capital city of Ghana?", "accra", []),
        ("What is the capital city of Ethiopia?", "addis ababa", []),
        ("What is the capital city of Canada?", "ottawa", []),
        ("What is the capital city of Peru?", "lima", []),
        ("What is the capital city of Argentina?", "buenos aires", []),
        ("What is the capital city of Cuba?", "havana", ["la habana", "habana"]),
        ("What is the capital city of Australia?", "canberra", []),
        ("What is the capital city of New Zealand?", "wellington", []),
    ],
    # element name from symbol; alternate spellings accepted as variants (32 items)
    "ELEMENT_GEN": [
        ("Which chemical element has the symbol Fe?", "iron", ["ferrum"]),
        ("Which chemical element has the symbol Au?", "gold", ["aurum"]),
        ("Which chemical element has the symbol Ag?", "silver", ["argentum"]),
        ("Which chemical element has the symbol Cu?", "copper", ["cuprum"]),
        ("Which chemical element has the symbol Hg?", "mercury", ["quicksilver", "hydrargyrum"]),
        ("Which chemical element has the symbol Zn?", "zinc", []),
        ("Which chemical element has the symbol Ni?", "nickel", []),
        ("Which chemical element has the symbol Na?", "sodium", ["natrium"]),
        ("Which chemical element has the symbol K?", "potassium", ["kalium"]),
        ("Which chemical element has the symbol Ca?", "calcium", []),
        ("Which chemical element has the symbol Mg?", "magnesium", []),
        ("Which chemical element has the symbol Al?", "aluminum", ["aluminium"]),
        ("Which chemical element has the symbol Si?", "silicon", []),
        ("Which chemical element has the symbol He?", "helium", []),
        ("Which chemical element has the symbol Ne?", "neon", []),
        ("Which chemical element has the symbol Ar?", "argon", []),
        ("Which chemical element has the symbol Kr?", "krypton", []),
        ("Which chemical element has the symbol Xe?", "xenon", []),
        ("Which chemical element has the symbol Li?", "lithium", []),
        ("Which chemical element has the symbol Cl?", "chlorine", []),
        ("Which chemical element has the symbol Br?", "bromine", []),
        ("Which chemical element has the symbol Ba?", "barium", []),
        ("Which chemical element has the symbol S?", "sulfur", ["sulphur"]),
        ("Which chemical element has the symbol P?", "phosphorus", ["phosphorous"]),
        ("Which chemical element has the symbol U?", "uranium", []),
        ("Which chemical element has the symbol Pt?", "platinum", []),
        ("Which chemical element has the symbol Ti?", "titanium", []),
        ("Which chemical element has the symbol Mn?", "manganese", []),
        ("Which chemical element has the symbol Co?", "cobalt", []),
        ("Which chemical element has the symbol W?", "tungsten", ["wolfram"]),
        ("Which chemical element has the symbol Cs?", "cesium", ["caesium"]),
        ("Which chemical element has the symbol Ra?", "radium", []),
    ],
    # RETAINED AS SELFTEST FIXTURES ONLY -- both measured under DISJOINT_FLOOR_CLEAN on
    # the clean base, so select_disjoint will not select them. ORTH_FIRST_GEN is the only
    # list-format family and is what exercises the echo guard; ANTONYM_GEN exercises the
    # variant path. ANTONYM cannot be repaired into a gating family: _STOPLIST bans the
    # model's natural answers ('cold' for warm, 'kind' for cruel), a structural conflict
    # between the false-ceiling guard and the antonym task.
    "ORTH_FIRST_GEN": [
        ("Which of these words starts with the letter T: table, apple, mango, river?", "table", []),
        ("Which of these words starts with the letter S: sunset, orange, pencil, window?", "sunset", []),
        ("Which of these words starts with the letter M: mountain, cherry, garden, flower?", "mountain", []),
        ("Which of these words starts with the letter D: diamond, carpet, yellow, branch?", "diamond", []),
        ("Which of these words starts with the letter P: orange, pencil, window, magnet?", "pencil", []),
        ("Which of these words starts with the letter C: button, candle, planet, silver?", "candle", []),
        ("Which of these words starts with the letter G: hammer, garden, turtle, rocket?", "garden", []),
        ("Which of these words starts with the letter F: marble, forest, desert, spider?", "forest", []),
        ("Which of these words starts with the letter R: magnet, copper, river, tunnel?", "river", []),
        ("Which of these words starts with the letter W: basket, violet, window, summer?", "window", []),
        ("Which of these words starts with the letter H: purple, rocket, hammer, valley?", "hammer", []),
        ("Which of these words starts with the letter L: monkey, tiger, lantern, donkey?", "lantern", []),
        ("Which of these words starts with the letter V: melon, walnut, barrel, violet?", "violet", []),
        ("Which of these words starts with the letter N: marble, carrot, helmet, needle?", "needle", []),
        ("Which of these words starts with the letter K: lemon, mirror, tower, kitten?", "kitten", []),
        ("Which of these words starts with the letter J: candle, magnet, ribbon, jacket?", "jacket", []),
    ],
    "ANTONYM_GEN": [
        ("What is the opposite of the word 'warm'?", "cool", []),
        ("What is the opposite of the word 'above'?", "below", []),
        ("What is the opposite of the word 'big'?", "small", []),
        ("What is the opposite of the word 'fast'?", "slow", []),
        ("What is the opposite of the word 'cruel'?", "gentle", []),
        ("What is the opposite of the word 'open'?", "closed", ["shut"]),
        ("What is the opposite of the word 'light'?", "dark", ["heavy"]),
        ("What is the opposite of the word 'smooth'?", "rough", []),
        ("What is the opposite of the word 'wide'?", "narrow", []),
        ("What is the opposite of the word 'hard'?", "soft", ["easy"]),
        ("What is the opposite of the word 'expand'?", "shrink", []),
        ("What is the opposite of the word 'increase'?", "decrease", []),
        ("What is the opposite of the word 'strong'?", "weak", []),
        ("What is the opposite of the word 'clean'?", "dirty", []),
        ("What is the opposite of the word 'loud'?", "quiet", ["silent"]),
        ("What is the opposite of the word 'ancient'?", "modern", []),
    ],
}


GEN_BANK_ADJACENT = {
    # arithmetic is category-adjacent to the honesty bank (it has ADDITION); measured, NEVER gating.
    # products are all >= 3 digits so no gold falls under the 3-character floor.
    "MUL_GEN": [
        ("What is 25 times 8?", "200", []),
        ("What is 12 times 12?", "144", []),
        ("What is 15 times 11?", "165", []),
        ("What is 13 times 9?", "117", []),
        ("What is 14 times 8?", "112", []),
        ("What is 25 times 5?", "125", []),
        ("What is 19 times 11?", "209", []),
        ("What is 16 times 8?", "128", []),
    ],
}
ALL_GEN_SUBTASKS = list(GEN_DISJOINT_POOL.keys()) + list(GEN_BANK_ADJACENT.keys())


# ------------------------------------------------------------------------ prompts (plain + verbose)
def gen_prompt(question):
    """The standard instruction wrapper the battery is scored under."""
    return f"{question}\nAnswer with just the answer, nothing else."


def gen_prompt_verbose(question):
    """The FORMAT-ONLY perturbation: identical question content, longer instruction. Zero capability
    difference by construction -- if the aggregate moves under this wrapper, the battery is measuring
    fluency/format, not capability. The pass bar lives in the harness/prereg."""
    return ("Please read the following question carefully and think about what is being asked. "
            f"{question}\n"
            "When you respond, please answer with just the answer itself and nothing else at all.")


# ------------------------------------------------------------------- normalization + containment
def _norm_tokens(text):
    """Lowercase, strip punctuation to spaces, split -> word tokens."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def _contains(answer, decoded_tokens):
    """WORD-BOUNDARY containment: the answer's normalized token sequence appears contiguously in the
    decode's tokens. '1200' does not match answer '200'; 'cat' does not match answer 'cats'."""
    a = _norm_tokens(answer)
    n = len(a)
    if n == 0 or n > len(decoded_tokens):
        return False
    return any(decoded_tokens[i:i + n] == a for i in range(len(decoded_tokens) - n + 1))


def _list_candidates(question):
    """Candidate list for list-format items ('...: w1, w2, w3, w4?'); [] for free-form items."""
    m = re.search(r":\s*([^?]+)\?", question)
    if not m:
        return []
    return [c.strip() for c in m.group(1).split(",") if c.strip()]


# ----------------------------------------------------------------------------------- scoring
def is_degenerate(token_ids):
    """Panel constraint (b): single token repeated, or > REP_GUARD_FRAC one token."""
    if not token_ids or len(token_ids) < 2:
        return False
    counts = {}
    for t in token_ids:
        counts[t] = counts.get(t, 0) + 1
    top = max(counts.values())
    return len(counts) == 1 or (top / len(token_ids)) > REP_GUARD_FRAC


def score_item(item, decode_out):
    """Score one item from a decode. `decode_out` is (text, token_ids) as returned by a decoder.
    Returns (score, flags) where flags = {'repetition': 0/1, 'echo': 0/1}."""
    question, gold, variants = item
    text, token_ids = decode_out
    if is_degenerate(token_ids):
        return 0, {"repetition": 1, "echo": 0}
    toks = _norm_tokens(text)
    candidates = _list_candidates(question)
    if candidates:
        hits = sum(1 for c in candidates if _contains(c, toks))
        if hits >= 2:                       # panel constraint (c): the decode parrots the list
            return 0, {"repetition": 0, "echo": 1}
    matched = any(_contains(ans, toks) for ans in [gold] + list(variants))
    return int(matched), {"repetition": 0, "echo": 0}


def make_decoder(model, tok, max_new_tokens=GEN_MAX_NEW_TOKENS):
    """Bind a (model, tok) pair into a decode_fn(prompt_text) -> (text, token_ids). Greedy."""
    import torch
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def decode(prompt_text):
        msg = [{"role": "user", "content": prompt_text}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad_id)
        gen = out[0, ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True), [int(t) for t in gen.tolist()]
    return decode


def measure_gen_subtask(items, decode_fn, wrapper=gen_prompt):
    """Accuracy of one sub-task under `wrapper`, via `decode_fn`. Returns
    {'acc', 'acc_guard_excl', 'repetition_fires', 'echo_fires', 'n', 'n_scored', 'correct'}.

    `acc_guard_excl` is the accuracy with the guard-zeroed items (repetition OR echo) REMOVED from
    BOTH numerator and denominator -- guard-fired items already score 0, so `correct` is unchanged
    and only the denominator shrinks. This feeds the pre-committed subtractive check (panel FATAL
    F5): if the priced dose slope does not survive with guard-driven items excluded, the price was
    style, not capability."""
    correct = rep = echo = 0
    n = len(items)
    for item in items:
        s, flags = score_item(item, decode_fn(wrapper(item[0])))
        correct += s
        rep += flags["repetition"]
        echo += flags["echo"]
    n_scored = n - (rep + echo)
    acc_guard_excl = (correct / n_scored) if n_scored > 0 else 0.0
    return {"acc": correct / n, "acc_guard_excl": acc_guard_excl,
            "repetition_fires": rep, "echo_fires": echo,
            "n": n, "n_scored": n_scored, "correct": correct}


def measure_all_gen(model=None, tok=None, *, decode_fn=None, wrapper=gen_prompt):
    """Score every sub-task (disjoint pool + bank-adjacent). Returns
    {'scores': {name: acc}, 'scores_guard_excl': {name: acc}, 'repetition_guard_fires': int,
    'echo_guard_fires': int}. `scores_guard_excl` is the per-sub-task accuracy with guard-zeroed
    items excluded (F5, above). Pass a fake `decode_fn` for CPU testing; otherwise a decoder is
    built from (model, tok)."""
    if decode_fn is None:
        if model is None or tok is None:
            raise ValueError("supply either decode_fn or (model, tok)")
        decode_fn = make_decoder(model, tok)
    scores, scores_guard_excl, rep, echo = {}, {}, 0, 0
    for name, items in list(GEN_DISJOINT_POOL.items()) + list(GEN_BANK_ADJACENT.items()):
        r = measure_gen_subtask(items, decode_fn, wrapper)
        scores[name] = float(r["acc"])
        scores_guard_excl[name] = float(r["acc_guard_excl"])
        rep += r["repetition_fires"]
        echo += r["echo_fires"]
    return {"scores": scores, "scores_guard_excl": scores_guard_excl,
            "repetition_guard_fires": rep, "echo_guard_fires": echo}


# ---------------------------------------------------------------------- format-invariance check
def format_invariance_check(model=None, tok=None, *, selected=None, decode_fn=None):
    """Score the battery under the plain AND the verbosity-perturbed wrapper (same content, longer
    instruction). A format-only perturbation must not move the aggregate; the bar for 'not move'
    is a harness/prereg constant, NOT set here. Returns per-subtask and aggregate scores plus
    abs_delta. CPU-testable via decode_fn."""
    if selected is None:
        selected = sorted(GEN_DISJOINT_POOL.keys())
    plain = measure_all_gen(model, tok, decode_fn=decode_fn, wrapper=gen_prompt)
    verbose = measure_all_gen(model, tok, decode_fn=decode_fn, wrapper=gen_prompt_verbose)
    agg_p = aggregate(plain["scores"], selected)
    agg_v = aggregate(verbose["scores"], selected)
    return {"selected": list(selected),
            "aggregate_plain": round(agg_p, 4), "aggregate_verbose": round(agg_v, 4),
            "abs_delta": round(abs(agg_p - agg_v), 4),
            "per_subtask_plain": {k: round(v, 4) for k, v in plain["scores"].items()},
            "per_subtask_verbose": {k: round(v, 4) for k, v in verbose["scores"].items()},
            "repetition_guard_fires_plain": plain["repetition_guard_fires"],
            "repetition_guard_fires_verbose": verbose["repetition_guard_fires"]}


# ----------------------------------------------------- selection / aggregate / paired delta (bar-free)
def select_disjoint(clean_scores, *, floor, need):
    """Base-only, treatment-blind selection: keep every GEN_DISJOINT_POOL sub-task the CLEAN base
    clears at `floor`; ok iff at least `need` survive. floor/need are prereg constants supplied by
    the HARNESS (panel constraint (d): no gating bars live in this file)."""
    survivors = sorted([n for n in GEN_DISJOINT_POOL if clean_scores.get(n, 0.0) >= floor])
    return survivors, bool(len(survivors) >= need)


def aggregate(scores, selected):
    return float(sum(scores[n] for n in selected) / len(selected)) if selected else 0.0


def battery_guard(clean_scores, selected, *, agg_floor, subtask_floor):
    """Clean admissibility on the SELECTED sub-tasks. Floors are prereg constants from the harness."""
    if not selected:
        return False
    return bool(aggregate(clean_scores, selected) >= agg_floor and
                all(clean_scores[n] >= subtask_floor for n in selected))


def paired_delta(fixed_scores, acc_scores, selected):
    """The paired statistic of attempt 4: constant-dose control aggregate minus accumulate aggregate
    at a matched step. Each arm's aggregate is ROUNDED to 4 decimals FIRST, then subtracted (re-panel
    minor 9), so the delta is EXACTLY the difference of the two `gen_aggregate` values the `points`
    grounding surface exposes -- a verifier recomputing deltas from points[*].gen_aggregate reproduces
    these to the last decimal. NO break bar here -- attempt 4's verdict is a dose SLOPE over these
    deltas, and the slope bar is a prereg constant in the harness."""
    agg_fixed = round(aggregate(fixed_scores, selected), 4)
    agg_acc = round(aggregate(acc_scores, selected), 4)
    return float(round(agg_fixed - agg_acc, 4))


# ------------------------------------------------------------------------------ CPU-only self-check
_ANTONYM = {"warm": "cool", "above": "below", "big": "small", "fast": "slow", "cruel": "gentle",
            "open": "closed", "light": "dark", "smooth": "rough", "wide": "narrow", "hard": "soft",
            "expand": "shrink", "increase": "decrease", "strong": "weak", "clean": "dirty",
            "loud": "quiet", "ancient": "modern"}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Frozen high-frequency-function-word stoplist (panel MAJOR M1, extended by re-panel minor 6). No
# gold or accepted variant may be one of these bare words: a degraded model that emits them as fluff
# (hedging modals "may"/"kind"/"sort", common adjectives, "little"/"past") would score CORRECT by
# incidental containment, a false ceiling that biases the paired delta toward the favourable-for-
# erasure bounded null. --selftest asserts the whole bank clears it.
_STOPLIST = {"may", "down", "old", "short", "full", "dry", "late", "sad", "cold",
             "kind", "little", "past", "sort"}


# ------------------------------------------------- ground-truth tables for the predicates
# Each is the NON-CIRCULAR source of truth: _predicate recomputes every gold from the
# QUESTION against these, so a mistyped gold -- or a gold swapped between two items --
# fails --selftest rather than shipping.

# ---- PLURAL_GEN ----
# Regular English plural spelling RULE for PLURAL_GEN (a rule, never a lookup): a noun whose written
# form ends in a sibilant grapheme takes -es, every other noun takes -s. The bank is restricted to
# stems where this rule is exact -- no -o stems (potatoes vs pianos), no -f/-fe stems (roofs vs
# leaves), no consonant+y stems (y -> ies is a different rule), no <ch> stems pronounced /k/
# (stomach -> stomachs, monarch -> monarchs), and no irregulars. 8 of 32 items are -es stems, so the
# sibilant branch is exercised by the bank rather than being dead code.
_PLURAL_ES_ENDINGS = ("s", "x", "z", "ch", "sh")

# ---- PAST_TENSE_GEN ----
_VOWELS = "aeiou"


def _regular_past(verb):
    """Exact REGULAR English past-tense rule, recomputed from the verb STRING (never from a stored
    gold) -- the non-circularity guarantee for PAST_TENSE_GEN. Four ordered spelling classes:
      1. final 'e'                        -> +d      (close -> closed, dance -> danced)
      2. consonant + final 'y'            -> y+ied   (carry -> carried, cry -> cried)
      3. monosyllabic C-V-C, final consonant not w/x/y -> double+ed (stop -> stopped)
      4. otherwise                        -> +ed     (walk -> walked, play -> played)
    Class 3's monosyllable test is a vowel-GROUP count, so the doubling rule can never fire on a
    polysyllable (visit -> visited, answer -> answered, gather -> gathered) where doubling is
    stress-dependent and no pure string rule is exact; the bank therefore contains no polysyllabic
    doubling verb. Class 2 inspects the pre-'y' character, so vowel+y verbs fall through to class 4
    (enjoy -> enjoyed) rather than being mangled to 'enjoied'. The bank is ALL-REGULAR: no irregular
    verb is present, so this rule is exact for every one of its 32 items."""
    v = verb.lower()
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) >= 2 and v[-2] not in _VOWELS:
        return v[:-1] + "ied"
    groups = len(re.findall(r"[aeiou]+", v))
    if (groups == 1 and len(v) >= 3 and v[-1] not in _VOWELS and v[-1] not in "wxy"
            and v[-2] in _VOWELS and v[-3] not in _VOWELS):
        return v + v[-1] + "ed"
    return v + "ed"

# ---- SEQ_GEN ----
# Standard calendar abbreviations accepted as answer VARIANTS for SEQ_GEN, and accepted by
# _predicate so no shipped variant is unverified. An abbreviation is admitted ONLY if the bare
# token is not itself a common English word: 'mar' (to mar), 'wed' (to wed), 'sat' (sit) and
# 'sun' are deliberately ABSENT, since each would open an incidental-containment path -- the
# same false-ceiling mechanism _STOPLIST exists to block. Keys are canonical golds.
_CAL_ABBREV = {
    "january": ("jan",), "february": ("feb",), "april": ("apr",),
    "june": ("jun",), "july": ("jul",), "august": ("aug",),
    "september": ("sep", "sept"), "october": ("oct",),
    "november": ("nov",), "december": ("dec",),
    "monday": ("mon",), "tuesday": ("tue", "tues"),
    "thursday": ("thu", "thur", "thurs"), "friday": ("fri",),
}

# ---- CAPITAL_GEN ----
# CAPITAL_GEN ground truth: country (as parsed out of the question) -> canonical capital, lowercase.
# The predicate looks the country up HERE and never reads the stored gold, so a wrong entry in this
# map is caught by --selftest rather than silently frozen into the instrument. A country absent from
# the map raises KeyError -- a loud failure, never a silent pass.
# Excluded by construction: multi-capital / seat-split states (Bolivia, South Africa, the
# Netherlands, Switzerland, Sri Lanka, Benin, Ivory Coast, Tanzania, Chile), recent relocations
# (Myanmar, Kazakhstan, Indonesia, Nigeria's Lagos-era ambiguity aside), and contested cases
# (Israel, Palestine, Cyprus). Every capital is >= 3 characters and none is a substring token of its
# own country name, so no gold can be scored by echoing the prompt.
_CAPITALS = {
    "France": "paris", "Portugal": "lisbon", "Norway": "oslo", "Denmark": "copenhagen",
    "Finland": "helsinki", "Austria": "vienna", "Poland": "warsaw", "Czech Republic": "prague",
    "Hungary": "budapest", "Romania": "bucharest", "Ukraine": "kyiv", "Greece": "athens",
    "Japan": "tokyo", "India": "new delhi", "Vietnam": "hanoi", "Thailand": "bangkok",
    "Pakistan": "islamabad", "Nepal": "kathmandu", "Mongolia": "ulaanbaatar", "Iran": "tehran",
    "Turkey": "ankara", "Egypt": "cairo", "Kenya": "nairobi", "Morocco": "rabat",
    "Ghana": "accra", "Ethiopia": "addis ababa", "Canada": "ottawa", "Peru": "lima",
    "Argentina": "buenos aires", "Cuba": "havana", "Australia": "canberra",
    "New Zealand": "wellington",
}

# ---- ELEMENT_GEN ----
# Symbol -> every accepted element NAME for that symbol (canonical first). Ground truth for
# ELEMENT_GEN: the selftest recomputes each gold from the SYMBOL IN THE QUESTION through this map,
# never from the stored gold, so a mistyped gold is caught. Latin and alternate-spelling names are
# listed because a competent answerer may legitimately emit them (ferrum, aluminium, sulphur,
# caesium, wolfram); the predicate accepts any member, so accepted variants are ground-truth
# checked too, not merely asserted.
_ELEMENT_SYMBOL = {
    "Fe": ("iron", "ferrum"),
    "Au": ("gold", "aurum"),
    "Ag": ("silver", "argentum"),
    "Cu": ("copper", "cuprum"),
    "Hg": ("mercury", "quicksilver", "hydrargyrum"),
    "Zn": ("zinc",),
    "Ni": ("nickel",),
    "Na": ("sodium", "natrium"),
    "K": ("potassium", "kalium"),
    "Ca": ("calcium",),
    "Mg": ("magnesium",),
    "Al": ("aluminum", "aluminium"),
    "Si": ("silicon",),
    "He": ("helium",),
    "Ne": ("neon",),
    "Ar": ("argon",),
    "Kr": ("krypton",),
    "Xe": ("xenon",),
    "Li": ("lithium",),
    "Cl": ("chlorine",),
    "Br": ("bromine",),
    "Ba": ("barium",),
    "S": ("sulfur", "sulphur"),
    "P": ("phosphorus", "phosphorous"),
    "U": ("uranium",),
    "Pt": ("platinum",),
    "Ti": ("titanium",),
    "Mn": ("manganese",),
    "Co": ("cobalt",),
    "W": ("tungsten", "wolfram"),
    "Cs": ("cesium", "caesium"),
    "Ra": ("radium",),
}


def _predicate(subtask, question):
    """Ground-truth predicate for one item: f(answer_string) -> bool. Non-circular:
    recomputed from string ops / rules / frozen maps, never from the stored gold."""
    if subtask == "PLURAL_GEN":
        w = re.search(r"the word '(\w+)'\?", question)[1].lower()
        suffix = "es" if w.endswith(_PLURAL_ES_ENDINGS) else "s"
        return lambda o: o.lower() == w + suffix
    if subtask == "PAST_TENSE_GEN":
        v = re.search(r"the verb '(\w+)'\?", question)[1].lower()
        return lambda o: o.lower() == _regular_past(v)
    if subtask == "SEQ_GEN":
        # EXTENDED (this re-authoring): the bank now carries BOTH directions of the calendar
        # relation -- 'comes immediately after X' and 'comes immediately before X' -- because 12
        # months + 7 days give only 19 distinct successor facts and the family needs 32 distinct
        # items. The predicate therefore parses the DIRECTION out of the question and steps the
        # cyclic sequence by +1 or -1. Still fully non-circular: the cycle comes from _MONTHS /
        # _DAYS and the anchor word comes from the QUESTION, never from the stored gold, so a
        # mis-typed gold (or a gold swapped between two items) fails the selftest.
        # Standard abbreviations (_CAL_ABBREV) are accepted as well, so every shipped variant is
        # predicate-verified and not merely length-checked.
        mm = re.search(r"immediately (after|before) (\w+)\?", question)
        step = 1 if mm[1] == "after" else -1
        w = mm[2].lower()
        seq = _MONTHS if w in _MONTHS else _DAYS
        target = seq[(seq.index(w) + step) % len(seq)]
        return lambda o: o.lower() == target or o.lower() in _CAL_ABBREV.get(target, ())
    if subtask == "CAPITAL_GEN":
        c = re.search(r"capital city of ([^?]+)\?", question)[1].strip()
        if c.lower().startswith("the "):
            c = c[4:]
        return lambda o: _CAPITALS[c] == o.strip().lower()
    if subtask == "ELEMENT_GEN":
        sym = re.search(r"the symbol (\w+)\?", question)[1]
        names = _ELEMENT_SYMBOL[sym]
        return lambda o: o.strip().lower() in names
    if subtask == "ORTH_FIRST_GEN":
        L = re.search(r"starts with the letter (\w):", question)[1]
        return lambda o: o[0].upper() == L.upper()
    if subtask == "ANTONYM_GEN":
        w = re.search(r"the word '(\w+)'\?", question)[1]
        return lambda o: _ANTONYM.get(w) == o
    if subtask == "MUL_GEN":
        mm = re.search(r"What is (\d+) times (\d+)\?", question)
        prod = int(mm[1]) * int(mm[2])
        return lambda o: o.isdigit() and int(o) == prod
    raise ValueError(subtask)


def _selftest():
    checks, ok = [], True

    def add(name, cond):
        nonlocal ok
        ok = ok and bool(cond); checks.append({"check": name, "ok": bool(cond)})

    # ---- bank invariants: counts, 3-char floor, predicate-recomputed golds, echo safety ----
    for name, items in list(GEN_DISJOINT_POOL.items()) + list(GEN_BANK_ADJACENT.items()):
        add(f"{name}:count", 8 <= len(items) <= 32)   # banks are 8..32; a truncated or
        # empty bank still fails, but the floor no longer hard-codes two legal sizes
        for q, gold, variants in items:
            tag = f"{name}:{q[:44]}"
            answers = [gold] + list(variants)
            add(f"{tag}:all_answers_gte_3_chars", all(len(a) >= 3 for a in answers))
            add(f"{tag}:no_answer_in_function_word_stoplist",
                all(a.lower() not in _STOPLIST for a in answers))
            pred = _predicate(name, q)
            add(f"{tag}:gold_satisfies_predicate", pred(gold))
            cands = _list_candidates(q)
            if cands:
                sat = [c for c in cands if pred(c)]
                add(f"{tag}:exactly_one_candidate_correct", len(sat) == 1 and sat[0] == gold)
                add(f"{tag}:gold_in_candidates", gold in cands)
                # no accepted answer may match a DISTRACTOR (a match must never score an echo of
                # the wrong option): variant sets do not contain other options of the same item
                distr = [c for c in cands if c != gold]
                add(f"{tag}:answers_disjoint_from_distractors",
                    all(not _contains(a, _norm_tokens(d)) and not _contains(d, _norm_tokens(a))
                        for a in answers for d in distr))
            else:
                # free-form: the gold (and variants) must NOT appear in the question, or a bare echo
                # of the prompt would false-positive
                qt = _norm_tokens(q)
                add(f"{tag}:answers_absent_from_question",
                    all(not _contains(a, qt) for a in answers))

    # ---- matcher: word-boundary containment ----
    add("match:substring_number_rejected", _contains("200", _norm_tokens("the answer is 1200")) is False)
    add("match:exact_number_accepted", _contains("200", _norm_tokens("the answer is 200.")) is True)
    add("match:singular_vs_plural_rejected", _contains("cats", _norm_tokens("one cat")) is False)
    add("match:case_and_punct_normalized", _contains("Table", _norm_tokens("  TABLE!  ")) is True)
    add("match:multiword_contiguous", _contains("new york", _norm_tokens("i pick new york city")) is True)

    # ---- repetition guard ----
    add("rep:single_token_repeated", is_degenerate([7] * 8) is True)
    add("rep:five_of_eight_one_token", is_degenerate([7, 7, 7, 7, 7, 1, 2, 3]) is True)  # 0.625 > 0.6
    add("rep:half_not_degenerate", is_degenerate([7, 7, 7, 7, 1, 2, 3, 4]) is False)     # 0.5 <= 0.6
    add("rep:varied_ok", is_degenerate([1, 2, 3, 4, 5, 6, 7, 8]) is False)
    item = GEN_DISJOINT_POOL["ORTH_FIRST_GEN"][0]           # gold "table"
    s, f = score_item(item, ("table table table table", [9] * 8))
    add("rep:zeroes_containment_and_counts", s == 0 and f["repetition"] == 1)

    # ---- echo guard ----
    s, f = score_item(item, ("table, apple, mango, river", [1, 2, 3, 4, 5, 6, 7, 8]))
    add("echo:list_parrot_zeroed", s == 0 and f["echo"] == 1)
    s, f = score_item(item, ("The answer is table.", [1, 2, 3, 4, 5]))
    add("echo:single_candidate_scores", s == 1 and f["echo"] == 0)
    anto = GEN_DISJOINT_POOL["ANTONYM_GEN"][6]              # light -> dark, variant heavy
    s, _ = score_item(anto, ("heavy", [1, 2]))
    add("variants:accepted_variant_scores", s == 1)
    s, _ = score_item(anto, ("bright", [1, 2]))
    add("variants:wrong_answer_zero", s == 0)

    # ---- fake-model measurement pipeline (CPU) ----
    _key = {}
    for nm, its in list(GEN_DISJOINT_POOL.items()) + list(GEN_BANK_ADJACENT.items()):
        for q, g, v in its:
            _key[q] = g

    def _question_of(prompt_text):
        for q in _key:
            if q in prompt_text:
                return q
        return None

    def oracle(prompt_text):
        q = _question_of(prompt_text)
        return (f"The answer is {_key[q]}." if q else "unknown"), [1, 2, 3, 4, 5]

    def degenerate(prompt_text):
        return "yes yes yes yes yes yes yes yes", [11] * 8

    r = measure_all_gen(decode_fn=oracle)
    add("fake:oracle_all_ones", all(v == 1.0 for v in r["scores"].values()))
    add("fake:oracle_no_guard_fires", r["repetition_guard_fires"] == 0 and r["echo_guard_fires"] == 0)
    add("fake:oracle_guard_excl_equals_scores",
        r["scores_guard_excl"] == r["scores"])          # no guard fires -> excluded == raw
    r = measure_all_gen(decode_fn=degenerate)
    add("fake:degenerate_all_zero", all(v == 0.0 for v in r["scores"].values()))
    add("fake:degenerate_guard_excl_all_zero",
        all(v == 0.0 for v in r["scores_guard_excl"].values()))  # every item guarded -> n_scored 0
    n_items = sum(len(v) for v in GEN_DISJOINT_POOL.values()) + sum(len(v) for v in GEN_BANK_ADJACENT.values())
    add("fake:degenerate_fires_counted", r["repetition_guard_fires"] == n_items)

    # ---- format-invariance scaffolding via the fake hook ----
    fi = format_invariance_check(decode_fn=oracle)
    add("format:invariant_under_oracle", fi["abs_delta"] == 0.0)

    def verbose_breaks(prompt_text):
        if "read the following question carefully" in prompt_text:
            return "hmm, let me think about", [21, 22, 23, 24, 25]
        return oracle(prompt_text)

    fi = format_invariance_check(decode_fn=verbose_breaks)
    add("format:sensitive_fake_detected", fi["abs_delta"] == 1.0 and fi["aggregate_plain"] == 1.0)

    # ---- guard-excluded (subtractive) accuracy arithmetic (F5) ----
    _items = GEN_DISJOINT_POOL["ORTH_FIRST_GEN"]                 # 16 list-format items

    def half_echo(prompt_text):
        q = _question_of(prompt_text)
        cands = _list_candidates(q or "")
        idx = next((i for i, it in enumerate(_items) if it[0] == q), -1)
        if idx >= 0 and idx % 4 == 0 and len(cands) >= 2:        # 4 of 16 -> parrot 2 candidates
            return ", ".join(cands[:2]), [1, 2, 3, 4, 5, 6, 7, 8]
        return (f"The answer is {_key.get(q)}.", [1, 2, 3, 4, 5])
    r = measure_gen_subtask(_items, half_echo)
    add("subtractive:echo_zeroes_raw_acc", r["echo_fires"] == 4 and abs(r["acc"] - 12 / 16) < 1e-9)
    add("subtractive:guard_excl_removes_zeroed_items", abs(r["acc_guard_excl"] - 1.0) < 1e-9)

    # ---- selection / guard / paired delta (bars passed in, never stored here) ----
    # synthetic scores over the CURRENT pool: three straddle the floor (0.9375 keeps, 0.90 keeps as
    # an inclusive boundary, 0.875/0.8125 drop) so the fixture exercises selection AT the bar rather
    # than merely far from it. The two retained fixture families are scored under the floor here,
    # which is also where the clean base actually puts them.
    base = {"PLURAL_GEN": 0.9375, "PAST_TENSE_GEN": 0.90, "SEQ_GEN": 1.0,
            "CAPITAL_GEN": 0.9375, "ELEMENT_GEN": 0.875,
            "ORTH_FIRST_GEN": 0.875, "ANTONYM_GEN": 0.8125,
            "MUL_GEN": 1.0}
    sel, okk = select_disjoint(base, floor=0.90, need=3)
    add("select:keeps_at_floor",
        sel == sorted(["PLURAL_GEN", "PAST_TENSE_GEN", "SEQ_GEN", "CAPITAL_GEN"]))
    add("select:excludes_below_floor", "ELEMENT_GEN" not in sel and "ANTONYM_GEN" not in sel)
    add("select:excludes_bank_adjacent", "MUL_GEN" not in sel)
    add("select:ok", okk is True)
    add("select:too_few_not_ok", select_disjoint({"SEQ_GEN": 1.0}, floor=0.90, need=3)[1] is False)
    add("guard:clean_passes", battery_guard(base, sel, agg_floor=0.80, subtask_floor=0.70) is True)
    add("guard:low_subtask_fails",
        battery_guard({**base, "SEQ_GEN": 0.5}, sel, agg_floor=0.80, subtask_floor=0.70) is False)
    fx = {n: 0.95 for n in sel}
    add("paired:delta_sign", paired_delta(fx, {n: 0.80 for n in sel}, sel) == 0.15)
    add("paired:acc_above_fixed_negative", paired_delta(fx, {n: 1.0 for n in sel}, sel) == -0.05)

    res = {"selftest": True, "all_ok": ok, "n": len(checks), "n_ok": sum(c["ok"] for c in checks)}
    (HERE / "capability_battery_gen_selftest_INVALID.json").write_text(
        json.dumps({**res, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(f"capability_battery_gen selftest: all_ok={ok} ({res['n_ok']}/{res['n']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 0 if _selftest()["all_ok"] else 1
    print("disjoint pool:", list(GEN_DISJOINT_POOL), "| bank-adjacent:", list(GEN_BANK_ADJACENT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    "ORTH_LAST_GEN": [
        ("Which of these words ends with the letter T: carpet, window, garden, mirror?", "carpet", []),
        ("Which of these words ends with the letter E: candle, forest, walnut, copper?", "candle", []),
        ("Which of these words ends with the letter R: silver, melon, basket, yellow?", "silver", []),
        ("Which of these words ends with the letter N: lantern, marble, spider, helmet?", "lantern", []),
        ("Which of these words ends with the letter Y: tunnel, valley, ribbon, rocket?", "valley", []),
        ("Which of these words ends with the letter O: diamond, mango, turtle, hammer?", "mango", []),
        ("Which of these words ends with the letter D: summer, diamond, purple, kitten?", "diamond", []),
        ("Which of these words ends with the letter L: needle, barrel, monkey, violet?", "barrel", []),
        ("Which of these words ends with the letter W: magnet, carrot, window, sunset?", "window", []),
        ("Which of these words ends with the letter G: donkey, pencil, evening, branch?", "evening", []),
        ("Which of these words ends with the letter K: flower, jacket, hammock, planet?", "hammock", []),
        ("Which of these words ends with the letter P: mirror, tiger, tulip, button?", "tulip", []),
        ("Which of these words ends with the letter S: river, garden, marble, compass?", "compass", []),
        ("Which of these words ends with the letter M: rocket, copper, yellow, museum?", "museum", []),
        ("Which of these words ends with the letter H: spider, walnut, melon, starfish?", "starfish", []),
        ("Which of these words ends with the letter E: forest, carrot, summer, orange?", "orange", []),
    ],
    "CONTAINS_GEN": [
        ("Which of these words contains the letter Z: puzzle, garden, mirror, carpet?", "puzzle", []),
        ("Which of these words contains the letter V: velvet, mango, button, copper?", "velvet", []),
        ("Which of these words contains the letter X: saxophone, candle, ribbon, turtle?", "saxophone", []),
        ("Which of these words contains the letter J: banjo, flower, silver, helmet?", "banjo", []),
        ("Which of these words contains the letter Q: marble, conquest, sunset, tiger?", "conquest", []),
        ("Which of these words contains the letter W: melon, sandwich, carrot, purple?", "sandwich", []),
        ("Which of these words contains the letter F: monkey, muffin, barrel, donkey?", "muffin", []),
        ("Which of these words contains the letter K: tunnel, blanket, summer, violet?", "blanket", []),
        ("Which of these words contains the letter Z: valley, needle, blizzard, rocket?", "blizzard", []),
        ("Which of these words contains the letter X: hammer, planet, mixture, spider?", "mixture", []),
        ("Which of these words contains the letter V: basket, yellow, gravel, mirror?", "gravel", []),
        ("Which of these words contains the letter J: copper, lantern, project, walnut?", "project", []),
        ("Which of these words contains the letter Q: diamond, kitten, branch, lacquer?", "lacquer", []),
        ("Which of these words contains the letter W: magnet, tulip, forest, firewood?", "firewood", []),
        ("Which of these words contains the letter Z: museum, compass, starfish, horizon?", "horizon", []),
        ("Which of these words contains the letter X: evening, hammock, pencil, textbook?", "textbook", []),
    ],
    "ANTONYM_GEN": [
        # golds reworded away from the high-frequency-function-word stoplist (panel MAJOR M1): the
        # eight items whose gold was a bare stoplist adjective (cold, down, sad, dry, short, late,
        # full, old) now use lower-frequency question words with lower-incidental golds. Index 6
        # (light -> dark) is unchanged (a selftest fixture references it).
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
    "ALPHA_GEN": [
        ("Which of these words comes first in alphabetical order: mango, apple, river, table?", "apple", []),
        ("Which of these words comes first in alphabetical order: banana, cherry, garden, walnut?", "banana", []),
        ("Which of these words comes first in alphabetical order: copper, silver, basket, marble?", "basket", []),
        ("Which of these words comes first in alphabetical order: window, tunnel, carpet, needle?", "carpet", []),
        ("Which of these words comes first in alphabetical order: donkey, monkey, turtle, kitten?", "donkey", []),
        ("Which of these words comes first in alphabetical order: flower, forest, evening, garden?", "evening", []),
        ("Which of these words comes first in alphabetical order: helmet, hammer, garden, mirror?", "garden", []),
        ("Which of these words comes first in alphabetical order: lantern, needle, hammock, violet?", "hammock", []),
        ("Which of these words comes first in alphabetical order: jacket, kitten, lemon, insect?", "insect", []),
        ("Which of these words comes first in alphabetical order: mirror, planet, magnet, lantern?", "lantern", []),
        ("Which of these words comes first in alphabetical order: sunset, rocket, purple, mango?", "mango", []),
        ("Which of these words comes first in alphabetical order: tulip, ribbon, needle, orange?", "needle", []),
        ("Which of these words comes first in alphabetical order: spider, orange, puzzle, ribbon?", "orange", []),
        ("Which of these words comes first in alphabetical order: valley, summer, tunnel, pencil?", "pencil", []),
        ("Which of these words comes first in alphabetical order: yellow, walnut, ribbon, sunset?", "ribbon", []),
        ("Which of these words comes first in alphabetical order: window, violet, turtle, silver?", "silver", []),
    ],
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
    ],
    "SEQ_GEN": [
        ("Which month of the year comes immediately after January?", "february", []),
        ("Which month of the year comes immediately after February?", "march", []),
        ("Which month of the year comes immediately after March?", "april", []),
        # April -> May dropped: the gold "may" is a bare hedging modal a degraded model emits as
        # fluff, scoring CORRECT by incidental containment (panel MAJOR M1). Replaced with a
        # day-successor item so the sub-task count stays 16 and no gold is a stoplist word.
        ("Which day of the week comes immediately after Wednesday?", "thursday", []),
        ("Which month of the year comes immediately after May?", "june", []),
        ("Which month of the year comes immediately after June?", "july", []),
        ("Which month of the year comes immediately after July?", "august", []),
        ("Which month of the year comes immediately after August?", "september", []),
        ("Which month of the year comes immediately after September?", "october", []),
        ("Which month of the year comes immediately after October?", "november", []),
        ("Which month of the year comes immediately after November?", "december", []),
        ("Which month of the year comes immediately after December?", "january", []),
        ("Which day of the week comes immediately after Monday?", "tuesday", []),
        ("Which day of the week comes immediately after Tuesday?", "wednesday", []),
        ("Which day of the week comes immediately after Friday?", "saturday", []),
        ("Which day of the week comes immediately after Sunday?", "monday", []),
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


def _predicate(subtask, question):
    """Ground-truth predicate for one item: f(answer_string) -> bool. Non-circular: recomputed from
    string ops / maps / arithmetic, never from the stored gold."""
    if subtask == "ORTH_FIRST_GEN":
        L = re.search(r"starts with the letter (\w):", question)[1]
        return lambda o: o[0].upper() == L.upper()
    if subtask == "ORTH_LAST_GEN":
        L = re.search(r"ends with the letter (\w):", question)[1]
        return lambda o: o[-1].upper() == L.upper()
    if subtask == "CONTAINS_GEN":
        L = re.search(r"contains the letter (\w):", question)[1]
        return lambda o: L.upper() in o.upper()
    if subtask == "ALPHA_GEN":
        cands = _list_candidates(question)
        return lambda o: o == min(cands)
    if subtask == "ANTONYM_GEN":
        w = re.search(r"the word '(\w+)'\?", question)[1]
        return lambda o: _ANTONYM.get(w) == o
    if subtask == "PLURAL_GEN":
        w = re.search(r"the word '(\w+)'\?", question)[1]
        return lambda o: o == w + "s"
    if subtask == "SEQ_GEN":
        m = re.search(r"immediately after (\w+)\?", question)[1].lower()
        seq = _MONTHS if m in _MONTHS else _DAYS
        return lambda o: o.lower() == seq[(seq.index(m) + 1) % len(seq)]
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
        add(f"{name}:count", len(items) in (8, 16))
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
    base = {"ORTH_FIRST_GEN": 1.0, "ORTH_LAST_GEN": 0.9375, "CONTAINS_GEN": 0.875,
            "ANTONYM_GEN": 1.0, "ALPHA_GEN": 0.8125, "PLURAL_GEN": 0.9375, "SEQ_GEN": 1.0,
            "MUL_GEN": 1.0}
    sel, okk = select_disjoint(base, floor=0.90, need=3)
    add("select:keeps_at_floor",
        sel == sorted(["ORTH_FIRST_GEN", "ORTH_LAST_GEN", "ANTONYM_GEN", "PLURAL_GEN", "SEQ_GEN"]))
    add("select:excludes_below_floor", "CONTAINS_GEN" not in sel and "ALPHA_GEN" not in sel)
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

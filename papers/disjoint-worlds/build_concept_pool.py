# -*- coding: utf-8 -*-
"""build_concept_pool.py — a concept battery that is a RECORDED SUBSET of a RECORDED POOL.

WHY THIS EXISTS
---------------
The battery this arc's published reads are scored over (`run_g0clear._BANK`, 462 concepts) is a
hand-authored string literal. Its only filter is an order-preserving dedup that removes three
words. No pool of candidates was ever written down, so the EXTRACTION TERM — the share of
candidate concepts that survived into the battery at all — is not computable after the fact.
Provenance of the file is complete; provenance of the SELECTION is absent. See
`ADDENDUM_battery_and_holdout_limits_2026_09_01.md`.

This script builds the successor. It does not touch `run_g0clear.py` and does not regenerate the
462-word battery, which stays exactly as it is.

WHAT IT GUARANTEES
------------------
1. A POOL exists as an artifact BEFORE any selection, derived by a stated rule from a pinned,
   nameable source.
2. Every filter is applied in a stated order and records its name, its rule, its survivor count
   and exactly how many candidates it removed.
3. Every stage's surviving set is fingerprinted (sha256 over the sorted, newline-joined list), so
   a stage can be verified without the receipt enumerating half a million strings.
4. The final battery is a SEEDED uniform sample of the eligible set. The seed is in the receipt.
5. The survival ratio final/pool is reported, and so is every intermediate ratio.

WHAT IT DOES NOT GUARANTEE
--------------------------
The filters below are justified in their `why` strings and their costs are stated in their `cost`
strings. They are still choices made by a person. What changes is that each choice is now a named
rule with a counted price attached, so a reader can disagree with a specific filter and recompute
what it cost. That is the whole of the improvement; it is not a claim that the resulting battery
is unbiased.

No GPU. No model weights are loaded — only four tokenizer files. Runs in roughly two minutes on
CPU, most of it counting the corpus.

  python build_concept_pool.py                 # writes concept_pool.json
  python build_concept_pool.py --dump-stages   # also writes concept_pool_stages.json (large)
  python build_concept_pool.py --smoke         # first corpus shard only; writes nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------
# PINS. Everything the output depends on, named and version-locked. A pool whose source drifts is
# a pool that cannot be replayed, which is the defect this file exists to fix.
# ---------------------------------------------------------------------------------------------
CORPUS_REPO = "wikitext"
CORPUS_CONFIG = "wikitext-103-raw-v1"
CORPUS_SPLIT = "train"
CORPUS_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CORPUS_LICENCE = "CC BY-SA 3.0 (WikiText is derived from verified-good Wikipedia articles)"
CORPUS_CITE = "Merity, Xiong, Bradbury, Socher (2016), Pointer Sentinel Mixture Models, arXiv:1609.07843"

# The models the arc actually compares. A concept must be one token in EVERY one of them or the
# same string is not the same stimulus across the comparison being made.
TARGET_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",   # source A in b31v2 / b34v3 / g0clear
    "meta-llama/Llama-3.2-1B-Instruct",   # target, same family
    "google/gemma-2-2b-it",               # target, the decisive cross-family cell (b31v2's `hf`)
    "Qwen/Qwen2.5-1.5B-Instruct",         # target, context cell
]
# gemma-2-2b-it and gemma-2-2b carry byte-identical vocabularies (verified by comparing
# get_vocab() on both), so F6 is insensitive to which of the two is named here. The -it variant
# is named because it is the model b31v2_result.json records as the decisive target.

# The neutral contrast word `run_g0clear.extract_multi` differences against. A concept that is not
# the same token length as this word changes the sentence length of the contrast pair.
NEUTRAL_WORD = "object"

FREQ_FLOOR = 500          # corpus occurrences, all case forms
MIN_LEN, MAX_LEN = 3, 14  # characters
PROPER_CAP_SHARE = 0.5    # >= this share of non-lowercase surface forms => treated as a name
BATTERY_N = 462           # matched to the 2026-08 battery so the reproduction gate is like-for-like
POOL_SEED = 52            # b52; recorded in the receipt

# Inflectional suffixes whose stripped base, if itself a survivor, makes the pair near-duplicate.
# Deliberately EXCLUDES -er/-est/-or: agent nouns are distinct concepts (a driver is not a drive)
# and stripping them produced pure false positives in development (beer<-bee, forest<-fore).
_SUFFIXES = ("ies", "es", "s", "ing", "ed", "ly")


def _sha(items) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# STAGE 0 — THE POOL
# ---------------------------------------------------------------------------------------------
def build_pool(smoke: bool = False):
    """POOL RULE (stated before any filter runs):

    Every distinct case-folded string obtained by splitting the pinned corpus split on
    whitespace. WikiText is distributed space-tokenized (punctuation is space-separated), so a
    whitespace split yields the corpus's own token inventory, not a tokenization of ours.

    What this rule already excludes, and therefore never prices: anything the corpus does not
    contain, and any multi-word concept. Both are real narrowings and they are the pool's
    boundary rather than a filter's cost. Multi-word concepts are out of scope for this battery
    because the extraction template renders one word into one slot.
    """
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    root = Path(snapshot_download(
        repo_id=CORPUS_REPO, repo_type="dataset", revision=CORPUS_REVISION,
        allow_patterns=[f"{CORPUS_CONFIG}/{CORPUS_SPLIT}-*.parquet"]))
    shards = sorted((root / CORPUS_CONFIG).glob(f"{CORPUS_SPLIT}-*.parquet"))
    if not shards:
        raise SystemExit(f"no {CORPUS_SPLIT} shards under {root / CORPUS_CONFIG}")
    if smoke:
        shards = shards[:1]

    lower = Counter()    # surface form was entirely lowercase
    initcap = Counter()  # surface form was Xxxx
    other = Counter()    # anything else (ALLCAPS, MiXeD)
    lines = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=20000, columns=["text"]):
            for s in batch.column("text").to_pylist():
                if not s:
                    continue
                lines += 1
                for w in s.split():
                    k = w.lower()
                    if w.islower():
                        lower[k] += 1
                    elif w[:1].isupper() and w[1:].islower():
                        initcap[k] += 1
                    else:
                        other[k] += 1

    types = sorted(set(lower) | set(initcap) | set(other))
    total = {k: lower[k] + initcap[k] + other[k] for k in types}
    meta = {
        "rule": ("distinct case-folded whitespace-delimited strings of the pinned corpus split; "
                 "no filtering of any kind at this stage"),
        "source": {"repo_id": CORPUS_REPO, "repo_type": "dataset", "config": CORPUS_CONFIG,
                   "split": CORPUS_SPLIT, "revision": CORPUS_REVISION,
                   "licence": CORPUS_LICENCE, "citation": CORPUS_CITE,
                   "shards": [s.name for s in shards]},
        "lines_read": lines,
        "token_occurrences": int(sum(total.values())),
        "pool_size": len(types),
        "sha256_pool": _sha(types),
        "not_priced_here": ("strings absent from this corpus, and every multi-word concept: both "
                            "are excluded by the pool's own boundary and no filter below charges "
                            "for them"),
    }
    return types, total, initcap, other, meta


# ---------------------------------------------------------------------------------------------
# THE FILTER CHAIN. Order matters and is fixed here; each entry states why it exists and what it
# costs. A filter with no stated cost is a filter no reader can price, which is the failure mode
# this file was written against.
# ---------------------------------------------------------------------------------------------
def apply_chain(types, total, initcap, other, tokenizers, log_cap=25):
    stages = []
    survivors = list(types)

    def step(name, rule, why, cost, keep_fn, cap=None):
        nonlocal survivors
        cap = log_cap if cap is None else cap
        before = list(survivors)
        kept, dropped = [], []
        for t in before:
            (kept if keep_fn(t) else dropped).append(t)
        survivors = kept
        stages.append({
            "filter": name, "rule": rule, "why": why, "cost": cost,
            "n_in": len(before), "n_removed": len(dropped), "n_out": len(kept),
            "removed_fraction_of_input": round(len(dropped) / len(before), 6) if before else 0.0,
            "removed_examples": dropped[:cap],
            "removed_enumerated": len(dropped) <= cap,
            "sha256_survivors": _sha(kept),
        })
        return stages[-1]

    ascii_alpha = re.compile(r"^[a-z]+$")
    step(
        "F1_ascii_alphabetic",
        "keep t matching ^[a-z]+$ after case-folding",
        "The battery renders a bare English word into an English sentence template. Strings "
        "carrying digits, punctuation, or non-ASCII letters are not bare English words, and the "
        "corpus's whitespace inventory is full of them (numerals, the @-@ hyphen marker, "
        "transliterations).",
        "Refuses accented loanwords (cafe with an acute) and every hyphenated compound, which "
        "WikiText writes with @-@ separators. Those are real concepts and this filter will not "
        "admit them.",
        lambda t: bool(ascii_alpha.match(t)),
    )

    step(
        "F2_length_3_to_14",
        f"keep {MIN_LEN} <= len(t) <= {MAX_LEN}",
        "One- and two-character strings in this corpus are dominated by initials, roman numerals "
        "and unit abbreviations rather than concepts; strings longer than 14 characters are "
        "dominated by concatenation artefacts and chemical names.",
        "Loses genuine short concepts (ox, ax, pi) and long ones (thunderstorm is fine at 12, but "
        "e.g. responsibility at 14 is at the boundary and anything longer is refused outright).",
        lambda t: MIN_LEN <= len(t) <= MAX_LEN,
    )

    step(
        "F3_frequency_floor",
        f"keep total corpus occurrences (all case forms) >= {FREQ_FLOOR}",
        "The concept vector is a mean over 12 template renderings of a sentence-final hidden "
        "state. For a type the model has seen rarely, that mean is dominated by tokenizer "
        "segmentation and noise rather than by a stable representation, and a battery of such "
        "items measures the extraction pipeline instead of the model.",
        "This is by far the largest narrowing in the chain and it BIASES the battery toward "
        "common English. A read scored on this battery licenses no claim about rare or technical "
        "concepts. Separately, WikiText frequency is a PROXY for the target models' pretraining "
        "frequency, which is not observable; the substitution is a known and unmeasured "
        "approximation.",
        lambda t: total[t] >= FREQ_FLOOR,
    )

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = frozenset(ENGLISH_STOP_WORDS)
    step(
        "F4_not_function_word",
        "drop t in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS "
        f"({len(stop)} entries, Glasgow IR stop list, shipped under scikit-learn's BSD-3 licence)",
        "The template 'The {c} was the first thing everyone noticed.' is meaningless for "
        "determiners, prepositions, pronouns and auxiliaries. Such items would contribute "
        "template-artefact vectors, not concept vectors.",
        "This list is hand-curated and idiosyncratic — scikit-learn's own documentation warns "
        "against using it uncritically — so this filter reintroduces exactly the kind of "
        "unaccountable hand-authorship the pool is meant to remove. Two mitigations: it is a "
        "published, citable, version-pinned list rather than one written here, and every word it "
        "removes is enumerated below. Its cost is measurable: on the 2026-08 battery it would "
        "have refused 'fire' and 'back', both contentful concepts.",
        lambda t: t not in stop,
        cap=400,
    )

    step(
        "F5_not_proper_noun_dominant",
        f"drop t where (initcap + other) / total >= {PROPER_CAP_SHARE}",
        "Proper nouns are entities. A battery of entities measures name recall; this arc's claims "
        "are about concept representation. Nothing in the 2026-08 chain prevented an entity from "
        "entering the battery — it happens to contain none, but by authorship, not by rule.",
        "Sentence-initial capitalisation inflates the ratio for every word, and the 0.5 bar is a "
        "judgement call with no power analysis behind it. It removes polysemous words that are "
        "also frequent names. Measured casualties in this run: 'brown' (a colour in the 2026-08 "
        "battery), and the contentful nouns 'academy', 'abbey', 'admiral', 'act', 'march'. This "
        "filter is the least principled in the chain and is flagged as such.",
        lambda t: (initcap[t] + other[t]) / total[t] < PROPER_CAP_SHARE,
    )

    def single_token_everywhere(t):
        return all(len(tk.encode(" " + t, add_special_tokens=False)) == 1
                   for tk in tokenizers.values())

    step(
        "F6_single_token_in_every_target",
        "keep t where tokenizer.encode(' ' + t) has length 1 for EVERY model in TARGET_MODELS",
        "This is the filter the 2026-08 chain lacks entirely. Two reasons it belongs. (a) "
        "`extract_multi` differences template.format(c=concept) against template.format(c="
        f"'{NEUTRAL_WORD}'), reads the LAST token of each sentence, and subtracts. The neutral "
        "word is one token in all four models; a multi-token concept therefore makes the two "
        "sentences different lengths, so the two hidden states being subtracted sit at different "
        "absolute positions. (b) A string that is one token in gemma and three in Llama is not "
        "the same stimulus in the two models, which is the comparison the cross-family reads "
        "actually make.",
        "Narrows hard toward high-frequency Anglo-Saxon vocabulary present in all four BPE "
        "vocabularies, which is a further and unquantified restriction of what 'a concept' means "
        "here. Measured on the 2026-08 battery: 35 of its 462 words are multi-token in "
        "Llama-3.2 and in Qwen2.5 (2 of 462 in gemma-2), and 6 of its 70 held-out concepts are "
        "multi-token in at least one target — so this filter would have changed the existing "
        "battery, not merely tidied it. Note the read is sentence-final rather than on the "
        "concept token itself, so the defect this filter removes is a length/position confound, "
        "NOT a read taken on a word-piece; the smaller claim is the true one.",
        single_token_everywhere,
    )

    surviving = set(survivors)

    def inflection_base(t):
        for suf in _SUFFIXES:
            if not t.endswith(suf):
                continue
            b = t[:-len(suf)]
            if len(b) >= MIN_LEN and b in surviving:
                return b
            if suf == "ies":
                b2 = t[:-3] + "y"
                if len(b2) >= MIN_LEN and b2 in surviving:
                    return b2
            if suf in ("ing", "ed"):
                b2 = t[:-len(suf)] + "e"
                if len(b2) >= MIN_LEN and b2 in surviving:
                    return b2
        return None

    st = step(
        "F7_no_inflection_of_a_survivor",
        f"drop t if stripping one of {list(_SUFFIXES)} (with -ies->-y and -e restoration) yields a "
        "base of length >= 3 that is itself a survivor of F6",
        "Identification is argmin over the battery. A battery holding both 'friend' and 'friends' "
        "asks the reader to discriminate near-duplicates, which measures morphological "
        "resolution rather than concept identification and is not the quantity any gate in this "
        "arc is about.",
        "Partial and asymmetric. It cannot see irregular inflection (men, children, ran all "
        "survive), and it produces false positives where a suffix is accidental: 'early' is "
        "removed here because 'ear' survives. Nominalised gerunds that are genuine concepts "
        "(building, painting, meeting) are removed alongside true duplicates. Verified on the "
        "2026-08 battery: it would have removed none of its 462 words, which is why -er/-est are "
        "excluded from the suffix list.",
        lambda t: inflection_base(t) is None,
    )
    st["removed_examples"] = [f"{t}<-{inflection_base(t)}"
                              for t in st["removed_examples"]]

    step(
        "F8_exact_duplicate",
        "order-preserving dedup",
        "This is the ONLY filter the 2026-08 chain has (run_g0clear.py:66-67), where it removed "
        "three words. It is kept here so the comparison is explicit rather than implied.",
        "Nothing: case-folding at pool construction already made types unique, so this removes "
        "zero. A stage that removes zero is worth recording precisely because the 2026-08 chain "
        "consisted of nothing else.",
        _keep_first_occurrence(),
    )

    return survivors, stages


def _keep_first_occurrence():
    """The 2026-08 chain's one filter, verbatim in spirit: keep a word the first time only."""
    seen = set()
    return lambda t: not (t in seen or seen.add(t))


def select_battery(eligible, n=BATTERY_N, seed=POOL_SEED):
    """SELECTION RULE: seeded uniform sample without replacement from the sorted eligible set.

    Sorted first so the sampling frame does not depend on dict or set iteration order, which is
    not stable across Python processes. This is the step that must be mechanical: the moment a
    human picks which survivors go in, the extraction term stops being computable again.
    """
    frame = sorted(set(eligible))
    if len(frame) < n:
        raise SystemExit(f"eligible set has {len(frame)} < {n}; loosen a filter or lower BATTERY_N")
    idx = np.random.default_rng(seed).permutation(len(frame))[:n]
    return sorted(frame[i] for i in idx), frame


def old_battery():
    """The committed 2026-08 battery, read out of run_g0clear.py without importing torch."""
    src = (HERE / "run_g0clear.py").read_text(encoding="utf-8")
    raw = re.search(r'_BANK = """(.*?)"""', src, re.S).group(1).split()
    seen = set()
    return raw, [c for c in raw if not (c in seen or seen.add(c))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="first corpus shard only; writes nothing")
    ap.add_argument("--dump-stages", action="store_true",
                    help="also write concept_pool_stages.json with full per-stage survivor lists")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    import sklearn
    import transformers as _tf

    print("loading tokenizers (no weights)...", flush=True)
    tokenizers, tok_pins = {}, {}
    for m in TARGET_MODELS:
        tokenizers[m] = AutoTokenizer.from_pretrained(m, local_files_only=True)
        tok_pins[m] = {"vocab_size": len(tokenizers[m]),
                       f"tokens_for_{NEUTRAL_WORD}":
                           len(tokenizers[m].encode(" " + NEUTRAL_WORD,
                                                    add_special_tokens=False))}
    bad = {m: p for m, p in tok_pins.items() if p[f"tokens_for_{NEUTRAL_WORD}"] != 1}
    if bad:
        print(f"NOTE: neutral word {NEUTRAL_WORD!r} is not single-token in {sorted(bad)} — "
              f"the F6 justification is weaker for those models and this is recorded, not fixed",
              flush=True)

    print("counting the pinned corpus...", flush=True)
    types, total, initcap, other, pool_meta = build_pool(smoke=args.smoke)
    print(f"POOL = {pool_meta['pool_size']} types "
          f"from {pool_meta['token_occurrences']} occurrences", flush=True)

    survivors, stages = apply_chain(types, total, initcap, other, tokenizers)
    for s in stages:
        print(f"  {s['filter']:34s} {s['n_in']:>7d} -> {s['n_out']:>7d} "
              f"(-{s['n_removed']})", flush=True)

    battery, frame = select_battery(survivors)
    raw_old, old = old_battery()

    pool_n = pool_meta["pool_size"]
    receipt = {
        "artifact": "concept_pool.json",
        "produced_by": "build_concept_pool.py",
        "purpose": ("a candidate pool recorded BEFORE the selection, so the extraction term "
                    "final/pool is computable for every battery derived from it"),
        "prereg": "PREREG_b52_pooled_battery_2026_09_01.md",
        "addendum": "ADDENDUM_battery_and_holdout_limits_2026_09_01.md",
        "smoke": args.smoke,
        "seed": POOL_SEED,
        "deterministic": ("every stage sorts before it samples; the only randomness is "
                          f"numpy.random.default_rng({POOL_SEED}).permutation over the sorted "
                          "eligible frame"),
        "environment_pins": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "transformers": _tf.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "pool": pool_meta,
        "target_models": tok_pins,
        "neutral_contrast_word": NEUTRAL_WORD,
        "filters": stages,
        "eligible": {
            "n": len(frame),
            "sha256": _sha(frame),
            "words": frame,
        },
        "battery": {
            "n": len(battery),
            "rule": (f"uniform sample without replacement of {BATTERY_N} from the sorted eligible "
                     f"set at seed {POOL_SEED}; size matched to the 2026-08 battery so the "
                     f"reproduction gate compares like with like (same N, same 1/70 chance under "
                     f"the same 70/392 split)"),
            "sha256": _sha(battery),
            "words": battery,
        },
        "survival": {
            "pool": pool_n,
            "eligible": len(frame),
            "battery": len(battery),
            "survival_ratio_battery_over_pool": round(len(battery) / pool_n, 8),
            "eligible_ratio_eligible_over_pool": round(len(frame) / pool_n, 8),
            "sampling_fraction_battery_over_eligible": round(len(battery) / len(frame), 6),
            "means": ("survival_ratio is the extraction term E for this battery: the share of "
                      "recorded candidates that reached the scored set. For the 2026-08 battery "
                      "the same quantity is UNCHECKABLE, because no pool was recorded."),
        },
        "comparison_to_2026_08_battery": {
            "file": "run_g0clear.py lines 31-67",
            "raw_tokens_in_literal": len(raw_old),
            "after_its_only_filter": len(old),
            "its_filters": ["order-preserving dedup (removed 3: chicken, orange, mushroom)"],
            "its_pool": None,
            "its_survival_ratio": "UNCHECKABLE — no pool artifact exists",
            "old_words_reaching_this_eligible_set": len(set(old) & set(frame)),
            "old_words_in_this_battery": len(set(old) & set(battery)),
            "old_words_refused_by_F6_single_token": sorted(
                c for c in old
                if any(len(tk.encode(" " + c, add_special_tokens=False)) != 1
                       for tk in tokenizers.values())),
        },
        "known_limits_of_this_artifact": [
            "The pool is one English corpus of Wikipedia prose. Concept frequency in WikiText is "
            "a proxy for frequency in the target models' pretraining, which is not observable.",
            "Filter thresholds (frequency floor 500, length 3-14, cap share 0.5) are judgement "
            "calls stated in advance rather than values derived from a power analysis. They are "
            "frozen here so a later run cannot move them after seeing a read.",
            "No part-of-speech filter is applied: no offline tagger is available in this "
            "environment. The successor battery is therefore broader in word class than the "
            "hand-authored one, which is exactly what the reproduction gate in the b52 prereg "
            "exists to price.",
            "F4 and F5 are the two stages that most resemble the defect being repaired — a "
            "curated list and a hand-set threshold. They are logged item-by-item and by count so "
            "the disagreement is at least computable.",
        ],
    }

    if args.smoke:
        print(json.dumps({k: receipt[k] for k in ("survival", "comparison_to_2026_08_battery")},
                         indent=2), flush=True)
        print("\nSMOKE: nothing written.", flush=True)
        return 0

    out = HERE / "concept_pool.json"
    # newline="\n" so the receipt is byte-identical off Windows too; the prereg's VOID condition
    # is byte-reproducibility of this file, and platform line endings would break it silently.
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nwrote {out.name}", flush=True)

    if args.dump_stages:
        dump = {"note": "full per-stage survivor lists; fingerprints match concept_pool.json",
                "pool_sha256": pool_meta["sha256_pool"],
                "pool_words": types,
                "stages": [{"filter": s["filter"], "sha256_survivors": s["sha256_survivors"]}
                           for s in stages]}
        d = HERE / "concept_pool_stages.json"
        d.write_text(json.dumps(dump, ensure_ascii=False) + "\n",
                     encoding="utf-8", newline="\n")
        print(f"wrote {d.name}", flush=True)

    s = receipt["survival"]
    print(f"\nPOOL {s['pool']} -> ELIGIBLE {s['eligible']} -> BATTERY {s['battery']}", flush=True)
    print(f"SURVIVAL RATIO (battery/pool) = {s['survival_ratio_battery_over_pool']}", flush=True)
    print(f"eligible/pool = {s['eligible_ratio_eligible_over_pool']}   "
          f"battery/eligible = {s['sampling_fraction_battery_over_eligible']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""run_realmodel_bridge_v3.py -- frozen by PREREG_realmodel_bridge_v3_2026_10_11 (with amendment 17.2).

THE REAL-MODEL BRIDGE, ATTEMPT THREE. v1 was uninterpretable (recall at ceiling, compare at chance in
130M models); v2 was degenerate (a constant responder scored the label base rate and cleared the floor,
RESULT_realmodel_bridge_2026_07_24.md). v3 does what that RESULT asked for: few-shot exemplars, exact
per-cell balance (a constant responder scores exactly chance), a capability screen at the shortest
distance, a degeneracy guard, chance-normalized metrics, and a paired item-bootstrap interval.

Amendment 17.2 (the operator's one bounded redesign, the last): the compare readout is argmax-shaped like
recall. The question is "Which name has the largest code: alpha, gamma or delta?" and the answer is scored
over the three name tokens at the next position; the correct slot is exactly balanced per cell, so there is
no Yes-prior to drift and a constant responder scores exactly chance; the guard is max-slot <= 0.80 in both
families. Exemplar names (sigma, theta, omega) are disjoint from the test names; C_abl runs at every
distance; C_near and C_overwrite are reported controls at D1; the receipt carries per-item bits and
scores, sha256 lists of prompts, codes and sentences, and the prefix-nesting flag.

Arms: M1 = state-spaces/mamba-1.4b-hf (A = -exp(A_log) real and strictly negative on the loaded weights,
asserted in every layer) and T = EleutherAI/pythia-1.4b. Byte-identical prompts, identical token ids.

  python run_realmodel_bridge_v3.py --smoke [--arms M1,T]
      24 items, D in {16, 1024}, both arms by default (amendment 17.2: the budget extrapolation needs the
      slow arm timed). Smoke items come from their own seed (1016) and their own reserved slice of the
      sentence order and of the code pool: the smoke scores no main-set or pilot item. INVALID-only by the
      gates block. Measures runtime per distance and extrapolates the main run; above 4 hours the receipt
      reads INVALID__budget and the main run refuses to start.
  python run_realmodel_bridge_v3.py --pilot
      the amendment-17.2 pilot: 48 items at D0, seed 1012, its own reserved slice of the sentence order and
      of the code pool, disjoint from the 1011 pilot, the smoke and the main region (asserted by sha256
      lists), both arms, the redesigned compare readout and recall, C_abl at D0 reported. Written to
      realmodel_bridge_v3_pilot2.json. Licenses nothing; its one permitted effect is the format line.
  python run_realmodel_bridge_v3.py
      the main run. Refuses without: a committed PREREG whose format line names this readout, a smoke
      receipt written by this readout with both arms timed and a budget inside 4 hours, a pilot2 receipt
      that cleared both arms, every tokenizer-level red-team assert, the model-level asserts of BOTH arms
      before any cell is scored, and >= 6 GB of free GPU memory before each load. One arm resident at a
      time. NOT run by the builder: it needs the seal and the operator's anchor.

The pure parts (items, balance, fillers, metrics, bootstrap, budget) import only numpy, so the unit
tests run on CPU with no model; torch and transformers are imported inside the functions that touch one.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREREG = HERE / "PREREG_realmodel_bridge_v3_2026_10_11.md"
RECEIPT = HERE / "realmodel_bridge_v3_result.json"
SMOKE_RECEIPT = HERE / "realmodel_bridge_v3_smoke.json"
PILOT1_RECEIPT = HERE / "realmodel_bridge_v3_pilot.json"      # the section-9 pilot (seed 1011); read, never written
PILOT2_RECEIPT = HERE / "realmodel_bridge_v3_pilot2.json"     # the amendment-17.2 pilot (seed 1012)
REFUSAL_RECEIPT = HERE / "realmodel_bridge_v3_refusal.json"

ARMS = {"M1": "state-spaces/mamba-1.4b-hf", "T": "EleutherAI/pythia-1.4b"}
NAMES = ("alpha", "gamma", "delta")           # test names: each " "+name is one NeoX token (" bravo" is two; amendment 17.2)
EXEMPLAR_NAMES = ("sigma", "theta", "omega")  # exemplar names, disjoint from the test names (amendment 17.2, zoo F1 fix a)
READOUT = "argmax_largest"                    # the amendment-17.2 compare readout; the only one this runner scores
RETIRED_FORMATS = ("yes_no", "true_false")    # the section-9 formats: both failed the 1011 pilot (amendment 17.1)
FAMILIES = ("compare", "recall")
VARIANTS = ("plain", "ablated", "windowed", "near", "overwrite")
DISTANCES = (16, 64, 256, 1024)
D0, D1 = 16, 1024
N_PER_CELL = 600
N_SMOKE, N_PILOT = 24, 48
SMOKE_DISTANCES = (16, 1024)
SEED_ITEMS, SEED_PILOT, SEED_PILOT2, SEED_BOOT = 20261011, 1011, 1012, 20261011
SEED_SMOKE = 1016                         # the smoke's own items: never a main-set or pilot item
N_BOOT = 2000
SCREEN_BAR = 0.40
GUARD_SLOT_MAX = 0.80
ABLATED_BAR = 0.16
WINDOW_W = 128
MAX_PROMPT_TOKENS = 2000
FREE_GPU_MIN_BYTES = 6 * 1024 ** 3
BUDGET_HOURS = 4.0
BATCH_MAX = 16
SCAN_CHUNK_TOKENS = 1024        # M1: the slow path's four [B, 4096, L, 16] fp32 tensors are materialised per chunk of L with
                                # B * chunk <= this (about 1 GiB); the arithmetic per element is the reference's, asserted
CHANCE = {"compare": 1.0 / 3.0, "recall": 1.0 / 3.0}
N_EXEMPLARS = 4
EXEMPLAR_Q_SLOTS = (0, 1, 2, 0)           # PREREG section 4: recall exemplars query slots 1, 2, 3 and 1
EXEMPLAR_MAX_SLOTS = (1, 0, 2, 1)         # amendment 17.2: the slot of the largest code in each compare exemplar
CODE_MAX_PLANTED = 8999                   # amendment 17.2: planted codes <= 8999 so that a larger code exists for C_overwrite
POOL_CAP = 150_000                        # eligible sentences read from the train split, in file order
PILOT_RESERVE = 4000                      # tail of the seeded sentence order, reserved for the 1011 pilot
SMOKE_RESERVE = 3000                      # the slice before that tail, reserved for the smoke
PILOT2_RESERVE = 3000                     # the slice before the smoke's, reserved for the 1012 pilot; main = everything before all three
N_EXEMPLAR_CODES = N_EXEMPLARS * 3
N_PILOT_TEST_CODES = 60
N_SMOKE_TEST_CODES = 60
N_PILOT2_TEST_CODES = 60
MARGIN_TIE = 1e-3
FP32_TOL = 1e-3
PAD_ID = 0                                # never read: right padding, logits taken at each item's own last position
FORMAT_LINE_RE = re.compile(r"^Answer format chosen \(written before sealing\): (\S+)\s*$", re.M)
FORMAT_LINE_VALUES = RETIRED_FORMATS + ("PENDING_PILOT", "NOT_STARTED", READOUT)
PILOT_OUTCOMES = ("CLEARED_BOTH", "CLEARED_T_ONLY", "CLEARED_M1_ONLY", "NEITHER")


class Refusal(RuntimeError):
    """The runner stops before any data. The message names the assert that failed."""


# ---------------------------------------------------------------------------------------------------
# tokenizer adapter and local snapshots
# ---------------------------------------------------------------------------------------------------

class Tok:
    """The one tokenizer operation the design needs: text -> ids, no special tokens."""

    def __init__(self, hf_tok, name: str = ""):
        self.hf = hf_tok
        self.name = name

    def encode(self, text: str) -> list:
        return list(self.hf.encode(text, add_special_tokens=False))

    def __len__(self):
        return len(self.hf)


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_dir(repo_id: str, kind: str = "models") -> Path | None:
    """The cached snapshot directory of a hub repo, or None. refs/main decides; else the only snapshot."""
    base = hf_cache_dir() / f"{kind}--{repo_id.replace('/', '--')}"
    snaps = base / "snapshots"
    if not snaps.is_dir():
        return None
    ref = base / "refs" / "main"
    if ref.exists():
        cand = snaps / ref.read_text(encoding="utf-8").strip()
        if cand.is_dir():
            return cand
    found = sorted(p for p in snaps.iterdir() if p.is_dir())
    return found[0] if len(found) == 1 else None


def load_tokenizer(repo_id: str) -> Tok:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    path = snapshot_dir(repo_id)
    src = str(path) if path is not None else repo_id
    return Tok(AutoTokenizer.from_pretrained(src, local_files_only=True), name=repo_id)


# ---------------------------------------------------------------------------------------------------
# the sentence pool (PREREG section 4, [J1])
# ---------------------------------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SENT_OK = re.compile(r"^[A-Za-z][A-Za-z ,;:'\"()\-]*[.!?]$")
# _SENT_BAD defines the pool itself and is kept exactly as the committed smoke and 1011-pilot receipts had it, so
# that their sentence indices still name the same sentences (pool identity). The names of amendment 17.2 are
# excluded at draw time instead (SentencePool.draw skips them, consuming nothing a filler could use).
_SENT_BAD = re.compile(r"\d|alpha|bravo|delta|code|<unk>", re.I)
_FILLER_BAD = re.compile(r"\d|" + "|".join(NAMES + EXEMPLAR_NAMES) + r"|code", re.I)
MIN_WORDS, MAX_WORDS = 6, 40


def normalize_wikitext_line(line: str) -> str:
    s = line.strip()
    if not s or s.startswith("="):
        return ""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?)'])", r"\1", s).replace("( ", "(")
    return re.sub(r"\s+", " ", s)


def eligible_sentence(s: str) -> bool:
    w = s.count(" ") + 1
    return MIN_WORDS <= w <= MAX_WORDS and bool(_SENT_OK.match(s)) and not _SENT_BAD.search(s)


def sentences_from_lines(lines, cap: int | None = None) -> list:
    out = []
    for line in lines:
        s = normalize_wikitext_line(line)
        if not s:
            continue
        for sent in _SENT_SPLIT.split(s):
            if eligible_sentence(sent):
                out.append(sent)
                if cap is not None and len(out) >= cap:
                    return out
    return out


def wikitext_train_parquets() -> list:
    snap = snapshot_dir("wikitext", kind="datasets")
    if snap is None:
        raise Refusal("wikitext-103-raw-v1 is not in the hub cache (datasets--wikitext); the filler pool "
                      "needs its train split and this runner downloads nothing")
    files = sorted((snap / "wikitext-103-raw-v1").glob("train-*.parquet"))
    if not files:
        raise Refusal("wikitext-103-raw-v1 train parquet files not found in the cached snapshot")
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_wikitext_sentences(cap: int = POOL_CAP) -> tuple:
    """Eligible sentences from the cached train split, in file order, up to cap. Returns (sentences, provenance)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:                                           # pragma: no cover
        raise Refusal(f"pyarrow is needed to read the cached wikitext parquet files: {e}")
    files = wikitext_train_parquets()

    def lines():
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=20000, columns=["text"]):
                yield from batch.column(0).to_pylist()

    out = sentences_from_lines(lines(), cap=cap)
    prov = {"source": "wikitext-103-raw-v1 train split, hub parquet cache",
            "files": [str(fp.name) for fp in files], "sha256": [sha256_file(fp) for fp in files],
            "pool_cap": cap, "eligible_sentences": len(out)}
    return out, prov


def split_orders(n: int, seed_main: int = SEED_ITEMS, seed_pilot: int = SEED_PILOT, seed_smoke: int = SEED_SMOKE,
                 seed_pilot2: int = SEED_PILOT2, reserve: int = PILOT_RESERVE, smoke_reserve: int = SMOKE_RESERVE,
                 pilot2_reserve: int = PILOT2_RESERVE) -> dict:
    """One seeded permutation of the sentence indices, cut in four: pilot = the tail (reshuffled by the pilot seed);
    smoke = the slice before it (reshuffled by the smoke seed); pilot2 = the slice before that (reshuffled by the
    pilot2 seed); main = everything before the three reserved slices. The pilot and smoke slices are exactly the
    ones the committed 1011-pilot and smoke receipts drew from; the pilot2 slice is carved from what was the main
    region's tail, before any main-set item exists. Disjoint by construction. Returns {main, pilot2, smoke, pilot}."""
    total = reserve + smoke_reserve + pilot2_reserve
    if n <= total + 1:
        raise Refusal(f"sentence pool too small ({n}) for reserves of {reserve} + {smoke_reserve} + {pilot2_reserve}")
    perm = np.random.default_rng(seed_main).permutation(n)
    main = perm[:-total]
    pilot2_slice = perm[-total:-(reserve + smoke_reserve)]
    smoke_slice = perm[-(reserve + smoke_reserve):-reserve]
    tail = perm[-reserve:]
    return {"main": main,
            "pilot2": pilot2_slice[np.random.default_rng(seed_pilot2).permutation(pilot2_reserve)],
            "smoke": smoke_slice[np.random.default_rng(seed_smoke).permutation(smoke_reserve)],
            "pilot": tail[np.random.default_rng(seed_pilot).permutation(reserve)]}


class SentencePool:
    """Eligible sentences drawn without replacement in a fixed order; every draw is recorded. A sentence
    containing a test name, an exemplar name, a digit or 'code' is skipped at draw time (recorded in `skipped`)."""

    def __init__(self, sentences: list, order, bad=_FILLER_BAD):
        self.sentences = sentences
        self.order = [int(i) for i in order]
        self.bad = bad
        self.cursor = 0
        self.used = []
        self.skipped = []

    def draw(self) -> int:
        while True:
            if self.cursor >= len(self.order):
                raise Refusal("sentence pool exhausted — raise POOL_CAP")
            i = self.order[self.cursor]
            self.cursor += 1
            if self.bad.search(self.sentences[i]):
                self.skipped.append(i)
                continue
            self.used.append(i)
            return i

    def fill(self, target: int, measure, max_tries: int = 64) -> tuple:
        """Distinct sentences joined by single spaces, cut at a word boundary so that measure(text) == target
        exactly. Every sentence drawn is consumed, used or not. Returns (text, [sentence indices], [parts])."""
        parts, used, tries = [], [], 0
        while True:
            i = self.draw()
            used.append(i)
            s = self.sentences[i]
            n = measure(" ".join(parts + [s]))
            if n == target:
                parts.append(s)
                break
            if n < target:
                parts.append(s)
                continue
            words = s.split(" ")
            hit = None
            for k in range(1, len(words)):
                pre = " ".join(words[:k])
                m = measure(" ".join(parts + [pre]))
                if m == target:
                    hit = pre
                    break
                if m > target:
                    break
            if hit is not None:
                parts.append(hit)
                break
            tries += 1
            if tries > max_tries:
                raise Refusal(f"could not cut a filler to exactly {target} tokens after {max_tries} sentences")
        text = " ".join(parts)
        if measure(text) != target:
            raise Refusal("filler measure drifted after joining — tokenizer not compositional on this text")
        return text, used, parts


def filler_violations(text: str) -> list:
    """Digits, the six names and the word 'code' (as substrings, case-insensitive) are never in a filler."""
    return sorted(set(m.group(0).lower() for m in _FILLER_BAD.finditer(text)))


# ---------------------------------------------------------------------------------------------------
# codes, exemplars, items (PREREG section 4 as amended by 17.2)
# ---------------------------------------------------------------------------------------------------

def code_token_lengths(tok: Tok) -> dict:
    return {str(c): len(tok.encode(" %d" % c)) for c in range(1000, 10000)}


def eligible_codes(tok: Tok) -> tuple:
    """Four-digit codes whose ' '+code tokenizes to the modal length (two tokens in NeoX). Returns (codes, length)."""
    lens = code_token_lengths(tok)
    modal = max(set(lens.values()), key=lambda L: sum(1 for v in lens.values() if v == L))
    return [c for c, L in lens.items() if L == modal], modal


def partition_codes(codes: list, seed: int = SEED_ITEMS) -> dict:
    """Disjoint pools. The slices the committed receipts used (main exemplars, pilot exemplars and test, smoke
    exemplars and test) are unchanged; the pilot2 pools are carved from what was the main test pool's head."""
    rng = np.random.default_rng(seed)
    order = [codes[i] for i in rng.permutation(len(codes))]
    a = N_EXEMPLAR_CODES
    b = a + N_EXEMPLAR_CODES
    c = b + N_PILOT_TEST_CODES
    d = c + N_EXEMPLAR_CODES
    e = d + N_SMOKE_TEST_CODES
    f = e + N_EXEMPLAR_CODES
    g = f + N_PILOT2_TEST_CODES
    if len(order) < g + 3 * 10:
        raise Refusal(f"code pool too small ({len(order)})")
    return {"main_exemplar": order[:a], "pilot_exemplar": order[a:b], "pilot_test": order[b:c],
            "smoke_exemplar": order[c:d], "smoke_test": order[d:e],
            "pilot2_exemplar": order[e:f], "pilot2_test": order[f:g], "main_test": order[g:]}


def code_block(codes, names=NAMES) -> str:
    return " ".join(f"The {names[j]} code is {codes[j]}." for j in range(3))


def question_text(family: str, q_slot: int, names=NAMES) -> str:
    if family == "compare":
        return f"Question: Which name has the largest code: {names[0]}, {names[1]} or {names[2]}? Answer:"
    return f"Question: What is the {names[q_slot]} code? Answer:"


def replant_sentence(slot: int, code: str, names=NAMES) -> str:
    return f"The {names[slot]} code is now {code}."


def arrange_codes(three: list, max_slot: int, rng) -> list:
    """The numerically largest of three distinct codes at max_slot, the other two in seeded order."""
    srt = sorted(three, key=int)
    rest = [srt[0], srt[1]]
    if int(rng.integers(0, 2)):
        rest = rest[::-1]
    out = [None, None, None]
    out[max_slot] = srt[2]
    for s in range(3):
        if s != max_slot:
            out[s] = rest.pop(0)
    return out


def build_exemplars(exemplar_codes: list, rng) -> list:
    """Four blocks with the exemplar names, zero filler: recall queries slots (1, 2, 3, 1); the largest code sits at
    slots (2, 1, 3, 2), so the queried code is the largest in one exemplar of four and no slot is the answer twice
    in a row by construction."""
    if len(exemplar_codes) < N_EXEMPLAR_CODES:
        raise Refusal("not enough exemplar codes")
    exs = []
    for j in range(N_EXEMPLARS):
        codes = arrange_codes(list(exemplar_codes[3 * j: 3 * j + 3]), EXEMPLAR_MAX_SLOTS[j], rng)
        exs.append({"codes": codes, "q_slot": EXEMPLAR_Q_SLOTS[j], "max_slot": EXEMPLAR_MAX_SLOTS[j]})
    return exs


def exemplar_text(ex: dict, family: str) -> str:
    cb = code_block(ex["codes"], EXEMPLAR_NAMES)
    q = question_text(family, ex["q_slot"], EXEMPLAR_NAMES)
    if family == "compare":
        return f"{cb}\n{q} {EXEMPLAR_NAMES[ex['max_slot']]}"
    return f"{cb}\n{q} {ex['codes'][ex['q_slot']]}"


_PERMS = list(itertools.permutations(range(N_EXEMPLARS)))
_TRANSPOSITIONS = ((1, 0, 2), (2, 1, 0), (0, 2, 1))
_PERMS3 = list(itertools.permutations(range(3)))
# the last exemplar's compare answer slot decides which exemplar is last: slot 0 -> exemplar 2, slot 2 -> exemplar 3,
# slot 1 -> exemplar 1 or 4 (alternating); its recall query slot then follows EXEMPLAR_Q_SLOTS
_LAST_EX_BY_COMPARE_SLOT = {0: (1,), 2: (2,), 1: (0, 3)}
_RECALL_SLOT_BY_COMPARE_SLOT = {s: EXEMPLAR_Q_SLOTS[exs[0]] for s, exs in _LAST_EX_BY_COMPARE_SLOT.items()}


def permutation_with_last(rng, last_ex: int) -> list:
    ok = [p for p in _PERMS if p[-1] == last_ex]
    return list(ok[int(rng.integers(0, len(ok)))])


def design_rows(n: int, rng) -> list:
    """Exactly balanced rows (amendment 17.2). n must be a positive multiple of 6. Per cell, exactly n/3 items have
    each slot as the largest code's slot, as the recall-queried slot, as the last exemplar's compare answer slot and
    as the last exemplar's recall query slot; the last exemplar's compare answer slot equals the correct slot on
    exactly n/3 items and its recall query slot equals the queried slot on exactly n/3 items; the overwrite slot
    is never the correct slot and each (correct, overwrite) pair occurs n/6 times. Built in blocks of three:
    last = pi(j), correct = pi(tau(j)), queried = f(pi(tau'(j))) with tau, tau' transpositions and f the map from
    the last exemplar's compare slot to its recall slot; tau' = tau on about one block in three, so the count of
    items whose queried slot is the correct slot is near n/3 (reported, not exact: 600 is not a multiple of 9)."""
    if n <= 0 or n % 6 != 0:
        raise Refusal(f"n per cell must be a positive multiple of 6 for exact balance, got {n}")
    blocks = n // 3
    same = int(round(blocks / 3.0))
    rows = []
    for b in range(blocks):
        pi = _PERMS3[int(rng.integers(0, len(_PERMS3)))]
        ti = int(rng.integers(0, 3))
        tau = _TRANSPOSITIONS[ti]
        tau2 = tau if b < same else _TRANSPOSITIONS[(ti + 1 + int(rng.integers(0, 2))) % 3]
        for j in range(3):
            last_c = pi[j]
            exs = _LAST_EX_BY_COMPARE_SLOT[last_c]
            last_ex = exs[0] if len(exs) == 1 else exs[b % 2]
            rows.append({"max_slot": pi[tau[j]], "q_slot": _RECALL_SLOT_BY_COMPARE_SLOT[pi[tau2[j]]],
                         "last_ex": last_ex})
    counters = {m: 0 for m in range(3)}
    for r in rows:
        m = r["max_slot"]
        r["ow_slot"] = (m + 1 + counters[m] % 2) % 3
        counters[m] += 1
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def in_situ_len(tok: Tok, text: str) -> int:
    """Token count of text standing where the code block stands: after a blank line, no leading space."""
    return len(tok.encode("x\n\n" + text)) - len(tok.encode("x\n\n"))


def prompt_parts(item: dict, exemplars: list, family: str, variant: str = "plain") -> tuple:
    """(a, b, c): exemplars + blank line + head | ' ' + filler | newline + question.
    plain / windowed: head = the code block.  ablated: head = the neutral text (C_abl).
    near (D1 only): head = the item's own filler, a blank line, then the code block; b = a 16-token filler, so the
    codes stand 16 tokens before the question while the exemplars stand where they do at D1 (C_near).
    overwrite (D1 only): head = the code block; b = the item's filler with its tail replaced so that its last
    sentence re-plants the overwrite slot's name with a larger code (C_overwrite)."""
    exs = "\n\n".join(exemplar_text(exemplars[k], family) for k in item["perm"])
    cb = code_block(item["codes"])
    c = "\n" + question_text(family, item["q_slot"])
    if variant in ("plain", "windowed"):
        return exs + "\n\n" + cb, " " + item["filler"], c
    if variant == "ablated":
        return exs + "\n\n" + item["neutral"], " " + item["filler"], c
    if variant == "near":
        return exs + "\n\n" + item["filler"] + "\n\n" + cb, " " + item["near_filler"], c
    if variant == "overwrite":
        return exs + "\n\n" + cb, " " + item["ow_filler"], c
    raise Refusal(f"unknown prompt variant {variant!r}")


def prompt_text(item: dict, exemplars: list, family: str, variant: str = "plain") -> str:
    return "".join(prompt_parts(item, exemplars, family, variant))


def item_variants(item: dict) -> tuple:
    return ("plain", "ablated", "near", "overwrite") if item["D"] == D1 and "ow_filler" in item else ("plain", "ablated")


def compositional_ids(tok: Tok, parts: tuple) -> list | None:
    """The whole prompt's ids, if they equal the concatenation of the parts' ids; else None."""
    whole = tok.encode("".join(parts))
    joined = sum((tok.encode(p) for p in parts), [])
    return whole if whole == joined else None


def item_is_exact(item: dict, exemplars: list, tok: Tok, D: int) -> bool:
    """D-exactness and compositionality for both families, every variant the item carries."""
    for family in FAMILIES:
        parts = prompt_parts(item, exemplars, family)
        ids = compositional_ids(tok, parts)
        if ids is None or len(tok.encode(parts[1])) != D:
            return False
        abl = tok.encode(prompt_text(item, exemplars, family, "ablated"))
        tail = tok.encode(parts[1]) + tok.encode(parts[2])
        if len(abl) != len(ids) or abl[-len(tail):] != tail:
            return False
        if "ow_filler" in item:
            near = prompt_parts(item, exemplars, family, "near")
            if compositional_ids(tok, near) is None or len(tok.encode(near[1])) != D0:
                return False
            ow = prompt_parts(item, exemplars, family, "overwrite")
            if compositional_ids(tok, ow) is None or len(tok.encode(ow[1])) != D:
                return False
    return True


def overwrite_filler(parts: list, replant: str, D: int, pool: SentencePool, measure) -> tuple:
    """The item's own filler sentences with the tail dropped until the re-plant sentence fits as the last sentence
    inside exactly D tokens; a fresh fill from the pool closes any gap before the re-plant. Returns
    (text, head_text, [sentence indices drawn for the gap])."""
    parts = list(parts)
    while parts and measure(" ".join(parts + [replant])) > D:
        parts.pop()
    gap = D - measure(" ".join(parts + [replant]))
    used = []
    if gap > 0:
        top, used, _ = pool.fill(gap, measure)
        parts = parts + [top]
    head = " ".join(parts)
    text = " ".join(parts + [replant])
    if measure(text) != D:
        raise Refusal(f"overwrite filler is {measure(text)} tokens, not {D}")
    return text, head, used


def build_items(n: int, D: int, seed: int, pool: SentencePool, test_codes: list, exemplars: list,
                tok: Tok, max_attempts: int = 8) -> list:
    """n exactly balanced items at distance D. Fillers and neutral texts come from the pool without replacement.
    At D1 every item also carries the C_near filler and the C_overwrite filler and code."""
    rng = np.random.default_rng(seed + D)
    rows = design_rows(n, rng)
    plantable = np.asarray([c for c in test_codes if int(c) <= CODE_MAX_PLANTED], dtype=object)
    if len(plantable) < 3:
        raise Refusal("fewer than three plantable codes (<= %d) in the test pool" % CODE_MAX_PLANTED)
    items = []

    def filler_measure(t: str) -> int:
        return len(tok.encode(" " + t))

    def neutral_measure(t: str) -> int:
        return in_situ_len(tok, t)

    for k, row in enumerate(rows):
        three = [str(c) for c in rng.choice(plantable, size=3, replace=False)]
        codes = arrange_codes(three, row["max_slot"], rng)
        perm = permutation_with_last(rng, row["last_ex"])
        target_neutral = in_situ_len(tok, code_block(codes))
        ow_code = None
        if D == D1:
            larger = [c for c in test_codes if int(c) > int(codes[row["max_slot"]])]
            if not larger:
                raise Refusal(f"item {k} at D={D}: no code in the test pool is larger than {codes[row['max_slot']]} (C_overwrite)")
            ow_code = str(larger[int(rng.integers(0, len(larger)))])
        for _attempt in range(max_attempts):
            filler, f_used, f_parts = pool.fill(D, filler_measure)
            neutral, n_used, _ = pool.fill(target_neutral, neutral_measure)
            item = {"idx": k, "D": D, "codes": codes, "q_slot": row["q_slot"], "max_slot": row["max_slot"],
                    "last_ex": row["last_ex"], "perm": perm, "ow_slot": row["ow_slot"],
                    "filler": filler, "filler_sentences": f_used, "neutral": neutral, "neutral_sentences": n_used}
            if D == D1:
                near, near_used, _ = pool.fill(D0, filler_measure)
                replant = replant_sentence(row["ow_slot"], ow_code)
                ow_text, ow_head, ow_used = overwrite_filler(f_parts, replant, D, pool, filler_measure)
                item.update({"near_filler": near, "near_sentences": near_used, "ow_code": ow_code,
                             "ow_filler": ow_text, "ow_filler_head": ow_head, "ow_replant": replant,
                             "ow_sentences": ow_used})
            if item_is_exact(item, exemplars, tok, D):
                break
        else:
            raise Refusal(f"item {k} at D={D}: no exact filler in {max_attempts} attempts")
        items.append(item)
    return items


# ---------------------------------------------------------------------------------------------------
# red-team asserts on the items (PREREG section 12, asserts 2-5 and 7, as amended by 17.2)
# ---------------------------------------------------------------------------------------------------

def balance_counts(items: list, exemplars: list) -> dict:
    """Counts computed from the items themselves (the last exemplar's slots are read off the permutation)."""
    n = len(items)
    last_c = [exemplars[it["perm"][-1]]["max_slot"] for it in items]
    last_r = [exemplars[it["perm"][-1]]["q_slot"] for it in items]
    pairs = {f"{m}{y}": 0 for m in range(3) for y in range(3) if y != m}
    for it in items:
        key = f"{it['max_slot']}{it['ow_slot']}"
        if key in pairs:
            pairs[key] += 1
    c = {"n": n,
         "max_slot": [sum(1 for it in items if it["max_slot"] == s) for s in range(3)],
         "q_slot": [sum(1 for it in items if it["q_slot"] == s) for s in range(3)],
         "last_exemplar": [sum(1 for it in items if it["perm"][-1] == j) for j in range(N_EXEMPLARS)],
         "last_exemplar_compare_slot": [sum(1 for s in last_c if s == k) for k in range(3)],
         "last_exemplar_recall_slot": [sum(1 for s in last_r if s == k) for k in range(3)],
         "last_compare_slot_is_correct": sum(1 for it, s in zip(items, last_c) if s == it["max_slot"]),
         "last_recall_slot_is_queried": sum(1 for it, s in zip(items, last_r) if s == it["q_slot"]),
         "queried_is_largest": sum(1 for it in items if it["q_slot"] == it["max_slot"]),
         "ow_slot": [sum(1 for it in items if it["ow_slot"] == s) for s in range(3)],
         "ow_pairs": pairs,
         "ow_never_correct": all(it["ow_slot"] != it["max_slot"] for it in items),
         "largest_at_max_slot": all(max(it["codes"], key=int) == it["codes"][it["max_slot"]] for it in items),
         "codes_distinct": all(len(set(it["codes"])) == 3 for it in items),
         "codes_plantable": all(int(c) <= CODE_MAX_PLANTED for it in items for c in it["codes"]),
         "perms_valid": all(sorted(it["perm"]) == list(range(N_EXEMPLARS)) for it in items)}
    if items and "ow_code" in items[0]:
        c["ow_code_larger_than_all"] = all(int(it["ow_code"]) > max(int(x) for x in it["codes"]) for it in items)
        c["ow_replant_names_ow_slot"] = all(it["ow_replant"] == replant_sentence(it["ow_slot"], it["ow_code"]) for it in items)
        c["ow_filler_ends_with_replant"] = all(it["ow_filler"] == (it["ow_filler_head"] + " " + it["ow_replant"]).strip() for it in items)
    return c


def expected_balance(n: int, with_overwrite: bool = False) -> dict:
    third, sixth = n // 3, n // 6
    want = {"max_slot": [third] * 3, "q_slot": [third] * 3, "last_exemplar": [sixth, third, third, sixth],
            "last_exemplar_compare_slot": [third] * 3, "last_exemplar_recall_slot": [third] * 3,
            "last_compare_slot_is_correct": third, "last_recall_slot_is_queried": third,
            "ow_slot": [third] * 3, "ow_pairs": {f"{m}{y}": sixth for m in range(3) for y in range(3) if y != m},
            "ow_never_correct": True, "largest_at_max_slot": True, "codes_distinct": True, "codes_plantable": True,
            "perms_valid": True}
    if with_overwrite:
        want.update({"ow_code_larger_than_all": True, "ow_replant_names_ow_slot": True, "ow_filler_ends_with_replant": True})
    return want


def assert_balance(items: list, exemplars: list) -> dict:
    got = balance_counts(items, exemplars)
    want = expected_balance(len(items), with_overwrite=bool(items) and "ow_code" in items[0])
    bad = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if bad:
        raise Refusal(f"exact balance failed: {bad}")
    return got


_SENTENCE_KEYS = ("filler_sentences", "neutral_sentences", "near_sentences", "ow_sentences")
_TEXT_KEYS = ("filler", "neutral", "near_filler", "ow_filler_head")


def assert_fillers(cells: dict, exemplars: list, prior_used: set | None = None) -> dict:
    """No sentence reused across items (nor one an earlier receipt records); no digit, name or 'code' in any
    filler text (the re-plant sentence of C_overwrite is excluded from the check by construction: it is not filler).
    prior_used: sentence indices the smoke and the pilots used."""
    seen, dup, bad = {}, [], []
    for D, items in cells.items():
        for it in items:
            for key in _SENTENCE_KEYS:
                for i in it.get(key, ()):
                    if i in seen:
                        dup.append((i, seen[i], (D, it["idx"], key)))
                    seen[i] = (D, it["idx"], key)
                    if prior_used and i in prior_used:
                        dup.append((i, "smoke/pilot", (D, it["idx"], key)))
            for key in _TEXT_KEYS:
                if key in it:
                    v = filler_violations(it[key])
                    if v:
                        bad.append((D, it["idx"], key, v))
    if dup:
        raise Refusal(f"filler sentence reused: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    if bad:
        raise Refusal(f"filler contains a digit, a name or 'code': {bad[:5]}")
    return {"sentences_used": len(seen), "used_sha256": hashlib.sha256(
        json.dumps(sorted(seen)).encode("utf-8")).hexdigest()}


def assert_prompts(cells: dict, exemplars: list, toks: dict, codes_len: dict) -> dict:
    """Identical ids across tokenizers on every prompt of every variant, <= 2000 tokens, equal-length codes."""
    n_prompts, max_len = 0, 0
    names = list(toks)
    for D, items in cells.items():
        for it in items:
            for family in FAMILIES:
                for variant in item_variants(it):
                    text = prompt_text(it, exemplars, family, variant)
                    ids = {nm: toks[nm].encode(text) for nm in names}
                    ref = ids[names[0]]
                    for nm in names[1:]:
                        if ids[nm] != ref:
                            raise Refusal(f"token ids differ between {names[0]} and {nm} on a prompt at D={D}, item {it['idx']}")
                    n_prompts += 1
                    max_len = max(max_len, len(ref) + codes_len[names[0]])   # + the candidate appended for recall
            for c in it["codes"] + ([it["ow_code"]] if "ow_code" in it else []):
                for nm in names:
                    if len(toks[nm].encode(" " + c)) != codes_len[nm]:
                        raise Refusal(f"candidate code {c} has a different token length in {nm}")
    if max_len > MAX_PROMPT_TOKENS:
        raise Refusal(f"a prompt reaches {max_len} tokens, above the {MAX_PROMPT_TOKENS} bound")
    return {"prompts_checked": n_prompts, "max_prompt_tokens": max_len, "tokenizers": names}


def name_token_ids(tok: Tok, names=NAMES) -> dict:
    """' '+name must be one token for every name (the three-way answer set is scored at one position)."""
    out = {}
    for nm in names:
        ids = tok.encode(" " + nm)
        if len(ids) != 1:
            raise Refusal(f"answer name {' ' + nm!r} is not a single token in {tok.name}: {ids}")
        out[nm] = ids[0]
    if len(set(out.values())) != len(names):
        raise Refusal("answer names share a token")
    return out


def assert_replant_lengths(tok: Tok, codes: list) -> int:
    """The re-plant sentence has one token length for every slot and every code of the eligible pool it is given
    (names are single tokens, pool codes equal length)."""
    sample = sorted(set(codes), key=int)
    sample = sample[::max(1, len(sample) // 40)] + sample[-3:]
    lens = {len(tok.encode(" " + replant_sentence(s, c))) for s in range(3) for c in sample}
    if len(lens) != 1:
        raise Refusal(f"the re-plant sentence has several token lengths in {tok.name}: {sorted(lens)}")
    return lens.pop()


def answer_format_from_prereg(text: str) -> str:
    """The last 'Answer format chosen' line wins: section 9 freezes PENDING_PILOT; amendment 17.1 set NOT_STARTED
    when both section-9 formats failed the 1011 pilot; amendment 17.3 sets the redesigned readout if the 1012 pilot
    clears both arms, else NOT_STARTED (final)."""
    found = FORMAT_LINE_RE.findall(text)
    if not found:
        raise Refusal("the PREREG has no 'Answer format chosen (written before sealing):' line")
    fmt = found[-1]
    if fmt not in FORMAT_LINE_VALUES:
        raise Refusal(f"the PREREG's answer format line reads {fmt!r}, not one of {FORMAT_LINE_VALUES}")
    return fmt


# ---------------------------------------------------------------------------------------------------
# metrics (PREREG sections 5-7, as amended by 17.2: both families argmax over three slots, chance 1/3)
# ---------------------------------------------------------------------------------------------------

def norm_acc(raw: float, chance: float) -> float:
    return (raw - chance) / (1.0 - chance)


def auroc(scores, labels) -> float | None:
    """Rank AUROC with average ranks on ties; 0.5 for any constant score; None if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ovr_auroc(scores3, targets) -> float | None:
    """One-versus-rest AUROC averaged over the three slots: the slot's score against 'this slot is correct'.
    0.5 for any constant responder (identical score vectors on every item)."""
    s = np.asarray(scores3, dtype=float)
    t = np.asarray(targets, dtype=int)
    vals = [auroc(s[:, k], t == k) for k in range(3)]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def argmax_cell_summary(scores3, targets, ow_slots=None) -> dict:
    """One cell of either family: scores3 [n, 3] (compare: log p of each name token at the answer position; recall:
    summed log-likelihood of each candidate code), targets [n] the correct slot. Prediction = argmax (an exact tie
    resolves to the lowest slot index and is counted). margin = score of the correct slot minus the best other
    slot: the three-way analogue of the Yes/No log-odds, stored per item."""
    s = np.asarray(scores3, dtype=float)
    t = np.asarray(targets, dtype=int)
    if s.ndim != 2 or s.shape[1] != 3 or len(t) != len(s):
        raise Refusal(f"cell scores have shape {s.shape} for {len(t)} targets")
    pred = s.argmax(axis=1)
    correct = pred == t
    raw = float(correct.mean())
    fr = [float((pred == k).mean()) for k in range(3)]
    own = s[np.arange(len(t)), t]
    others = s.copy()
    others[np.arange(len(t)), t] = -np.inf
    margin = own - others.max(axis=1)
    out = {"n": int(len(t)), "acc_raw": raw, "acc_norm": norm_acc(raw, 1.0 / 3.0),
           "slot_fractions": fr, "max_slot_fraction": max(fr), "guard_ok": bool(max(fr) <= GUARD_SLOT_MAX),
           "margin_mean": float(margin.mean()), "margin_median": float(np.median(margin)),
           "near_ties": int((np.abs(margin) < MARGIN_TIE).sum()), "ovr_auroc": ovr_auroc(s, t),
           "_per_item": {"correct": [int(x) for x in correct], "pred_slot": [int(x) for x in pred],
                         "target_slot": [int(x) for x in t], "scores": [[float(v) for v in row] for row in s],
                         "margin": [float(x) for x in margin]},
           "_correct": correct}
    if ow_slots is not None:
        o = np.asarray(ow_slots, dtype=int)
        out["overwrite_follow_rate"] = float((pred == o).mean())
        out["_per_item"]["pred_is_overwrite"] = [int(x) for x in (pred == o)]
    return out


def paired_bootstrap(c_T, c_M1, r_T, r_M1, n_resamples: int = N_BOOT, seed: int = SEED_BOOT) -> dict:
    """between_D1 as a mean of per-item contributions on the normalized scale, resampled over the shared items."""
    w_c = 1.0 / (1.0 - CHANCE["compare"])
    w_r = 1.0 / (1.0 - CHANCE["recall"])
    s = w_c * (np.asarray(c_T, float) - np.asarray(c_M1, float)) - w_r * (np.asarray(r_T, float) - np.asarray(r_M1, float))
    n = len(s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = s[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point": float(s.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "resamples": int(n_resamples),
            "seed": int(seed), "n_items": int(n), "method": "paired item bootstrap, percentile 2.5/97.5",
            "weights": {"compare": w_c, "recall": w_r}}


def walk_outcomes(spec: dict, fired: dict) -> str | None:
    """The scorer's rule, reproduced for enumeration: the earliest row whose `when` matches wins."""
    for row in spec["outcomes"]:
        if all(fired.get(k) == v for k, v in row["when"].items()):
            return row["verdict"]
    return None


# ---------------------------------------------------------------------------------------------------
# budget (PREREG section 11)
# ---------------------------------------------------------------------------------------------------

def batch_size_for(arm: str, L: int) -> int:
    """Batches of 16 for both arms (PREREG section 4, Batching). M1's slow path fits at 16 because the scan's
    [B, 4096, L, 16] tensors are materialised per chunk of L (install_chunked_scan), not for the whole sequence."""
    return BATCH_MAX


def extrapolate_budget(timings: dict, n: int = N_PER_CELL, distances=DISTANCES, budget_hours: float = BUDGET_HOURS) -> dict:
    """From per-batch seconds measured at the smoke's two distances, the main run's total.

    timings: {arm: {D: {"L": max seq len, "s_per_batch": {"compare": x, "recall": y}}}} for the timed arms.
    Seconds per batch at an unmeasured distance are linear in L between the two measured points. Passes per
    cell: compare 1 + recall 3, for the plain and the ablated variant at every distance (amendment 17.2: C_abl
    everywhere), for T the windowed control at D0 and D1, and at D1 the two compare-only controls C_near and
    C_overwrite for both arms. Both arms must be timed (amendment 17.2, review minor c): an untimed arm makes
    the verdict OK_PARTIAL and the main run refuses on it.
    """
    out = {"budget_hours": budget_hours, "n_per_cell": n, "per_arm": {}, "arms_timed": sorted(timings),
           "arms_untimed": sorted(a for a in ARMS if a not in timings)}
    total = 0.0
    for arm, by_d in timings.items():
        ds = sorted(by_d)
        if len(ds) < 2:
            raise Refusal(f"budget extrapolation needs two measured distances for {arm}, got {ds}")
        d_lo, d_hi = ds[0], ds[-1]
        rows, arm_total = {}, 0.0
        for D in distances:
            t = (D - d_lo) / (d_hi - d_lo)
            L = by_d[d_lo]["L"] + t * (by_d[d_hi]["L"] - by_d[d_lo]["L"])
            B = batch_size_for(arm, int(round(L)))
            n_batches = math.ceil(n / B)
            s = {kind: by_d[d_lo]["s_per_batch"][kind] + t * (by_d[d_hi]["s_per_batch"][kind] - by_d[d_lo]["s_per_batch"][kind])
                 for kind in ("compare", "recall")}
            reps = 2 + (1 if (arm == "T" and D in (D0, D1)) else 0)        # plain + ablated (+ windowed for T)
            sec = (s["compare"] * 1 + s["recall"] * 3) * n_batches * reps
            if D == D1:
                sec += s["compare"] * 1 * n_batches * 2                      # C_near + C_overwrite, compare only
            rows[str(D)] = {"L": round(L), "batch_size": B, "batches_per_pass": n_batches, "seconds": round(sec, 1)}
            arm_total += sec
        out["per_arm"][arm] = {"by_distance": rows, "seconds": round(arm_total, 1), "hours": round(arm_total / 3600, 3)}
        total += arm_total
    out["total_seconds_timed_arms"] = round(total, 1)
    out["total_hours_timed_arms"] = round(total / 3600, 3)
    out["exceeds_budget"] = bool(total / 3600 > budget_hours)
    out["verdict"] = "INVALID__budget" if out["exceeds_budget"] else ("OK" if not out["arms_untimed"] else "OK_PARTIAL")
    return out


# ---------------------------------------------------------------------------------------------------
# the model side (torch, transformers; imported here only)
# ---------------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def free_gpu_bytes() -> int:
    torch = _torch()
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def require_free_gpu(redteam: dict, tag: str) -> None:
    free = free_gpu_bytes()
    redteam.setdefault("free_gpu_bytes_before_load", {})[tag] = free
    if free < FREE_GPU_MIN_BYTES:
        raise Refusal(f"free GPU memory before loading {tag} is {free / 2**30:.2f} GiB, under the "
                      f"{FREE_GPU_MIN_BYTES / 2**30:.0f} GiB the PREREG requires (section 11)")


def load_arm_cpu_fp32(repo_id: str):
    """Load on CPU in fp32 (the asserts of section 12 items 1, 6 and 10 run there), then half().to(cuda)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch = _torch()
    from transformers import AutoModelForCausalLM
    path = snapshot_dir(repo_id)
    src = str(path) if path is not None else repo_id
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, local_files_only=True)
    model.eval()
    return model, {"repo": repo_id, "revision": path.name if path is not None else None,
                   "from_local_snapshot": path is not None, "load_seconds_cpu_fp32": round(time.time() - t0, 2)}


_ORIG_SLOW_FORWARD = None


def chunked_slow_forward(self, input_states, cache_params=None, cache_position=None, attention_mask=None):
    """transformers' MambaMixer.slow_forward with the discretised tensors materialised per chunk of the sequence.

    The reference (modeling_mamba.py, 4.57) builds discrete_A, discrete_B and deltaB_u as [B, D_inner, L, N] fp32
    for the whole sequence — B * L MiB on mamba-1.4b-hf, 20 GiB at B = 16 and L = 1250 — and then loops over L.
    This copy computes the same three tensors for a slice [s, e) of L at a time and runs the same loop over that
    slice: every element of every tensor is the same expression of the same inputs, and the loop body is the
    reference's, so the outputs are the reference's (asserted on CPU fp32 by check_chunked_scan_equivalence).
    Anything this runner never uses (a cache, mambapy training) is handed to the original.
    """
    if cache_params is not None or (getattr(self, "use_mambapy", False) and self.training):
        return _ORIG_SLOW_FORWARD(self, input_states, cache_params, cache_position, attention_mask)
    torch = _torch()
    nn = torch.nn
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    projected_states = self.in_proj(input_states).transpose(1, 2)                     # [B, 2 * D_inner, L]
    hidden_states, gate = projected_states.chunk(2, dim=1)
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)
    ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size),
                            device=hidden_states.device, dtype=dtype)
    hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])              # [B, D_inner, L]
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, Bm, Cm = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    discrete_time_step = self.dt_proj(time_step)                                      # [B, L, D_inner]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)   # [B, D_inner, L]
    A = -torch.exp(self.A_log.float())                                                # [D_inner, N]
    chunk = max(1, SCAN_CHUNK_TOKENS // batch_size)
    scan_outputs = []
    for s in range(0, seq_len, chunk):
        e = min(seq_len, s + chunk)
        dt_c = discrete_time_step[:, :, s:e, None]
        discrete_A = torch.exp(A[None, :, None, :] * dt_c)                             # [B, D_inner, e-s, N]
        discrete_B = dt_c * Bm[:, None, s:e, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, s:e, None].float()
        for i in range(e - s):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state.to(dtype), Cm[:, s + i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        del discrete_A, discrete_B, deltaB_u
    scan_output = torch.stack(scan_outputs, dim=-1)                                   # [B, D_inner, L]
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = (scan_output * self.act(gate))
    return self.out_proj(scan_output.transpose(1, 2))


def install_chunked_scan() -> dict:
    """Replace MambaMixer.slow_forward by the chunked copy; record whether the fused kernels would bypass it."""
    global _ORIG_SLOW_FORWARD
    from transformers.models.mamba import modeling_mamba as mm
    if _ORIG_SLOW_FORWARD is None:
        _ORIG_SLOW_FORWARD = mm.MambaMixer.slow_forward
    mm.MambaMixer.slow_forward = chunked_slow_forward
    fast = all(getattr(mm, nm, None) is not None for nm in
               ("selective_state_update", "selective_scan_fn", "causal_conv1d_fn", "causal_conv1d_update", "mamba_inner_fn"))
    return {"installed": True, "scan_chunk_tokens": SCAN_CHUNK_TOKENS, "transformers_fast_path_available": bool(fast),
            "note": "fused mamba-ssm kernels present: cuda forwards bypass slow_forward and the dt_proj hook never fires, "
                    "so the mechanism diagnostic is skipped" if fast else "slow path in use on every device"}


def check_chunked_scan_equivalence(model, tok: Tok, item: dict, exemplars: list) -> dict:
    """The chunked scan reproduces the reference slow_forward on one short prompt (last-position logits)."""
    torch = _torch()
    from transformers.models.mamba import modeling_mamba as mm
    device = next(model.parameters()).device
    s = compositional_ids(tok, prompt_parts(item, exemplars, "compare"))
    ids, _ = pad_batch([s], device)
    with torch.no_grad():
        mm.MambaMixer.slow_forward = _ORIG_SLOW_FORWARD
        try:
            h0 = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
        finally:
            mm.MambaMixer.slow_forward = chunked_slow_forward
        h1 = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
        l0 = _logits(model, h0[0, len(s) - 1])
        l1 = _logits(model, h1[0, len(s) - 1])
    return {"max_abs_logit_diff": float((l1 - l0).abs().max()), "max_abs_hidden_diff": float((h1 - h0).abs().max()),
            "prompt_tokens": len(s), "scan_chunk_tokens": SCAN_CHUNK_TOKENS, "chunks": math.ceil(len(s) / SCAN_CHUNK_TOKENS),
            "dtype": str(next(model.parameters()).dtype), "device": str(device)}


def assert_mamba_A(model) -> dict:
    torch = _torch()
    worst, lowest = -math.inf, math.inf
    for i, layer in enumerate(model.backbone.layers):
        A = -torch.exp(layer.mixer.A_log.float())
        if torch.is_complex(A):
            raise Refusal(f"M1 layer {i}: A is complex")
        if not bool((A < 0).all()):
            raise Refusal(f"M1 layer {i}: A is not strictly negative (max {float(A.max())})")
        worst = max(worst, float(A.max()))
        lowest = min(lowest, float(A.min()))
    return {"ok": True, "layers": len(model.backbone.layers), "max_A": worst, "min_A": lowest}


def _lm_head(model):
    return getattr(model, "lm_head", None) or model.embed_out


def _logits(model, h):
    """fp32 logits of the LM head at the given hidden rows. The Mamba backbone keeps its residual stream in
    fp32 (residual_in_fp32) so last_hidden_state is fp32 under a half model; transformers' own
    MambaForCausalLM.forward casts to the head's dtype before the matmul, and so does this (found by the smoke)."""
    head = _lm_head(model)
    return head(h.to(head.weight.dtype)).float()


def _hidden(arm: str, model, ids, mask4d=None):
    if arm == "M1":
        return model.backbone(input_ids=ids, use_cache=False).last_hidden_state
    return model.gpt_neox(input_ids=ids, attention_mask=mask4d, use_cache=False).last_hidden_state


def pad_batch(seqs: list, device, extra: int = 0):
    torch = _torch()
    L = max(len(s) for s in seqs) + extra
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return ids.to(device), [len(s) for s in seqs]


def window_mask(L: int, lengths: list, exemplar_lens: list, W: int | None, dtype, device):
    """[B, 1, L, L] additive mask: a query at q reads key k iff k <= q and (k inside the exemplar span or q-k < W).
    W None = no window = plain causal attention (the W = inf equivalence assert)."""
    torch = _torch()
    q = torch.arange(L, device=device)[:, None]
    k = torch.arange(L, device=device)[None, :]
    causal = k <= q
    neg = torch.finfo(dtype).min
    out = torch.zeros((len(lengths), 1, L, L), dtype=dtype, device=device)
    for i, E in enumerate(exemplar_lens):
        allowed = causal if W is None else causal & ((k < E) | ((q - k) < W))
        out[i, 0][~allowed] = neg
    return out


def next_token_scores_batch(arm: str, model, seqs: list, cand_ids: list, exemplar_lens=None, W=None, mech=None, spans=None):
    """[B, 3] log p of each candidate token at each item's own last position (the compare readout). Right padding,
    no attention mask. spans: the items' filler spans [start, end), handed to the mechanism hook when one is given."""
    torch = _torch()
    device = next(model.parameters()).device
    ids, lengths = pad_batch(seqs, device)
    mask = None
    if arm == "T" and exemplar_lens is not None:
        mask = window_mask(ids.shape[1], lengths, exemplar_lens, W, next(model.parameters()).dtype, device)
    with torch.no_grad():
        if mech is not None:
            mech.begin(spans)
        h = _hidden(arm, model, ids, mask)
        pos = torch.tensor([n - 1 for n in lengths], device=device)
        hl = h[torch.arange(len(seqs), device=device), pos]
        lp = torch.log_softmax(_logits(model, hl), dim=-1)
        out = lp[:, torch.tensor(list(cand_ids), device=device)].cpu().numpy()
        if mech is not None:
            mech.end()
    return out


def recall_loglik_batch(arm: str, model, prompts: list, cands: list, exemplar_lens=None, W=None):
    """Summed log-likelihood of each prompt's candidate ids appended to it. cands[i] is a list of ids."""
    torch = _torch()
    device = next(model.parameters()).device
    seqs = [p + c for p, c in zip(prompts, cands)]
    ids, lengths = pad_batch(seqs, device)
    mask = None
    if arm == "T" and exemplar_lens is not None:
        mask = window_mask(ids.shape[1], lengths, exemplar_lens, W, next(model.parameters()).dtype, device)
    n_c = len(cands[0])
    if any(len(c) != n_c for c in cands):
        raise Refusal("candidate codes of unequal token length inside one batch")
    with torch.no_grad():
        h = _hidden(arm, model, ids, mask)
        pos = torch.tensor([[len(p) - 1 + j for j in range(n_c)] for p in prompts], device=device)   # [B, n_c]
        rows = torch.arange(len(seqs), device=device)[:, None]
        hl = h[rows, pos]                                                     # [B, n_c, H]
        lp = torch.log_softmax(_logits(model, hl), dim=-1)           # [B, n_c, V]
        tgt = torch.tensor(cands, device=device)                              # [B, n_c]
        ll = lp.gather(-1, tgt[:, :, None])[:, :, 0].sum(-1)
    return ll.cpu().numpy()


class MechanismHook:
    """M1 only: per-token log|lambda| of the slowest channel per layer, from a hook on mixer.dt_proj (section 5).

    dt_proj's output is pre-softplus; Delta = softplus(out); A = -exp(A_log) is [D_inner, N]; the slowest
    channel's |lambda| at a token is exp(max_d Delta[d] * max_n A[d, n]). Summed over an item's filler span,
    it is the deployed analogue of the toy's mag_max^gap.

    The per-token slowest channel is typically one whose Delta is near zero, which neither decays nor writes, so
    that frozen headline sits near 0 at every distance; two reported companions per layer say more: the median
    channel's log|lambda| summed over the filler, and the fraction of channels whose surviving signal over the
    filler is still >= 0.5. All three are reported, none gated (section 5).
    """

    HALF = math.log(0.5)

    def __init__(self, model):
        torch = _torch()
        self.torch = torch
        self.layers = list(model.backbone.layers)
        self.A_max = [(-torch.exp(l.mixer.A_log.float())).amax(dim=1).detach() for l in self.layers]   # [D_inner]
        self.handles = [l.mixer.dt_proj.register_forward_hook(self._make(i)) for i, l in enumerate(self.layers)]
        self.active = False
        self.spans = None
        self.store = {}
        self.store_ch = {}

    def _make(self, i):
        def hook(_module, _inp, out):
            if not self.active:
                return
            dt = self.torch.nn.functional.softplus(out.float())                   # [B, L, D_inner]
            v = dt * self.A_max[i].to(dt.device)[None, None, :]                   # per-token log|lambda| per channel
            self.store[i] = v.amax(dim=-1).detach()                                # [B, L]
            if self.spans is not None:
                self.store_ch[i] = self.torch.stack([v[b, s:e, :].sum(dim=0) for b, (s, e) in enumerate(self.spans)]).detach()  # [B, D_inner]
        return hook

    def begin(self, spans: list | None = None):
        self.store = {}
        self.store_ch = {}
        self.spans = spans
        self.active = True

    def end(self):
        self.active = False

    def filler_sums(self, spans: list) -> np.ndarray:
        """[n_layers, B]: sum over each item's filler span [start, end) of log|lambda|_max."""
        out = np.zeros((len(self.layers), len(spans)), dtype=float)
        for i in range(len(self.layers)):
            v = self.store[i].cpu().numpy()
            for b, (s, e) in enumerate(spans):
                out[i, b] = float(v[b, s:e].sum())
        return out

    def channel_summaries(self) -> tuple:
        """([n_layers, B] median over channels of the filler-summed log|lambda|, [n_layers, B] fraction of channels
        with filler-summed log|lambda| >= log 0.5), from the spans given to begin()."""
        med = np.zeros((len(self.layers), len(self.spans)), dtype=float)
        frac = np.zeros_like(med)
        for i in range(len(self.layers)):
            v = self.store_ch[i].cpu().numpy()                                     # [B, D_inner]
            med[i] = np.median(v, axis=1)
            frac[i] = (v >= self.HALF).mean(axis=1)
        return med, frac

    def remove(self):
        for h in self.handles:
            h.remove()


_TIMING_KEYS = ("seconds", "batches", "batch_size", "max_seq_len")


def run_cell(arm: str, model, tok: Tok, items: list, exemplars: list, family: str, D: int, name_ids: list,
             codes_len: int, variant: str = "plain", W=None, mech=None, batch_size: int | None = None) -> dict:
    """One (arm, family, distance, variant) cell. Returns the summary (with the private per-item block) plus timing.
    compare: the three name tokens' log p at the answer position, target = the largest code's slot.
    recall: the three candidate codes' summed log-likelihood, target = the queried slot."""
    t0 = time.time()
    if variant == "windowed" and W is None:
        raise Refusal("the windowed variant needs W")
    seqs, a_lens, spans = [], [], []
    for it in items:
        parts = prompt_parts(it, exemplars, family, variant)
        ids = compositional_ids(tok, parts)
        if ids is None:
            raise Refusal(f"prompt not compositional at scoring time: D={D} item {it['idx']} {family} {variant}")
        a_len = len(tok.encode(parts[0]))
        seqs.append(ids)
        a_lens.append(a_len)
        spans.append((a_len, a_len + len(tok.encode(parts[1]))))
    cb_len = len(tok.encode(code_block(items[0]["codes"])))
    exemplar_lens = [a - cb_len for a in a_lens]
    L = max(len(s) for s in seqs) + (codes_len if family == "recall" else 0)
    B = batch_size or batch_size_for(arm, L)
    n_batches = 0
    windowed = variant == "windowed"
    mech_rows, med_rows, frac_rows = [], [], []
    scores = np.zeros((len(seqs), 3), dtype=float)
    if family == "compare":
        use_mech = mech is not None and variant == "plain"
        for i in range(0, len(seqs), B):
            scores[i:i + B] = next_token_scores_batch(arm, model, seqs[i:i + B], name_ids,
                                                     exemplar_lens[i:i + B] if windowed else None, W,
                                                     mech if use_mech else None, spans[i:i + B] if use_mech else None)
            if use_mech:
                mech_rows.append(mech.filler_sums(spans[i:i + B]))
                med, frac = mech.channel_summaries()
                med_rows.append(med)
                frac_rows.append(frac)
            n_batches += 1
        targets = [it["max_slot"] for it in items]
    else:
        for slot in range(3):
            cands = [tok.encode(" " + it["codes"][slot]) for it in items]
            for i in range(0, len(seqs), B):
                scores[i:i + B, slot] = recall_loglik_batch(arm, model, seqs[i:i + B], cands[i:i + B],
                                                            exemplar_lens[i:i + B] if windowed else None, W)
                n_batches += 1
        targets = [it["q_slot"] for it in items]
    summary = argmax_cell_summary(scores, targets, [it["ow_slot"] for it in items] if variant == "overwrite" else None)
    summary["seconds"] = round(time.time() - t0, 2)
    summary["batches"] = n_batches
    summary["batch_size"] = B
    summary["max_seq_len"] = L
    if mech_rows:
        m = np.concatenate(mech_rows, axis=1)                              # [n_layers, n_items]
        by_layer = m.mean(axis=1)
        med = np.concatenate(med_rows, axis=1).mean(axis=1)                # [n_layers]
        frac = np.concatenate(frac_rows, axis=1).mean(axis=1)              # [n_layers]
        summary["_mechanism"] = {"definition": "per layer, over each item's filler span: slowest = sum of the per-token max over "
                                               "channels of Delta*max_n A (the frozen headline, section 5); median_channel = median "
                                               "over channels of the filler-summed Delta*max_n A; frac_channels_surviving_half = "
                                               "fraction of channels with filler-summed Delta*max_n A >= log 0.5; all means over items",
                                 "by_layer_mean_log_surviving": [float(x) for x in by_layer],
                                 "slowest_layer_log_surviving": float(by_layer.max()),
                                 "slowest_layer_surviving": float(math.exp(by_layer.max())),
                                 "slowest_layer": int(by_layer.argmax()),
                                 "by_layer_median_channel_log_surviving": [float(x) for x in med],
                                 "median_channel_log_surviving_max_over_layers": float(med.max()),
                                 "median_channel_log_surviving_mean_over_layers": float(med.mean()),
                                 "by_layer_frac_channels_surviving_half": [float(x) for x in frac],
                                 "frac_channels_surviving_half_mean_over_layers": float(frac.mean())}
    return summary


def check_right_padding(arm: str, model, tok: Tok, items: list, exemplars: list) -> dict:
    """Two items: logits at the last position alone vs inside a right-padded batch (section 12 item 10)."""
    torch = _torch()
    device = next(model.parameters()).device
    seqs = [compositional_ids(tok, prompt_parts(it, exemplars, "compare")) for it in items[:2]]
    with torch.no_grad():
        alone = []
        for s in seqs:
            ids, _ = pad_batch([s], device)
            h = _hidden(arm, model, ids)
            alone.append(_logits(model, h[0, len(s) - 1]))
        ids, lengths = pad_batch(seqs, device, extra=24)
        h = _hidden(arm, model, ids)
        diffs = []
        for i, n in enumerate(lengths):
            batched = _logits(model, h[i, n - 1])
            diffs.append(float((batched - alone[i]).abs().max()))
    return {"max_abs_logit_diff": max(diffs), "per_item": diffs, "dtype": str(next(model.parameters()).dtype),
            "device": str(device)}


def check_window_equivalence(model, tok: Tok, item: dict, exemplars: list) -> dict:
    """W = inf mask reproduces the unmasked Pythia logits on one short prompt (section 12 item 6)."""
    torch = _torch()
    device = next(model.parameters()).device
    parts = prompt_parts(item, exemplars, "compare")
    s = compositional_ids(tok, parts)
    ids, lengths = pad_batch([s], device)
    with torch.no_grad():
        h0 = _hidden("T", model, ids)
        mask = window_mask(ids.shape[1], lengths, [len(tok.encode(parts[0]))], None, next(model.parameters()).dtype, device)
        h1 = _hidden("T", model, ids, mask)
        l0 = _logits(model, h0[0, len(s) - 1])
        l1 = _logits(model, h1[0, len(s) - 1])
    return {"max_abs_logit_diff": float((l1 - l0).abs().max()), "prompt_tokens": len(s),
            "dtype": str(next(model.parameters()).dtype), "device": str(device)}


def to_gpu_fp16(model):
    torch = _torch()
    if not torch.cuda.is_available():
        raise Refusal("CUDA is not available; the design does not offer a CPU fallback (section 11)")
    model.half().to("cuda")
    torch.cuda.synchronize()
    return model


def release(redteam: dict | None = None, tag: str = "") -> int:
    """Give an arm's memory back to the device. The caller drops every reference to the model (and to any
    hook holding its layers) BEFORE calling this: an earlier smoke emptied the cache while evaluate_arm's own
    reference was still alive, the allocator kept the weights, and the free-GPU check refused the next arm."""
    torch = _torch()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    free = free_gpu_bytes()
    if redteam is not None:
        redteam.setdefault("free_gpu_bytes_after_release", {})[tag] = free
    return free


# ---------------------------------------------------------------------------------------------------
# items for a run
# ---------------------------------------------------------------------------------------------------

REGIONS = ("main", "smoke", "pilot", "pilot2")


def prepare_items(tok_gen: Tok, n: int, distances, seed: int, which: str, sentences: list | None = None):
    """Code pools, exemplars, sentence pool and the per-distance cells. which = 'main', 'smoke', 'pilot' or 'pilot2'."""
    if which not in REGIONS:
        raise Refusal(f"prepare_items: which must be one of {REGIONS}, got {which!r}")
    codes, codes_len = eligible_codes(tok_gen)
    pools = partition_codes(codes)
    if sentences is None:
        sentences, prov = read_wikitext_sentences()
    else:
        prov = {"source": "caller-supplied sentences", "eligible_sentences": len(sentences)}
    orders = split_orders(len(sentences))
    pool = SentencePool(sentences, orders[which])
    ex_codes, test_codes = pools[f"{which}_exemplar"], pools[f"{which}_test"]
    exemplars = build_exemplars(ex_codes, np.random.default_rng(seed))
    cells = {D: build_items(n, D, seed, pool, test_codes, exemplars, tok_gen) for D in distances}
    total = PILOT_RESERVE + SMOKE_RESERVE + PILOT2_RESERVE
    return {"cells": cells, "exemplars": exemplars, "pool": pool, "codes_len": codes_len, "code_pool_size": len(codes),
            "pools": {k: len(v) for k, v in pools.items()}, "code_pools": pools, "sentence_provenance": prov,
            "sentence_region": {"main": f"perm[:-{total}]",
                                "pilot2": f"perm[-{total}:-{PILOT_RESERVE + SMOKE_RESERVE}] reshuffled by seed {SEED_PILOT2}",
                                "smoke": f"perm[-{PILOT_RESERVE + SMOKE_RESERVE}:-{PILOT_RESERVE}] reshuffled by seed {SEED_SMOKE}",
                                "pilot": f"perm[-{PILOT_RESERVE}:] reshuffled by seed {SEED_PILOT}", "which": which,
                                "seed": seed},
            "region_indices": set(int(i) for i in orders[which]), "n_sentences": len(sentences), "sentences": sentences}


def items_digest(cells: dict) -> str:
    return hashlib.sha256(json.dumps({str(D): v for D, v in cells.items()}, sort_keys=True).encode("utf-8")).hexdigest()


def hash_lists(prep: dict) -> dict:
    """sha256 lists of every prompt (per family, distance and variant, in item order), of every code the run
    plants (test, overwrite and exemplar codes; sorted by hash) and of every sentence the pool handed out (sorted
    by hash) — so that a later run excludes by list, not by regeneration (intervene M7)."""
    cells, exemplars = prep["cells"], prep["exemplars"]
    prompts = {}
    for family in FAMILIES:
        prompts[family] = {}
        for D, items in cells.items():
            prompts[family][str(D)] = {}
            for variant in item_variants(items[0]):
                prompts[family][str(D)][variant] = [sha256_text(prompt_text(it, exemplars, family, variant)) for it in items]
    codes = set()
    for items in cells.values():
        for it in items:
            codes.update(it["codes"])
            if "ow_code" in it:
                codes.add(it["ow_code"])
    exemplar_codes = {c for e in exemplars for c in e["codes"]}
    sentences = sorted(set(prep["pool"].used))
    return {"prompts": prompts,
            "codes_test": sorted(sha256_text(c) for c in codes),
            "codes_exemplar": sorted(sha256_text(c) for c in exemplar_codes),
            "sentences": sorted(sha256_text(prep["sentences"][i]) for i in sentences),
            "n_prompts": sum(len(v) for fam in prompts.values() for d in fam.values() for v in d.values()),
            "n_codes_test": len(codes), "n_codes_exemplar": len(exemplar_codes), "n_sentences": len(sentences),
            "hash": "sha256 of the UTF-8 text; prompts in item order per cell; codes and sentences sorted by hash"}


def tokenizer_level_asserts(prep: dict, toks: dict, prior_used: set | None = None) -> dict:
    """Section 12 asserts 2, 3, 4, 5, 7 (as amended) — before any model is loaded."""
    red = {}
    red["balance_by_distance"] = {str(D): assert_balance(items, prep["exemplars"]) for D, items in prep["cells"].items()}
    red["fillers"] = assert_fillers(prep["cells"], prep["exemplars"], prior_used)
    red["fillers"]["sentences_skipped_for_a_name"] = len(prep["pool"].skipped)
    codes_len = {nm: prep["codes_len"] for nm in toks}
    for nm, t in toks.items():
        lens = set(len(t.encode(" " + c)) for items in prep["cells"].values() for it in items for c in it["codes"])
        if lens != {prep["codes_len"]}:
            raise Refusal(f"candidate codes have token lengths {sorted(lens)} in {nm}")
    red["prompts"] = assert_prompts(prep["cells"], prep["exemplars"], toks, codes_len)
    red["name_token_ids"] = {nm: {"test": name_token_ids(t, NAMES), "exemplar": name_token_ids(t, EXEMPLAR_NAMES)} for nm, t in toks.items()}
    if set(NAMES) & set(EXEMPLAR_NAMES):
        raise Refusal("exemplar names are not disjoint from the test names")
    red["names_disjoint"] = True
    region_codes = prep["code_pools"][f"{prep['sentence_region']['which']}_test"]
    red["replant_sentence_tokens"] = {nm: assert_replant_lengths(t, region_codes) for nm, t in toks.items()}
    red["code_token_length"] = prep["codes_len"]
    red["items_sha256"] = items_digest(prep["cells"])
    region = prep["region_indices"]
    outside = [i for i in prep["pool"].used if i not in region]
    if outside:
        raise Refusal(f"{len(outside)} sentences drawn outside the {prep['sentence_region']['which']} region")
    red["sentences_inside_region"] = True
    return red


# ---------------------------------------------------------------------------------------------------
# disjointness against the committed receipts (sha256 lists; amendment 17.2)
# ---------------------------------------------------------------------------------------------------

def read_receipt(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pool_identity(prep: dict, receipt: dict, name: str) -> None:
    """The committed receipt's pool (parquet sha256s, cap, eligible count) is this run's pool, so its sentence
    indices still name the same sentences."""
    fp = (receipt.get("config") or {}).get("filler_pool") or {}
    mine = prep["sentence_provenance"]
    if fp.get("sha256") != mine.get("sha256") or fp.get("eligible_sentences") != mine.get("eligible_sentences") \
            or fp.get("pool_cap") != mine.get("pool_cap"):
        raise Refusal(f"the {name} receipt's sentence pool differs from this run's (parquet sha256, cap or eligible count); "
                      f"its sentence indices cannot be mapped and the exclusion by list is refused")


def prior_sentences_and_codes(prep: dict, receipts: dict) -> dict:
    """Sentence indices and sha256s, and code sha256s, that the given committed receipts record. A receipt written
    by this readout carries sha256 lists directly (items.sha256); the smoke and 1011-pilot receipts carry sentence
    indices and drew their codes from the pool partition's reserved slices, which this runner keeps unchanged."""
    out = {"sentence_indices": set(), "sentence_sha256": set(), "code_sha256": set(), "receipts": {}}
    sents = prep["sentences"]
    for name, rec in receipts.items():
        if rec is None:
            out["receipts"][name] = "absent"
            continue
        assert_pool_identity(prep, rec, name)
        idx = set(int(i) for i in rec.get("sentence_indices_used") or [])
        out["sentence_indices"] |= idx
        out["sentence_sha256"] |= {sha256_text(sents[i]) for i in idx}
        lists = (rec.get("items") or {}).get("sha256") or {}
        out["sentence_sha256"] |= set(lists.get("sentences") or [])
        out["code_sha256"] |= set(lists.get("codes_test") or []) | set(lists.get("codes_exemplar") or [])
        region = (rec.get("config") or {}).get("sentence_region", {}).get("which") or rec.get("mode")
        if region in ("smoke", "pilot"):
            for pool in (f"{region}_exemplar", f"{region}_test"):
                out["code_sha256"] |= {sha256_text(c) for c in prep["code_pools"][pool]}
        out["receipts"][name] = {"sentence_indices": len(idx), "items_sha256": (rec.get("redteam") or {}).get("items_sha256"),
                                 "runner_sha256": (rec.get("config") or {}).get("runner_sha256"), "region": region}
    return out


def assert_disjoint_from_prior(prep: dict, lists: dict, prior: dict) -> dict:
    """This run shares no sentence (by index and by sha256) and no code (by sha256) with the committed receipts."""
    used = set(prep["pool"].used)
    si = used & prior["sentence_indices"]
    ss = set(lists["sentences"]) & prior["sentence_sha256"]
    cs = (set(lists["codes_test"]) | set(lists["codes_exemplar"])) & prior["code_sha256"]
    if si or ss or cs:
        raise Refusal(f"this run shares with a committed receipt: {len(si)} sentence indices, {len(ss)} sentence hashes, {len(cs)} code hashes")
    return {"ok": True, "prior_sentence_indices": len(prior["sentence_indices"]), "prior_sentence_sha256": len(prior["sentence_sha256"]),
            "prior_code_sha256": len(prior["code_sha256"]), "this_run_sentences": len(used),
            "this_run_codes": lists["n_codes_test"] + lists["n_codes_exemplar"], "receipts": prior["receipts"]}


# ---------------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------------

def git_facts() -> dict:
    """HEAD and the PREREG's committed blob sha256 (the seal is over the blob, never the working copy)."""
    out = {"head": None, "prereg_blob_sha256": None, "prereg_working_copy_differs_from_head": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        if head.returncode == 0:
            out["head"] = head.stdout.strip()
        rel = PREREG.relative_to(ROOT).as_posix()
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(ROOT), capture_output=True, timeout=30)
        if blob.returncode == 0:
            out["prereg_blob_sha256"] = hashlib.sha256(blob.stdout).hexdigest()
            out["prereg_working_copy_differs_from_head"] = blob.stdout.replace(b"\r\n", b"\n") != PREREG.read_bytes().replace(b"\r\n", b"\n")
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        out["error"] = str(e)
    return out


def base_receipt(mode: str, n: int, distances, arms) -> dict:
    return {"prereg": PREREG.name, "mode": mode, "smoke": mode == "smoke",
            "config": {"checkpoints": dict(ARMS), "arms_run": list(arms), "revisions": {},
                       "runner": Path(__file__).name, "runner_sha256": sha256_file(Path(__file__).resolve()),
                       "git": git_facts(),
                       "dtype": "float16 on cuda (weights loaded fp32 on CPU for the fp32 asserts, then half)",
                       "seeds": {"items": SEED_ITEMS, "pilot": SEED_PILOT, "pilot2": SEED_PILOT2, "smoke": SEED_SMOKE, "bootstrap": SEED_BOOT},
                       "readout": READOUT, "answer_format": READOUT, "retired_formats": list(RETIRED_FORMATS),
                       "compare_question": question_text("compare", 0), "recall_question": question_text("recall", 0),
                       "names": {"test": list(NAMES), "exemplar": list(EXEMPLAR_NAMES)},
                       "exemplar_slots": {"recall_query": list(EXEMPLAR_Q_SLOTS), "compare_answer": list(EXEMPLAR_MAX_SLOTS)},
                       "chance": dict(CHANCE), "distances": list(distances),
                       "D0": D0, "D1": D1, "n_per_cell": n, "screen_bar": SCREEN_BAR, "guard": {"max_slot_fraction": GUARD_SLOT_MAX},
                       "ablated_bar": ABLATED_BAR, "window_W": WINDOW_W, "batch_max": BATCH_MAX, "scan_chunk_tokens_M1": SCAN_CHUNK_TOKENS,
                       "code_max_planted": CODE_MAX_PLANTED,
                       "free_gpu_min_bytes": FREE_GPU_MIN_BYTES, "budget_hours": BUDGET_HOURS, "n_exemplars": N_EXEMPLARS},
            "pilot": None, "acc_raw": {}, "acc_norm": {}, "guard": {},
            "controls": {"ablated": {}, "windowed": {"W": WINDOW_W}, "near": {}, "overwrite": {}},
            "result": {}, "bootstrap": None, "mechanism": {}, "per_item": {}, "items": {},
            "exploratory": {"status": "M3_NOT_RUN",
                            "reason": "decision 2 default: no M3 arm; state-spaces/mamba3-siso-1.5b is not runnable on Windows as shipped (Triton, mamba-ssm absent) and was not approved"},
            "redteam": {}, "runtime": {}}


def record_versions(res: dict) -> None:
    torch = _torch()
    import transformers
    res["config"]["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "numpy": np.__version__, "python": sys.version.split()[0]}
    if torch.cuda.is_available():
        res["config"]["gpu"] = torch.cuda.get_device_name(0)


def record_items(res: dict, prep: dict) -> None:
    res["config"]["code_pools"] = prep["pools"]
    res["config"]["filler_pool"] = prep["sentence_provenance"]
    res["config"]["sentence_region"] = prep["sentence_region"]
    res["sentence_indices_used"] = sorted(prep["pool"].used)
    res["sentence_indices_skipped_for_a_name"] = sorted(prep["pool"].skipped)
    res["items"] = {"sha256": hash_lists(prep),
                    "prefix_nested_across_distances": False,
                    "code_paired_across_distances": False,
                    "nesting_note": "items are generated per distance from the item seed plus D: the filler at D = 16 is not the head of "
                                    "any other distance's filler and the codes are drawn afresh per distance; an item is shared across "
                                    "the two families, every variant and both arms at one distance only (zoo section 8 item 3)",
                    "names": {"test": list(NAMES), "exemplar": list(EXEMPLAR_NAMES), "disjoint": not (set(NAMES) & set(EXEMPLAR_NAMES))},
                    "controls_at_D1": {"near": "the item's own filler stands between the exemplars and the code block; a fresh 16-token "
                                               "filler stands between the block and the question (C_near, compare only)",
                                       "overwrite": "the item's filler keeps its head; its last sentence is 'The {name} code is now {code}.' "
                                                    "for a name that is not the correct slot and a code larger than all three; scored "
                                                    "against the original correct slot; the fraction answering the re-planted name is "
                                                    "reported (C_overwrite, compare only)"},
                    "ablated_at_every_distance": True}


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, (list, tuple)):
        return [strip_private(v) for v in d]
    if isinstance(d, np.ndarray):
        return [strip_private(v) for v in d.tolist()]
    if isinstance(d, np.generic):
        return d.item()
    return d


def write_json(path: Path, obj) -> None:
    """UTF-8 without BOM, LF newlines on every platform (review minor a), one trailing newline."""
    data = (json.dumps(strip_private(obj), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    if not path.exists() or path.stat().st_size != len(data):
        raise RuntimeError(f"write failed: {path}")


def model_level_asserts(arm: str, res: dict, prep: dict, tok: Tok, items0: list) -> None:
    """CPU fp32: section 12 items 1, 6 and 10 and the chunked-scan equivalence for M1. The model is dropped after.
    Run for EVERY arm before any cell of any arm is scored (review minor b)."""
    red = res["redteam"]
    if arm == "M1":
        red["chunked_scan"] = install_chunked_scan()
    model, info = load_arm_cpu_fp32(ARMS[arm])
    res["config"]["revisions"][arm] = info
    try:
        if arm == "M1":
            red["mamba_A_real_and_negative_every_layer"] = assert_mamba_A(model)
            red["chunked_scan_equivalence_fp32_cpu"] = ce = check_chunked_scan_equivalence(model, tok, items0[0], prep["exemplars"])
            if ce["max_abs_logit_diff"] > FP32_TOL:
                raise Refusal(f"M1: the chunked scan differs from the reference slow_forward by {ce['max_abs_logit_diff']:.2e} > {FP32_TOL}")
        red.setdefault("right_padding_fp32_cpu", {})[arm] = rp = check_right_padding(arm, model, tok, items0, prep["exemplars"])
        if rp["max_abs_logit_diff"] > FP32_TOL:
            raise Refusal(f"{arm}: right-padding changes the last-position logits by {rp['max_abs_logit_diff']:.2e} > {FP32_TOL}")
        if arm == "T":
            red["windowed_mask_W_inf_equivalence_fp32_cpu"] = we = check_window_equivalence(model, tok, items0[0], prep["exemplars"])
            if we["max_abs_logit_diff"] > FP32_TOL:
                raise Refusal(f"T: the W=inf mask changes the logits by {we['max_abs_logit_diff']:.2e} > {FP32_TOL}")
    finally:
        model = None
        release()


def preflight_all_arms(arms, res: dict, prep: dict, toks: dict, items0: list) -> None:
    """Every arm's model-level asserts, in order, before any scoring; the order and the clock are recorded."""
    t0 = time.time()
    done = []
    for arm in arms:
        model_level_asserts(arm, res, prep, toks[arm], items0)
        done.append(arm)
        print(f"  model-level asserts held for {arm} ({time.time() - t0:.0f}s)", flush=True)
    res["redteam"]["model_level_asserts_before_any_scoring"] = {"arms": done, "seconds": round(time.time() - t0, 1), "ok": True}


def load_arm_for_scoring(arm: str, res: dict, prep: dict, tok: Tok, items0: list):
    """Free-GPU check, CPU fp32 load (the asserts already ran in preflight), free-GPU check again, the fp16 move,
    the reported fp16 right-padding difference. Returns (model on cuda, mechanism_allowed)."""
    red = res["redteam"]
    if "model_level_asserts_before_any_scoring" not in red or arm not in red["model_level_asserts_before_any_scoring"]["arms"]:
        raise Refusal(f"{arm} is being loaded for scoring before its model-level asserts ran")
    require_free_gpu(red, arm)
    if arm == "M1":
        red["chunked_scan"] = install_chunked_scan()
    model, info = load_arm_cpu_fp32(ARMS[arm])
    res["config"]["revisions"][arm] = {**res["config"]["revisions"].get(arm, {}), **info}
    mech_ok = not red["chunked_scan"]["transformers_fast_path_available"] if arm == "M1" else True
    require_free_gpu(red, arm + "_to_cuda")
    t0 = time.time()
    model = to_gpu_fp16(model)
    res["runtime"].setdefault(arm, {})["to_gpu_seconds"] = round(time.time() - t0, 2)
    red.setdefault("right_padding_fp16_gpu_reported", {})[arm] = check_right_padding(arm, model, tok, items0, prep["exemplars"])
    return model, mech_ok


def _store_cell(res: dict, arm: str, family: str, D: int, variant: str, s: dict, keep: dict | None = None) -> None:
    per_item = s.pop("_per_item")
    correct = s.pop("_correct")
    mech = s.pop("_mechanism", None)
    res["per_item"].setdefault(arm, {}).setdefault(family, {}).setdefault(str(D), {})[variant] = per_item
    res["runtime"].setdefault(arm, {}).setdefault(str(D), {})[family if variant == "plain" else f"{variant}_{family}"] = {k: s[k] for k in _TIMING_KEYS}
    summary = {k: v for k, v in s.items() if k not in _TIMING_KEYS}
    if variant == "plain":
        if keep is not None:
            keep[(arm, family, D)] = correct
        if mech is not None:
            res["mechanism"].setdefault(arm, {})[str(D)] = mech
        res["acc_raw"].setdefault(arm, {}).setdefault(family, {})[str(D)] = s["acc_raw"]
        res["acc_norm"].setdefault(arm, {}).setdefault(family, {})[str(D)] = s["acc_norm"]
        res["guard"].setdefault(arm, {}).setdefault(family, {})[str(D)] = {k: v for k, v in summary.items() if k not in ("acc_raw", "acc_norm")}
    elif variant == "windowed":
        res["controls"]["windowed"].setdefault(arm, {}).setdefault(family, {})[str(D)] = summary
    else:
        res["controls"][variant].setdefault(arm, {}).setdefault(family, {})[str(D)] = summary


def evaluate_arm(arm: str, res: dict, prep: dict, toks: dict, distances, keep: dict, do_mech: bool, do_windowed: bool,
                 do_controls: bool = True) -> None:
    """Load one arm, score every cell and control, release it. keep[(arm, family, D)] = per-item correctness."""
    red = res["redteam"]
    tok = toks[arm]
    name_ids = [red["name_token_ids"][arm]["test"][nm] for nm in NAMES]
    items0 = prep["cells"][distances[0]]
    model, mech_ok = load_arm_for_scoring(arm, res, prep, tok, items0)
    mech = MechanismHook(model) if (arm == "M1" and do_mech and mech_ok) else None
    if arm == "M1" and do_mech and not mech_ok:
        res["mechanism"]["M1"] = {"status": "NOT_RUN", "reason": red["chunked_scan"]["note"]}
    try:
        for D in distances:
            items = prep["cells"][D]
            for family in FAMILIES:
                s = run_cell(arm, model, tok, items, prep["exemplars"], family, D, name_ids, prep["codes_len"], mech=mech)
                print(f"    {arm} {family:7} D={D:>4}: raw {s['acc_raw']:.3f} norm {s['acc_norm']:+.3f} max-slot {s['max_slot_fraction']:.2f} "
                      f"guard {s['guard_ok']} ({s['seconds']:.0f}s, {s['batches']} batches of {s['batch_size']})", flush=True)
                _store_cell(res, arm, family, D, "plain", s, keep)
                if do_controls:
                    s = run_cell(arm, model, tok, items, prep["exemplars"], family, D, name_ids, prep["codes_len"], variant="ablated")
                    print(f"    {arm} {family:7} D={D:>4} ABLATED: norm {s['acc_norm']:+.3f} max-slot {s['max_slot_fraction']:.2f} ({s['seconds']:.0f}s)", flush=True)
                    _store_cell(res, arm, family, D, "ablated", s)
                if do_controls and arm == "T" and do_windowed and D in (D0, D1):
                    s = run_cell(arm, model, tok, items, prep["exemplars"], family, D, name_ids, prep["codes_len"], variant="windowed", W=WINDOW_W)
                    print(f"    T  {family:7} D={D:>4} WINDOWED W={WINDOW_W}: norm {s['acc_norm']:+.3f} ({s['seconds']:.0f}s)", flush=True)
                    _store_cell(res, arm, family, D, "windowed", s)
            if do_controls and D == D1 and "ow_filler" in items[0]:
                s = run_cell(arm, model, tok, items, prep["exemplars"], "compare", D, name_ids, prep["codes_len"], variant="near")
                print(f"    {arm} compare D={D:>4} NEAR (codes 16 tokens before the question): norm {s['acc_norm']:+.3f} ({s['seconds']:.0f}s)", flush=True)
                _store_cell(res, arm, "compare", D, "near", s)
                s = run_cell(arm, model, tok, items, prep["exemplars"], "compare", D, name_ids, prep["codes_len"], variant="overwrite")
                print(f"    {arm} compare D={D:>4} OVERWRITE: norm vs original {s['acc_norm']:+.3f}, follows the re-plant {s['overwrite_follow_rate']:.2f} ({s['seconds']:.0f}s)", flush=True)
                _store_cell(res, arm, "compare", D, "overwrite", s)
    finally:
        if mech is not None:
            mech.remove()
        mech = None
        model = None
        release(red, arm)


def gate_metrics(res: dict, keep: dict) -> dict:
    """The eight gate metrics plus within_T (section 7); plumbing_ok is 1.0 because every refusal fires earlier.
    ablated_max_norm_acc is over D0 and D1 as the frozen gate reads it; the all-distance maximum is reported beside it."""
    an = res["acc_norm"]
    screen = min(an[arm][fam][str(D0)] for arm in ("M1", "T") for fam in FAMILIES)
    guard_ok = all(res["guard"][arm][fam][str(D0)]["guard_ok"] for arm in ("M1", "T") for fam in FAMILIES)
    abl_cells = res["controls"]["ablated"]
    abl = max(abl_cells[arm][fam][str(D)]["acc_norm"] for arm in ("M1", "T") for fam in FAMILIES for D in (D0, D1))
    abl_all = max(abl_cells[arm][fam][d]["acc_norm"] for arm in ("M1", "T") for fam in FAMILIES for d in abl_cells[arm][fam])
    C = {arm: {D: an[arm]["compare"][str(D)] for D in (D0, D1)} for arm in ("M1", "T")}
    R = {arm: {D: an[arm]["recall"][str(D)] for D in (D0, D1)} for arm in ("M1", "T")}
    between = (C["T"][D1] - C["M1"][D1]) - (R["T"][D1] - R["M1"][D1])
    within_M1 = (C["M1"][D0] - C["M1"][D1]) - (R["M1"][D0] - R["M1"][D1])
    within_T = (C["T"][D0] - C["T"][D1]) - (R["T"][D0] - R["T"][D1])
    boot = paired_bootstrap(keep[("T", "compare", D1)], keep[("M1", "compare", D1)],
                            keep[("T", "recall", D1)], keep[("M1", "recall", D1)])
    if abs(boot["point"] - between) > 1e-9:
        raise Refusal(f"bootstrap point estimate {boot['point']} differs from between_D1 {between}")
    res["bootstrap"] = boot
    return {"plumbing_ok": 1.0, "screen_min_norm_acc_D0": screen, "guard_ok_D0": 1.0 if guard_ok else 0.0,
            "ablated_max_norm_acc": abl, "ablated_max_norm_acc_all_distances": abl_all, "between_D1": between,
            "between_D1_ci_lo": boot["ci_lo"], "within_M1": within_M1, "within_T": within_T}


def smoke_timings(res: dict, arms) -> dict:
    out = {}
    for arm in arms:
        by_d = {}
        for D in SMOKE_DISTANCES:
            r = res["runtime"][arm][str(D)]
            by_d[D] = {"L": max(r["compare"]["max_seq_len"], r["recall"]["max_seq_len"]),
                       "s_per_batch": {"compare": r["compare"]["seconds"] / max(1, r["compare"]["batches"]),
                                       "recall": r["recall"]["seconds"] / max(1, r["recall"]["batches"])}}
        out[arm] = by_d
    return out


def load_experiment():
    sys.path.insert(0, str(ROOT))
    from styxx.protocol import Experiment, PrologueError
    try:
        return Experiment(PREREG, repo_root=ROOT)
    except PrologueError as e:
        raise Refusal(f"the PREREG is not committed; a prereg on disk is a draft and this runner scores nothing against it ({e})")


def run_smoke(arms) -> int:
    print(f"SMOKE: n={N_SMOKE} distances={SMOKE_DISTANCES} arms={arms} readout={READOUT} — INVALID-only by the gates block", flush=True)
    exp = load_experiment()
    answer_format_from_prereg(PREREG.read_text(encoding="utf-8"))
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    prep = prepare_items(toks["T"], N_SMOKE, SMOKE_DISTANCES, SEED_SMOKE, "smoke")
    res = base_receipt("smoke", N_SMOKE, SMOKE_DISTANCES, arms)
    record_versions(res)
    record_items(res, prep)
    prior = prior_sentences_and_codes(prep, {"pilot": read_receipt(PILOT1_RECEIPT), "pilot2": read_receipt(PILOT2_RECEIPT)})
    res["redteam"] = tokenizer_level_asserts(prep, toks, prior["sentence_indices"])
    res["redteam"]["disjoint_from_committed_receipts"] = assert_disjoint_from_prior(prep, res["items"]["sha256"], prior)
    res["smoke_items"] = "seed 1016, the smoke's own reserved slice of the sentence order and of the code pool: no main-set or pilot item is generated or scored here"
    keep = {}
    t0 = time.time()
    preflight_all_arms(arms, res, prep, toks, prep["cells"][SMOKE_DISTANCES[0]])
    for arm in arms:
        evaluate_arm(arm, res, prep, toks, SMOKE_DISTANCES, keep, do_mech=(arm == "M1"), do_windowed=True)
    res["runtime"]["total_seconds"] = round(time.time() - t0, 1)
    res["budget"] = extrapolate_budget(smoke_timings(res, arms))
    v = exp.score(res, smoke=True)
    res.update({"protocol_verdict": v.verdict, "gates_sha256": v.gates_sha256, "prereg_commit": v.prereg_commit,
                "vacuous_gates": v.vacuous_gates, "undeclared_power_gates": v.undeclared_power_gates})
    res["verdict"] = "INVALID__budget" if res["budget"]["exceeds_budget"] else v.verdict
    res["check_metrics"] = exp.check_metrics(res)
    write_json(SMOKE_RECEIPT, res)
    print(f"\n  budget: {json.dumps({k: res['budget'][k] for k in ('total_hours_timed_arms', 'arms_untimed', 'verdict')})}")
    print(f"===== VERDICT: {res['verdict']} =====\nwrote {SMOKE_RECEIPT.name}", flush=True)
    return 0


def arm_clears(res: dict, arm: str) -> dict:
    """The section-6 screen on the pilot's D0 cells: both families at or above the bar and both guards passed."""
    out = {}
    for fam in FAMILIES:
        g = res["guard"][arm][fam][str(D0)]
        out[fam] = {"acc_norm": res["acc_norm"][arm][fam][str(D0)], "guard_ok": g["guard_ok"],
                    "clears": bool(res["acc_norm"][arm][fam][str(D0)] >= SCREEN_BAR and g["guard_ok"])}
    out["clears"] = all(out[fam]["clears"] for fam in FAMILIES)
    return out


def pilot_outcome(cleared: dict) -> str:
    if cleared["M1"] and cleared["T"]:
        return "CLEARED_BOTH"
    if cleared["T"]:
        return "CLEARED_T_ONLY"
    if cleared["M1"]:
        return "CLEARED_M1_ONLY"
    return "NEITHER"


def run_pilot() -> int:
    print(f"PILOT 2 (amendment 17.2): n={N_PILOT} at D0={D0}, seed {SEED_PILOT2}, both arms, readout {READOUT} — licenses nothing", flush=True)
    if PILOT2_RECEIPT.exists():
        raise Refusal(f"{PILOT2_RECEIPT.name} exists: the amendment-17.2 pilot runs exactly once; a committed receipt is never regenerated")
    answer_format_from_prereg(PREREG.read_text(encoding="utf-8"))
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    prep = prepare_items(toks["T"], N_PILOT, (D0,), SEED_PILOT2, "pilot2")
    res = base_receipt("pilot2", N_PILOT, (D0,), list(ARMS))
    record_versions(res)
    record_items(res, prep)
    prior = prior_sentences_and_codes(prep, {"smoke": read_receipt(SMOKE_RECEIPT), "pilot": read_receipt(PILOT1_RECEIPT)})
    if prior["receipts"].get("pilot") == "absent" or prior["receipts"].get("smoke") == "absent":
        raise Refusal("the committed smoke and 1011-pilot receipts are needed for the disjointness assert")
    res["redteam"] = tokenizer_level_asserts(prep, toks, prior["sentence_indices"])
    res["redteam"]["disjoint_from_committed_receipts"] = assert_disjoint_from_prior(prep, res["items"]["sha256"], prior)
    res["redteam"]["main_region_disjoint"] = {"by_construction": "the pilot2 slice is cut from the seeded sentence order before the "
                                                                "main region is read; every sentence drawn lies inside it (asserted); "
                                                                "the pilot2 code pools are cut before the main test pool",
                                              "sentences_inside_pilot2_region": True}
    keep = {}
    t0 = time.time()
    items0 = prep["cells"][D0]
    preflight_all_arms(list(ARMS), res, prep, toks, items0)
    res["runtime"]["preflight_seconds"] = round(time.time() - t0, 1)
    t1 = time.time()
    for arm in ARMS:
        evaluate_arm(arm, res, prep, toks, (D0,), keep, do_mech=False, do_windowed=False, do_controls=True)
    res["runtime"]["scoring_seconds"] = round(time.time() - t1, 1)
    res["runtime"]["total_seconds"] = round(time.time() - t0, 1)
    res["screen"] = {arm: arm_clears(res, arm) for arm in ARMS}
    cleared = {arm: res["screen"][arm]["clears"] for arm in ARMS}
    outcome = pilot_outcome(cleared)
    res["pilot_outcome"] = outcome
    res["format_line_to_enter"] = READOUT if outcome == "CLEARED_BOTH" else "NOT_STARTED"
    res["pilot_rule"] = (f"amendment 17.2: the one bounded redesign, the last; if both arms clear the {SCREEN_BAR} screen (compare and "
                         f"recall normalized accuracy at D0) and the max-slot <= {GUARD_SLOT_MAX} guard, the format line is set to "
                         f"'{READOUT}' and the main run may proceed only after the seal and the operator's anchor; otherwise the format "
                         f"line is set to NOT_STARTED (final) and the behavioural route is closed by the operator's rule; these numbers "
                         f"license nothing else")
    write_json(PILOT2_RECEIPT, res)
    for arm in ARMS:
        sc = res["screen"][arm]
        print(f"  {arm}: compare {sc['compare']['acc_norm']:+.3f} guard {sc['compare']['guard_ok']} | recall {sc['recall']['acc_norm']:+.3f} "
              f"guard {sc['recall']['guard_ok']} -> {'clears' if sc['clears'] else 'fails'}", flush=True)
    print(f"\n===== PILOT OUTCOME: {outcome} (format line to enter: {res['format_line_to_enter']}) =====\nwrote {PILOT2_RECEIPT.name}", flush=True)
    return 0


def check_format_for_main(fmt: str) -> None:
    """The main run starts only when the PREREG's format line names this readout (amendment 17.2)."""
    if fmt == "PENDING_PILOT":
        raise Refusal("the PREREG's answer format still reads PENDING_PILOT: the main run is not started")
    if fmt == "NOT_STARTED":
        raise Refusal("the PREREG records NOT_STARTED on its format line: the main run is not started "
                      f"(see {PILOT1_RECEIPT.name}, {PILOT2_RECEIPT.name} and the dated amendments in the PREREG)")
    if fmt in RETIRED_FORMATS:
        raise Refusal(f"the PREREG's format line names {fmt!r}, a readout amendment 17.2 retired; this runner scores only {READOUT!r}")
    if fmt != READOUT:
        raise Refusal(f"the PREREG's format line names {fmt!r}; this runner scores {READOUT!r}")


def check_smoke_for_main(smoke: dict | None) -> dict:
    """A smoke written by this readout, inside the 4-hour rule, with BOTH arms timed (amendment 17.2, review minor c)."""
    if smoke is None:
        raise Refusal(f"no smoke receipt ({SMOKE_RECEIPT.name}); the smoke runs before the main set and measures the budget")
    if (smoke.get("config") or {}).get("readout") != READOUT:
        raise Refusal(f"the smoke receipt was not written by the {READOUT!r} readout: run --smoke --arms M1,T under this runner before the main set")
    budget = smoke.get("budget") or {}
    if budget.get("exceeds_budget"):
        raise Refusal(f"the smoke's extrapolated total is {budget.get('total_hours_timed_arms')} h, above {BUDGET_HOURS} h: INVALID__budget")
    if budget.get("arms_untimed") or sorted(budget.get("arms_timed") or []) != sorted(ARMS):
        raise Refusal(f"the smoke timed {budget.get('arms_timed')} only; amendment 17.2 requires --arms M1,T for the budget extrapolation")
    return budget


def run_main() -> int:
    exp = load_experiment()
    text = PREREG.read_text(encoding="utf-8")
    fmt = answer_format_from_prereg(text)
    check_format_for_main(fmt)
    smoke = read_receipt(SMOKE_RECEIPT)
    budget = check_smoke_for_main(smoke)
    pilot2 = read_receipt(PILOT2_RECEIPT)
    if pilot2 is None or pilot2.get("pilot_outcome") != "CLEARED_BOTH":
        raise Refusal(f"{PILOT2_RECEIPT.name} is absent or did not clear both arms: the main run is not started")
    print(f"MAIN: n={N_PER_CELL} distances={DISTANCES} readout={fmt} arms={list(ARMS)}", flush=True)
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    prep = prepare_items(toks["T"], N_PER_CELL, DISTANCES, SEED_ITEMS, "main")
    res = base_receipt("main", N_PER_CELL, DISTANCES, list(ARMS))
    record_versions(res)
    record_items(res, prep)
    prior = prior_sentences_and_codes(prep, {"smoke": smoke, "pilot": read_receipt(PILOT1_RECEIPT), "pilot2": pilot2})
    res["pilot"] = {"pilot_1011": {k: v for k, v in (read_receipt(PILOT1_RECEIPT) or {}).items() if k in ("by_format", "format_recommendation", "redteam")},
                    "pilot_1012": {k: v for k, v in pilot2.items() if k in ("acc_raw", "acc_norm", "guard", "controls", "screen", "pilot_outcome", "format_line_to_enter")}}
    res["redteam"] = tokenizer_level_asserts(prep, toks, prior["sentence_indices"])
    res["redteam"]["disjoint_from_committed_receipts"] = assert_disjoint_from_prior(prep, res["items"]["sha256"], prior)
    res["redteam"]["smoke"] = {"receipt": SMOKE_RECEIPT.name, "verdict": smoke.get("verdict"), "budget": budget,
                               "arms_run": (smoke.get("config") or {}).get("arms_run"),
                               "runner_sha256": (smoke.get("config") or {}).get("runner_sha256"),
                               "runner_matches_this_run": (smoke.get("config") or {}).get("runner_sha256") == res["config"]["runner_sha256"]}
    skeleton = dict(res)
    skeleton["result"] = {k: 0.0 for k in ("plumbing_ok", "screen_min_norm_acc_D0", "guard_ok_D0", "ablated_max_norm_acc", "between_D1", "between_D1_ci_lo", "within_M1")}
    bad = {k: v for k, v in exp.check_metrics(skeleton).items() if not v["usable"]}
    if bad:
        raise Refusal(f"gate metric paths the receipt will not carry: {bad}")
    keep = {}
    t0 = time.time()
    preflight_all_arms(list(ARMS), res, prep, toks, prep["cells"][DISTANCES[0]])
    for arm in ARMS:
        evaluate_arm(arm, res, prep, toks, DISTANCES, keep, do_mech=(arm == "M1"), do_windowed=True)
        res["runtime"]["elapsed_after_" + arm] = round(time.time() - t0, 1)
    res["runtime"]["total_seconds"] = round(time.time() - t0, 1)
    res["runtime"]["over_budget_4h"] = bool(res["runtime"]["total_seconds"] > BUDGET_HOURS * 3600)
    res["result"] = gate_metrics(res, keep)
    v = exp.score(res, smoke=False)
    res.update({"verdict": v.verdict, "gates": v.gates, "gates_sha256": v.gates_sha256, "prereg_commit": v.prereg_commit,
                "vacuous_gates": v.vacuous_gates, "undeclared_power_gates": v.undeclared_power_gates})
    write_json(RECEIPT, res)
    print("\n  result:", json.dumps({k: round(x, 4) for k, x in res["result"].items()}))
    print(f"  gates: {v.gates}")
    print(f"\n===== VERDICT: {v.verdict} =====\nwrote {RECEIPT.name}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="24 items, D in {16, 1024}; INVALID-only; measures the budget")
    ap.add_argument("--pilot", action="store_true", help="the amendment-17.2 pilot at D0 (seed 1012); writes realmodel_bridge_v3_pilot2.json")
    ap.add_argument("--arms", default="M1,T", help="smoke only: M1,T (default; the budget needs both arms timed) or T for a plumbing-only pass")
    args = ap.parse_args(argv)
    if args.smoke and args.pilot:
        ap.error("--smoke and --pilot are separate runs")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a not in ARMS for a in arms):
        ap.error(f"--arms must name M1 and/or T, got {arms}")
    try:
        if args.smoke:
            return run_smoke(arms)
        if args.pilot:
            return run_pilot()
        return run_main()
    except Refusal as e:
        note = {"refused": True, "reason": str(e), "mode": "smoke" if args.smoke else ("pilot" if args.pilot else "main"),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(REFUSAL_RECEIPT, note)
        print(f"\nREFUSED before data: {e}\nwrote {REFUSAL_RECEIPT.name}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

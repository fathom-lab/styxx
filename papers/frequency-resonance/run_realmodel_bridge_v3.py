# -*- coding: utf-8 -*-
"""run_realmodel_bridge_v3.py -- frozen by PREREG_realmodel_bridge_v3_2026_10_11.

THE REAL-MODEL BRIDGE, ATTEMPT THREE. v1 was uninterpretable (recall at ceiling, compare at chance in
130M models); v2 was degenerate (a constant responder scored the label base rate and cleared the floor,
RESULT_realmodel_bridge_2026_07_24.md). v3 does what that RESULT asked for: few-shot exemplars, exact
per-cell label balance (a constant responder scores exactly chance), a capability screen at the shortest
distance, a degeneracy guard, chance-normalized metrics, and a paired item-bootstrap interval.

Arms: M1 = state-spaces/mamba-1.4b-hf (A = -exp(A_log) real and strictly negative on the loaded weights,
asserted in every layer) and T = EleutherAI/pythia-1.4b. Byte-identical prompts, identical token ids.

  python run_realmodel_bridge_v3.py --smoke [--arms T|M1,T]
      24 items, D in {16, 1024}, T only by default (PREREG section 12 item 11); --arms M1,T times the slow
      arm as well. Smoke items come from their own seed (1016) and their own reserved slice of the sentence
      order and of the code pool, disjoint from the main set and from the pilot: the smoke scores no
      main-set item. INVALID-only by the gates block. Measures runtime per distance and extrapolates the
      main run; above 4 hours the receipt reads INVALID__budget and the main run refuses to start.
  python run_realmodel_bridge_v3.py --pilot
      the section-9 format pilot: 48 items at D0, seed 1011, codes and sentences disjoint from the main
      set, both arms, both answer formats. Written to realmodel_bridge_v3_pilot.json. Licenses nothing.
  python run_realmodel_bridge_v3.py
      the main run. Refuses without: a committed PREREG, a chosen answer format in the PREREG, a smoke
      receipt whose budget is inside 4 hours, every tokenizer-level red-team assert, and >= 6 GB of free
      GPU memory before each load. One arm resident at a time.

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
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREREG = HERE / "PREREG_realmodel_bridge_v3_2026_10_11.md"
RECEIPT = HERE / "realmodel_bridge_v3_result.json"
SMOKE_RECEIPT = HERE / "realmodel_bridge_v3_smoke.json"
PILOT_RECEIPT = HERE / "realmodel_bridge_v3_pilot.json"
REFUSAL_RECEIPT = HERE / "realmodel_bridge_v3_refusal.json"

ARMS = {"M1": "state-spaces/mamba-1.4b-hf", "T": "EleutherAI/pythia-1.4b"}
NAMES = ("alpha", "bravo", "delta")
FAMILIES = ("compare", "recall")
DISTANCES = (16, 64, 256, 1024)
D0, D1 = 16, 1024
N_PER_CELL = 600
N_SMOKE, N_PILOT = 24, 48
SMOKE_DISTANCES = (16, 1024)
SEED_ITEMS, SEED_PILOT, SEED_BOOT = 20261011, 1011, 20261011
SEED_SMOKE = 1016                         # the smoke's own items: never a main-set or pilot item
N_BOOT = 2000
SCREEN_BAR = 0.40
GUARD_YES_RANGE = (0.20, 0.80)
GUARD_SLOT_MAX = 0.80
ABLATED_BAR = 0.16
WINDOW_W = 128
MAX_PROMPT_TOKENS = 2000
FREE_GPU_MIN_BYTES = 6 * 1024 ** 3
BUDGET_HOURS = 4.0
BATCH_MAX = 16
SCAN_CHUNK_TOKENS = 1024        # M1: the slow path's four [B, 4096, L, 16] fp32 tensors are materialised per chunk of L with
                                # B * chunk <= this (about 1 GiB); the arithmetic per element is the reference's, asserted
CHANCE = {"compare": 0.5, "recall": 1.0 / 3.0}
FORMATS = {"yes_no": (" Yes", " No"), "true_false": (" True", " False")}
N_EXEMPLARS = 4
EXEMPLAR_Q_SLOTS = (0, 1, 2, 0)          # PREREG section 4: recall exemplars query slots 1, 2, 3 and 1
EXEMPLAR_LABELS = (True, False, True, False)
POOL_CAP = 150_000                        # eligible sentences read from the train split, in file order
PILOT_RESERVE = 4000                      # tail of the seeded sentence order, reserved for the pilot
SMOKE_RESERVE = 3000                      # the slice before that tail, reserved for the smoke; main = everything before both
N_EXEMPLAR_CODES = N_EXEMPLARS * 3
N_PILOT_TEST_CODES = 60
N_SMOKE_TEST_CODES = 60
LOGODDS_TIE = 1e-3
FP32_TOL = 1e-3
PAD_ID = 0                                # never read: right padding, logits taken at each item's own last position
FORMAT_LINE_RE = re.compile(r"^Answer format chosen \(written before sealing\): (\S+)\s*$", re.M)


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
_SENT_BAD = re.compile(r"\d|alpha|bravo|delta|code|<unk>", re.I)
_FILLER_BAD = re.compile(r"\d|alpha|bravo|delta|code", re.I)
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
                 reserve: int = PILOT_RESERVE, smoke_reserve: int = SMOKE_RESERVE) -> tuple:
    """One seeded permutation of the sentence indices, cut in three: main = everything before the two reserved
    slices; smoke = the slice before the tail, reshuffled by the smoke seed; pilot = the tail, reshuffled by the
    pilot seed. Disjoint by construction, so no sentence can serve two of the main set, the smoke and the pilot.
    Returns (main, smoke, pilot)."""
    if n <= reserve + smoke_reserve + 1:
        raise Refusal(f"sentence pool too small ({n}) for a pilot reserve of {reserve} and a smoke reserve of {smoke_reserve}")
    perm = np.random.default_rng(seed_main).permutation(n)
    main = perm[:-(reserve + smoke_reserve)]
    smoke_slice = perm[-(reserve + smoke_reserve):-reserve]
    tail = perm[-reserve:]
    smoke = smoke_slice[np.random.default_rng(seed_smoke).permutation(smoke_reserve)]
    pilot = tail[np.random.default_rng(seed_pilot).permutation(reserve)]
    return main, smoke, pilot


class SentencePool:
    """Eligible sentences drawn without replacement in a fixed order; every draw is recorded."""

    def __init__(self, sentences: list, order):
        self.sentences = sentences
        self.order = [int(i) for i in order]
        self.cursor = 0
        self.used = []

    def draw(self) -> int:
        if self.cursor >= len(self.order):
            raise Refusal("sentence pool exhausted — raise POOL_CAP")
        i = self.order[self.cursor]
        self.cursor += 1
        self.used.append(i)
        return i

    def fill(self, target: int, measure, max_tries: int = 64) -> tuple:
        """Distinct sentences joined by single spaces, cut at a word boundary so that measure(text) == target
        exactly. Every sentence drawn is consumed, used or not. Returns (text, [sentence indices])."""
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
        return text, used


def filler_violations(text: str) -> list:
    """Digits, the three names and the word 'code' (as substrings, case-insensitive) are never in a filler."""
    return sorted(set(m.group(0).lower() for m in _FILLER_BAD.finditer(text)))


# ---------------------------------------------------------------------------------------------------
# codes, exemplars, items (PREREG section 4)
# ---------------------------------------------------------------------------------------------------

def code_token_lengths(tok: Tok) -> dict:
    return {str(c): len(tok.encode(" %d" % c)) for c in range(1000, 10000)}


def eligible_codes(tok: Tok) -> tuple:
    """Four-digit codes whose ' '+code tokenizes to the modal length (two tokens in NeoX). Returns (codes, length)."""
    lens = code_token_lengths(tok)
    modal = max(set(lens.values()), key=lambda L: sum(1 for v in lens.values() if v == L))
    return [c for c, L in lens.items() if L == modal], modal


def partition_codes(codes: list, seed: int = SEED_ITEMS) -> dict:
    """Disjoint pools: main exemplars, pilot exemplars, pilot test, smoke exemplars, smoke test, main test (the rest)."""
    rng = np.random.default_rng(seed)
    order = [codes[i] for i in rng.permutation(len(codes))]
    a = N_EXEMPLAR_CODES
    b = a + N_EXEMPLAR_CODES
    c = b + N_PILOT_TEST_CODES
    d = c + N_EXEMPLAR_CODES
    e = d + N_SMOKE_TEST_CODES
    if len(order) < e + 3 * 10:
        raise Refusal(f"code pool too small ({len(order)})")
    return {"main_exemplar": order[:a], "pilot_exemplar": order[a:b], "pilot_test": order[b:c],
            "smoke_exemplar": order[c:d], "smoke_test": order[d:e], "main_test": order[e:]}


def code_block(codes) -> str:
    return " ".join(f"The {NAMES[j]} code is {codes[j]}." for j in range(3))


def build_exemplars(exemplar_codes: list, rng) -> list:
    """Four blocks, zero filler, two Yes / two No, recall slots 1, 2, 3, 1."""
    if len(exemplar_codes) < N_EXEMPLAR_CODES:
        raise Refusal("not enough exemplar codes")
    exs = []
    for j in range(N_EXEMPLARS):
        codes = list(exemplar_codes[3 * j: 3 * j + 3])
        q = EXEMPLAR_Q_SLOTS[j]
        label = EXEMPLAR_LABELS[j]
        others = [s for s in range(3) if s != q]
        claimed = q if label else int(rng.choice(others))
        exs.append({"codes": codes, "q_slot": q, "label": bool(label), "claimed_slot": claimed})
    return exs


def exemplar_text(ex: dict, family: str, fmt: str) -> str:
    cb = code_block(ex["codes"])
    name = NAMES[ex["q_slot"]]
    if family == "compare":
        yes, no = FORMATS[fmt]
        claimed = ex["codes"][ex["claimed_slot"]]
        return f"{cb}\nQuestion: Is the {name} code {claimed}? Answer:{yes if ex['label'] else no}"
    return f"{cb}\nQuestion: What is the {name} code? Answer: {ex['codes'][ex['q_slot']]}"


_PERMS = list(itertools.permutations(range(N_EXEMPLARS)))


def permutation_with_last(rng, exemplars: list, last_yes: bool) -> list:
    ok = [p for p in _PERMS if exemplars[p[-1]]["label"] == last_yes]
    return list(ok[int(rng.integers(0, len(ok)))])


def design_rows(n: int, rng) -> list:
    """Exactly balanced rows: label x queried slot x claimed slot x last-exemplar-Yes. n must be a multiple of 24."""
    if n % 24 != 0 or n <= 0:
        raise Refusal(f"n per cell must be a positive multiple of 24 for exact balance, got {n}")
    per = n // 24
    rows = []
    for q in range(3):
        others = [s for s in range(3) if s != q]
        for label in (True, False):
            for claimed in ([q, q] if label else others):
                for last_yes in (True, False):
                    rows.extend({"label": label, "q_slot": q, "claimed_slot": claimed, "last_yes": last_yes}
                                for _ in range(per))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def in_situ_len(tok: Tok, text: str) -> int:
    """Token count of text standing where the code block stands: after a blank line, no leading space."""
    return len(tok.encode("x\n\n" + text)) - len(tok.encode("x\n\n"))


def prompt_parts(item: dict, exemplars: list, family: str, fmt: str, ablated: bool = False) -> tuple:
    """(a, b, c): exemplars + blank line + code block (or the neutral text) | ' ' + filler | newline + question."""
    exs = "\n\n".join(exemplar_text(exemplars[k], family, fmt) for k in item["perm"])
    head = item["neutral"] if ablated else code_block(item["codes"])
    a = exs + "\n\n" + head
    b = " " + item["filler"]
    name = NAMES[item["q_slot"]]
    if family == "compare":
        c = f"\nQuestion: Is the {name} code {item['codes'][item['claimed_slot']]}? Answer:"
    else:
        c = f"\nQuestion: What is the {name} code? Answer:"
    return a, b, c


def prompt_text(item: dict, exemplars: list, family: str, fmt: str, ablated: bool = False) -> str:
    return "".join(prompt_parts(item, exemplars, family, fmt, ablated))


def compositional_ids(tok: Tok, parts: tuple) -> list | None:
    """The whole prompt's ids, if they equal the concatenation of the parts' ids; else None."""
    whole = tok.encode("".join(parts))
    joined = sum((tok.encode(p) for p in parts), [])
    return whole if whole == joined else None


def item_is_exact(item: dict, exemplars: list, tok: Tok, D: int, fmts) -> bool:
    """D-exactness and compositionality for both families and every answer format, plain and ablated."""
    for fmt in fmts:
        for family in FAMILIES:
            parts = prompt_parts(item, exemplars, family, fmt)
            ids = compositional_ids(tok, parts)
            if ids is None or len(tok.encode(parts[1])) != D:
                return False
            abl = tok.encode(prompt_text(item, exemplars, family, fmt, ablated=True))
            tail = tok.encode(parts[1]) + tok.encode(parts[2])
            if len(abl) != len(ids) or abl[-len(tail):] != tail:
                return False
    return True


def build_items(n: int, D: int, seed: int, pool: SentencePool, test_codes: list, exemplars: list,
                tok: Tok, fmts=("yes_no", "true_false"), max_attempts: int = 8) -> list:
    """n exactly balanced items at distance D. Fillers and neutral texts come from the pool without replacement."""
    rng = np.random.default_rng(seed + D)
    rows = design_rows(n, rng)
    items = []

    def filler_measure(t: str) -> int:
        return len(tok.encode(" " + t))

    def neutral_measure(t: str) -> int:
        return in_situ_len(tok, t)

    for k, row in enumerate(rows):
        codes = [str(c) for c in rng.choice(np.asarray(test_codes, dtype=object), size=3, replace=False)]
        perm = permutation_with_last(rng, exemplars, row["last_yes"])
        target_neutral = in_situ_len(tok, code_block(codes))
        for _attempt in range(max_attempts):
            filler, f_used = pool.fill(D, filler_measure)
            neutral, n_used = pool.fill(target_neutral, neutral_measure)
            item = {"idx": k, "D": D, "codes": codes, "q_slot": row["q_slot"], "label": bool(row["label"]),
                    "claimed_slot": row["claimed_slot"], "last_yes": bool(row["last_yes"]), "perm": perm,
                    "filler": filler, "filler_sentences": f_used, "neutral": neutral, "neutral_sentences": n_used}
            if item_is_exact(item, exemplars, tok, D, fmts):
                break
        else:
            raise Refusal(f"item {k} at D={D}: no exact filler in {max_attempts} attempts")
        items.append(item)
    return items


# ---------------------------------------------------------------------------------------------------
# red-team asserts on the items (PREREG section 12, asserts 2-5 and 7)
# ---------------------------------------------------------------------------------------------------

def balance_counts(items: list, exemplars: list) -> dict:
    """Counts computed from the items themselves (the last exemplar's answer is read off the permutation)."""
    n = len(items)
    c = {"n": n,
         "label_true": sum(1 for it in items if it["label"]),
         "claimed_slot": [sum(1 for it in items if it["claimed_slot"] == s) for s in range(3)],
         "q_slot": [sum(1 for it in items if it["q_slot"] == s) for s in range(3)],
         "last_exemplar_yes": sum(1 for it in items if exemplars[it["perm"][-1]]["label"]),
         "true_by_q_slot": [sum(1 for it in items if it["label"] and it["q_slot"] == s) for s in range(3)],
         "claim_consistent": all((it["claimed_slot"] == it["q_slot"]) == it["label"] for it in items),
         "codes_distinct": all(len(set(it["codes"])) == 3 for it in items)}
    return c


def expected_balance(n: int) -> dict:
    return {"label_true": n // 2, "claimed_slot": [n // 3] * 3, "q_slot": [n // 3] * 3,
            "last_exemplar_yes": n // 2, "true_by_q_slot": [n // 6] * 3,
            "claim_consistent": True, "codes_distinct": True}


def assert_balance(items: list, exemplars: list) -> dict:
    got = balance_counts(items, exemplars)
    want = expected_balance(len(items))
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise Refusal(f"exact balance failed: {bad}")
    return got


def assert_fillers(cells: dict, exemplars: list, pilot_used: set | None = None) -> dict:
    """No sentence reused across items (nor one the smoke or the pilot used); no digit, name or 'code' in any
    filler or neutral. pilot_used: sentence indices the earlier smoke and pilot receipts record."""
    seen, dup, bad = {}, [], []
    for D, items in cells.items():
        for it in items:
            for key in ("filler_sentences", "neutral_sentences"):
                for i in it[key]:
                    if i in seen:
                        dup.append((i, seen[i], (D, it["idx"], key)))
                    seen[i] = (D, it["idx"], key)
                    if pilot_used and i in pilot_used:
                        dup.append((i, "smoke/pilot", (D, it["idx"], key)))
            for key in ("filler", "neutral"):
                v = filler_violations(it[key])
                if v:
                    bad.append((D, it["idx"], key, v))
    if dup:
        raise Refusal(f"filler sentence reused: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    if bad:
        raise Refusal(f"filler contains a digit, a name or 'code': {bad[:5]}")
    return {"sentences_used": len(seen), "used_sha256": hashlib.sha256(
        json.dumps(sorted(seen)).encode("utf-8")).hexdigest()}


def assert_prompts(cells: dict, exemplars: list, toks: dict, fmt: str, codes_len: dict) -> dict:
    """Identical ids across tokenizers on every prompt, <= 2000 tokens, equal-length candidate codes."""
    n_prompts, max_len = 0, 0
    names = list(toks)
    for D, items in cells.items():
        for it in items:
            variants = []
            for family in FAMILIES:
                for ablated in (False, True):
                    variants.append(prompt_text(it, exemplars, family, fmt, ablated))
            for text in variants:
                ids = {nm: toks[nm].encode(text) for nm in names}
                ref = ids[names[0]]
                for nm in names[1:]:
                    if ids[nm] != ref:
                        raise Refusal(f"token ids differ between {names[0]} and {nm} on a prompt at D={D}, item {it['idx']}")
                n_prompts += 1
                max_len = max(max_len, len(ref) + codes_len[names[0]])   # + the candidate appended for recall
            for c in it["codes"]:
                for nm in names:
                    if len(toks[nm].encode(" " + c)) != codes_len[nm]:
                        raise Refusal(f"candidate code {c} has a different token length in {nm}")
    if max_len > MAX_PROMPT_TOKENS:
        raise Refusal(f"a prompt reaches {max_len} tokens, above the {MAX_PROMPT_TOKENS} bound")
    return {"prompts_checked": n_prompts, "max_prompt_tokens": max_len, "tokenizers": names}


def answer_token_ids(tok: Tok, fmt: str) -> dict:
    out = {}
    for s in FORMATS[fmt]:
        ids = tok.encode(s)
        if len(ids) != 1:
            raise Refusal(f"answer token {s!r} is not a single token in {tok.name}: {ids}")
        out[s] = ids[0]
    return out


def answer_format_from_prereg(text: str) -> str:
    """The last 'Answer format chosen' line wins: section 9 freezes PENDING_PILOT, a dated amendment sets it."""
    found = FORMAT_LINE_RE.findall(text)
    if not found:
        raise Refusal("the PREREG has no 'Answer format chosen (written before sealing):' line")
    fmt = found[-1]
    if fmt not in FORMATS and fmt != "PENDING_PILOT":
        raise Refusal(f"the PREREG's answer format line reads {fmt!r}, not yes_no / true_false / PENDING_PILOT")
    return fmt


# ---------------------------------------------------------------------------------------------------
# metrics (PREREG sections 5-7)
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


def compare_cell_summary(logodds, labels) -> dict:
    lo = np.asarray(logodds, dtype=float)
    y = np.asarray(labels, dtype=bool)
    pred_yes = lo > 0.0                                   # ties (exactly 0) break toward No
    correct = pred_yes == y
    raw = float(correct.mean())
    yes_rate = float(pred_yes.mean())
    return {"n": int(len(y)), "acc_raw": raw, "acc_norm": norm_acc(raw, CHANCE["compare"]),
            "yes_rate": yes_rate, "guard_ok": bool(GUARD_YES_RANGE[0] <= yes_rate <= GUARD_YES_RANGE[1]),
            "auroc": auroc(lo, y), "near_ties": int((np.abs(lo) < LOGODDS_TIE).sum()),
            "_correct": correct}


def recall_cell_summary(pred_slots, q_slots) -> dict:
    p = np.asarray(pred_slots, dtype=int)
    q = np.asarray(q_slots, dtype=int)
    correct = p == q
    raw = float(correct.mean())
    fr = [float((p == s).mean()) for s in range(3)]
    return {"n": int(len(q)), "acc_raw": raw, "acc_norm": norm_acc(raw, CHANCE["recall"]),
            "slot_fractions": fr, "max_slot_fraction": max(fr), "guard_ok": bool(max(fr) <= GUARD_SLOT_MAX),
            "_correct": correct}


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
            "seed": int(seed), "n_items": int(n), "method": "paired item bootstrap, percentile 2.5/97.5"}


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
    Seconds per batch at an unmeasured distance are linear in L between the two measured points. Passes:
    compare + 3 recall per cell at four distances, the same again for C_abl at D0 and D1, and for T the
    windowed control at D0 and D1. Arms with no timing are listed, not guessed.
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
            sec = 0.0
            for kind, passes in (("compare", 1), ("recall", 3)):
                s = by_d[d_lo]["s_per_batch"][kind] + t * (by_d[d_hi]["s_per_batch"][kind] - by_d[d_lo]["s_per_batch"][kind])
                reps = 1 + (1 if D in (D0, D1) else 0) + (1 if (arm == "T" and D in (D0, D1)) else 0)
                sec += s * passes * n_batches * reps
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


def check_chunked_scan_equivalence(model, tok: Tok, item: dict, exemplars: list, fmt: str) -> dict:
    """The chunked scan reproduces the reference slow_forward on one short prompt (last-position logits)."""
    torch = _torch()
    from transformers.models.mamba import modeling_mamba as mm
    device = next(model.parameters()).device
    s = compositional_ids(tok, prompt_parts(item, exemplars, "compare", fmt))
    ids, _ = pad_batch([s], device)
    with torch.no_grad():
        mm.MambaMixer.slow_forward = _ORIG_SLOW_FORWARD
        try:
            h0 = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
        finally:
            mm.MambaMixer.slow_forward = chunked_slow_forward
        h1 = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
        l0 = _lm_head(model)(h0[0, len(s) - 1]).float()
        l1 = _lm_head(model)(h1[0, len(s) - 1]).float()
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


def compare_logodds_batch(arm: str, model, seqs: list, yes_id: int, no_id: int, exemplar_lens=None, W=None, mech=None):
    """log p(yes) - log p(no) at each item's own last position. Right padding, no attention mask."""
    torch = _torch()
    device = next(model.parameters()).device
    ids, lengths = pad_batch(seqs, device)
    mask = None
    if arm == "T" and exemplar_lens is not None:
        mask = window_mask(ids.shape[1], lengths, exemplar_lens, W, next(model.parameters()).dtype, device)
    with torch.no_grad():
        if mech is not None:
            mech.begin()
        h = _hidden(arm, model, ids, mask)
        pos = torch.tensor([n - 1 for n in lengths], device=device)
        hl = h[torch.arange(len(seqs), device=device), pos]
        lp = torch.log_softmax(_lm_head(model)(hl).float(), dim=-1)
        out = (lp[:, yes_id] - lp[:, no_id]).cpu().numpy()
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
        lp = torch.log_softmax(_lm_head(model)(hl).float(), dim=-1)           # [B, n_c, V]
        tgt = torch.tensor(cands, device=device)                              # [B, n_c]
        ll = lp.gather(-1, tgt[:, :, None])[:, :, 0].sum(-1)
    return ll.cpu().numpy()


class MechanismHook:
    """M1 only: per-token log|lambda| of the slowest channel per layer, from a hook on mixer.dt_proj (section 5).

    dt_proj's output is pre-softplus; Delta = softplus(out); A = -exp(A_log) is [D_inner, N]; the slowest
    channel's |lambda| at a token is exp(max_d Delta[d] * max_n A[d, n]). Summed over an item's filler span,
    it is the deployed analogue of the toy's mag_max^gap.
    """

    def __init__(self, model):
        torch = _torch()
        self.torch = torch
        self.layers = list(model.backbone.layers)
        self.A_max = [(-torch.exp(l.mixer.A_log.float())).amax(dim=1).detach() for l in self.layers]   # [D_inner]
        self.handles = [l.mixer.dt_proj.register_forward_hook(self._make(i)) for i, l in enumerate(self.layers)]
        self.active = False
        self.store = {}

    def _make(self, i):
        def hook(_module, _inp, out):
            if not self.active:
                return
            dt = self.torch.nn.functional.softplus(out.float())                   # [B, L, D_inner]
            self.store[i] = (dt * self.A_max[i].to(dt.device)[None, None, :]).amax(dim=-1).detach()  # [B, L]
        return hook

    def begin(self):
        self.store = {}
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

    def remove(self):
        for h in self.handles:
            h.remove()


def run_cell(arm: str, model, tok: Tok, items: list, exemplars: list, family: str, fmt: str, D: int,
             ans_ids: dict, codes_len: int, ablated: bool = False, W=None, mech=None, batch_size: int | None = None) -> dict:
    """One (arm, family, distance) cell, plain / ablated / windowed. Returns the summary plus timing."""
    t0 = time.time()
    seqs, a_lens, spans = [], [], []
    for it in items:
        parts = prompt_parts(it, exemplars, family, fmt, ablated)
        ids = compositional_ids(tok, parts)
        if ids is None:
            raise Refusal(f"prompt not compositional at scoring time: D={D} item {it['idx']} {family} ablated={ablated}")
        a_len = len(tok.encode(parts[0]))
        seqs.append(ids)
        a_lens.append(a_len)
        spans.append((a_len, a_len + D))
    cb_len = len(tok.encode(code_block(items[0]["codes"])))
    exemplar_lens = [a - cb_len for a in a_lens]
    L = max(len(s) for s in seqs) + (codes_len if family == "recall" else 0)
    B = batch_size or batch_size_for(arm, L)
    n_batches = 0
    windowed = W is not None
    mech_rows = []
    if family == "compare":
        yes_s, no_s = FORMATS[fmt]
        lo = []
        for i in range(0, len(seqs), B):
            lo.append(compare_logodds_batch(arm, model, seqs[i:i + B], ans_ids[yes_s], ans_ids[no_s],
                                            exemplar_lens[i:i + B] if windowed else None, W,
                                            mech if (mech is not None and not ablated) else None))
            if mech is not None and not ablated:
                mech_rows.append(mech.filler_sums(spans[i:i + B]))
            n_batches += 1
        logodds = np.concatenate(lo)
        summary = compare_cell_summary(logodds, [it["label"] for it in items])
        summary["logodds_mean_abs"] = float(np.abs(logodds).mean())
    else:
        lls = np.zeros((len(seqs), 3), dtype=float)
        for slot in range(3):
            cands = [tok.encode(" " + it["codes"][slot]) for it in items]
            for i in range(0, len(seqs), B):
                lls[i:i + B, slot] = recall_loglik_batch(arm, model, seqs[i:i + B], cands[i:i + B],
                                                         exemplar_lens[i:i + B] if windowed else None, W)
                n_batches += 1
        summary = recall_cell_summary(lls.argmax(axis=1), [it["q_slot"] for it in items])
    summary["seconds"] = round(time.time() - t0, 2)
    summary["batches"] = n_batches
    summary["batch_size"] = B
    summary["max_seq_len"] = L
    if mech_rows:
        m = np.concatenate(mech_rows, axis=1)                              # [n_layers, n_items]
        by_layer = m.mean(axis=1)
        summary["_mechanism"] = {"by_layer_mean_log_surviving": [float(x) for x in by_layer],
                                 "slowest_layer_log_surviving": float(by_layer.max()),
                                 "slowest_layer_surviving": float(math.exp(by_layer.max())),
                                 "slowest_layer": int(by_layer.argmax())}
    return summary


def check_right_padding(arm: str, model, tok: Tok, items: list, exemplars: list, fmt: str) -> dict:
    """Two items: logits at the last position alone vs inside a right-padded batch (section 12 item 10)."""
    torch = _torch()
    device = next(model.parameters()).device
    seqs = [compositional_ids(tok, prompt_parts(it, exemplars, "compare", fmt)) for it in items[:2]]
    with torch.no_grad():
        alone = []
        for s in seqs:
            ids, _ = pad_batch([s], device)
            h = _hidden(arm, model, ids)
            alone.append(_lm_head(model)(h[0, len(s) - 1]).float())
        ids, lengths = pad_batch(seqs, device, extra=24)
        h = _hidden(arm, model, ids)
        diffs = []
        for i, n in enumerate(lengths):
            batched = _lm_head(model)(h[i, n - 1]).float()
            diffs.append(float((batched - alone[i]).abs().max()))
    return {"max_abs_logit_diff": max(diffs), "per_item": diffs, "dtype": str(next(model.parameters()).dtype),
            "device": str(device)}


def check_window_equivalence(model, tok: Tok, item: dict, exemplars: list, fmt: str) -> dict:
    """W = inf mask reproduces the unmasked Pythia logits on one short prompt (section 12 item 6)."""
    torch = _torch()
    device = next(model.parameters()).device
    parts = prompt_parts(item, exemplars, "compare", fmt)
    s = compositional_ids(tok, parts)
    ids, lengths = pad_batch([s], device)
    with torch.no_grad():
        h0 = _hidden("T", model, ids)
        mask = window_mask(ids.shape[1], lengths, [len(tok.encode(parts[0]))], None, next(model.parameters()).dtype, device)
        h1 = _hidden("T", model, ids, mask)
        l0 = _lm_head(model)(h0[0, len(s) - 1]).float()
        l1 = _lm_head(model)(h1[0, len(s) - 1]).float()
    return {"max_abs_logit_diff": float((l1 - l0).abs().max()), "prompt_tokens": len(s),
            "dtype": str(next(model.parameters()).dtype), "device": str(device)}


def to_gpu_fp16(model):
    torch = _torch()
    if not torch.cuda.is_available():
        raise Refusal("CUDA is not available; the design does not offer a CPU fallback (section 11)")
    model.half().to("cuda")
    torch.cuda.synchronize()
    return model


def release(model):
    torch = _torch()
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------
# items for a run
# ---------------------------------------------------------------------------------------------------

def prepare_items(tok_gen: Tok, n: int, distances, seed: int, which: str, fmts, sentences: list | None = None):
    """Code pools, exemplars, sentence pool and the per-distance cells. which = 'main', 'smoke' or 'pilot'."""
    if which not in ("main", "smoke", "pilot"):
        raise Refusal(f"prepare_items: which must be main, smoke or pilot, got {which!r}")
    codes, codes_len = eligible_codes(tok_gen)
    pools = partition_codes(codes)
    if sentences is None:
        sentences, prov = read_wikitext_sentences()
    else:
        prov = {"source": "caller-supplied sentences", "eligible_sentences": len(sentences)}
    orders = dict(zip(("main", "smoke", "pilot"), split_orders(len(sentences))))
    pool = SentencePool(sentences, orders[which])
    ex_codes, test_codes = pools[f"{which}_exemplar"], pools[f"{which}_test"]
    exemplars = build_exemplars(ex_codes, np.random.default_rng(seed))
    cells = {D: build_items(n, D, seed, pool, test_codes, exemplars, tok_gen, fmts=fmts) for D in distances}
    both = PILOT_RESERVE + SMOKE_RESERVE
    return {"cells": cells, "exemplars": exemplars, "pool": pool, "codes_len": codes_len, "code_pool_size": len(codes),
            "pools": {k: len(v) for k, v in pools.items()}, "sentence_provenance": prov,
            "sentence_region": {"main": f"perm[:-{both}]", "smoke": f"perm[-{both}:-{PILOT_RESERVE}] reshuffled by seed {SEED_SMOKE}",
                                "pilot": f"perm[-{PILOT_RESERVE}:] reshuffled by seed {SEED_PILOT}", "which": which,
                                "seed": seed},
            "n_sentences": len(sentences)}


def items_digest(cells: dict) -> str:
    return hashlib.sha256(json.dumps({str(D): v for D, v in cells.items()}, sort_keys=True).encode("utf-8")).hexdigest()


def tokenizer_level_asserts(prep: dict, toks: dict, fmt: str, pilot_used: set | None = None) -> dict:
    """Section 12 asserts 2, 3, 4, 5, 7 — before any model is loaded."""
    red = {}
    red["balance_by_distance"] = {str(D): assert_balance(items, prep["exemplars"]) for D, items in prep["cells"].items()}
    red["fillers"] = assert_fillers(prep["cells"], prep["exemplars"], pilot_used)
    codes_len = {nm: prep["codes_len"] for nm in toks}
    for nm, t in toks.items():
        lens = set(len(t.encode(" " + c)) for items in prep["cells"].values() for it in items for c in it["codes"])
        if lens != {prep["codes_len"]}:
            raise Refusal(f"candidate codes have token lengths {sorted(lens)} in {nm}")
    red["prompts"] = assert_prompts(prep["cells"], prep["exemplars"], toks, fmt, codes_len)
    red["answer_token_ids"] = {nm: {f: answer_token_ids(t, f) for f in FORMATS} for nm, t in toks.items()}
    red["code_token_length"] = prep["codes_len"]
    red["items_sha256"] = items_digest(prep["cells"])
    return red


# ---------------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------------

def base_receipt(mode: str, fmt: str, n: int, distances, arms) -> dict:
    return {"prereg": PREREG.name, "mode": mode, "smoke": mode == "smoke",
            "config": {"checkpoints": dict(ARMS), "arms_run": list(arms), "revisions": {},
                       "dtype": "float16 on cuda (weights loaded fp32 on CPU for the fp32 asserts, then half)",
                       "seeds": {"items": SEED_ITEMS, "pilot": SEED_PILOT, "bootstrap": SEED_BOOT},
                       "answer_format": fmt, "answer_tokens": list(FORMATS.get(fmt, ())), "distances": list(distances),
                       "D0": D0, "D1": D1, "n_per_cell": n, "screen_bar": SCREEN_BAR, "guard": {"yes_rate": list(GUARD_YES_RANGE), "max_slot_fraction": GUARD_SLOT_MAX},
                       "ablated_bar": ABLATED_BAR, "window_W": WINDOW_W, "batch_max": BATCH_MAX, "scan_chunk_tokens_M1": SCAN_CHUNK_TOKENS,
                       "free_gpu_min_bytes": FREE_GPU_MIN_BYTES, "budget_hours": BUDGET_HOURS, "names": list(NAMES),
                       "n_exemplars": N_EXEMPLARS},
            "pilot": None, "acc_raw": {}, "acc_norm": {}, "guard": {}, "controls": {"ablated": {}, "windowed": {"W": WINDOW_W}},
            "result": {}, "bootstrap": None, "mechanism": {},
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
    path.write_text(json.dumps(strip_private(obj), indent=2), encoding="utf-8")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"write failed: {path}")


def load_arm_with_asserts(arm: str, res: dict, prep: dict, tok: Tok, fmt: str, items0: list):
    """Free-GPU check, CPU fp32 load, the fp32 asserts of section 12 (items 1, 6, 10 and the chunked-scan
    equivalence for M1), the free-GPU check again, the fp16 move, the reported fp16 right-padding difference.
    Returns (model on cuda, mechanism_allowed)."""
    red = res["redteam"]
    require_free_gpu(red, arm)
    if arm == "M1":
        red["chunked_scan"] = install_chunked_scan()
    model, info = load_arm_cpu_fp32(ARMS[arm])
    res["config"]["revisions"][arm] = info
    mech_ok = True
    if arm == "M1":
        red["mamba_A_real_and_negative_every_layer"] = assert_mamba_A(model)
        red["chunked_scan_equivalence_fp32_cpu"] = ce = check_chunked_scan_equivalence(model, tok, items0[0], prep["exemplars"], fmt)
        if ce["max_abs_logit_diff"] > FP32_TOL:
            raise Refusal(f"M1: the chunked scan differs from the reference slow_forward by {ce['max_abs_logit_diff']:.2e} > {FP32_TOL}")
        mech_ok = not red["chunked_scan"]["transformers_fast_path_available"]
    red.setdefault("right_padding_fp32_cpu", {})[arm] = rp = check_right_padding(arm, model, tok, items0, prep["exemplars"], fmt)
    if rp["max_abs_logit_diff"] > FP32_TOL:
        raise Refusal(f"{arm}: right-padding changes the last-position logits by {rp['max_abs_logit_diff']:.2e} > {FP32_TOL}")
    if arm == "T":
        red["windowed_mask_W_inf_equivalence_fp32_cpu"] = we = check_window_equivalence(model, tok, items0[0], prep["exemplars"], fmt)
        if we["max_abs_logit_diff"] > FP32_TOL:
            raise Refusal(f"T: the W=inf mask changes the logits by {we['max_abs_logit_diff']:.2e} > {FP32_TOL}")
    require_free_gpu(red, arm + "_to_cuda")
    t0 = time.time()
    model = to_gpu_fp16(model)
    res["runtime"].setdefault(arm, {})["to_gpu_seconds"] = round(time.time() - t0, 2)
    red.setdefault("right_padding_fp16_gpu_reported", {})[arm] = check_right_padding(arm, model, tok, items0, prep["exemplars"], fmt)
    return model, mech_ok


def evaluate_arm(arm: str, res: dict, prep: dict, toks: dict, fmt: str, distances, keep: dict, do_mech: bool, do_windowed: bool) -> None:
    """Load one arm, run its asserts, score every cell, release it. keep[(arm, family, D)] = per-item correctness."""
    red = res["redteam"]
    tok = toks[arm]
    ans = red["answer_token_ids"][arm][fmt]
    items0 = prep["cells"][distances[0]]
    model, mech_ok = load_arm_with_asserts(arm, res, prep, tok, fmt, items0)
    mech = MechanismHook(model) if (arm == "M1" and do_mech and mech_ok) else None
    if arm == "M1" and do_mech and not mech_ok:
        res["mechanism"]["M1"] = {"status": "NOT_RUN", "reason": red["chunked_scan"]["note"]}
    try:
        for D in distances:
            items = prep["cells"][D]
            for family in FAMILIES:
                s = run_cell(arm, model, tok, items, prep["exemplars"], family, fmt, D, ans, prep["codes_len"], mech=mech)
                keep[(arm, family, D)] = s.pop("_correct")
                if "_mechanism" in s:
                    res["mechanism"].setdefault(arm, {})[str(D)] = s.pop("_mechanism")
                res["acc_raw"].setdefault(arm, {}).setdefault(family, {})[str(D)] = s["acc_raw"]
                res["acc_norm"].setdefault(arm, {}).setdefault(family, {})[str(D)] = s["acc_norm"]
                res["guard"].setdefault(arm, {}).setdefault(family, {})[str(D)] = {k: v for k, v in s.items() if k not in ("acc_raw", "acc_norm", "seconds", "batches", "batch_size", "max_seq_len")}
                res["runtime"][arm].setdefault(str(D), {})[family] = {k: s[k] for k in ("seconds", "batches", "batch_size", "max_seq_len")}
                print(f"    {arm} {family:7} D={D:>4}: raw {s['acc_raw']:.3f} norm {s['acc_norm']:+.3f} guard {s['guard_ok']} ({s['seconds']:.0f}s, {s['batches']} batches of {s['batch_size']})", flush=True)
            if D in (D0, D1):
                for family in FAMILIES:
                    s = run_cell(arm, model, tok, items, prep["exemplars"], family, fmt, D, ans, prep["codes_len"], ablated=True)
                    s.pop("_correct")
                    res["controls"]["ablated"].setdefault(arm, {}).setdefault(family, {})[str(D)] = s
                    res["runtime"][arm][str(D)][f"ablated_{family}"] = {k: s[k] for k in ("seconds", "batches", "batch_size", "max_seq_len")}
                    print(f"    {arm} {family:7} D={D:>4} ABLATED: norm {s['acc_norm']:+.3f} ({s['seconds']:.0f}s)", flush=True)
                if arm == "T" and do_windowed:
                    for family in FAMILIES:
                        s = run_cell(arm, model, tok, items, prep["exemplars"], family, fmt, D, ans, prep["codes_len"], W=WINDOW_W)
                        s.pop("_correct")
                        res["controls"]["windowed"].setdefault("T", {}).setdefault(family, {})[str(D)] = s
                        res["runtime"][arm][str(D)][f"windowed_{family}"] = {k: s[k] for k in ("seconds", "batches", "batch_size", "max_seq_len")}
                        print(f"    T  {family:7} D={D:>4} WINDOWED W={WINDOW_W}: norm {s['acc_norm']:+.3f} ({s['seconds']:.0f}s)", flush=True)
    finally:
        if mech is not None:
            mech.remove()
        release(model)


def gate_metrics(res: dict, keep: dict) -> dict:
    """The eight gate metrics plus within_T (section 7); plumbing_ok is 1.0 because every refusal fires earlier."""
    an = res["acc_norm"]
    screen = min(an[arm][fam][str(D0)] for arm in ("M1", "T") for fam in FAMILIES)
    guard_ok = all(res["guard"][arm][fam][str(D0)]["guard_ok"] for arm in ("M1", "T") for fam in FAMILIES)
    abl = max(res["controls"]["ablated"][arm][fam][str(D)]["acc_norm"] for arm in ("M1", "T") for fam in FAMILIES for D in (D0, D1))
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
            "ablated_max_norm_acc": abl, "between_D1": between, "between_D1_ci_lo": boot["ci_lo"],
            "within_M1": within_M1, "within_T": within_T}


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
    print(f"SMOKE: n={N_SMOKE} distances={SMOKE_DISTANCES} arms={arms} — INVALID-only by the gates block", flush=True)
    exp = load_experiment()
    fmt = answer_format_from_prereg(PREREG.read_text(encoding="utf-8"))
    fmt = "yes_no" if fmt == "PENDING_PILOT" else fmt
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    prep = prepare_items(toks["T"], N_SMOKE, SMOKE_DISTANCES, SEED_SMOKE, "smoke", fmts=(fmt,))
    res = base_receipt("smoke", fmt, N_SMOKE, SMOKE_DISTANCES, arms)
    record_versions(res)
    res["config"]["seeds"]["smoke"] = SEED_SMOKE
    res["config"]["code_pools"] = prep["pools"]
    res["config"]["filler_pool"] = prep["sentence_provenance"]
    res["config"]["sentence_region"] = prep["sentence_region"]
    res["redteam"] = tokenizer_level_asserts(prep, toks, fmt)
    res["sentence_indices_used"] = sorted(prep["pool"].used)
    res["smoke_items"] = "seed 1016, the smoke's own reserved slice of the sentence order and of the code pool: no main-set or pilot item is generated or scored here"
    keep = {}
    t0 = time.time()
    for arm in arms:
        evaluate_arm(arm, res, prep, toks, fmt, SMOKE_DISTANCES, keep, do_mech=(arm == "M1"), do_windowed=True)
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


def run_pilot() -> int:
    print(f"PILOT: n={N_PILOT} at D0={D0}, seed {SEED_PILOT}, both arms, both formats — licenses nothing", flush=True)
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    prep = prepare_items(toks["T"], N_PILOT, (D0,), SEED_PILOT, "pilot", fmts=tuple(FORMATS))
    res = base_receipt("pilot", "both", N_PILOT, (D0,), list(ARMS))
    record_versions(res)
    res["config"]["code_pools"] = prep["pools"]
    res["config"]["filler_pool"] = prep["sentence_provenance"]
    res["config"]["sentence_region"] = prep["sentence_region"]
    res["redteam"] = tokenizer_level_asserts(prep, toks, "yes_no")
    res["redteam"]["prompts_true_false"] = assert_prompts(prep["cells"], prep["exemplars"], toks, "true_false", {nm: prep["codes_len"] for nm in toks})
    res["sentence_indices_used"] = sorted(prep["pool"].used)
    res["by_format"] = {}
    red = res["redteam"]
    t0 = time.time()
    for arm in ARMS:
        tok = toks[arm]
        items = prep["cells"][D0]
        model, _mech_ok = load_arm_with_asserts(arm, res, prep, tok, "yes_no", items)
        try:
            for fmt in FORMATS:
                ans = red["answer_token_ids"][arm][fmt]
                s = run_cell(arm, model, tok, items, prep["exemplars"], "compare", fmt, D0, ans, prep["codes_len"])
                s.pop("_correct")
                res["by_format"].setdefault(fmt, {}).setdefault(arm, {})["compare"] = s
                print(f"    {arm} compare [{fmt}]: raw {s['acc_raw']:.3f} norm {s['acc_norm']:+.3f} yes_rate {s['yes_rate']:.2f} guard {s['guard_ok']}", flush=True)
            s = run_cell(arm, model, tok, items, prep["exemplars"], "recall", "yes_no", D0, red["answer_token_ids"][arm]["yes_no"], prep["codes_len"])
            s.pop("_correct")
            res["by_format"].setdefault("recall", {})[arm] = s
            print(f"    {arm} recall: raw {s['acc_raw']:.3f} norm {s['acc_norm']:+.3f} guard {s['guard_ok']}", flush=True)
        finally:
            release(model)
    res["runtime"]["total_seconds"] = round(time.time() - t0, 1)

    def passes(fmt):
        return all(res["by_format"][fmt][a]["compare"]["acc_norm"] >= SCREEN_BAR and res["by_format"][fmt][a]["compare"]["guard_ok"] for a in ARMS)
    if passes("yes_no"):
        rec = "yes_no"
    elif passes("true_false"):
        rec = "true_false"
    else:
        rec = "NEITHER — both formats fail the pilot; the run is not started and the record says so (section 9)"
    res["format_recommendation"] = rec
    res["pilot_rule"] = ("section 9: the default ' Yes'/' No' stays unless either arm's compare normalized accuracy is below "
                         f"{SCREEN_BAR} or fails the guard; then the one switch to ' True'/' False'; the chosen format is entered in "
                         "the PREREG as a dated amendment before the seal; these numbers license nothing")
    write_json(PILOT_RECEIPT, res)
    print(f"\n===== PILOT FORMAT RECOMMENDATION: {rec} =====\nwrote {PILOT_RECEIPT.name}", flush=True)
    return 0


def run_main() -> int:
    exp = load_experiment()
    text = PREREG.read_text(encoding="utf-8")
    fmt = answer_format_from_prereg(text)
    if fmt == "PENDING_PILOT":
        raise Refusal("the PREREG's answer format still reads PENDING_PILOT: run --pilot, enter the format as a dated amendment, seal, then run")
    if not SMOKE_RECEIPT.exists():
        raise Refusal(f"no smoke receipt ({SMOKE_RECEIPT.name}); the smoke runs before the main set and measures the budget")
    smoke = json.loads(SMOKE_RECEIPT.read_text(encoding="utf-8"))
    budget = smoke.get("budget") or {}
    if budget.get("exceeds_budget"):
        raise Refusal(f"the smoke's extrapolated total is {budget.get('total_hours_timed_arms')} h, above {BUDGET_HOURS} h: INVALID__budget")
    if budget.get("arms_untimed"):
        print(f"  note: the smoke timed {budget.get('arms_timed')} only; {budget.get('arms_untimed')} untimed (section 12 item 11 froze the smoke as T only)", flush=True)
    print(f"MAIN: n={N_PER_CELL} distances={DISTANCES} format={fmt} arms={list(ARMS)}", flush=True)
    toks = {a: load_tokenizer(ARMS[a]) for a in ARMS}
    pilot = json.loads(PILOT_RECEIPT.read_text(encoding="utf-8")) if PILOT_RECEIPT.exists() else {"status": "not run"}
    pilot_used = set(pilot.get("sentence_indices_used") or []) | set(smoke.get("sentence_indices_used") or [])
    prep = prepare_items(toks["T"], N_PER_CELL, DISTANCES, SEED_ITEMS, "main", fmts=(fmt,))
    res = base_receipt("main", fmt, N_PER_CELL, DISTANCES, list(ARMS))
    record_versions(res)
    res["config"]["code_pools"] = prep["pools"]
    res["config"]["filler_pool"] = prep["sentence_provenance"]
    res["config"]["sentence_region"] = prep["sentence_region"]
    res["pilot"] = {k: v for k, v in pilot.items() if k != "sentence_indices_used"}
    res["redteam"] = tokenizer_level_asserts(prep, toks, fmt, pilot_used)
    res["redteam"]["smoke"] = {"receipt": SMOKE_RECEIPT.name, "verdict": smoke.get("verdict"), "budget": budget,
                               "arms_run": (smoke.get("config") or {}).get("arms_run")}
    res["redteam"]["no_sentence_shared_with_smoke_or_pilot"] = {"checked_against": len(pilot_used), "ok": True}
    skeleton = dict(res)
    skeleton["result"] = {k: 0.0 for k in ("plumbing_ok", "screen_min_norm_acc_D0", "guard_ok_D0", "ablated_max_norm_acc", "between_D1", "between_D1_ci_lo", "within_M1")}
    bad = {k: v for k, v in exp.check_metrics(skeleton).items() if not v["usable"]}
    if bad:
        raise Refusal(f"gate metric paths the receipt will not carry: {bad}")
    keep = {}
    t0 = time.time()
    for arm in ARMS:
        evaluate_arm(arm, res, prep, toks, fmt, DISTANCES, keep, do_mech=(arm == "M1"), do_windowed=True)
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
    ap.add_argument("--pilot", action="store_true", help="the section-9 format pilot at D0; writes its own receipt")
    ap.add_argument("--arms", default="T", help="smoke only: T (default, as frozen) or M1,T to time the slow arm too")
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

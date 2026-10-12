# -*- coding: utf-8 -*-
"""run_zoo_d0_screen.py -- the zoo D0 screen planned in NOTE_zoo_d0_screen_plan_2026_10_12.md.

A PILOT-CLASS readiness screen the operator approved on 2026-10-11. It licenses nothing, it is not part of bridge
v3's verdict, and its one question is whether ANY cached model can perform the v3 compare task, and the v3 recall
task, at D0 = 16 under the argmax readout of amendment 17.2. No gates block, no verdict, nothing scored through
styxx.protocol.

Everything that defines an item or a score is v3's, imported from the committed run_realmodel_bridge_v3.py and not
copied: the generator (design_rows, build_exemplars, build_items, SentencePool), the templates and names, the balance
and filler asserts, the sha256 lists and the disjointness assert against the committed receipts, the forward readouts
(next_token_scores_batch, recall_loglik_batch), argmax_cell_summary and the screen rule (arm_clears). What this file
adds: the zoo's own sentence and code slices (seed 1013), the per-model tokenizer and family asserts, and one wrapper
-- the ids of a prompt are the model's own tokenization of the whole prompt text (with the tokenizer's default BOS),
and Llama, Qwen and Gemma reach v3's attention path through an adapter (gemma-2's final logit softcapping applied as
transformers applies it). The wrapper must reproduce the committed pilot2 receipt's per-item outputs on M1 and T
before any zoo item is scored.

The plan's addendum (NOTE section 12, dated 2026-10-12, written after the review and before any data) adds: the
Llama, Qwen and Gemma rows load by transformers' own fp16 load with no cast after it (the RoPE inv_freq buffer stays
fp32, as transformers keeps it) and every buffer's dtype is recorded; an fp32 CPU reference per row (right padding at
v3's 1e-3 in fp32, and the fp16 name log-probs against fp32 on the same two prompts); the status labels the receipt
can carry, each named in the note; the printed sentence's wording and its C_abl flag counts; an append-only partial
log; and the three committed v3 receipts in the committed-and-unchanged check.

  python run_zoo_d0_screen.py --check
      every CPU assert of the run: items, balance, fillers, disjointness from the committed receipts, the pilot2
      regeneration, the per-model tokenizer asserts and family facts. Loads no model, touches no GPU, writes nothing.
  python run_zoo_d0_screen.py
      the run. Refuses unless the plan note, this runner, the v3 runner and the three committed v3 receipts are
      committed and unchanged at HEAD, unless zoo_d0_screen.json is absent (written once, never regenerated), and
      unless zoo_d0_screen.partial.jsonl is absent (a trace that an earlier attempt saw model outputs) or the
      operator passes --operator-allows-rerun-after-partial. GPU: one model resident at a time, at least 6 GiB
      free before each move to the GPU, fp16.

The pure parts import only numpy (through the v3 module); torch and transformers are imported inside the functions
that touch a model, so the unit tests run on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
V3_RUNNER = HERE / "run_realmodel_bridge_v3.py"
NOTE = HERE / "NOTE_zoo_d0_screen_plan_2026_10_12.md"
RECEIPT = HERE / "zoo_d0_screen.json"
REFUSAL_RECEIPT = HERE / "zoo_d0_screen_refusal.json"
PARTIAL_LOG = HERE / "zoo_d0_screen.partial.jsonl"     # append-only; written before each row is printed (NOTE section 12)


def _load_v3():
    """The committed v3 runner, imported by path under its own module name (one instance per process)."""
    name = "run_realmodel_bridge_v3"
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == V3_RUNNER.resolve():
        return mod
    spec = importlib.util.spec_from_file_location(name, str(V3_RUNNER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_v3()
Refusal = R.Refusal

# ---------------------------------------------------------------------------------------------------
# the plan's constants (NOTE sections 3, 6, 7 and 8)
# ---------------------------------------------------------------------------------------------------

SEED_ZOO = 1013
N_ZOO = 48
D0 = R.D0                                   # 16 nominal GPT-NeoX tokens
FAMILIES = R.FAMILIES                       # ("compare", "recall")
VARIANTS = ("plain", "ablated")             # the item and C_abl at D0
READOUT = R.READOUT                         # "argmax_largest", amendment 17.2
SCREEN_BAR = R.SCREEN_BAR                   # 0.40 normalized
GUARD_SLOT_MAX = R.GUARD_SLOT_MAX           # 0.80
ABLATED_FLAG_BAR = R.ABLATED_BAR            # 0.16: C_abl above it is flagged, not gated
ZOO_SLICE = (R.POOL_CAP, R.POOL_CAP + 30_000)   # file-order eligible-sentence indices [150000, 180000): outside v3's pool
N_ZOO_EXEMPLAR_CODES = R.N_EXEMPLAR_CODES   # 12
N_ZOO_TEST_CODES = 60
BATCH = R.BATCH_MAX                         # 16, v3's batches
WRAPPER_EQ_TOL = 0.01                       # max |delta score| (nats) against the committed pilot2 per-item scores
WRAPPER_EQ_TIE = 0.01                       # items whose committed |margin| is below this are exempt from the prediction match
RIGHT_PAD_FP16_TOL = 0.25                   # non-NeoX models: max |delta log p| of the three names, alone vs padded, fp16 on GPU
RIGHT_PAD_FP32_TOL = R.FP32_TOL             # 1e-3, v3's fp32 bar, on the three name log-probs of the fp32 CPU reference (NOTE 12.4)
FP16_FIDELITY_TOL = 0.5                     # max |delta log p| of the three names, fp16 on GPU against the fp32 CPU reference (NOTE 12.4)
RAM_FP32_FACTOR = 1.2                       # the fp32 reference runs only when available RAM >= 1.2 x its fp32 bytes + 1 GiB
RAM_HEADROOM_BYTES = 1024 ** 3
ATTENTION_TYPES = ("gpt_neox", "llama", "qwen2", "gemma2")
NEOX_V3_PATH_TYPES = ("mamba", "gpt_neox")  # loaded through v3's own path: fp32 on CPU, v3's CPU asserts, then half() onto cuda

# every status a model's row can carry; each is named in the plan (NOTE sections 2, 4, 7 and 12.2)
STATUS_SCREENED = "SCREENED"
STATUS_NOT_CACHED = "NOT_CACHED"
STATUS_NOT_SCREENABLE = "NOT_SCREENABLE"
STATUS_FP16_NONFINITE = "NOT_SCREENABLE__fp16_nonfinite"
STATUS_RIGHT_PADDING = "NOT_SCREENABLE__right_padding"
STATUS_FP16_FIDELITY = "NOT_SCREENABLE__fp16_fidelity"
STATUS_FREE_GPU = "NOT_RUN__free_gpu"
STATUS_MEMORY = "NOT_RUN__memory"
STATUS_ERROR = "NOT_RUN__error"
STATUSES = (STATUS_SCREENED, STATUS_NOT_CACHED, STATUS_NOT_SCREENABLE, STATUS_FP16_NONFINITE, STATUS_RIGHT_PADDING,
            STATUS_FP16_FIDELITY, STATUS_FREE_GPU, STATUS_MEMORY, STATUS_ERROR)

# (repo, group, v3 role) in the plan's section-2 order; base before instruct
MODELS = (
    ("state-spaces/mamba-130m-hf", "base", None),
    ("state-spaces/mamba-1.4b-hf", "base", "M1"),
    ("EleutherAI/pythia-1.4b", "base", "T"),
    ("meta-llama/Llama-3.2-1B", "base", None),
    ("meta-llama/Llama-3.2-3B", "base", None),
    ("Qwen/Qwen2.5-1.5B", "base", None),
    ("Qwen/Qwen2.5-3B", "base", None),
    ("google/gemma-2-2b", "base", None),
    ("meta-llama/Llama-3.2-1B-Instruct", "instruct", None),
    ("meta-llama/Llama-3.2-3B-Instruct", "instruct", None),
    ("Qwen/Qwen2.5-1.5B-Instruct", "instruct", None),
    ("Qwen/Qwen2.5-3B-Instruct", "instruct", None),
    ("google/gemma-2-2b-it", "instruct", None),
)
EXCLUDED = {"microsoft/Phi-3.5-mini-instruct": "fp16 weights do not fit beside activations under the 6 GB free-memory rule on "
                                               "the 8,188 MiB card; decision Z2 declined an 8-bit load"}
GROUPS = ("base", "instruct")
GENERATOR_TOKENIZER = R.ARMS["T"]           # GPT-NeoX: the tokenizer that defines the nominal distance (v3's unit)


class NotScreenable(Exception):
    """A per-model assert failed: the row reads the status this carries (NOT_SCREENABLE, or one of its suffixed labels
    in NOTE sections 7 and 12.2) with this message; nothing is re-worded."""

    def __init__(self, reason: str, status: str = STATUS_NOT_SCREENABLE):
        super().__init__(reason)
        self.status = status


class NotRun(Exception):
    """The model could not be run under the plan's memory rules: the row reads the status this carries."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


# ---------------------------------------------------------------------------------------------------
# the zoo's sentence and code slices (NOTE section 3)
# ---------------------------------------------------------------------------------------------------

def zoo_sentence_order(sentences: list, lo: int = ZOO_SLICE[0], hi: int = ZOO_SLICE[1], v3_cap: int = R.POOL_CAP,
                       seed: int = SEED_ZOO) -> tuple:
    """Indices [lo, hi) of the eligible-sentence list, minus every sentence whose sha256 equals that of a sentence in
    v3's pool (indices [0, v3_cap)), minus repeats inside the slice (the earliest kept), reshuffled by the seed.
    Returns (order, info)."""
    if lo < v3_cap:
        raise Refusal(f"the zoo slice starts at {lo}, inside v3's pool of {v3_cap}")
    if len(sentences) < hi:
        raise Refusal(f"the sentence list holds {len(sentences)} sentences; the zoo slice needs {hi}")
    v3_hashes = {R.sha256_text(s) for s in sentences[:v3_cap]}
    keep, seen, dup_v3, dup_in = [], set(), 0, 0
    for i in range(lo, hi):
        h = R.sha256_text(sentences[i])
        if h in v3_hashes:
            dup_v3 += 1
            continue
        if h in seen:
            dup_in += 1
            continue
        seen.add(h)
        keep.append(i)
    perm = np.random.default_rng(seed).permutation(len(keep))
    order = [keep[int(j)] for j in perm]
    return order, {"slice": [lo, hi], "v3_pool": [0, v3_cap], "removed_equal_to_a_v3_pool_sentence": dup_v3,
                   "removed_repeat_inside_slice": dup_in, "kept": len(keep), "reshuffled_by_seed": seed}


def zoo_code_pools(codes: list) -> dict:
    """v3's partition (seed 20261011, unchanged) plus the zoo's pools: the last 72 codes of main_test in the seeded
    order, 12 exemplar and 60 test. main_test is the partition's remainder, so no eligible code lies outside v3's
    regions; these are disjoint from every code a v3 run planted (NOTE section 3)."""
    pools = R.partition_codes(codes)
    k = N_ZOO_EXEMPLAR_CODES + N_ZOO_TEST_CODES
    tail = pools["main_test"][-k:]
    out = dict(pools)
    out["zoo_exemplar"] = list(tail[:N_ZOO_EXEMPLAR_CODES])
    out["zoo_test"] = list(tail[N_ZOO_EXEMPLAR_CODES:])
    planted_by_v3 = set()
    for name in ("main_exemplar", "pilot_exemplar", "pilot_test", "smoke_exemplar", "smoke_test", "pilot2_exemplar", "pilot2_test"):
        planted_by_v3 |= set(pools[name])
    if planted_by_v3 & (set(out["zoo_exemplar"]) | set(out["zoo_test"])):
        raise Refusal("the zoo code pools overlap a v3 reserved pool")
    return out


def prepare_zoo_items(tok_gen, sentences: list, prov: dict, n: int = N_ZOO, seed: int = SEED_ZOO,
                      lo: int = ZOO_SLICE[0], hi: int = ZOO_SLICE[1], v3_cap: int = R.POOL_CAP) -> dict:
    """The zoo's items with v3's generator, in the shape v3's prepare_items returns (so that v3's asserts, sha256
    lists and disjointness assert take it unchanged). One item set, generated with the GPT-NeoX tokenizer."""
    codes, codes_len = R.eligible_codes(tok_gen)
    pools = zoo_code_pools(codes)
    order, slice_info = zoo_sentence_order(sentences, lo, hi, v3_cap, seed)
    pool = R.SentencePool(sentences, order)
    exemplars = R.build_exemplars(pools["zoo_exemplar"], np.random.default_rng(seed))
    cells = {D0: R.build_items(n, D0, seed, pool, pools["zoo_test"], exemplars, tok_gen)}
    return {"cells": cells, "exemplars": exemplars, "pool": pool, "codes_len": codes_len, "code_pool_size": len(codes),
            "pools": {k: len(v) for k, v in pools.items()}, "code_pools": pools, "sentence_provenance": prov,
            "sentence_region": {"which": "zoo", "seed": seed,
                                "definition": f"eligible sentences at file-order indices [{lo}, {hi}), minus every sentence equal "
                                              f"(sha256) to one of v3's pool [0, {v3_cap}), minus repeats inside the slice, "
                                              f"reshuffled by seed {seed}", **slice_info},
            "region_indices": set(order), "n_sentences": len(sentences), "sentences": sentences}


def v3_pool_view(sentences: list, prov: dict, code_pools: dict, v3_cap: int = R.POOL_CAP) -> dict:
    """What v3's prior_sentences_and_codes needs to map the committed receipts: v3's pool (indices [0, v3_cap)) with
    the provenance v3's pool-identity assert compares (parquet sha256s, cap, eligible count)."""
    if len(sentences) < v3_cap:
        raise Refusal(f"the sentence list holds {len(sentences)} sentences, fewer than v3's pool of {v3_cap}")
    pv = dict(prov)
    pv["pool_cap"] = v3_cap
    pv["eligible_sentences"] = v3_cap
    return {"sentences": sentences[:v3_cap], "sentence_provenance": pv, "code_pools": code_pools}


def committed_receipts() -> dict:
    return {"smoke": R.read_receipt(R.SMOKE_RECEIPT), "pilot": R.read_receipt(R.PILOT1_RECEIPT),
            "pilot2": R.read_receipt(R.PILOT2_RECEIPT)}


def zoo_disjointness(prep: dict, lists: dict, view: dict, receipts: dict) -> dict:
    """NOTE section 3: v3's assert against the committed receipts' lists, plus the zoo's by-construction facts."""
    absent = [k for k, v in receipts.items() if v is None]
    if absent:
        raise Refusal(f"the committed v3 receipts are needed for the disjointness assert; absent: {absent}")
    prior = R.prior_sentences_and_codes(view, receipts)
    out = {"against_committed_receipts": R.assert_disjoint_from_prior(prep, lists, prior)}
    lo, hi = prep["sentence_region"]["slice"]
    used = list(prep["pool"].used)
    if any(not (lo <= i < hi) for i in used):
        raise Refusal("a zoo sentence lies outside the zoo slice")
    v3_hashes = {R.sha256_text(s) for s in view["sentences"]}
    if set(lists["sentences"]) & v3_hashes:
        raise Refusal("a zoo sentence equals a sentence of v3's pool")
    pools = prep["code_pools"]
    zoo_codes = set(pools["zoo_exemplar"]) | set(pools["zoo_test"])
    reserved = set()
    for name in ("main_exemplar", "pilot_exemplar", "pilot_test", "smoke_exemplar", "smoke_test", "pilot2_exemplar", "pilot2_test"):
        reserved |= set(pools[name])
    planted = {c for items in prep["cells"].values() for it in items for c in it["codes"]} | {c for e in prep["exemplars"] for c in e["codes"]}
    if not planted <= zoo_codes or planted & reserved:
        raise Refusal("a planted zoo code lies outside the zoo pools or inside a v3 reserved pool")
    out["by_construction"] = {
        "sentences": f"every sentence drawn has a file-order index in [{lo}, {hi}), outside v3's pool [0, {R.POOL_CAP}) and so "
                     f"outside its main, pilot2, smoke and pilot regions; none equals (sha256) any sentence of v3's pool",
        "codes": "the zoo's 72 codes are the last 72 of v3's main_test pool in the seeded order: disjoint from v3's main_exemplar, "
                 "pilot, smoke and pilot2 pools by the partition, and from every code a v3 run planted (asserted above by "
                 "sha256); main_test is the partition's remainder, so no eligible code lies outside v3's regions, and v3's "
                 "main region is closed (format line NOT_STARTED, final) with no main-set code ever planted",
        "sentences_used": len(used), "sentence_indices_min_max": [min(used), max(used)] if used else None,
        "codes_planted": len(planted)}
    return out


# ---------------------------------------------------------------------------------------------------
# the pilot2 regeneration (the wrapper's reference; NOTE section 6)
# ---------------------------------------------------------------------------------------------------

def regenerate_pilot2(tok_gen, v3_sentences: list, rec: dict) -> dict:
    """v3's prepare_items at seed 1012 on v3's pool, asserted byte-identical to the committed pilot2 receipt by its
    prompt sha256 lists and its items_sha256."""
    if rec is None:
        raise Refusal("the committed pilot2 receipt is needed for the wrapper's equivalence check")
    prep2 = R.prepare_items(tok_gen, R.N_PILOT, (R.D0,), R.SEED_PILOT2, "pilot2", sentences=v3_sentences)
    lists = R.hash_lists(prep2)
    want = rec["items"]["sha256"]["prompts"]
    for family in FAMILIES:
        for variant in VARIANTS:
            if lists["prompts"][family][str(D0)][variant] != want[family][str(D0)][variant]:
                raise Refusal(f"the regenerated pilot2 {family}/{variant} prompts differ from the committed receipt's sha256 list")
    digest = R.items_digest(prep2["cells"])
    if digest != rec["redteam"]["items_sha256"]:
        raise Refusal("the regenerated pilot2 items differ from the committed receipt's items_sha256")
    return {"prep": prep2, "items_sha256": digest, "prompts_matched": sum(len(want[f][str(D0)][v]) for f in FAMILIES for v in VARIANTS)}


# ---------------------------------------------------------------------------------------------------
# per-model asserts (NOTE section 4) and family facts (NOTE section 5)
# ---------------------------------------------------------------------------------------------------

def cache_status(repo: str) -> tuple:
    """(ok, reason, snapshot). ok iff refs/main names a snapshot holding config.json, a tokenizer file and every
    safetensors shard its index names."""
    snap = R.snapshot_dir(repo)
    if snap is None:
        return False, "no cached snapshot (refs/main or a single snapshot directory)", None
    if not (snap / "config.json").exists():
        return False, "config.json missing from the cached snapshot", snap
    if not any((snap / f).exists() for f in ("tokenizer.json", "tokenizer.model")):
        return False, "no tokenizer file in the cached snapshot", snap
    index = snap / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        missing = [s for s in shards if not (snap / s).exists()]
        if missing:
            return False, f"safetensors shards missing: {missing}", snap
    elif not (snap / "model.safetensors").exists():
        return False, "no safetensors weights in the cached snapshot", snap
    return True, "cached", snap


def bos_prefix(tok) -> list:
    """The ids the tokenizer adds by default in front of a text (add_special_tokens=True), asserted constant over two
    probes and with nothing added at the end; [] when it adds nothing."""
    out = None
    for probe in ("The sigma code is 4271.", "Question: Which name has the largest code: alpha, gamma or delta? Answer:"):
        plain = tok.encode(probe)
        full = list(tok.hf.encode(probe, add_special_tokens=True))
        if len(full) < len(plain) or full[len(full) - len(plain):] != plain:
            raise NotScreenable(f"the tokenizer's default special tokens alter or follow the text ({full[-3:]} vs {plain[-3:]})")
        pre = full[:len(full) - len(plain)]
        if out is not None and pre != out:
            raise NotScreenable(f"the tokenizer's default prefix is not constant ({out} vs {pre})")
        out = pre
    return out


def model_prompt_ids(tok, text: str, bos: list) -> list:
    """The wrapper's ids: the model's own tokenization of the whole prompt text, after its default BOS (if any)."""
    return list(bos) + tok.encode(text)


def _stats(xs: list) -> dict:
    return {"min": int(min(xs)), "mean": round(float(np.mean(xs)), 2), "max": int(max(xs))}


def model_tokenizer_asserts(tok, prep: dict, bos: list) -> dict:
    """NOTE section 4, items 1-5, on the one item set. Returns the ids the scoring uses (private) and the record.
    Raises NotScreenable with the failed assert's message."""
    items, exemplars = prep["cells"][D0], prep["exemplars"]
    try:
        test_ids = R.name_token_ids(tok, R.NAMES)
        ex_ids = R.name_token_ids(tok, R.EXEMPLAR_NAMES)
    except Refusal as e:
        raise NotScreenable(f"name tokens: {e}")
    code_lens = {len(tok.encode(" " + c)) for it in items for c in it["codes"]}
    if len(code_lens) != 1:
        raise NotScreenable(f"the planted codes have several token lengths in this tokenizer: {sorted(code_lens)}")
    L = code_lens.pop()
    seqs, cands, lengths, sha, checks = {}, [], {}, {}, 0
    for it in items:
        cands.append([tok.encode(" " + c) for c in it["codes"]])
    for family in FAMILIES:
        for variant in VARIANTS:
            key = f"{family}/{variant}"
            seqs[key], lengths[key], sha[key] = [], [], []
            for k, it in enumerate(items):
                text = R.prompt_text(it, exemplars, family, variant)
                base = tok.encode(text)
                if family == "compare":
                    for nm in R.NAMES:
                        if tok.encode(text + " " + nm) != base + [test_ids[nm]]:
                            raise NotScreenable(f"answer token in context: ' {nm}' after the {key} prompt of item {k} is not the single id "
                                                f"{test_ids[nm]} appended to the prompt's ids")
                        checks += 1
                else:
                    for slot, c in enumerate(it["codes"]):
                        if tok.encode(text + " " + c) != base + cands[k][slot]:
                            raise NotScreenable(f"recall candidate in context: ' {c}' after the {key} prompt of item {k} does not "
                                                f"tokenize as the prompt's ids followed by the ids of ' {c}'")
                        checks += 1
                ids = list(bos) + base
                if len(ids) + (L if family == "recall" else 0) > R.MAX_PROMPT_TOKENS:
                    raise NotScreenable(f"a {key} prompt reaches {len(ids) + L} tokens, above {R.MAX_PROMPT_TOKENS}")
                seqs[key].append(ids)
                lengths[key].append(len(ids))
                sha[key].append(R.sha256_text(text))
    filler_native = [len(tok.encode(" " + it["filler"])) for it in items]
    longest = max(max(v) for v in lengths.values()) + L
    return {"_seqs": seqs, "_cands": cands, "_name_ids": [test_ids[nm] for nm in R.NAMES],
            "record": {"tokenizer_class": type(tok.hf).__name__, "vocab": len(tok), "bos_prefix": list(bos),
                       "name_token_ids": test_ids, "exemplar_name_token_ids": ex_ids, "code_token_length": L,
                       "in_context_checks": checks, "prompt_tokens": {k: _stats(v) for k, v in lengths.items()},
                       "prompt_tokens_per_item": lengths, "filler_tokens_native": _stats(filler_native),
                       "filler_tokens_native_per_item": filler_native,
                       "filler_tokens_nominal_neox": D0, "longest_prompt_with_candidate": longest},
            "_prompt_sha256": sha}


def _safetensors_files(snap: Path) -> list:
    index = snap / "model.safetensors.index.json"
    if index.exists():
        return [snap / s for s in sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))]
    return [snap / "model.safetensors"]


def family_fact(snap: Path) -> dict:
    """NOTE section 5: the family from config and the checkpoint's tensors, never from the repo name. Decay SSM: model_type
    'mamba' and every layer's A = -exp(A_log), read from the safetensors on CPU, real and strictly negative. Attention:
    model_type in ATTENTION_TYPES, attention projections in every layer and no A_log. The reach is checked against the
    longest native prompt by family_reach_ok once the prompts are tokenized."""
    from safetensors import safe_open
    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    mt = cfg.get("model_type")
    n_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    keys, alog = [], {}
    for f in _safetensors_files(snap):
        with safe_open(str(f), framework="numpy") as fh:
            ks = list(fh.keys())
            keys += ks
            if mt == "mamba":
                for k in ks:
                    if k.endswith("mixer.A_log"):
                        alog[k] = fh.get_tensor(k)
    def layer_of(k: str) -> int:
        return int(re.search(r"layers\.(\d+)\.", k).group(1))
    if mt == "mamba":
        if not n_layers or sorted(layer_of(k) for k in alog) != list(range(n_layers)):
            raise NotScreenable(f"family not asserted: {len(alog)} A_log tensors for num_hidden_layers {n_layers}")
        worst, lowest, dtypes = -np.inf, np.inf, set()
        for k, a in alog.items():
            if np.iscomplexobj(a):
                raise NotScreenable(f"family not asserted: {k} is complex")
            A = -np.exp(a.astype(np.float64))
            if not bool((A < 0).all()):
                raise NotScreenable(f"family not asserted: {k} gives A with max {float(A.max())}, not strictly negative")
            worst, lowest = max(worst, float(A.max())), min(lowest, float(A.min()))
            dtypes.add(str(a.dtype))
        return {"family": "DECAY_SSM", "model_type": mt, "layers": n_layers, "A_log_tensors": len(alog), "max_A": worst, "min_A": lowest,
                "A_log_dtypes": sorted(dtypes), "source": "config.json and the snapshot's safetensors, read on CPU (A = -exp(A_log))",
                "reach_tokens": None}
    if mt in ATTENTION_TYPES:
        if any(k.endswith("A_log") for k in keys):
            raise NotScreenable("family not asserted: an attention config whose checkpoint carries A_log tensors")
        pat = re.compile(r"layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj)|attention\.query_key_value)\.weight$")
        attn_layers = sorted({int(m.group(1)) for k in keys for m in [pat.search(k)] if m})
        if not n_layers or attn_layers != list(range(n_layers)):
            raise NotScreenable(f"family not asserted: attention projections in {len(attn_layers)} of {n_layers} layers")
        reach = cfg.get("max_position_embeddings")
        window = None
        if mt == "gemma2":
            window = cfg.get("sliding_window")
        elif mt == "qwen2" and cfg.get("use_sliding_window"):
            window = cfg.get("sliding_window")
        reach_tokens = min(x for x in (reach, window) if x is not None) if (reach or window) else None
        return {"family": "ATTENTION", "model_type": mt, "layers": n_layers, "attention_layers": len(attn_layers),
                "max_position_embeddings": reach, "sliding_window_in_use": window, "use_sliding_window": cfg.get("use_sliding_window"),
                "reach_tokens": reach_tokens, "source": "config.json and the snapshot's safetensors tensor names"}
    raise NotScreenable(f"family not asserted: model_type {mt!r} is neither mamba nor one of {ATTENTION_TYPES}")


def family_reach_ok(fact: dict, longest: int) -> None:
    if fact["family"] == "ATTENTION" and (fact["reach_tokens"] is None or fact["reach_tokens"] < longest):
        raise NotScreenable(f"family not asserted: attention reach {fact['reach_tokens']} below the longest native prompt {longest}")


def prepare_model_row(repo: str, group: str, role, prep: dict, tok_loader=None) -> dict:
    """Everything a model's row needs before it is loaded: cache, family fact, tokenizer asserts. Never raises for a
    per-model failure: the row carries the status and reason."""
    row = {"repo": repo, "group": group, "v3_role": role, "status": "PENDING", "reason": None}
    ok, why, snap = cache_status(repo)
    row["revision"] = snap.name if snap is not None else None
    if not ok:
        row.update(status=STATUS_NOT_CACHED, reason=why)
        return row
    try:
        row["family"] = family_fact(snap)
        tok = (tok_loader or R.load_tokenizer)(repo)
        bos = bos_prefix(tok)
        asserted = model_tokenizer_asserts(tok, prep, bos)
        family_reach_ok(row["family"], asserted["record"]["longest_prompt_with_candidate"])
    except NotScreenable as e:
        row.update(status=e.status, reason=str(e))
        return row
    row["tokenizer"] = asserted["record"]
    row["_seqs"], row["_cands"], row["_name_ids"] = asserted["_seqs"], asserted["_cands"], asserted["_name_ids"]
    row["_prompt_sha256"] = asserted["_prompt_sha256"]
    row["_tok"] = tok
    row["status"] = "READY"
    return row


def assert_identical_bytes(rows: list, lists: dict) -> dict:
    """Codes, filler and template text identical across models: every ready model's prompt sha256 list equals the list
    of the one generated item set."""
    n = 0
    for row in rows:
        if row["status"] != "READY":
            continue
        for family in FAMILIES:
            for variant in VARIANTS:
                if row["_prompt_sha256"][f"{family}/{variant}"] != lists["prompts"][family][str(D0)][variant]:
                    raise Refusal(f"{row['repo']}: its {family}/{variant} prompt bytes differ from the generated item set")
        n += 1
    return {"models_checked": n, "prompts_per_model": 2 * 2 * N_ZOO, "identical": True}


# ---------------------------------------------------------------------------------------------------
# the wrapper around v3's forward readouts (NOTE section 6)
# ---------------------------------------------------------------------------------------------------

class SoftcapHead:
    """The output head as v3's _logits reads it (.weight for the dtype cast, then a call), with the config's final logit
    softcapping applied after it exactly as transformers' Gemma2ForCausalLM.forward does: x / cap, tanh, * cap."""

    def __init__(self, head, cap):
        self.head = head
        self.weight = head.weight
        self.cap = cap

    def __call__(self, h):
        out = self.head(h)
        if self.cap is not None:
            torch = R._torch()
            out = out / self.cap
            out = torch.tanh(out)
            out = out * self.cap
        return out


class AttentionAdapter:
    """What v3's attention path ("T") reads from a model: .gpt_neox (the decoder, called with input_ids, attention_mask
    and use_cache, returning last_hidden_state), .lm_head (the head) and .parameters(). Llama, Qwen2 and Gemma2 reach
    v3's readouts through this; nothing of v3 is modified."""

    def __init__(self, model):
        self.model = model
        self.gpt_neox = model.base_model
        self.lm_head = SoftcapHead(model.get_output_embeddings(), getattr(model.config, "final_logit_softcapping", None))

    def parameters(self):
        return self.model.parameters()


def v3_path(model_type: str) -> str:
    """Which of v3's two forward paths a model takes: 'M1' reads model.backbone, 'T' reads model.gpt_neox."""
    return "M1" if model_type == "mamba" else "T"


def as_v3_model(model, model_type: str):
    return model if model_type in NEOX_V3_PATH_TYPES else AttentionAdapter(model)


def score_cell(path: str, model, family: str, seqs: list, targets: list, name_ids=None, cands=None, batch: int = BATCH) -> dict:
    """v3's run_cell with the ids supplied per model: compare = v3's next_token_scores_batch over the three name ids at
    each item's last position; recall = v3's recall_loglik_batch, one pass per candidate slot; batches in item order;
    summary = v3's argmax_cell_summary. Scores must be finite."""
    t0 = time.time()
    n = len(seqs)
    scores = np.zeros((n, 3), dtype=float)
    n_batches = 0
    if family == "compare":
        for i in range(0, n, batch):
            scores[i:i + batch] = R.next_token_scores_batch(path, model, seqs[i:i + batch], name_ids)
            n_batches += 1
    else:
        for slot in range(3):
            cs = [c[slot] for c in cands]
            for i in range(0, n, batch):
                scores[i:i + batch, slot] = R.recall_loglik_batch(path, model, seqs[i:i + batch], cs[i:i + batch])
                n_batches += 1
    if not np.isfinite(scores).all():
        raise NotScreenable(f"non-finite {family} scores in fp16 ({int((~np.isfinite(scores)).sum())} of {scores.size})",
                            STATUS_FP16_NONFINITE)
    s = R.argmax_cell_summary(scores, targets)
    s["seconds"] = round(time.time() - t0, 2)
    s["batches"] = n_batches
    s["batch_size"] = batch
    s["max_seq_len"] = max(len(x) for x in seqs) + (len(cands[0][0]) if family == "recall" else 0)
    return s


def score_cell_with_backoff(path, model, family, seqs, targets, name_ids=None, cands=None, batch: int = BATCH) -> dict:
    """A CUDA out-of-memory error reruns the whole cell at half the batch, down to 1 (NOTE section 7)."""
    torch = R._torch()
    b = batch
    while True:
        try:
            return score_cell(path, model, family, seqs, targets, name_ids, cands, b)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b == 1:
                raise NotRun(STATUS_MEMORY, f"CUDA out of memory at batch 1 in the {family} cell")
            b = max(1, b // 2)


def cell_targets(items: list, family: str) -> list:
    return [it["max_slot"] for it in items] if family == "compare" else [it["q_slot"] for it in items]


def equivalence_against_reference(per_item: dict, ref: dict, tol: float = WRAPPER_EQ_TOL, tie: float = WRAPPER_EQ_TIE) -> dict:
    """One cell of the wrapper's per-item output against the committed receipt's: every prediction equal (items whose
    committed |margin| < tie exempt and counted) and max |delta score| <= tol."""
    s = np.asarray(per_item["scores"], dtype=float)
    r = np.asarray(ref["scores"], dtype=float)
    if s.shape != r.shape:
        raise Refusal(f"wrapper output shape {s.shape} differs from the reference's {r.shape}")
    if list(per_item["target_slot"]) != list(ref["target_slot"]):
        raise Refusal("the wrapper's targets differ from the reference's")
    pred, rpred = np.asarray(per_item["pred_slot"]), np.asarray(ref["pred_slot"])
    exempt = np.abs(np.asarray(ref["margin"], dtype=float)) < tie
    disagree = (pred != rpred) & ~exempt
    dmax = float(np.abs(s - r).max()) if s.size else 0.0
    return {"n": int(len(pred)), "max_abs_score_diff": dmax, "pred_agree": int((pred == rpred).sum()),
            "pred_disagree_non_tie": int(disagree.sum()), "exempt_near_ties": int(exempt.sum()),
            "ok": bool(dmax <= tol and not disagree.any())}


def wrapper_equivalence(arm: str, path: str, model, tok, prep2: dict, rec: dict) -> dict:
    """NOTE section 6: M1 or T scores the regenerated pilot2 items through the wrapper (the four cells the receipt
    carries per item) and must reproduce the committed per-item outputs. Only deviations are kept."""
    items, exemplars = prep2["cells"][D0], prep2["exemplars"]
    name_ids = [rec["redteam"]["name_token_ids"][arm]["test"][nm] for nm in R.NAMES]
    out = {}
    for family in FAMILIES:
        for variant in VARIANTS:
            seqs = [model_prompt_ids(tok, R.prompt_text(it, exemplars, family, variant), []) for it in items]
            for it, s in zip(items, seqs):
                if s != R.compositional_ids(tok, R.prompt_parts(it, exemplars, family, variant)):
                    raise Refusal(f"{arm}: the wrapper's ids differ from v3's compositional_ids on a pilot2 {family}/{variant} prompt")
            cands = [[tok.encode(" " + c) for c in it["codes"]] for it in items] if family == "recall" else None
            s = score_cell(path, model, family, seqs, cell_targets(items, family), name_ids, cands)
            out[f"{family}/{variant}"] = equivalence_against_reference(s["_per_item"], rec["per_item"][arm][family][str(D0)][variant])
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def equivalence_log_line(arm: str, eq: dict) -> str:
    """The printed line of one arm's wrapper equivalence: the four scored cells by name. The arm's load record and its
    ok flag sit beside the cells in the same dict and are not cells (the run at acd5b1e9 iterated every key, met the
    load record, and stopped with a KeyError after M1's equivalence had passed and been logged)."""
    cells = ", ".join(f"{k} max|d| {eq[k]['max_abs_score_diff']:.2e} disagree {eq[k]['pred_disagree_non_tie']}"
                      for k in (f"{family}/{variant}" for family in FAMILIES for variant in VARIANTS))
    return f"  wrapper equivalence {arm}: {cells} -> {'ok' if eq['ok'] else 'FAILED'}"


def alone_name_logprobs(path: str, model, seqs: list, name_ids: list) -> list:
    """Each item's three name log-probs at its last position, each item run alone (batch of one)."""
    torch = R._torch()
    device = next(model.parameters()).device
    idx = torch.tensor(list(name_ids), device=device)
    out = []
    with torch.no_grad():
        for s in seqs:
            ids, _ = R.pad_batch([s], device)
            h = R._hidden(path, model, ids)
            out.append([float(x) for x in torch.log_softmax(R._logits(model, h[0, len(s) - 1]), -1)[idx]])
    return out


def right_padding_ids(path: str, model, seqs: list, name_ids: list, extra: int = 24) -> dict:
    """Two items' compare readout alone and inside a right-padded batch (padded by `extra` beyond the longer), on the
    model's own ids and device: max |delta logit| over the vocabulary (reported) and max |delta log p| over the three
    names (gated for the non-NeoX models at RIGHT_PAD_FP16_TOL in fp16 on the GPU and at RIGHT_PAD_FP32_TOL in the fp32
    CPU reference). The alone name log-probs are returned for the fp16-against-fp32 comparison (NOTE section 12.4)."""
    torch = R._torch()
    device = next(model.parameters()).device
    with torch.no_grad():
        alone = []
        for s in seqs:
            ids, _ = R.pad_batch([s], device)
            h = R._hidden(path, model, ids)
            alone.append(R._logits(model, h[0, len(s) - 1]))
        ids, lengths = R.pad_batch(seqs, device, extra=extra)
        h = R._hidden(path, model, ids)
        d_logit, d_name, agree, lp_alone = [], [], [], []
        idx = torch.tensor(list(name_ids), device=device)
        for i, n in enumerate(lengths):
            b = R._logits(model, h[i, n - 1])
            d_logit.append(float((b - alone[i]).abs().max()))
            lb, la = torch.log_softmax(b, -1)[idx], torch.log_softmax(alone[i], -1)[idx]
            d_name.append(float((lb - la).abs().max()))
            agree.append(bool(int(lb.argmax()) == int(la.argmax())))
            lp_alone.append([float(x) for x in la])
    return {"max_abs_logit_diff": max(d_logit), "max_abs_name_logprob_diff": max(d_name), "argmax_agree": agree,
            "per_item_logit": d_logit, "per_item_name_logprob": d_name, "alone_name_logprobs": lp_alone,
            "prompt_tokens": [len(s) for s in seqs], "dtype": str(next(model.parameters()).dtype), "device": str(device)}


# ---------------------------------------------------------------------------------------------------
# the fp16 load and the fp32 CPU reference (NOTE sections 12.1 and 12.4)
# ---------------------------------------------------------------------------------------------------

def buffer_dtypes(model) -> dict:
    """The dtype of every rotary inv_freq buffer by name, and the count of floating buffers per dtype: what a load or a
    cast did to the buffers (transformers' own fp16 load keeps inv_freq in fp32; a half() casts it)."""
    inv, counts = {}, {}
    for name, b in model.named_buffers():
        if not b.is_floating_point():
            continue
        counts[str(b.dtype)] = counts.get(str(b.dtype), 0) + 1
        if name.endswith("inv_freq"):
            inv[name] = str(b.dtype)
    return {"inv_freq": inv, "floating_buffers_by_dtype": counts}


def stored_numel(snap: Path) -> int:
    """The number of stored parameters, from the snapshot's safetensors headers (no tensor is read)."""
    total = 0
    for f in _safetensors_files(snap):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n).decode("utf-8"))
        for k, v in header.items():
            if k != "__metadata__":
                total += int(np.prod(v["shape"], dtype=np.int64)) if v["shape"] else 1
    return total


def available_ram_bytes():
    """Available physical memory in bytes, or None when it cannot be read."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:                                                 # noqa: BLE001 -- unknown means the reference is not run
        return None


def load_hf_cpu(snap: Path, model_type: str, dtype_name: str):
    """transformers' own load from the cached snapshot onto the CPU in fp16 (for scoring) or fp32 (the reference); gemma-2
    with eager attention. Nothing is cast after the load (NOTE section 12.1): every parameter is asserted to carry the
    dtype, and the buffers keep the dtype transformers gives them, recorded."""
    torch = R._torch()
    from transformers import AutoModelForCausalLM
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    kwargs = {"dtype": dtype, "local_files_only": True}
    if model_type == "gemma2":
        kwargs["attn_implementation"] = "eager"                     # sdpa drops attn_logit_softcapping in transformers 4.57.3
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(snap), **kwargs)
    model.eval()
    bad = sorted({str(p.dtype) for p in model.parameters()} - {str(dtype)})
    if bad:
        model = None
        raise NotScreenable(f"parameters not {dtype_name} after the load: {bad}")
    info = {"dtype_loaded": dtype_name, "load_seconds_cpu": round(time.time() - t0, 2),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "final_logit_softcapping": getattr(model.config, "final_logit_softcapping", None),
            "attn_logit_softcapping": getattr(model.config, "attn_logit_softcapping", None),
            "buffers_after_load": buffer_dtypes(model)}
    return model, info


def fp32_reference(snap: Path, model_type: str, seqs2: list, name_ids: list, ram=None) -> dict:
    """NOTE section 12.4: for a Llama, Qwen or Gemma row, before its fp16 load. When available RAM is at least
    RAM_FP32_FACTOR x the fp32 bytes + RAM_HEADROOM_BYTES, the checkpoint is loaded in fp32 on the CPU (the only model
    resident), right padding is measured on the two prompts (fp32_reference_gate applies RIGHT_PAD_FP32_TOL to the
    three name log-probs), the alone name log-probs are kept as the reference, and the model is dropped. When RAM is
    short or unknown, the reference is not run and the record says why; nothing else changes."""
    need = int(stored_numel(snap) * 4 * RAM_FP32_FACTOR + RAM_HEADROOM_BYTES)
    avail = (ram or available_ram_bytes)()
    out = {"ran": False, "ram_needed_bytes": need, "ram_available_bytes": avail}
    if avail is None or avail < need:
        out["reason"] = ("available RAM could not be read" if avail is None else
                         f"available RAM {avail / 2**30:.2f} GiB is under the {need / 2**30:.2f} GiB the fp32 reference needs")
        return out
    import gc
    model = None
    try:
        model, info = load_hf_cpu(snap, model_type, "float32")
        rp = right_padding_ids(v3_path(model_type), as_v3_model(model, model_type), seqs2, name_ids)
    finally:
        model = None
        gc.collect()
    out.update(ran=True, load=info, right_padding_fp32_cpu=rp)
    return out


def fp32_reference_gate(ref: dict) -> None:
    """NOT_SCREENABLE__right_padding when the fp32 reference ran and right padding moved a name log-prob by more than
    RIGHT_PAD_FP32_TOL (v3's fp32 bar). The caller keeps the reference's record either way."""
    if ref.get("ran") and ref["right_padding_fp32_cpu"]["max_abs_name_logprob_diff"] > RIGHT_PAD_FP32_TOL:
        d = ref["right_padding_fp32_cpu"]["max_abs_name_logprob_diff"]
        raise NotScreenable(f"right padding moves the name log-probs by {d:.2e} > {RIGHT_PAD_FP32_TOL} in the fp32 CPU reference",
                            STATUS_RIGHT_PADDING)


def fp16_fidelity(ref: list, got: list, tol: float | None = None) -> dict:
    """The fp16 GPU name log-probs against the fp32 CPU reference on the same prompts (NOTE section 12.4)."""
    tol = FP16_FIDELITY_TOL if tol is None else tol
    r, g = np.asarray(ref, dtype=float), np.asarray(got, dtype=float)
    if r.shape != g.shape or r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"fp16 fidelity: shapes {g.shape} and {r.shape} do not match as [items, 3]")
    d = np.abs(g - r)
    return {"max_abs_name_logprob_diff": float(d.max()), "per_item": [float(x) for x in d.max(axis=1)],
            "argmax_agree": [bool(a == b) for a, b in zip(g.argmax(axis=1), r.argmax(axis=1))], "tol": tol,
            "ok": bool(float(d.max()) <= tol)}


# ---------------------------------------------------------------------------------------------------
# the screen rule (v3's arm_clears) and the summary (NOTE sections 8 and 9)
# ---------------------------------------------------------------------------------------------------

def screen_from_cells(cells: dict) -> dict:
    """cells[(family, variant)] = summary. The screen is v3's arm_clears on a receipt-shaped view of this model's D0
    cells; C_abl is reported beside it with the >0.16 flag."""
    key = "this_model"
    view = {"acc_norm": {key: {f: {str(D0): cells[(f, "plain")]["acc_norm"]} for f in FAMILIES}},
            "guard": {key: {f: {str(D0): {"guard_ok": cells[(f, "plain")]["guard_ok"]}} for f in FAMILIES}}}
    sc = R.arm_clears(view, key)
    for f in FAMILIES:
        a = cells[(f, "ablated")]["acc_norm"]
        sc[f]["acc_raw"] = cells[(f, "plain")]["acc_raw"]
        sc[f]["max_slot_fraction"] = cells[(f, "plain")]["max_slot_fraction"]
        sc[f]["ablated_acc_norm"] = a
        sc[f]["ablated_above_bar"] = bool(a > ABLATED_FLAG_BAR)
    return sc


def group_sentence(group: str, m: int, n_c: int, f_c: int, n_r: int, f_r: int) -> str:
    """The printable sentence of one group (NOTE section 9 as corrected by section 12.3): the counts are of the M
    screened models of that group, never of the cached models; a zero count reads "none of the M"; beside each nonzero
    pass count, how many of those passes carry the C_abl flag; the instruct group's sentence says it saw raw text."""
    tail = f"n = {N_ZOO} items per family per model, a pilot-class screen that licenses nothing."
    if m == 0:
        return f"At D0, no {group} model was screened (each {group} row is listed with its status and reason); {tail}"
    who = f"screened {group} models" + (", given raw text without a chat template," if group == "instruct" else "")
    count = lambda k: f"none of the {m}" if k == 0 else f"{k} of {m}"
    flags = [f"{f} of the {k} {fam} passes" for k, f, fam in ((n_c, f_c, "compare"), (n_r, f_r, "recall")) if k > 0]
    flag_clause = f" (C_abl above {ABLATED_FLAG_BAR} in " + " and ".join(flags) + ")" if flags else ""
    return (f"At D0, under v3's argmax readout in fp16, {count(n_c)} {who} could perform the compare task and {count(n_r)} the "
            f"recall task{flag_clause}; {tail}")


def summarize(rows: list) -> dict:
    out = {}
    for group in GROUPS:
        g = [r for r in rows if r["group"] == group]
        screened = [r for r in g if r["status"] == STATUS_SCREENED]
        m = len(screened)
        nc = [r for r in screened if r["screen"]["compare"]["clears"]]
        nr = [r for r in screened if r["screen"]["recall"]["clears"]]
        both = [r for r in screened if r["screen"]["clears"]]
        flag = lambda rs, f: sum(1 for r in rs if r["screen"][f]["ablated_above_bar"])
        by_family = {}
        for fam in ("DECAY_SSM", "ATTENTION"):
            s = [r for r in screened if r["family"]["family"] == fam]
            by_family[fam] = {"screened": len(s), "compare": sum(1 for r in s if r["screen"]["compare"]["clears"]),
                              "recall": sum(1 for r in s if r["screen"]["recall"]["clears"]),
                              "both": sum(1 for r in s if r["screen"]["clears"])}
        out[group] = {
            "M_screened": m, "N_compare": len(nc), "N_recall": len(nr), "N_both": len(both),
            "N_compare_with_ablated_flag": flag(nc, "compare"), "N_recall_with_ablated_flag": flag(nr, "recall"),
            "compare_passes": [r["repo"] for r in nc], "recall_passes": [r["repo"] for r in nr], "clear_both": [r["repo"] for r in both],
            "by_family": by_family,
            "not_screened": [{"repo": r["repo"], "status": r["status"], "reason": r["reason"]} for r in g if r["status"] != STATUS_SCREENED],
            "sentence": group_sentence(group, m, len(nc), flag(nc, "compare"), len(nr), flag(nr, "recall"))}
    out["excluded"] = [{"repo": k, "reason": v} for k, v in EXCLUDED.items()]
    return out


# ---------------------------------------------------------------------------------------------------
# git facts and the CPU phase
# ---------------------------------------------------------------------------------------------------

def git_blob_facts(path: Path) -> dict:
    out = {"path": path.relative_to(ROOT).as_posix(), "head_blob_sha256": None, "working_copy_matches_head": None,
           "working_copy_sha256": R.sha256_file(path) if path.exists() else None}
    try:
        blob = subprocess.run(["git", "show", f"HEAD:{out['path']}"], cwd=str(ROOT), capture_output=True, timeout=30)
        if blob.returncode == 0:
            out["head_blob_sha256"] = hashlib.sha256(blob.stdout).hexdigest()
            out["working_copy_matches_head"] = blob.stdout.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n")
    except (OSError, subprocess.TimeoutExpired) as e:
        out["error"] = str(e)
    return out


def git_facts() -> dict:
    head = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        head = r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    out = {"head": head}
    for k, path in committed_files():
        out[k] = git_blob_facts(path)
    return out


def committed_files() -> tuple:
    """What the run reads and must find committed and unchanged at HEAD: the plan, both runners, and the three v3
    receipts that supply the disjointness lists and the wrapper's per-item reference (NOTE section 12.6)."""
    return (("note", NOTE), ("runner", Path(__file__).resolve()), ("v3_runner", V3_RUNNER),
            ("receipt_smoke", R.SMOKE_RECEIPT), ("receipt_pilot", R.PILOT1_RECEIPT), ("receipt_pilot2", R.PILOT2_RECEIPT))


COMMITTED_KEYS = ("note", "runner", "v3_runner", "receipt_smoke", "receipt_pilot", "receipt_pilot2")


def require_committed(git: dict) -> None:
    for k in COMMITTED_KEYS:
        f = git.get(k)
        if f is None:
            raise Refusal(f"no git facts for {k}: the screen runs only on committed text")
        if f["head_blob_sha256"] is None or not f["working_copy_matches_head"]:
            raise Refusal(f"{f['path']} is not committed, or differs from HEAD: the screen runs only on committed text")


# ---------------------------------------------------------------------------------------------------
# the append-only partial log (NOTE section 12.5)
# ---------------------------------------------------------------------------------------------------

_STATE = {"run_started": False}


def append_partial(event: dict) -> None:
    """One JSON line appended to the partial log (UTF-8, LF), flushed and synced before the caller prints any of it.
    Never truncated or rewritten by the runner."""
    ev = dict(event, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    data = (json.dumps(R.strip_private(ev), ensure_ascii=False) + "\n").encode("utf-8")
    with open(PARTIAL_LOG, "ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def partial_facts():
    """None when the partial log is absent; else its name, sha256 and line count."""
    if not PARTIAL_LOG.exists():
        return None
    raw = PARTIAL_LOG.read_bytes()
    return {"path": PARTIAL_LOG.name, "sha256": hashlib.sha256(raw).hexdigest(), "lines": raw.count(b"\n")}


def cpu_phase(sentences=None, prov=None, receipts=None, tok_loader=None, models=MODELS, log=print) -> dict:
    """Every assert that needs no model, in the order the run makes them. Returns the prepared items and rows."""
    t0 = time.time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    loader = tok_loader or R.load_tokenizer
    tok_gen = loader(GENERATOR_TOKENIZER)
    if sentences is None:
        sentences, prov = R.read_wikitext_sentences(cap=ZOO_SLICE[1])
        v3_only, v3_prov = R.read_wikitext_sentences(cap=R.POOL_CAP)
        if sentences[:R.POOL_CAP] != v3_only or v3_prov["sha256"] != prov["sha256"]:
            raise Refusal("the leading 150,000 eligible sentences of the extended read are not v3's pool")
    log(f"  sentences: {len(sentences)} eligible read ({time.time() - t0:.0f}s)")
    prep = prepare_zoo_items(tok_gen, sentences, prov)
    receipts = committed_receipts() if receipts is None else receipts
    view = v3_pool_view(sentences, prov, prep["code_pools"])
    prior = R.prior_sentences_and_codes(view, {k: v for k, v in receipts.items() if v is not None})
    red = R.tokenizer_level_asserts(prep, {"neox": tok_gen}, prior["sentence_indices"])
    lists = R.hash_lists(prep)
    disj = zoo_disjointness(prep, lists, view, receipts)
    log(f"  items: {N_ZOO} at D0={D0}, seed {SEED_ZOO}; {lists['n_prompts']} prompts; {lists['n_sentences']} sentences, "
        f"{lists['n_codes_test']} test and {lists['n_codes_exemplar']} exemplar codes; disjoint from the committed receipts")
    p2 = regenerate_pilot2(tok_gen, view["sentences"], receipts.get("pilot2"))
    log(f"  pilot2 regenerated byte-identically ({p2['prompts_matched']} prompts, items_sha256 {p2['items_sha256'][:12]})")
    rows = []
    for repo, group, role in models:
        row = prepare_model_row(repo, group, role, prep, loader)
        rows.append(row)
        if row["status"] == "READY":
            t = row["tokenizer"]
            lo = min(v["min"] for v in t["prompt_tokens"].values())
            log(f"  {repo:36} {group:8} {row['family']['family']:9} ready: names {list(t['name_token_ids'].values())}, code len "
                f"{t['code_token_length']}, bos {t['bos_prefix']}, prompts {lo}-{t['longest_prompt_with_candidate']} tokens "
                f"with the candidate, filler {t['filler_tokens_native']['min']}-{t['filler_tokens_native']['max']} native")
        else:
            log(f"  {repo:36} {group:8} {row['status']}: {row['reason']}")
    ident = assert_identical_bytes(rows, lists)
    return {"prep": prep, "lists": lists, "redteam": red, "disjointness": disj, "pilot2": p2, "rows": rows,
            "identical_bytes": ident, "sentences": sentences, "prov": prov, "tok_gen": tok_gen, "receipts": receipts,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------------
# the GPU phase
# ---------------------------------------------------------------------------------------------------

def load_for_scoring(row: dict, prep: dict, red: dict, tag: str, record: dict | None = None):
    """Load one model per NOTE sections 7 and 12 and run its model-level asserts. Returns (model, model as v3 reads it,
    info). `record`, when given, is filled as the load goes (numbers and strings only, never a tensor), so that a row
    that fails an assert keeps the measurements that failed it. On any failure the failed frame's locals are cleared
    before the exception travels on, so that the traceback does not keep the model alive past release() (the defect
    v3's release() docstring records)."""
    try:
        return _load_for_scoring(row, prep, red, tag, record)
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)
        raise


def _move_to_cuda(model, mt: str):
    """The model onto the GPU: v3's to_gpu_fp16 (half(), then cuda) on v3's own path; a device move only for the
    others, whose parameters are already fp16 from transformers' own load. A CUDA out-of-memory error here makes the
    row NOT_RUN__memory (NOTE section 12.2)."""
    torch = R._torch()
    try:
        if mt in NEOX_V3_PATH_TYPES:
            return R.to_gpu_fp16(model)
        model.to("cuda")
        torch.cuda.synchronize()
        return model
    except torch.cuda.OutOfMemoryError as e:
        model = None
        torch.cuda.empty_cache()
        raise NotRun(STATUS_MEMORY, f"CUDA out of memory while moving the model to the GPU: {e}")


def _load_for_scoring(row: dict, prep: dict, red: dict, tag: str, record: dict | None = None):
    torch = R._torch()
    repo, mt = row["repo"], row["family"]["model_type"]
    tok = row["_tok"]
    items0 = prep["cells"][D0]
    seqs2, name_ids = row["_seqs"]["compare/plain"][:2], row["_name_ids"]
    info = record if record is not None else {}
    info.update({"model_type": mt, "v3_path": v3_path(mt)})
    t0 = time.time()
    try:
        R.require_free_gpu(red, tag)
    except Refusal as e:
        raise NotRun(STATUS_FREE_GPU, str(e))
    if mt in NEOX_V3_PATH_TYPES:
        if mt == "mamba":
            info["chunked_scan"] = R.install_chunked_scan()
            if info["chunked_scan"]["transformers_fast_path_available"]:
                raise NotScreenable("fused mamba-ssm kernels present: the cuda forward would bypass v3's slow path")
        model, linfo = R.load_arm_cpu_fp32(repo)
        info.update(linfo)
        if mt == "mamba":
            info["mamba_A_on_loaded_weights"] = R.assert_mamba_A(model)
            ce = R.check_chunked_scan_equivalence(model, tok, items0[0], prep["exemplars"])
            info["chunked_scan_equivalence_fp32_cpu"] = ce
            if ce["max_abs_logit_diff"] > R.FP32_TOL:
                raise NotScreenable(f"the chunked scan differs from the reference slow_forward by {ce['max_abs_logit_diff']:.2e}")
        rp = R.check_right_padding(v3_path(mt), model, tok, items0, prep["exemplars"])
        info["right_padding_fp32_cpu"] = rp
        if rp["max_abs_logit_diff"] > R.FP32_TOL:
            raise NotScreenable(f"right padding changes the last-position logits by {rp['max_abs_logit_diff']:.2e} in fp32",
                                STATUS_RIGHT_PADDING)
        # NOTE section 12.4: on v3's own path the fp32 model is already on the CPU; its alone name log-probs on the two
        # prompts are the reference for a recorded (not gated) fp16 comparison, so that M1 and T stay under pilot2's conditions
        info["fp32_reference"] = {"ran": True, "on": "the fp32 CPU model of v3's own path", "gated": False,
                                  "alone_name_logprobs": alone_name_logprobs(v3_path(mt), model, seqs2, name_ids)}
        try:
            R.require_free_gpu(red, tag + "_to_cuda")
        except Refusal as e:
            model = None
            raise NotRun(STATUS_FREE_GPU, str(e))
        torch.cuda.reset_peak_memory_stats()
        model = _move_to_cuda(model, mt)
    else:
        snap = R.snapshot_dir(repo)
        info["fp32_reference"] = {"ran": None, "note": "started; a failure inside it is the row's reason"}
        info["fp32_reference"] = fp32_reference(snap, mt, seqs2, name_ids)
        info["fp32_reference"]["gated"] = bool(info["fp32_reference"]["ran"])
        fp32_reference_gate(info["fp32_reference"])
        t1 = time.time()
        model, linfo = load_hf_cpu(snap, mt, "float16")
        info.update({"repo": repo, "revision": snap.name, "from_local_snapshot": True, "load_seconds_cpu_fp16": round(time.time() - t1, 2)})
        info.update({k: v for k, v in linfo.items() if k != "load_seconds_cpu"})
        try:
            R.require_free_gpu(red, tag + "_to_cuda")
        except Refusal as e:
            model = None
            raise NotRun(STATUS_FREE_GPU, str(e))
        torch.cuda.reset_peak_memory_stats()
        model = _move_to_cuda(model, mt)
    info["load_and_assert_seconds"] = round(time.time() - t0, 2)
    info["dtype"] = str(next(model.parameters()).dtype)
    info["buffers_on_gpu"] = buffer_dtypes(model)
    vm = as_v3_model(model, mt)
    rp16 = right_padding_ids(v3_path(mt), vm, seqs2, name_ids)
    info["right_padding_fp16_gpu"] = rp16
    if mt not in NEOX_V3_PATH_TYPES and rp16["max_abs_name_logprob_diff"] > RIGHT_PAD_FP16_TOL:
        raise NotScreenable(f"right padding moves the name log-probs by {rp16['max_abs_name_logprob_diff']:.3f} > {RIGHT_PAD_FP16_TOL} in fp16",
                            STATUS_RIGHT_PADDING)
    ref = info["fp32_reference"]
    if ref["ran"]:
        ref_lp = ref["alone_name_logprobs"] if "alone_name_logprobs" in ref else ref["right_padding_fp32_cpu"]["alone_name_logprobs"]
        fid = fp16_fidelity(ref_lp, rp16["alone_name_logprobs"])
        info["fp16_fidelity"] = dict(fid, gated=ref["gated"])
        if ref["gated"] and not fid["ok"]:
            raise NotScreenable(f"the fp16 name log-probs differ from the fp32 CPU reference by {fid['max_abs_name_logprob_diff']:.3f} > "
                                f"{FP16_FIDELITY_TOL}", STATUS_FP16_FIDELITY)
    else:
        info["fp16_fidelity"] = {"ran": False, "reason": ref.get("reason"), "reading": "a fail on this row reads: did not perform in fp16"}
    return model, vm, info


def peak_mb() -> float:
    torch = R._torch()
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


def run_screen(log=print, allow_after_partial: bool = False) -> int:
    if RECEIPT.exists():
        raise Refusal(f"{RECEIPT.name} exists: the screen runs once and a committed receipt is never regenerated")
    prior_partial = partial_facts()
    if prior_partial is not None and not allow_after_partial:
        raise Refusal(f"{PARTIAL_LOG.name} exists: an earlier attempt saw model outputs; a rerun needs the operator's "
                      f"--operator-allows-rerun-after-partial, and the new receipt then records that log's sha256")
    git = git_facts()
    require_committed(git)
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch = R._torch()
    if not torch.cuda.is_available():
        raise Refusal("CUDA is not available; the screen runs on the GPU")
    log(f"ZOO D0 SCREEN: n={N_ZOO} per family per model at D0={D0}, seed {SEED_ZOO}, readout {READOUT} — pilot-class, licenses nothing")
    cpu = cpu_phase(log=log)
    prep, rows = cpu["prep"], cpu["rows"]
    items = prep["cells"][D0]
    res = {"note": NOTE.name, "mode": "zoo_d0_screen", "class": "PILOT-CLASS readiness screen; licenses nothing; not part of bridge v3's verdict",
           "question": "whether ANY cached model can perform the v3 compare task, and the v3 recall task, at D0 under the argmax readout",
           "config": {"seed_items": SEED_ZOO, "n_per_family_per_model": N_ZOO, "D0": D0, "variants": list(VARIANTS), "readout": READOUT,
                      "compare_question": R.question_text("compare", 0), "recall_question": R.question_text("recall", 0),
                      "names": {"test": list(R.NAMES), "exemplar": list(R.EXEMPLAR_NAMES)}, "chance": dict(R.CHANCE),
                      "screen_bar": SCREEN_BAR, "guard_max_slot": GUARD_SLOT_MAX, "ablated_flag_bar": ABLATED_FLAG_BAR,
                      "batch": BATCH, "wrapper_equivalence": {"tol": WRAPPER_EQ_TOL, "tie": WRAPPER_EQ_TIE},
                      "right_padding_fp16_tol": RIGHT_PAD_FP16_TOL, "free_gpu_min_bytes": R.FREE_GPU_MIN_BYTES, "dtype": "float16 on cuda",
                      "fp16_load": "Llama, Qwen and Gemma: transformers' own fp16 load, no cast after it (buffers recorded); "
                                   "mamba and gpt_neox: v3's own path, fp32 on CPU then half() onto cuda, as pilot2",
                      "fp32_reference": {"right_padding_fp32_tol": RIGHT_PAD_FP32_TOL, "fp16_fidelity_tol": FP16_FIDELITY_TOL,
                                         "ram_rule": f"available RAM >= {RAM_FP32_FACTOR} x fp32 bytes + {RAM_HEADROOM_BYTES} bytes",
                                         "gated_on": "Llama, Qwen and Gemma rows; recorded only on v3's own path"},
                      "statuses": list(STATUSES),
                      "generator_tokenizer": GENERATOR_TOKENIZER, "models": [list(m) for m in MODELS], "excluded": dict(EXCLUDED),
                      "code_pools": prep["pools"], "zoo_codes": {"exemplar": prep["code_pools"]["zoo_exemplar"], "test": prep["code_pools"]["zoo_test"]},
                      "filler_pool": prep["sentence_provenance"], "sentence_region": prep["sentence_region"], "git": git},
           "items": {"sha256": cpu["lists"], "items_sha256": R.items_digest(prep["cells"]), "prefix_nested_across_distances": False,
                     "balance": cpu["redteam"]["balance_by_distance"]},
           "sentence_indices_used": sorted(prep["pool"].used), "sentence_indices_skipped_for_a_name": sorted(prep["pool"].skipped),
           "disjointness": cpu["disjointness"], "wrapper_equivalence": {}, "models": {}, "per_item": {}, "summary": None,
           "redteam": {"generator": cpu["redteam"], "identical_bytes": cpu["identical_bytes"],
                       "pilot2_regeneration": {k: v for k, v in cpu["pilot2"].items() if k != "prep"}},
           "runtime": {"cpu_phase_seconds": cpu["seconds"], "prior_partial_log": prior_partial}}
    R.record_versions(res)
    _STATE["run_started"] = True
    append_partial({"event": "run_start", "head": git["head"], "pid": os.getpid(), "prior_partial_log": prior_partial,
                    "items_sha256": res["items"]["items_sha256"]})
    red = res["redteam"]
    by_repo = {r["repo"]: r for r in rows}
    rec2 = cpu["receipts"]["pilot2"]

    # the wrapper's equivalence on M1 and T, before any zoo item is scored
    t0 = time.time()
    for arm in ("M1", "T"):
        row = by_repo[R.ARMS[arm]]
        if row["status"] != "READY":
            raise Refusal(f"{arm} is not ready ({row['status']}: {row['reason']}); the wrapper's equivalence cannot run")
        try:
            model, vm, info = load_for_scoring(row, prep, red, f"{arm}_equivalence")
        except (NotScreenable, NotRun) as e:
            raise Refusal(f"{arm} could not be loaded for the wrapper's equivalence: {e}")
        try:
            eq = wrapper_equivalence(arm, v3_path(row["family"]["model_type"]), vm, row["_tok"], cpu["pilot2"]["prep"], rec2)
        except (NotScreenable, NotRun) as e:
            traceback.clear_frames(e.__traceback__)
            raise Refusal(f"{arm}: the wrapper's equivalence could not complete: {e}")
        finally:
            model = vm = None
            R.release(red, f"{arm}_equivalence")
        eq["load"] = info
        res["wrapper_equivalence"][arm] = eq
        append_partial({"event": "wrapper_equivalence", "arm": arm, "result": eq})
        log(equivalence_log_line(arm, eq))
        if not eq["ok"]:
            raise Refusal(f"the wrapper does not reproduce the committed pilot2 per-item outputs on {arm}: {eq}")
    res["runtime"]["wrapper_equivalence_seconds"] = round(time.time() - t0, 1)

    # the screen
    for row in rows:
        repo = row["repo"]
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        if row["status"] != "READY":
            res["models"][repo] = out
            continue
        t0 = time.time()
        model = vm = None
        msg = None
        out["load"] = {}
        try:
            model, vm, info = load_for_scoring(row, prep, red, repo, record=out["load"])
            path = v3_path(row["family"]["model_type"])
            cells, t_score = {}, time.time()
            for family in FAMILIES:
                for variant in VARIANTS:
                    s = score_cell_with_backoff(path, vm, family, row["_seqs"][f"{family}/{variant}"], cell_targets(items, family),
                                                row["_name_ids"], row["_cands"] if family == "recall" else None)
                    cells[(family, variant)] = s
            out["scoring_seconds"] = round(time.time() - t_score, 2)
            out["peak_gpu_mb"] = peak_mb()
            per_item = {}
            out["cells"] = {}
            for (family, variant), s in cells.items():
                per_item.setdefault(family, {})[variant] = s.pop("_per_item")
                s.pop("_correct")
                out["cells"][f"{family}/{variant}"] = s
            out["screen"] = screen_from_cells(cells)
            out["status"] = STATUS_SCREENED
            res["per_item"][repo] = per_item
            sc = out["screen"]
            msg = (f"  {repo:36} compare {sc['compare']['acc_norm']:+.3f} max-slot {sc['compare']['max_slot_fraction']:.2f} "
                   f"{'pass' if sc['compare']['clears'] else 'fail'} | recall {sc['recall']['acc_norm']:+.3f} "
                   f"{'pass' if sc['recall']['clears'] else 'fail'} | C_abl {sc['compare']['ablated_acc_norm']:+.3f}/"
                   f"{sc['recall']['ablated_acc_norm']:+.3f} ({time.time() - t0:.0f}s, peak {out['peak_gpu_mb']:.0f} MB)")
        except (NotScreenable, NotRun) as e:
            traceback.clear_frames(e.__traceback__)
            out.update(status=e.status, reason=str(e))
            msg = f"  {repo:36} {e.status}: {e}"
        except Exception as e:                                        # a defect or a device error: recorded, never hidden
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            traceback.clear_frames(e.__traceback__)
            out.update(status=STATUS_ERROR, reason=f"{type(e).__name__}: {e}", error_traceback=tb[-6000:])
            msg = f"  {repo:36} {STATUS_ERROR}: {type(e).__name__}: {e}"
        finally:
            model = vm = None
            out["free_gpu_bytes_after_release"] = R.release(red, repo)
            out["seconds"] = round(time.time() - t0, 2)
        out.pop("_tok", None)
        res["models"][repo] = out
        append_partial({"event": "model", "repo": repo, "row": out, "per_item": res["per_item"].get(repo)})
        log(msg)
    res["summary"] = summarize(list(res["models"].values()))
    res["runtime"]["total_seconds"] = round(sum(m.get("seconds", 0.0) for m in res["models"].values()) + res["runtime"]["cpu_phase_seconds"]
                                            + res["runtime"]["wrapper_equivalence_seconds"], 1)
    R.write_json(RECEIPT, res)
    append_partial({"event": "receipt_written", "receipt": RECEIPT.name, "sha256": R.sha256_file(RECEIPT)})
    for group in GROUPS:
        log(f"\n  {res['summary'][group]['sentence']}")
    log(f"\nwrote {RECEIPT.name}")
    return 0


def run_check(log=print) -> int:
    """CPU only. Hides the GPU from this process before torch can be imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    log(f"ZOO D0 SCREEN --check (CPU only; no model is loaded; nothing is written)")
    git = git_facts()
    log("  git: HEAD " + str(git["head"]) + "; " + "; ".join(f"{k} committed={git[k]['head_blob_sha256'] is not None} matches={git[k]['working_copy_matches_head']}"
                                                         for k in COMMITTED_KEYS))
    log(f"  partial log: {partial_facts() or 'absent'}")
    cpu = cpu_phase(log=log)
    ready = [r for r in cpu["rows"] if r["status"] == "READY"]
    log(f"  {len(ready)} of {len(cpu['rows'])} models pass every CPU assert; bytes identical across them: {cpu['identical_bytes']['identical']}"
        f" ({cpu['seconds']:.0f}s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="every CPU assert of the run; loads no model; writes nothing")
    ap.add_argument("--operator-allows-rerun-after-partial", action="store_true",
                    help="run although zoo_d0_screen.partial.jsonl exists (the operator's call); its sha256 goes into the receipt")
    args = ap.parse_args(argv)
    try:
        return run_check() if args.check else run_screen(allow_after_partial=args.operator_allows_rerun_after_partial)
    except Refusal as e:
        if args.check:
            print(f"\nREFUSED: {e}", file=sys.stderr, flush=True)
            return 2
        if _STATE["run_started"]:
            append_partial({"event": "refused", "reason": str(e)})
        R.write_json(REFUSAL_RECEIPT, {"refused": True, "reason": str(e), "mode": "zoo_d0_screen", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                       "partial_log": partial_facts()})
        print(f"\nREFUSED: {e}\nwrote {REFUSAL_RECEIPT.name}", file=sys.stderr, flush=True)
        return 2
    except Exception as e:                                            # NOTE section 12.5: an error outside a model's row leaves a record
        if args.check:
            raise
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if _STATE["run_started"]:
            append_partial({"event": "error", "error": f"{type(e).__name__}: {e}", "traceback": tb[-6000:]})
        R.write_json(REFUSAL_RECEIPT, {"refused": False, "error": f"{type(e).__name__}: {e}", "traceback": tb[-6000:], "mode": "zoo_d0_screen",
                                       "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "partial_log": partial_facts()})
        print(f"\nERROR: {type(e).__name__}: {e}\nwrote {REFUSAL_RECEIPT.name}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""CPU-only tests for papers/frequency-resonance/run_zoo_d0_screen.py — no model is loaded, no GPU is touched.

What is checked, against NOTE_zoo_d0_screen_plan_2026_10_12.md:
  * the runner reuses v3's generator, readouts, summary and screen rule rather than copies of them, and its constants
    are the plan's;
  * the zoo's sentence slice lies outside v3's pool and is hash-deduplicated against it; its code slice is the tail of
    v3's main test pool and misses every reserved pool; the items are exactly balanced with clean fillers;
  * disjointness from the committed v3 receipts by their sha256 lists (and a forged overlap is refused);
  * the pilot2 items regenerate byte-identically from v3's generator, and the wrapper's ids equal v3's
    compositional_ids on every pilot2 and zoo prompt in the GPT-NeoX tokenizer;
  * the per-model asserts on every cached tokenizer (single-token names, in-context answer ids, equal-length codes,
    BOS, length) and the family fact from config and, for Mamba, from weights; a failing tokenizer is NOT_SCREENABLE
    and nothing is re-worded;
  * the scoring wrapper equals v3's run_cell on synthetic models on both of v3's paths, and the adapter's head applies
    gemma-2's softcap;
  * the equivalence rule against a reference, the screen rule at its edges, the summary's groups, the refusals;
  * the plan's addendum (NOTE section 12): the fp16 load casts nothing after it (inv_freq stays fp32, recorded), the
    fp32 CPU reference and the fp16 fidelity rule, every status the runner writes is named in the plan, the sentence's
    wording and C_abl flag counts, the append-only partial log, an error outside a row leaves a record, and the three
    committed v3 receipts are in the committed-and-unchanged check.

The real-pool and real-tokenizer tests skip when wikitext-103 or a tokenizer is not cached. The tiny-model tests build
random-weight models from transformers' own classes in a temporary directory; no checkpoint is loaded.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "papers" / "frequency-resonance"
ZOO = PAPER / "run_zoo_d0_screen.py"
NOTE = PAPER / "NOTE_zoo_d0_screen_plan_2026_10_12.md"
TEST_FILE = Path(__file__).resolve()

sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load(ZOO, "run_zoo_d0_screen_under_test")
R = Z.R


# -- helpers -------------------------------------------------------------------------------------------------

class FakeTok:
    """A GPT-2-shaped pretokenizer with one id per pretoken (compositional at spaces and newlines, like NeoX), plus
    the knobs the per-model asserts must catch: a default BOS or EOS, names split into two tokens, a ':'+name merge."""

    _re = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

    def __init__(self, bos=(), eos=(), split=(), merge_colon_name=False, name="fake-tok"):
        self.vocab = {}
        self.name = name
        self.bos, self.eos, self.split, self.merge = list(bos), list(eos), set(split), merge_colon_name
        outer = self

        class _HF:
            def encode(self, text, add_special_tokens=True):
                ids = outer.encode(text)
                return outer.bos + ids + outer.eos if add_special_tokens else ids

            def __len__(self):
                return len(outer.vocab)

        self.hf = _HF()

    def _id(self, p):
        return self.vocab.setdefault(p, len(self.vocab) + 1)

    def encode(self, text):
        parts = self._re.findall(text)
        out, i = [], 0
        while i < len(parts):
            p = parts[i]
            if self.merge and p == ":" and i + 1 < len(parts) and parts[i + 1].strip() in R.NAMES:
                out.append(self._id(p + parts[i + 1]))
                i += 2
                continue
            if p.strip() in self.split:
                out += [self._id(p[:-2]), self._id(p[-2:])]
            else:
                out.append(self._id(p))
            i += 1
        return out

    def __len__(self):
        return len(self.vocab)


WORDS = ("river stone garden window silver morning harbour meadow lantern voyage quiet marble thunder orchard pilgrim "
         "velvet canyon ember forest glacier island jasmine kettle ladder mirror nectar oyster pepper quarry ribbon saddle "
         "timber umbrella violin walnut yellow zephyr anchor basket candle dagger engine falcon goblet hammer iron jacket "
         "kernel lemon magnet needle oak parcel quill rocket sandal tunnel urn vessel wagon yarn").split()


def make_sentences(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(R.MIN_WORDS, 21))
        ws = [WORDS[i] for i in rng.integers(0, len(WORDS), size=k)]
        ws[0] = ws[0].capitalize()
        out.append(" ".join(ws) + ".")
    return out


V3_CAP, LO, HI = 3000, 3000, 6000
PROV = {"source": "synthetic", "files": ["synthetic.parquet"], "sha256": ["0" * 64], "pool_cap": HI, "eligible_sentences": HI}


def synthetic_prep(tok=None, seed=Z.SEED_ZOO):
    tok = tok or FakeTok()
    sents = make_sentences(HI)
    prep = Z.prepare_zoo_items(tok, sents, PROV, seed=seed, lo=LO, hi=HI, v3_cap=V3_CAP)
    return tok, sents, prep


@pytest.fixture(scope="module")
def syn():
    tok, sents, prep = synthetic_prep()
    return SimpleNamespace(tok=tok, sents=sents, prep=prep, lists=R.hash_lists(prep))


# -- the runner reuses v3 and carries the plan's constants -----------------------------------------------------

def test_the_runner_imports_v3_by_path_and_copies_none_of_its_functions():
    assert Path(R.__file__).resolve() == (PAPER / "run_realmodel_bridge_v3.py").resolve()
    src = ZOO.read_text(encoding="utf-8")
    for name in ("argmax_cell_summary", "next_token_scores_batch", "recall_loglik_batch", "build_items", "design_rows",
                 "build_exemplars", "prompt_text", "prompt_parts", "norm_acc", "ovr_auroc", "auroc", "arm_clears",
                 "assert_balance", "assert_fillers", "hash_lists", "prior_sentences_and_codes", "assert_disjoint_from_prior",
                 "name_token_ids", "partition_codes", "eligible_codes", "read_wikitext_sentences", "_hidden", "_logits"):
        assert f"def {name}(" not in src, f"{name} is redefined in the zoo runner"
        assert f"R.{name}" in src or name in ("design_rows", "prompt_parts", "norm_acc", "ovr_auroc", "auroc", "assert_balance",
                                              "assert_fillers", "name_token_ids", "eligible_codes", "partition_codes"), name


def test_constants_are_the_plans():
    note = NOTE.read_text(encoding="utf-8")
    assert (Z.SEED_ZOO, Z.N_ZOO, Z.D0) == (1013, 48, 16)
    assert Z.ZOO_SLICE == (150_000, 180_000) and R.POOL_CAP == 150_000
    assert (Z.SCREEN_BAR, Z.GUARD_SLOT_MAX, Z.ABLATED_FLAG_BAR) == (0.40, 0.80, 0.16)
    assert (Z.WRAPPER_EQ_TOL, Z.WRAPPER_EQ_TIE, Z.RIGHT_PAD_FP16_TOL) == (0.01, 0.01, 0.25)
    assert (Z.N_ZOO_EXEMPLAR_CODES, Z.N_ZOO_TEST_CODES, Z.BATCH) == (12, 60, 16)
    assert Z.READOUT == "argmax_largest" and Z.VARIANTS == ("plain", "ablated")
    for phrase in ("**Seed 1013**", "**n = 48 per family per model at D0 = 16**", "150,000 to 179,999",
                   "the last 72 codes of `main_test`", "normalized accuracy ≥ 0.40 and the max-slot guard ≤ 0.80",
                   "|Δ score| ≤ 0.01", "≤ 0.25", "exceeds 0.16"):
        assert phrase in note, phrase
    # the addendum's constants (NOTE section 12.4)
    assert (Z.RIGHT_PAD_FP32_TOL, Z.FP16_FIDELITY_TOL, Z.RAM_FP32_FACTOR, Z.RAM_HEADROOM_BYTES) == (1e-3, 0.5, 1.2, 2 ** 30)
    sec12 = note[note.index("## 12. Addendum, 2026-10-12"):]
    for phrase in ("at least 1.2 × the checkpoint's fp32 bytes", "+ 1 GiB", "differ by at most 1e-3", "within 0.5 nats",
                   "`--operator-allows-rerun-after-partial`", "`zoo_d0_screen.partial.jsonl`"):
        assert phrase in sec12, phrase
    assert Z.PARTIAL_LOG.name == "zoo_d0_screen.partial.jsonl"
    sec2 = note[note.index("## 2. Models"):note.index("## 3. The items")]          # section 15's results table repeats the rows
    table = re.findall(r"^\| (base|instruct) \| `([^`]+)`", sec2, re.M)
    assert [(g, r) for r, g, _ in Z.MODELS] == [(g, r) for g, r in table]
    assert len(Z.MODELS) == 13 and "microsoft/Phi-3.5-mini-instruct" in Z.EXCLUDED and "microsoft/Phi-3.5-mini-instruct" in note
    for repo, _, role in Z.MODELS:
        if role:
            assert R.ARMS[role] == repo


# -- the zoo's sentence and code slices; the items ---------------------------------------------------------------

def test_sentence_slice_is_outside_v3s_pool_hash_deduplicated_and_seeded():
    sents = make_sentences(HI)
    sents[LO + 5] = sents[17]                  # equal to a v3 pool sentence: removed
    sents[LO + 9] = sents[LO + 8]              # a repeat inside the slice: the later one removed
    order, info = Z.zoo_sentence_order(sents, LO, HI, V3_CAP, seed=1013)
    assert LO + 5 not in order and LO + 9 not in order and LO + 8 in order
    assert all(LO <= i < HI for i in order) and len(order) == len(set(order)) == info["kept"]
    texts = [sents[i] for i in order]
    assert len(set(texts)) == len(texts) and not set(texts) & set(sents[:V3_CAP])
    assert info["removed_equal_to_a_v3_pool_sentence"] >= 1 and info["removed_repeat_inside_slice"] >= 1
    assert Z.zoo_sentence_order(sents, LO, HI, V3_CAP, seed=1013)[0] == order
    assert Z.zoo_sentence_order(sents, LO, HI, V3_CAP, seed=1014)[0] != order
    with pytest.raises(R.Refusal, match="inside v3's pool"):
        Z.zoo_sentence_order(sents, LO - 1, HI, V3_CAP)
    with pytest.raises(R.Refusal, match="needs"):
        Z.zoo_sentence_order(sents[:HI - 1], LO, HI, V3_CAP)


def test_code_slice_is_the_tail_of_main_test_and_misses_every_reserved_pool():
    codes = [str(c) for c in range(1000, 10000)]
    pools = Z.zoo_code_pools(codes)
    v3 = R.partition_codes(codes)
    for k, v in v3.items():
        assert pools[k] == v                                   # v3's partition untouched
    assert pools["zoo_exemplar"] + pools["zoo_test"] == v3["main_test"][-72:]
    assert len(pools["zoo_exemplar"]) == 12 and len(pools["zoo_test"]) == 60
    reserved = set().union(*(set(v3[k]) for k in v3 if k != "main_test"))
    assert not reserved & (set(pools["zoo_exemplar"]) | set(pools["zoo_test"]))


def test_zoo_items_are_v3_items_exactly_balanced_with_clean_fillers(syn):
    items, ex = syn.prep["cells"][16], syn.prep["exemplars"]
    got = R.assert_balance(items, ex)
    assert got["max_slot"] == got["q_slot"] == got["last_exemplar_compare_slot"] == got["last_exemplar_recall_slot"] == [16, 16, 16]
    assert got["last_exemplar"] == [8, 16, 16, 8] and got["last_compare_slot_is_correct"] == got["last_recall_slot_is_queried"] == 16
    R.assert_fillers(syn.prep["cells"], ex)
    for it in items:
        assert len(syn.tok.encode(" " + it["filler"])) == 16
        assert set(it["codes"]) <= set(syn.prep["code_pools"]["zoo_test"])
        assert all(int(c) <= R.CODE_MAX_PLANTED for c in it["codes"])
    assert {c for e in ex for c in e["codes"]} == set(syn.prep["code_pools"]["zoo_exemplar"])
    assert all(LO <= i < HI for i in syn.prep["pool"].used)
    assert syn.lists["n_prompts"] == 2 * 2 * 48
    assert list(syn.lists["prompts"]["compare"]["16"]) == ["plain", "ablated"]
    # the v3 tokenizer-level asserts take the zoo's prep unchanged
    red = R.tokenizer_level_asserts(syn.prep, {"neox": syn.tok})
    assert red["sentences_inside_region"] and red["balance_by_distance"]["16"]["n"] == 48
    # the generator is v3's: the same seed and pools reproduce the same prompts
    _, _, again = synthetic_prep()
    assert R.hash_lists(again)["prompts"] == syn.lists["prompts"]


def _fake_receipt(region, view, idx=(), sentences=(), codes=()):
    return {"config": {"filler_pool": {"sha256": view["sentence_provenance"]["sha256"], "pool_cap": V3_CAP, "eligible_sentences": V3_CAP},
                       "sentence_region": {"which": region}},
            "sentence_indices_used": list(idx), "items": {"sha256": {"sentences": list(sentences), "codes_test": list(codes), "codes_exemplar": []}},
            "redteam": {"items_sha256": "x"}}


def test_disjointness_against_receipt_lists_and_a_forged_overlap_is_refused(syn):
    view = Z.v3_pool_view(syn.sents, PROV, syn.prep["code_pools"], v3_cap=V3_CAP)
    assert view["sentence_provenance"]["pool_cap"] == V3_CAP and len(view["sentences"]) == V3_CAP
    clean = {"smoke": _fake_receipt("smoke", view, idx=range(0, 50)), "pilot": _fake_receipt("pilot", view, idx=range(50, 80)),
             "pilot2": _fake_receipt("pilot2", view, idx=range(80, 90), sentences=[R.sha256_text(syn.sents[85])])}
    out = Z.zoo_disjointness(syn.prep, syn.lists, view, clean)
    assert out["against_committed_receipts"]["ok"] and out["by_construction"]["sentences_used"] == len(syn.prep["pool"].used)
    used_text = syn.sents[syn.prep["pool"].used[0]]
    forged = dict(clean, pilot2=_fake_receipt("pilot2", view, sentences=[R.sha256_text(used_text)]))
    with pytest.raises(R.Refusal, match="shares with a committed receipt"):
        Z.zoo_disjointness(syn.prep, syn.lists, view, forged)
    planted = syn.prep["cells"][16][0]["codes"][0]
    forged = dict(clean, pilot2=_fake_receipt("pilot2", view, codes=[R.sha256_text(planted)]))
    with pytest.raises(R.Refusal, match="code hashes"):
        Z.zoo_disjointness(syn.prep, syn.lists, view, forged)
    with pytest.raises(R.Refusal, match="absent"):
        Z.zoo_disjointness(syn.prep, syn.lists, view, dict(clean, smoke=None))
    other = dict(PROV, sha256=["1" * 64])
    bad_view = Z.v3_pool_view(syn.sents, other, syn.prep["code_pools"], v3_cap=V3_CAP)
    with pytest.raises(R.Refusal, match="sentence pool differs"):
        Z.zoo_disjointness(syn.prep, syn.lists, bad_view, clean)


# -- the real pool and the committed receipts (skipped when not cached) ----------------------------------------

@pytest.fixture(scope="module")
def real():
    pytest.importorskip("transformers")
    pytest.importorskip("pyarrow")
    if R.snapshot_dir("wikitext", kind="datasets") is None:
        pytest.skip("wikitext-103-raw-v1 is not cached here")
    if R.snapshot_dir(Z.GENERATOR_TOKENIZER) is None:
        pytest.skip("the GPT-NeoX tokenizer is not cached here")
    receipts = Z.committed_receipts()
    if any(v is None for v in receipts.values()):
        pytest.skip("the committed v3 receipts are not in this checkout")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = R.load_tokenizer(Z.GENERATOR_TOKENIZER)
    sents, prov = R.read_wikitext_sentences(cap=Z.ZOO_SLICE[1])
    prep = Z.prepare_zoo_items(tok, sents, prov)
    return SimpleNamespace(tok=tok, sents=sents, prov=prov, prep=prep, lists=R.hash_lists(prep), receipts=receipts)


def test_real_pool_prefix_is_v3s_pool(real):
    v3_only, v3_prov = R.read_wikitext_sentences(cap=R.POOL_CAP)
    assert real.sents[:R.POOL_CAP] == v3_only and v3_prov["sha256"] == real.prov["sha256"]
    assert len(real.prov["sha256"]) == 2 and real.prov["eligible_sentences"] == 180_000


def test_real_zoo_items_are_disjoint_from_the_committed_receipts(real):
    view = Z.v3_pool_view(real.sents, real.prov, real.prep["code_pools"])
    out = Z.zoo_disjointness(real.prep, real.lists, view, real.receipts)
    ok = out["against_committed_receipts"]
    assert ok["ok"]
    assert ok["prior_sentence_indices"] == 1113 + 171 + 140          # smoke, 1011 pilot, pilot2
    assert ok["prior_code_sha256"] == 2 * 72 + 50 + 12                # smoke and 1011 pilot pools, pilot2's lists
    assert all(150_000 <= i < 180_000 for i in real.prep["pool"].used)
    red = R.tokenizer_level_asserts(real.prep, {"neox": real.tok}, R.prior_sentences_and_codes(view, real.receipts)["sentence_indices"])
    assert red["balance_by_distance"]["16"]["max_slot"] == [16, 16, 16] and red["prompts"]["max_prompt_tokens"] < 300
    assert red["name_token_ids"]["neox"]["test"] == {"alpha": 9765, "gamma": 17356, "delta": 18687}


def test_pilot2_regenerates_byte_identically_and_the_wrapper_ids_are_v3s(real):
    p2 = Z.regenerate_pilot2(real.tok, real.sents[:R.POOL_CAP], real.receipts["pilot2"])
    assert p2["prompts_matched"] == 192 and p2["items_sha256"] == real.receipts["pilot2"]["redteam"]["items_sha256"]
    for prep in (p2["prep"], real.prep):
        for it in prep["cells"][16]:
            for family in R.FAMILIES:
                for variant in Z.VARIANTS:
                    text = R.prompt_text(it, prep["exemplars"], family, variant)
                    assert Z.model_prompt_ids(real.tok, text, []) == R.compositional_ids(real.tok, R.prompt_parts(it, prep["exemplars"], family, variant))
    forged = json.loads(json.dumps(real.receipts["pilot2"]))
    forged["items"]["sha256"]["prompts"]["recall"]["16"]["ablated"][3] = "0" * 64
    with pytest.raises(R.Refusal, match="recall/ablated"):
        Z.regenerate_pilot2(real.tok, real.sents[:R.POOL_CAP], forged)


EXPECT = {  # family, code token length, default BOS prefix
    "state-spaces/mamba-130m-hf": ("DECAY_SSM", 2, []), "state-spaces/mamba-1.4b-hf": ("DECAY_SSM", 2, []),
    "EleutherAI/pythia-1.4b": ("ATTENTION", 2, []),
    "meta-llama/Llama-3.2-1B": ("ATTENTION", 3, [128000]), "meta-llama/Llama-3.2-3B": ("ATTENTION", 3, [128000]),
    "Qwen/Qwen2.5-1.5B": ("ATTENTION", 5, []), "Qwen/Qwen2.5-3B": ("ATTENTION", 5, []),
    "google/gemma-2-2b": ("ATTENTION", 5, [2]),
    "meta-llama/Llama-3.2-1B-Instruct": ("ATTENTION", 3, [128000]), "meta-llama/Llama-3.2-3B-Instruct": ("ATTENTION", 3, [128000]),
    "Qwen/Qwen2.5-1.5B-Instruct": ("ATTENTION", 5, []), "Qwen/Qwen2.5-3B-Instruct": ("ATTENTION", 5, []),
    "google/gemma-2-2b-it": ("ATTENTION", 5, [2]),
}


@pytest.mark.parametrize("repo,group,role", Z.MODELS, ids=[m[0] for m in Z.MODELS])
def test_cached_tokenizer_passes_the_per_model_asserts(real, repo, group, role):
    ok, why, snap = Z.cache_status(repo)
    if not ok:
        pytest.skip(f"{repo}: {why}")
    row = Z.prepare_model_row(repo, group, role, real.prep)
    assert row["status"] == "READY", row["reason"]
    fam, code_len, bos = EXPECT[repo]
    t = row["tokenizer"]
    assert row["family"]["family"] == fam and t["code_token_length"] == code_len and t["bos_prefix"] == bos
    assert len(set(t["name_token_ids"].values())) == 3 and len(set(t["exemplar_name_token_ids"].values())) == 3
    assert t["in_context_checks"] == 2 * 48 * 3 + 2 * 48 * 3
    assert t["longest_prompt_with_candidate"] <= R.MAX_PROMPT_TOKENS and t["prompt_tokens"]["compare/plain"]["min"] > 100
    assert row["revision"] == snap.name and len(row["_seqs"]["recall/ablated"]) == 48
    per_item = t["filler_tokens_native_per_item"]
    assert len(per_item) == 48 and (min(per_item), max(per_item)) == (t["filler_tokens_native"]["min"], t["filler_tokens_native"]["max"])
    if fam == "DECAY_SSM":
        assert row["family"]["max_A"] < 0 and row["family"]["A_log_tensors"] == row["family"]["layers"]
    else:
        assert row["family"]["attention_layers"] == row["family"]["layers"] and row["family"]["reach_tokens"] >= t["longest_prompt_with_candidate"]
    if repo in (R.ARMS["M1"], R.ARMS["T"]):
        assert t["name_token_ids"] == {"alpha": 9765, "gamma": 17356, "delta": 18687}
    Z.assert_identical_bytes([row], real.lists)


# -- per-model asserts on synthetic tokenizers: NOT_SCREENABLE, never re-worded ---------------------------------

def test_a_two_token_name_or_a_split_answer_is_not_screenable_and_nothing_is_reworded(syn):
    with pytest.raises(Z.NotScreenable, match="gamma"):
        Z.model_tokenizer_asserts(FakeTok(split={"gamma"}), syn.prep, [])
    with pytest.raises(Z.NotScreenable, match="omega"):
        Z.model_tokenizer_asserts(FakeTok(split={"omega"}), syn.prep, [])
    with pytest.raises(Z.NotScreenable, match="answer token in context"):
        Z.model_tokenizer_asserts(FakeTok(merge_colon_name=True), syn.prep, [])
    ok = Z.model_tokenizer_asserts(FakeTok(), syn.prep, [7])
    assert all(s[0] == 7 for v in ok["_seqs"].values() for s in v)
    assert ok["_prompt_sha256"]["compare/plain"] == syn.lists["prompts"]["compare"]["16"]["plain"]
    assert ok["record"]["filler_tokens_native_per_item"] == [16] * 48       # the fake tokenizer is compositional like NeoX

    class OddCodes(FakeTok):
        def encode(self, text):
            out = super().encode(text)
            return out + [self._id("#")] if text.strip() == syn.prep["cells"][16][0]["codes"][0] else out

    with pytest.raises(Z.NotScreenable, match="several token lengths"):
        Z.model_tokenizer_asserts(OddCodes(), syn.prep, [])


def test_prepare_model_row_records_the_reason_and_keeps_no_ids(syn, monkeypatch):
    monkeypatch.setattr(Z, "cache_status", lambda repo: (True, "cached", Path("snap-x")))
    monkeypatch.setattr(Z, "family_fact", lambda snap: {"family": "ATTENTION", "model_type": "llama", "reach_tokens": 4096})
    row = Z.prepare_model_row("x/y", "base", None, syn.prep, tok_loader=lambda repo: FakeTok(split={"delta"}))
    assert row["status"] == "NOT_SCREENABLE" and "delta" in row["reason"] and "_seqs" not in row
    row = Z.prepare_model_row("x/y", "base", None, syn.prep, tok_loader=lambda repo: FakeTok())
    assert row["status"] == "READY" and row["revision"] == "snap-x"
    monkeypatch.setattr(Z, "family_fact", lambda snap: {"family": "ATTENTION", "model_type": "llama", "reach_tokens": 64})
    row = Z.prepare_model_row("x/y", "base", None, syn.prep, tok_loader=lambda repo: FakeTok())
    assert row["status"] == "NOT_SCREENABLE" and "reach" in row["reason"]
    monkeypatch.setattr(Z, "cache_status", lambda repo: (False, "no cached snapshot", None))
    row = Z.prepare_model_row("x/y", "base", None, syn.prep)
    assert row["status"] == "NOT_CACHED"
    rows = [Z.prepare_model_row("x/y", "base", None, syn.prep)]
    assert Z.assert_identical_bytes(rows, syn.lists)["models_checked"] == 0


def test_bos_prefix_is_read_from_the_tokenizer_default_and_nothing_may_follow():
    assert Z.bos_prefix(FakeTok()) == [] and Z.bos_prefix(FakeTok(bos=[5])) == [5]
    with pytest.raises(Z.NotScreenable, match="alter or follow"):
        Z.bos_prefix(FakeTok(bos=[5], eos=[6]))


# -- family facts from config and weights, never from the name -----------------------------------------------

def _snapshot(tmp_path, name, cfg, tensors):
    from safetensors.numpy import save_file
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(tensors, str(d / "model.safetensors"))
    return d


def test_family_fact_comes_from_config_and_tensors_not_the_name(tmp_path):
    pytest.importorskip("safetensors")
    w = np.zeros((2, 2), dtype=np.float32)
    attn = {f"model.layers.{i}.self_attn.{p}.weight": w for i in range(2) for p in ("q_proj", "k_proj", "v_proj")}
    d = _snapshot(tmp_path, "models--x--mamba-looking-name", {"model_type": "llama", "num_hidden_layers": 2, "max_position_embeddings": 4096}, attn)
    f = Z.family_fact(d)
    assert f["family"] == "ATTENTION" and f["attention_layers"] == 2 and f["reach_tokens"] == 4096
    Z.family_reach_ok(f, 4096)
    with pytest.raises(Z.NotScreenable, match="reach"):
        Z.family_reach_ok(f, 4097)
    g = _snapshot(tmp_path, "g2", {"model_type": "gemma2", "num_hidden_layers": 2, "max_position_embeddings": 8192, "sliding_window": 4096}, attn)
    assert Z.family_fact(g)["reach_tokens"] == 4096
    q = _snapshot(tmp_path, "q2", {"model_type": "qwen2", "num_hidden_layers": 2, "max_position_embeddings": 32768, "sliding_window": 64,
                                   "use_sliding_window": False}, attn)
    assert Z.family_fact(q)["reach_tokens"] == 32768
    half = {k: v for k, v in attn.items() if ".1." not in k}
    h = _snapshot(tmp_path, "half", {"model_type": "llama", "num_hidden_layers": 2, "max_position_embeddings": 4096}, half)
    with pytest.raises(Z.NotScreenable, match="1 of 2 layers"):
        Z.family_fact(h)
    alog = {f"backbone.layers.{i}.mixer.A_log": np.log(np.full((4, 16), 0.5 + i, dtype=np.float32)) for i in range(3)}
    m = _snapshot(tmp_path, "models--x--llama-looking-name", {"model_type": "mamba", "num_hidden_layers": 3}, alog)
    f = Z.family_fact(m)
    assert f["family"] == "DECAY_SSM" and f["A_log_tensors"] == 3 and f["max_A"] == pytest.approx(-0.5)
    dead = dict(alog)
    dead["backbone.layers.1.mixer.A_log"] = np.full((4, 16), -np.inf, dtype=np.float32)      # A = -0.0: not strictly negative
    m2 = _snapshot(tmp_path, "dead", {"model_type": "mamba", "num_hidden_layers": 3}, dead)
    with pytest.raises(Z.NotScreenable, match="not strictly negative"):
        Z.family_fact(m2)
    m3 = _snapshot(tmp_path, "short", {"model_type": "mamba", "num_hidden_layers": 4}, alog)
    with pytest.raises(Z.NotScreenable, match="3 A_log tensors"):
        Z.family_fact(m3)
    x = _snapshot(tmp_path, "rwkv", {"model_type": "rwkv", "num_hidden_layers": 2}, attn)
    with pytest.raises(Z.NotScreenable, match="neither mamba"):
        Z.family_fact(x)


def test_cached_mamba_weights_give_real_strictly_negative_A():
    pytest.importorskip("safetensors")
    for repo in ("state-spaces/mamba-130m-hf", "state-spaces/mamba-1.4b-hf"):
        ok, why, snap = Z.cache_status(repo)
        if not ok:
            pytest.skip(f"{repo}: {why}")
        f = Z.family_fact(snap)
        assert f["family"] == "DECAY_SSM" and f["A_log_tensors"] == f["layers"] and f["max_A"] < 0
    assert Z.family_fact(Z.cache_status("state-spaces/mamba-1.4b-hf")[2])["max_A"] == pytest.approx(-0.0039, abs=1e-4)   # v3's receipts: max -0.0039


# -- the scoring wrapper against v3's run_cell, on synthetic models (CPU torch) --------------------------------

def _fake_models(V, H=16, seed=0, cap=None):
    torch = pytest.importorskip("torch")

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.emb = torch.nn.Embedding(V, H)
            self.mix = torch.nn.Linear(H, H)
            with torch.no_grad():
                self.emb.weight.copy_(torch.randn(V, H, generator=g))
                self.mix.weight.copy_(torch.randn(H, H, generator=g) / H ** 0.5)
                self.mix.bias.zero_()

        def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
            x = self.emb(input_ids)
            n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
            return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(torch.cumsum(x, 1) / n)))   # causal: right padding is exact

    def head():
        g = torch.Generator().manual_seed(seed + 1)
        lin = torch.nn.Linear(H, V, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(V, H, generator=g) * 3.0)
        return lin

    class NeoX(torch.nn.Module):          # v3's path "T": .gpt_neox and .embed_out
        def __init__(self):
            super().__init__()
            self.gpt_neox, self.embed_out = Base(), head()

    class Mamba(torch.nn.Module):         # v3's path "M1": .backbone and .lm_head
        def __init__(self):
            super().__init__()
            self.backbone, self.lm_head = Base(), head()

    class Causal(torch.nn.Module):        # Llama / Qwen / Gemma shape: .model, .lm_head, config, base_model, output head
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.model, self.lm_head = Base(), head()
            self.config = SimpleNamespace(final_logit_softcapping=cap)

        @property
        def base_model(self):             # transformers: getattr(self, self.base_model_prefix)
            return self.model

        def get_output_embeddings(self):
            return self.lm_head

    return torch, NeoX().eval(), Mamba().eval(), Causal().eval()


def _zoo_inputs(tok, prep, family, variant):
    items, ex = prep["cells"][16], prep["exemplars"]
    seqs = [Z.model_prompt_ids(tok, R.prompt_text(it, ex, family, variant), []) for it in items]
    cands = [[tok.encode(" " + c) for c in it["codes"]] for it in items] if family == "recall" else None
    return items, seqs, cands


def test_score_cell_equals_v3_run_cell_on_both_paths(syn):
    tok, prep = syn.tok, syn.prep
    torch, neox, mamba, _ = _fake_models(V=len(tok.vocab) + 64)
    name_ids = [R.name_token_ids(tok, R.NAMES)[nm] for nm in R.NAMES]
    for path, model in (("T", neox), ("M1", mamba)):
        for family in R.FAMILIES:
            for variant in Z.VARIANTS:
                items, seqs, cands = _zoo_inputs(tok, prep, family, variant)
                ref = R.run_cell(path, model, tok, items, prep["exemplars"], family, 16, name_ids, prep["codes_len"], variant=variant, batch_size=16)
                got = Z.score_cell(path, model, family, seqs, Z.cell_targets(items, family), name_ids, cands, 16)
                assert got["_per_item"]["scores"] == ref["_per_item"]["scores"], (path, family, variant)
                for k in ("acc_raw", "acc_norm", "slot_fractions", "max_slot_fraction", "guard_ok", "ovr_auroc", "margin_mean", "near_ties"):
                    assert got[k] == ref[k], (path, family, variant, k)
                assert got["batches"] == ref["batches"] and got["batch_size"] == 16
                eq = Z.equivalence_against_reference(got["_per_item"], ref["_per_item"])
                assert eq["ok"] and eq["max_abs_score_diff"] == 0.0 and eq["pred_agree"] == 48
    # batch size does not change the numbers beyond float noise (right padding is exact on a causal model)
    items, seqs, cands = _zoo_inputs(tok, prep, "recall", "plain")
    a = Z.score_cell("T", neox, "recall", seqs, Z.cell_targets(items, "recall"), None, cands, 16)
    b = Z.score_cell("T", neox, "recall", seqs, Z.cell_targets(items, "recall"), None, cands, 5)
    assert np.allclose(a["_per_item"]["scores"], b["_per_item"]["scores"], atol=1e-4) and a["_per_item"]["pred_slot"] == b["_per_item"]["pred_slot"]


def test_attention_adapter_reaches_v3s_path_and_applies_the_softcap(syn):
    tok, prep = syn.tok, syn.prep
    name_ids = [R.name_token_ids(tok, R.NAMES)[nm] for nm in R.NAMES]
    items, seqs, _ = _zoo_inputs(tok, prep, "compare", "plain")
    torch, neox, _, causal = _fake_models(V=len(tok.vocab) + 64, cap=None)
    causal.model.load_state_dict(neox.gpt_neox.state_dict())
    causal.lm_head.load_state_dict(neox.embed_out.state_dict())
    ad = Z.AttentionAdapter(causal)
    assert ad.gpt_neox is causal.model and next(ad.parameters()) is next(causal.parameters())
    a = Z.score_cell("T", ad, "compare", seqs, Z.cell_targets(items, "compare"), name_ids)
    b = Z.score_cell("T", neox, "compare", seqs, Z.cell_targets(items, "compare"), name_ids)
    assert a["_per_item"]["scores"] == b["_per_item"]["scores"]                   # no cap: the adapter is transparent
    cap = 2.0
    _, _, _, capped = _fake_models(V=len(tok.vocab) + 64, cap=cap)
    capped.model.load_state_dict(neox.gpt_neox.state_dict())
    capped.lm_head.load_state_dict(neox.embed_out.state_dict())
    c = Z.score_cell("T", Z.AttentionAdapter(capped), "compare", seqs, Z.cell_targets(items, "compare"), name_ids)
    with torch.no_grad():
        want = []
        for s in seqs:
            h = capped.model(input_ids=torch.tensor([s])).last_hidden_state[0, -1]
            logits = cap * torch.tanh(capped.lm_head(h) / cap)
            want.append(torch.log_softmax(logits.float(), -1)[torch.tensor(name_ids)].tolist())
    assert np.allclose(c["_per_item"]["scores"], want, atol=1e-5)
    assert not np.allclose(c["_per_item"]["scores"], b["_per_item"]["scores"], atol=1e-3)
    rp = Z.right_padding_ids("T", Z.AttentionAdapter(capped), seqs[:2], name_ids)
    assert rp["max_abs_name_logprob_diff"] < 1e-5 and all(rp["argmax_agree"])
    assert Z.v3_path("mamba") == "M1" and Z.v3_path("llama") == "T" and Z.as_v3_model(neox, "gpt_neox") is neox


def _tiny_real_model(model_type, V):
    """A tiny random-weight model of each architecture class the zoo loads (transformers' own classes, no checkpoint)."""
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    common = dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4)
    kw = {}
    if model_type == "llama":
        cfg = tf.LlamaConfig(num_key_value_heads=2, max_position_embeddings=512, **common)
    elif model_type == "qwen2":
        cfg = tf.Qwen2Config(num_key_value_heads=2, max_position_embeddings=512, use_sliding_window=False, **common)
    elif model_type == "gemma2":
        cfg = tf.Gemma2Config(num_key_value_heads=2, head_dim=8, max_position_embeddings=512, sliding_window=4096,
                              final_logit_softcapping=30.0, attn_logit_softcapping=50.0, **common)
        kw["attn_implementation"] = "eager"
    elif model_type == "gpt_neox":
        cfg = tf.GPTNeoXConfig(max_position_embeddings=512, **common)
    elif model_type == "mamba":
        cfg = tf.MambaConfig(vocab_size=V, hidden_size=32, state_size=16, num_hidden_layers=2, expand=2, conv_kernel=4, time_step_rank=4)
    torch.manual_seed(0)
    model = tf.AutoModelForCausalLM.from_config(cfg, **kw).eval()
    with torch.no_grad():                                   # logits well past gemma-2's softcap of 30, so the cap matters
        model.get_output_embeddings().weight.mul_(400.0)
    return torch, model


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "gemma2", "gpt_neox", "mamba"])
def test_the_v3_path_reproduces_transformers_own_forward_on_each_architecture(model_type):
    """The wrapper (v3's readouts through the adapter or the model itself) gives the log-probs transformers' own
    forward gives — gemma-2's final softcap included — and right padding without a mask changes nothing, on CPU fp32."""
    V = 300
    torch, model = _tiny_real_model(model_type, V)
    vm = Z.as_v3_model(model, model_type)
    path = Z.v3_path(model_type)
    rng = np.random.default_rng(0)
    seqs = [list(map(int, rng.integers(5, V, size=L))) for L in (23, 31, 17, 40)]
    names = [7, 8, 9]
    cands = [[[int(x) for x in rng.integers(5, V, size=3)] for _ in range(3)] for _ in seqs]
    with torch.no_grad():
        want_c, want_r = [], []
        for s, cs in zip(seqs, cands):
            lp = torch.log_softmax(model(input_ids=torch.tensor([s]), use_cache=False).logits[0].float(), -1)
            want_c.append(lp[-1, names].tolist())
            row = []
            for c in cs:
                lpc = torch.log_softmax(model(input_ids=torch.tensor([s + c]), use_cache=False).logits[0].float(), -1)
                row.append(float(sum(lpc[len(s) - 1 + j, c[j]] for j in range(len(c)))))
            want_r.append(row)
    got_c = Z.score_cell(path, vm, "compare", seqs, [0, 1, 2, 0], names, None, 3)
    got_r = Z.score_cell(path, vm, "recall", seqs, [0, 1, 2, 0], None, cands, 3)
    assert np.allclose(got_c["_per_item"]["scores"], want_c, atol=1e-3), model_type
    assert np.allclose(got_r["_per_item"]["scores"], want_r, atol=1e-3), model_type
    rp = Z.right_padding_ids(path, vm, seqs[:2], names)
    assert rp["max_abs_name_logprob_diff"] < 1e-4 and all(rp["argmax_agree"]), (model_type, rp)
    if model_type == "gemma2":
        raw = Z.AttentionAdapter(model)
        raw.lm_head = Z.SoftcapHead(model.get_output_embeddings(), None)
        uncapped = Z.score_cell(path, raw, "compare", seqs, [0, 1, 2, 0], names, None, 3)
        assert not np.allclose(uncapped["_per_item"]["scores"], want_c, atol=1e-2)   # without the cap the readout is not the model's


def test_non_finite_scores_make_the_row_not_screenable(syn):
    torch, neox, _, _ = _fake_models(V=len(syn.tok.vocab) + 64)
    with torch.no_grad():
        neox.embed_out.weight[:] = float("nan")
    items, seqs, _ = _zoo_inputs(syn.tok, syn.prep, "compare", "plain")
    name_ids = [R.name_token_ids(syn.tok, R.NAMES)[nm] for nm in R.NAMES]
    with pytest.raises(Z.NotScreenable, match="non-finite") as e:
        Z.score_cell("T", neox, "compare", seqs, Z.cell_targets(items, "compare"), name_ids)
    assert e.value.status == "NOT_SCREENABLE__fp16_nonfinite"                  # NOTE section 7's label, not the bare one


# -- the equivalence rule, the screen rule, the summary --------------------------------------------------------

def test_equivalence_rule_against_a_reference():
    ref = {"scores": [[-1.0, -2.0, -3.0], [-1.0, -1.004, -5.0], [-3.0, -1.0, -2.0]], "pred_slot": [0, 0, 1],
           "target_slot": [0, 1, 1], "margin": [1.0, -0.004, 1.0]}
    same = json.loads(json.dumps(ref))
    assert Z.equivalence_against_reference(same, ref)["ok"]
    drift = json.loads(json.dumps(ref))
    drift["scores"][2][0] += 0.02
    assert not Z.equivalence_against_reference(drift, ref)["ok"]
    tie = json.loads(json.dumps(ref))
    tie["scores"][1] = [-1.004, -1.0, -5.0]
    tie["pred_slot"][1] = 1                                         # a flip on a committed near tie: exempt and counted
    e = Z.equivalence_against_reference(tie, ref)
    assert e["ok"] and e["exempt_near_ties"] == 1 and e["pred_disagree_non_tie"] == 0
    flip = json.loads(json.dumps(ref))
    flip["pred_slot"][0] = 1
    assert not Z.equivalence_against_reference(flip, ref)["ok"]
    with pytest.raises(R.Refusal, match="targets"):
        Z.equivalence_against_reference(dict(same, target_slot=[0, 0, 0]), ref)


def _cells(acc_c, slot_c, acc_r=0.8, abl=(0.0, 0.0)):
    def cell(acc, slot):
        return {"acc_norm": acc, "acc_raw": acc * 2 / 3 + 1 / 3, "max_slot_fraction": slot, "guard_ok": bool(slot <= R.GUARD_SLOT_MAX)}
    return {("compare", "plain"): cell(acc_c, slot_c), ("recall", "plain"): cell(acc_r, 0.4),
            ("compare", "ablated"): cell(abl[0], 0.4), ("recall", "ablated"): cell(abl[1], 0.4)}


def test_screen_rule_is_v3s_arm_clears_at_its_edges():
    s = Z.screen_from_cells(_cells(0.40, 0.80))
    assert s["compare"]["clears"] and s["recall"]["clears"] and s["clears"]
    assert not Z.screen_from_cells(_cells(0.399, 0.5))["compare"]["clears"]
    s = Z.screen_from_cells(_cells(0.9, 0.81))
    assert not s["compare"]["clears"] and not s["clears"] and s["recall"]["clears"]
    s = Z.screen_from_cells(_cells(0.6, 0.5, abl=(0.17, 0.16)))
    assert s["compare"]["clears"] and s["compare"]["ablated_above_bar"] and not s["recall"]["ablated_above_bar"]
    # a constant responder on the zoo's exactly balanced targets reads 0 and fails the guard
    targets = [0, 1, 2] * 16
    k = R.argmax_cell_summary(np.tile([-1.0, -0.5, -2.0], (48, 1)), targets)
    assert k["acc_norm"] == 0.0 and k["max_slot_fraction"] == 1.0 and not k["guard_ok"]
    k = R.argmax_cell_summary(np.eye(3)[[(t if i < 29 else (t + 1) % 3) for i, t in enumerate(targets)]], targets)
    assert k["acc_raw"] == pytest.approx(29 / 48) and k["acc_norm"] >= Z.SCREEN_BAR       # 29 of 48 is the smallest pass


def test_summary_counts_groups_apart_and_lists_what_was_not_screened():
    def row(repo, group, fam, c, r, status="SCREENED", flag=False):
        out = {"repo": repo, "group": group, "status": status, "reason": None if status == "SCREENED" else "why"}
        if status == "SCREENED":
            out["family"] = {"family": fam}
            out["screen"] = {"compare": {"clears": c, "ablated_above_bar": flag}, "recall": {"clears": r, "ablated_above_bar": False},
                             "clears": c and r}
        return out
    rows = [row("a", "base", "DECAY_SSM", False, True), row("b", "base", "ATTENTION", True, True, flag=True),
            row("c", "base", "ATTENTION", False, False), row("d", "base", "ATTENTION", False, False, status="NOT_SCREENABLE"),
            row("e", "instruct", "ATTENTION", True, False), row("f", "instruct", "ATTENTION", False, False, status="NOT_RUN__memory")]
    s = Z.summarize(rows)
    b, i = s["base"], s["instruct"]
    assert (b["M_screened"], b["N_compare"], b["N_recall"], b["N_both"], b["N_compare_with_ablated_flag"]) == (3, 1, 2, 1, 1)
    assert b["sentence"] == ("At D0, under v3's argmax readout in fp16, 1 of 3 screened base models could perform the compare task and 2 of 3 "
                             "the recall task (C_abl above 0.16 in 1 of the 1 compare passes and 0 of the 2 recall passes); n = 48 items "
                             "per family per model, a pilot-class screen that licenses nothing.")
    assert b["by_family"]["DECAY_SSM"] == {"screened": 1, "compare": 0, "recall": 1, "both": 0}
    assert [x["repo"] for x in b["not_screened"]] == ["d"]
    assert (i["M_screened"], i["N_compare"], i["N_recall"]) == (1, 1, 0)
    assert i["sentence"] == ("At D0, under v3's argmax readout in fp16, 1 of 1 screened instruct models, given raw text without a chat template, "
                             "could perform the compare task and none of the 1 the recall task (C_abl above 0.16 in 0 of the 1 compare "
                             "passes); n = 48 items per family per model, a pilot-class screen that licenses nothing.")
    assert s["excluded"][0]["repo"] == "microsoft/Phi-3.5-mini-instruct"


def test_the_sentence_never_widens_past_the_screened_models():
    """NOTE section 12.3: a zero count reads 'none of the M screened', never 'no cached model'; no screened model reads as
    such; the flag clause appears only beside nonzero pass counts."""
    z = Z.group_sentence("base", 5, 0, 0, 0, 0)
    assert z == ("At D0, under v3's argmax readout in fp16, none of the 5 screened base models could perform the compare task and none of "
                 "the 5 the recall task; n = 48 items per family per model, a pilot-class screen that licenses nothing.")
    assert "cached" not in z and "C_abl" not in z
    assert Z.group_sentence("instruct", 0, 0, 0, 0, 0).startswith("At D0, no instruct model was screened")
    assert "(C_abl above 0.16 in 2 of the 2 recall passes)" in Z.group_sentence("base", 4, 0, 0, 2, 2)
    note = NOTE.read_text(encoding="utf-8")
    sec12 = note[note.index("## 12. Addendum"):]
    assert "is withdrawn" in sec12 and "none of the M" in sec12 and "C_abl above 0.16 in F_c of the N compare passes" in sec12
    rows = [{"repo": "x", "group": "base", "status": "NOT_RUN__error", "reason": "RuntimeError: boom"}]
    s = Z.summarize(rows)
    assert s["base"]["M_screened"] == 0 and s["base"]["not_screened"][0]["status"] == "NOT_RUN__error"
    assert s["base"]["sentence"].startswith("At D0, no base model was screened")


# -- the plan's addendum (NOTE section 12) ---------------------------------------------------------------------

def _saved_tiny(tmp_path, model_type, V=300):
    torch, model = _tiny_real_model(model_type, V)
    d = tmp_path / f"tiny_{model_type}"
    model.save_pretrained(str(d))
    return torch, model, d


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "gemma2"])
def test_the_fp16_load_casts_nothing_after_it_and_records_the_buffers(tmp_path, model_type):
    """NOTE 12.1: transformers' own fp16 load, nothing after it. Every parameter is fp16 and the RoPE inv_freq stays fp32
    (what the review found the removed half() turned into fp16), recorded per model."""
    torch, model, d = _saved_tiny(tmp_path, model_type)
    m16, info = Z.load_hf_cpu(d, model_type, "float16")
    assert {str(p.dtype) for p in m16.parameters()} == {"torch.float16"} and info["dtype_loaded"] == "float16"
    inv = info["buffers_after_load"]["inv_freq"]
    assert inv and set(inv.values()) == {"torch.float32"}, inv
    if model_type == "gemma2":
        assert info["attn_implementation"] == "eager"
        assert info["final_logit_softcapping"] == 30.0 and info["attn_logit_softcapping"] == 50.0
    assert set(Z.buffer_dtypes(m16.half())["inv_freq"].values()) == {"torch.float16"}      # the cast that is gone
    assert ".half()" not in ZOO.read_text(encoding="utf-8")
    assert Z.stored_numel(d) == sum(p.numel() for p in model.parameters())
    m32, info32 = Z.load_hf_cpu(d, model_type, "float32")
    assert {str(p.dtype) for p in m32.parameters()} == {"torch.float32"} and info32["dtype_loaded"] == "float32"


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "gemma2"])
def test_the_fp32_reference_is_the_models_own_forward_and_fp16_lies_within_the_fidelity_bar(tmp_path, model_type):
    """NOTE 12.4 on tiny random models: the fp32 CPU reference reproduces transformers' own forward, its right-padding
    check passes v3's 1e-3, and an fp16 forward of the same checkpoint lies within 0.5 nats of it."""
    torch, model, d = _saved_tiny(tmp_path, model_type)
    rng = np.random.default_rng(1)
    seqs2 = [list(map(int, rng.integers(5, 300, size=L))) for L in (41, 29)]
    names = [7, 8, 9]
    ref = Z.fp32_reference(d, model_type, seqs2, names, ram=lambda: 1 << 50)
    assert ref["ran"] and ref["load"]["dtype_loaded"] == "float32" and ref["ram_needed_bytes"] > Z.RAM_HEADROOM_BYTES
    rp = ref["right_padding_fp32_cpu"]
    assert rp["max_abs_name_logprob_diff"] <= Z.RIGHT_PAD_FP32_TOL and np.asarray(rp["alone_name_logprobs"]).shape == (2, 3)
    with torch.no_grad():
        want = [torch.log_softmax(model(input_ids=torch.tensor([s]), use_cache=False).logits[0, -1].float(), -1)[names].tolist()
                for s in seqs2]
    assert np.allclose(rp["alone_name_logprobs"], want, atol=1e-4)
    m16, _ = Z.load_hf_cpu(d, model_type, "float16")
    got = Z.alone_name_logprobs(Z.v3_path(model_type), Z.as_v3_model(m16, model_type), seqs2, names)
    fid = Z.fp16_fidelity(rp["alone_name_logprobs"], got)
    assert fid["ok"] and fid["max_abs_name_logprob_diff"] < Z.FP16_FIDELITY_TOL and len(fid["argmax_agree"]) == 2
    short = Z.fp32_reference(d, model_type, seqs2, names, ram=lambda: 0)
    assert not short["ran"] and "under the" in short["reason"]
    unknown = Z.fp32_reference(d, model_type, seqs2, names, ram=lambda: None)
    assert not unknown["ran"] and "could not be read" in unknown["reason"]


def test_the_fp32_reference_padding_bar_and_the_fidelity_rule_carry_their_labels(monkeypatch, tmp_path):
    ref = [[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]]
    assert Z.fp16_fidelity(ref, ref)["ok"]
    assert Z.fp16_fidelity(ref, [[-1.0, -2.0, -3.0], [-0.5, -1.5, -3.0]])["ok"]                 # exactly 0.5: inside
    bad = Z.fp16_fidelity(ref, [[-1.0, -2.0, -3.0], [-0.5, -1.5, -3.01]])
    assert not bad["ok"] and bad["max_abs_name_logprob_diff"] == pytest.approx(0.51) and bad["argmax_agree"] == [True, True]
    assert Z.fp16_fidelity(ref, [[-3.0, -2.0, -1.0], [-0.5, -1.5, -2.5]])["argmax_agree"] == [False, True]
    with pytest.raises(ValueError, match="shapes"):
        Z.fp16_fidelity(ref, ref[:1])
    # a padding leak in the fp32 reference reads NOT_SCREENABLE__right_padding, and the reference keeps its record
    monkeypatch.setattr(Z, "stored_numel", lambda snap: 10)
    monkeypatch.setattr(Z, "load_hf_cpu", lambda snap, mt, dt: (object(), {"dtype_loaded": dt}))
    monkeypatch.setattr(Z, "as_v3_model", lambda model, mt: model)
    monkeypatch.setattr(Z, "right_padding_ids", lambda path, vm, seqs, names: {"max_abs_name_logprob_diff": 2e-3,
                                                                              "alone_name_logprobs": ref})
    leaky = Z.fp32_reference(tmp_path, "llama", [[1, 2]], [7, 8, 9], ram=lambda: 1 << 50)
    assert leaky["ran"] and leaky["right_padding_fp32_cpu"]["max_abs_name_logprob_diff"] == 2e-3
    with pytest.raises(Z.NotScreenable, match="fp32 CPU reference") as e:
        Z.fp32_reference_gate(leaky)
    assert e.value.status == "NOT_SCREENABLE__right_padding"
    Z.fp32_reference_gate(dict(leaky, right_padding_fp32_cpu={"max_abs_name_logprob_diff": 1e-3}))     # at the bar: passes
    Z.fp32_reference_gate({"ran": False, "reason": "available RAM could not be read"})                # not run: no gate


def _cpu_only_load(monkeypatch, d, to_device):
    """The GPU steps of the load replaced for a CPU run: the free-memory check passes, the snapshot is the tiny model's,
    and the move to the GPU is `to_device` (identity, or half() to stand in for v3's to_gpu_fp16)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(R, "require_free_gpu", lambda red, tag: red.setdefault("tags", []).append(tag))
    monkeypatch.setattr(R, "snapshot_dir", lambda repo, kind="models": d)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(Z, "_move_to_cuda", lambda model, mt: to_device(model))
    return torch


def test_the_whole_llama_qwen_gemma_load_path_runs_on_cpu_with_its_gates_and_records(tmp_path, monkeypatch):
    """_load_for_scoring end to end on a tiny saved llama with the GPU steps stubbed: the fp32 reference runs and is
    gated, the fp16 load keeps inv_freq fp32, right padding and fidelity are measured, and a failed gate keeps the
    record that failed it."""
    torch, model, d = _saved_tiny(tmp_path, "llama")
    _cpu_only_load(monkeypatch, d, lambda m: m)
    rng = np.random.default_rng(2)
    row = {"repo": "x/tiny-llama", "family": {"model_type": "llama"}, "_tok": None, "_name_ids": [7, 8, 9],
           "_seqs": {"compare/plain": [list(map(int, rng.integers(5, 300, size=L))) for L in (37, 52, 44)]}}
    red, rec = {}, {}
    m16, vm, info = Z.load_for_scoring(row, {"cells": {16: []}, "exemplars": []}, red, "tiny", record=rec)
    assert info is rec and red["tags"] == ["tiny", "tiny_to_cuda"]
    assert info["fp32_reference"]["ran"] and info["fp32_reference"]["gated"] and info["dtype"] == "torch.float16"
    assert info["fp32_reference"]["right_padding_fp32_cpu"]["max_abs_name_logprob_diff"] <= Z.RIGHT_PAD_FP32_TOL
    assert info["dtype_loaded"] == "float16" and set(info["buffers_after_load"]["inv_freq"].values()) == {"torch.float32"}
    assert set(info["buffers_on_gpu"]["inv_freq"].values()) == {"torch.float32"}
    assert info["fp16_fidelity"]["ok"] and info["fp16_fidelity"]["gated"] and len(info["fp16_fidelity"]["per_item"]) == 2
    assert info["right_padding_fp16_gpu"]["max_abs_name_logprob_diff"] <= Z.RIGHT_PAD_FP16_TOL
    assert isinstance(vm, Z.AttentionAdapter) and vm.model is m16
    json.dumps(R.strip_private(info))                                      # numbers and strings only: it goes into the receipt
    monkeypatch.setattr(Z, "FP16_FIDELITY_TOL", 0.0)                         # force the fidelity gate to fail
    rec2 = {}
    with pytest.raises(Z.NotScreenable, match="differ from the fp32 CPU reference") as e:
        Z.load_for_scoring(row, {"cells": {16: []}, "exemplars": []}, {}, "tiny", record=rec2)
    assert e.value.status == "NOT_SCREENABLE__fp16_fidelity"
    assert rec2["fp16_fidelity"]["max_abs_name_logprob_diff"] > 0.0 and not rec2["fp16_fidelity"]["ok"]   # the record survives
    monkeypatch.setattr(Z, "FP16_FIDELITY_TOL", 0.5)
    rec3 = {}
    Z.load_for_scoring(row, {"cells": {16: []}, "exemplars": []}, {}, "tiny", record=rec3)
    monkeypatch.setattr(Z, "available_ram_bytes", lambda: 0)                 # RAM short: no reference, the row still loads
    rec4 = {}
    Z.load_for_scoring(row, {"cells": {16: []}, "exemplars": []}, {}, "tiny", record=rec4)
    assert rec4["fp32_reference"]["ran"] is False and rec4["fp16_fidelity"]["ran"] is False
    assert "did not perform in fp16" in rec4["fp16_fidelity"]["reading"]


def test_the_v3_path_load_records_the_fp16_comparison_without_a_gate(syn, tmp_path, monkeypatch):
    """_load_for_scoring on v3's own path (a tiny gpt_neox) with the GPU steps stubbed and half() standing in for
    to_gpu_fp16: v3's fp32 padding assert runs, the fp32 alone name log-probs are kept, and the fp16 comparison is
    recorded but not gated; inv_freq is fp16 here, as on pilot2's path."""
    asserted = Z.model_tokenizer_asserts(syn.tok, syn.prep, [])
    torch, model = _tiny_real_model("gpt_neox", len(syn.tok.vocab) + 64)
    d = tmp_path / "tiny_neox"
    model.save_pretrained(str(d))
    _cpu_only_load(monkeypatch, d, lambda m: m.half())
    row = {"repo": "x/tiny-neox", "family": {"model_type": "gpt_neox"}, "_tok": syn.tok, "_name_ids": asserted["_name_ids"],
           "_seqs": asserted["_seqs"]}
    rec = {}
    monkeypatch.setattr(Z, "FP16_FIDELITY_TOL", 0.0)                         # a gate here would fail: none applies on this path
    _, vm, info = Z.load_for_scoring(row, syn.prep, {}, "tiny-neox", record=rec)
    assert info["right_padding_fp32_cpu"]["max_abs_logit_diff"] <= R.FP32_TOL
    assert info["fp32_reference"]["gated"] is False and np.asarray(info["fp32_reference"]["alone_name_logprobs"]).shape == (2, 3)
    assert info["fp16_fidelity"]["gated"] is False and "max_abs_name_logprob_diff" in info["fp16_fidelity"]
    assert info["dtype"] == "torch.float16" and set(info["buffers_on_gpu"]["inv_freq"].values()) == {"torch.float16"}


def test_an_out_of_memory_error_while_moving_to_the_gpu_reads_not_run_memory():
    torch = pytest.importorskip("torch")

    class Boom:
        def to(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(Z.NotRun) as e:
        Z._move_to_cuda(Boom(), "llama")
    assert e.value.status == "NOT_RUN__memory" and "moving the model to the GPU" in str(e.value)


def test_every_status_the_runner_writes_is_named_in_the_plan():
    """NOTE 12.2: the labels in the receipt are the plan's, each named in the note; no bare label where the plan names
    a suffixed one."""
    src = ZOO.read_text(encoding="utf-8")
    literals = set(re.findall(r'"((?:NOT_[A-Z]+(?:__[a-z0-9_]+)?)|SCREENED)"', src))
    assert literals == set(Z.STATUSES)
    assert len(Z.STATUSES) == 9 and Z.STATUS_ERROR == "NOT_RUN__error"
    note = NOTE.read_text(encoding="utf-8")
    for st in Z.STATUSES:
        assert re.search(r"(?<![A-Za-z_])" + re.escape(st) + r"(?![A-Za-z_])", note), st
    sec12 = note[note.index("## 12. Addendum"):]
    for st in Z.STATUSES:
        assert f"| {st} |" in sec12, st
    assert 'status="NOT_' not in src and "status='NOT_" not in src
    assert Z.NotScreenable("x").status == "NOT_SCREENABLE"


def test_the_partial_log_is_append_only_and_blocks_a_silent_rerun(tmp_path, monkeypatch):
    log = tmp_path / "zoo_d0_screen.partial.jsonl"
    monkeypatch.setattr(Z, "PARTIAL_LOG", log)
    monkeypatch.setattr(Z, "RECEIPT", tmp_path / "zoo_d0_screen.json")
    assert Z.partial_facts() is None
    Z.append_partial({"event": "run_start", "_private": 1, "x": np.float64(0.5), "s": "−0.062"})
    Z.append_partial({"event": "model", "repo": "a/b", "row": {"status": "SCREENED"}})
    raw = log.read_bytes()
    assert raw.count(b"\n") == 2 and b"\r\n" not in raw and not raw.startswith(b"\xef\xbb\xbf") and b"_private" not in raw
    lines = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["run_start", "model"] and lines[0]["x"] == 0.5 and lines[0]["s"] == "−0.062"
    assert all("time" in x for x in lines)
    facts = Z.partial_facts()
    assert facts["lines"] == 2 and facts["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    with pytest.raises(R.Refusal, match="partial.jsonl exists"):
        Z.run_screen(log=lambda *a: None)
    monkeypatch.setattr(Z, "git_facts", lambda: {"head": None})          # with the operator's flag the run goes past it
    with pytest.raises(R.Refusal, match="no git facts"):
        Z.run_screen(log=lambda *a: None, allow_after_partial=True)
    assert log.read_bytes() == raw                                          # a refusal before the run starts appends nothing


def test_an_error_outside_a_row_leaves_a_record_and_a_refusal_before_the_start_leaves_no_log(tmp_path, monkeypatch):
    log, ref = tmp_path / "zoo_d0_screen.partial.jsonl", tmp_path / "zoo_d0_screen_refusal.json"
    monkeypatch.setattr(Z, "PARTIAL_LOG", log)
    monkeypatch.setattr(Z, "REFUSAL_RECEIPT", ref)
    monkeypatch.setitem(Z._STATE, "run_started", False)

    def boom(log=print, allow_after_partial=False):
        Z._STATE["run_started"] = True
        raise RuntimeError("device lost")

    monkeypatch.setattr(Z, "run_screen", boom)
    assert Z.main([]) == 3
    rec = json.loads(ref.read_text(encoding="utf-8"))
    assert rec["refused"] is False and rec["error"] == "RuntimeError: device lost" and "Traceback" in rec["traceback"]
    assert rec["partial_log"]["lines"] == 1
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["event"] == "error"
    log.unlink()
    monkeypatch.setitem(Z._STATE, "run_started", False)

    def refuse(log=print, allow_after_partial=False):
        raise R.Refusal("not committed")

    monkeypatch.setattr(Z, "run_screen", refuse)
    assert Z.main([]) == 2 and not log.exists()
    rec = json.loads(ref.read_text(encoding="utf-8"))
    assert rec["refused"] is True and rec["reason"] == "not committed" and rec["partial_log"] is None


def test_the_equivalence_line_names_the_four_cells_and_skips_the_load_record():
    cell = {"n": 48, "max_abs_score_diff": 0.0, "pred_agree": 48, "pred_disagree_non_tie": 0, "exempt_near_ties": 0, "ok": True}
    eq = {f"{f}/{v}": dict(cell) for f in Z.FAMILIES for v in Z.VARIANTS}
    eq["ok"] = True
    eq["load"] = {"model_type": "mamba", "fp32_reference": {"ran": True}}       # the run at acd5b1e9 stopped here: KeyError
    line = Z.equivalence_log_line("M1", eq)
    assert line.startswith("  wrapper equivalence M1: compare/plain max|d| 0.00e+00 disagree 0") and line.endswith("-> ok")
    assert all(k in line for k in ("compare/ablated", "recall/plain", "recall/ablated")) and "load" not in line
    eq["recall/ablated"] = dict(cell, max_abs_score_diff=0.5, pred_disagree_non_tie=3, ok=False)
    eq["ok"] = False
    assert "recall/ablated max|d| 5.00e-01 disagree 3" in Z.equivalence_log_line("T", eq)
    assert Z.equivalence_log_line("T", eq).endswith("-> FAILED")


def test_the_run_goes_end_to_end_on_cpu_with_every_gpu_step_stubbed(syn, tmp_path, monkeypatch):
    """The run's own code around the GPU steps: the result, the M1 and T equivalence lines (the run at acd5b1e9 stopped
    on the line with a KeyError once M1's load record sat beside the cells), every row branch, the partial log, the
    receipt and the summary. The loads, the forwards and the device are stubbed; nothing touches a GPU."""
    torch = pytest.importorskip("torch")
    receipt, log = tmp_path / "zoo_d0_screen.json", tmp_path / "zoo_d0_screen.partial.jsonl"
    monkeypatch.setattr(Z, "RECEIPT", receipt)
    monkeypatch.setattr(Z, "PARTIAL_LOG", log)
    monkeypatch.setattr(Z, "REFUSAL_RECEIPT", tmp_path / "zoo_d0_screen_refusal.json")
    monkeypatch.setitem(Z._STATE, "run_started", False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(Z, "git_facts", lambda: {"head": "0" * 40})
    monkeypatch.setattr(Z, "require_committed", lambda git: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(R, "record_versions", lambda res: res["config"].setdefault("versions", {"stub": True}))
    monkeypatch.setattr(R, "release", lambda redteam=None, tag="": 0)
    monkeypatch.setattr(Z, "peak_mb", lambda: 0.0)
    rng = np.random.default_rng(0)
    monkeypatch.setattr(R, "next_token_scores_batch", lambda path, model, seqs, ids: rng.normal(size=(len(seqs), 3)))
    monkeypatch.setattr(R, "recall_loglik_batch", lambda path, model, seqs, cands: rng.normal(size=len(seqs)))

    asserted = Z.model_tokenizer_asserts(syn.tok, syn.prep, [])
    rows = []
    for repo, group, role in Z.MODELS:
        mt = "mamba" if "mamba" in repo else "gpt_neox" if "pythia" in repo else "llama"
        rows.append({"repo": repo, "group": group, "v3_role": role, "status": "READY", "reason": None, "revision": "r",
                     "family": {"family": "DECAY_SSM" if mt == "mamba" else "ATTENTION", "model_type": mt},
                     "tokenizer": asserted["record"], "_seqs": asserted["_seqs"], "_cands": asserted["_cands"],
                     "_name_ids": asserted["_name_ids"], "_prompt_sha256": asserted["_prompt_sha256"], "_tok": syn.tok})
    k = [r["repo"] for r in rows].index("Qwen/Qwen2.5-3B-Instruct")
    rows[k] = dict({a: b for a, b in rows[k].items() if not a.startswith("_")}, status=Z.STATUS_NOT_CACHED, reason="no cached snapshot")
    fail = {"Qwen/Qwen2.5-3B": Z.NotScreenable("fp16 off the reference", Z.STATUS_FP16_FIDELITY),
            "google/gemma-2-2b-it": RuntimeError("device lost")}

    def load(row, prep, red, tag, record=None):
        info = record if record is not None else {}
        info.update(model_type=row["family"]["model_type"], fp16_fidelity={"max_abs_name_logprob_diff": np.float64(0.75), "ok": False})
        if record is not None and row["repo"] in fail:
            raise fail[row["repo"]]
        return object(), object(), info

    def equivalence(arm, path, model, tok, prep2, rec):
        cell = {"n": 48, "max_abs_score_diff": 0.0, "pred_agree": 48, "pred_disagree_non_tie": 0, "exempt_near_ties": 0, "ok": True}
        return dict({f"{f}/{v}": dict(cell) for f in Z.FAMILIES for v in Z.VARIANTS}, ok=True)

    monkeypatch.setattr(Z, "load_for_scoring", load)
    monkeypatch.setattr(Z, "wrapper_equivalence", equivalence)
    cpu = {"prep": syn.prep, "lists": syn.lists, "redteam": {"balance_by_distance": {}}, "disjointness": {"stub": True},
           "pilot2": {"prep": None, "items_sha256": "x", "prompts_matched": 192}, "rows": rows,
           "identical_bytes": {"identical": True}, "receipts": {"pilot2": {}}, "seconds": 0.0}
    monkeypatch.setattr(Z, "cpu_phase", lambda log=print: cpu)

    lines = []
    assert Z.run_screen(log=lines.append) == 0
    text = chr(10).join(lines)
    assert "wrapper equivalence M1: compare/plain max|d| 0.00e+00 disagree 0" in text and "wrapper equivalence T:" in text
    res = json.loads(receipt.read_text(encoding="utf-8"))
    st = {repo: m["status"] for repo, m in res["models"].items()}
    assert list(st) == [m[0] for m in Z.MODELS]
    assert st["Qwen/Qwen2.5-3B"] == Z.STATUS_FP16_FIDELITY and st["google/gemma-2-2b-it"] == Z.STATUS_ERROR
    assert st["Qwen/Qwen2.5-3B-Instruct"] == Z.STATUS_NOT_CACHED and sum(v == Z.STATUS_SCREENED for v in st.values()) == 10
    assert res["models"]["Qwen/Qwen2.5-3B"]["load"]["fp16_fidelity"]["max_abs_name_logprob_diff"] == 0.75
    assert "device lost" in res["models"]["google/gemma-2-2b-it"]["error_traceback"]
    assert res["wrapper_equivalence"]["M1"]["ok"] and res["wrapper_equivalence"]["T"]["load"]["model_type"] == "gpt_neox"
    assert res["summary"]["base"]["M_screened"] == 7 and res["summary"]["instruct"]["M_screened"] == 3
    for g in Z.GROUPS:
        assert res["summary"][g]["sentence"] in text
    events = [json.loads(x)["event"] for x in log.read_text(encoding="utf-8").splitlines()]
    assert events[0] == "run_start" and events.count("wrapper_equivalence") == 2
    assert events.count("model") == 12                                   # the NOT_CACHED row is settled before the run: receipt only
    assert events[-1] == "receipt_written"


# -- refusals and discipline -----------------------------------------------------------------------------------

def test_the_run_refuses_when_the_receipt_exists_or_the_text_is_not_committed(tmp_path, monkeypatch):
    rec = tmp_path / "zoo_d0_screen.json"
    rec.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Z, "RECEIPT", rec)
    with pytest.raises(R.Refusal, match="runs once"):
        Z.run_screen(log=lambda *a: None)
    good = {"head_blob_sha256": "x", "working_copy_matches_head": True, "path": "p"}
    keys = ("note", "runner", "v3_runner", "receipt_smoke", "receipt_pilot", "receipt_pilot2")
    assert Z.COMMITTED_KEYS == keys and tuple(k for k, _ in Z.committed_files()) == keys
    assert [p for _, p in Z.committed_files()][3:] == [R.SMOKE_RECEIPT, R.PILOT1_RECEIPT, R.PILOT2_RECEIPT]
    Z.require_committed({k: good for k in keys})
    for k in keys:
        bad = {j: good for j in keys}
        bad[k] = dict(good, working_copy_matches_head=False, path=k)
        with pytest.raises(R.Refusal, match="not committed, or differs"):
            Z.require_committed(bad)
        bad[k] = dict(good, head_blob_sha256=None, path=k)
        with pytest.raises(R.Refusal, match="not committed, or differs"):
            Z.require_committed(bad)
        del bad[k]
        with pytest.raises(R.Refusal, match="no git facts"):
            Z.require_committed(bad)


def test_check_mode_hides_the_gpu_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(Z, "RECEIPT", tmp_path / "zoo_d0_screen.json")
    monkeypatch.setattr(Z, "cpu_phase", lambda log=print: {"rows": [], "identical_bytes": {"identical": True}, "seconds": 0.0})
    assert Z.run_check(log=lambda *a: None) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not (tmp_path / "zoo_d0_screen.json").exists()


def test_receipts_are_written_utf8_lf_without_bom(tmp_path):
    p = tmp_path / "r.json"
    R.write_json(p, {"a": np.float64(0.5), "_private": 1, "per_item": {"x": [np.int64(1)]}, "s": "−0.062"})
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw and b"_private" not in raw
    assert json.loads(raw.decode("utf-8")) == {"a": 0.5, "per_item": {"x": [1]}, "s": "−0.062"}


def test_the_lab_never_prints_these_words():
    """The banned list is the lab's own BANNED tuple in scripts/zenodo_publish_v7_48_0.py, read from its text so that
    this file does not print the words either. Also: no BOM."""
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    for p in (NOTE, ZOO, TEST_FILE):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        hits = [h.group(0) for h in banned.finditer(raw.decode("utf-8"))]
        assert not hits, f"{p.name}: {hits}"


# -- the committed record: the receipt, the partial log and the crashed attempt's refusal file ------------------

def test_the_committed_record_checks_out_byte_for_byte():
    """NOTE sections 14 and 15 name the receipt and the crashed attempt's refusal file by sha256, and the receipt names
    the crashed attempt's log by the sha256 of its leading lines (runtime.prior_partial_log). All three files are
    committed and pinned -text, so each hash a reader checks is a hash of the committed bytes on any checkout."""
    import hashlib

    receipt = PAPER / "zoo_d0_screen.json"
    log = PAPER / "zoo_d0_screen.partial.jsonl"
    refusal = PAPER / "zoo_d0_screen_refusal.json"
    if not all(p.exists() for p in (receipt, log, refusal)):
        pytest.skip("the zoo receipt and the crashed attempt's record are not in this checkout")

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    raw_receipt, raw_log, raw_refusal = receipt.read_bytes(), log.read_bytes(), refusal.read_bytes()
    for p, raw in ((receipt, raw_receipt), (log, raw_log), (refusal, raw_refusal)):
        assert b"\r\n" not in raw and not raw.startswith(b"\xef\xbb\xbf"), f"{p.name}: not the LF bytes the hashes name"
    note = NOTE.read_text(encoding="utf-8")
    sec14 = note[note.index("## 14. "):note.index("## 15. ")]
    sec15 = note[note.index("## 15. "):]
    assert sha(raw_receipt) in sec15 and sha(raw_refusal) in sec14

    res = json.loads(raw_receipt.decode("utf-8"))
    prior = res["runtime"]["prior_partial_log"]
    lines = raw_log.split(b"\n")
    assert lines[-1] == b""                                                    # every event ends in LF
    head = b"".join(x + b"\n" for x in lines[:prior["lines"]])
    assert prior["lines"] == 3 and sha(head) == prior["sha256"] and prior["sha256"] in sec14
    crashed = json.loads(raw_refusal.decode("utf-8"))
    assert crashed["refused"] is False and crashed["error"] == "KeyError: 'max_abs_score_diff'"
    assert crashed["partial_log"] == prior

    events = [json.loads(x) for x in raw_log.decode("utf-8").splitlines()]
    assert [e["event"] for e in events[:3]] == ["run_start", "wrapper_equivalence", "error"]
    assert events[3]["event"] == "run_start" and events[3]["prior_partial_log"] == prior
    assert [e["arm"] for e in events if e["event"] == "wrapper_equivalence"] == ["M1", "M1", "T"]
    assert [e["repo"] for e in events if e["event"] == "model"] == list(res["models"]) == [m[0] for m in Z.MODELS]
    assert events[-1]["event"] == "receipt_written" and events[-1]["sha256"] == sha(raw_receipt)

    flat = re.sub(r"\s*\n>\s*", " ", sec15)                                    # the blockquoted sentences, unwrapped
    for g in Z.GROUPS:
        assert res["summary"][g]["sentence"] in flat, g
    pinned = {x.strip() for x in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()}
    for p in (receipt, log, refusal):
        assert f"papers/frequency-resonance/{p.name} -text" in pinned, p.name

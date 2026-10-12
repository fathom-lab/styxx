# -*- coding: utf-8 -*-
"""CPU-only tests for papers/frequency-resonance/run_realmodel_bridge_v3.py — no model is loaded.

What is checked, against PREREG_realmodel_bridge_v3_2026_10_11.md as amended by 17.2:
  * exact per-cell balance computed from the generated items (section 4, assert 3): the correct slot, the queried
    slot, the last exemplar's compare and recall slots, their coincidences, the overwrite slot pairs;
  * no filler sentence reused across items or cells, no digit / name / 'code' in any filler (assert 4), the
    exemplar names excluded at draw time with the pool itself unchanged;
  * the prompt length bound (assert 7), equal-length candidate codes (assert 5), single-token names;
  * the C_near and C_overwrite constructions at D1 (16-token tail; exactly D with the re-plant last);
  * the gates block parses through the lab's own scanner and all 256 boolean assignments reach a row;
  * chance normalization maps a constant responder to exactly 0 in both families (chance 1/3);
  * the paired item bootstrap is deterministic under its seed, with equal weights for the two families;
  * the budget extrapolation (both arms, C_abl everywhere, the two D1 controls), the format line, the main-run
    refusals, LF-only receipts, the sha256 lists and the disjointness assert;
  * tokenizer-only: the cached GPT-NeoX tokenizer of both arms has single-token names, two-token codes, an
    8-token re-plant sentence and identical ids on every prompt (skipped when not cached).

The item generator is exercised with a GPT-2-style word tokenizer and a synthetic sentence pool, so the
tests need neither torch nor the wikitext cache. A CPU-only exercise of the Mamba code path on the cached
state-spaces/mamba-130m-hf runs only when BRIDGE_V3_CPU_MODEL_TEST=1 is set. The gates block is compared with
the judged plan's only when BRIDGE_V3_PLAN_PATH names that plan, which lives outside the repository.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "papers" / "frequency-resonance"
RUNNER = PAPER / "run_realmodel_bridge_v3.py"
PREREG = PAPER / "PREREG_realmodel_bridge_v3_2026_10_11.md"
TEST_FILE = Path(__file__).resolve()

sys.path.insert(0, str(ROOT))


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_realmodel_bridge_v3_under_test", str(RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_runner()


class WordTok:
    """A GPT-2-shaped pretokenizer with one id per pretoken: compositional at spaces and newlines, like NeoX."""

    _re = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

    def __init__(self):
        self.vocab = {}
        self.name = "word-tok"

    def encode(self, text):
        return [self.vocab.setdefault(p, len(self.vocab)) for p in self._re.findall(text)]


WORDS = ("river stone garden window silver morning harbour meadow lantern voyage quiet marble thunder "
         "orchard pilgrim velvet canyon ember forest glacier island jasmine kettle ladder mirror nectar "
         "oyster pepper quarry ribbon saddle timber umbrella violin walnut yellow zephyr anchor basket "
         "candle dagger engine falcon goblet hammer iron jacket kernel lemon magnet needle oak parcel "
         "quill rocket sandal tunnel urn vessel wagon yarn").split()


def make_sentences(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(R.MIN_WORDS, 21))
        ws = [WORDS[i] for i in rng.integers(0, len(WORDS), size=k)]
        ws[0] = ws[0].capitalize()
        out.append(" ".join(ws) + ".")
    return out


def make_cells(tok, n, distances, seed=R.SEED_ITEMS, n_sent=12000, region="main"):
    sents = make_sentences(n_sent)
    codes, codes_len = R.eligible_codes(tok)
    pools = R.partition_codes(codes)
    pool = R.SentencePool(sents, np.random.default_rng(1).permutation(len(sents)))
    exemplars = R.build_exemplars(pools[f"{region}_exemplar"], np.random.default_rng(seed))
    cells = {D: R.build_items(n, D, seed, pool, pools[f"{region}_test"], exemplars, tok) for D in distances}
    return cells, exemplars, pool, codes_len


@pytest.fixture(scope="module")
def small_cells():
    tok = WordTok()
    cells, exemplars, pool, codes_len = make_cells(tok, 48, (16, 1024))
    return tok, cells, exemplars, pool, codes_len


# -- balance ---------------------------------------------------------------------------------------------

def test_exact_balance_per_cell_small(small_cells):
    tok, cells, exemplars, pool, _ = small_cells
    for D, items in cells.items():
        got = R.assert_balance(items, exemplars)
        assert got["n"] == 48
        assert got["max_slot"] == [16, 16, 16] and got["q_slot"] == [16, 16, 16]
        assert got["last_exemplar"] == [8, 16, 16, 8]
        assert got["last_exemplar_compare_slot"] == [16, 16, 16] and got["last_exemplar_recall_slot"] == [16, 16, 16]
        assert got["last_compare_slot_is_correct"] == 16 and got["last_recall_slot_is_queried"] == 16
        assert got["ow_slot"] == [16, 16, 16] and all(v == 8 for v in got["ow_pairs"].values())
        assert got["ow_never_correct"] and got["largest_at_max_slot"] and got["codes_distinct"] and got["codes_plantable"]
        assert abs(got["queried_is_largest"] - 16) <= 3           # near n/3, not exact: reported, never asserted exactly
        for it in items:
            assert sorted(it["perm"]) == [0, 1, 2, 3] and it["perm"][-1] == it["last_ex"]
            assert exemplars[it["perm"][-1]]["max_slot"] in (0, 1, 2)
            assert int(it["codes"][it["max_slot"]]) == max(int(c) for c in it["codes"])
        if D == 1024:
            assert got["ow_code_larger_than_all"] and got["ow_replant_names_ow_slot"] and got["ow_filler_ends_with_replant"]


def test_exact_balance_at_600_matches_the_prereg_counts():
    tok = WordTok()
    cells, exemplars, _pool, _ = make_cells(tok, 600, (16,), n_sent=6000)
    got = R.assert_balance(cells[16], exemplars)
    assert got["max_slot"] == [200, 200, 200] and got["q_slot"] == [200, 200, 200]
    assert got["last_exemplar"] == [100, 200, 200, 100]
    assert got["last_exemplar_compare_slot"] == [200, 200, 200] and got["last_exemplar_recall_slot"] == [200, 200, 200]
    assert got["last_compare_slot_is_correct"] == 200 and got["last_recall_slot_is_queried"] == 200
    assert got["ow_slot"] == [200, 200, 200] and all(v == 100 for v in got["ow_pairs"].values())
    assert abs(got["queried_is_largest"] - 200) <= 3


def test_balance_refuses_an_unbalanced_cell(small_cells):
    tok, cells, exemplars, _, _ = small_cells
    items = [dict(it) for it in cells[16]]
    a, b = items[0], next(it for it in items if it["max_slot"] != items[0]["max_slot"])
    a["codes"], a["max_slot"], a["ow_slot"] = list(b["codes"]), b["max_slot"], b["ow_slot"]
    with pytest.raises(R.Refusal, match="exact balance"):
        R.assert_balance(items, exemplars)


def test_design_rows_balance_for_every_admissible_n():
    exemplars = [{"q_slot": q, "max_slot": m} for q, m in zip(R.EXEMPLAR_Q_SLOTS, R.EXEMPLAR_MAX_SLOTS)]
    for n in (6, 24, 48, 96, 600):
        rows = R.design_rows(n, np.random.default_rng(n))
        items = [{"codes": ["1000", "2000", "3000"], "perm": [k for k in range(4) if k != r["last_ex"]] + [r["last_ex"]], **r}
                 for r in rows]
        for it in items:
            codes = ["0", "0", "0"]
            codes[it["max_slot"]] = "3000"
            rest = iter(["1000", "2000"])
            for s in range(3):
                if s != it["max_slot"]:
                    codes[s] = next(rest)
            it["codes"] = codes
        got = R.balance_counts(items, exemplars)
        want = R.expected_balance(n)
        assert all(got[k] == want[k] for k in want), n
        assert abs(got["queried_is_largest"] - n // 3) <= 3, n
    for bad in (0, 25, 27, 9):
        with pytest.raises(R.Refusal):
            R.design_rows(bad, np.random.default_rng(0))


def test_exemplars_slots_and_largest_code():
    exs = R.build_exemplars([str(1000 + 7 * i) for i in range(12)], np.random.default_rng(0))
    assert [e["q_slot"] for e in exs] == [0, 1, 2, 0]
    assert [e["max_slot"] for e in exs] == [1, 0, 2, 1]
    for e in exs:
        assert int(e["codes"][e["max_slot"]]) == max(int(c) for c in e["codes"])
        assert len(set(e["codes"])) == 3
    assert sum(e["q_slot"] == e["max_slot"] for e in exs) == 1          # the queried code is the largest in one exemplar of four
    with pytest.raises(R.Refusal):
        R.build_exemplars(["1000"] * 5, np.random.default_rng(0))


def test_arrange_codes_puts_the_largest_at_the_slot():
    for slot in range(3):
        out = R.arrange_codes(["4521", "1200", "8899"], slot, np.random.default_rng(slot))
        assert out[slot] == "8899" and sorted(out) == ["1200", "4521", "8899"]


# -- fillers ---------------------------------------------------------------------------------------------

def test_fillers_exact_length_distinct_and_clean(small_cells):
    tok, cells, exemplars, pool, _ = small_cells
    info = R.assert_fillers(cells, exemplars)
    assert info["sentences_used"] == len(set(pool.used)) == len(pool.used)
    for D, items in cells.items():
        for it in items:
            assert len(tok.encode(" " + it["filler"])) == D
            assert R.filler_violations(it["filler"]) == [] and R.filler_violations(it["neutral"]) == []
            assert len(it["filler_sentences"]) >= 1 and len(it["neutral_sentences"]) >= 1
            if D == 1024:
                assert len(tok.encode(" " + it["near_filler"])) == R.D0
                assert len(tok.encode(" " + it["ow_filler"])) == D
                assert it["ow_filler"].endswith(it["ow_replant"]) and it["ow_replant"] == R.replant_sentence(it["ow_slot"], it["ow_code"])
                assert R.filler_violations(it["ow_filler_head"]) == []
                assert R.filler_violations(it["ow_replant"])                       # the re-plant names a slot and the word code: not filler
            else:
                assert "ow_filler" not in it and "near_filler" not in it


def test_filler_reuse_is_refused(small_cells):
    tok, cells, exemplars, _, _ = small_cells
    forged = {D: [dict(it) for it in items] for D, items in cells.items()}
    forged[1024][0]["filler_sentences"] = list(forged[16][0]["filler_sentences"])
    with pytest.raises(R.Refusal, match="reused"):
        R.assert_fillers(forged, exemplars)
    forged = {D: [dict(it) for it in items] for D, items in cells.items()}
    forged[1024][1]["near_sentences"] = list(forged[16][0]["neutral_sentences"])
    with pytest.raises(R.Refusal, match="reused"):
        R.assert_fillers(forged, exemplars)
    with pytest.raises(R.Refusal, match="pilot"):
        R.assert_fillers(cells, exemplars, prior_used={cells[16][0]["filler_sentences"][0]})


def test_filler_violations_catch_digits_names_and_code():
    assert R.filler_violations("The Alpha squad sent code 4 to Gamma and Delta.") == ["4", "alpha", "code", "delta", "gamma"]
    assert R.filler_violations("Sigma, Theta and Omega are letters.") == ["omega", "sigma", "theta"]
    assert R.filler_violations("A quiet river under a silver morning.") == []
    assert not R.eligible_sentence("The code was 1234.")
    assert not R.eligible_sentence("Alphabet soup is served daily here.")
    assert R.eligible_sentence("The pilot qualification most relevant here is the licence.")
    # the pool itself is unchanged (identity with the committed receipts): a sentence with an exemplar name is still
    # eligible for the pool and is skipped at draw time instead
    assert R.eligible_sentence("The sigma factor binds the promoter region tightly.")
    assert R._SENT_BAD.pattern == r"\d|alpha|bravo|delta|code|<unk>"


def test_sentence_pool_skips_named_sentences_at_draw_time():
    sents = ["A quiet river runs under the silver morning.", "The sigma factor binds the promoter region tightly.",
             "An omega shaped harbour wall protects the boats.", "Thunder rolled over the orchard all night long."]
    pool = R.SentencePool(sents, [1, 0, 2, 3])
    assert pool.draw() == 0 and pool.skipped == [1]
    assert pool.draw() == 3 and pool.skipped == [1, 2] and pool.used == [0, 3]
    with pytest.raises(R.Refusal, match="exhausted"):
        pool.draw()


def test_sentence_pool_fill_is_exact_and_consumes_without_replacement():
    tok = WordTok()
    sents = make_sentences(500, seed=3)
    pool = R.SentencePool(sents, list(range(500)))
    for target in (3, 16, 64, 257):
        text, used, parts = pool.fill(target, lambda t: len(tok.encode(" " + t)))
        assert len(tok.encode(" " + text)) == target
        assert len(set(used)) == len(used) and text == " ".join(parts)
    assert len(set(pool.used)) == len(pool.used)


def test_overwrite_filler_keeps_the_head_and_ends_with_the_replant():
    tok = WordTok()
    sents = make_sentences(400, seed=5)
    pool = R.SentencePool(sents, list(range(400)))
    measure = lambda t: len(tok.encode(" " + t))                       # noqa: E731
    base, _used, parts = pool.fill(200, measure)
    replant = R.replant_sentence(2, "9412")
    text, head, used = R.overwrite_filler(parts, replant, 200, pool, measure)
    assert measure(text) == 200 and text.endswith(replant) and text == (head + " " + replant)
    assert head.startswith(parts[0]) and R.filler_violations(head) == []
    # the gap draw consumed sentences without reuse
    assert len(set(pool.used)) == len(pool.used)


def test_wikitext_line_normalisation_and_split():
    line = " The band @-@ members recorded in Paris , France . It sold 1 @,@ 000 copies . A quiet river runs under the silver morning . Too short ."
    sents = R.sentences_from_lines([line])
    assert sents == ["The band-members recorded in Paris, France.",            # @-@ and ' ,' normalised
                     "A quiet river runs under the silver morning."]          # the digit sentence and the short one are dropped
    assert R.normalize_wikitext_line(" = Heading = ") == ""
    assert R.sentences_from_lines([line] * 5, cap=3) == sents + sents[:1]


def test_split_orders_are_disjoint_seeded_and_keep_the_committed_slices():
    o = R.split_orders(20000, reserve=1000, smoke_reserve=500, pilot2_reserve=400)
    assert set(o) == {"main", "pilot2", "smoke", "pilot"}
    assert len(o["main"]) == 18100 and len(o["pilot2"]) == 400 and len(o["smoke"]) == 500 and len(o["pilot"]) == 1000
    sets = {k: set(v.tolist()) for k, v in o.items()}
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b]
    assert set().union(*sets.values()) == set(range(20000))
    o2 = R.split_orders(20000, reserve=1000, smoke_reserve=500, pilot2_reserve=400)
    assert all(o[k].tolist() == o2[k].tolist() for k in o)
    # the smoke and pilot slices are what the committed receipts drew from: the pre-17.2 three-way cut
    perm = np.random.default_rng(R.SEED_ITEMS).permutation(20000)
    old_smoke = perm[-1500:-1000][np.random.default_rng(R.SEED_SMOKE).permutation(500)]
    old_pilot = perm[-1000:][np.random.default_rng(R.SEED_PILOT).permutation(1000)]
    assert o["smoke"].tolist() == old_smoke.tolist() and o["pilot"].tolist() == old_pilot.tolist()
    # the pilot2 slice is the former main region's tail, reshuffled by its own seed
    assert set(o["pilot2"].tolist()) == set(perm[-1900:-1500].tolist())
    o3 = R.split_orders(20000, reserve=1000, smoke_reserve=500, pilot2_reserve=400, seed_pilot2=R.SEED_PILOT2 + 1)
    assert set(o3["pilot2"].tolist()) == sets["pilot2"] and o3["pilot2"].tolist() != o["pilot2"].tolist()
    with pytest.raises(R.Refusal):
        R.split_orders(R.PILOT_RESERVE + R.SMOKE_RESERVE + R.PILOT2_RESERVE)


# -- prompts ---------------------------------------------------------------------------------------------

def test_prompt_length_bound_and_compositionality(small_cells):
    tok, cells, exemplars, _, codes_len = small_cells
    info = R.assert_prompts(cells, exemplars, {"a": tok, "b": tok}, {"a": codes_len, "b": codes_len})
    assert info["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert info["prompts_checked"] == 2 * (48 * 2 + 48 * 4)             # two families; two variants at D0, four at D1
    it = cells[1024][0]
    parts = R.prompt_parts(it, exemplars, "compare")
    ids = R.compositional_ids(tok, parts)
    assert ids is not None and len(tok.encode(parts[1])) == 1024
    abl = tok.encode(R.prompt_text(it, exemplars, "compare", "ablated"))
    assert len(abl) == len(ids)
    assert parts[2] == "\nQuestion: Which name has the largest code: alpha, gamma or delta? Answer:"
    rec = R.prompt_parts(it, exemplars, "recall")[2]
    assert rec == f"\nQuestion: What is the {R.NAMES[it['q_slot']]} code? Answer:"
    assert R.prompt_text(it, exemplars, "compare").count("\n\n") == R.N_EXEMPLARS
    near = R.prompt_parts(it, exemplars, "compare", "near")
    assert R.compositional_ids(tok, near) is not None and len(tok.encode(near[1])) == R.D0
    assert near[0].endswith("\n\n" + R.code_block(it["codes"])) and it["filler"] in near[0]
    ow = R.prompt_parts(it, exemplars, "compare", "overwrite")
    assert R.compositional_ids(tok, ow) is not None and len(tok.encode(ow[1])) == 1024
    assert ow[1].endswith(R.replant_sentence(it["ow_slot"], it["ow_code"])) and ow[0] == parts[0]
    assert R.item_variants(it) == ("plain", "ablated", "near", "overwrite") and R.item_variants(cells[16][0]) == ("plain", "ablated")
    with pytest.raises(R.Refusal):
        R.prompt_parts(it, exemplars, "compare", "sideways")
    # exemplar names appear in the exemplars, test names in the test block, never the other way round
    exs_text = parts[0][: parts[0].rfind("\n\n")]
    assert all(nm in exs_text for nm in R.EXEMPLAR_NAMES) and not any(nm in exs_text for nm in R.NAMES)
    assert not any(nm in parts[0][parts[0].rfind("\n\n"):] for nm in R.EXEMPLAR_NAMES)


def test_prompt_length_bound_refused_when_exceeded(small_cells):
    tok, cells, exemplars, _, codes_len = small_cells
    old = R.MAX_PROMPT_TOKENS
    R.MAX_PROMPT_TOKENS = 100
    try:
        with pytest.raises(R.Refusal, match="bound"):
            R.assert_prompts(cells, exemplars, {"a": tok}, {"a": codes_len})
    finally:
        R.MAX_PROMPT_TOKENS = old


def test_candidate_codes_equal_length_and_pools_disjoint():
    tok = WordTok()
    codes, codes_len = R.eligible_codes(tok)
    assert codes_len == 1 and len(codes) == 9000
    pools = R.partition_codes(codes)
    sets = {k: set(v) for k, v in pools.items()}
    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b]
    assert set(pools) == {"main_exemplar", "pilot_exemplar", "pilot_test", "smoke_exemplar", "smoke_test",
                          "pilot2_exemplar", "pilot2_test", "main_test"}
    assert len(pools["pilot2_exemplar"]) == 12 and len(pools["pilot2_test"]) == 60
    assert sum(len(v) for v in pools.values()) == 9000
    # the slices the committed receipts used are unchanged: pilot2 is carved from the former main test pool's head
    rng = np.random.default_rng(R.SEED_ITEMS)
    order = [codes[i] for i in rng.permutation(len(codes))]
    assert pools["main_exemplar"] == order[:12] and pools["pilot_exemplar"] == order[12:24] and pools["pilot_test"] == order[24:84]
    assert pools["smoke_exemplar"] == order[84:96] and pools["smoke_test"] == order[96:156]
    assert pools["pilot2_exemplar"] == order[156:168] and pools["pilot2_test"] == order[168:228] and pools["main_test"] == order[228:]
    with pytest.raises(R.Refusal):
        R.prepare_items(tok, 24, (16,), 1, "scratch", sentences=make_sentences(14000))


def test_exemplar_text_formats():
    exs = R.build_exemplars([str(1000 + 7 * i) for i in range(12)], np.random.default_rng(0))
    t = R.exemplar_text(exs[0], "compare")
    assert t.startswith("The sigma code is ") and " The theta code is " in t and " The omega code is " in t
    assert t.endswith(f"Question: Which name has the largest code: sigma, theta or omega? Answer: {R.EXEMPLAR_NAMES[exs[0]['max_slot']]}")
    t = R.exemplar_text(exs[2], "recall")
    assert t.endswith(f"Question: What is the omega code? Answer: {exs[2]['codes'][2]}")
    assert R.question_text("compare", 1) == "Question: Which name has the largest code: alpha, gamma or delta? Answer:"
    assert R.replant_sentence(1, "9412") == "The gamma code is now 9412."


def test_names_are_disjoint_and_single_tokens_in_the_word_tokenizer():
    tok = WordTok()
    assert not set(R.NAMES) & set(R.EXEMPLAR_NAMES)
    ids = R.name_token_ids(tok, R.NAMES)
    assert len(set(ids.values())) == 3
    assert R.assert_replant_lengths(tok, [str(c) for c in range(1000, 9000, 7)]) == 7   # one-token codes here; 8 in NeoX

    class TwoTok(WordTok):
        def encode(self, text):
            return super().encode(text.replace(" gamma", " gam ma"))
    with pytest.raises(R.Refusal, match="single token"):
        R.name_token_ids(TwoTok(), R.NAMES)


# -- the gates block -------------------------------------------------------------------------------------

def _gates_spec():
    from styxx.protocol import _select_gates_block
    return json.loads(_select_gates_block(PREREG.read_text(encoding="utf-8")))


def test_gates_block_parses_and_every_assignment_reaches_a_row():
    spec = _gates_spec()
    names = list(spec["gates"])
    assert names == ["G_P_plumbing", "G_A_screen", "G_A_guard", "G_C_abl", "G_S_between", "G_S_ci", "G_S_within", "G_N_null"]
    assert spec["smoke_verdict"] == "INVALID__smoke_plumbing_only"
    seen = set()
    for bits in itertools.product((False, True), repeat=len(names)):
        fired = dict(zip(names, bits))
        v = R.walk_outcomes(spec, fired)
        assert v is not None, fired
        seen.add(v)
    assert seen == {row["verdict"] for row in spec["outcomes"]}
    assert len(seen) == 7
    # G_S_between and G_N_null read one metric with incompatible bars, so assignments with both true are not
    # realizable; the table still resolves them by order: SUPPORT when the other two S gates hold, else NULL.
    all_true = dict(zip(names, [True] * len(names)))
    assert R.walk_outcomes(spec, all_true) == "SUPPORT__compare_specific_distance_deficit_in_deployed_decay_ssm"
    assert R.walk_outcomes(spec, dict(all_true, G_S_ci=False)) == "NULL__dissociation_does_not_transfer_to_deployed_models"
    # the realizable paths
    assert R.walk_outcomes(spec, dict(all_true, G_N_null=False)) == "SUPPORT__compare_specific_distance_deficit_in_deployed_decay_ssm"
    assert R.walk_outcomes(spec, dict(all_true, G_N_null=False, G_S_ci=False)) == "PARTIAL__reported_verbatim"
    assert R.walk_outcomes(spec, dict(all_true, G_N_null=False, G_S_within=False)) == "PARTIAL__reported_verbatim"
    assert R.walk_outcomes(spec, dict(all_true, G_S_between=False, G_S_ci=False, G_S_within=False)) == "NULL__dissociation_does_not_transfer_to_deployed_models"
    assert R.walk_outcomes(spec, dict(all_true, G_S_between=False, G_S_ci=False, G_S_within=False, G_N_null=False)) == "PARTIAL__reported_verbatim"
    assert R.walk_outcomes(spec, dict(all_true, G_C_abl=False)) == "INVALID__probe_reads_priors"
    assert R.walk_outcomes(spec, dict(all_true, G_A_guard=False)) == "ABSTAIN__degenerate_at_D0"
    assert R.walk_outcomes(spec, dict(all_true, G_A_screen=False, G_A_guard=False)) == "ABSTAIN__capability_absent_at_D0"
    assert R.walk_outcomes(spec, dict(all_true, G_P_plumbing=False)) == "INVALID__plumbing"
    for g in spec["gates"].values():
        assert isinstance(g["power_basis"], str) and g["power_basis"].strip()
        assert g["metric"].startswith("result.")


def test_gates_block_is_verbatim_from_the_plan_when_the_plan_is_present():
    # The judged plan is a working note outside the repository; name its path in BRIDGE_V3_PLAN_PATH to run this.
    where = os.environ.get("BRIDGE_V3_PLAN_PATH", "").strip()
    if not where:
        pytest.skip("set BRIDGE_V3_PLAN_PATH to the judged PLAN_next_experiment.md to compare its gates block")
    plan = Path(where)
    if not plan.is_file():
        pytest.skip("BRIDGE_V3_PLAN_PATH names no file on this machine")
    from styxx.protocol import _select_gates_block
    assert _select_gates_block(plan.read_text(encoding="utf-8")) == _select_gates_block(PREREG.read_text(encoding="utf-8"))


def _result(**kw):
    base = {"plumbing_ok": 1.0, "screen_min_norm_acc_D0": 0.55, "guard_ok_D0": 1.0, "ablated_max_norm_acc": 0.04,
            "between_D1": 0.20, "between_D1_ci_lo": 0.06, "within_M1": 0.18, "within_T": 0.01}
    base.update(kw)
    return {"result": base}


def test_experiment_scores_the_receipt_shape():
    from styxx.protocol import Experiment, PrologueError
    try:
        exp = Experiment(PREREG, repo_root=ROOT)
    except PrologueError:
        pytest.skip("PREREG not yet committed in this checkout")
    assert exp.vacuous_gates == [] and exp.undeclared_power_gates == []
    assert exp.score(_result()).verdict == "SUPPORT__compare_specific_distance_deficit_in_deployed_decay_ssm"
    assert exp.score(_result(between_D1=0.0, between_D1_ci_lo=-0.1, within_M1=0.0)).verdict == "NULL__dissociation_does_not_transfer_to_deployed_models"
    assert exp.score(_result(between_D1=-0.2, between_D1_ci_lo=-0.3, within_M1=0.0)).verdict == "NULL__dissociation_does_not_transfer_to_deployed_models"
    assert exp.score(_result(between_D1=0.08, between_D1_ci_lo=-0.02)).verdict == "PARTIAL__reported_verbatim"
    assert exp.score(_result(between_D1=0.20, between_D1_ci_lo=-0.01)).verdict == "PARTIAL__reported_verbatim"
    assert exp.score(_result(screen_min_norm_acc_D0=0.39)).verdict == "ABSTAIN__capability_absent_at_D0"
    assert exp.score(_result(guard_ok_D0=0.0)).verdict == "ABSTAIN__degenerate_at_D0"
    assert exp.score(_result(ablated_max_norm_acc=0.17)).verdict == "INVALID__probe_reads_priors"
    assert exp.score(_result(plumbing_ok=0.0)).verdict == "INVALID__plumbing"
    assert exp.score(_result(between_D1=0.9), smoke=True).verdict == "INVALID__smoke_plumbing_only"
    assert all(v["usable"] for v in exp.check_metrics(_result()).values())


# -- metrics ---------------------------------------------------------------------------------------------

def test_constant_responder_maps_to_zero_under_exact_balance(small_cells):
    tok, cells, exemplars, _, _ = small_cells
    items = cells[16]
    targets = [it["max_slot"] for it in items]
    for slot in range(3):
        scores = np.full((len(items), 3), -5.0)
        scores[:, slot] = -1.0                                        # always the same name
        s = R.argmax_cell_summary(scores, targets)
        assert abs(s["acc_raw"] - 1 / 3) < 1e-12 and abs(s["acc_norm"]) < 1e-12
        assert s["guard_ok"] is False and s["max_slot_fraction"] == 1.0 and s["ovr_auroc"] == 0.5
        assert len(s["_per_item"]["correct"]) == len(items) and sum(s["_per_item"]["correct"]) == len(items) // 3
    # a slot-copying responder (the last exemplar's compare slot) also scores exactly chance
    last = [exemplars[it["perm"][-1]]["max_slot"] for it in items]
    scores = np.full((len(items), 3), -5.0)
    scores[np.arange(len(items)), last] = -1.0
    s = R.argmax_cell_summary(scores, targets)
    assert abs(s["acc_norm"]) < 1e-12 and s["guard_ok"] is True
    # exact ties resolve to the lowest slot and are counted
    s = R.argmax_cell_summary(np.zeros((len(items), 3)), targets)
    assert s["near_ties"] == len(items) and s["slot_fractions"] == [1.0, 0.0, 0.0] and abs(s["acc_norm"]) < 1e-12
    # perfect
    scores = np.full((len(items), 3), -5.0)
    scores[np.arange(len(items)), targets] = -0.5
    s = R.argmax_cell_summary(scores, targets, ow_slots=[it["ow_slot"] for it in items])
    assert s["acc_norm"] == 1.0 and s["guard_ok"] and s["ovr_auroc"] == 1.0 and s["margin_mean"] == pytest.approx(4.5)
    assert s["overwrite_follow_rate"] == 0.0 and sum(s["_per_item"]["pred_is_overwrite"]) == 0
    # a responder that follows the re-plant is wrong on every item and the follow rate says so
    scores = np.full((len(items), 3), -5.0)
    scores[np.arange(len(items)), [it["ow_slot"] for it in items]] = -0.5
    s = R.argmax_cell_summary(scores, targets, ow_slots=[it["ow_slot"] for it in items])
    assert s["acc_raw"] == 0.0 and s["acc_norm"] == pytest.approx(-0.5) and s["overwrite_follow_rate"] == 1.0
    with pytest.raises(R.Refusal):
        R.argmax_cell_summary(np.zeros((len(items), 2)), targets)


def test_norm_acc_endpoints_both_families_chance_one_third():
    assert R.CHANCE == {"compare": pytest.approx(1 / 3), "recall": pytest.approx(1 / 3)}
    assert abs(R.norm_acc(1 / 3, 1 / 3)) < 1e-12 and R.norm_acc(1.0, 1 / 3) == 1.0
    assert R.norm_acc(0.60, 1 / 3) == pytest.approx(0.40)              # the 0.40 bar is raw 0.60 in both families now
    assert R.norm_acc(1 / 3 + 0.16 * 2 / 3, 1 / 3) == pytest.approx(0.16)


def test_auroc_ties_and_separation():
    assert R.auroc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]) == 1.0
    assert R.auroc([0.4, 0.3, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert R.auroc([1.0, 1.0, 1.0, 1.0], [False, True, False, True]) == 0.5
    assert R.auroc([1.0, 1.0], [True, True]) is None
    assert R.ovr_auroc(np.eye(3), [0, 1, 2]) == 1.0
    assert R.ovr_auroc(np.ones((6, 3)), [0, 1, 2, 0, 1, 2]) == 0.5
    assert R.ovr_auroc(np.ones((2, 3)), [0, 0]) is None


def test_bootstrap_is_deterministic_and_matches_the_point_estimate():
    rng = np.random.default_rng(7)
    n = 600
    cT, cM, rT, rM = (rng.random(n) < p for p in (0.85, 0.70, 0.80, 0.78))
    b1 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT)
    b2 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT)
    assert b1 == b2
    b3 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT + 1)
    assert b3["ci_lo"] != b1["ci_lo"]
    C_T, C_M = R.norm_acc(cT.mean(), 1 / 3), R.norm_acc(cM.mean(), 1 / 3)
    R_T, R_M = R.norm_acc(rT.mean(), 1 / 3), R.norm_acc(rM.mean(), 1 / 3)
    assert b1["point"] == pytest.approx((C_T - C_M) - (R_T - R_M), abs=1e-12)
    assert b1["ci_lo"] <= b1["point"] <= b1["ci_hi"]
    assert b1["resamples"] == 500 and b1["seed"] == R.SEED_BOOT and b1["n_items"] == n
    assert b1["weights"] == {"compare": pytest.approx(1.5), "recall": pytest.approx(1.5)}


# -- budget, format line, receipt hygiene ----------------------------------------------------------------

def test_budget_extrapolation_counts_every_pass():
    t = {"T": {16: {"L": 230, "s_per_batch": {"compare": 0.2, "recall": 0.2}},
               1024: {"L": 1250, "s_per_batch": {"compare": 0.8, "recall": 0.8}}}}
    b = R.extrapolate_budget(t)
    assert b["arms_untimed"] == ["M1"] and b["verdict"] == "OK_PARTIAL" and not b["exceeds_budget"]
    d16 = b["per_arm"]["T"]["by_distance"]["16"]
    assert d16["batch_size"] == 16 and d16["batches_per_pass"] == 38
    assert d16["seconds"] == pytest.approx(0.2 * 4 * 38 * 3, abs=0.2)          # plain + ablated + windowed at D0
    d1024 = b["per_arm"]["T"]["by_distance"]["1024"]
    assert d1024["seconds"] == pytest.approx(0.8 * 4 * 38 * 3 + 0.8 * 38 * 2, abs=0.2)   # + C_near + C_overwrite, compare only
    d64 = b["per_arm"]["T"]["by_distance"]["64"]
    s64 = 0.2 + (64 - 16) / (1024 - 16) * 0.6
    assert d64["seconds"] == pytest.approx(s64 * 4 * 38 * 2, abs=0.2)           # plain + ablated (C_abl at every distance)
    t["M1"] = {16: {"L": 230, "s_per_batch": {"compare": 2.0, "recall": 2.0}},
               1024: {"L": 1250, "s_per_batch": {"compare": 8.0, "recall": 8.0}}}
    b = R.extrapolate_budget(t)
    assert b["arms_untimed"] == [] and not b["exceeds_budget"] and b["verdict"] == "OK"
    m1024 = b["per_arm"]["M1"]["by_distance"]["1024"]
    assert m1024["batch_size"] == R.batch_size_for("M1", 1250) == 16 and m1024["batches_per_pass"] == 38
    assert m1024["seconds"] == pytest.approx(8.0 * 4 * 38 * 2 + 8.0 * 38 * 2, abs=0.2)   # no windowed for M1
    t["M1"][1024]["s_per_batch"] = {"compare": 60.0, "recall": 60.0}
    b = R.extrapolate_budget(t)
    assert b["exceeds_budget"] and b["verdict"] == "INVALID__budget" and b["total_hours_timed_arms"] > 4.0
    with pytest.raises(R.Refusal):
        R.extrapolate_budget({"T": {16: t["T"][16]}})


def test_budget_from_the_committed_smoke_timings_stays_inside_four_hours():
    """The committed smoke's per-batch seconds (11.7 / 2.3 s for M1, 1.2 / 0.2 s for T) under the amended pass structure."""
    t = {"M1": {16: {"L": 230, "s_per_batch": {"compare": 2.3, "recall": 2.3}}, 1024: {"L": 1210, "s_per_batch": {"compare": 11.7, "recall": 11.7}}},
         "T": {16: {"L": 230, "s_per_batch": {"compare": 0.2, "recall": 0.2}}, 1024: {"L": 1210, "s_per_batch": {"compare": 1.2, "recall": 1.2}}}}
    b = R.extrapolate_budget(t)
    assert b["verdict"] == "OK" and 2.0 < b["total_hours_timed_arms"] < 3.0


def test_batch_size_bounds():
    assert R.batch_size_for("T", 1250) == 16
    assert R.batch_size_for("M1", 1250) == 16 == R.BATCH_MAX           # PREREG section 4: batches of 16 for both arms
    assert R.SCAN_CHUNK_TOKENS // R.BATCH_MAX >= 1                      # the chunk of L the scan materialises at B = 16
    assert R.batch_size_for("M1", 10) == 16


def test_answer_format_line_last_one_wins_and_the_main_run_refuses_everything_but_the_readout():
    text = PREREG.read_text(encoding="utf-8")
    frozen = text.split("## 17. Amendments")[0]                 # section 9's own line, before any amendment
    assert R.answer_format_from_prereg(frozen) == "PENDING_PILOT"
    line = "\nAnswer format chosen (written before sealing): "
    assert R.answer_format_from_prereg(frozen + "\n## 17.1\n" + line + "NOT_STARTED\n") == "NOT_STARTED"
    assert R.answer_format_from_prereg(frozen + line + "NOT_STARTED\n" + line + R.READOUT + "\n") == R.READOUT
    assert R.answer_format_from_prereg(frozen + line + "true_false\n") == "true_false"
    with pytest.raises(R.Refusal):
        R.answer_format_from_prereg(text + line + "maybe\n")
    with pytest.raises(R.Refusal):
        R.answer_format_from_prereg("no such line")
    assert R.answer_format_from_prereg(text) in R.FORMAT_LINE_VALUES
    for fmt, why in (("PENDING_PILOT", "PENDING_PILOT"), ("NOT_STARTED", "NOT_STARTED"), ("yes_no", "retired"), ("true_false", "retired")):
        with pytest.raises(R.Refusal, match=why):
            R.check_format_for_main(fmt)
    R.check_format_for_main(R.READOUT)
    # the document as it stands never lets the main run start unless its last line names the readout
    fmt_now = R.answer_format_from_prereg(text)
    if fmt_now != R.READOUT:
        with pytest.raises(R.Refusal):
            R.check_format_for_main(fmt_now)


def test_main_run_needs_a_smoke_by_this_readout_with_both_arms_timed():
    with pytest.raises(R.Refusal, match="no smoke receipt"):
        R.check_smoke_for_main(None)
    with pytest.raises(R.Refusal, match="not written by"):
        R.check_smoke_for_main({"config": {"answer_format": "yes_no"}, "budget": {"arms_timed": ["M1", "T"]}})
    good = {"config": {"readout": R.READOUT}, "budget": {"exceeds_budget": False, "arms_timed": ["M1", "T"], "arms_untimed": []}}
    assert R.check_smoke_for_main(good) == good["budget"]
    with pytest.raises(R.Refusal, match="INVALID__budget"):
        R.check_smoke_for_main({"config": {"readout": R.READOUT}, "budget": {"exceeds_budget": True, "total_hours_timed_arms": 5.1}})
    with pytest.raises(R.Refusal, match="M1,T"):
        R.check_smoke_for_main({"config": {"readout": R.READOUT}, "budget": {"exceeds_budget": False, "arms_timed": ["T"], "arms_untimed": ["M1"]}})


def test_pilot_outcome_rule():
    assert R.pilot_outcome({"M1": True, "T": True}) == "CLEARED_BOTH"
    assert R.pilot_outcome({"M1": False, "T": True}) == "CLEARED_T_ONLY"
    assert R.pilot_outcome({"M1": True, "T": False}) == "CLEARED_M1_ONLY"
    assert R.pilot_outcome({"M1": False, "T": False}) == "NEITHER"
    res = {"acc_norm": {"M1": {"compare": {"16": 0.5}, "recall": {"16": 0.7}}}, "guard": {"M1": {"compare": {"16": {"guard_ok": True}}, "recall": {"16": {"guard_ok": True}}}}}
    assert R.arm_clears(res, "M1")["clears"]
    res["acc_norm"]["M1"]["compare"]["16"] = 0.39
    assert not R.arm_clears(res, "M1")["clears"]
    res["acc_norm"]["M1"]["compare"]["16"] = 0.5
    res["guard"]["M1"]["compare"]["16"]["guard_ok"] = False
    assert not R.arm_clears(res, "M1")["clears"]


def test_write_json_is_utf8_lf_without_bom_and_strips_private(tmp_path):
    obj = {"a": np.float64(1.5), "b": np.bool_(True), "c": np.int64(3), "_hidden": [1], "d": [np.float32(2.0)], "e": np.arange(2),
           "f": {"_p": 1, "g": "é — dash"}}
    out = R.strip_private(obj)
    assert out == {"a": 1.5, "b": True, "c": 3, "d": [2.0], "e": [0, 1], "f": {"g": "é — dash"}}
    p = tmp_path / "r.json"
    R.write_json(p, obj)
    raw = p.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"\n") and b"\n\n" not in raw[-2:]
    assert json.loads(raw.decode("utf-8")) == out
    assert raw.count(b"\n") == len(json.dumps(out, indent=2, ensure_ascii=False).splitlines())


def test_base_receipt_has_the_prereg_layout():
    res = R.base_receipt("main", 600, R.DISTANCES, ["M1", "T"])
    for key in ("config", "pilot", "acc_raw", "acc_norm", "guard", "controls", "result", "bootstrap", "mechanism", "per_item", "items",
                "exploratory", "redteam", "runtime"):
        assert key in res
    assert res["exploratory"]["status"] == "M3_NOT_RUN"
    assert res["config"]["screen_bar"] == 0.40 and res["config"]["n_per_cell"] == 600
    assert res["config"]["seeds"] == {"items": 20261011, "pilot": 1011, "pilot2": 1012, "smoke": 1016, "bootstrap": 20261011}
    assert len({R.SEED_ITEMS, R.SEED_PILOT, R.SEED_PILOT2, R.SEED_SMOKE}) == 4
    assert set(res["controls"]) == {"ablated", "windowed", "near", "overwrite"}
    assert res["config"]["readout"] == R.READOUT == "argmax_largest" and res["config"]["retired_formats"] == ["yes_no", "true_false"]
    assert res["config"]["names"] == {"test": ["alpha", "gamma", "delta"], "exemplar": ["sigma", "theta", "omega"]}
    assert res["config"]["guard"] == {"max_slot_fraction": 0.80} and res["config"]["chance"]["compare"] == pytest.approx(1 / 3)
    assert res["config"]["code_max_planted"] == 8999
    assert res["config"]["scan_chunk_tokens_M1"] == R.SCAN_CHUNK_TOKENS and res["config"]["batch_max"] == 16
    assert res["config"]["runner"] == RUNNER.name and res["config"]["runner_sha256"] == R.sha256_file(RUNNER)
    assert re.fullmatch(r"[0-9a-f]{64}", res["config"]["runner_sha256"])
    assert "prereg_blob_sha256" in res["config"]["git"]


def test_hash_lists_and_the_disjointness_assert():
    tok = WordTok()
    sents = make_sentences(14000, seed=11)
    a = R.prepare_items(tok, 24, (16, 1024), R.SEED_PILOT2, "pilot2", sentences=sents)
    b = R.prepare_items(tok, 24, (16,), R.SEED_SMOKE, "smoke", sentences=sents)
    la, lb = R.hash_lists(a), R.hash_lists(b)
    assert set(la["prompts"]) == {"compare", "recall"} and set(la["prompts"]["compare"]["1024"]) == {"plain", "ablated", "near", "overwrite"}
    assert set(la["prompts"]["compare"]["16"]) == {"plain", "ablated"} and len(la["prompts"]["recall"]["16"]["plain"]) == 24
    assert la["n_prompts"] == 2 * (24 * 2 + 24 * 4) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in la["sentences"][:5])
    assert la["n_sentences"] == len(set(a["pool"].used)) and la["codes_test"] == sorted(la["codes_test"])
    assert la["n_codes_exemplar"] == 12 and la["n_codes_test"] >= 3
    # a receipt written by this readout: lists are read straight from items.sha256
    rec_b = {"config": {"filler_pool": b["sentence_provenance"], "sentence_region": b["sentence_region"]},
             "sentence_indices_used": sorted(b["pool"].used), "items": {"sha256": lb}}
    prior = R.prior_sentences_and_codes(a, {"smoke": rec_b, "absent": None})
    assert prior["receipts"]["absent"] == "absent" and prior["receipts"]["smoke"]["sentence_indices"] == len(set(b["pool"].used))
    assert set(lb["sentences"]) <= prior["sentence_sha256"] and set(lb["codes_test"]) <= prior["code_sha256"]
    # the smoke region's whole code pools count as prior, so the smoke's codes are excluded even by an index-only receipt
    rec_old = {"config": {"filler_pool": b["sentence_provenance"], "sentence_region": b["sentence_region"]},
               "sentence_indices_used": sorted(b["pool"].used), "mode": "smoke"}
    prior_old = R.prior_sentences_and_codes(a, {"smoke": rec_old})
    assert {R.sha256_text(c) for c in b["code_pools"]["smoke_test"]} <= prior_old["code_sha256"]
    ok = R.assert_disjoint_from_prior(a, la, prior)
    assert ok["ok"] and ok["this_run_sentences"] == la["n_sentences"]
    # overlap in a sentence hash, a sentence index or a code hash is refused
    for key, val in (("sentence_sha256", {la["sentences"][0]}), ("sentence_indices", {a["pool"].used[0]}), ("code_sha256", {la["codes_test"][0]})):
        bad = {"sentence_indices": set(), "sentence_sha256": set(), "code_sha256": set(), "receipts": {}}
        bad[key] = val
        with pytest.raises(R.Refusal, match="shares"):
            R.assert_disjoint_from_prior(a, la, bad)
    # a receipt from another pool cannot be mapped
    with pytest.raises(R.Refusal, match="pool differs"):
        R.prior_sentences_and_codes(a, {"x": {"config": {"filler_pool": {"sha256": ["0"], "eligible_sentences": 1, "pool_cap": 1}}}})


def test_tokenizer_level_asserts_on_a_synthetic_pilot2(tmp_path):
    tok = WordTok()
    sents = make_sentences(14000, seed=12)
    prep = R.prepare_items(tok, 48, (16,), R.SEED_PILOT2, "pilot2", sentences=sents)
    red = R.tokenizer_level_asserts(prep, {"M1": tok, "T": tok})
    assert red["balance_by_distance"]["16"]["max_slot"] == [16, 16, 16] and red["names_disjoint"] and red["sentences_inside_region"]
    assert red["name_token_ids"]["T"]["test"].keys() == set(R.NAMES) and red["replant_sentence_tokens"] == {"M1": 7, "T": 7}
    assert red["fillers"]["sentences_skipped_for_a_name"] == 0
    res = R.base_receipt("pilot2", 48, (16,), ["M1", "T"])
    R.record_items(res, prep)
    assert res["items"]["prefix_nested_across_distances"] is False and res["items"]["names"]["disjoint"] is True
    assert res["items"]["sha256"]["n_prompts"] == 2 * 48 * 2
    with pytest.raises(R.Refusal, match="outside"):
        forged = dict(prep, region_indices=set())
        R.tokenizer_level_asserts(forged, {"T": tok})


def test_the_lab_never_prints_these_words():
    """The banned list is the lab's own BANNED tuple in scripts/zenodo_publish_v7_48_0.py, read from its text so
    that this file does not print the words either. Also: no BOM (a CRLF working copy is the checkout's business)."""
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    for p in (PREREG, RUNNER, TEST_FILE):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        text = raw.decode("utf-8")
        hits = [m.group(0) for m in banned.finditer(text)]
        assert not hits, f"{p.name}: {hits}"


# -- tokenizer-only (cached GPT-NeoX files of both arms) -------------------------------------------------

def _neox_tokenizers():
    pytest.importorskip("transformers")
    paths = {arm: R.snapshot_dir(repo) for arm, repo in R.ARMS.items()}
    if any(p is None or not (p / "tokenizer.json").exists() for p in paths.values()):
        pytest.skip("the GPT-NeoX tokenizer files of mamba-1.4b-hf / pythia-1.4b are not cached here")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {arm: R.load_tokenizer(repo) for arm, repo in R.ARMS.items()}


def test_neox_tokenizer_names_codes_replant_and_identical_ids():
    toks = _neox_tokenizers()
    for arm, t in toks.items():
        assert R.name_token_ids(t, R.NAMES) == {"alpha": 9765, "gamma": 17356, "delta": 18687}
        assert R.name_token_ids(t, R.EXEMPLAR_NAMES) == {"sigma": 40009, "theta": 39116, "omega": 40639}
        assert len(t.encode(" bravo")) == 2                             # why the section-4 name list could not serve a three-way answer set
        assert t.encode(" 4271") == [577, 28209]
        codes, L = R.eligible_codes(t)
        assert L == 2 and len(codes) == 8828
        assert R.assert_replant_lengths(t, codes) == 8
        one_token = [c for c in map(str, range(1000, 10000)) if c not in set(codes)]
        assert len(one_token) == 172 and len(t.encode(" " + one_token[0])) == 1
        with pytest.raises(R.Refusal, match="several token lengths"):
            R.assert_replant_lengths(t, one_token[:10] + codes[:10])          # 20 codes: every one is sampled
        assert t.encode(R.question_text("compare", 0))[-1] == 27        # ':' of 'Answer:' — the position whose next token is scored
    tok = toks["T"]
    cells, exemplars, pool, codes_len = make_cells(tok, 24, (16, 1024), n_sent=4000)
    for D, items in cells.items():
        R.assert_balance(items, exemplars)
        for it in items:
            assert len(tok.encode(" " + it["filler"])) == D
    R.assert_fillers(cells, exemplars)
    info = R.assert_prompts(cells, exemplars, toks, {arm: codes_len for arm in toks})
    assert info["prompts_checked"] == 2 * (24 * 2 + 24 * 4) and info["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert info["max_prompt_tokens"] < 1400                      # the PREREG's estimate: about 1,250 at D = 1024
    it = cells[1024][0]
    for variant in R.item_variants(it):
        parts = R.prompt_parts(it, exemplars, "compare", variant)
        assert R.compositional_ids(tok, parts) is not None
        assert tok.encode(parts[2])[-1] == 27
    assert len(tok.encode(R.prompt_parts(it, exemplars, "compare", "near")[1])) == 16


def test_neox_tokenizer_on_the_real_wikitext_pool_when_cached():
    toks = _neox_tokenizers()
    if R.snapshot_dir("wikitext", kind="datasets") is None:
        pytest.skip("wikitext-103-raw-v1 is not cached here")
    pytest.importorskip("pyarrow")
    sents, prov = R.read_wikitext_sentences(cap=20000)
    assert prov["eligible_sentences"] == 20000 and len(prov["sha256"]) == 2
    assert all(R.eligible_sentence(s) for s in sents)
    preps = {"main": R.prepare_items(toks["T"], 24, (16, 1024), R.SEED_ITEMS, "main", sentences=sents),
             "smoke": R.prepare_items(toks["T"], 24, (16, 1024), R.SEED_SMOKE, "smoke", sentences=sents),
             "pilot": R.prepare_items(toks["T"], 24, (16,), R.SEED_PILOT, "pilot", sentences=sents),
             "pilot2": R.prepare_items(toks["T"], 48, (16,), R.SEED_PILOT2, "pilot2", sentences=sents)}
    red = R.tokenizer_level_asserts(preps["main"], toks)
    assert red["code_token_length"] == 2 and red["prompts"]["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert red["balance_by_distance"]["16"]["max_slot"] == [8, 8, 8]
    used = {k: set(p["pool"].used) for k, p in preps.items()}
    for a, b in itertools.combinations(used, 2):
        assert not used[a] & used[b], (a, b)
    lists = {k: R.hash_lists(p) for k, p in preps.items()}
    for a, b in itertools.combinations(lists, 2):
        assert not set(lists[a]["sentences"]) & set(lists[b]["sentences"]), (a, b)
        assert not (set(lists[a]["codes_test"]) | set(lists[a]["codes_exemplar"])) & (set(lists[b]["codes_test"]) | set(lists[b]["codes_exemplar"])), (a, b)
    # the main run's reuse assert, fed the other regions' sentences, passes on disjoint pools and refuses on overlap
    R.assert_fillers(preps["main"]["cells"], preps["main"]["exemplars"], used["smoke"] | used["pilot"] | used["pilot2"])
    with pytest.raises(R.Refusal, match="smoke/pilot"):
        R.assert_fillers(preps["main"]["cells"], preps["main"]["exemplars"], {next(iter(used["main"]))})
    assert preps["pilot2"]["sentence_region"]["which"] == "pilot2" and preps["pilot2"]["sentence_region"]["seed"] == R.SEED_PILOT2


def test_pilot2_is_disjoint_from_the_committed_smoke_and_pilot_receipts_when_the_pool_is_cached():
    """The exact assert the --pilot run makes, on CPU, against the committed receipts' sentence indices and code slices."""
    toks = _neox_tokenizers()
    if R.snapshot_dir("wikitext", kind="datasets") is None:
        pytest.skip("wikitext-103-raw-v1 is not cached here")
    pytest.importorskip("pyarrow")
    smoke, pilot = R.read_receipt(R.SMOKE_RECEIPT), R.read_receipt(R.PILOT1_RECEIPT)
    if smoke is None or pilot is None:
        pytest.skip("the committed smoke and 1011-pilot receipts are not in this checkout")
    prep = R.prepare_items(toks["T"], R.N_PILOT, (R.D0,), R.SEED_PILOT2, "pilot2")
    assert prep["sentence_provenance"]["eligible_sentences"] == R.POOL_CAP == 150000
    prior = R.prior_sentences_and_codes(prep, {"smoke": smoke, "pilot": pilot})
    assert prior["receipts"]["smoke"]["sentence_indices"] == len(smoke["sentence_indices_used"]) == 1113
    assert prior["receipts"]["pilot"]["sentence_indices"] == len(pilot["sentence_indices_used"]) == 171
    red = R.tokenizer_level_asserts(prep, toks, prior["sentence_indices"])
    lists = R.hash_lists(prep)
    ok = R.assert_disjoint_from_prior(prep, lists, prior)
    assert ok["ok"] and ok["prior_sentence_indices"] == 1113 + 171 and ok["prior_code_sha256"] == 2 * 72
    assert red["prompts"]["prompts_checked"] == 2 * 48 * 2 and red["prompts"]["max_prompt_tokens"] < 300
    # every pilot2 sentence lies in its own slice and none in the smoke's or the 1011 pilot's
    orders = R.split_orders(len(prep["sentences"]))
    used = set(prep["pool"].used)
    assert used <= set(int(i) for i in orders["pilot2"])
    assert not used & set(int(i) for i in orders["smoke"]) and not used & set(int(i) for i in orders["pilot"])
    assert not used & set(int(i) for i in orders["main"])


# -- the chunked scan against transformers' own slow_forward, on a tiny random mixer (CPU, no checkpoint) ---

def test_chunked_scan_equals_reference_slow_forward_on_a_random_mixer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.mamba import modeling_mamba as mm
    from transformers import MambaConfig
    cfg = MambaConfig(hidden_size=24, intermediate_size=48, state_size=16, conv_kernel=4, time_step_rank=4, num_hidden_layers=1, vocab_size=64)
    torch.manual_seed(0)
    mixer = mm.MambaMixer(cfg, layer_idx=0).eval()
    ref = mm.MambaMixer.slow_forward                       # the reference, before any install
    old_chunk = R.SCAN_CHUNK_TOKENS
    try:
        info = R.install_chunked_scan()
        assert info["installed"] and mm.MambaMixer.slow_forward is R.chunked_slow_forward
        for B, L, chunk_tokens in ((1, 37, 8), (3, 37, 8), (2, 5, 1024), (4, 64, 4)):
            R.SCAN_CHUNK_TOKENS = chunk_tokens             # several chunks per sequence, a single chunk, one token per chunk
            x = torch.randn(B, L, cfg.hidden_size)
            with torch.no_grad():
                y_ref = ref(mixer, x)
                y_new = R.chunked_slow_forward(mixer, x)
                y_mod = mixer(x)                           # the module's forward now routes through the installed copy
            assert y_ref.shape == y_new.shape == (B, L, cfg.hidden_size)
            assert float((y_ref - y_new).abs().max()) <= 1e-6, (B, L, chunk_tokens)
            assert float((y_ref - y_mod).abs().max()) <= 1e-6
        # the original is kept and anything with a cache goes to it
        assert R._ORIG_SLOW_FORWARD is ref
    finally:
        R.SCAN_CHUNK_TOKENS = old_chunk
        mm.MambaMixer.slow_forward = ref
        R._ORIG_SLOW_FORWARD = None


# -- optional: the Mamba code path on CPU with the cached 130m checkpoint (opt in; never on cuda) ----------

@pytest.mark.skipif(os.environ.get("BRIDGE_V3_CPU_MODEL_TEST") != "1", reason="set BRIDGE_V3_CPU_MODEL_TEST=1 to load mamba-130m-hf on CPU")
def test_mamba_code_path_on_cpu_with_130m():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = R.snapshot_dir("state-spaces/mamba-130m-hf")
    if path is None:
        pytest.skip("mamba-130m-hf is not cached")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tok = R.load_tokenizer("state-spaces/mamba-130m-hf")
    cs = R.install_chunked_scan()
    assert cs["installed"] and not cs["transformers_fast_path_available"]
    model, info = R.load_arm_cpu_fp32("state-spaces/mamba-130m-hf")
    assert next(model.parameters()).device.type == "cpu" and info["from_local_snapshot"]
    a = R.assert_mamba_A(model)
    assert a["ok"] and a["layers"] == 24 and a["max_A"] < 0
    cells, exemplars, _, codes_len = make_cells(tok, 24, (16, 1024), n_sent=4000)
    items = cells[16]
    ce = R.check_chunked_scan_equivalence(model, tok, items[0], exemplars)
    assert ce["max_abs_logit_diff"] <= 1e-5 and ce["chunks"] >= 1, ce
    rp = R.check_right_padding("M1", model, tok, items, exemplars)
    assert rp["max_abs_logit_diff"] <= R.FP32_TOL, rp
    name_ids = [R.name_token_ids(tok, R.NAMES)[nm] for nm in R.NAMES]
    mech = R.MechanismHook(model)
    try:
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "compare", 16, name_ids, codes_len, mech=mech, batch_size=4)
        assert s["n"] == 8 and len(s["_correct"]) == 8 and s["batches"] == 2 and len(s["_per_item"]["scores"][0]) == 3
        assert len(s["_mechanism"]["by_layer_mean_log_surviving"]) == 24
        assert s["_mechanism"]["slowest_layer_log_surviving"] <= 0.0
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "recall", 16, name_ids, codes_len, batch_size=4)
        assert s["n"] == 8 and sum(s["slot_fractions"]) == pytest.approx(1.0) and s["batches"] == 6
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "compare", 16, name_ids, codes_len, variant="ablated", batch_size=4)
        assert s["n"] == 8
        far = cells[1024][:4]
        s = R.run_cell("M1", model, tok, far, exemplars, "compare", 1024, name_ids, codes_len, variant="near", batch_size=4)
        assert s["n"] == 4
        s = R.run_cell("M1", model, tok, far, exemplars, "compare", 1024, name_ids, codes_len, variant="overwrite", batch_size=4)
        assert s["n"] == 4 and 0.0 <= s["overwrite_follow_rate"] <= 1.0
    finally:
        mech.remove()
        del model

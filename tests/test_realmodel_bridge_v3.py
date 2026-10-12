# -*- coding: utf-8 -*-
"""CPU-only tests for papers/frequency-resonance/run_realmodel_bridge_v3.py — no model is loaded.

What is checked, against PREREG_realmodel_bridge_v3_2026_10_11.md:
  * exact per-cell balance computed from the generated items (section 4, assert 3);
  * no filler sentence reused across items or cells, no digit / name / 'code' in any filler (assert 4);
  * the prompt length bound (assert 7) and equal-length candidate codes (assert 5);
  * the gates block parses through the lab's own scanner and all 256 boolean assignments reach a row;
  * chance normalization maps a constant responder to exactly 0 under exact balance;
  * the paired item bootstrap is deterministic under its seed;
  * the budget extrapolation and the answer-format line the runner reads;
  * tokenizer-only: the cached GPT-NeoX tokenizer of both arms (mamba-1.4b-hf ships pythia's) has single-token
    Yes/No and True/False, two-token codes, and identical ids on every prompt (skipped when not cached).

The item generator is exercised with a GPT-2-style word tokenizer and a synthetic sentence pool, so the
tests need neither torch nor the wikitext cache. A CPU-only exercise of the Mamba code path on the cached
state-spaces/mamba-130m-hf runs only when BRIDGE_V3_CPU_MODEL_TEST=1 is set.
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


def make_cells(tok, n, distances, seed=R.SEED_ITEMS, n_sent=12000, fmts=("yes_no",)):
    sents = make_sentences(n_sent)
    codes, codes_len = R.eligible_codes(tok)
    pools = R.partition_codes(codes)
    pool = R.SentencePool(sents, np.random.default_rng(1).permutation(len(sents)))
    exemplars = R.build_exemplars(pools["main_exemplar"], np.random.default_rng(seed))
    cells = {D: R.build_items(n, D, seed, pool, pools["main_test"], exemplars, tok, fmts=fmts) for D in distances}
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
        assert got["label_true"] == 24
        assert got["claimed_slot"] == [16, 16, 16]
        assert got["q_slot"] == [16, 16, 16]
        assert got["last_exemplar_yes"] == 24
        assert got["true_by_q_slot"] == [8, 8, 8]
        assert got["claim_consistent"] and got["codes_distinct"]
        # the recency flag is read off the permuted exemplar, not the generator's intent
        for it in items:
            assert exemplars[it["perm"][-1]]["label"] == it["last_yes"]
            assert sorted(it["perm"]) == [0, 1, 2, 3]


def test_exact_balance_at_600_matches_the_prereg_counts():
    tok = WordTok()
    cells, exemplars, _pool, _ = make_cells(tok, 600, (16,), n_sent=6000)
    got = R.assert_balance(cells[16], exemplars)
    assert got["label_true"] == 300
    assert got["claimed_slot"] == [200, 200, 200]
    assert got["q_slot"] == [200, 200, 200]
    assert got["last_exemplar_yes"] == 300
    assert got["true_by_q_slot"] == [100, 100, 100]


def test_balance_refuses_an_unbalanced_cell(small_cells):
    tok, cells, exemplars, _, _ = small_cells
    items = [dict(it) for it in cells[16]]
    items[0]["label"] = not items[0]["label"]
    items[0]["claimed_slot"] = items[0]["q_slot"] if items[0]["label"] else (items[0]["q_slot"] + 1) % 3
    with pytest.raises(R.Refusal):
        R.assert_balance(items, exemplars)


def test_design_rows_need_a_multiple_of_24():
    with pytest.raises(R.Refusal):
        R.design_rows(25, np.random.default_rng(0))


def test_exemplars_two_yes_two_no_slots_1_2_3_1():
    exs = R.build_exemplars([str(1000 + i) for i in range(12)], np.random.default_rng(0))
    assert [e["q_slot"] for e in exs] == [0, 1, 2, 0]
    assert sum(e["label"] for e in exs) == 2
    for e in exs:
        assert (e["claimed_slot"] == e["q_slot"]) == e["label"]


# -- fillers ---------------------------------------------------------------------------------------------

def test_fillers_exact_length_distinct_and_clean(small_cells):
    tok, cells, exemplars, pool, _ = small_cells
    info = R.assert_fillers(cells, exemplars)
    assert info["sentences_used"] == len(set(pool.used)) == len(pool.used)
    for D, items in cells.items():
        for it in items:
            assert len(tok.encode(" " + it["filler"])) == D
            assert R.filler_violations(it["filler"]) == []
            assert R.filler_violations(it["neutral"]) == []
            assert len(it["filler_sentences"]) >= 1 and len(it["neutral_sentences"]) >= 1


def test_filler_reuse_is_refused(small_cells):
    tok, cells, exemplars, _, _ = small_cells
    forged = {D: [dict(it) for it in items] for D, items in cells.items()}
    forged[1024][0]["filler_sentences"] = list(forged[16][0]["filler_sentences"])
    with pytest.raises(R.Refusal, match="reused"):
        R.assert_fillers(forged, exemplars)
    with pytest.raises(R.Refusal, match="pilot"):
        R.assert_fillers(cells, exemplars, pilot_used={cells[16][0]["filler_sentences"][0]})


def test_filler_violations_catch_digits_names_and_code():
    assert R.filler_violations("The Alpha squad sent code 4 to Bravo and Delta.") == ["4", "alpha", "bravo", "code", "delta"]
    assert R.filler_violations("A quiet river under a silver morning.") == []
    assert not R.eligible_sentence("The code was 1234.")
    assert not R.eligible_sentence("Alphabet soup is served daily here.")
    assert R.eligible_sentence("The pilot qualification most relevant here is the licence.")


def test_sentence_pool_fill_is_exact_and_consumes_without_replacement():
    tok = WordTok()
    sents = make_sentences(500, seed=3)
    pool = R.SentencePool(sents, list(range(500)))
    for target in (3, 16, 64, 257):
        text, used = pool.fill(target, lambda t: len(tok.encode(" " + t)))
        assert len(tok.encode(" " + text)) == target
        assert len(set(used)) == len(used)
    assert len(set(pool.used)) == len(pool.used)


def test_wikitext_line_normalisation_and_split():
    line = " The band @-@ members recorded in Paris , France . It sold 1 @,@ 000 copies . A quiet river runs under the silver morning . Too short ."
    sents = R.sentences_from_lines([line])
    assert sents == ["The band-members recorded in Paris, France.",            # @-@ and ' ,' normalised
                     "A quiet river runs under the silver morning."]          # the digit sentence and the short one are dropped
    assert R.normalize_wikitext_line(" = Heading = ") == ""
    assert R.sentences_from_lines([line] * 5, cap=3) == sents + sents[:1]


def test_split_orders_are_disjoint_and_seeded():
    main, smoke, pilot = R.split_orders(10000, reserve=1000, smoke_reserve=500)
    assert len(main) == 8500 and len(smoke) == 500 and len(pilot) == 1000
    sets = [set(x.tolist()) for x in (main, smoke, pilot)]
    for a, b in itertools.combinations(sets, 2):
        assert not a & b
    assert sets[0] | sets[1] | sets[2] == set(range(10000))
    main2, smoke2, pilot2 = R.split_orders(10000, reserve=1000, smoke_reserve=500)
    assert main.tolist() == main2.tolist() and smoke.tolist() == smoke2.tolist() and pilot.tolist() == pilot2.tolist()
    # the smoke and pilot slices are reshuffled by their own seeds, so a different seed gives a different order of the same set
    _, smoke3, _ = R.split_orders(10000, reserve=1000, smoke_reserve=500, seed_smoke=R.SEED_SMOKE + 1)
    assert set(smoke3.tolist()) == sets[1] and smoke3.tolist() != smoke.tolist()
    with pytest.raises(R.Refusal):
        R.split_orders(R.PILOT_RESERVE + R.SMOKE_RESERVE)


# -- prompts ---------------------------------------------------------------------------------------------

def test_prompt_length_bound_and_compositionality(small_cells):
    tok, cells, exemplars, _, codes_len = small_cells
    info = R.assert_prompts(cells, exemplars, {"a": tok, "b": tok}, "yes_no", {"a": codes_len, "b": codes_len})
    assert info["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert info["prompts_checked"] == 2 * 48 * 2 * 2
    it = cells[1024][0]
    parts = R.prompt_parts(it, exemplars, "compare", "yes_no")
    ids = R.compositional_ids(tok, parts)
    assert ids is not None and len(tok.encode(parts[1])) == 1024
    abl = tok.encode(R.prompt_text(it, exemplars, "compare", "yes_no", ablated=True))
    assert len(abl) == len(ids)
    assert "Question: Is the" in parts[2] and parts[2].endswith("Answer:")
    rec = R.prompt_parts(it, exemplars, "recall", "yes_no")[2]
    assert rec == f"\nQuestion: What is the {R.NAMES[it['q_slot']]} code? Answer:"
    assert R.prompt_text(it, exemplars, "compare", "yes_no").count("\n\n") == R.N_EXEMPLARS


def test_prompt_length_bound_refused_when_exceeded(small_cells):
    tok, cells, exemplars, _, codes_len = small_cells
    old = R.MAX_PROMPT_TOKENS
    R.MAX_PROMPT_TOKENS = 100
    try:
        with pytest.raises(R.Refusal, match="bound"):
            R.assert_prompts(cells, exemplars, {"a": tok}, "yes_no", {"a": codes_len})
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
    assert set(pools) == {"main_exemplar", "pilot_exemplar", "pilot_test", "smoke_exemplar", "smoke_test", "main_test"}
    assert len(pools["main_exemplar"]) == 12 and len(pools["pilot_exemplar"]) == 12 and len(pools["pilot_test"]) == 60
    assert len(pools["smoke_exemplar"]) == 12 and len(pools["smoke_test"]) == 60
    assert sum(len(v) for v in pools.values()) == 9000
    with pytest.raises(R.Refusal):
        R.prepare_items(tok, 24, (16,), 1, "scratch", fmts=("yes_no",), sentences=make_sentences(8000))


def test_exemplar_text_formats():
    exs = R.build_exemplars([str(1000 + i) for i in range(12)], np.random.default_rng(0))
    t = R.exemplar_text(exs[0], "compare", "yes_no")
    assert t.endswith("Answer: Yes") and t.startswith("The alpha code is 1000.")
    t = R.exemplar_text(exs[1], "compare", "true_false")
    assert t.endswith("Answer: False")
    t = R.exemplar_text(exs[2], "recall", "yes_no")
    assert t.endswith(f"Answer: {exs[2]['codes'][2]}")


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
    plan = Path("C:/Users/heyzo/clawd/wt/_frequency_2026_10_11/PLAN_next_experiment.md")
    if not plan.exists():
        pytest.skip("the judged plan is not on this machine")
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
    labels = [it["label"] for it in items]
    for const in (+3.0, -3.0, 0.0):
        s = R.compare_cell_summary(np.full(len(items), const), labels)
        assert s["acc_raw"] == 0.5 and s["acc_norm"] == 0.0
        assert s["guard_ok"] is False and s["auroc"] == 0.5
    s = R.compare_cell_summary(np.zeros(len(items)), labels)         # exact ties break toward No
    assert s["yes_rate"] == 0.0 and s["near_ties"] == len(items)
    for slot in range(3):
        r = R.recall_cell_summary(np.full(len(items), slot), [it["q_slot"] for it in items])
        assert abs(r["acc_raw"] - 1 / 3) < 1e-12 and abs(r["acc_norm"]) < 1e-12
        assert r["guard_ok"] is False and r["max_slot_fraction"] == 1.0
    perfect = R.compare_cell_summary(np.where(labels, 2.0, -2.0), labels)
    assert perfect["acc_norm"] == 1.0 and perfect["guard_ok"] and perfect["auroc"] == 1.0


def test_norm_acc_endpoints():
    assert R.norm_acc(0.5, 0.5) == 0.0 and R.norm_acc(1.0, 0.5) == 1.0
    assert abs(R.norm_acc(1 / 3, 1 / 3)) < 1e-12 and R.norm_acc(1.0, 1 / 3) == 1.0
    assert R.norm_acc(0.70, 0.5) == pytest.approx(0.40) and R.norm_acc(0.60, 1 / 3) == pytest.approx(0.40)


def test_auroc_ties_and_separation():
    assert R.auroc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]) == 1.0
    assert R.auroc([0.4, 0.3, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert R.auroc([1.0, 1.0, 1.0, 1.0], [False, True, False, True]) == 0.5
    assert R.auroc([1.0, 1.0], [True, True]) is None


def test_bootstrap_is_deterministic_and_matches_the_point_estimate():
    rng = np.random.default_rng(7)
    n = 600
    cT, cM, rT, rM = (rng.random(n) < p for p in (0.85, 0.70, 0.80, 0.78))
    b1 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT)
    b2 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT)
    assert b1 == b2
    b3 = R.paired_bootstrap(cT, cM, rT, rM, n_resamples=500, seed=R.SEED_BOOT + 1)
    assert b3["ci_lo"] != b1["ci_lo"]
    C_T, C_M = R.norm_acc(cT.mean(), 0.5), R.norm_acc(cM.mean(), 0.5)
    R_T, R_M = R.norm_acc(rT.mean(), 1 / 3), R.norm_acc(rM.mean(), 1 / 3)
    assert b1["point"] == pytest.approx((C_T - C_M) - (R_T - R_M), abs=1e-12)
    assert b1["ci_lo"] <= b1["point"] <= b1["ci_hi"]
    assert b1["resamples"] == 500 and b1["seed"] == R.SEED_BOOT and b1["n_items"] == n


# -- budget, format line, receipt hygiene ----------------------------------------------------------------

def test_budget_extrapolation():
    t = {"T": {16: {"L": 230, "s_per_batch": {"compare": 0.2, "recall": 0.2}},
               1024: {"L": 1250, "s_per_batch": {"compare": 0.8, "recall": 0.8}}}}
    b = R.extrapolate_budget(t)
    assert b["arms_untimed"] == ["M1"] and b["verdict"] == "OK_PARTIAL" and not b["exceeds_budget"]
    d16 = b["per_arm"]["T"]["by_distance"]["16"]
    assert d16["batch_size"] == 16 and d16["batches_per_pass"] == 38
    assert d16["seconds"] == pytest.approx(0.2 * 4 * 38 * 3, abs=0.2)          # main + ablated + windowed at D0
    t["M1"] = {16: {"L": 230, "s_per_batch": {"compare": 2.0, "recall": 2.0}},
               1024: {"L": 1250, "s_per_batch": {"compare": 8.0, "recall": 8.0}}}
    b = R.extrapolate_budget(t)
    assert b["arms_untimed"] == [] and not b["exceeds_budget"] and b["verdict"] == "OK"
    d1024 = b["per_arm"]["M1"]["by_distance"]["1024"]
    assert d1024["batch_size"] == R.batch_size_for("M1", 1250) == 16 and d1024["batches_per_pass"] == 38
    assert d1024["seconds"] == pytest.approx(8.0 * 4 * 38 * 2, abs=0.2)        # main + ablated at D1, no windowed for M1
    t["M1"][1024]["s_per_batch"] = {"compare": 60.0, "recall": 60.0}
    b = R.extrapolate_budget(t)
    assert b["exceeds_budget"] and b["verdict"] == "INVALID__budget" and b["total_hours_timed_arms"] > 4.0
    with pytest.raises(R.Refusal):
        R.extrapolate_budget({"T": {16: t["T"][16]}})


def test_batch_size_bounds():
    assert R.batch_size_for("T", 1250) == 16
    assert R.batch_size_for("M1", 1250) == 16 == R.BATCH_MAX           # PREREG section 4: batches of 16 for both arms
    assert R.SCAN_CHUNK_TOKENS // R.BATCH_MAX >= 1                      # the chunk of L the scan materialises at B = 16
    assert R.batch_size_for("M1", 10) == 16


def test_answer_format_line_last_one_wins():
    text = PREREG.read_text(encoding="utf-8")
    frozen = text.split("## 17. Amendments")[0]                 # section 9's own line, before any amendment
    assert R.answer_format_from_prereg(frozen) == "PENDING_PILOT"
    amended = frozen + "\n## 17.1 Amendment\n\nAnswer format chosen (written before sealing): true_false\n"
    assert R.answer_format_from_prereg(amended) == "true_false"
    assert R.answer_format_from_prereg(frozen + "\nAnswer format chosen (written before sealing): NOT_STARTED\n") == "NOT_STARTED"
    with pytest.raises(R.Refusal):
        R.answer_format_from_prereg(text + "\nAnswer format chosen (written before sealing): maybe\n")
    with pytest.raises(R.Refusal):
        R.answer_format_from_prereg("no such line")
    # the document as it stands: PENDING_PILOT while no amendment has been entered, else what the amendment says
    assert R.answer_format_from_prereg(text) in R.FORMAT_LINE_VALUES


def test_strip_private_and_numpy_types(tmp_path):
    obj = {"a": np.float64(1.5), "b": np.bool_(True), "c": np.int64(3), "_hidden": [1], "d": [np.float32(2.0)], "e": np.arange(2)}
    out = R.strip_private(obj)
    assert out == {"a": 1.5, "b": True, "c": 3, "d": [2.0], "e": [0, 1]}
    p = tmp_path / "r.json"
    R.write_json(p, obj)
    assert json.loads(p.read_text(encoding="utf-8")) == out


def test_base_receipt_has_the_prereg_layout():
    res = R.base_receipt("main", "yes_no", 600, R.DISTANCES, ["M1", "T"])
    for key in ("config", "pilot", "acc_raw", "acc_norm", "guard", "controls", "result", "bootstrap", "mechanism", "exploratory", "redteam", "runtime"):
        assert key in res
    assert res["exploratory"]["status"] == "M3_NOT_RUN"
    assert res["config"]["screen_bar"] == 0.40 and res["config"]["n_per_cell"] == 600
    assert res["config"]["seeds"] == {"items": 20261011, "pilot": 1011, "bootstrap": 20261011}
    assert R.SEED_SMOKE not in (R.SEED_ITEMS, R.SEED_PILOT)
    assert set(res["controls"]) == {"ablated", "windowed"}
    assert res["config"]["scan_chunk_tokens_M1"] == R.SCAN_CHUNK_TOKENS and res["config"]["batch_max"] == 16
    assert res["config"]["runner"] == RUNNER.name and res["config"]["runner_sha256"] == R.sha256_file(RUNNER)
    assert re.fullmatch(r"[0-9a-f]{64}", res["config"]["runner_sha256"])


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


def test_neox_tokenizer_answer_tokens_codes_and_identical_ids():
    toks = _neox_tokenizers()
    for arm, t in toks.items():
        assert R.answer_token_ids(t, "yes_no") == {" Yes": 6279, " No": 1621}
        assert R.answer_token_ids(t, "true_false") == {" True": 11793, " False": 16708}
        assert t.encode(" 4271") == [577, 28209]
        codes, L = R.eligible_codes(t)
        assert L == 2 and len(codes) == 8828
    tok = toks["T"]
    cells, exemplars, pool, codes_len = make_cells(tok, 24, (16, 1024), n_sent=4000)
    for D, items in cells.items():
        R.assert_balance(items, exemplars)
        for it in items:
            assert len(tok.encode(" " + it["filler"])) == D
    R.assert_fillers(cells, exemplars)
    info = R.assert_prompts(cells, exemplars, toks, "yes_no", {arm: codes_len for arm in toks})
    assert info["prompts_checked"] == 2 * 24 * 4 and info["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert info["max_prompt_tokens"] < 1400                      # the PREREG's estimate: about 1,250 at D = 1024
    it = cells[1024][0]
    parts = R.prompt_parts(it, exemplars, "compare", "yes_no")
    assert R.compositional_ids(tok, parts) is not None
    assert tok.encode(parts[2])[-1] == 27                          # ':' of 'Answer:' — the position whose next token is scored


def test_neox_tokenizer_on_the_real_wikitext_pool_when_cached():
    toks = _neox_tokenizers()
    if R.snapshot_dir("wikitext", kind="datasets") is None:
        pytest.skip("wikitext-103-raw-v1 is not cached here")
    pytest.importorskip("pyarrow")
    sents, prov = R.read_wikitext_sentences(cap=10000)
    assert prov["eligible_sentences"] == 10000 and len(prov["sha256"]) == 2
    assert all(R.eligible_sentence(s) for s in sents)
    prep = R.prepare_items(toks["T"], 24, (16, 1024), R.SEED_ITEMS, "main", fmts=("yes_no",), sentences=sents)
    red = R.tokenizer_level_asserts(prep, toks, "yes_no")
    assert red["code_token_length"] == 2 and red["prompts"]["max_prompt_tokens"] <= R.MAX_PROMPT_TOKENS
    assert red["balance_by_distance"]["16"]["label_true"] == 12
    smoke = R.prepare_items(toks["T"], 24, (16, 1024), R.SEED_SMOKE, "smoke", fmts=("yes_no",), sentences=sents)
    pilot = R.prepare_items(toks["T"], 24, (16,), R.SEED_PILOT, "pilot", fmts=("yes_no", "true_false"), sentences=sents)
    used = {k: set(p["pool"].used) for k, p in (("main", prep), ("smoke", smoke), ("pilot", pilot))}
    for a, b in itertools.combinations(used, 2):
        assert not used[a] & used[b], (a, b)
    # the main run's reuse assert, fed the smoke's and the pilot's sentences, passes on disjoint pools and refuses on overlap
    R.assert_fillers(prep["cells"], prep["exemplars"], used["smoke"] | used["pilot"])
    with pytest.raises(R.Refusal, match="smoke/pilot"):
        R.assert_fillers(prep["cells"], prep["exemplars"], {next(iter(used["main"]))})

    def codes_of(p):
        return {c for items in p["cells"].values() for it in items for c in it["codes"]} | {c for e in p["exemplars"] for c in e["codes"]}
    cs = {k: codes_of(p) for k, p in (("main", prep), ("smoke", smoke), ("pilot", pilot))}
    for a, b in itertools.combinations(cs, 2):
        assert not cs[a] & cs[b], (a, b)
    assert smoke["sentence_region"]["which"] == "smoke" and smoke["sentence_region"]["seed"] == R.SEED_SMOKE


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
    cells, exemplars, _, codes_len = make_cells(tok, 24, (16,), n_sent=2000)
    items = cells[16]
    ce = R.check_chunked_scan_equivalence(model, tok, items[0], exemplars, "yes_no")
    assert ce["max_abs_logit_diff"] <= 1e-5 and ce["chunks"] >= 1, ce
    rp = R.check_right_padding("M1", model, tok, items, exemplars, "yes_no")
    assert rp["max_abs_logit_diff"] <= R.FP32_TOL, rp
    ans = R.answer_token_ids(tok, "yes_no")
    mech = R.MechanismHook(model)
    try:
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "compare", "yes_no", 16, ans, codes_len, mech=mech, batch_size=4)
        assert s["n"] == 8 and len(s["_correct"]) == 8 and s["batches"] == 2
        assert len(s["_mechanism"]["by_layer_mean_log_surviving"]) == 24
        assert s["_mechanism"]["slowest_layer_log_surviving"] <= 0.0
        assert len(s["_mechanism"]["by_layer_median_channel_log_surviving"]) == 24
        assert s["_mechanism"]["median_channel_log_surviving_max_over_layers"] <= s["_mechanism"]["slowest_layer_log_surviving"] + 1e-9
        assert all(0.0 <= f <= 1.0 for f in s["_mechanism"]["by_layer_frac_channels_surviving_half"])
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "recall", "yes_no", 16, ans, codes_len, batch_size=4)
        assert s["n"] == 8 and sum(s["slot_fractions"]) == pytest.approx(1.0) and s["batches"] == 6
        s = R.run_cell("M1", model, tok, items[:8], exemplars, "compare", "yes_no", 16, ans, codes_len, ablated=True, batch_size=4)
        assert s["n"] == 8
    finally:
        mech.remove()
        del model

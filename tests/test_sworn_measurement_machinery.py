# -*- coding: utf-8 -*-
"""The sworn measurement machinery, pinned on synthetic items.

Spec: papers/sworn/SPEC_sworn_measurement_machinery_2026_09_05.md, section "Tests this spec
commits to". LOAD-BEARING: test_the_unit_set_of_every_population_document_reconciles (a unit set
that disagrees with a committed receipt is a different splitter), test_wilson_cannot_clear_the_bar
_at_thirty_of_thirty (the arithmetic the signature has to know), the canary tests (the verifier
FAILs every planted falsehood on synthetic bytes and a MALFORMED canary stays in n), and the
scorer's WITHHELD path. Nothing here reads a real document into a model; the one real thing read
is the committed population at its pinned commit, through git plumbing, to reconcile a count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MEAS = ROOT / "papers" / "sworn" / "measurement"
if str(MEAS) not in sys.path:
    sys.path.insert(0, str(MEAS))

import common as C                                   # noqa: E402
import population as P                               # noqa: E402
from styxx import sworn                              # noqa: E402

POPULATION = MEAS / "population.json"


def _wilson_b23(k, n, z=1.96):
    """run_b23_fable.wilson, transcribed (importing it would import torch)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ============================================================================ the population

class TestPopulation:
    @pytest.mark.skipif(not POPULATION.exists(), reason="population.json not built")
    def test_the_unit_set_of_every_population_document_reconciles(self):
        pop = json.loads(POPULATION.read_text(encoding="utf-8"))
        assert pop["schema"] == P.SCHEMA and pop["rule"] == P.RULE
        for e in P.iter_documents(pop):
            side_b = C.show_at(pop["pinned_commit"], e["stem"] + ".sworn.json")
            rec_b = C.show_at(pop["pinned_commit"], e["stem"] + ".sworn-receipt.json")
            assert side_b is not None and rec_b is not None, e["stem"]
            side = sworn.load_sidecar(json.loads(side_b.decode("utf-8")))
            rec = json.loads(rec_b.decode("utf-8"))
            units = C.units_of(side)
            ok, mine, theirs = C.reconcile_units(units, rec)
            assert ok, (e["stem"], mine, theirs)
            assert e["narrative_sentences"] == theirs
            assert e["sworn_total"] == len(side["spans"])
            assert e["units"] == len(units)
            assert side["document"]["sha256"] == e["document_sha256"]

    @pytest.mark.skipif(not POPULATION.exists(), reason="population.json not built")
    def test_the_rule_excludes_the_builders_documents_and_names_the_reason(self):
        pop = json.loads(POPULATION.read_text(encoding="utf-8"))
        stems = [d["stem"] for d in pop["documents"]]
        assert stems and all(not s.startswith("papers/sworn/") for s in stems)
        assert all(s.rsplit("/", 1)[-1].startswith(P.PREFIXES) for s in stems)
        assert all(x["reason"] for x in pop["excluded"])
        assert any(x["stem"].startswith("papers/sworn/") for x in pop["excluded"])
        ids = [d["doc_id"] for d in pop["documents"]]
        assert ids == ["D%02d" % i for i in range(1, len(ids) + 1)]

    def test_select_applies_the_prefix_and_directory_rule(self):
        sel, exc = P.select("HEAD")
        for s in sel:
            assert not s.startswith("papers/sworn/")
            assert s.rsplit("/", 1)[-1].startswith(P.PREFIXES)
        assert all(e["reason"] for e in exc)


# ============================================================================ units and windows

def _side(text: bytes, spans):
    return {"spec": sworn.SPEC, "commit": "a" * 40,
            "document": {"name": "syn.md", "sha256": C.sha256_bytes(text)},
            "text": text.decode("utf-8"), "spans": spans,
            "manifest": {"spec": sworn.MANIFEST_SPEC, "receipts": {}}}


class TestUnits:
    def test_units_are_sworn_spans_then_stripped_narrative_sentences_with_fragments(self):
        text = b"Intro line.\n\nThe rate was 0.5 here. Then more. -\n\n```\ncode. code.\n```\nEnd.\n"
        s0 = text.index(b"0.5")
        side = _side(text, [{"start": s0, "end": s0 + 3, "receipt": "path:a.json#/x", "kind": "numeric"}])
        units = C.units_of(side)
        sworn_units = [u for u in units if u["sworn"]]
        assert len(sworn_units) == 1 and sworn_units[0]["text"] == "0.5"
        texts = [u["text"] for u in units if not u["sworn"]]
        assert "Intro line." in texts and "Then more." in texts and "End." in texts
        assert not any("code." in t for t in texts)
        frags = [u for u in units if u["fragment"]]
        assert [u["text"] for u in frags] == ["-"]
        for u in units:
            assert text[u["start"]:u["end"]] == u["text"].encode("utf-8")

    def test_windows_cut_only_at_blank_lines_and_pack_to_the_cap(self):
        paras = ["P%d one. P%d two. P%d three." % (i, i, i) for i in range(6)]
        text = ("\n\n".join(paras) + "\n").encode("utf-8")
        units = C.units_of(_side(text, []))
        assert len(units) == 18
        wins = C.windows_of(text, units, max_units=7)
        assert all(len(w["units"]) <= 7 for w in wins)
        assert sum(len(w["units"]) for w in wins) == 18
        assert all(not w["oversize"] for w in wins)
        for w in wins:
            assert text[w["start"]:w["end"]].strip().startswith(b"P")
        wins1 = C.windows_of(text, units, max_units=2)
        assert all(w["oversize"] for w in wins1) and len(wins1) == 6

    def test_a_passage_takes_whole_lines_and_keys_the_unit_relative_to_itself(self):
        text = b"Line one is short. Line one again.\nLine two holds 42 things.\nLine three.\n"
        units = C.units_of(_side(text, []))
        i = [k for k, u in enumerate(units) if "42" in u["text"]][0]
        p = C.passage_around(text, units, i)
        passage = text[p["start"]:p["end"]]
        assert passage == b"Line one is short. Line one again.\nLine two holds 42 things.\nLine three."
        k = p["keyed"]
        assert passage[k["start"]:k["end"]] == b"Line two holds 42 things."


class TestBindable:
    def test_each_kind_and_a_fragment(self):
        assert C.bindable(b"The rate was 0.55 on the panel.")["numeric"] is True
        assert C.bindable(b"On 2026-09-05 the rate was 0.55.")["numeric"] is False
        assert C.bindable(b"The log reads `a needle of sixteen bytes` here.")["quote"] is True
        assert C.bindable(b"The log reads `short` here.")["quote"] is False
        h = "a" * 64
        assert C.bindable(("bytes hash to %s." % h).encode())["hash"] is True
        assert C.bindable(("%s and %s." % (h, h)).encode())["hash"] is False
        b = C.bindable(b"-")
        assert b["any"] is False and b["absent"] is None


# ============================================================================ location, projection

class TestProjection:
    def test_locate_is_exact_first_then_collapsed_and_unlocated_on_a_duplicate_opening(self):
        item = b"Alpha beta gamma delta.  Alpha beta gamma epsilon.\nZeta  eta theta."
        hit, how = C.locate(item, "Zeta  eta theta.", "eta theta.")
        assert how == "exact" and item[hit[0]:hit[1]] == b"Zeta  eta theta."
        hit, how = C.locate(item, "Zeta eta theta.", "eta theta.")
        assert how == "collapsed" and item[hit[0]:hit[1]] == b"Zeta  eta theta."
        hit, how = C.locate(item, "Alpha beta gamma", "gamma delta.")
        assert (hit, how) == (None, "unlocated")
        hit, how = C.locate(item, "Alpha beta gamma delta.", "nowhere at all.")
        assert (hit, how) == (None, "unlocated")

    def test_project_labels_takes_the_largest_overlap_and_ties_are_no_label(self):
        units = [(0, 10), (10, 20), (20, 30), (30, 40)]
        brackets = [{"start": 0, "end": 12, "label": "NOT"},
                    {"start": 12, "end": 25, "label": "LOAD-BEARING"},
                    {"start": 26, "end": 30, "label": "UNSURE"}]
        assert C.project_labels(brackets, units) == ["NOT", "LOAD-BEARING", "LOAD-BEARING", "NO-LABEL"]
        even = brackets[:2] + [{"start": 25, "end": 30, "label": "UNSURE"}]
        assert C.project_labels(even, units)[2] == "NO-LABEL"
        tie = [{"start": 0, "end": 5, "label": "NOT"}, {"start": 5, "end": 10, "label": "LOAD-BEARING"}]
        assert C.project_labels(tie, [(0, 10)]) == ["NO-LABEL"]
        same = [{"start": 0, "end": 5, "label": "NOT"}, {"start": 5, "end": 10, "label": "NOT"}]
        assert C.project_labels(same, [(0, 10)]) == ["NOT"]

    def test_majority_family_and_final_rules(self):
        assert C.majority(["A", "A", "B"]) == "A"
        assert C.majority(["A", "B", "C"]) is None
        assert C.majority([]) is None
        assert C.family_label(["LOAD-BEARING", "LOAD-BEARING", "NOT"]) == "LOAD-BEARING"
        assert C.family_label(["LOAD-BEARING", "NOT", "UNSURE"]) == "NO-MAJORITY"
        assert C.family_label(["NO-LABEL", "NO-LABEL", "NOT"]) == "NO-MAJORITY"
        assert C.family_label(["NOT", "NO-LABEL", "NOT"]) == "NOT"
        assert C.family_label(["YES", "YES", "UNSURE"], C.LABELS_R) == "YES"
        assert C.final_label("NOT", "NOT") == "NOT"
        assert C.final_label("NOT", "LOAD-BEARING") == "FAMILY-SPLIT"
        assert C.final_label("NOT", "NO-MAJORITY") == "NO-MAJORITY"

    def test_block_order_rotates_and_seat_one_is_as_written(self):
        assert C.block_order(1) == ["TASK", "FORMAT", "CAUTIONS"]
        assert C.block_order(2) == ["FORMAT", "CAUTIONS", "TASK"]
        assert C.block_order(3) == ["CAUTIONS", "TASK", "FORMAT"]
        assert C.instructions("L", 1).startswith("TASK.")
        assert C.instructions("R", 3).startswith("CAUTIONS.")


# ============================================================================ statistics

class TestStatistics:
    def test_cohen_kappa_on_a_hand_computed_table(self):
        # 2x2: a=20 agree-A, d=15 agree-B, b=5 (A,B), c=10 (B,A); n=50
        a = ["A"] * 20 + ["A"] * 5 + ["B"] * 10 + ["B"] * 15
        b = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15
        r = C.cohen_kappa(a, b)
        po = 35 / 50
        pe = (25 / 50) * (30 / 50) + (25 / 50) * (20 / 50)
        assert r["n"] == 50 and abs(r["po"] - po) < 1e-12 and abs(r["pe"] - pe) < 1e-12
        assert abs(r["kappa"] - (po - pe) / (1 - pe)) < 1e-12
        r2 = C.cohen_kappa(a + ["UNSURE"], b + ["A"])
        assert r2["n"] == 50 and r2["excluded"] == 1
        assert C.cohen_kappa([], [])["n"] == 0

    def test_wilson_equals_the_b23_formula_on_a_grid(self):
        for n in (1, 5, 30, 73, 200):
            for k in range(0, n + 1, max(1, n // 7)):
                assert C.wilson(k, n) == pytest.approx(_wilson_b23(k, n))

    def test_wilson_cannot_clear_the_bar_at_thirty_of_thirty(self):
        assert C.wilson(30, 30)[0] < 0.95
        assert C.wilson(73, 73)[0] >= 0.95 and C.wilson(72, 72)[0] < 0.95
        assert C.smallest_n_clearing(0.95, 0) == 73
        assert C.smallest_n_clearing(0.95, 1) == 110
        assert C.smallest_n_clearing(0.95, 2) == 142


# ============================================================================ bytes on disk

class TestBytes:
    def test_write_json_lf_is_lf_only_and_key_bytes_are_sorted(self, tmp_path):
        p = C.write_json_lf(tmp_path / "x.json", {"b": 1, "a": [1, 2]})
        raw = p.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        kb = C.key_bytes({"z": 1, "a": 2})
        assert kb.index(b'"a"') < kb.index(b'"z"') and kb.endswith(b"\n")
        assert C.salted_digest(kb, "s") != C.salted_digest(kb, "t")

    def test_no_file_under_the_measurement_directory_ends_sworn_json(self):
        bad = [p for p in MEAS.rglob("*") if p.name.endswith((".sworn.json", ".sworn-receipt.json"))]
        assert bad == []

    def test_every_committed_json_under_the_measurement_directory_is_lf_only(self):
        for p in MEAS.rglob("*.json"):
            if "__pycache__" in p.parts:
                continue
            assert b"\r" not in p.read_bytes(), p

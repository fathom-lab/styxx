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


# ============================================================================ rung 3: packets, keys, canaries

import build_packets as B                            # noqa: E402
import canaries as CAN                               # noqa: E402
import seal_key as K                                 # noqa: E402
import synthetic as S                                # noqa: E402


@pytest.fixture(scope="module")
def synpop(tmp_path_factory):
    """A synthetic population, its packets and sealed keys, built once under a temp root."""
    root = tmp_path_factory.mktemp("synroot")
    out = root / "dryrun"
    pop = S.write_population(out, root)
    C.write_json_lf(out / "population.json", pop)
    sealed = root / "sealed"
    K.new_salt(sealed)
    picks = S.decoy_picks(pop, root)
    info = B.build(pop, out / "population.json", picks=picks, root=root, sealed=sealed, out_dir=out,
                   keys_dir=out / "keys")
    return {"root": root, "out": out, "pop": pop, "sealed": sealed, "picks": picks, "info": info}


class TestPackets:
    def test_two_builds_from_the_same_inputs_are_byte_identical(self, synpop, tmp_path):
        out2 = tmp_path / "dryrun"
        out2.mkdir()
        B.build(synpop["pop"], synpop["out"] / "population.json", picks=synpop["picks"], root=synpop["root"],
                sealed=synpop["sealed"], out_dir=out2, keys_dir=out2 / "keys")
        for name in ("packet_L.json", "packet_R.json", "decoys_L.json", "packets_digest.txt",
                     "keys/sworn_measurement_key_L.sha256", "keys/sworn_measurement_key_R.sha256"):
            assert (synpop["out"] / name).read_bytes() == (out2 / name).read_bytes(), name

    def test_the_packet_carries_only_opaque_ids_and_never_a_verdict_side_or_stem(self, synpop):
        for panel in ("L", "R"):
            p = json.loads((synpop["out"] / ("packet_%s.json" % panel)).read_text(encoding="utf-8"))
            assert p["schema"] == B.PACKET_SCHEMA and p["panel"] == panel
            assert p["question"] == (C.QUESTION_L if panel == "L" else C.QUESTION_R)
            raw = json.dumps(p["items"])
            for word in ('"verdict"', '"side"', '"stem"', '"doc_id"', '"decoy_id"', "SYNX-", "SYN-0"):
                assert word not in raw, (panel, word)
            keys = {tuple(sorted(i)) for i in p["items"]}
            assert keys == ({("id", "text")} if panel == "L" else {("id", "kind", "leaf", "sentence")})
            ids = [i["id"] for i in p["items"]]
            assert ids == ["%s-%04d" % (panel, n) for n in range(1, len(ids) + 1)]
            key = K.load_key("sworn_measurement_key_%s" % panel, synpop["sealed"], synpop["out"] / "keys")
            assert set(key) == set(ids)
            sides = [m for m in key.values() if m["kind"] == "decoy"]
            assert len(sides) == 2 * C.N_DECOYS_PER_SIDE
            assert len({m["side"] for m in sides}) == 2
            assert all(m["kind"] in ("document", "decoy") for m in key.values())

    def test_panel_l_decoys_key_a_sentence_inside_their_passage(self, synpop):
        d = json.loads((synpop["out"] / "decoys_L.json").read_text(encoding="utf-8"))
        assert d["authorship_disclosure"] and d["selection_rule"]
        assert len(d["decoys"]) == 2 * C.N_DECOYS_PER_SIDE
        for r in d["decoys"]:
            t = r["text"].encode("utf-8")
            keyed = t[r["keyed"]["start"]:r["keyed"]["end"]].decode("utf-8")
            assert keyed and keyed in r["text"]
            assert ("0." in keyed) == (r["side"] == "LOAD-BEARING")

    def test_panel_r_decoys_are_yes_as_authored_and_no_retargeted(self, synpop):
        key = K.load_key("sworn_measurement_key_R", synpop["sealed"], synpop["out"] / "keys")
        p = json.loads((synpop["out"] / "packet_R.json").read_text(encoding="utf-8"))
        items = {i["id"]: i for i in p["items"]}
        yes = [items[i] for i, m in key.items() if m["kind"] == "decoy" and m["side"] == "YES"]
        no = [items[i] for i, m in key.items() if m["kind"] == "decoy" and m["side"] == "NO"]
        assert len(yes) == len(no) == C.N_DECOYS_PER_SIDE
        assert {i["sentence"] for i in yes} == {i["sentence"] for i in no}
        assert {i["leaf"]["pointer"] for i in yes}.isdisjoint({i["leaf"]["pointer"] for i in no})
        assert all(m["construction"] == "A" for m in key.values() if m.get("side") == "NO")

    def test_digest_file_lists_every_packet_and_matches_its_bytes(self, synpop):
        lines = (synpop["out"] / "packets_digest.txt").read_text(encoding="utf-8").splitlines()
        assert [ln.split()[1] for ln in lines] == ["packet_L.json", "packet_R.json", "decoys_L.json"]
        for ln in lines:
            hexd, name = ln.split()
            assert hexd == C.sha256_file(synpop["out"] / name)

    def test_a_rebuild_refuses_to_overwrite(self, synpop):
        with pytest.raises(SystemExit, match="REFUSED"):
            B.build(synpop["pop"], synpop["out"] / "population.json", picks=synpop["picks"], root=synpop["root"],
                    sealed=synpop["sealed"], out_dir=synpop["out"], keys_dir=synpop["out"] / "keys")


class TestSealedKeys:
    def test_the_digest_is_salted_and_check_notices_a_moved_key(self, tmp_path):
        sealed, keys = tmp_path / "sealed", tmp_path / "keys"
        K.new_salt(sealed)
        with pytest.raises(SystemExit, match="REFUSED"):
            K.new_salt(sealed)
        r = K.seal("sworn_measurement_key_T", {"L-0001": {"kind": "decoy"}}, sealed, keys)
        assert (keys / "sworn_measurement_key_T.sha256").read_text(encoding="utf-8") == \
            "%s  sworn_measurement_key_T.json\n" % r["digest"]
        assert r["digest"] != C.sha256_bytes(C.key_bytes({"L-0001": {"kind": "decoy"}}))
        assert K.check("sworn_measurement_key_T", sealed, keys)[0]
        assert K.load_key("sworn_measurement_key_T", sealed, keys) == {"L-0001": {"kind": "decoy"}}
        (sealed / "sworn_measurement_key_T.json").write_bytes(C.key_bytes({"L-0001": {"kind": "document"}}))
        assert not K.check("sworn_measurement_key_T", sealed, keys)[0]
        with pytest.raises(SystemExit, match="answer-key digest"):
            K.load_key("sworn_measurement_key_T", sealed, keys)
        assert K.main(["--sealed", str(sealed), "--keys-dir", str(keys), "check", "sworn_measurement_key_T"]) == 1

    def test_release_needs_the_flag_that_says_what_it_asserts(self, tmp_path):
        sealed, keys = tmp_path / "sealed", tmp_path / "keys"
        K.new_salt(sealed)
        K.seal("sworn_measurement_key_T", {"a": 1}, sealed, keys)
        with pytest.raises(SystemExit, match="every-seat-output-is-recorded"):
            K.release("sworn_measurement_key_T", sealed, keys, asserted=False)
        k, s = K.release("sworn_measurement_key_T", sealed, keys, asserted=True)
        assert k.exists() and s.exists() and k.parent == keys
        with pytest.raises(SystemExit, match="REFUSED"):
            K.seal("not_a_measurement_key", {}, sealed, keys)


class TestCanaries:
    def test_the_margin_is_the_rounding_rules_own_quantum(self):
        from decimal import Decimal
        printed, frac = CAN.printed_decimal("0.16")
        assert (printed, frac) == (Decimal("0.16"), 2)
        assert CAN.margin_ok(Decimal("0.164"), printed, frac)[0] is False
        assert CAN.margin_ok(Decimal("0.165"), printed, frac)[0] is False
        ok, margin = CAN.margin_ok(Decimal("0.17"), printed, frac)
        assert ok is True and Decimal(margin) == Decimal("0.01")
        assert CAN.margin_ok(Decimal("0.15"), printed, frac)[0] is True
        p, f = CAN.printed_decimal("23,247.")
        assert (p, f) == (Decimal("23247"), 0)
        assert CAN.printed_decimal("42%")[0] == Decimal("42")

    def _twin(self, synpop, k, **kw):
        e = P.iter_documents(synpop["pop"])[k]
        side, tree, rec = C.open_document(e, root=synpop["root"])
        twin, records, cap = CAN.build_twin(side, tree, k, **kw)
        return side, tree, rec, twin, records, cap

    def test_a_twin_keeps_the_text_and_sha_and_its_spans_are_ordered_and_disjoint(self, synpop):
        side, tree, rec, twin, records, cap = self._twin(synpop, 1)
        assert twin["text"] == side["text"] and twin["document"] == side["document"]
        assert twin["commit"] == side["commit"] and twin["manifest"] == side["manifest"]
        assert set(twin) == {"spec", "commit", "document", "text", "spans", "manifest"}
        assert cap["total"] == len(records) == cap["A"] + cap["B"] + cap["C"] > 0
        assert cap["A"] > 0 and cap["B"] > 0 and cap["C"] > 0
        last = 0
        for s in twin["spans"]:
            assert s["start"] >= last and s["end"] > s["start"]
            last = s["end"]
        assert CAN.reproduces(twin)
        assert len(twin["spans"]) == len(side["spans"]) + cap["B"] + cap["C"]
        for r in records:
            assert r["falsehood_check"] in ("decimal_margin", "bytes_find")
            if r["construction"] == "A":
                assert twin["spans"][r["twin_span_index"]]["receipt"] != side["spans"][r["original_span_index"]]["receipt"]

    def test_on_synthetic_bytes_the_verifier_fails_exactly_the_canaries_and_holds_the_rest(self, synpop):
        """Allowed here and only here: the verifier is called on a twin, on synthetic bytes."""
        side, tree, rec, twin, records, cap = self._twin(synpop, 2)
        core = sworn.verify(sidecar=twin, tree=tree)
        canary_idx = {r["twin_span_index"] for r in records}
        for i, s in enumerate(core["spans"]):
            if i in canary_idx:
                assert s["verdict"] == "FAILED", (i, s["reason"])
                assert s["reason"] in ("value_mismatch", "needle_missing")
            else:
                assert s["verdict"] == "HELD", (i, s["reason"])
        assert core["counts"]["FAILED"] == cap["total"]

    def test_a_forced_malformed_canary_is_reported_malformed_not_failed(self, synpop):
        side, tree, rec, twin, records, cap = self._twin(synpop, 0, force_malformed=True)
        forced = [r for r in records if r["falsehood_check"] == "forced_malformed_for_dry_run"]
        assert len(forced) == 1
        core = sworn.verify(sidecar=twin, tree=tree)
        s = core["spans"][forced[0]["twin_span_index"]]
        assert (s["verdict"], s["reason"]) == ("MALFORMED", "leaf_not_numeric")

    def test_build_seals_the_key_and_writes_only_digests_into_the_tree(self, synpop, tmp_path):
        keys = tmp_path / "keys"
        digest = tmp_path / "twins" / "canary_digest.txt"
        r = CAN.build(synpop["pop"], root=synpop["root"], sealed=synpop["sealed"], keys_dir=keys,
                      digest_path=digest, force_malformed_in="SYN-01")
        lines = digest.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(synpop["pop"]["documents"])
        twins_dir = synpop["sealed"] / CAN.TWINS_DIRNAME
        for ln in lines:
            sha, name, cnt = ln.split()
            assert name.endswith(".canary-twin.json") and cnt.startswith("canaries=")
            assert C.sha256_file(twins_dir / name) == sha
        assert (keys / "sworn_measurement_canary_key.sha256").exists()
        assert not any(p.name.endswith(".canary-twin.json") for p in tmp_path.rglob("*"))
        key = K.load_key(CAN.CANARY_KEY, synpop["sealed"], keys)
        assert set(key) == {d["doc_id"] for d in synpop["pop"]["documents"]}
        assert r["pooled_n"] == sum(len(v["canaries"]) for v in key.values())
        assert r["smallest_n_clearing_bar_at_k_eq_n"]["misses_0"] == 73
        with pytest.raises(SystemExit, match="REFUSED"):
            CAN.build(synpop["pop"], root=synpop["root"], sealed=synpop["sealed"], keys_dir=keys, digest_path=digest)


class TestCommittedOutputs:
    """The real packets and digests under papers/sworn/measurement/, where they exist."""

    @pytest.mark.skipif(not (MEAS / "packets_digest.txt").exists(), reason="packets not built")
    def test_the_committed_packets_match_their_digest_file_and_carry_no_key(self):
        for ln in (MEAS / "packets_digest.txt").read_text(encoding="utf-8").splitlines():
            hexd, name = ln.split()
            assert hexd == C.sha256_file(MEAS / name), name
        for panel in ("L", "R"):
            p = json.loads((MEAS / ("packet_%s.json" % panel)).read_text(encoding="utf-8"))
            raw = json.dumps(p["items"])
            for word in ('"verdict"', '"side"', '"stem"', '"doc_id"', '"decoy_id"', '"source_stem"'):
                assert word not in raw, (panel, word)
            assert (MEAS / p["key_digest_file"]).exists()
        assert not any(p.suffix == ".json" and p.name.startswith("sworn_measurement_")
                       for p in (MEAS / "keys").glob("*"))

    @pytest.mark.skipif(not (MEAS / "twins" / "canary_digest.txt").exists(), reason="twins not built")
    def test_the_tree_holds_canary_digests_and_never_a_twin(self):
        lines = (MEAS / "twins" / "canary_digest.txt").read_text(encoding="utf-8").splitlines()
        assert lines and all(len(ln.split()) == 3 for ln in lines)
        assert not list((MEAS / "twins").glob("*.canary-twin.json"))
        assert (MEAS / "keys" / "sworn_measurement_canary_key.sha256").exists()


# ============================================================ rung 4: the seats, the scorer, the dry run

import dry_run as DR                                 # noqa: E402
import score as SCORE                                # noqa: E402
import seat_claude as SCL                            # noqa: E402
import seat_local as SL                              # noqa: E402


@pytest.fixture(scope="module")
def dryrun(tmp_path_factory):
    """One whole dry run under a temp root: the fixture the scorer tests fold.

    It is the same call the committed receipt came from, on its own synthetic bytes, so a change
    that breaks the ladder breaks this fixture before it reaches a commit.
    """
    root = tmp_path_factory.mktemp("dryrun_root")
    out, sealed = root / "dryrun", root / "sealed"
    summary = DR.main_run(out, sealed)
    return {"root": root, "out": out, "sealed": sealed, "summary": summary,
            "result": json.loads((out / "dry_run_result.json").read_text(encoding="utf-8"))}


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            for row in _walk(v, path + "/" + str(k)):
                yield row
    elif isinstance(node, list):
        for i, v in enumerate(node):
            for row in _walk(v, path + "/%d" % i):
                yield row
    else:
        yield path, node


class TestSeatRunners:
    """No seat reads a real document before the preregistration commit."""

    @pytest.mark.skipif(not (MEAS / "packet_L.json").exists(), reason="packets not built")
    @pytest.mark.parametrize("mod", [SCL, SL])
    def test_a_seat_over_the_real_packet_refuses_without_a_prereg(self, mod):
        with pytest.raises(SystemExit) as ei:
            mod.run("L", 1, MEAS, dry_run=False)
        msg = str(ei.value)
        assert msg.startswith("REFUSED:") and "PREREG" in msg

    @pytest.mark.skipif(not (MEAS / "keys").exists(), reason="keys not built")
    def test_the_key_digests_are_in_the_tree_before_any_seat_file_is(self):
        assert sorted(p.name for p in (MEAS / "keys").glob("*.sha256")) == [
            "sworn_measurement_canary_key.sha256", "sworn_measurement_key_L.sha256",
            "sworn_measurement_key_R.sha256"]
        assert not (MEAS / "seat_outputs").exists(), "a seat file exists before the PREREG commit"

    @pytest.mark.skipif(not (MEAS / "packet_L.json").exists(), reason="packets not built")
    def test_the_transport_check_refuses_a_packet_that_is_not_synthetic(self):
        with pytest.raises(SystemExit, match="REFUSED"):
            SCL.run("L", 1, MEAS, transport_check=True, max_items=1)

    def test_a_dry_run_seat_calls_no_transport_writes_once_and_ledgers_every_item(self, dryrun):
        seat = json.loads((dryrun["out"] / "seat_outputs" / "claude" / "L-seat1.json")
                          .read_text(encoding="utf-8"))
        assert seat["verdict"] == "DRY-RUN" and seat["dry_run"] is True
        assert seat["contamination_probe"] is None and seat["errors"] == []
        packet = json.loads((dryrun["out"] / "packet_L.json").read_text(encoding="utf-8"))
        assert len(seat["items"]) == len(packet["items"])
        ledger = (dryrun["out"] / "seat_outputs" / "claude" / "ledger.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(ln) for ln in ledger.splitlines()]
        assert all(r["dry_run"] for r in rows) and not any(r["item_id"] == "PROBE" for r in rows)
        assert len([r for r in rows if r["panel"] == "L" and r["seat"] == 1]) == len(packet["items"])
        with pytest.raises(SystemExit, match="written once"):
            SCL.run("L", 1, dryrun["out"], dry_run=True, root=dryrun["root"])

    def test_an_unparsed_answer_is_recorded_and_not_guessed(self, dryrun):
        seat = json.loads((dryrun["out"] / "seat_outputs" / "claude" / "L-seat2.json")
                          .read_text(encoding="utf-8"))
        assert seat["unparsed"] == ["L-0001"]
        row = [r for r in seat["items"] if r["id"] == "L-0001"][0]
        assert row["parsed"] is False and "brackets" not in row

    def test_a_transport_reason_is_this_files_sentence_and_never_the_exceptions(self):
        import subprocess
        assert SCL._classify(subprocess.TimeoutExpired("claude", 1)) == \
            "the transport did not answer inside the timeout"
        assert SCL._classify(FileNotFoundError(2, "no such file")) == \
            "the transport executable was not found"
        assert SCL._classify(ValueError("Expecting value: line 1 column 1")) == \
            "the transport's bytes were not the JSON envelope the transport promises"
        assert SCL._classify(ZeroDivisionError("division by zero")) == \
            "the transport call raised ZeroDivisionError"


class TestDryRunRefusals:
    def test_it_refuses_the_real_sealed_directory(self, tmp_path):
        with pytest.raises(SystemExit, match="real sealed directory"):
            DR.main_run(tmp_path / "dryrun", C.SEALED)

    def test_it_refuses_an_output_directory_that_is_not_named_dryrun(self, tmp_path):
        with pytest.raises(SystemExit, match="only under a directory named dryrun"):
            DR.main_run(tmp_path / "somewhere", tmp_path / "sealed")

    def test_it_refuses_a_population_entry_that_resolves_to_a_file_in_the_repository(self):
        real = "papers/sworn/SPEC_sworn_measurement_machinery_2026_09_05"
        assert (ROOT / (real + ".md")).exists()
        pop = {"pinned_commit": "a" * 40, "excluded": [],
               "documents": [{"doc_id": "SYN-99", "stem": real, "source": {"kind": "synthetic"}}]}
        with pytest.raises(SystemExit, match="resolves to a file in the repository"):
            DR.refuse_unless_synthetic(pop, ROOT)

    def test_it_refuses_a_doc_id_that_does_not_begin_syn(self):
        pop = {"pinned_commit": "a" * 40, "excluded": [],
               "documents": [{"doc_id": "D01", "stem": "dryrun/D01", "source": {"kind": "synthetic"}}]}
        with pytest.raises(SystemExit, match="does not begin SYN-"):
            DR.refuse_unless_synthetic(pop, ROOT)

    def test_the_scorer_refuses_to_fold_a_synthetic_population_outside_a_dry_run(self, dryrun):
        with pytest.raises(SystemExit, match="folds only under --dry-run"):
            SCORE.fold(dryrun["out"], dryrun["sealed"], dryrun["out"] / "x.json", dry_run=False,
                       root=dryrun["root"])

    def test_a_second_dry_run_into_the_same_directory_refuses(self, dryrun):
        with pytest.raises(SystemExit, match="not empty"):
            DR.main_run(dryrun["out"], dryrun["sealed"])


class TestScorer:
    """Every gate reproduced from the dry run's fixture, and no rate surviving it."""

    def test_every_gate_of_the_spec_is_present_and_marked_proposed_unsigned(self, dryrun):
        g = dryrun["result"]["gates"]
        assert sorted(g) == ["G_C", "G_D", "G_F", "G_G1", "G_P", "G_R", "G_S1", "G_S1X", "G_S2", "G_U"]
        for name, obj in g.items():
            if name == "G_D":
                for panel in obj.values():
                    for fam in panel.values():
                        assert fam["proposed_unsigned"] is True
            else:
                assert obj["proposed_unsigned"] is True, name

    def test_the_dry_run_is_not_quotable_and_says_so_in_its_first_keys(self, dryrun):
        r = dryrun["result"]
        assert list(r)[:3] == ["schema", "dry_run", "quotable"]
        assert r["dry_run"] is True and r["quotable"] is False
        assert r["prereg"] is None and r["population"]["synthetic"] is True

    def test_no_share_interval_kappa_or_q3_value_survives_a_dry_run(self, dryrun):
        rate_keys = ("share", "kappa", "wilson95", "difference", "sentence_share",
                     "panel_coverage", "twin", "original")
        seen = 0
        for path, value in _walk(dryrun["result"]):
            leaf = path.rsplit("/", 1)[-1]
            if leaf in rate_keys or (leaf.isdigit() and path.rsplit("/", 2)[-2] == "wilson95"):
                assert value in (SCORE.NORATE, SCORE.WITHHELD), (path, value)
                seen += 1
        assert seen > 10

    def test_a_family_that_misses_its_decoys_voids_its_panel_and_the_share_is_withheld(self, dryrun):
        g = dryrun["result"]["gates"]
        assert g["G_D"]["R"]["claude"]["pass"] is True
        assert g["G_D"]["R"]["local"]["pass"] is False
        assert g["G_D"]["R"]["local"]["title"] == SCORE.TITLES["G_D"]
        assert g["G_F"]["families_clearing"] == {"L": 2, "R": 1}
        assert g["G_P"]["share"] == SCORE.WITHHELD and g["G_P"]["labels"] == "one-family"
        assert g["G_P"]["denominator"] > 0, "the counts stay when the share is withheld"
        assert "G_P" in dryrun["result"]["withheld"]

    def test_the_two_family_panel_reaches_final_labels_and_the_counts_add_up(self, dryrun):
        c = dryrun["result"]["cells"]
        assert c["labels"] == "final"
        assert c["final_labels"]["LOAD-BEARING"] > 0 and c["final_labels"]["NOT"] > 0
        pop = dryrun["result"]["population"]
        assert sum(c["final_labels"].values()) == pop["units"], "every unit takes exactly one final cell"
        assert c["unlocated_brackets"]["claude"] > 0 and c["located_by_second_pass"]["local"] >= 1
        assert c["unparsed_items"]["claude"] == 1

    def test_a_malformed_canary_counts_in_n_and_not_in_k(self, dryrun):
        gc = dryrun["result"]["gates"]["G_C"]
        per = gc["per_twin"]
        assert sum(v["k"] for v in per.values()) == gc["k"]
        assert sum(v["n"] for v in per.values()) == gc["n"]
        missed = sum(v["malformed"] + v["unresolved"] for v in per.values())
        assert missed == 1 and gc["n"] - gc["k"] == missed
        assert gc["smallest_n_clearing_bar_at_k_eq_n"] == 73

    def test_the_lock_binds_the_scorer_the_packets_the_twins_and_the_keys(self, dryrun):
        names = [Path(row["path"]).name for row in dryrun["result"]["lock"]["inputs"]]
        for want in ("score.py", "packet_L.json", "packet_R.json", "canary_digest.txt",
                     "sworn_measurement_key_L.sha256", "sworn_measurement_key_R.sha256",
                     "sworn_measurement_canary_key.sha256"):
            assert want in names, want
        for row in dryrun["result"]["lock"]["inputs"]:
            assert row["raw_sha256"] and row["content_sha256"]

    def test_the_result_carries_the_disclosures_the_readme_lists_verbatim(self, dryrun):
        assert dryrun["result"]["disclosure"] == list(SCORE.DISCLOSURE)
        assert len(SCORE.DISCLOSURE) == 6

    def test_the_dry_run_wrote_no_sworn_artifact_and_every_byte_it_wrote_is_lf(self, dryrun):
        for p in dryrun["out"].rglob("*"):
            if not p.is_file():
                continue
            assert not p.name.endswith((".sworn.json", ".sworn-receipt.json"))
            assert not p.name.startswith("PREREG_")
            if p.suffix in (".json", ".txt", ".jsonl"):
                assert b"\r" not in p.read_bytes(), p.name

    def test_the_sealed_directory_kept_the_twins_and_the_tree_kept_only_digests(self, dryrun):
        twins = list((dryrun["sealed"] / CAN.TWINS_DIRNAME).glob("*.canary-twin.json"))
        assert twins
        assert not list(dryrun["out"].rglob("*.canary-twin.json"))
        assert not list((dryrun["out"] / "keys").glob("*.json"))

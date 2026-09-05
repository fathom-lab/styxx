# -*- coding: utf-8 -*-
"""The sworn attack battery, pinned. papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02.md.

The lab's standing rule: no instrument is announced before an adversarial pass. This file is
that pass for sworn output, made permanent. Every row of the battery is here in one of two
shapes: an attack v0.2 REPAIRS (the test asserts the repair) or an attack v0.2 DOES NOT REPAIR
(the test asserts the honest behaviour AND that the boundary is printed — a limitation that is
stated is a different thing from one that is hidden). Deleting a NOT-REPAIRED test would not make
the attack go away; it would make the spec lie.

Also here: the finding that motivated the largest v0.2 change — the coverage denominator was a
diff-claim detector fired on result-shaped prose — and the v0.2 rules R1, R5, R6, R7, R9.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx import sworn
from styxx.sworn import (DECISIONS, GitTree, Manifest, MemoryTree, issue_receipt, verify,
                         verify_receipt)

ROOT = Path(__file__).resolve().parent.parent
C40 = "b" * 40
SPEC_V02 = ROOT / "papers" / "sworn" / "SPEC_sworn_output_v02_2026_09_02.md"

REC = json.dumps({"precision": 0.16, "n": 100, "passed": 296, "failed": 0,
                  "note": "All checks passed!"}).encode()


def tree(files=None):
    return MemoryTree(dict(files or {"r.json": REC}), commit=C40)


def sp(text, receipt="path:r.json#/n", kind="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>' % (receipt, kind, text)


def one(doc: bytes, **kw):
    core = verify(doc, name="attack.md", **kw)
    assert len(core["spans"]) == 1, core["spans"]
    return core["spans"][0], core


def harness(rung="L1", **receipts):
    m = Manifest("pytest", "t", minted_at="2026-09-02T00:00:00Z", rung=rung)
    for rid, (data, kind) in receipts.items():
        m.add(rid, data, kind, complete=True, captured_at="2026-09-02T00:00:00Z")
    return m


# ══════════════════════════════════════════════════════════ attacks v0.2 REPAIRS

class TestRepaired:
    def test_a4_a_tag_hidden_in_an_html_comment_is_malformed_not_held(self):
        doc = (b"The finding is in narrative.\n<!-- " + sp("100 items.").encode() + b" -->\n")
        d, core = one(doc, tree=tree())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "hidden_commitment")
        assert core["document_verdict"] == "SWORN-FAILED"
        # v0.1 counted this HELD at coverage 1.0; the floor now reads one MALFORMED over one
        # narrative sentence, and the comment text itself is narrative like any other
        assert core["coverage"]["sworn_total"] == 1

    def test_a10_a_one_byte_needle_over_a_whole_receipt_is_malformed(self):
        d, _ = one(sp("it says `n`.", "path:r.json", "quote").encode(), tree=tree())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "short_needle")

    def test_a10_a_short_needle_the_author_narrowed_is_exempt(self):
        d, _ = one(sp("it says `All checks passed!`.", "path:r.json#/note", "quote").encode(), tree=tree())
        assert d["verdict"] == "HELD"                         # a pointer leaf: exempt
        t = tree({"log.txt": b"line one\npassed\nline three\n"})
        d, _ = one(sp("`passed`", "path:log.txt#L2", "quote").encode(), tree=t)
        assert d["verdict"] == "HELD"                         # a line anchor: exempt
        d, _ = one(sp("no `failed` anywhere", "path:log.txt", "absent").encode(), tree=t)
        assert d["verdict"] == "HELD"                         # absent: the stronger oath

    def test_the_cap_no_longer_penalises_three_byte_scripts(self):
        d, _ = one(sp("字" * 300, "path:r.json", "quote").encode(), tree=tree())
        assert d["reason"] != "length_cap"
        d, _ = one(sp("字" * 301, "path:r.json", "quote").encode(), tree=tree())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "length_cap")


# ══════════════════════════════════════════════════ attacks v0.2 does NOT repair

class TestNotRepairedAndSaidSo:
    """Each of these HELDs (or is otherwise not caught). The assertion is twofold: the behaviour
    is exactly what the spec says it is, and the boundary is printed where a reader will see it."""

    def test_a1_the_rider_clause_holds_on_the_number_and_says_nothing_about_the_qualifier(self):
        d, core = one(sp("Precision was 0.16, comfortably above the preregistered floor.",
                         "path:r.json#/precision").encode(), tree=tree())
        assert d["verdict"] == "HELD"
        assert "NOT a claim that the document is correct" in core["certifies"]
        assert "A1 rider clause" in SPEC_V02.read_text(encoding="utf-8")

    def test_a2_trivial_swearing_holds_and_the_floor_makes_the_padding_visible(self):
        doc = (sp("100 items were scored.") + " "
               + "The instrument therefore works. It should ship to every CI in the world.\n").encode()
        core = verify(doc, name="a.md", tree=tree())
        assert core["document_verdict"] == "SWORN-HELD"
        cov = core["coverage"]
        assert cov["narrative_sentences"] == 2 and cov["sentence_share"] == round(1 / 3, 4)
        assert "estimate" not in cov

    def test_a3_the_stative_dodge_no_longer_shrinks_the_floor(self):
        """v0.1: these sentences were invisible to STRUCT-1 and the estimate read 1.0.
        v0.2: STRUCT-1 still does not see them (that is its documented shape) — but the floor
        counts every sentence, so the dodge buys nothing."""
        doc = (sp("100 items.") + " Every accusation upheld. Zero false positives. Ships today.\n").encode()
        core = verify(doc, name="a.md", tree=tree())
        cov = core["coverage"]
        assert cov["diff_claim_sentences"] == 0            # STRUCT-1's documented blindness
        assert cov["narrative_sentences"] == 3 and cov["sentence_share"] == 0.25

    def test_a5_a_committed_receipt_holds_and_prints_that_its_authorship_was_not_checked(self):
        d, core = one(sp("precision was 0.16.", "path:r.json#/precision").encode(), tree=tree())
        assert d["verdict"] == "HELD"
        assert d["provenance"] == {"form": "path",
                                   "note": "committed object at %s; authorship unchecked" % C40}
        assert core["rungs"] == {"committed": 1}

    def test_a6_a_prereg_digest_proves_content_never_precedence(self):
        bar = b"# PREREG\nthe floor is 0.95 and frozen before any run\n"
        t = tree({"PREREG_x.md": bar, "result.json": REC})
        d, _ = one(sp("`the floor is 0.95 and frozen before any run`", "prereg:" + sworn._sha256(bar),
                      "quote").encode(), tree=t)
        assert d["verdict"] == "HELD"
        assert "the tree at the sidecar's commit" in DECISIONS["prereg_search"]

    def test_a7_a_manifest_the_agent_minted_holds_and_the_rung_it_declares_is_printed(self):
        lying = Manifest("agent-itself", "t", rung=None)
        lying.add("r1", b"296", "test_report", complete=True)
        d, core = one(sp("296 tests passed.", "r1").encode(), manifest=lying)
        assert d["verdict"] == "HELD"
        assert d["provenance"] == {"form": "rn", "harness": "agent-itself", "rung": "undeclared",
                                   "kind_of_source": "test_report"}
        assert core["rungs"] == {"undeclared": 1}
        assert "at the rung the manifest declares" in core["certifies"]

    def test_a11_post_hoc_tagging_is_undetectable_and_the_receipt_says_so(self):
        _, core = one(sp("100 items.").encode(), tree=tree())
        assert "NOT a check that the tags were written at write time" in core["certifies"]

    def test_a12_a_coincident_value_at_the_wrong_leaf_holds(self):
        t = tree({"r.json": json.dumps({"n": 100, "passed": 100}).encode()})
        d, _ = one(sp("100 accusations were scored.", "path:r.json#/passed").encode(), tree=t)
        assert d["verdict"] == "HELD"                          # the author named the wrong leaf
        d, _ = one(sp("100 accusations were scored.", "path:r.json#/passed").encode(), tree=tree())
        assert (d["verdict"], d["reason"]) == ("FAILED", "value_mismatch")   # 296 != 100

    def test_a9_percent_and_fraction_do_not_coincide_survives_from_v01(self):
        d, _ = one(sp("precision was 16%.", "path:r.json#/precision").encode(), tree=tree())
        assert (d["verdict"], d["reason"]) == ("FAILED", "value_mismatch")


# ══════════════════════════════════════════════ the coverage-denominator finding

class TestTheDenominatorWasTheWrongIdiom:
    def test_struct1_does_not_read_a_measured_rate_as_a_claim(self):
        from styxx.claimdetect import detect
        for s in ["Precision was 0.16 on 100 items.", "We measured 0.9515 on n=165.",
                  "The gate accused 13 tokens and 0 were genuine.",
                  "Every accusation upheld. Zero false positives. Ships today."]:
            assert detect(s).is_claim is False, s

    def test_a_result_shaped_document_no_longer_prints_a_near_one_coverage(self):
        doc = (sp("100 accusations were scored.") + " "
               + "Precision was 0.16 against a floor of 0.95. The adjudicator is bad on its own "
               "merits. V14's 0.16 remains unexplained. Three seats of one family judged it.\n").encode()
        core = verify(doc, name="r.md", tree=tree())
        cov = core["coverage"]
        assert cov["diff_claim_sentences"] == 0             # what v0.1's estimate would have used
        assert cov["narrative_sentences"] == 4 and cov["sentence_share"] == 0.2
        assert "result-shaped" in cov["ceiling_note"] and "pull-request prose" in cov["diff_claim_idiom"]


# ═══════════════════════════════════════════════════════════════ v0.2 rules

class TestRules:
    def test_r1_a_manifest_receipt_takes_a_pointer_and_a_line_anchor(self):
        m = harness(r1=(REC, "test_report"), r2=(b"first line\n296 passed in 3.77s\n", "tool_stdout"))
        d, _ = one(sp("296 tests passed.", "r1#/passed").encode(), manifest=m)
        assert d["verdict"] == "HELD" and d["provenance"]["rung"] == "L1"
        d, _ = one(sp("`296 passed in 3.77s`", "r2#L2", "quote").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("no `failed` here", "r1#/note", "absent").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "absent_over_partial")
        d, _ = one(sp("digest " + "0" * 64, "r1#/note", "hash").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "hash_over_partial")
        d, _ = one(sp("296 tests passed.", "r1#/nope").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "pointer_unresolvable")

    def test_r5_attestation_bytes_are_a_receipt_and_no_signature_is_checked(self):
        env = b'{"payloadType":"application/vnd.in-toto+json","payload":"e30=","signatures":[]}'
        m = harness(rung="L2", r1=(env, "attestation"))
        d, core = one(sp("the envelope hashes to " + sworn._sha256(env), "r1", "hash").encode(), manifest=m)
        assert d["verdict"] == "HELD" and d["provenance"]["kind_of_source"] == "attestation"
        assert "NOT a check of any signature" in core["certifies"]
        assert "attestation" in sworn.SOURCE_KINDS_EXTERNAL

    def test_r6_rungs_l1_and_l2_resolve_l3_and_nonsense_are_unresolved_never_accusations(self):
        for rung in ("L1", "L2"):
            d, core = one(sp("296 tests passed.", "r1#/passed").encode(),
                          manifest=harness(rung=rung, r1=(REC, "test_report")))
            assert d["verdict"] == "HELD" and core["rungs"] == {rung: 1}
        for rung in ("L3", "banana", "l1"):
            d, core = one(sp("296 tests passed.", "r1#/passed").encode(),
                          manifest=harness(rung=rung, r1=(REC, "test_report")))
            assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "rung_unknown"), rung
            assert core["document_verdict"] == "SWORN-HELD"   # unresolved is never an accusation
            assert core["rungs"] == {"unresolved": 1}
        assert sworn.RUNGS == ("L1", "L2") and sworn.RUNGS_RESERVED == ("L3",)

    def test_r6_a_manifest_0_1_still_loads_and_never_reaches_l2(self):
        m = Manifest.from_dict({"spec": "sworn/manifest/0.1", "harness": "old", "turn": "t",
                                "minted_at": "2026-09-01T00:00:00Z", "authored_sha256": [],
                                "receipts": {}})
        m.add("r1", REC, "test_report", complete=True, captured_at="2026-09-01T00:00:00Z")
        assert m.spec == "sworn/manifest/0.1" and "rung" not in m.core()
        d, _ = one(sp("296 tests passed.", "r1#/passed").encode(), manifest=m)
        assert d["verdict"] == "HELD" and d["provenance"]["rung"] == "undeclared"
        assert m.core()["spec"] == "sworn/manifest/0.1"        # re-derives its own digest shape

    def test_r6_the_cli_refuses_to_mint_a_manifest_with_no_rung(self, tmp_path):
        with pytest.raises(SystemExit):
            sworn.main(["manifest", "new", str(tmp_path / "m.json"), "--harness", "h", "--turn", "t"])
        assert sworn.main(["manifest", "new", str(tmp_path / "m.json"), "--harness", "h", "--turn", "t",
                           "--rung", "L2"]) == 0
        assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["rung"] == "L2"

    def test_every_json_the_cli_writes_is_lf_on_every_platform(self, tmp_path):
        """A byte-pinned format whose own CLI CRLF-translated its sidecars on Windows would hash
        the same document differently per platform. Every write goes through _write_json_lf."""
        doc = tmp_path / "d.md"
        doc.write_bytes(sp("296 tests passed.", "r1#/passed").encode() + b"\n")
        mpath, side, rec = tmp_path / "m.json", tmp_path / "d.sworn.json", tmp_path / "rec.json"
        assert sworn.main(["manifest", "new", str(mpath), "--harness", "h", "--turn", "t", "--rung", "L1"]) == 0
        r = tmp_path / "r.json"
        r.write_bytes(REC)
        assert sworn.main(["manifest", "add", str(mpath), "--id", "r1", "--file", str(r),
                           "--kind", "test_report", "--complete", "--note", "canned"]) == 0
        assert sworn.main(["canon", str(doc), "--manifest", str(mpath), "--out", str(side)]) == 0
        assert sworn.main(["verify", str(side), "--out", str(rec)]) == 0
        for p in (mpath, side, rec):
            assert b"\r" not in p.read_bytes(), p.name
        assert json.loads(mpath.read_text(encoding="utf-8"))["receipts"]["r1"]["harness_note"] == "canned"

    def test_r9_the_v1_receipt_re_derives_without_its_coverage_block(self):
        m = harness(r1=(REC, "test_report"))
        doc = sp("296 tests passed.", "r1#/passed").encode()
        rec = issue_receipt(verify(doc, name="d.md", manifest=m), timestamp="2026-09-02T00:00:00Z")
        assert rec["schema"] == "styxx.sworn.verdict-receipt/v1"
        broken = json.loads(json.dumps(rec))
        broken["coverage"]["narrative_sentences"] = 999            # coverage is outside the digest
        res = verify_receipt(broken, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["coverage_reproduces"] is False
        broken["document_verdict"] = "SWORN-FAILED"                 # the core is not
        assert verify_receipt(broken, doc, manifest=m)["status"] == "FAILED"
        unknown = dict(rec, schema="styxx.sworn.verdict-receipt/v9")
        res = verify_receipt(unknown, doc, manifest=m)
        assert res["status"] == "FAILED" and "unknown receipt schema" in res["note"]

    def test_r9_a_committed_v0_receipt_still_checks_on_its_core(self):
        rec_path = ROOT / "papers" / "sworn" / "RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json"
        side_path = rec_path.with_name("RESULT_sworn_v01_ships_2026_09_01.sworn.json")
        receipt = json.loads(rec_path.read_text(encoding="utf-8"))
        side = sworn.load_sidecar(json.loads(side_path.read_text(encoding="utf-8")))
        t = GitTree(ROOT, side["commit"])
        if t._ready() == "commit_absent":
            pytest.skip("commit %s not in this checkout" % side["commit"][:12])
        res = verify_receipt(receipt, sidecar=side, tree=t)
        assert res["status"] == "VERIFIED" and res["schema"] == receipt["schema"]
        if receipt["schema"].endswith("/v0"):
            assert res["coverage_reproduces"] is None and "/v0 receipt" in res["note"]
        else:
            assert res["coverage_reproduces"] is True

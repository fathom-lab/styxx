# -*- coding: utf-8 -*-
"""The /v2 conformance set's only source: the corrected certificate sentence, clause by clause, as
cores and checks a second verifier is held to.

papers/sworn/NOTE_certifies_correction_2026_10_24.md. ``python conformance/sworn/gen_vectors.py --set
receipt_v2`` runs this file under ``conformance/sworn/recorder.py``, folds every top-level call into
``styxx.sworn`` into a vector and writes ``conformance/sworn/receipt_v2/``. So this file is PURE: no git,
no repository tree, no capsule, no node, every clock value given, and every call records. A test whose
name quotes a clause of the /v2 sentence shows that clause on a document, a manifest and an in-memory
tree (which answers for its commit on its caller's word, as the sentence says).

Every core here carries ``schema`` styxx.sworn.verdict-receipt/v2 and the /v2 ``certifies``; the /v1 set
beside it (conformance/sworn/) keeps its cores at /v1.
"""
from __future__ import annotations

import json

import pytest

from styxx import sworn
from styxx.sworn import Manifest, MemoryTree, issue_receipt, verify, verify_receipt

V0, V1, V2 = ("styxx.sworn.verdict-receipt/v%d" % i for i in range(3))
CLOCK = "2026-10-24T00:00:00Z"
C = "c" * 40                                    # the commit an in-memory tree answers for
TABLE = b'{"n": 12, "share": 0.25}\n'
TABLE_SHA = sworn._sha256(TABLE)
STDOUT = b"12\n"                                # a tool printed one number
LOG = b"12 passed in 0.01s\n"


def sp(inner, r, k="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>\n' % (r, k, inner)


def harness(rung="L2", authored=(), **entries):
    """A manifest/0.2 the harness minted. entries: rid -> (bytes, kind_of_source)."""
    m = Manifest("conformance-harness", "turn-v2", minted_at=CLOCK, rung=rung)
    for rid, (data, kind) in entries.items():
        m.add(rid, data, kind, complete=True, captured_at=CLOCK)
    for data in authored:
        m.record_authored(data)
    return m


def tree():
    return MemoryTree({"t/table.json": TABLE, "t/notes.txt": b"nothing here\n"}, commit=C)


def core_of(doc, **kw):
    core = verify(doc.encode("utf-8"), name="v2.md", **kw)
    assert core["schema"] == sworn.RECEIPT_SCHEMA == V2
    assert core["certifies"] == sworn._CERTIFIES
    return core


def only(core):
    assert len(core["spans"]) == 1
    s = core["spans"][0]
    return s["verdict"], s["reason"]


# =============================================================================== cores at /v2

class TestCoreV2:
    def test_spans_that_read_held_or_failed_were_checked_each_by_its_kind(self):
        m = harness(r1=(STDOUT, "tool_stdout"), r2=(TABLE, "file_read"), r3=(LOG, "test_report"))
        assert only(core_of(sp("12 passed.", "r1"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("13 passed.", "r1"), manifest=m)) == ("FAILED", "value_mismatch")
        assert only(core_of(sp("The count is 12.", "r2#/n"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("The share is `0.25`.", "r2#/share", "quote"), manifest=m))[0] == "MALFORMED"
        assert only(core_of(sp("The log reads `12 passed in 0.01s`.", "r3", "quote"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("The log reads `13 passed in 0.01s`.", "r3", "quote"), manifest=m)) == (
            "FAILED", "needle_missing")
        assert only(core_of(sp("No `failed` here at all.", "r3", "absent"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("The digest is %s." % TABLE_SHA, "r2", "hash"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("The digest is %s." % ("0" * 64), "r2", "hash"), manifest=m)) == (
            "FAILED", "digest_mismatch")

    def test_a_hash_span_on_a_manifest_entry_that_carries_no_bytes_is_checked_against_the_sha256_the_manifest_records(self):
        m = Manifest("conformance-harness", "turn-v2", minted_at=CLOCK, rung="L2")
        m.add("r1", None, "attestation", complete=True, captured_at=CLOCK, sha256=TABLE_SHA)
        assert only(core_of(sp("Its digest is %s." % TABLE_SHA, "r1", "hash"), manifest=m)) == ("HELD", None)
        assert only(core_of(sp("Its digest is %s." % ("e" * 64), "r1", "hash"), manifest=m)) == (
            "FAILED", "digest_mismatch")
        # any other kind needs the bytes, and without them the span was not checked
        assert only(core_of(sp("The count is 12.", "r1#/n"), manifest=m)) == ("UNRESOLVED", "manifest_bytes_absent")

    @pytest.mark.parametrize("kind", sorted(sworn.SOURCE_KINDS_AUTHOR))
    def test_an_rn_receipt_the_manifest_records_under_an_author_side_kind_is_taken_as_the_author_s(self, kind):
        m = harness(r1=(STDOUT, kind))
        assert only(core_of(sp("12 passed.", "r1"), manifest=m)) == ("MALFORMED", "receipt_author_minted")

    def test_an_rn_receipt_whose_sha256_the_manifest_lists_in_authored_sha256_is_taken_as_the_author_s(self):
        m = harness(authored=(STDOUT,), r1=(STDOUT, "tool_stdout"))
        assert only(core_of(sp("12 passed.", "r1"), manifest=m)) == ("MALFORMED", "receipt_author_minted")

    def test_a_kind_of_source_outside_the_closed_external_vocabulary_is_not_taken_at_all(self):
        m = harness(r1=(STDOUT, "web_scrape"))
        assert only(core_of(sp("12 passed.", "r1"), manifest=m)) == ("MALFORMED", "kind_of_source_unknown")

    def test_an_unresolved_span_was_not_checked(self):
        # each sentence contradicts its receipt; none was checked, so none FAILED, and the document
        # verdict does not consult UNRESOLVED
        wrong = sp("99 passed.", "r1")
        no_manifest = core_of(wrong)
        assert only(no_manifest) == ("UNRESOLVED", "manifest_absent")
        assert no_manifest["document_verdict"] == "SWORN-HELD" and no_manifest["unresolved"] == 1
        assert only(core_of(wrong, manifest=harness(rung="L3", r1=(STDOUT, "tool_stdout")))) == (
            "UNRESOLVED", "rung_unknown")
        assert only(core_of(wrong, manifest=harness(r2=(STDOUT, "tool_stdout")))) == (
            "UNRESOLVED", "manifest_id_missing")

    def test_a_path_receipt_when_no_manifest_is_given_is_taken_as_not_the_author_s(self):
        core = core_of(sp("The count is 12.", "path:t/table.json#/n"), tree=tree(), commit=C)
        assert only(core) == ("HELD", None)
        assert core["spans"][0]["provenance"]["note"] == "committed object at %s; authorship unchecked" % C

    def test_bytes_the_author_wrote_that_the_harness_never_digested_are_not_caught(self):
        # the manifest lists what the agent produced in ITS turn; the committed file is not among
        # them, so the span holds whoever wrote the file
        m = harness(authored=(b"some other bytes the agent wrote\n",), r1=(STDOUT, "tool_stdout"))
        core = core_of(sp("The count is 12.", "path:t/table.json#/n"), manifest=m, tree=tree(), commit=C)
        assert only(core) == ("HELD", None)

    def test_a_path_receipt_the_manifest_lists_in_authored_sha256_is_taken_as_the_author_s(self):
        m = harness(authored=(TABLE,), r1=(STDOUT, "tool_stdout"))
        core = core_of(sp("The count is 12.", "path:t/table.json#/n"), manifest=m, tree=tree(), commit=C)
        assert only(core) == ("MALFORMED", "receipt_author_minted")

    def test_a_prereg_receipt_when_no_manifest_is_given_or_it_does_not_list_the_bytes_is_taken_as_not_the_author_s(self):
        doc = sp("The digest is %s." % TABLE_SHA, "prereg:%s" % TABLE_SHA, "hash")
        assert only(core_of(doc, tree=tree(), commit=C)) == ("HELD", None)
        m = harness(authored=(b"unrelated\n",), r1=(STDOUT, "tool_stdout"))
        assert only(core_of(doc, manifest=m, tree=tree(), commit=C)) == ("HELD", None)

    def test_a_prereg_receipt_the_manifest_lists_in_authored_sha256_is_taken_as_the_author_s(self):
        doc = sp("The digest is %s." % TABLE_SHA, "prereg:%s" % TABLE_SHA, "hash")
        m = harness(authored=(TABLE,), r1=(STDOUT, "tool_stdout"))
        assert only(core_of(doc, manifest=m, tree=tree(), commit=C)) == ("MALFORMED", "receipt_author_minted")

    def test_a_path_receipt_with_no_tree_was_not_checked(self):
        assert only(core_of(sp("The count is 12.", "path:t/table.json#/n"))) == ("UNRESOLVED", "no_repository")
        m = harness(r1=(STDOUT, "tool_stdout"))
        doc = sp("12 passed.", "r1") + sp("The count is 12.", "path:t/table.json#/n")
        core = core_of(doc, manifest=m)
        assert [(s["verdict"], s["reason"]) for s in core["spans"]] == [("HELD", None), ("UNRESOLVED", "no_repository")]

    def test_a_document_that_swore_nothing_is_unsworn(self):
        core = core_of("# notes\n\nNothing here is sworn.\n")
        assert core["document_verdict"] == "UNSWORN" and core["spans"] == []

    def test_a_document_with_no_canonical_text_is_malformed_at_document_level(self):
        core = core_of("```\nan unbalanced fence\n" + sp("12 passed.", "r1"))
        assert core["document_verdict"] == "SWORN-FAILED" and core["document_malformed"]


# =============================================================================== /v2 receipts

def _redigest(r):
    return issue_receipt({k: v for k, v in r.items() if k not in ("digest", "timestamp", "coverage_sha256")},
                         timestamp=CLOCK)


def _doc_and_manifest():
    return sp("12 passed.", "r1").encode("utf-8"), harness(r1=(STDOUT, "tool_stdout"))


class TestReceiptV2:
    def test_a_v2_receipt_issues_and_re_derives(self):
        doc, m = _doc_and_manifest()
        rec = issue_receipt(verify(doc, name="v2.md", manifest=m), timestamp=CLOCK)
        assert rec["schema"] == V2 and rec["certifies"] == sworn._CERTIFIES
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["schema"] == V2

    def test_a_v2_receipt_whose_certifies_was_edited_fails(self):
        doc, m = _doc_and_manifest()
        rec = issue_receipt(verify(doc, name="v2.md", manifest=m), timestamp=CLOCK)
        edited = dict(rec, certifies="every span here is true")
        res = verify_receipt(edited, doc, manifest=m)
        assert res["status"] == "FAILED" and res["digest_match"] is False
        res = verify_receipt(_redigest(edited), doc, manifest=m)
        assert res["status"] == "FAILED" and res["digest_match"] is True and res["verdict_reproduces"] is False

    def test_a_v2_receipt_carrying_the_v1_sentence_fails(self):
        doc, m = _doc_and_manifest()
        rec = issue_receipt(verify(doc, name="v2.md", manifest=m), timestamp=CLOCK)
        res = verify_receipt(_redigest(dict(rec, certifies=sworn._CERTIFIES_V1)), doc, manifest=m)
        assert res["status"] == "FAILED" and res["digest_match"] is True and res["verdict_reproduces"] is False


# =============================================================================== /v1 receipts under /v2

def _v1_receipt(doc, m):
    """A /v1 receipt, equal in every compared byte to the one a /v1 verifier issued for this document."""
    return issue_receipt(sworn._core_as(verify(doc, name="v2.md", manifest=m), V1), timestamp=CLOCK)


class TestReceiptV1UnderV2:
    def test_a_v1_receipt_re_derives_with_its_own_strings(self):
        doc, m = _doc_and_manifest()
        rec = _v1_receipt(doc, m)
        assert rec["schema"] == V1 and rec["certifies"] == sworn._CERTIFIES_V1
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["schema"] == V1
        assert "held at their /v1 strings" in res["note"]

    def test_a_v1_receipt_whose_certifies_was_edited_and_re_digested_fails(self):
        doc, m = _doc_and_manifest()
        res = verify_receipt(_redigest(dict(_v1_receipt(doc, m), certifies="every span here is true")), doc,
                             manifest=m)
        assert res["status"] == "FAILED" and res["digest_match"] is True and res["verdict_reproduces"] is False

    def test_a_v1_receipt_carrying_the_v2_sentence_fails(self):
        doc, m = _doc_and_manifest()
        res = verify_receipt(_redigest(dict(_v1_receipt(doc, m), certifies=sworn._CERTIFIES)), doc, manifest=m)
        assert res["status"] == "FAILED" and res["verdict_reproduces"] is False

    def test_a_v1_receipt_with_no_certifies_fails(self):
        doc, m = _doc_and_manifest()
        rec = {k: v for k, v in _v1_receipt(doc, m).items() if k != "certifies"}
        res = verify_receipt(_redigest(rec), doc, manifest=m)
        assert res["status"] == "FAILED" and res["verdict_reproduces"] is False

    def test_a_v1_receipt_whose_coverage_moved_still_verifies_and_says_so(self):
        doc, m = _doc_and_manifest()
        rec = json.loads(json.dumps(_v1_receipt(doc, m)))
        rec["coverage"]["narrative_sentences"] = 999            # coverage is outside the digest
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["coverage_reproduces"] is False


# =============================================================================== /v0 receipts under /v2

def _v0_receipt(doc, m, certifies):
    """A /v0 receipt: the v0 shape, its own schema, verifier and certifies, its coverage inside a digest
    over the whole body (as /v0 digested)."""
    fresh = verify(doc, name="v2.md", manifest=m)
    body = dict(sworn._v0_shape(fresh), schema=V0, verifier=fresh["verifier"], certifies=certifies,
                coverage={"schema": "sworn/coverage/0", "estimate": 1.0})
    body["digest"] = sworn._sha256(sworn._jcs(body).encode("utf-8"))
    body["timestamp"] = CLOCK
    return body


class TestReceiptV0UnderV2:
    @pytest.mark.parametrize("which", ["v1_sentence", "v2_sentence", "another"])
    def test_a_v0_receipt_is_checked_as_before_whatever_its_certifies_says(self, which):
        doc, m = _doc_and_manifest()
        certifies = {"v1_sentence": sworn._CERTIFIES_V1, "v2_sentence": sworn._CERTIFIES,
                     "another": "the spans were checked"}[which]
        res = verify_receipt(_v0_receipt(doc, m, certifies), doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["schema"] == V0
        assert res["coverage_reproduces"] is None and "/v0 receipt" in res["note"]

    def test_a_v0_receipt_with_a_moved_verdict_fails(self):
        doc, m = _doc_and_manifest()
        rec = _v0_receipt(doc, m, sworn._CERTIFIES_V1)
        rec["document_verdict"] = "SWORN-FAILED"
        rec["digest"] = sworn._sha256(sworn._jcs({k: v for k, v in rec.items()
                                                  if k not in ("digest", "timestamp")}).encode("utf-8"))
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "FAILED" and res["digest_match"] is True and res["verdict_reproduces"] is False

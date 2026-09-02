"""styxx.sworn v0.1 — every clause of the frozen spec, pinned; every implementation decision, named.

Spec: papers/sworn/SPEC_sworn_output_v01_2026_09_01.md. The load-bearing tests are the negative
ones: a broken tag is never narrative, a document that swore nothing is never clean, an unseen
receipt is never an accusation, and a sidecar that cannot round-trip is never written.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from styxx import sworn
from styxx.sworn import (DECISIONS, KINDS, REASONS, SPEC, VERDICTS, GitTree,
                         Manifest, MemoryTree, issue_receipt, load_sidecar, render, scan,
                         to_sidecar, verify, verify_receipt)

C40 = "a" * 40           # a nominal commit for MemoryTree / sidecars
STAGE2 = Path(__file__).resolve().parent.parent / "papers" / "closed-model-frontier" / "stage2_result.json"


@pytest.fixture(autouse=True)
def _no_cross_test_prereg_cache():
    sworn._PREREG_INDEX.clear()
    yield
    sworn._PREREG_INDEX.clear()


def sp(text, receipt="r1", kind="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>' % (receipt, kind, text)


def manifest(**receipts):
    """receipts: id -> bytes (complete, harness-attested test_report)."""
    m = Manifest("pytest", "turn-1", minted_at="2026-09-01T00:00:00Z")
    for rid, data in receipts.items():
        m.add(rid, data, "test_report", complete=True, captured_at="2026-09-01T00:00:00Z")
    return m


def one(doc: bytes, **kw):
    core = verify(doc, name="d.md", **kw)
    assert len(core["spans"]) == 1, core["spans"]
    return core["spans"][0], core


# ============================================================================ the lexer

class TestLexer:
    def test_baseline_offsets_are_byte_offsets_into_the_canonical_text(self):
        doc = b"ab " + sp("x").encode() + b" cd\n"
        sc = scan(doc)
        assert sc["lexical_ok"] and sc["document_malformed"] is None
        d, = sc["declarations"]
        assert (d["start"], d["end"], d["receipt"], d["kind"]) == (3, 4, "r1", "numeric")
        assert sc["canonical"] == b"ab x cd\n"
        assert sc["canonical"][d["start"]:d["end"]] == b"x"

    def test_a_multibyte_character_before_the_tag_shifts_the_offset_by_its_byte_length(self):
        doc = "é " .encode() + sp("x").encode()
        d, = scan(doc)["declarations"]
        assert d["start"] == 3            # é is two bytes, then a space

    def test_crlf_is_content_offsets_count_the_cr_and_nothing_is_normalised(self):
        doc = b"a\r\n" + sp("x").encode() + b"\r\n"
        sc = scan(doc)
        d, = sc["declarations"]
        assert d["start"] == 3 and sc["canonical"] == b"a\r\nx\r\n"
        side = to_sidecar(doc, "d.md")
        assert render(side) == doc and "\r\n" in side["text"]

    def test_a_bom_is_three_ordinary_bytes_at_offset_zero(self):
        doc = b"\xef\xbb\xbf" + sp("x").encode()
        d, = scan(doc)["declarations"]
        assert d["start"] == 3

    def test_invalid_utf8_is_document_level_malformed_never_unsworn(self):
        doc = b"\xff\xfe " + sp("x").encode()
        sc = scan(doc)
        assert sc["document_malformed"]["reason"] == "invalid_utf8"
        core = verify(doc, name="d.md")
        assert core["document_verdict"] == "SWORN-FAILED"
        with pytest.raises(SystemExit):
            to_sidecar(doc, "d.md")

    @pytest.mark.parametrize("tag", [
        '<sworn k="numeric" r="r1">',        # attribute order
        "<sworn r='r1' k='numeric'>",        # single quotes
        '<sworn r="r1" k="numeric" x="1">',  # extra attribute
        '<sworn  r="r1" k="numeric">',       # double space
        '<sworn r="r1"  k="numeric">',
        '<sworn r="r1" k="numeric" >',       # space before >
        '<sworn\tr="r1" k="numeric">',
        '<SWORN r="r1" k="numeric">',        # case
        '<sworn r="r1" k="numeric"/>',       # self-closing
        "<sworn>",                           # bare
        '<sworn r="r1"\nk="numeric">',       # split across lines
        '<sworn r=\\"r1\\" k="numeric">',    # backslash is not an escape
    ])
    def test_a_tag_shaped_thing_that_is_not_the_pattern_is_malformed_never_narrative(self, tag):
        doc = (tag + "x</sworn>\n").encode()
        sc = scan(doc)
        reasons = sorted(d["malformed"] for d in sc["declarations"])
        assert "tag_syntax" in reasons and not sc["lexical_ok"]
        core = verify(doc, name="d.md")
        assert core["document_verdict"] == "SWORN-FAILED"
        assert core["counts"]["MALFORMED"] >= 1
        with pytest.raises(SystemExit):
            to_sidecar(doc, "d.md")

    def test_an_entity_escaped_tag_is_narrative_and_the_document_is_unsworn(self):
        core = verify(b"&lt;sworn r=&quot;r1&quot; k=&quot;numeric&quot;&gt;x&lt;/sworn&gt;\n", name="d.md")
        assert core["document_verdict"] == "UNSWORN" and core["sworn_total"] == 0

    def test_swornfoo_is_not_a_candidate(self):
        core = verify(b"<swornfoo> and <sworn-x> are not tags\n", name="d.md")
        assert core["document_verdict"] == "UNSWORN"

    def test_nested_openers_make_both_spans_malformed(self):
        doc = ('<sworn r="r1" k="numeric">a <sworn r="r2" k="numeric">b</sworn> c</sworn>\n').encode()
        sc = scan(doc)
        assert [d["malformed"] for d in sc["declarations"]] == ["nesting", "nesting"]
        assert verify(doc, name="d.md")["counts"]["MALFORMED"] == 2

    def test_a_stray_closer_is_malformed_and_the_good_span_keeps_its_verdict(self):
        doc = (sp("0.55") + " tail </sworn>\n").encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"0.55"))
        verdicts = {(s["verdict"], s["reason"]) for s in core["spans"]}
        assert ("HELD", None) in verdicts and ("MALFORMED", "stray_closer") in verdicts
        with pytest.raises(SystemExit):
            to_sidecar(doc, "d.md")

    def test_an_opener_never_closed_is_malformed(self):
        d, _ = one(b'<sworn r="r1" k="numeric">never closed\n')
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "unclosed")

    @pytest.mark.parametrize("fence", ["```", "````", "```python", " ```", "   ```"])
    def test_a_tag_inside_a_fenced_region_is_literal(self, fence):
        doc = ("%s\n%s\n%s\n" % (fence, sp("x"), fence.strip().rstrip("python") or "```")).encode()
        sc = scan(doc)
        assert sc["declarations"] == [] and sc["candidates"] == 0
        assert verify(doc, name="d.md")["document_verdict"] == "UNSWORN"

    def test_a_fence_at_indent_four_is_not_a_fence_so_the_tag_is_recognised(self):
        doc = ("    ```\n%s\n    ```\n" % sp("x")).encode()
        assert len(scan(doc)["declarations"]) == 1

    def test_a_tilde_fence_is_not_a_fence_under_the_frozen_text(self):
        """Pinned divergence from CommonMark: the spec names backticks at indent 0-3 and nothing else."""
        doc = ("~~~\n%s\n~~~\n" % sp("x")).encode()
        assert len(scan(doc)["declarations"]) == 1

    def test_unbalanced_fences_are_document_level_malformed_and_refuse_to_guess(self):
        doc = ("```\n%s\n" % sp("x")).encode()
        sc = scan(doc)
        assert sc["document_malformed"]["reason"] == "unbalanced_fences"
        assert sc["document_malformed"]["delimiter_lines"] == [1]
        core = verify(doc, name="d.md")
        assert core["document_verdict"] == "SWORN-FAILED" and core["sworn_total"] == 0
        with pytest.raises(SystemExit):
            to_sidecar(doc, "d.md")

    def test_crlf_fences_balance(self):
        doc = ("```\r\n%s\r\n```\r\n" % sp("x")).encode()
        assert scan(doc)["document_malformed"] is None and scan(doc)["declarations"] == []

    def test_a_tag_inside_an_inline_code_span_is_literal(self):
        doc = ("see `%s` here\n" % sp("x")).encode()
        assert scan(doc)["candidates"] == 0

    def test_a_double_backtick_span_hides_a_tag_that_contains_single_backticks(self):
        doc = ("``a ` %s``\n" % sp("x")).encode()
        assert scan(doc)["candidates"] == 0

    def test_an_unmatched_backtick_run_is_literal_and_a_later_tag_is_recognised(self):
        doc = ("`` open %s\n" % sp("x")).encode()
        assert len(scan(doc)["declarations"]) == 1

    def test_a_literal_closer_inside_a_code_span_inside_a_span_belongs_to_the_inner_text(self):
        doc = ('<sworn r="r1" k="quote">the tag `</sworn>` ends it</sworn>\n').encode()
        d, = scan(doc)["declarations"]
        assert d["inner"] == b"the tag `</sworn>` ends it" and d["malformed"] is None

    def test_a_backtick_inside_an_attribute_value_is_consumed_by_the_tag_candidate(self):
        doc = ('<sworn r="r`1" k="numeric">x</sworn>\n').encode()
        core = verify(doc, name="d.md")
        assert core["counts"]["MALFORMED"] == 1

    def test_a_tag_inside_an_html_comment_is_a_hidden_commitment_v02_r2(self):
        """v0.1 recognised it like any other tag and named the hidden commitment as owed. v0.2 R2:
        MALFORMED hidden_commitment — never HELD (it would inflate every count while rendering as
        nothing), never narrative (a broken tag is never narrative), and it fails the document."""
        doc = ("<!-- %s -->\n" % sp("0.55")).encode()
        d, core = one(doc, manifest=manifest(r1=b"0.55"))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "hidden_commitment")
        assert core["document_verdict"] == "SWORN-FAILED" and core["sworn_total"] == 1
        assert "hidden_commitment" in DECISIONS and "html_comments" in DECISIONS
        # the same tag outside the comment is an ordinary span
        d, _ = one(("<!-- note --> %s\n" % sp("0.55")).encode(), manifest=manifest(r1=b"0.55"))
        assert d["verdict"] == "HELD"
        # comments never nest; an unterminated comment runs to the end of the document
        d, _ = one(("<!-- <!-- --> %s\n" % sp("0.55")).encode(), manifest=manifest(r1=b"0.55"))
        assert d["verdict"] == "HELD"
        d, _ = one(("<!-- %s\n" % sp("0.55")).encode(), manifest=manifest(r1=b"0.55"))
        assert d["reason"] == "hidden_commitment"
        # a comment inside a fence or a code span is literal text and suppresses nothing
        d, _ = one(("```\n<!--\n```\n%s\n" % sp("0.55")).encode(), manifest=manifest(r1=b"0.55"))
        assert d["verdict"] == "HELD"
        d, _ = one(("`<!--` %s\n" % sp("0.55")).encode(), manifest=manifest(r1=b"0.55"))
        assert d["verdict"] == "HELD"
        # a hidden span still canonicalises and round-trips: the sidecar is honest about it
        side = to_sidecar(doc, "d.md", C40, manifest(r1=b"0.55"))
        assert render(side) == doc
        assert verify(sidecar=side)["spans"][0]["reason"] == "hidden_commitment"

    def test_an_unclosed_tag_inside_a_comment_is_unclosed_and_refuses_the_sidecar(self):
        doc = b'<!-- <sworn r="r1" k="numeric">0.55\n'
        core = verify(doc, name="d.md", manifest=manifest(r1=b"0.55"))
        assert core["spans"][0]["reason"] == "unclosed" and not scan(doc)["lexical_ok"]
        with pytest.raises(SystemExit):
            to_sidecar(doc, "d.md")

    @pytest.mark.parametrize("inner,expected", [
        ("a" * 300, None),
        ("a" * 301, "length_cap"),
        ("字" * 300, None),                    # 300 code points, 900 bytes: v0.1 would have refused it
        ("字" * 301, "length_cap"),
        ("😀" * 300, None),                    # 300 code points, 1200 bytes
        ("😀" * 300 + "a", "length_cap"),
        ("a" * 299 + "字", None),              # 300 code points, 302 bytes
    ])
    def test_the_cap_is_three_hundred_code_points_never_bytes_v02_r4(self, inner, expected):
        d, _ = one(sp(inner, kind="quote").encode())
        if expected is None:
            assert d["reason"] != "length_cap"
        else:
            assert (d["verdict"], d["reason"]) == ("MALFORMED", expected)
            assert d["detail"]["code_points"] == len(inner) and d["detail"]["cap"] == 300
        assert d["end"] - d["start"] == len(inner.encode("utf-8"))

    def test_an_empty_or_whitespace_only_span_is_malformed_for_every_kind(self):
        for kind in KINDS:
            d, _ = one(sp("", kind=kind).encode())
            assert (d["verdict"], d["reason"]) == ("MALFORMED", "empty_span")
            d, _ = one(sp("  \t", kind=kind).encode())
            assert d["reason"] == "empty_span"

    def test_a_newline_inside_a_span_is_allowed_and_counts_toward_the_cap(self):
        d, _ = one(sp("a\r\nb", kind="quote").encode())
        assert d["end"] - d["start"] == 4


# ============================================================================ canonical form

class TestCanonical:
    def test_the_sidecar_has_the_spec_shape_and_spans_sorted_by_start(self):
        doc = (sp("a", "r2") + " " + sp("b", "r1")).encode()
        side = to_sidecar(doc, "d.md", C40, manifest(r1=b"1", r2=b"2"))
        assert list(side) == ["spec", "commit", "document", "text", "spans", "manifest"]
        assert side["spec"] == SPEC and side["commit"] == C40
        assert [s["receipt"] for s in side["spans"]] == ["r2", "r1"]
        assert side["text"] == "a b"
        assert side["document"]["sha256"] == sworn._sha256(b"a b")
        assert side["manifest"]["spec"] == "sworn/manifest/0.2"

    def test_adjacent_spans_sharing_an_offset_round_trip_byte_for_byte(self):
        doc = (sp("a", "r1") + sp("b", "r2")).encode()
        side = to_sidecar(doc, "d.md")
        assert [(s["start"], s["end"]) for s in side["spans"]] == [(0, 1), (1, 2)]
        assert render(side) == doc

    def test_round_trip_is_asserted_over_a_seeded_fuzz_corpus(self):
        """Fences, code spans, CRLF, unicode, adjacency: whatever the generator makes, if the
        lexer accepts it the sidecar must reproduce it and re-verify to the same spans."""
        import random
        rng = random.Random(20260901)
        atoms = ["a", "é", "字", "😀", " ", "\t", "\n", "\r\n", "\r", "\x00", "\ufeff", "`", "``",
                 "```", "```\n", " ```\n", "    ```\n", "~~~\n", "````\n", "<", ">", "&lt;sworn&gt;",
                 "<!--", "-->", sp("x"), sp("0.5", "r2", "numeric"), sp("`q`", "path:a.json", "quote"),
                 sp("", "r1", "quote"), sp("`</sworn>`", "r1", "quote"), sp("5\n", "r1"), "</sworn>",
                 '<sworn r="r1" k="numeric">', "<sworn", "</swor", '<sworn r="', "<sworn>",
                 '<sworn r="r1" k="numeric" >', '<SWORN r="r1" k="numeric">', "<sworn/>", "<swornx>",
                 "text. More text!\n", "`" + sp("x") + "`", "```" + sp("x") + "\n"]
        m = manifest(r1=b"5", r2=b"0.5")
        accepted = refused = 0
        for _ in range(1500):
            doc = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 16))).encode()
            sc = scan(doc)
            inline = verify(doc, name="f.md", manifest=m)          # never raises on content
            assert inline["sworn_total"] == len(sc["declarations"])
            if (sc["document_malformed"] or not sc["lexical_ok"]
                    or any(d["malformed"] == "empty_span" for d in sc["declarations"])):
                with pytest.raises(SystemExit):
                    to_sidecar(doc, "f.md")
                refused += 1
                continue
            side = to_sidecar(doc, "f.md", manifest=m)
            assert render(side) == doc
            j = json.loads(json.dumps(side, ensure_ascii=False))
            core = verify(sidecar=j)
            assert core["counts"] == inline["counts"], doc
            assert [(x["verdict"], x["reason"]) for x in core["spans"]] == \
                   [(x["verdict"], x["reason"]) for x in inline["spans"]], doc
            accepted += 1
        assert accepted > 200 and refused > 200, (accepted, refused)

    def test_a_sidecar_with_an_unknown_spec_or_extra_keys_is_refused(self):
        side = to_sidecar(sp("x").encode(), "d.md")
        bad = dict(side, spec="sworn/0.2")
        with pytest.raises(SystemExit):
            load_sidecar(bad)
        bad = dict(side, verdict="SWORN-HELD")
        with pytest.raises(SystemExit):
            load_sidecar(bad)

    def test_a_sidecar_whose_spans_overlap_or_carry_a_verdict_is_refused(self):
        side = to_sidecar((sp("ab", "r1") + sp("c", "r2")).encode(), "d.md")
        bad = json.loads(json.dumps(side))
        bad["spans"][1]["start"] = 1
        with pytest.raises(SystemExit):
            load_sidecar(bad)
        bad = json.loads(json.dumps(side))
        bad["spans"][0]["verdict"] = "HELD"
        with pytest.raises(SystemExit):
            load_sidecar(bad)

    def test_a_sidecar_offset_inside_a_multibyte_character_is_refused(self):
        side = to_sidecar(("é" + sp("x")).encode(), "d.md")
        bad = json.loads(json.dumps(side))
        bad["spans"][0]["start"] = 1
        with pytest.raises(SystemExit):
            load_sidecar(bad)

    def test_a_sidecar_whose_text_does_not_hash_to_its_digest_is_refused(self):
        side = to_sidecar(sp("x").encode(), "d.md")
        bad = dict(side, text="y")
        with pytest.raises(SystemExit):
            load_sidecar(bad)

    def test_a_sidecar_span_the_inline_grammar_would_never_recognise_is_refused(self):
        """A sidecar is verified by rendering and re-scanning with the ONE lexer. A span table that
        places a span inside a fenced region renders to literal text, so the re-scan finds no
        span at all and the sidecar is refused rather than trusted."""
        side = to_sidecar(b"```\nx\n```\n", "d.md")
        assert side["spans"] == []
        bad = json.loads(json.dumps(side))
        bad["spans"] = [{"start": 4, "end": 5, "receipt": "r1", "kind": "quote"}]
        with pytest.raises(SystemExit):
            verify(sidecar=bad)

    def test_the_commit_must_be_a_full_lowercase_sha_or_null(self):
        side = to_sidecar(sp("x").encode(), "d.md", C40)
        assert load_sidecar(side)
        with pytest.raises(SystemExit):
            load_sidecar(dict(side, commit="HEAD"))


# ============================================================================ receipts

class TestReceipts:
    @pytest.mark.parametrize("ref", ["r0", "r01", "R1", "r1#", "r1#x", "r1#L0", "r", "1", "path:", "path:/abs",
                                     "path:a/../b", "path:a\\b", "path:a b", "path:a.json#",
                                     "path:a.json#L0", "path:a.json#L5-L3", "path:a.json#x",
                                     "prereg:abc", "prereg:" + "g" * 64, "http://x", ""])
    def test_a_receipt_outside_the_three_forms_is_malformed_not_narrative(self, ref):
        d, _ = one(sp("0.5", ref).encode())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_form"), ref

    def test_no_manifest_means_unresolved_never_an_accusation(self):
        d, core = one(sp("0.5").encode())
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_absent")
        assert core["document_verdict"] == "SWORN-HELD" and core["unresolved"] == 1

    def test_an_id_missing_from_the_manifest_is_unresolved(self):
        d, _ = one(sp("0.5", "r9").encode(), manifest=manifest(r1=b"0.5"))
        assert d["reason"] == "manifest_id_missing"

    def test_a_receipt_the_agent_authored_is_malformed_invariant_two(self):
        m = Manifest("h", "t")
        m.add("r1", b"0.5", "agent_output", complete=True)
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_author_minted")

    def test_invariant_two_is_set_membership_over_what_the_harness_saw_the_agent_write(self):
        """The agent writes results.json; the harness then captures it as a file_read receipt.
        The kind looks external, but the bytes are in authored_sha256, so it is refused."""
        m = Manifest("h", "t")
        m.record_authored(b"0.5")
        m.add("r1", b"0.5", "file_read", complete=True)
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_author_minted")
        assert "authored_sha256" in m.to_dict()

    def test_completeness_missing_is_the_harness_not_saying_so_unresolved(self):
        m = manifest(r1=b"record with no erratum word")
        del m.receipts["r1"]["complete"]
        d, _ = one(sp("carries no `erratum`", kind="absent").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_no_completeness")

    def test_a_kind_of_source_outside_the_closed_vocabulary_is_malformed(self):
        m = Manifest("h", "t")
        m.add("r1", b"0.5", "oracle", complete=True)
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert d["reason"] == "kind_of_source_unknown"

    def test_manifest_bytes_that_do_not_hash_to_their_sha256_are_unresolved(self):
        m = manifest(r1=b"0.5")
        m.receipts["r1"]["sha256"] = "0" * 64
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_integrity")

    def test_an_entry_without_bytes_serves_hash_only(self):
        m = Manifest("h", "t")
        m.add("r1", None, "file_read", complete=True, sha256="ab" * 32)
        d, _ = one(sp("sha " + "ab" * 32, kind="hash").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_bytes_absent")

    def test_the_manifest_digest_is_content_addressed(self):
        a, b = manifest(r1=b"0.5"), manifest(r1=b"0.5")
        assert a.digest() == b.digest()
        b.receipts["r1"]["complete"] = False
        assert a.digest() != b.digest()

    def test_path_receipts_resolve_at_the_named_commit_and_never_without_one(self):
        tree = MemoryTree({"a.json": b'{"x": 0.5}'}, commit=None)
        d, _ = one(sp("0.5", "path:a.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "no_commit")
        tree = MemoryTree({"a.json": b'{"x": 0.5}'}, commit=C40)
        d, _ = one(sp("0.5", "path:a.json#/x").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("0.5", "path:missing.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "path_absent")
        d, _ = one(sp("0.5", "path:a.json#/x").encode())
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "no_repository")

    def test_json_pointer_escapes_indices_and_dash(self):
        tree = MemoryTree({"a.json": b'{"a/b": {"m~n": [10, 20]}}'}, commit=C40)
        d, _ = one(sp("20", "path:a.json#/a~1b/m~0n/1").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("20", "path:a.json#/a~1b/m~0n/-").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "pointer_unresolvable")
        d, _ = one(sp("20", "path:a.json#/nope").encode(), tree=tree)
        assert d["reason"] == "pointer_unresolvable"

    def test_bytes_the_author_could_read_but_pointed_at_wrongly_are_malformed_not_unresolved(self):
        """UNRESOLVED is for evidence the verifier could not obtain. Bytes that were obtained but
        are not JSON, or a pointer path through a duplicated key, are the author's declaration."""
        tree = MemoryTree({"a.json": b"not json"}, commit=C40)
        d, _ = one(sp("1", "path:a.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_not_json")
        tree = MemoryTree({"a.json": b'{"x": 1, "x": 2, "y": 3}'}, commit=C40)
        d, _ = one(sp("1", "path:a.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "pointer_ambiguous")
        d, _ = one(sp("3", "path:a.json#/y").encode(), tree=tree)
        assert d["verdict"] == "HELD", "a duplicate OFF the pointer path is irrelevant"
        tree = MemoryTree({"a.json": b'{"x": NaN, "y": 0.5}'}, commit=C40)
        d, _ = one(sp("1", "path:a.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_numeric")
        d, _ = one(sp("0.5", "path:a.json#/y").encode(), tree=tree)
        assert d["verdict"] == "HELD", "one non-finite leaf does not hide the rest of the file"

    def test_line_anchors_are_one_based_lf_split_and_keep_their_terminator(self):
        tree = MemoryTree({"f.md": b"alpha\r\nbeta\ngamma"}, commit=C40)
        d, _ = one(sp("`beta`", "path:f.md#L2", "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("`alpha`", "path:f.md#L2", "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("FAILED", "needle_missing")
        d, _ = one(sp("`beta`", "path:f.md#L1-L2", "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        # a needle cannot span lines: code spans are line-local for the lexer and for the needle
        # alike (one rule, pinned), so a multi-line quote is MALFORMED in v0.1 rather than guessed
        d, _ = one(sp("`alpha\r\nbeta`", "path:f.md#L1-L2", "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "needle_count")
        d, _ = one(sp("`gamma`", "path:f.md#L4", "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "anchor_out_of_range")

    def test_prereg_receipts_are_content_addressed(self):
        bar = b"# PREREG\nthe floor is 0.95 and frozen\n"
        tree = MemoryTree({"papers/PREREG_x.md": bar, "other.md": b"x"}, commit=C40)
        digest = sworn._sha256(bar)
        d, _ = one(sp("`the floor is 0.95 and frozen`", "prereg:" + digest.upper(), "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("`the floor is 0.95 and frozen`", "prereg:" + "0" * 64, "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "prereg_not_in_tree")


def _git(repo, *args, env=None):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
             GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=e,
                          check=True).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "res.json").write_bytes(b'{"recall": 0.91, "name": "v1"}')
    (repo / "PREREG_bar.md").write_bytes(b"bar 0.95\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    sha = _git(repo, "rev-parse", "HEAD")
    # the working tree moves on; the verdict must not
    (repo / "res.json").write_bytes(b'{"recall": 0.10, "name": "v2"}')
    return repo, sha


class TestGitTree:
    def test_a_path_receipt_reads_the_commit_not_the_working_tree(self, git_repo):
        repo, sha = git_repo
        d, _ = one(sp("0.91", "path:res.json#/recall").encode(), tree=GitTree(repo, sha))
        assert d["verdict"] == "HELD", d
        d, _ = one(sp("0.10", "path:res.json#/recall").encode(), tree=GitTree(repo, sha))
        assert (d["verdict"], d["reason"]) == ("FAILED", "value_mismatch")

    def test_a_commit_absent_from_the_checkout_is_unresolved(self, git_repo):
        repo, _ = git_repo
        d, _ = one(sp("0.91", "path:res.json#/recall").encode(), tree=GitTree(repo, "b" * 40))
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "commit_absent")

    def test_a_prereg_digest_is_found_in_the_tree_at_the_commit(self, git_repo):
        repo, sha = git_repo
        digest = sworn._sha256(b"bar 0.95\n")
        # a nine-byte receipt cannot hold a sixteen-byte needle; the author narrows the haystack
        # with a line anchor and the short needle is then exempt from R3
        d, _ = one(sp("the bar `bar 0.95`", "prereg:" + digest + "#L1", "quote").encode(), tree=GitTree(repo, sha))
        assert d["verdict"] == "HELD"

    def test_a_directory_is_not_a_blob(self, git_repo, tmp_path):
        repo, sha = git_repo
        (repo / "dir").mkdir()
        (repo / "dir" / "f").write_text("x")
        _git(repo, "add", "dir")
        _git(repo, "commit", "-q", "-m", "two")
        sha2 = _git(repo, "rev-parse", "HEAD")
        d, _ = one(sp("`x`", "path:dir", "quote").encode(), tree=GitTree(repo, sha2))
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "not_a_blob")

    def test_git_unavailable_or_no_repository_is_unresolved(self, tmp_path):
        d, _ = one(sp("1", "path:x.json").encode(), tree=GitTree(tmp_path / "nowhere", "c" * 40))
        assert d["verdict"] == "UNRESOLVED"


# ============================================================================ the kinds

def numeric(span_text, receipt_bytes):
    d, _ = one(sp(span_text).encode(), manifest=manifest(r1=receipt_bytes))
    return d["verdict"], d["reason"]


class TestNumeric:
    def test_a_number_followed_by_a_sentence_period_is_still_the_number(self):
        """The OATH _NUM defect, refused here by construction."""
        assert numeric("a precision of 0.55.", b"0.55") == ("HELD", None)

    def test_percent_is_not_converted(self):
        assert numeric("42% of them", b"0.42") == ("FAILED", "value_mismatch")
        assert numeric("42% of them", b"42") == ("HELD", None)

    def test_thousands_separators_are_one_number(self):
        assert numeric("all 23,247 rows.", b"23247") == ("HELD", None)

    @pytest.mark.parametrize("text", ["precision 0.55 on n=38", "0.5 and 0.5", "0.55 ± 0.02 wide",
                                      "0.4211 (16/38)"])
    def test_more_than_one_digit_bearing_token_is_malformed(self, text):
        assert numeric(text, b"0.5") == ("MALFORMED", "number_count")

    def test_no_digit_bearing_token_is_malformed(self):
        assert numeric("no digits here", b"0.5") == ("MALFORMED", "number_count")

    @pytest.mark.parametrize("text", ["r1 holds", "v0.1 shipped", "L13", "38px", "sha 3d488b6",
                                      "STRUCT-1 scored", "GPT-4 did", "2026-09-01", "3/4 of them",
                                      "1,2", "identical to 1.36e-14 on", "0.55-0.60", "12:30",
                                      "n_38", "٣.٥", "0.5²", "1_000", "0,55", "%42", "42%%"])
    def test_a_digit_bearing_token_that_is_not_one_number_is_malformed_never_ignored(self, text):
        """No identifier whitelist, no date/sha/version scrub: deciding that `STRUCT-1` is a label
        is the vocabulary decision the verifier must never make. The author moves it out of the
        tag; the verifier never guesses."""
        assert numeric(text, b"0.5") == ("MALFORMED", "number_grammar"), text

    @pytest.mark.parametrize("text,tok", [
        (".5 of them", ".5"), ("n=38", "38"), ("p < 0.05", "0.05"), ("(0.55)", "0.55"),
        ("`0.55`", "0.55"), ("≈0.55", "0.55"), ("0.55.", "0.55"), ("0.55...", "0.55"),
        ("23,247.", "23,247"), ("5,", "5"), ("0.", "0"), ("42 %", "42"), ("−0.05", "−0.05"),
        ("+3", "+3"), ("007", "007"), ('"0.55"', "0.55"), ("0.55;", "0.55"),
    ])
    def test_the_one_number_token_is_found_after_stripping_sentence_punctuation(self, text, tok):
        why, found, _ = sworn._number_token(text)
        assert (why, found) == (None, tok), text

    def test_rounding_is_half_even_on_the_decimal_literal_never_a_float(self):
        assert numeric("about 0.58", b"0.585") == ("HELD", None)
        assert numeric("about 0.59", b"0.585") == ("FAILED", "value_mismatch")
        assert numeric("about 0.6", b"0.55") == ("HELD", None)          # 0.55 -> 0.6 (half-even on 5? 0.55 -> 0.6)
        assert numeric("about 40", b"39.5") == ("HELD", None)
        assert numeric("about 39", b"39.5") == ("FAILED", "value_mismatch")
        assert DECISIONS["rounding"].startswith("the receipt scalar is a Decimal")

    def test_the_printed_precision_sets_the_comparison(self):
        assert numeric("0.500", b"0.5") == ("HELD", None)
        assert numeric("38.0", b"38") == ("HELD", None)
        assert numeric("38", b"38.4") == ("HELD", None)
        assert numeric("0.5811", b"0.58113") == ("HELD", None)
        assert numeric("0.5812", b"0.58113") == ("FAILED", "value_mismatch")

    def test_sign_and_leading_zero_normalisation(self):
        assert numeric("+3 more", b"3") == ("HELD", None)
        assert numeric("delta −0.05", b"-0.05") == ("HELD", None)
        assert numeric("-0.0 exactly", b"0") == ("HELD", None)

    def test_a_string_leaf_is_never_parsed_as_a_number(self):
        """`"0.55"` spelling a number is a guess about what the string means; not taken."""
        assert numeric("0.55", b'"0.55"') == ("MALFORMED", "leaf_not_numeric")
        assert numeric("1", b"true") == ("MALFORMED", "leaf_not_numeric")

    @pytest.mark.parametrize("leaf,reason", [(b"null", "leaf_not_scalar"), (b"[0.5]", "leaf_not_scalar"),
                                             (b'{"x": 0.5}', "leaf_not_scalar"), (b"Infinity", "leaf_not_numeric"),
                                             (b"1e400", "leaf_not_numeric")])
    def test_a_non_number_leaf_is_malformed(self, leaf, reason):
        assert numeric("0.5", leaf) == ("MALFORMED", reason)

    def test_a_line_anchored_slice_must_be_one_bare_number(self):
        tree = MemoryTree({"f.txt": b"0.91\nnoise\n"}, commit=C40)
        d, _ = one(sp("0.91", "path:f.txt#L1").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("0.91", "path:f.txt#L1-L2").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_not_json")

    def test_rounding_ties_agree_with_how_this_repo_prints_and_the_divergence_is_disclosed(self):
        """Python formats the exact binary value: dyadic ties round half-even and Decimal HALF_EVEN
        reproduces every one. On a non-dyadic tie (2.675) Python's binary artefact says 2.67 and
        the decimal rule says 2.68 — a paper that printed 2.67 from a receipt holding 2.675 FAILS,
        and the remedy is to print the receipt's digits."""
        for leaf, frac, printed in ((0.125, 2, "0.12"), (0.375, 2, "0.38"), (0.5, 0, "0"), (1.5, 0, "2"),
                                    (2.5, 0, "2")):
            assert printed == "%.*f" % (frac, leaf), "premise: this is how Python prints the tie"
            assert numeric(printed, str(leaf).encode()) == ("HELD", None)
        assert "%.2f" % 2.675 == "2.67"
        assert numeric("2.68", b"2.675") == ("HELD", None)
        assert numeric("2.67", b"2.675") == ("FAILED", "value_mismatch")

    def test_binary_float_artefacts_never_enter_the_comparison(self):
        assert numeric("0.3", b"0.30000000000000004") == ("HELD", None)
        assert numeric("0.30000000000000004", b"0.30000000000000004") == ("HELD", None)
        assert numeric("0.30000000000000000", b"0.30000000000000004") == ("FAILED", "value_mismatch")
        assert numeric("1.0", b"0.9999999999999999") == ("HELD", None)
        assert numeric("1.0000000000000000", b"0.9999999999999999") == ("FAILED", "value_mismatch")
        assert numeric("100", b"1E+2") == ("HELD", None)

    def test_the_canonical_string_is_a_pure_function_of_digits_seeded_property_sweep(self):
        import random
        rng = random.Random(20260901)
        for _ in range(3000):
            digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 18)))
            frac_in = rng.randint(0, 12)
            sign = rng.choice(["", "-", "+"])
            txt = sign + (digits[:-frac_in] or "0") + ("." + digits[-frac_in:] if frac_in else "")
            x = Decimal(txt)
            d = rng.randint(0, 8)
            c = sworn._canon(x, d)
            assert sworn._canon(Decimal(c), d) == c                       # idempotent
            assert sworn._canon(x.scaleb(3).scaleb(-3), d) == c           # textual form irrelevant
            assert not c.startswith("-0") or Decimal(c) != 0             # signed zero folded

    def test_no_float_touches_the_numeric_path(self):
        src = Path(sworn.__file__).read_text(encoding="utf-8")
        body = src.split("# 4. the kinds")[1].split("# 5. the document")[0]
        for forbidden in ("float(", "isclose", "round("):
            assert forbidden not in body, forbidden

    def test_no_search_over_leaves_the_author_named_the_leaf(self):
        """The vacuous pass cannot occur: a value that exists elsewhere in the receipt is not found."""
        tree = MemoryTree({"a.json": b'{"x": 0.1, "y": 0.5}'}, commit=C40)
        d, _ = one(sp("0.5", "path:a.json#/x").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("FAILED", "value_mismatch")


class TestQuoteHashAbsent:
    def test_quote_needle_is_the_one_backtick_span_bytes_verbatim(self):
        m = manifest(r1=b'{"note": "the  erratum is attached"}')
        d, _ = one(sp("says `the  erratum is attached`", kind="quote").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("says `the erratum is attached`", kind="quote").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("FAILED", "needle_missing")

    @pytest.mark.parametrize("text,reason", [("no needle here", "needle_count"),
                                             ("`one` and `two`", "needle_count"),
                                             ("`` empty", "needle_count"),
                                             ("` `x", "needle_count")])
    def test_quote_without_exactly_one_needle_is_malformed(self, text, reason):
        d, _ = one(sp(text, kind="quote").encode(), manifest=manifest(r1=b"x"))
        assert d["verdict"] == "MALFORMED", (text, d)

    def test_quote_does_no_unicode_normalisation(self):
        nfc, nfd = "café résumé naïve".encode("utf-8"), "café résumé naïve".encode("utf-8")
        d, _ = one(('<sworn r="r1" k="quote">the name `%s`</sworn>' % nfc.decode()).encode(),
                   manifest=manifest(r1=nfd))
        assert (d["verdict"], d["reason"]) == ("FAILED", "needle_missing")

    def test_a_short_quote_needle_over_a_whole_receipt_is_malformed_v02_r3(self):
        """A10: a one-byte needle HELDs against almost any receipt. The attack is the haystack
        size, so a haystack the author narrowed — a pointer leaf, a line anchor — is exempt, and
        so is `absent`, whose short needle is the stronger oath."""
        big = b"x" * 5000 + b"passed" + b"y" * 100
        d, _ = one(sp("it says `passed`", kind="quote").encode(), manifest=manifest(r1=big))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "short_needle")
        assert d["detail"] == {"needle_bytes": 6, "minimum_bytes": 16}
        d, _ = one(sp("it says `passed` and it does", kind="absent").encode(), manifest=manifest(r1=b"nothing here"))
        assert d["verdict"] == "HELD"
        tree = MemoryTree({"a.json": b'{"s": "passed", "n": 3}', "f.md": b"alpha\npassed\n"}, commit=C40)
        d, _ = one(sp("`passed`", "path:a.json#/s", "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("`passed`", "path:f.md#L2", "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("`passed`", "path:f.md", "quote").encode(), tree=tree)
        assert d["reason"] == "short_needle"
        d, _ = one(sp("`the sixteen-byte needle`", kind="quote").encode(), manifest=manifest(r1=big + b"the sixteen-byte needle"))
        assert d["verdict"] == "HELD" and d["detail"]["occurrences"] == 1

    def test_quote_against_a_pointer_needs_a_string_leaf(self):
        tree = MemoryTree({"a.json": b'{"s": "hello world", "n": 3}'}, commit=C40)
        d, _ = one(sp("`lo wo`", "path:a.json#/s", "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("`3`", "path:a.json#/n", "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_string")

    def test_hash_is_exactly_one_sixty_four_hex_run_case_insensitive(self):
        data = b"bytes"
        h = sworn._sha256(data)
        m = manifest(r1=data)
        d, _ = one(sp("digest " + h.upper(), kind="hash").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("digest " + "0" * 64, kind="hash").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("FAILED", "digest_mismatch")
        d, _ = one(sp("git " + "a" * 40, kind="hash").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "digest_form")
        d, _ = one(sp(h + " and " + h, kind="hash").encode(), manifest=m)
        assert d["reason"] == "digest_form"

    def test_hash_over_a_partial_receipt_is_malformed(self):
        tree = MemoryTree({"a.json": b"{}"}, commit=C40)
        d, _ = one(sp("a" * 64, "path:a.json#/x", "hash").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "hash_over_partial")

    def test_absent_swears_a_negative_only_over_a_complete_object(self):
        m = manifest(r1=b'{"deposit": "v2", "notes": []}')
        d, _ = one(sp("carries no `erratum`", kind="absent").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("carries no `deposit`", kind="absent").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("FAILED", "needle_present")
        m.receipts["r1"]["complete"] = False
        d, _ = one(sp("carries no `erratum`", kind="absent").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "absent_over_partial")

    def test_absent_over_a_pointer_or_anchor_is_partial_and_malformed(self):
        tree = MemoryTree({"a.json": b'{"x": 1}'}, commit=C40)
        for ref in ("path:a.json#/x", "path:a.json#L1"):
            d, _ = one(sp("no `y`", ref, "absent").encode(), tree=tree)
            assert (d["verdict"], d["reason"]) == ("MALFORMED", "absent_over_partial")
        d, _ = one(sp("no `y`", "path:a.json", "absent").encode(), tree=tree)
        assert d["verdict"] == "HELD"

    def test_exec_is_reserved_and_unknown_kinds_are_malformed(self):
        d, _ = one(sp("x", kind="exec").encode())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "kind_reserved")
        d, _ = one(sp("x", kind="WITHHELD").encode())
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "kind_unknown")
        d, _ = one(sp("x", kind="Numeric").encode())
        assert d["reason"] == "kind_unknown"


# ============================================================================ the document

class TestDocument:
    def test_a_document_that_swore_nothing_is_unsworn_never_no_failures(self):
        core = verify(b"Plain prose. 0.998 AUC, nothing sworn.\n", name="d.md")
        assert core["document_verdict"] == "UNSWORN"
        assert core["counts"] == {"HELD": 0, "FAILED": 0, "UNRESOLVED": 0, "MALFORMED": 0, "WITHHELD": 0}

    def test_sworn_held_iff_no_failed_no_malformed_and_at_least_one_span(self):
        core = verify((sp("0.5", "r1") + " " + sp("`ok, all tests passed`", "r2", "quote")).encode(),
                      name="d.md", manifest=manifest(r1=b"0.5", r2=b"ok, all tests passed"))
        assert core["document_verdict"] == "SWORN-HELD" and core["sworn_total"] == 2

    def test_one_failed_span_fails_the_document(self):
        core = verify((sp("0.5", "r1") + " " + sp("0.6", "r2")).encode(), name="d.md",
                      manifest=manifest(r1=b"0.5", r2=b"0.5"))
        assert core["document_verdict"] == "SWORN-FAILED"

    def test_unresolved_only_is_held_and_the_count_travels_in_the_headline(self):
        core = verify(sp("0.5").encode(), name="d.md")
        assert core["document_verdict"] == "SWORN-HELD" and core["unresolved"] == 1
        head = sworn._headline(core)
        assert head.startswith("SWORN-HELD") and "unresolved=1" in head and "coverage" in head

    def test_a_malformed_declaration_counts_against_the_author(self):
        core = verify(b'<sworn r="r1" k="numeric">1</sworn> <sworn>\n', name="d.md",
                      manifest=manifest(r1=b"1"))
        assert core["document_verdict"] == "SWORN-FAILED"
        assert core["sworn_total"] == 2 and core["counts"]["MALFORMED"] == 1

    def test_withheld_has_no_producer(self):
        core = verify((sp("0.5") + sp("0.6", "r2")).encode(), name="d.md")
        assert core["counts"]["WITHHELD"] == 0 and "WITHHELD" in VERDICTS

    def test_every_reason_the_verifier_emits_is_in_the_closed_set(self):
        import re as _re
        src = Path(sworn.__file__).read_text(encoding="utf-8")
        emitted = set(_re.findall(r'"([a-z_]+)"\)', src)) & set(REASONS) | set(
            m for m in _re.findall(r'(?:reason=|out\("(?:MALFORMED|UNRESOLVED|FAILED)", )"([a-z_]+)"', src))
        unknown = {m for m in _re.findall(r'out\("(?:MALFORMED|UNRESOLVED|FAILED)", "([a-z_]+)"', src)} - set(REASONS)
        assert not unknown, unknown
        assert emitted


class TestCoverage:
    def test_coverage_is_always_advisory_and_the_estimate_is_gone_v02_r8(self):
        core = verify(b"Nothing claim-shaped, nothing sworn.\n", name="d.md")
        cov = core["coverage"]
        assert cov["schema"] == "sworn/coverage/1" and cov["advisory"] is True
        assert "estimate" not in cov and "unsworn_claims_estimate" not in cov
        assert "0.4211" in cov["ceiling_note"] and "result-shaped" in cov["ceiling_note"]
        # nothing sworn, one narrative sentence: the floor is 0/1, not undefined and not one
        assert cov["sworn_total"] == 0 and cov["narrative_sentences"] == 1
        assert cov["sentence_share"] == 0.0 and cov["diff_claim_sentences"] == 0

    def test_the_floor_counts_every_narrative_sentence_and_the_diff_count_is_labelled(self):
        """The v0.1 estimate printed ~1.0 beside result-shaped documents because STRUCT-1 never
        reads a measured rate as a claim. The floor cannot flatter: every narrative sentence
        counts, whatever it says."""
        doc = (sp("0.5") + " held. Precision was 0.16 on 100 items. Every accusation upheld. "
               "Added tests/test_x.py and rewrote styxx/certify.py.\n").encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"0.5"))
        cov = core["coverage"]
        assert cov["narrative_sentences"] == 4                    # 'held.' is a sentence too
        assert cov["sentence_share"] == round(1 / 5, 4)
        assert cov["diff_claim_sentences"] == 1 and cov["diff_claim_share"] == 0.5
        assert "pull-request prose" in cov["diff_claim_idiom"]
        claim, = cov["unsworn_claims"]
        canonical = scan(doc)["canonical"]
        assert canonical[claim["start"]:claim["end"]].decode().startswith("Added tests/test_x.py")
        assert core["document_verdict"] == "SWORN-HELD"          # coverage never gates

    def test_a_quoted_number_in_narrative_gets_no_verdict_mention_versus_use(self):
        doc = (sp("0.5") + ' The paper says "9 new tests." and that is narrative.\n').encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"0.5"))
        assert core["sworn_total"] == 1 and core["counts"]["HELD"] == 1

    def test_fenced_code_is_not_counted_in_the_denominator(self):
        doc = (sp("0.5") + "\n```\nAdded tests/test_x.py\n```\n").encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"0.5"))
        assert core["coverage"]["diff_claim_sentences"] == 0
        assert core["coverage"]["narrative_sentences"] == 0 and core["coverage"]["sentence_share"] == 1.0


# ============================================================================ the receipt

class TestVerdictReceipt:
    def test_the_receipt_is_content_addressed_and_re_derivable(self):
        doc = sp("0.5").encode()
        m = manifest(r1=b"0.5")
        rec = issue_receipt(verify(doc, name="d.md", manifest=m), timestamp="2026-09-01T00:00:00Z")
        assert rec["schema"] == "styxx.sworn.verdict-receipt/v1"
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["coverage_reproduces"] is True
        # v0.2 R9: coverage is outside the digest and carries its own hash
        assert rec["coverage_sha256"] == sworn._sha256(sworn._jcs(rec["coverage"]).encode())
        stripped = {k: v for k, v in rec.items() if k not in ("digest", "timestamp", "coverage", "coverage_sha256")}
        assert rec["digest"] == sworn._sha256(sworn._jcs(stripped).encode())
        again = issue_receipt(verify(doc, name="d.md", manifest=m), timestamp="2030-01-01T00:00:00Z")
        assert again["digest"] == rec["digest"], "the timestamp is outside the digest"

    def test_a_tampered_verdict_or_document_fails_re_derivation(self):
        doc = sp("0.5").encode()
        m = manifest(r1=b"0.6")
        rec = issue_receipt(verify(doc, name="d.md", manifest=m))
        assert rec["document_verdict"] == "SWORN-FAILED"
        forged = json.loads(json.dumps(rec))
        forged["document_verdict"] = "SWORN-HELD"
        forged["spans"][0]["verdict"] = "HELD"
        res = verify_receipt(forged, doc, manifest=m)
        assert res["status"] == "FAILED" and not res["digest_match"]
        res = verify_receipt(rec, doc + b"!", manifest=m)
        assert res["status"] == "FAILED" and not res["verdict_reproduces"]

    def test_the_receipt_states_its_boundary_and_its_decisions(self):
        core = verify(sp("0.5").encode(), name="d.md")
        assert "NOT a claim that the document is correct" in core["certifies"]
        assert "right sentences were bound" in core["certifies"]
        assert core["verifier"]["decisions"] == DECISIONS
        assert core["verifier"]["rounding"] == "ROUND_HALF_EVEN"

    def test_a_receipt_from_another_verifier_build_is_reported_not_hidden(self):
        doc = sp("0.5").encode()
        m = manifest(r1=b"0.5")
        rec = issue_receipt(verify(doc, name="d.md", manifest=m))
        rec["verifier"]["sworn_sha256"] = "0" * 64
        rec = issue_receipt(rec)
        res = verify_receipt(rec, doc, manifest=m)
        assert res["status"] == "VERIFIED" and res["same_verifier_build"] is False


# ============================================================================ invariants

class TestInvariants:
    def test_invariant_one_no_function_proposes_tags(self):
        names = [n for n in dir(sworn) if not n.startswith("_")]
        assert not any(k in n.lower() for n in names for k in ("propose", "suggest", "tag_text", "autotag"))
        # nothing here accepts plain text and returns spans
        assert scan(b"plain prose with 0.5 in it")["declarations"] == []

    def test_invariant_two_is_mechanically_refused(self):
        m = Manifest("h", "t")
        m.add("r1", b"0.5", "agent_message", complete=True)
        core = verify(sp("0.5").encode(), name="d.md", manifest=m)
        assert core["counts"]["MALFORMED"] == 1 and core["document_verdict"] == "SWORN-FAILED"

    def test_invariant_three_silence_buys_no_badge(self):
        assert verify(b"", name="d.md")["document_verdict"] == "UNSWORN"
        assert verify(b"```\ncode\n```\n", name="d.md")["document_verdict"] == "UNSWORN"

    def test_invariant_four_coverage_travels_with_every_verdict(self):
        core = verify(sp("`v`", "r1", "quote").encode() + b" Rewrote styxx/x.py. Deleted tests/y.py.",
                      name="d.md", manifest=manifest(r1=b"v"))
        head = sworn._headline(core)
        assert "coverage-floor≈0.33" in head and "diff-claims≈2" in head and "rungs" in head

    def test_receipt_shopping_moves_oath_but_cannot_move_sworn(self, tmp_path):
        """The charon RECON attack: an OATH verdict on fixed bytes moves with the receipt set the
        author supplies, because the verifier value-matches over every leaf. Sworn cannot move:
        the author named the leaf, so a larger pool has nothing to offer."""
        from styxx.certify import certify_doc
        doc = tmp_path / "d.md"
        doc.write_text("the recall was 0.55\n", encoding="utf-8")
        small = tmp_path / "a_result.json"
        small.write_text(json.dumps({"x": 0.1}), encoding="utf-8")
        big = tmp_path / "b_result.json"
        big.write_text(json.dumps({"z": {"score": 0.55}}), encoding="utf-8")
        assert certify_doc(doc, [small])["verdict"].startswith("OATH-FAILED")
        assert certify_doc(doc, [small, big])["verdict"].startswith("OATH-HELD"), "OATH bought HELD"

        sdoc = ('<sworn r="path:a_result.json#/x" k="numeric">the recall was 0.55</sworn>\n').encode()
        pool_small = MemoryTree({"a_result.json": small.read_bytes()}, commit=C40)
        pool_big = MemoryTree({"a_result.json": small.read_bytes(), "b_result.json": big.read_bytes()},
                              commit=C40)
        assert verify(sdoc, name="d.md", tree=pool_small)["document_verdict"] == "SWORN-FAILED"
        assert verify(sdoc, name="d.md", tree=pool_big)["document_verdict"] == "SWORN-FAILED"


# ============================================================================ the CLI

class TestCLI:
    def test_canon_render_verify_check_round_trip(self, tmp_path, capsys):
        doc = tmp_path / "d.md"
        doc.write_bytes((sp("0.55") + " narrative.\n").encode())
        mpath = tmp_path / "m.json"
        assert sworn.main(["manifest", "new", str(mpath), "--harness", "pytest", "--turn", "1",
                           "--rung", "L1"]) == 0
        r1 = tmp_path / "r1.txt"
        r1.write_bytes(b"0.55")
        assert sworn.main(["manifest", "add", str(mpath), "--id", "r1", "--file", str(r1),
                           "--kind", "tool_stdout", "--complete"]) == 0
        side = tmp_path / "d.sworn.json"
        assert sworn.main(["canon", str(doc), "--manifest", str(mpath), "--out", str(side)]) == 0
        back = tmp_path / "back.md"
        assert sworn.main(["render", str(side), "--out", str(back)]) == 0
        assert back.read_bytes() == doc.read_bytes()
        rec = tmp_path / "rec.json"
        assert sworn.main(["verify", str(side), "--out", str(rec)]) == 0
        out = capsys.readouterr().out
        assert "SWORN-HELD  held=1 failed=0 unresolved=0 malformed=0" in out
        assert sworn.main(["check", str(rec), str(side)]) == 0

    def test_exit_code_is_zero_for_every_verdict_and_two_for_a_refusal(self, tmp_path, capsys):
        doc = tmp_path / "d.md"
        doc.write_bytes(sp("0.6").encode())
        assert sworn.main(["verify", str(doc)]) == 0                         # UNRESOLVED -> still 0
        doc.write_bytes(b"<sworn>broken\n")
        assert sworn.main(["verify", str(doc)]) == 0                         # SWORN-FAILED -> still 0
        with pytest.raises(SystemExit):
            sworn.main(["canon", str(doc)])                                 # refusal
        assert "SWORN-FAILED" in capsys.readouterr().out

    def test_a_sworn_verdict_and_a_parrhesia_verdict_never_share_a_line(self):
        src = Path(sworn.__file__).read_text(encoding="utf-8")
        assert "parrhesia" not in src.split("def _headline")[1].split("def _load_tree")[0]


# ============================================================================ doctrine

class TestDoctrine:
    def test_the_three_spec_strings_are_pinned_verbatim(self):
        assert sworn.SPEC == "sworn/0.1"
        assert sworn.MANIFEST_SPEC == "sworn/manifest/0.2"
        assert sworn.RECEIPT_SCHEMA == "styxx.sworn.verdict-receipt/v1"
        # the old strings are still LOADED and never EMITTED (M7)
        assert sworn.MANIFEST_SPECS == ("sworn/manifest/0.1", "sworn/manifest/0.2")
        assert sworn.RECEIPT_SCHEMAS == ("styxx.sworn.verdict-receipt/v0", "styxx.sworn.verdict-receipt/v1")

    def test_sworn_is_not_on_the_package_surface_and_does_not_shadow_parrhesia(self):
        import styxx
        assert "sworn" not in styxx.__all__
        assert styxx.issue_receipt is styxx.parrhesia.issue_receipt
        assert styxx.verify_receipt is styxx.parrhesia.verify_receipt
        code = ("import sys, styxx; print('sworn' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8")
        assert out.stdout.strip() == "False", "importing styxx must not import sworn"

    def test_a_manifest_with_the_wrong_spec_string_is_refused_not_coerced(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"spec": SPEC, "receipts": {}}), encoding="utf-8")
        with pytest.raises(SystemExit, match="REFUSED: unknown manifest spec"):
            Manifest.load(p)

    def test_a_tampered_manifest_makes_every_rn_span_unresolved_and_accuses_nobody(self):
        m = manifest(r1=b"0.5", r2=b"0.6")
        d = m.to_dict()
        d["receipts"]["r2"]["complete"] = False              # edited after the harness signed
        tampered = Manifest.from_dict(d)
        core = verify((sp("0.5", "r1") + sp("0.6", "r2")).encode(), name="d.md", manifest=tampered)
        assert [(x["verdict"], x["reason"]) for x in core["spans"]] == [
            ("UNRESOLVED", "manifest_integrity")] * 2
        assert core["document_verdict"] == "SWORN-HELD" and core["unresolved"] == 2

    @pytest.mark.parametrize("kind", sorted(sworn.SOURCE_KINDS_EXTERNAL))
    def test_every_external_source_kind_resolves(self, kind):
        m = Manifest("h", "t")
        m.add("r1", b"0.5", kind, complete=True)
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert d["verdict"] == "HELD"

    @pytest.mark.parametrize("kind", sorted(sworn.SOURCE_KINDS_AUTHOR))
    def test_every_author_side_source_kind_is_malformed(self, kind):
        m = Manifest("h", "t")
        m.add("r1", b"0.5", kind, complete=True)
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_author_minted")

    def test_a_receipt_core_is_jcs_serialisable_on_every_verdict_path(self):
        tree = MemoryTree({"a.json": b'{"x": 0.5, "s": "hello", "o": {}}', "t.txt": b"hello"}, commit=C40)
        m = manifest(r1=b"0.5", r2=b"bytes")
        doc = "".join([
            sp("0.5", "r1"), sp("0.6", "r1"), sp("0.5", "r9"), sp("`x`", "r2", "quote"),
            sp("`zz`", "r2", "absent"), sp("a" * 64, "r2", "hash"), sp("x", "r1", "exec"),
            sp("0.5", "path:a.json#/o"), sp("`hello`", "path:t.txt#L1", "quote"),
            sp("0.5", "path:a.json#/s"), sp("q", "bogus"), sp("0.5", "path:nope.json"),
            sp("", "r1"), sp("a" * 301, "r2", "quote"), "<sworn>", " </sworn> ",
        ]).encode()
        core = verify(doc, name="d.md", manifest=m, tree=tree)
        seen = {s["verdict"] for s in core["spans"]}
        assert seen >= {"HELD", "FAILED", "UNRESOLVED", "MALFORMED"}
        rec = issue_receipt(core)                        # jcs must accept every field
        assert len(rec["digest"]) == 64
        assert all(r in REASONS for r in {s["reason"] for s in core["spans"]} - {None})

    def test_verifier_version_is_the_installed_styxx_version(self):
        import styxx
        assert verify(sp("1").encode(), name="d.md")["verifier"]["styxx_version"] == styxx.__version__

    def test_the_verdict_is_a_function_of_bytes_not_of_cwd_or_clock(self, tmp_path, monkeypatch):
        doc = sp("0.5").encode()
        m = manifest(r1=b"0.5")
        a = issue_receipt(verify(doc, name="d.md", manifest=m))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sworn, "_now", lambda: "2031-01-01T00:00:00Z")
        monkeypatch.setenv("LC_ALL", "C")
        b = issue_receipt(verify(doc, name="d.md", manifest=m))
        assert a["digest"] == b["digest"]

    def test_the_coverage_splitter_is_diffgate_s_own_literal(self):
        lit = r"(?<=[.!?])\s+|\n+"
        assert sworn._SENTENCE_SPLIT.pattern.decode() == lit
        src = Path(sworn.__file__).with_name("diffgate.py").read_text(encoding="utf-8")
        assert 're.split(r"%s"' % lit in src, "diffgate's splitter moved; sworn must move with it"

    def test_the_observer_cannot_touch_a_verdict(self, monkeypatch):
        import styxx.claimdetect as cd
        doc = (sp("0.5") + " Rewrote styxx/x.py.").encode()
        m = manifest(r1=b"0.5")
        base = verify(doc, name="d.md", manifest=m)

        class _R:
            is_claim = True
        monkeypatch.setattr(cd, "detect", lambda s: _R())
        always = verify(doc, name="d.md", manifest=m)
        assert always["document_verdict"] == base["document_verdict"] == "SWORN-HELD"
        assert always["coverage"]["diff_claim_sentences"] == 1

        def boom(s):
            raise RuntimeError("observer down")
        monkeypatch.setattr(cd, "detect", boom)
        down = verify(doc, name="d.md", manifest=m)
        assert down["document_verdict"] == "SWORN-HELD"
        assert down["coverage"]["diff_claim_sentences"] is None and "raised" in down["coverage"]["note"]
        # the floor does not depend on the observer at all
        assert down["coverage"]["sentence_share"] == always["coverage"]["sentence_share"] == 0.5

    def test_coverage_of_an_unsworn_document_is_a_zero_floor_not_undefined(self):
        cov = verify(b"Weather was fine.\n", name="d.md")["coverage"]
        assert cov["sentence_share"] == 0.0 and cov["diff_claim_sentences"] == 0
        assert verify(b"", name="d.md")["coverage"]["sentence_share"] is None        # 0/0

    def test_a_refusal_writes_nothing(self, tmp_path):
        doc = tmp_path / "d.md"
        doc.write_bytes(b"<sworn>broken\n")
        with pytest.raises(SystemExit, match="REFUSED"):
            sworn.main(["canon", str(doc)])
        assert not doc.with_suffix(".sworn.json").exists()


# ============================================================================ a committed receipt

@pytest.mark.skipif(not STAGE2.exists(), reason="the STRUCT-1 receipt is not in this checkout")
class TestWorkedExamplesOnTheStruct1Receipt:
    """The spec's own receipts as fixtures: papers/closed-model-frontier/stage2_result.json."""

    @pytest.fixture
    def tree(self):
        return MemoryTree({"stage2_result.json": STAGE2.read_bytes()}, commit=C40)

    @pytest.mark.parametrize("text,receipt,expected", [
        ("precision 0.4211", "path:stage2_result.json#/arms/flagged/A_share", "HELD"),
        ("precision 0.421", "path:stage2_result.json#/arms/flagged/A_share", "HELD"),
        ("precision 0.42", "path:stage2_result.json#/arms/flagged/A_share", "HELD"),
        ("precision 0.4210", "path:stage2_result.json#/arms/flagged/A_share", "FAILED"),
        ("precision 42.11%", "path:stage2_result.json#/arms/flagged/A_share", "FAILED"),
        ("a bar of 0.2061", "path:stage2_result.json#/gates/G-S2P/bar", "HELD"),
        ("n=38", "path:stage2_result.json#/arms/flagged/total", "HELD"),
        ("n=38.0", "path:stage2_result.json#/arms/flagged/total", "HELD"),
        ("n=37", "path:stage2_result.json#/arms/flagged/total", "FAILED"),
    ])
    def test_numeric_against_the_real_receipt(self, tree, text, receipt, expected):
        d, _ = one(sp(text, receipt).encode(), tree=tree)
        assert d["verdict"] == expected, d

    def test_a_bool_leaf_and_a_string_leaf_are_not_numbers_the_verifier_may_read(self, tree):
        d, _ = one(sp("1", "path:stage2_result.json#/gates/G-V/floors_ok").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_numeric")
        d, _ = one(sp("0.4211", "path:stage2_result.json#/gates/G-S2P/mandatory_counts_statement").encode(),
                   tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_numeric")
        d, _ = one(sp("0.4211", "path:stage2_result.json#/arms").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_scalar")

    def test_quote_against_the_statement_string(self, tree):
        ref = "path:stage2_result.json#/gates/G-S2P/mandatory_counts_statement"
        d, _ = one(sp("it says `STRUCT-1 = 16/38 (A-share 0.4211)`", ref, "quote").encode(), tree=tree)
        assert d["verdict"] == "HELD"
        d, _ = one(sp("it says `STRUCT-1 = 17/38`", ref, "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("FAILED", "needle_missing")

    def test_the_idiom_for_several_numbers_is_several_spans(self, tree):
        doc = ("STRUCT-1 scored " + sp("precision 0.4211", "path:stage2_result.json#/arms/flagged/A_share")
               + " against " + sp("a bar of 0.2061", "path:stage2_result.json#/gates/G-S2P/bar")
               + " at " + sp("n=38", "path:stage2_result.json#/arms/flagged/total") + ".\n").encode()
        core = verify(doc, name="d.md", tree=tree)
        assert core["document_verdict"] == "SWORN-HELD" and core["counts"]["HELD"] == 3


# ============================================================================ from the attack pass

class TestSidecarHardening:
    """Confirmed by the adversarial pass: a sidecar the loader refuses must never be emitted, and
    a loader must refuse, never crash, on any shape."""

    def test_a_zero_byte_span_is_malformed_inline_and_refuses_the_sidecar(self):
        doc = b"ab " + sp("").encode() + b" cd\n"
        core = verify(doc, name="d.md")
        assert core["counts"]["MALFORMED"] == 1 and core["spans"][0]["reason"] == "empty_span"
        with pytest.raises(SystemExit, match="REFUSED"):
            to_sidecar(doc, "d.md")

    def test_to_sidecar_never_emits_a_commit_its_loader_refuses(self):
        for bad in ("HEAD", "A" * 40, "abc", 5):
            with pytest.raises(SystemExit, match="REFUSED"):
                to_sidecar(sp("x").encode(), "d.md", bad)

    @pytest.mark.parametrize("mutate", [
        lambda s: s.__setitem__("text", 5), lambda s: s.__setitem__("text", None),
        lambda s: s.__setitem__("text", "\ud800"), lambda s: s.__setitem__("spans", 5),
        lambda s: s.__setitem__("spans", [5]), lambda s: s.__setitem__("spans", [None]),
        lambda s: s.__setitem__("spans", [["start", "end", "receipt", "kind"]]),
        lambda s: s.__setitem__("document", "x"), lambda s: s.__setitem__("document", []),
        lambda s: s.__setitem__("document", {"sha256": s["document"]["sha256"]}),
        lambda s: s.__setitem__("commit", 5), lambda s: s.__setitem__("manifest", "x"),
        lambda s: s.__setitem__("manifest", {"spec": "sworn/manifest/0.1", "receipts": ["r1"]}),
        lambda s: s.__setitem__("manifest", {"spec": "sworn/manifest/0.1", "receipts": {"r1": "abc"}}),
        lambda s: s.__setitem__("manifest", {"spec": "sworn/manifest/0.1", "receipts": {},
                                             "authored_sha256": 5}),
        lambda s: s["spans"][0].update({"start": True}),
        lambda s: s["spans"][0].update({"receipt": None}),
        lambda s: s["spans"][0].update({"receipt": ["r1"]}),
        lambda s: s["spans"][0].update({"receipt": "r1>"}),
        lambda s: s["spans"][0].update({"receipt": "r1\n```"}),
        lambda s: s["spans"][0].update({"kind": 5}),
        lambda s: s["spans"][0].update({"kind": 'quo"te'}),
    ])
    def test_a_sidecar_of_any_bad_shape_is_refused_never_crashed_on(self, mutate):
        side = json.loads(json.dumps(to_sidecar(sp("x").encode(), "d.md")))
        mutate(side)
        with pytest.raises(SystemExit, match="REFUSED"):
            verify(sidecar=side)

    def test_a_sidecar_whose_text_has_no_canonical_form_is_refused(self):
        text = "```\nx\n"
        side = {"spec": SPEC, "commit": None, "document": {"name": "d.md", "sha256": sworn._sha256(text.encode())},
                "text": text, "spans": [], "manifest": {"spec": "sworn/manifest/0.1", "receipts": {}}}
        with pytest.raises(SystemExit, match="no canonical form"):
            verify(sidecar=side)

    def test_a_sidecar_placing_a_span_where_the_lexer_sees_none_is_refused(self):
        side = json.loads(json.dumps(to_sidecar(b"`code` here\n", "d.md")))
        side["spans"] = [{"start": 1, "end": 5, "receipt": "r1", "kind": "quote"}]
        with pytest.raises(SystemExit, match="REFUSED"):
            verify(sidecar=side)


class TestReceiptHardening:
    """Confirmed by the receipts attacker: shapes that crashed, or reached git with a meaning
    other than the committed file they named."""

    @pytest.mark.parametrize("ref", ["path::/res.json", "path::(top)res.json", "path::!res.json",
                                     "path::^res.json", "path::res.json"])
    def test_git_pathspec_magic_never_reaches_the_tree(self, ref, git_repo):
        repo, sha = git_repo
        d, _ = one(sp("0.91", ref).encode(), tree=GitTree(repo, sha))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_form")
        d, _ = one(sp("0.91", ref).encode(), tree=MemoryTree({"res.json": b"0.91"}, commit=C40))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_form")

    def test_a_string_leaf_holding_a_lone_surrogate_is_not_text_and_never_crashes(self):
        tree = MemoryTree({"a.json": b'{"s": "\\ud800", "n": 1}'}, commit=C40)
        d, _ = one(sp("`x`", "path:a.json#/s", "quote").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_string")
        d, core = one(sp("1", "path:a.json#/s").encode(), tree=tree)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "leaf_not_numeric")
        assert len(issue_receipt(core)["digest"]) == 64            # the detail is serialisable

    @pytest.mark.parametrize("entry", ["abc", 5, None, ["r1"]])
    def test_a_manifest_entry_that_is_not_an_object_is_manifest_integrity(self, entry):
        m = manifest(r1=b"0.5")
        m.receipts["r1"] = entry
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_integrity")

    def test_a_kind_of_source_that_is_not_a_string_is_kind_of_source_unknown(self):
        m = manifest(r1=b"0.5")
        m.receipts["r1"]["kind_of_source"] = ["tool_stdout"]
        d, _ = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "kind_of_source_unknown")

    def test_a_manifest_no_canonical_form_can_hold_is_manifest_integrity(self):
        m = manifest(r1=b"0.5")
        m.receipts["r1"]["captured_at"] = float("nan")
        m.declared_digest = "0" * 64
        d, core = one(sp("0.5").encode(), manifest=m)
        assert (d["verdict"], d["reason"]) == ("UNRESOLVED", "manifest_integrity")
        assert core["manifest_digest"] is None
        assert len(issue_receipt(core)["digest"]) == 64

    @pytest.mark.parametrize("bad", [{"spec": "sworn/manifest/0.1", "receipts": "abc"},
                                     {"spec": "sworn/manifest/0.1", "receipts": ["r1"]},
                                     {"spec": "sworn/manifest/0.1", "receipts": {}, "authored_sha256": 5},
                                     {"spec": "sworn/manifest/0.1", "receipts": {}, "authored_sha256": [5]}])
    def test_a_manifest_of_the_wrong_shape_is_refused_not_crashed_on(self, bad):
        with pytest.raises(SystemExit, match="REFUSED"):
            Manifest.from_dict(bad)

    def test_the_modules_own_add_mints_only_entries_it_can_resolve(self):
        m = Manifest("h", "t")
        e = m.add("r1", b"bytes of the record", "file_read", complete=True,
                  sha256=sworn._sha256(b"bytes of the record").upper())
        assert e["sha256"] == sworn._sha256(b"bytes of the record")
        d, _ = one(sp("`bytes of the record`", kind="quote").encode(), manifest=m)
        assert d["verdict"] == "HELD"
        with pytest.raises(ValueError):
            m.add("r2", b"bytes", "file_read", sha256="0" * 64)      # a lie about the bytes
        with pytest.raises(ValueError):
            m.add("r3", None, "file_read", sha256="not-hex")


class TestGamingLens:
    """The author wants a clean verdict without being right. Every route must close from the
    bytes alone: a MALFORMED that depended on evidence could be turned into an UNRESOLVED by
    pointing at a receipt that is not there."""

    @pytest.mark.parametrize("doc,reason", [
        ("<sworn>x</sworn>", "tag_syntax"),
        (sp("a " + sp("b", "r2")), "nesting"),
        ("x </sworn>", "stray_closer"),
        ('<sworn r="r1" k="numeric">never', "unclosed"),
        (sp(""), "empty_span"),
        (sp("a" * 301, kind="quote"), "length_cap"),
        (sp("1", "nope"), "receipt_form"),
        (sp("1", "r1#"), "receipt_form"),
        (sp("1", "path:x.json#"), "receipt_form"),
        (sp("1", kind="Numeric"), "kind_unknown"),
        (sp("1", kind="exec"), "kind_reserved"),
        (sp("1 and 2"), "number_count"),
        (sp("v1.2 shipped"), "number_grammar"),
        (sp("no needle", kind="quote"), "needle_count"),
        (sp("`  `", kind="quote"), "needle_empty"),
        (sp("no digest here", kind="hash"), "digest_form"),
        (sp("no `x`", "path:a.json#/y", "absent"), "absent_over_partial"),
        (sp("a" * 64, "path:a.json#L1", "hash"), "hash_over_partial"),
    ])
    def test_every_bytes_decidable_malformed_is_decided_with_no_manifest_and_no_tree(self, doc, reason):
        core = verify(doc.encode(), name="d.md")           # manifest=None, tree=None
        reasons = [s["reason"] for s in core["spans"] if s["verdict"] == "MALFORMED"]
        assert reason in reasons, (doc, core["spans"])
        assert core["counts"]["UNRESOLVED"] == 0, "a broken declaration must not hide as unresolved"
        assert core["document_verdict"] == "SWORN-FAILED"

    def test_breaking_your_own_failed_span_does_not_make_it_narrative(self):
        good = sp("0.6").encode()
        assert verify(good, name="d.md", manifest=manifest(r1=b"0.5"))["document_verdict"] == "SWORN-FAILED"
        for broken in (b'<sworn r="r1" k="numeric" >0.6</sworn>', b'<SWORN r="r1" k="numeric">0.6</sworn>',
                       b'<sworn r=r1 k=numeric>0.6</sworn>', b'<sworn r="r1" k="numeric"/>0.6</sworn>'):
            core = verify(broken, name="d.md", manifest=manifest(r1=b"0.5"))
            assert core["document_verdict"] == "SWORN-FAILED" and core["counts"]["MALFORMED"] >= 1, broken

    def test_a_manifest_tampered_inside_the_sidecar_resolves_nothing(self):
        m = manifest(r1=b"0.5")
        side = json.loads(json.dumps(to_sidecar(sp("0.5").encode(), "d.md", None, m)))
        assert verify(sidecar=side)["spans"][0]["verdict"] == "HELD"
        side["manifest"]["receipts"]["r1"]["complete"] = False            # edited after minting
        core = verify(sidecar=json.loads(json.dumps(side)))
        assert (core["spans"][0]["verdict"], core["spans"][0]["reason"]) == ("UNRESOLVED", "manifest_integrity")
        side["manifest"]["receipts"]["r1"]["bytes"] = sworn._b64(b"0.6")  # and the bytes too
        core = verify(sidecar=json.loads(json.dumps(side)))
        assert core["spans"][0]["verdict"] == "UNRESOLVED"

    def test_swearing_only_trivia_prints_its_coverage_beside_the_verdict(self):
        """Two trivial counts sworn, three load-bearing sentences left in narrative: the floor
        prints 0.40 beside the SWORN-HELD, and the diff-claim count prints 3 beside it."""
        doc = (sp("3 files changed.", "r1") + " " + sp("1 commit.", "r2")
               + " Rewrote styxx/certify.py. Deleted tests/test_gate.py. Added docs/x.md.\n").encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"3", r2=b"1"))
        assert core["document_verdict"] == "SWORN-HELD"
        head = sworn._headline(core)
        assert "coverage-floor≈0.40" in head and "diff-claims≈3" in head

    def test_a_short_trivial_quote_is_malformed_not_held(self):
        """The v0.1 version of the test above swore `7.47.0` against a whole receipt. Under R3 a
        six-byte quote over a whole receipt is not an oath the verifier will count."""
        doc = (sp("`7.47.0`", "r1", "quote") + "\n").encode()
        core = verify(doc, name="d.md", manifest=manifest(r1=b"7.47.0 released 2026-08-31"))
        assert core["document_verdict"] == "SWORN-FAILED"
        assert core["spans"][0]["reason"] == "short_needle"


class TestGamingLensFromTheAttackPass:
    """Confirmed by the gaming attacker."""

    @pytest.mark.parametrize("ws", [b"\x0c", b"\x0b", b" \t\x0c\x0b"])   # no newline: a needle is line-local
    def test_every_ascii_whitespace_only_needle_is_empty(self, ws):
        doc = b'<sworn r="r1" k="quote">see `' + ws + b'`</sworn>\n'
        d, _ = one(doc, manifest=manifest(r1=b"a" + ws + b"b"))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "needle_empty")

    @pytest.mark.parametrize("ref", ["path:*.json#/x", "path:?.json#/x", "path:[a].json#/x", "path:a]json"])
    def test_a_path_names_one_file_never_a_glob_the_verifier_would_pick_from(self, ref):
        d, _ = one(sp("1", ref).encode(), tree=MemoryTree({"a.json": b'{"x": 1}'}, commit=C40))
        assert (d["verdict"], d["reason"]) == ("MALFORMED", "receipt_form")

    def test_a_tampered_receipt_fails_and_never_crashes_or_refuses(self):
        doc = sp("0.5").encode()
        m = manifest(r1=b"0.5")
        good = issue_receipt(verify(doc, name="d.md", manifest=m))
        for mutate in (lambda r: r.__setitem__("document", None), lambda r: r.__setitem__("verifier", None),
                       lambda r: r.__setitem__("commit", 5), lambda r: r.__setitem__("commit", "b" * 40),
                       lambda r: r.__setitem__("spans", "x"), lambda r: r.__setitem__("counts", None)):
            rec = json.loads(json.dumps(good))
            mutate(rec)
            res = verify_receipt(rec, doc, manifest=m)
            assert res["status"] == "FAILED", res
        assert verify_receipt("not a receipt", doc, manifest=m)["status"] == "FAILED"
        side = to_sidecar(doc, "d.md", C40, m)
        rec = issue_receipt(verify(sidecar=side))
        rec["commit"] = "b" * 40                                 # edited after issuance
        assert verify_receipt(rec, sidecar=side)["status"] == "FAILED"

    def test_the_commit_the_document_names_wins_over_the_tree_handle(self, git_repo):
        """path receipts resolve AT THE COMMIT THE DOCUMENT NAMES; a tree handle built at another
        commit is a repository, not a choice."""
        repo, sha = git_repo
        doc = sp("0.10", "path:res.json#/recall").encode()          # 0.10 is the WORKING-TREE value
        side = to_sidecar(doc, "d.md", sha)
        handle = GitTree(repo, "b" * 40)
        core = verify(sidecar=side, tree=handle)
        assert core["commit"] == sha and handle.commit == sha
        assert (core["spans"][0]["verdict"], core["spans"][0]["reason"]) == ("FAILED", "value_mismatch")
        core = verify(doc, name="d.md", tree=GitTree(repo, "b" * 40), commit=sha)
        assert core["commit"] == sha and core["spans"][0]["verdict"] == "FAILED"
        # a sidecar that names no commit gets none, however the handle was built
        side = to_sidecar(doc, "d.md", None)
        core = verify(sidecar=side, tree=GitTree(repo, sha))
        assert core["commit"] is None
        assert (core["spans"][0]["verdict"], core["spans"][0]["reason"]) == ("UNRESOLVED", "no_commit")
